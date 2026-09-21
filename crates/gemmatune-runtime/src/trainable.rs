//! Differentiable, full-sequence Gemma 3 decoder used by local LoRA training.
//!
//! Candle's upstream Gemma model intentionally returns only its final logits
//! and mutates a KV cache. This fork uses frozen checkpoint tensors plus
//! `Var`-backed LoRA matrices, so a teacher-forced forward pass is cache-free
//! and its loss can be backpropagated into the adapters.
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, D};
use candle_nn::VarBuilder;
use candle_transformers::utils::repeat_kv;
use gemmatune_core::Device as RequestedDevice;
use gemmatune_lora::{TrainableAdapter, TrainableTensor};
use std::{path::Path, sync::Arc};

pub const GEMMA_3_1B_LAYERS: usize = 26;
const HIDDEN: usize = 1152;
const HEAD_DIM: usize = 256;
const HEADS: usize = 4;
const VOCABULARY: usize = 262_144;
const MAX_POSITIONS: usize = 32_768;
type Result<T> = CandleResult<T>;
struct RmsNorm {
    weight: Tensor,
}
impl RmsNorm {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(HIDDEN, "weight")?.to_dtype(DType::F32)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = (x.sqr()?.sum_keepdim(D::Minus1)? / HIDDEN as f64)?;
        x.broadcast_div(&(variance + 1e-6)?.sqrt()?)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}
struct FrozenLinear {
    weight: Tensor,
}
impl FrozenLinear {
    fn load(vb: VarBuilder, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get((output, input), "weight")?.to_dtype(DType::F32)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, tokens, input) = x.dims3()?;
        x.reshape((batch * tokens, input))?
            .matmul(&self.weight.t()?)?
            .reshape((batch, tokens, self.weight.dim(0)?))
    }
}
/// A frozen Gemma projection with the trainable `scale * B @ A` path attached.
struct LoraLinear {
    base: FrozenLinear,
    a: Var,
    b: Var,
    scale: f64,
}
impl LoraLinear {
    fn load(vb: VarBuilder, adapter: &TrainableTensor) -> Result<Self> {
        let spec = &adapter.spec;
        let base = FrozenLinear::load(vb, spec.input_features, spec.output_features)?;
        let device = base.weight.device().clone();
        Ok(Self {
            base,
            a: Var::from_vec(adapter.a.clone(), spec.shape_a, &device)?,
            b: Var::from_vec(adapter.b.clone(), spec.shape_b, &device)?,
            scale: adapter.scaling() as f64,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, tokens, input) = x.dims3()?;
        let update = (x
            .reshape((batch * tokens, input))?
            .matmul(&self.a.t()?)?
            .matmul(&self.b.t()?)?
            .reshape((batch, tokens, self.b.dim(0)?))?
            * self.scale)?;
        self.base.forward(x)?.add(&update)
    }

    fn variables(&self) -> [Var; 2] {
        [self.a.clone(), self.b.clone()]
    }
}
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}
impl Rotary {
    fn new(theta: f64, device: &Device) -> Result<Self> {
        let frequencies = (0..HEAD_DIM)
            .step_by(2)
            .map(|index| 1f32 / theta.powf(index as f64 / HEAD_DIM as f64) as f32)
            .collect::<Vec<_>>();
        let positions = Tensor::arange(0u32, MAX_POSITIONS as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_POSITIONS, 1))?;
        let values =
            positions.matmul(&Tensor::from_vec(frequencies, (1, HEAD_DIM / 2), device)?)?;
        Ok(Self {
            sin: values.sin()?,
            cos: values.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let length = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, length)?;
        let sin = self.sin.narrow(0, 0, length)?;
        Ok((
            candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}
struct Attention {
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    o: LoraLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: Arc<Rotary>,
}
impl Attention {
    fn load(vb: VarBuilder, adapter: &TrainableAdapter, rotary: Arc<Rotary>) -> Result<Self> {
        let q = adapter
            .tensor("q_proj")
            .expect("validated adapter has q_proj");
        let k = adapter
            .tensor("k_proj")
            .expect("validated adapter has k_proj");
        let v = adapter
            .tensor("v_proj")
            .expect("validated adapter has v_proj");
        let o = adapter
            .tensor("o_proj")
            .expect("validated adapter has o_proj");
        Ok(Self {
            q: LoraLinear::load(vb.pp("q_proj"), q)?,
            k: LoraLinear::load(vb.pp("k_proj"), k)?,
            v: LoraLinear::load(vb.pp("v_proj"), v)?,
            o: LoraLinear::load(vb.pp("o_proj"), o)?,
            q_norm: RmsNorm::load(vb.pp("q_norm"))?,
            k_norm: RmsNorm::load(vb.pp("k_norm"))?,
            rotary,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, tokens, _) = x.dims3()?;
        let q = self
            .q
            .forward(x)?
            .reshape((batch, tokens, HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(x)?
            .reshape((batch, tokens, 1, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(x)?
            .reshape((batch, tokens, 1, HEAD_DIM))?
            .transpose(1, 2)?;
        let (q, k) = self
            .rotary
            .apply(&self.q_norm.forward(&q)?, &self.k_norm.forward(&k)?)?;
        let scores =
            (q.matmul(&repeat_kv(k, HEADS)?.transpose(2, 3)?)? / (HEAD_DIM as f64).sqrt())?;
        let weights = candle_nn::ops::softmax_last_dim(
            &((scores / 50.0)?.tanh()? * 50.0)?.broadcast_add(mask)?,
        )?;
        let output = weights
            .matmul(&repeat_kv(v, HEADS)?)?
            .transpose(1, 2)?
            .reshape((batch, tokens, HIDDEN))?;
        self.o.forward(&output)
    }

    fn variables(&self) -> Vec<Var> {
        [
            self.q.variables(),
            self.k.variables(),
            self.v.variables(),
            self.o.variables(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}
struct Mlp {
    gate: FrozenLinear,
    up: FrozenLinear,
    down: FrozenLinear,
}
impl Mlp {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate: FrozenLinear::load(vb.pp("gate_proj"), HIDDEN, 6912)?,
            up: FrozenLinear::load(vb.pp("up_proj"), HIDDEN, 6912)?,
            down: FrozenLinear::load(vb.pp("down_proj"), 6912, HIDDEN)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = (self.gate.forward(x)?.gelu_erf()? * self.up.forward(x)?)?;
        self.down.forward(&hidden)
    }
}
struct Layer {
    attention: Attention,
    mlp: Mlp,
    input_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    post_ff_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    local: bool,
}
impl Layer {
    fn load(
        vb: VarBuilder,
        adapter: &TrainableAdapter,
        rotary: Arc<Rotary>,
        local: bool,
    ) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(vb.pp("self_attn"), adapter, rotary)?,
            mlp: Mlp::load(vb.pp("mlp"))?,
            input_norm: RmsNorm::load(vb.pp("input_layernorm"))?,
            pre_ff_norm: RmsNorm::load(vb.pp("pre_feedforward_layernorm"))?,
            post_ff_norm: RmsNorm::load(vb.pp("post_feedforward_layernorm"))?,
            post_attn_norm: RmsNorm::load(vb.pp("post_attention_layernorm"))?,
            local,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let attention = self.attention.forward(&self.input_norm.forward(x)?, mask)?;
        let x = (x + self.post_attn_norm.forward(&attention)?)?;
        let feed_forward = self.mlp.forward(&self.pre_ff_norm.forward(&x)?)?;
        &x + self.post_ff_norm.forward(&feed_forward)?
    }
}
/// Frozen Gemma 3 1B base weights with 208 trainable per-layer LoRA variables.
///
/// This model never maintains a KV cache. `forward` returns a logits row for
/// every input token (`[batch, sequence, 262144]`) for teacher-forced loss.
pub struct TrainableGemmaDecoder {
    embeddings: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    device: Device,
}
impl TrainableGemmaDecoder {
    pub fn load_1b_it(
        root: impl AsRef<Path>,
        requested: RequestedDevice,
        adapter: &TrainableAdapter,
    ) -> std::result::Result<Self, String> {
        if adapter.base_checkpoint != "gemma-3-1b-it" {
            return Err("trainable decoder requires a gemma-3-1b-it adapter".into());
        }
        for module in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            if adapter.tensor(module).is_none() {
                return Err(format!("trainable decoder requires {module} LoRA weights"));
            }
        }
        let (device, _) = super::select_backend(requested)?;
        let paths = super::safetensor_paths(root.as_ref())?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device)
                .map_err(|error| format!("cannot map Gemma safetensors: {error}"))?
        }
        .pp("model");
        let embeddings = vb
            .pp("embed_tokens")
            .get((VOCABULARY, HIDDEN), "weight")
            .and_then(|weight| weight.to_dtype(DType::F32))
            .map_err(|error| format!("cannot load Gemma token embeddings: {error}"))?;
        let global =
            Arc::new(Rotary::new(1_000_000.0, &device).map_err(|error| error.to_string())?);
        let local = Arc::new(Rotary::new(10_000.0, &device).map_err(|error| error.to_string())?);
        let mut layers = Vec::with_capacity(GEMMA_3_1B_LAYERS);
        for index in 0..GEMMA_3_1B_LAYERS {
            let uses_local = (index + 1) % 6 != 0;
            layers.push(
                Layer::load(
                    vb.pp("layers").pp(index),
                    adapter,
                    if uses_local {
                        local.clone()
                    } else {
                        global.clone()
                    },
                    uses_local,
                )
                .map_err(|error| format!("cannot load Gemma layer {index}: {error}"))?,
            );
        }
        let norm = RmsNorm::load(vb.pp("norm"))
            .map_err(|error| format!("cannot load Gemma output norm: {error}"))?;
        Ok(Self {
            embeddings,
            layers,
            norm,
            device,
        })
    }

    /// Returns the actual Candle variables that an optimizer must update.
    pub fn adapter_variables(&self) -> Vec<Var> {
        self.layers
            .iter()
            .flat_map(|layer| layer.attention.variables())
            .collect()
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, tokens) = input_ids.dims2()?;
        if tokens == 0 || tokens > MAX_POSITIONS {
            candle_core::bail!("sequence length must be in 1..={MAX_POSITIONS}");
        }
        let mut x = (self.embeddings.embedding(input_ids)? * (HIDDEN as f64).sqrt())?;
        let global_mask = causal_mask(batch, tokens, None, &self.device)?;
        let local_mask = causal_mask(batch, tokens, Some(512), &self.device)?;
        for layer in &self.layers {
            x = layer.forward(
                &x,
                if layer.local {
                    &local_mask
                } else {
                    &global_mask
                },
            )?;
        }
        let logits = self.norm.forward(&x)?.matmul(&self.embeddings.t()?)?;
        (logits / 30.0)?.tanh()?.affine(30.0, 0.0)
    }
}
fn causal_mask(
    batch: usize,
    tokens: usize,
    window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let values = (0..tokens)
        .flat_map(|row| {
            (0..tokens).map(move |column| {
                if column > row || window.is_some_and(|width| column + width < row) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (tokens, tokens), device)?.expand((batch, 1, tokens, tokens))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lora_variables_receive_a_real_candle_gradient() -> Result<()> {
        let device = Device::Cpu;
        let projection = LoraLinear {
            base: FrozenLinear {
                weight: Tensor::zeros((2, 2), DType::F32, &device)?,
            },
            a: Var::from_vec(vec![1f32, 2.], (1, 2), &device)?,
            b: Var::from_vec(vec![3f32, 4.], (2, 1), &device)?,
            scale: 1.0,
        };
        let input = Tensor::from_vec(vec![1f32, 1.], (1, 1, 2), &device)?;
        let gradients = projection.forward(&input)?.sum_all()?.backward()?;
        assert!(gradients.get(&projection.a).is_some());
        assert!(gradients.get(&projection.b).is_some());
        Ok(())
    }

    #[test]
    fn creates_a_causal_local_mask() -> Result<()> {
        let mask = causal_mask(1, 3, Some(1), &Device::Cpu)?
            .squeeze(0)?
            .squeeze(0)?;
        assert_eq!(mask.to_vec2::<f32>()?[0][1], f32::NEG_INFINITY);
        assert_eq!(mask.to_vec2::<f32>()?[2][0], f32::NEG_INFINITY);
        Ok(())
    }
}
