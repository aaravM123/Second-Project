pub mod training;

use candle_core::{DType, Device as CandleDevice, Result as CandleResult, Shape, Tensor, D};
use candle_nn::{var_builder::SimpleBackend, Activation, Init, VarBuilder};
use candle_transformers::models::gemma3;
use gemmatune_core::Device;
use gemmatune_lora::TrainableAdapter;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub requested: Device,
    pub active: Device,
    pub implementation: String,
    pub accelerated: bool,
    pub note: String,
}

pub fn select_backend(requested: Device) -> Result<(CandleDevice, BackendStatus), String> {
    match requested {
        Device::Cpu => Ok((
            CandleDevice::Cpu,
            BackendStatus {
                requested,
                active: Device::Cpu,
                implementation: "candle-cpu".into(),
                accelerated: false,
                note: "Candle executes Gemma tensors on the local CPU.".into(),
            },
        )),
        Device::Metal => metal_backend(),
        Device::Cuda => cuda_backend(),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_backend() -> Result<(CandleDevice, BackendStatus), String> {
    let device = CandleDevice::new_metal(0).map_err(|error| format!("Metal initialization failed: {error}"))?;
    Ok((device, BackendStatus {
        requested: Device::Metal, active: Device::Metal, implementation: "candle-metal".into(),
        accelerated: true, note: "Candle Metal kernels execute the loaded Gemma model locally.".into(),
    }))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_backend() -> Result<(CandleDevice, BackendStatus), String> {
    Err("Metal requires macOS and `gemmatune-runtime`'s `metal` feature".into())
}

#[cfg(feature = "cuda")]
fn cuda_backend() -> Result<(CandleDevice, BackendStatus), String> {
    let device = CandleDevice::new_cuda(0).map_err(|error| format!("CUDA initialization failed: {error}"))?;
    Ok((device, BackendStatus {
        requested: Device::Cuda, active: Device::Cuda, implementation: "candle-cuda".into(),
        accelerated: true, note: "Candle CUDA kernels execute the loaded Gemma model locally.".into(),
    }))
}

#[cfg(not(feature = "cuda"))]
fn cuda_backend() -> Result<(CandleDevice, BackendStatus), String> {
    Err("CUDA requires building `gemmatune-runtime` with its `cuda` feature".into())
}

fn gemma_3_1b_config() -> gemma3::Config {
    gemma3::Config {
        attention_bias: false,
        head_dim: 256,
        hidden_activation: Activation::Gelu,
        hidden_size: 1152,
        intermediate_size: 6912,
        num_attention_heads: 4,
        num_hidden_layers: 26,
        num_key_value_heads: 1,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_local_base_freq: 10_000.0,
        vocab_size: 262_144,
        final_logit_softcapping: Some(30.0),
        attn_logit_softcapping: Some(50.0),
        query_pre_attn_scalar: 256,
        sliding_window: 512,
        sliding_window_pattern: 6,
        max_position_embeddings: 32_768,
    }
}

fn safetensor_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = fs::read_dir(root)
        .map_err(|error| format!("cannot read checkpoint directory {}: {error}", root.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "safetensors"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no .safetensors files found in {}", root.display()));
    }
    Ok(paths)
}

struct LoraBackend {
    base: candle_core::safetensors::MmapedSafetensors,
    adapter: Arc<Mutex<TrainableAdapter>>,
}

impl LoraBackend {
    fn module_for(name: &str) -> Option<&str> {
        ["q_proj", "k_proj", "v_proj", "o_proj"]
            .into_iter()
            .find(|module| name.ends_with(&format!("self_attn.{module}.weight")))
    }
}

impl SimpleBackend for LoraBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _init: Init,
        dtype: DType,
        device: &CandleDevice,
    ) -> CandleResult<Tensor> {
        let base = self.base.load(name, device)?.to_dtype(dtype)?;
        let Some(module) = Self::module_for(name) else {
            return Ok(base);
        };
        let adapter = self.adapter.lock().expect("adapter lock is poisoned");
        let tensor = adapter.tensor(module).expect("LoRA plan has every attention projection");
        let delta = Tensor::from_vec(
            tensor.delta_weight(),
            (tensor.spec.output_features, tensor.spec.input_features),
            device,
        )?.to_dtype(dtype)?;
        if shape.dims() != delta.dims() {
            candle_core::bail!("LoRA shape mismatch for {name}");
        }
        base.add(&delta)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.base.contains_tensor(name)
    }
}

pub struct LocalGemma {
    model: Mutex<gemma3::Model>,
    device: CandleDevice,
    pub backend: BackendStatus,
}

pub struct AdapterRuntime {
    root: PathBuf,
    requested: Device,
    adapter: Arc<Mutex<TrainableAdapter>>,
    model: LocalGemma,
}

impl AdapterRuntime {
    pub fn load(
        root: impl AsRef<Path>,
        requested: Device,
        adapter: TrainableAdapter,
    ) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        let adapter = Arc::new(Mutex::new(adapter));
        let model = LocalGemma::load_1b_it_with_adapter(&root, requested, adapter.clone())?;
        Ok(Self { root, requested, adapter, model })
    }

    pub fn adapter(&self) -> Arc<Mutex<TrainableAdapter>> {
        self.adapter.clone()
    }

    pub fn generate(&self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>, String> {
        self.model.generate(prompt_ids, max_new_tokens)
    }

    pub fn reload_after_update(&mut self) -> Result<(), String> {
        self.model = LocalGemma::load_1b_it_with_adapter(
            &self.root,
            self.requested,
            self.adapter.clone(),
        )?;
        Ok(())
    }

    pub fn backend(&self) -> &BackendStatus {
        &self.model.backend
    }
}

impl LocalGemma {
    pub fn load_1b_it(root: impl AsRef<Path>, requested: Device) -> Result<Self, String> {
        let (device, backend) = select_backend(requested)?;
        let paths = safetensor_paths(root.as_ref())?;
        let builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &device)
                .map_err(|error| format!("cannot map Gemma safetensors: {error}"))?
        };
        let model = gemma3::Model::new(false, &gemma_3_1b_config(), builder)
            .map_err(|error| format!("cannot load Gemma 3 1B IT tensor layout: {error}"))?;
        Ok(Self { model: Mutex::new(model), device, backend })
    }

    pub fn load_1b_it_with_adapter(
        root: impl AsRef<Path>,
        requested: Device,
        adapter: Arc<Mutex<TrainableAdapter>>,
    ) -> Result<Self, String> {
        let (device, backend) = select_backend(requested)?;
        let paths = safetensor_paths(root.as_ref())?;
        let base = unsafe {
            candle_core::safetensors::MmapedSafetensors::multi(&paths)
                .map_err(|error| format!("cannot map Gemma safetensors: {error}"))?
        };
        let builder = VarBuilder::from_backend(
            Box::new(LoraBackend { base, adapter }),
            DType::BF16,
            device.clone(),
        );
        let model = gemma3::Model::new(false, &gemma_3_1b_config(), builder)
            .map_err(|error| format!("cannot inject LoRA into Gemma 3 1B IT: {error}"))?;
        Ok(Self { model: Mutex::new(model), device, backend })
    }

    pub fn generate(&self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>, String> {
        if prompt_ids.is_empty() {
            return Err("a Gemma prompt needs at least one token".into());
        }
        if prompt_ids.len() + max_new_tokens > 32_768 {
            return Err("Gemma 3 1B IT is limited to 32768 total tokens".into());
        }
        let mut model = self.model.lock().map_err(|_| "Gemma model lock is poisoned")?;
        model.clear_kv_cache();
        let mut next_input = prompt_ids.to_vec();
        let mut offset = 0;
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let input = Tensor::from_vec(next_input.clone(), (1, next_input.len()), &self.device)
                .map_err(|error| format!("cannot create Gemma input tensor: {error}"))?;
            let logits = model.forward(&input, offset).map_err(|error| format!("Gemma forward pass failed: {error}"))?;
            let next = logits.squeeze(0).and_then(|tensor| tensor.argmax(D::Minus1))
                .and_then(|tensor| tensor.to_scalar::<u32>())
                .map_err(|error| format!("cannot sample Gemma output: {error}"))?;
            offset += next_input.len();
            next_input = vec![next];
            generated.push(next);
        }
        Ok(generated)
    }
}
