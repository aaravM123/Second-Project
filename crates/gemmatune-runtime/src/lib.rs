//! Local Gemma 3 1B IT inference using Candle and checkpoint-local safetensors.

use candle_core::{DType, Device as CandleDevice, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::gemma3;
use gemmatune_core::Device;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
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

/// Loaded, local Gemma 3 1B IT weights. Candle's Gemma implementation performs
/// GQA, Q/K RMS normalization, local/global masks, RoPE, and a KV cache.
pub struct LocalGemma {
    model: Mutex<gemma3::Model>,
    device: CandleDevice,
    pub backend: BackendStatus,
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
