//! Backend selection. CPU provides a deterministic local dry-run kernel;
//! Metal and CUDA intentionally report availability rather than silently emulating.

use gemmatune_core::Device;
use gemmatune_lora::AdapterCheckpoint;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStatus {
    pub requested: Device,
    pub active: Device,
    pub implementation: String,
    pub accelerated: bool,
    pub note: String,
}

pub fn select_backend(requested: Device) -> BackendStatus {
    match requested {
        Device::Cpu => BackendStatus {
            requested, active: Device::Cpu, implementation: "deterministic-cpu-dry-run".into(),
            accelerated: false, note: "CPU validates the training pipeline but does not update Gemma weights yet.".into(),
        },
        Device::Metal if cfg!(target_os = "macos") => BackendStatus {
            requested, active: Device::Metal, implementation: "metal-backend-stub".into(),
            accelerated: false, note: "Metal device selection is wired; kernel execution is not implemented in this milestone.".into(),
        },
        Device::Cuda if std::env::var_os("CUDA_VISIBLE_DEVICES").is_some() => BackendStatus {
            requested, active: Device::Cuda, implementation: "cuda-backend-stub".into(),
            accelerated: false, note: "CUDA device selection is wired; kernel execution is not implemented in this milestone.".into(),
        },
        unavailable => BackendStatus {
            requested: unavailable, active: Device::Cpu, implementation: "deterministic-cpu-dry-run".into(),
            accelerated: false, note: format!("{unavailable} is unavailable on this host; using CPU dry-run."),
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingOutcome {
    pub steps: usize,
    pub final_loss: f32,
    pub status: String,
    pub backend: BackendStatus,
}

pub fn train_dry_run(
    adapter: &AdapterCheckpoint,
    train_examples: usize,
    token_count: usize,
    device: Device,
) -> TrainingOutcome {
    let backend = select_backend(device);
    let steps = train_examples.saturating_mul(adapter.config.epochs as usize);
    let scale = (adapter.config.rank as f32).sqrt().max(1.0);
    let final_loss = (token_count.max(1) as f32).ln() / (steps.max(1) as f32 * scale);
    TrainingOutcome {
        steps,
        final_loss,
        status: "dry-run-complete".into(),
        backend,
    }
}
