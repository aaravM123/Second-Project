//! Evaluation report generation for comparing a base model with its LoRA adapter.

use gemmatune_core::TrainingManifest;
use gemmatune_lora::AdapterCheckpoint;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub schema_version: u8,
    pub base_checkpoint: String,
    pub adapter_format: String,
    pub validation_examples: usize,
    pub base_style_score: f32,
    pub adapter_style_score: f32,
    pub improvement: f32,
    pub note: String,
}

pub fn evaluate(manifest: &TrainingManifest, adapter: &AdapterCheckpoint) -> EvaluationReport {
    let base_style_score = 0.50;
    // This predictable score makes dry-run output meaningful without falsely
    // representing itself as a model-quality measurement.
    let improvement = (adapter.config.rank as f32 / 256.0).min(0.20);
    EvaluationReport {
        schema_version: 1,
        base_checkpoint: manifest.config.model.checkpoint.clone(),
        adapter_format: adapter.format.clone(),
        validation_examples: manifest.validation_examples,
        base_style_score,
        adapter_style_score: base_style_score + improvement,
        improvement,
        note: "Synthetic pipeline metric: replace with held-out inference metrics when a runtime backend is enabled.".into(),
    }
}

impl EvaluationReport {
    pub fn write_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }
}
