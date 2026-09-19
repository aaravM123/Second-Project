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
    pub base_token_accuracy: f32,
    pub adapter_token_accuracy: f32,
    pub improvement: f32,
}

/// Compare generated token IDs to held-out target token IDs. Scores are
/// measured from model output, never inferred from adapter rank or metadata.
pub fn evaluate(
    manifest: &TrainingManifest,
    adapter: &AdapterCheckpoint,
    base_predictions: &[u32],
    adapter_predictions: &[u32],
    targets: &[u32],
) -> EvaluationReport {
    let score = |predictions: &[u32]| {
        let compared = predictions.len().min(targets.len());
        if compared == 0 { 0.0 } else {
            predictions.iter().zip(targets).take(compared).filter(|(a, b)| a == b).count() as f32 / compared as f32
        }
    };
    let base_token_accuracy = score(base_predictions);
    let adapter_token_accuracy = score(adapter_predictions);
    EvaluationReport {
        schema_version: 1,
        base_checkpoint: manifest.config.model.checkpoint.clone(),
        adapter_format: adapter.format.clone(),
        validation_examples: manifest.validation_examples,
        base_token_accuracy,
        adapter_token_accuracy,
        improvement: adapter_token_accuracy - base_token_accuracy,
    }
}

impl EvaluationReport {
    pub fn write_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }
}
