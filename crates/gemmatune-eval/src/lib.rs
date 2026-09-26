//! Evaluation report generation for comparing a base model with its LoRA adapter.

use gemmatune_core::TrainingManifest;
use gemmatune_lora::AdapterCheckpoint;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

/// Held-out tokenized conversations captured with a completed local run.
///
/// The sequences are already redacted when the source dataset requested PII
/// redaction. Keeping the token IDs with the run makes later evaluation
/// reproducible even when the original dataset directory has moved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeldOutDataset {
    pub schema_version: u8,
    pub sequences: Vec<Vec<u32>>,
}

impl HeldOutDataset {
    pub fn from_sequences(sequences: &[Vec<u32>]) -> Result<Self, String> {
        if sequences.is_empty() {
            return Err("the held-out split contains no conversations".into());
        }
        Ok(Self {
            schema_version: 1,
            sequences: sequences.to_vec(),
        })
    }

    pub fn token_count(&self) -> usize {
        self.sequences.iter().map(Vec::len).sum()
    }

    pub fn write_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn read_from(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let dataset: Self = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if dataset.schema_version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported held-out dataset schema {}",
                    dataset.schema_version
                ),
            ));
        }
        Self::from_sequences(&dataset.sequences)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub schema_version: u8,
    pub base_checkpoint: String,
    pub adapter_format: String,
    pub validation_examples: usize,
    /// Number of generated target tokens included in both accuracy values.
    pub scored_tokens: usize,
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
        if compared == 0 {
            0.0
        } else {
            predictions
                .iter()
                .zip(targets)
                .take(compared)
                .filter(|(a, b)| a == b)
                .count() as f32
                / compared as f32
        }
    };
    let base_token_accuracy = score(base_predictions);
    let adapter_token_accuracy = score(adapter_predictions);
    EvaluationReport {
        schema_version: 1,
        base_checkpoint: manifest.config.model.checkpoint.clone(),
        adapter_format: adapter.format.clone(),
        validation_examples: manifest.validation_examples,
        scored_tokens: base_predictions
            .len()
            .min(adapter_predictions.len())
            .min(targets.len()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use gemmatune_core::TrainingConfig;
    use gemmatune_lora::{injection_plan, TrainableAdapter};

    fn manifest() -> TrainingManifest {
        TrainingManifest {
            schema_version: 1,
            created_at_unix_secs: 0,
            config: TrainingConfig::default(),
            train_examples: 2,
            validation_examples: 1,
            token_count: 6,
            backend: "candle-cpu".into(),
            training_status: "completed".into(),
            adapter_checkpoint: "adapter.json".into(),
        }
    }

    fn adapter() -> AdapterCheckpoint {
        let plan = injection_plan("gemma-3-1b-it", &Default::default()).unwrap();
        let trainable = TrainableAdapter::from_plan(plan.clone());
        assert!(!trainable.has_updated_weights());
        plan
    }

    #[test]
    fn compares_measured_predictions_and_reports_delta() {
        let report = evaluate(&manifest(), &adapter(), &[3, 7, 9], &[3, 8, 9], &[3, 7, 9]);

        assert_eq!(report.validation_examples, 1);
        assert_eq!(report.scored_tokens, 3);
        assert_eq!(report.base_token_accuracy, 1.0);
        assert!((report.adapter_token_accuracy - 2.0 / 3.0).abs() < f32::EPSILON);
        assert!((report.improvement + 1.0 / 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn held_out_data_round_trips_and_counts_tokens() {
        let dataset = HeldOutDataset::from_sequences(&[vec![1, 2, 3], vec![4, 5]]).unwrap();
        let path = std::env::temp_dir().join("gemmatune-heldout-test.json");

        dataset.write_to(&path).unwrap();
        let restored = HeldOutDataset::read_from(&path).unwrap();

        assert_eq!(restored, dataset);
        assert_eq!(restored.token_count(), 5);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn held_out_data_rejects_an_empty_split() {
        assert!(HeldOutDataset::from_sequences(&[]).is_err());
        assert_eq!(
            HeldOutDataset::from_sequences(&[vec![42]])
                .unwrap()
                .sequences,
            vec![vec![42]]
        );
    }

    #[test]
    fn report_scores_only_aligned_generated_tokens() {
        let report = evaluate(&manifest(), &adapter(), &[1, 2, 3], &[1], &[1, 2]);

        assert_eq!(report.base_token_accuracy, 1.0);
        assert_eq!(report.adapter_token_accuracy, 1.0);
        assert_eq!(report.improvement, 0.0);
        assert_eq!(report.scored_tokens, 1);
    }
}
