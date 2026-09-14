//! GemmaTune's stable public configuration and run-manifest API.
//! All persisted files are JSON so local runs remain inspectable and portable.

use serde::{Deserialize, Serialize};
use std::{fmt, fs, io, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        })
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!(
                "unsupported device `{value}`; expected cpu, metal, or cuda"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub checkpoint: String,
    #[serde(default)]
    pub local_path: Option<String>,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub device: Device,
}

fn default_method() -> String {
    "lora".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: u16,
    pub alpha: f32,
    pub epochs: u16,
    pub learning_rate: f32,
    #[serde(default = "default_targets")]
    pub target_modules: Vec<String>,
}

fn default_targets() -> Vec<String> {
    ["q_proj", "k_proj", "v_proj", "o_proj"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            epochs: 3,
            learning_rate: 0.0002,
            target_modules: default_targets(),
        }
    }
}

impl LoraConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.rank == 0 {
            return Err("LoRA rank must be greater than zero".into());
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err("LoRA alpha must be positive".into());
        }
        if self.epochs == 0 {
            return Err("epochs must be greater than zero".into());
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.target_modules.is_empty() {
            return Err("at least one LoRA target module is required".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetConfig {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_train_split")]
    pub train_split: f32,
    #[serde(default)]
    pub redact_pii: bool,
    #[serde(default = "default_data_file")]
    pub file: String,
}

fn default_format() -> String {
    "chat".into()
}
fn default_train_split() -> f32 {
    0.9
}
fn default_data_file() -> String {
    "conversations.jsonl".into()
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            format: default_format(),
            train_split: default_train_split(),
            redact_pii: true,
            file: default_data_file(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    #[serde(default)]
    pub dataset: DatasetConfig,
    #[serde(default)]
    pub lora: LoraConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                checkpoint: "gemma-3-1b-it".into(),
                local_path: None,
                method: default_method(),
                device: Device::Cpu,
            },
            dataset: DatasetConfig::default(),
            lora: LoraConfig::default(),
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(
            self.model.checkpoint.as_str(),
            "gemma-3-1b-it" | "gemma-3-4b-it"
        ) {
            return Err("checkpoint must be gemma-3-1b-it or gemma-3-4b-it".into());
        }
        if self.model.method != "lora" {
            return Err("only the lora training method is supported".into());
        }
        if self.dataset.format != "chat" {
            return Err("only chat datasets are supported in this milestone".into());
        }
        if !(0.0..1.0).contains(&self.dataset.train_split) {
            return Err(
                "dataset train_split must be between 0 (exclusive) and 1 (exclusive)".into(),
            );
        }
        self.lora.validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingManifest {
    pub schema_version: u8,
    pub created_at_unix_secs: u64,
    pub config: TrainingConfig,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub token_count: usize,
    pub backend: String,
    pub training_status: String,
    pub adapter_checkpoint: String,
}

impl TrainingManifest {
    pub fn write_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn read_from(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_lora() {
        let mut config = LoraConfig::default();
        config.rank = 0;
        assert!(config.validate().is_err());
    }
}
