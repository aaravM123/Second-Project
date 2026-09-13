//! LoRA injection plans and JSON adapter checkpoints.

use gemmatune_core::LoraConfig;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterTensor {
    pub module: String,
    pub rank: u16,
    pub alpha: f32,
    pub shape_a: (usize, usize),
    pub shape_b: (usize, usize),
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterCheckpoint {
    pub format: String,
    pub base_checkpoint: String,
    pub config: LoraConfig,
    pub tensors: Vec<AdapterTensor>,
}

pub fn injection_plan(
    base_checkpoint: &str,
    config: &LoraConfig,
) -> Result<AdapterCheckpoint, String> {
    config.validate()?;
    let hidden_size = match base_checkpoint {
        "gemma-3-1b-it" => 1152,
        "gemma-3-4b-it" => 2560,
        _ => return Err(format!("unknown Gemma checkpoint `{base_checkpoint}`")),
    };
    let tensors = config
        .target_modules
        .iter()
        .enumerate()
        .map(|(index, module)| AdapterTensor {
            module: module.clone(),
            rank: config.rank,
            alpha: config.alpha,
            shape_a: (hidden_size, config.rank as usize),
            shape_b: (config.rank as usize, hidden_size),
            seed: 0x4745_4d4d_4154_554e_u64 + index as u64,
        })
        .collect();
    Ok(AdapterCheckpoint {
        format: "gemmatune-lora-v1".into(),
        base_checkpoint: base_checkpoint.into(),
        config: config.clone(),
        tensors,
    })
}

impl AdapterCheckpoint {
    pub fn write_to(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let content = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, content)
    }

    pub fn read_from(path: impl AsRef<Path>) -> io::Result<Self> {
        let content = fs::read(path)?;
        serde_json::from_slice(&content)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plans_the_four_attention_projections() {
        assert_eq!(
            injection_plan("gemma-3-1b-it", &LoraConfig::default())
                .unwrap()
                .tensors
                .len(),
            4
        );
    }
}
