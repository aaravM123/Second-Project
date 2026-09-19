//! LoRA injection plans and JSON adapter checkpoints.

use gemmatune_core::LoraConfig;
use serde::{Deserialize, Serialize};
use std::{fs, io, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterTensor {
    pub module: String,
    pub rank: u16,
    pub alpha: f32,
    /// Base linear input width.
    pub input_features: usize,
    /// Base linear output width.
    pub output_features: usize,
    /// LoRA A has layout `[rank, input_features]`.
    pub shape_a: (usize, usize),
    /// LoRA B has layout `[output_features, rank]`.
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
    if base_checkpoint != "gemma-3-1b-it" {
        return Err(format!(
            "exact LoRA layouts are currently implemented only for gemma-3-1b-it, not `{base_checkpoint}`"
        ));
    }
    let tensors = config
        .target_modules
        .iter()
        .enumerate()
        .map(|(index, module)| {
            let (input_features, output_features) = match module.as_str() {
                // Gemma 3 1B: 4 Q heads × 256 dims and 1 KV head × 256 dims.
                "q_proj" => (1152, 1024),
                "k_proj" | "v_proj" => (1152, 256),
                "o_proj" => (1024, 1152),
                unsupported => {
                    return Err(format!(
                        "unsupported Gemma 3 1B LoRA module `{unsupported}`"
                    ))
                }
            };
            Ok(AdapterTensor {
                module: module.clone(),
                rank: config.rank,
                alpha: config.alpha,
                input_features,
                output_features,
                shape_a: (config.rank as usize, input_features),
                shape_b: (output_features, config.rank as usize),
                seed: 0x4745_4d4d_4154_554e_u64 + index as u64,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
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

    #[test]
    fn uses_gqa_projection_dimensions() {
        let plan = injection_plan("gemma-3-1b-it", &LoraConfig::default()).unwrap();
        assert_eq!(plan.tensors[0].shape_a, (16, 1152));
        assert_eq!(plan.tensors[0].shape_b, (1024, 16));
        assert_eq!(plan.tensors[1].shape_b, (256, 16));
        assert_eq!(plan.tensors[3].shape_a, (16, 1024));
    }
}
