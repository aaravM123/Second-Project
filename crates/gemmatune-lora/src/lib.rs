use gemmatune_core::LoraConfig;
use safetensors::{
    tensor::{Dtype, TensorView},
    SafeTensors,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, io, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdapterTensor {
    pub module: String,
    pub rank: u16,
    pub alpha: f32,
    pub input_features: usize,
    pub output_features: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainableTensor {
    pub spec: AdapterTensor,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    step: u64,
    m_a: Vec<f32>,
    v_a: Vec<f32>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
}

impl TrainableTensor {
    pub fn new(spec: AdapterTensor) -> Self {
        let mut state = spec.seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f32 / u64::MAX as f32) - 0.5) * 0.02
        };
        let a = (0..spec.shape_a.0 * spec.shape_a.1)
            .map(|_| next())
            .collect::<Vec<_>>();
        let b = vec![0.0; spec.shape_b.0 * spec.shape_b.1];
        Self {
            spec,
            m_a: vec![0.0; a.len()],
            v_a: vec![0.0; a.len()],
            m_b: vec![0.0; b.len()],
            v_b: vec![0.0; b.len()],
            a,
            b,
            step: 0,
        }
    }

    pub fn scaling(&self) -> f32 {
        self.spec.alpha / self.spec.rank as f32
    }

    pub fn delta_weight(&self) -> Vec<f32> {
        let input = self.spec.input_features;
        let output = self.spec.output_features;
        let rank = self.spec.rank as usize;
        let mut delta = vec![0.0; input * output];
        for out in 0..output {
            for inner in 0..rank {
                let b = self.b[out * rank + inner] * self.scaling();
                for input_index in 0..input {
                    delta[out * input + input_index] += b * self.a[inner * input + input_index];
                }
            }
        }
        delta
    }

    pub fn apply_adamw(
        &mut self,
        gradient_a: &[f32],
        gradient_b: &[f32],
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<(), String> {
        if gradient_a.len() != self.a.len() || gradient_b.len() != self.b.len() {
            return Err(format!(
                "gradient shape does not match adapter {}",
                self.spec.module
            ));
        }
        self.step += 1;
        Self::adamw_update(
            &mut self.a,
            &mut self.m_a,
            &mut self.v_a,
            gradient_a,
            self.step,
            learning_rate,
            weight_decay,
        );
        Self::adamw_update(
            &mut self.b,
            &mut self.m_b,
            &mut self.v_b,
            gradient_b,
            self.step,
            learning_rate,
            weight_decay,
        );
        Ok(())
    }

    fn adamw_update(
        weights: &mut [f32],
        first_moment: &mut [f32],
        second_moment: &mut [f32],
        gradients: &[f32],
        step: u64,
        learning_rate: f32,
        weight_decay: f32,
    ) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPSILON: f32 = 1e-8;
        let bias1 = 1.0 - BETA1.powi(step as i32);
        let bias2 = 1.0 - BETA2.powi(step as i32);
        for (((weight, moment), variance), gradient) in weights
            .iter_mut()
            .zip(first_moment)
            .zip(second_moment)
            .zip(gradients)
        {
            *moment = BETA1 * *moment + (1.0 - BETA1) * *gradient;
            *variance = BETA2 * *variance + (1.0 - BETA2) * *gradient * *gradient;
            let update = (*moment / bias1) / ((*variance / bias2).sqrt() + EPSILON);
            *weight -= learning_rate * (update + weight_decay * *weight);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainableAdapter {
    pub base_checkpoint: String,
    pub config: LoraConfig,
    pub tensors: Vec<TrainableTensor>,
}

impl TrainableAdapter {
    pub fn from_plan(plan: AdapterCheckpoint) -> Self {
        Self {
            base_checkpoint: plan.base_checkpoint,
            config: plan.config,
            tensors: plan.tensors.into_iter().map(TrainableTensor::new).collect(),
        }
    }

    pub fn has_updated_weights(&self) -> bool {
        self.tensors
            .iter()
            .any(|tensor| tensor.b.iter().any(|value| *value != 0.0))
    }

    pub fn tensor(&self, module: &str) -> Option<&TrainableTensor> {
        self.tensors
            .iter()
            .find(|tensor| tensor.spec.module == module)
    }

    pub fn tensor_mut(&mut self, module: &str) -> Option<&mut TrainableTensor> {
        self.tensors
            .iter_mut()
            .find(|tensor| tensor.spec.module == module)
    }

    fn floats_as_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bytes_as_floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
        if bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err("adapter safetensor data is not f32 aligned".into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect())
    }

    pub fn write_safetensors(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut buffers = HashMap::new();
        for tensor in &self.tensors {
            buffers.insert(
                format!("{}.lora_A", tensor.spec.module),
                Self::floats_as_bytes(&tensor.a),
            );
            buffers.insert(
                format!("{}.lora_B", tensor.spec.module),
                Self::floats_as_bytes(&tensor.b),
            );
        }
        let mut views = HashMap::new();
        for tensor in &self.tensors {
            let a_name = format!("{}.lora_A", tensor.spec.module);
            let b_name = format!("{}.lora_B", tensor.spec.module);
            views.insert(
                a_name.clone(),
                TensorView::new(
                    Dtype::F32,
                    vec![tensor.spec.shape_a.0, tensor.spec.shape_a.1],
                    &buffers[&a_name],
                )
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
            );
            views.insert(
                b_name.clone(),
                TensorView::new(
                    Dtype::F32,
                    vec![tensor.spec.shape_b.0, tensor.spec.shape_b.1],
                    &buffers[&b_name],
                )
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
            );
        }
        safetensors::serialize_to_file(&views, &None, path.as_ref())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }

    pub fn load_safetensors(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = fs::read(path)?;
        let file = SafeTensors::deserialize(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        for tensor in &mut self.tensors {
            let a = file
                .tensor(&format!("{}.lora_A", tensor.spec.module))
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            let b = file
                .tensor(&format!("{}.lora_B", tensor.spec.module))
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            if a.shape() != [tensor.spec.shape_a.0, tensor.spec.shape_a.1]
                || b.shape() != [tensor.spec.shape_b.0, tensor.spec.shape_b.1]
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "adapter tensor shape does not match Gemma module",
                ));
            }
            tensor.a = Self::bytes_as_floats(a.data())
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            tensor.b = Self::bytes_as_floats(b.data())
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        }
        Ok(())
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

    #[test]
    fn adamw_changes_adapter_weights() {
        let plan = injection_plan("gemma-3-1b-it", &LoraConfig::default()).unwrap();
        let mut adapter = TrainableAdapter::from_plan(plan);
        let tensor = adapter.tensor_mut("q_proj").unwrap();
        let before = tensor.b.clone();
        tensor
            .apply_adamw(
                &vec![0.0; tensor.a.len()],
                &vec![0.25; tensor.b.len()],
                0.001,
                0.01,
            )
            .unwrap();
        assert_ne!(before, tensor.b);
        assert!(adapter.has_updated_weights());
    }

    #[test]
    fn builds_projection_delta_in_weight_layout() {
        let plan = injection_plan("gemma-3-1b-it", &LoraConfig::default()).unwrap();
        let adapter = TrainableAdapter::from_plan(plan);
        let delta = adapter.tensor("k_proj").unwrap().delta_weight();
        assert_eq!(delta.len(), 256 * 1152);
    }

    #[test]
    fn round_trips_adapter_safetensors() {
        let plan = injection_plan("gemma-3-1b-it", &LoraConfig::default()).unwrap();
        let mut adapter = TrainableAdapter::from_plan(plan.clone());
        let tensor = adapter.tensor_mut("v_proj").unwrap();
        tensor
            .apply_adamw(
                &vec![0.0; tensor.a.len()],
                &vec![1.0; tensor.b.len()],
                0.01,
                0.0,
            )
            .unwrap();
        let path = std::env::temp_dir().join("gemmatune-adapter-test.safetensors");
        adapter.write_safetensors(&path).unwrap();
        let mut restored = TrainableAdapter::from_plan(plan);
        restored.load_safetensors(&path).unwrap();
        assert_eq!(
            adapter.tensor("v_proj").unwrap().b,
            restored.tensor("v_proj").unwrap().b
        );
        fs::remove_file(path).unwrap();
    }
}
