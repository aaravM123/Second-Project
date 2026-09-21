//! Loss accounting and token batching shared by the adapter trainer.

/// A causal language-model sample with every position shifted one token ahead.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalBatch {
    pub inputs: Vec<u32>,
    pub targets: Vec<u32>,
}

impl CausalBatch {
    pub fn from_tokens(tokens: &[u32]) -> Result<Self, String> {
        if tokens.len() < 2 {
            return Err("a causal training example needs at least two tokens".into());
        }
        Ok(Self {
            inputs: tokens[..tokens.len() - 1].to_vec(),
            targets: tokens[1..].to_vec(),
        })
    }

    pub fn token_count(&self) -> usize {
        self.targets.len()
    }
}

/// Numerically stable negative log-likelihood for one vocabulary-logit row.
pub fn token_cross_entropy(logits: &[f32], target: u32) -> Result<f32, String> {
    let target = target as usize;
    if target >= logits.len() {
        return Err(format!("target token {target} exceeds vocabulary {}", logits.len()));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer = logits.iter().map(|value| (value - maximum).exp()).sum::<f32>().ln() + maximum;
    Ok(normalizer - logits[target])
}

/// Gradient of softmax cross entropy with respect to its logits.
pub fn token_cross_entropy_gradient(logits: &[f32], target: u32) -> Result<Vec<f32>, String> {
    let target = target as usize;
    if target >= logits.len() {
        return Err(format!("target token {target} exceeds vocabulary {}", logits.len()));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents = logits.iter().map(|value| (value - maximum).exp()).collect::<Vec<_>>();
    let normalizer = exponents.iter().sum::<f32>();
    Ok(exponents
        .into_iter()
        .enumerate()
        .map(|(index, value)| value / normalizer - usize::from(index == target) as f32)
        .collect())
}

/// Aggregate teacher-forced sequence loss. Runtime callers feed one logits row
/// for each shifted target in a `CausalBatch`.
pub fn causal_cross_entropy(rows: &[Vec<f32>], batch: &CausalBatch) -> Result<f32, String> {
    if rows.len() != batch.targets.len() {
        return Err(format!(
            "received {} logits rows for {} causal targets",
            rows.len(),
            batch.targets.len()
        ));
    }
    let sum = rows
        .iter()
        .zip(&batch.targets)
        .map(|(row, target)| token_cross_entropy(row, *target))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<f32>();
    Ok(sum / batch.targets.len() as f32)
}

/// Select every token except masked positions. A caller can use this for
/// assistant-only chat losses without allocating a dense attention mask.
pub fn assistant_targets(tokens: &[u32], loss_mask: &[bool]) -> Result<Vec<u32>, String> {
    if tokens.len() != loss_mask.len() {
        return Err("token and loss-mask lengths differ".into());
    }
    Ok(tokens
        .iter()
        .zip(loss_mask)
        .filter_map(|(token, include)| include.then_some(*token))
        .collect())
}

/// L2 gradient norm used to keep a small LoRA update stable on CPU training.
pub fn gradient_norm(gradients: &[&[f32]]) -> f32 {
    gradients
        .iter()
        .flat_map(|gradient| gradient.iter())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

/// Scale gradients in place when their combined norm exceeds `maximum`.
pub fn clip_gradients(gradients: &mut [&mut [f32]], maximum: f32) -> Result<f32, String> {
    if !maximum.is_finite() || maximum <= 0.0 {
        return Err("maximum gradient norm must be positive".into());
    }
    let norm = gradients
        .iter()
        .flat_map(|gradient| gradient.iter())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > maximum {
        let scale = maximum / norm;
        for gradient in gradients {
            for value in gradient.iter_mut() {
                *value *= scale;
            }
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_tokens_for_causal_training() {
        let batch = CausalBatch::from_tokens(&[4, 9, 12]).unwrap();
        assert_eq!(batch.inputs, vec![4, 9]);
        assert_eq!(batch.targets, vec![9, 12]);
    }

    #[test]
    fn computes_cross_entropy_and_gradient() {
        let logits = vec![0.0, 2.0, -1.0];
        assert!(token_cross_entropy(&logits, 1).unwrap() < 0.2);
        let gradient = token_cross_entropy_gradient(&logits, 1).unwrap();
        assert!(gradient[1] < 0.0);
        assert!((gradient.iter().sum::<f32>()).abs() < 1e-6);
    }

    #[test]
    fn averages_each_teacher_forced_position() {
        let batch = CausalBatch::from_tokens(&[0, 1, 2]).unwrap();
        let rows = vec![vec![0.0, 2.0, 0.0], vec![0.0, 0.0, 2.0]];
        assert!(causal_cross_entropy(&rows, &batch).unwrap() < 0.3);
    }

    #[test]
    fn filters_only_assistant_positions() {
        assert_eq!(
            assistant_targets(&[1, 2, 3], &[false, true, true]).unwrap(),
            vec![2, 3]
        );
    }

    #[test]
    fn clips_combined_adapter_gradients() {
        let mut a = vec![3.0, 4.0];
        let mut b = vec![0.0, 0.0];
        assert_eq!(clip_gradients(&mut [&mut a, &mut b], 1.0).unwrap(), 5.0);
        assert!((gradient_norm(&[&a, &b]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_clip_threshold() {
        let mut gradient = vec![1.0];
        assert!(clip_gradients(&mut [&mut gradient], 0.0).is_err());
    }

    #[test]
    fn rejects_mismatched_loss_rows() {
        let batch = CausalBatch::from_tokens(&[1, 2, 3]).unwrap();
        assert!(causal_cross_entropy(&[vec![0.0, 1.0]], &batch).is_err());
    }

    #[test]
    fn rejects_single_token_causal_examples() {
        assert!(CausalBatch::from_tokens(&[42]).is_err());
    }
}
