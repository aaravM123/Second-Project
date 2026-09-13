//! Fully local JSONL chat dataset preparation.

use gemmatune_core::DatasetConfig;
use gemmatune_gemma::{apply_chat_template, ChatMessage};
use serde::Deserialize;
use std::{fs, io, path::Path};

#[derive(Debug, Deserialize)]
struct JsonConversation {
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone)]
pub struct PreparedDataset {
    pub train: Vec<Vec<u32>>,
    pub validation: Vec<Vec<u32>>,
    pub redacted_fields: usize,
}

impl PreparedDataset {
    pub fn token_count(&self) -> usize {
        self.train
            .iter()
            .chain(&self.validation)
            .map(Vec::len)
            .sum()
    }
}

/// Stable local tokenizer placeholder. It preserves the pipeline contract without
/// shipping Google's tokenizer model; replace it with a SentencePiece model at runtime.
pub fn tokenize(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .map(|word| {
            let mut hash: u32 = 2_166_136_261;
            for byte in word.bytes() {
                hash ^= byte as u32;
                hash = hash.wrapping_mul(16_777_619);
            }
            hash % 262_144
        })
        .collect()
}

fn redact(text: &str) -> (String, usize) {
    let mut replacements = 0;
    let words = text
        .split_whitespace()
        .map(|word| {
            if word.contains('@') && word.contains('.') {
                replacements += 1;
                "[REDACTED_EMAIL]".to_owned()
            } else if word.chars().filter(char::is_ascii_digit).count() >= 7 {
                replacements += 1;
                "[REDACTED_NUMBER]".to_owned()
            } else {
                word.to_owned()
            }
        })
        .collect::<Vec<_>>();
    (words.join(" "), replacements)
}

pub fn prepare_chat_dataset(
    root: impl AsRef<Path>,
    config: &DatasetConfig,
) -> io::Result<PreparedDataset> {
    let path = root.as_ref().join(&config.file);
    let source = fs::read_to_string(&path)?;
    let mut records = Vec::new();
    let mut redacted_fields = 0;
    for (index, line) in source
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let mut record: JsonConversation = serde_json::from_str(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.display(), index + 1),
            )
        })?;
        for message in &mut record.messages {
            if config.redact_pii {
                let (content, count) = redact(&message.content);
                message.content = content;
                redacted_fields += count;
            }
        }
        if record.messages.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{} has no messages", path.display(), index + 1),
            ));
        }
        records.push(tokenize(&apply_chat_template(&record.messages, false)));
    }
    if records.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a training dataset needs at least two conversations",
        ));
    }
    let split = ((records.len() as f32) * config.train_split).floor() as usize;
    let split = split.clamp(1, records.len() - 1);
    let validation = records.split_off(split);
    Ok(PreparedDataset {
        train: records,
        validation,
        redacted_fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokenizer_is_deterministic() {
        assert_eq!(tokenize("a local test"), tokenize("a local test"));
    }
}
