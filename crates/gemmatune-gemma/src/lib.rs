//! Architecture metadata and the Gemma instruction chat template.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GemmaArchitecture {
    pub checkpoint: String,
    pub parameters_billions: u8,
    pub hidden_size: u16,
    pub layers: u8,
    pub vocabulary_size: usize,
}

pub fn architecture(checkpoint: &str) -> Option<GemmaArchitecture> {
    match checkpoint {
        "gemma-3-1b-it" => Some(GemmaArchitecture {
            checkpoint: checkpoint.into(),
            parameters_billions: 1,
            hidden_size: 1152,
            layers: 26,
            vocabulary_size: 262_144,
        }),
        "gemma-3-4b-it" => Some(GemmaArchitecture {
            checkpoint: checkpoint.into(),
            parameters_billions: 4,
            hidden_size: 2560,
            layers: 34,
            vocabulary_size: 262_144,
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn apply_chat_template(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str("<start_of_turn>");
        rendered.push_str(&message.role);
        rendered.push('\n');
        rendered.push_str(&message.content);
        rendered.push_str("<end_of_turn>\n");
    }
    if add_generation_prompt {
        rendered.push_str("<start_of_turn>model\n");
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn renders_generation_prompt() {
        let prompt = apply_chat_template(
            &[ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            }],
            true,
        );
        assert_eq!(
            prompt,
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
    }
}
