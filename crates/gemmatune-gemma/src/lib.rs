//! Architecture metadata and the Gemma instruction chat template.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GemmaArchitecture {
    pub checkpoint: String,
    pub parameters_billions: u8,
    pub hidden_size: u16,
    pub layers: u8,
    pub vocabulary_size: usize,
    pub attention: Option<GemmaAttentionConfig>,
    pub context_length: usize,
}

/// Text-only Gemma 3 attention parameters used by the 1B IT runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GemmaAttentionConfig {
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub local_attention_window: usize,
    pub local_rope_theta: f64,
    pub global_rope_theta: f64,
    /// Gemma uses five local layers followed by one global layer.
    pub local_attention_pattern: usize,
}

impl GemmaAttentionConfig {
    pub fn is_local_layer(&self, layer_index: usize) -> bool {
        (layer_index + 1) % self.local_attention_pattern != 0
    }
}

pub fn architecture(checkpoint: &str) -> Option<GemmaArchitecture> {
    match checkpoint {
        "gemma-3-1b-it" => Some(GemmaArchitecture {
            checkpoint: checkpoint.into(),
            parameters_billions: 1,
            hidden_size: 1152,
            layers: 26,
            vocabulary_size: 262_144,
            attention: Some(GemmaAttentionConfig {
                intermediate_size: 6912,
                query_heads: 4,
                key_value_heads: 1,
                head_dim: 256,
                local_attention_window: 512,
                local_rope_theta: 10_000.0,
                global_rope_theta: 1_000_000.0,
                local_attention_pattern: 6,
            }),
            context_length: 32_768,
        }),
        "gemma-3-4b-it" => Some(GemmaArchitecture {
            checkpoint: checkpoint.into(),
            parameters_billions: 4,
            hidden_size: 2560,
            layers: 34,
            vocabulary_size: 262_144,
            // 1B is the supported implementation target. Keep 4B as a
            // higher-memory model identifier without asserting its full shape.
            attention: None,
            context_length: 131_072,
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub const EOS_TOKEN: &str = "<eos>";

fn gemma_role(role: &str) -> &str {
    // Common chat exports call the model turn `assistant`; Gemma IT calls it
    // `model`, and the serialized prompt must always use the latter.
    match role {
        "assistant" | "model" => "model",
        "user" => "user",
        _ => role,
    }
}

pub fn apply_chat_template(messages: &[ChatMessage], add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    for message in messages {
        rendered.push_str("<start_of_turn>");
        rendered.push_str(gemma_role(&message.role));
        rendered.push('\n');
        rendered.push_str(&message.content);
        rendered.push_str("<end_of_turn>");
        rendered.push_str(EOS_TOKEN);
        rendered.push('\n');
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
            "<start_of_turn>user\nHello<end_of_turn><eos>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn normalizes_assistant_turns_to_model() {
        let prompt = apply_chat_template(
            &[ChatMessage {
                role: "assistant".into(),
                content: "Hello".into(),
            }],
            false,
        );
        assert!(prompt.starts_with("<start_of_turn>model\n"));
    }

    #[test]
    fn identifies_five_local_then_global_layers() {
        let attention = architecture("gemma-3-1b-it").unwrap().attention.unwrap();
        assert!(attention.is_local_layer(4));
        assert!(!attention.is_local_layer(5));
    }
}
