# GemmaTune

GemmaTune is a Rust-native, fully local SDK and CLI for configuring Gemma LoRA
fine-tuning runs. It never downloads a model, transmits prompts, or requires a
Python runtime. The first milestone provides an inspectable end-to-end pipeline
for Gemma 3 1B IT, plus 4B architecture metadata for higher-memory machines.

## Quick start

```bash
cargo run -p gemmatune-cli -- train ./examples/writing-style
cargo run -p gemmatune-cli -- evaluate ./runs/latest
cargo run -p gemmatune-cli -- serve ./runs/latest --port 8080
```

The `train` command prepares local JSONL chat data, optionally redacts simple
email/phone-like values, applies the Gemma chat template, creates a validated
LoRA injection plan, and writes `runs/latest/manifest.json` plus
`runs/latest/adapter.json`. `evaluate` writes `evaluation.json`.

`serve` binds only to `127.0.0.1` and exposes `POST /v1/chat/completions`. Add
`--once` when using it in scripts or tests.

## SDK attributes

The public macros are re-exported by `gemmatune-core`:

```rust
use gemmatune_core::{fine_tune, gemma_model, training_dataset};

#[gemma_model(checkpoint = "gemma-3-1b-it", method = "lora", device = "metal")]
struct PersonalModel;

#[training_dataset(format = "chat", train_split = 0.9, redact_pii = true)]
struct ConversationDataset;

#[fine_tune(rank = 16, alpha = 32, epochs = 3, learning_rate = 0.0002)]
fn train_personal_model() {}
```

Macros reject unsupported checkpoint/device/dataset choices and non-positive
training values during compilation.

## Run configuration and data

Each input directory needs `gemmatune.toml` and JSONL conversations such as:

```json
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"}]}
```

See [`examples/writing-style`](examples/writing-style). The TOML has `model`,
`dataset`, and `lora` tables and is deliberately simple enough to inspect and
check into source control.

## Workspace crates

| Crate | Responsibility |
| --- | --- |
| `gemmatune-macros` | Attribute macros and compile-time validation |
| `gemmatune-core` | Public configuration and training manifests |
| `gemmatune-gemma` | Gemma 3 metadata and chat template |
| `gemmatune-data` | Local JSONL preparation, redaction, tokenization |
| `gemmatune-lora` | LoRA injection plans and adapter checkpoints |
| `gemmatune-runtime` | CPU/Metal/CUDA backend selection |
| `gemmatune-eval` | Base-vs-adapter evaluation reports |
| `gemmatune-cli` | `train`, `evaluate`, and localhost `serve` |

## Milestone boundary

The pipeline, validation, manifests, adapters, evaluation report, and HTTP API
are functional today. Tokenization is deterministic word hashing rather than
Gemma's SentencePiece tokenizer; CPU computes a dry-run loss rather than
updating tensors; and Metal/CUDA are backend-selection stubs. Consequently the
adapter and evaluation score are pipeline artifacts, not trained model weights
or quality claims. Real Gemma/SentencePiece loading and accelerated kernels are
the next required runtime work before production fine-tuning.
