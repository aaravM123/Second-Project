# GemmaTune

GemmaTune is a local Rust SDK for fine-tuning Gemma 3 1B IT with LoRA. It
loads a checkpoint from disk and does not send prompts, training examples, or
adapter weights to a hosted service.

## Local checkpoint layout

Set `model.local_path` in the dataset's `gemmatune.toml` to a Gemma 3 1B IT
directory containing:

```text
gemma-3-1b-it/
├── gemma3_cleaned_262144_v2.spiece.model
└── model-00001-of-0000N.safetensors
```

All safetensor shards in that directory are loaded. The reference runtime is
Gemma 3 1B IT: 26 decoder layers, 4 query heads, 1 KV head, and 32K context.

## Fine-tune locally

```bash
gemmatune finetune ./examples/writing-style
```

`finetune` formats conversations with Gemma IT turns, tokenizes them with the
local SentencePiece model, and teacher-forces shifted causal targets. It keeps
the base checkpoint frozen and optimizes only per-layer LoRA A/B matrices on
Q/K/V/O projections using AdamW.

The command writes an inspectable run directory:

```text
runs/latest/
├── manifest.json
├── adapter.json
└── adapter.safetensors
```

`adapter.safetensors` contains LoRA matrices; `adapter.json` records the base
checkpoint and rank/alpha configuration required to attach them safely.
Runs are local and resumable from their stored adapter values; base checkpoint
weights are never modified by the fine-tuning command.
Use the same checkpoint directory when resuming a run.
Keep adapter artifacts with their matching model revision.

## Evaluate an adapter

```bash
gemmatune evaluate ./runs/latest
```

Evaluation runs the frozen base and adapter-injected model on held-out token
sequences. It reports measured token accuracy for both paths rather than a
rank-derived synthetic score.

## Serve locally

```bash
gemmatune serve ./runs/latest --port 8080
```

The server listens only on `127.0.0.1` and accepts
`POST /v1/chat/completions`. It applies the Gemma IT `user`/`model` prompt
template, tokenizes locally, and generates from the run's adapter.

## Current constraints

GemmaTune currently targets text-only Gemma 3 1B IT. CPU execution is the
reference training path. Metal and CUDA builds remain optional Candle features
for compatible machines. The 4B model is a higher-memory future target.
