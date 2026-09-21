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
├── adapter.safetensors
└── heldout.json
```

`adapter.safetensors` contains LoRA matrices; `adapter.json` records the base
checkpoint and rank/alpha configuration required to attach them safely.
`heldout.json` stores the redacted, tokenized validation conversations used by
the run. It is an evaluation artifact, not a copy of the source JSONL file.
Runs are local and resumable from their stored adapter values; base checkpoint
weights are never modified by the fine-tuning command.
Use the same checkpoint directory when resuming a run.
Keep adapter artifacts with their matching model revision.

## Evaluate an adapter

```bash
gemmatune evaluate ./runs/latest
```

Evaluation loads `adapter.safetensors`, rebuilds a separate adapter-injected
Gemma model, and compares it with a frozen-base model. For each stored held-out
conversation, both models receive every token except the final one and greedily
predict that final token. `evaluation.json` records the two measured accuracies
and their difference; it does not infer a score from rank, alpha, or adapter
metadata.

Evaluation needs `manifest.json`, `adapter.json`, `adapter.safetensors`, and
`heldout.json` from the same completed run, plus the local base checkpoint
named by `model.local_path`. It never retrains, rewrites adapter weights, or
reads the original dataset directory.

## Serve locally

```bash
gemmatune serve ./runs/latest --port 8080
```

The server listens only on `127.0.0.1` and accepts
`POST /v1/chat/completions`. Before it binds the port, `serve` reads
`manifest.json` and `adapter.json`, loads `adapter.safetensors`, and merges its
LoRA matrices into the matching Q/K/V/O Gemma projections. It therefore serves
the trained adapter-injected model, never a fallback frozen-base model.

For example:

```bash
curl --json '{
  "model": "gemma-3-1b-it",
  "messages": [
    {"role": "user", "content": "Write a one-line welcome message."}
  ],
  "max_tokens": 48
}' http://127.0.0.1:8080/v1/chat/completions
```

Requests use a JSON subset compatible with OpenAI chat completions:

- `messages` is required and must contain non-empty `user`, `assistant`, or
  `model` text turns. Previous assistant turns are rendered as Gemma `model`
  turns.
- `model`, when supplied, must exactly match the checkpoint recorded in the
  run manifest. This prevents accidentally addressing an incompatible adapter.
- `max_tokens` defaults to 64 and is limited to 1–512. The prompt and output
  together must still fit Gemma's 32K-token context window.
- Streaming is not implemented; omit `stream` or set it to `false`.

Malformed HTTP, JSON, or request fields receive a JSON `400 Bad Request`
response. The server processes one request per connection and caps request
bodies at 64 KiB. It is intentionally loopback-only, so exposing it beyond
the local machine requires a separate, authenticated reverse proxy.

## Current constraints

GemmaTune currently targets text-only Gemma 3 1B IT. CPU execution is the
reference training path. Metal and CUDA builds remain optional Candle features
for compatible machines. The 4B model is a higher-memory future target.
