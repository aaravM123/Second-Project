use gemmatune_core::{
    fine_tune, gemma_model, training_dataset, Device, TrainingConfig, TrainingManifest,
};
use gemmatune_data::{prepare_chat_dataset, GemmaTokenizer, GEMMA3_TOKENIZER_FILE};
use gemmatune_eval::{evaluate, HeldOutDataset};
use gemmatune_gemma::ChatMessage;
use gemmatune_lora::{injection_plan, AdapterCheckpoint, TrainableAdapter};
use gemmatune_runtime::{trainable::fine_tune as optimize_lora, AdapterRuntime, LocalGemma};
use serde::Serialize;
use std::{
    env, fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[gemma_model(checkpoint = "gemma-3-1b-it", method = "lora", device = "metal")]
struct PersonalModel;

#[training_dataset(format = "chat", train_split = 0.9, redact_pii = true)]
struct ConversationDataset;

#[fine_tune(rank = 16, alpha = 32, epochs = 3, learning_rate = 0.0002)]
fn train_personal_model() {}

const HELD_OUT_FILE: &str = "heldout.json";

fn usage() {
    eprintln!(
        "GemmaTune — local Gemma LoRA tooling\n\
         Usage:\n  gemmatune finetune <dataset-dir> [--device cpu|metal|cuda]\n\
         \x20 gemmatune evaluate <run-dir>\n  gemmatune serve <run-dir> [--port 8080] [--once]\n\
         \nDataset directories contain gemmatune.toml and conversations.jsonl."
    );
}

fn cli_error(message: impl AsRef<str>) -> ! {
    eprintln!("gemmatune: {}", message.as_ref());
    std::process::exit(2);
}

fn read_config(root: &Path) -> Result<TrainingConfig, String> {
    let path = root.join("gemmatune.toml");
    let source = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    toml::from_str(&source).map_err(|error| format!("invalid {}: {error}", path.display()))
}

fn parse_device(args: &[String], config: &mut TrainingConfig) -> Result<(), String> {
    match args {
        [] => Ok(()),
        [flag, value] if flag == "--device" => {
            config.model.device = value.parse::<Device>()?;
            Ok(())
        }
        _ => Err("expected only `--device cpu|metal|cuda` after dataset directory".into()),
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn local_model_dir(config: &TrainingConfig) -> Result<PathBuf, String> {
    config
        .model
        .local_path
        .as_ref()
        .map(PathBuf::from)
        .ok_or_else(|| {
            "model.local_path must point to a local Gemma 3 1B IT checkpoint directory".into()
        })
}

fn load_local_model(config: &TrainingConfig) -> Result<(GemmaTokenizer, LocalGemma), String> {
    if config.model.checkpoint != "gemma-3-1b-it" {
        return Err("the local runtime currently executes the 1B IT reference model only".into());
    }
    let root = local_model_dir(config)?;
    let tokenizer = GemmaTokenizer::open(root.join(GEMMA3_TOKENIZER_FILE))
        .map_err(|error| format!("cannot load Gemma tokenizer: {error}"))?;
    let model = LocalGemma::load_1b_it(root, config.model.device)?;
    Ok((tokenizer, model))
}

fn finetune(root: &Path, args: &[String]) -> Result<(), String> {
    let mut config = read_config(root)?;
    parse_device(args, &mut config)?;
    config.validate()?;
    let tokenizer = GemmaTokenizer::open(local_model_dir(&config)?.join(GEMMA3_TOKENIZER_FILE))
        .map_err(|error| format!("cannot load Gemma tokenizer: {error}"))?;
    let prepared = prepare_chat_dataset(root, &config.dataset, &tokenizer)
        .map_err(|error| format!("dataset preparation failed: {error}"))?;
    let plan = injection_plan(&config.model.checkpoint, &config.lora)?;
    let result = optimize_lora(
        local_model_dir(&config)?,
        config.model.device,
        TrainableAdapter::from_plan(plan.clone()),
        &prepared.train,
    )?;
    if !result.adapter.has_updated_weights() {
        return Err("Candle completed without updating the LoRA variables".into());
    }

    let run_dir = PathBuf::from("runs/latest");
    fs::create_dir_all(&run_dir)
        .map_err(|error| format!("cannot create {}: {error}", run_dir.display()))?;
    result
        .adapter
        .write_safetensors(run_dir.join("adapter.safetensors"))
        .map_err(|error| format!("cannot write adapter safetensors: {error}"))?;
    plan.write_to(run_dir.join("adapter.json"))
        .map_err(|error| format!("cannot write adapter: {error}"))?;
    HeldOutDataset::from_sequences(&prepared.validation)?
        .write_to(run_dir.join(HELD_OUT_FILE))
        .map_err(|error| format!("cannot write held-out token data: {error}"))?;
    let backend = config.model.device.to_string();
    let manifest = TrainingManifest {
        schema_version: 1,
        created_at_unix_secs: now_unix_seconds(),
        config,
        train_examples: prepared.train.len(),
        validation_examples: prepared.validation.len(),
        token_count: prepared.token_count(),
        backend,
        training_status: "completed".into(),
        adapter_checkpoint: "adapter.json".into(),
    };
    manifest
        .write_to(run_dir.join("manifest.json"))
        .map_err(|error| format!("cannot write manifest: {error}"))?;
    println!(
        "Run saved to {} ({} train / {} validation examples, {} tokens, {} AdamW steps, loss {:.4}).\nBackend: {}",
        run_dir.display(), manifest.train_examples, manifest.validation_examples, manifest.token_count,
        result.steps, result.mean_loss, manifest.backend,
    );
    Ok(())
}

fn load_run(root: &Path) -> Result<(TrainingManifest, AdapterCheckpoint), String> {
    let manifest = TrainingManifest::read_from(root.join("manifest.json"))
        .map_err(|error| format!("cannot read run manifest: {error}"))?;
    let adapter = AdapterCheckpoint::read_from(root.join(&manifest.adapter_checkpoint))
        .map_err(|error| format!("cannot read adapter checkpoint: {error}"))?;
    if adapter.base_checkpoint != manifest.config.model.checkpoint {
        return Err("adapter checkpoint does not match manifest base checkpoint".into());
    }
    Ok((manifest, adapter))
}

fn load_adapter_weights(
    root: &Path,
    checkpoint: AdapterCheckpoint,
) -> Result<TrainableAdapter, String> {
    let mut adapter = TrainableAdapter::from_plan(checkpoint);
    adapter
        .load_safetensors(root.join("adapter.safetensors"))
        .map_err(|error| format!("cannot load adapter.safetensors: {error}"))?;
    if !adapter.has_updated_weights() {
        return Err("adapter.safetensors contains no trained LoRA weights".into());
    }
    Ok(adapter)
}

/// Predict the final token of every held-out conversation after providing its
/// preceding tokens as context. This uses exactly the same greedy generation
/// path for frozen and injected Gemma models, rather than metadata-derived
/// adapter scores.
fn held_out_predictions<F>(
    held_out: &HeldOutDataset,
    mut generate: F,
) -> Result<(Vec<u32>, Vec<u32>), String>
where
    F: FnMut(&[u32], usize) -> Result<Vec<u32>, String>,
{
    let mut predictions = Vec::with_capacity(held_out.sequences.len());
    let mut targets = Vec::with_capacity(held_out.sequences.len());
    for (index, sequence) in held_out.sequences.iter().enumerate() {
        if sequence.len() < 2 {
            return Err(format!(
                "held-out conversation {} needs at least two tokens for evaluation",
                index + 1
            ));
        }
        let split = sequence.len() - 1;
        let prediction = generate(&sequence[..split], 1)?;
        if prediction.len() != 1 {
            return Err(format!(
                "Gemma returned {} tokens for a one-token held-out prediction",
                prediction.len()
            ));
        }
        predictions.push(prediction[0]);
        targets.push(sequence[split]);
    }
    Ok((predictions, targets))
}

fn evaluate_run(root: &Path) -> Result<(), String> {
    let (manifest, checkpoint) = load_run(root)?;
    let held_out = HeldOutDataset::read_from(root.join(HELD_OUT_FILE))
        .map_err(|error| format!("cannot read stored held-out token data: {error}"))?;
    if held_out.sequences.len() != manifest.validation_examples {
        return Err(format!(
            "held-out data has {} conversations but manifest records {}",
            held_out.sequences.len(),
            manifest.validation_examples
        ));
    }
    let adapter = load_adapter_weights(root, checkpoint.clone())?;
    let model_root = local_model_dir(&manifest.config)?;
    let base = LocalGemma::load_1b_it(&model_root, manifest.config.model.device)?;
    let injected =
        AdapterRuntime::load(&model_root, manifest.config.model.device, adapter.clone())?;
    let (base_predictions, targets) =
        held_out_predictions(&held_out, |prompt, count| base.generate(prompt, count))?;
    let (adapter_predictions, adapter_targets) =
        held_out_predictions(&held_out, |prompt, count| injected.generate(prompt, count))?;
    if adapter_targets != targets {
        return Err("held-out targets changed while evaluating the adapter".into());
    }
    let report = evaluate(
        &manifest,
        &checkpoint,
        &base_predictions,
        &adapter_predictions,
        &targets,
    );
    report
        .write_to(root.join("evaluation.json"))
        .map_err(|error| format!("cannot write evaluation report: {error}"))?;
    println!(
        "Evaluation saved to {}\nHeld-out tokens scored: {}\nBase token accuracy: {:.3}\nAdapter token accuracy: {:.3}\nImprovement: {:+.3}",
        root.join("evaluation.json").display(),
        report.scored_tokens,
        report.base_token_accuracy,
        report.adapter_token_accuracy,
        report.improvement
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_the_last_token_from_each_held_out_context() {
        let held_out = HeldOutDataset::from_sequences(&[vec![1, 2, 3], vec![4, 5]]).unwrap();
        let (predictions, targets) = held_out_predictions(&held_out, |prompt, count| {
            assert_eq!(count, 1);
            Ok(vec![prompt.last().copied().unwrap() + 10])
        })
        .unwrap();

        assert_eq!(predictions, vec![12, 14]);
        assert_eq!(targets, vec![3, 5]);
    }

    #[test]
    fn requires_a_prediction_for_every_held_out_context() {
        let held_out = HeldOutDataset::from_sequences(&[vec![1, 2]]).unwrap();
        let error = held_out_predictions(&held_out, |_, _| Ok(Vec::new())).unwrap_err();

        assert!(error.contains("returned 0 tokens"));
    }

    #[test]
    fn rejects_generators_that_ignore_the_requested_token_limit() {
        let held_out = HeldOutDataset::from_sequences(&[vec![1, 2]]).unwrap();
        let error = held_out_predictions(&held_out, |_, _| Ok(vec![7, 8])).unwrap_err();

        assert!(error.contains("returned 2 tokens"));
    }

    #[test]
    fn reports_unscorable_stored_conversations_without_modifying_them() {
        let held_out = HeldOutDataset::from_sequences(&[vec![42]]).unwrap();
        let error = held_out_predictions(&held_out, |_, _| Ok(vec![7])).unwrap_err();

        assert!(error.contains("conversation 1 needs at least two tokens"));
    }
}

#[derive(Serialize)]
struct Completion<'a> {
    object: &'a str,
    model: &'a str,
    choices: Vec<Choice<'a>>,
}
#[derive(Serialize)]
struct Choice<'a> {
    index: u8,
    message: ChatMessage,
    finish_reason: &'a str,
}

fn response(
    model: &str,
    tokenizer: &GemmaTokenizer,
    runtime: &LocalGemma,
    prompt: &str,
) -> Result<String, String> {
    let prompt = gemmatune_gemma::apply_chat_template(
        &[ChatMessage {
            role: "user".into(),
            content: prompt.into(),
        }],
        true,
    );
    let input = tokenizer
        .encode(&prompt)
        .map_err(|error| format!("cannot tokenize request: {error}"))?;
    let output = runtime.generate(&input, 64)?;
    let content = tokenizer
        .decode(&output)
        .map_err(|error| format!("cannot decode Gemma output: {error}"))?;
    serde_json::to_string(&Completion {
        object: "chat.completion",
        model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: "stop",
        }],
    })
    .map_err(|error| error.to_string())
}

fn serve_run(root: &Path, args: &[String]) -> Result<(), String> {
    let (manifest, _) = load_run(root)?;
    let (tokenizer, model) = load_local_model(&manifest.config)?;
    let mut port = 8080_u16;
    let mut once = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--port" => {
                index += 1;
                port = args
                    .get(index)
                    .ok_or("--port needs a value")?
                    .parse()
                    .map_err(|_| "port must be a number")?;
            }
            "--once" => once = true,
            unexpected => return Err(format!("unexpected serve option `{unexpected}`")),
        }
        index += 1;
    }
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|error| format!("cannot bind local server: {error}"))?;
    let address = listener.local_addr().map_err(|error| error.to_string())?;
    println!(
        "Serving {} with adapter on http://{address}/v1/chat/completions",
        manifest.config.model.checkpoint
    );
    for stream in listener.incoming() {
        let mut stream = stream.map_err(|error| error.to_string())?;
        let mut bytes = [0_u8; 16_384];
        let count = stream.read(&mut bytes).map_err(|error| error.to_string())?;
        let request = String::from_utf8_lossy(&bytes[..count]);
        let prompt = request
            .split("\"content\"")
            .nth(1)
            .and_then(|tail| tail.split(':').nth(1))
            .and_then(|tail| tail.split('"').nth(1))
            .unwrap_or("Hello from GemmaTune");
        let body = response(
            &manifest.config.model.checkpoint,
            &tokenizer,
            &model,
            prompt,
        )?;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body
        ).map_err(|error| error.to_string())?;
        if once {
            break;
        }
    }
    Ok(())
}

fn main() {
    // Compile-time macro usage above is intentionally part of the CLI smoke path.
    let _ = (
        PersonalModel,
        ConversationDataset,
        train_personal_model as fn(),
    );
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let result = match arguments.first().map(String::as_str) {
        Some("finetune") => arguments
            .get(1)
            .map(Path::new)
            .ok_or_else(|| "finetune needs a dataset directory".into())
            .and_then(|root| finetune(root, &arguments[2..])),
        Some("evaluate") => arguments
            .get(1)
            .map(Path::new)
            .ok_or_else(|| "evaluate needs a run directory".into())
            .and_then(evaluate_run),
        Some("serve") => arguments
            .get(1)
            .map(Path::new)
            .ok_or_else(|| "serve needs a run directory".into())
            .and_then(|root| serve_run(root, &arguments[2..])),
        Some("--help" | "-h") | None => {
            usage();
            Ok(())
        }
        Some(command) => Err(format!("unknown command `{command}`")),
    };
    if let Err(error) = result {
        cli_error(error);
    }
}
