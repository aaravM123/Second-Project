use gemmatune_core::{
    fine_tune, gemma_model, training_dataset, Device, TrainingConfig, TrainingManifest,
};
use gemmatune_data::prepare_chat_dataset;
use gemmatune_eval::evaluate;
use gemmatune_gemma::{apply_chat_template, ChatMessage};
use gemmatune_lora::{injection_plan, AdapterCheckpoint};
use gemmatune_runtime::train_dry_run;
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

fn usage() {
    eprintln!(
        "GemmaTune — local Gemma LoRA tooling\n\
         Usage:\n  gemmatune train <dataset-dir> [--device cpu|metal|cuda]\n\
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

fn train(root: &Path, args: &[String]) -> Result<(), String> {
    let mut config = read_config(root)?;
    parse_device(args, &mut config)?;
    config.validate()?;
    let prepared = prepare_chat_dataset(root, &config.dataset)
        .map_err(|error| format!("dataset preparation failed: {error}"))?;
    let adapter = injection_plan(&config.model.checkpoint, &config.lora)?;
    let outcome = train_dry_run(
        &adapter,
        prepared.train.len(),
        prepared.token_count(),
        config.model.device,
    );

    let run_dir = PathBuf::from("runs/latest");
    fs::create_dir_all(&run_dir)
        .map_err(|error| format!("cannot create {}: {error}", run_dir.display()))?;
    let adapter_path = run_dir.join("adapter.json");
    adapter
        .write_to(&adapter_path)
        .map_err(|error| format!("cannot write adapter: {error}"))?;
    let manifest = TrainingManifest {
        schema_version: 1,
        created_at_unix_secs: now_unix_seconds(),
        config,
        train_examples: prepared.train.len(),
        validation_examples: prepared.validation.len(),
        token_count: prepared.token_count(),
        backend: outcome.backend.implementation.clone(),
        training_status: outcome.status.clone(),
        adapter_checkpoint: "adapter.json".into(),
    };
    manifest
        .write_to(run_dir.join("manifest.json"))
        .map_err(|error| format!("cannot write manifest: {error}"))?;
    println!(
        "Run saved to {} ({} train / {} validation examples, {} tokens).\nBackend: {} — {}\nSteps: {}, dry-run loss: {:.4}",
        run_dir.display(), manifest.train_examples, manifest.validation_examples, manifest.token_count,
        outcome.backend.implementation, outcome.backend.note, outcome.steps, outcome.final_loss,
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

fn evaluate_run(root: &Path) -> Result<(), String> {
    let (manifest, adapter) = load_run(root)?;
    let report = evaluate(&manifest, &adapter);
    report
        .write_to(root.join("evaluation.json"))
        .map_err(|error| format!("cannot write evaluation report: {error}"))?;
    println!(
        "Evaluation saved to {}\nBase style score: {:.3}\nAdapter style score: {:.3}\nImprovement: +{:.3}\n{}",
        root.join("evaluation.json").display(), report.base_style_score, report.adapter_style_score, report.improvement, report.note
    );
    Ok(())
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

fn response(model: &str, prompt: &str) -> String {
    let template = apply_chat_template(
        &[ChatMessage {
            role: "user".into(),
            content: prompt.into(),
        }],
        true,
    );
    let content = format!(
        "GemmaTune local runtime stub received your prompt ({} template bytes). Enable a Metal or CUDA kernel to generate Gemma tokens.",
        template.len()
    );
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
    .expect("serializing a fixed response cannot fail")
}

fn serve_run(root: &Path, args: &[String]) -> Result<(), String> {
    let (manifest, _) = load_run(root)?;
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
        let body = response(&manifest.config.model.checkpoint, prompt);
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
        Some("train") => arguments
            .get(1)
            .map(Path::new)
            .ok_or_else(|| "train needs a dataset directory".into())
            .and_then(|root| train(root, &arguments[2..])),
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
