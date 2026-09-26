//! `needle` — the Needle 3 command line, in Rust.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use needle_core::render::build_prompt;
use needle_engine::generate::{GenOptions, generate};
use needle_engine::loader;

mod fetch;
mod generate;
mod net;
mod platform;
mod playground;

#[derive(Parser)]
#[command(name = "needle", version, about = "Needle 3: tool calling for tiny devices", disable_help_subcommand = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum PlatformVerb {
    /// Upload data, fine-tune every size, download the .cact files.
    Finetune {
        train: String,
        validation: String,
        test: String,
        #[arg(long)]
        max_depth: Option<u64>,
        #[arg(long)]
        suffix: Option<String>,
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long)]
        depth: Option<u64>,
        #[arg(long)]
        no_wait: bool,
    },
    /// Generate training data from tool definitions.
    Generate {
        #[arg(long)]
        tools: PathBuf,
        #[arg(long, default_value_t = 1000)]
        examples: usize,
        #[arg(long)]
        description: Option<String>,
        #[arg(long)]
        message: Option<Vec<String>>,
        #[arg(long)]
        suffix: Option<String>,
        #[arg(long, default_value = ".")]
        out: PathBuf,
        #[arg(long)]
        no_wait: bool,
    },
    Jobs {
        job_id: Option<String>,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        depth: Option<u64>,
    },
    Models {
        model_id: Option<String>,
    },
    Files,
    Billing,
}

#[derive(Subcommand)]
enum Command {
    /// Run a checkpoint (or .cact archive) on a query.
    Run {
        #[arg(long)]
        checkpoint: PathBuf,
        /// Query text for tool-call generation.
        #[arg(long)]
        query: Option<String>,
        /// Tools JSON for tool-call generation.
        #[arg(long)]
        tools: Option<PathBuf>,
        #[arg(long, default_value_t = 512)]
        max_len: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Print prefill and decode speed to stderr.
        #[arg(long)]
        stats: bool,
    },
    /// Train a LoRA adapter on JSONL data.
    Finetune {
        /// Path to JSONL training data.
        jsonl_path: PathBuf,
        /// Base model checkpoint.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 16)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-4)]
        lr: f32,
        /// LoRA adapter rank.
        #[arg(long, default_value_t = 16)]
        lora_rank: usize,
        /// LoRA scaling alpha.
        #[arg(long, default_value_t = 32.0)]
        lora_alpha: f32,
        /// Max training sequence length.
        #[arg(long, default_value_t = 1024)]
        max_len: usize,
        /// Fraction of examples held out for validation (0 disables).
        #[arg(long, default_value_t = 0.1)]
        val_split: f64,
        /// Random seed for LoRA init, validation split, and epoch shuffling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "checkpoints")]
        checkpoint_dir: PathBuf,
        /// Output adapter path (.safetensors).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Skip the held-out exact-call scoring after training.
        #[arg(long)]
        no_score: bool,
        /// Stop after N optimizer steps (benchmarking).
        #[arg(long)]
        max_steps: Option<usize>,
        /// Generate N extra examples via OpenRouter before training (0 = off).
        #[arg(long, default_value_t = 0)]
        generate: usize,
        /// OpenRouter model for --generate.
        #[arg(long, default_value = generate::DEFAULT_MODEL)]
        model: String,
        /// Concurrent OpenRouter requests when generating.
        #[arg(long, default_value_t = 8)]
        workers: usize,
    },
    /// Synthesise training data via OpenRouter.
    GenerateData {
        /// Tool schemas JSON to seed generation.
        #[arg(long)]
        tools: Option<PathBuf>,
        /// Existing JSONL to expand.
        #[arg(long)]
        augment: Option<PathBuf>,
        #[arg(long, default_value_t = 100)]
        num_samples: usize,
        #[arg(long, default_value_t = 25)]
        batch_size: usize,
        /// Concurrent OpenRouter requests.
        #[arg(long, default_value_t = 16)]
        workers: usize,
        #[arg(long, default_value = generate::DEFAULT_MODEL)]
        model: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// needle3 | needle3.safetensors | <platform> | model-<id> | <org>/<repo>[/<file>.cact]
    Download {
        spec: String,
        /// Directory to place the files.
        #[arg(long, default_value = ".")]
        out: PathBuf,
        /// One size of a fine-tuned model-<id> (default: every size).
        #[arg(long)]
        depth: Option<u64>,
        /// Engine generation when downloading a platform build.
        #[arg(long, default_value_t = 3)]
        generation: u32,
    },
    /// Fetch the native engine library for this platform.
    Fetch {
        /// Directory to place the engine (default: the cache).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Fetch the build for another device, e.g. manylinux2014_aarch64.
        #[arg(long)]
        platform_tag: Option<String>,
        #[arg(long, default_value_t = 3)]
        generation: u32,
    },
    /// Fine-tune, generate data and download models on cactuscompute.com.
    Platform {
        #[command(subcommand)]
        verb: PlatformVerb,
    },
    /// Serve the browser playground.
    Playground {
        /// Tuned .cact to serve (defaults to the base model).
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long, default_value_t = 7860)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// The native runner's HTTP mode: POST /complete {"input"}, POST /reset.
    Serve {
        #[arg(long, default_value = "models/needle3.cact")]
        model: PathBuf,
        #[arg(long)]
        tools: Option<PathBuf>,
        /// System facts file.
        #[arg(long)]
        system: Option<PathBuf>,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 512)]
        max: usize,
    },
    /// Export a checkpoint (+ adapter) to a .cact archive.
    Build {
        /// Base checkpoint (.safetensors); defaults to the adapter's base.
        checkpoint: Option<PathBuf>,
        /// LoRA adapter to merge before export.
        #[arg(long)]
        lora: Option<PathBuf>,
        /// Output .cact path.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Export the N-layer rung of the base (2..20); default the full depth.
        #[arg(long)]
        layers: Option<usize>,
        /// The published base archive (default: downloaded fresh, as the
        /// reference does; its tokenizer ships in the export).
        #[arg(long)]
        base_archive: Option<PathBuf>,
        /// Push the .cact to $NEEDLE_HF_REPO.
        #[arg(long)]
        upload: bool,
        /// Also download that platform's engine and header and place the
        /// archive beside them as needle3.cact (--out is then a directory).
        #[arg(long)]
        platform: Option<String>,
    },
    /// One agent turn (or several): the engine's JSON envelope per query.
    Complete {
        /// Weights: a .cact archive or a .safetensors checkpoint.
        #[arg(long, default_value = "models/needle3.cact")]
        model: PathBuf,
        /// Tools JSON (array of schemas).
        #[arg(long)]
        tools: PathBuf,
        /// System facts text.
        #[arg(long, default_value = "")]
        system: String,
        #[arg(long, default_value_t = 512)]
        max: usize,
        /// Queries; each continues the same conversation.
        queries: Vec<String>,
    },
    /// Run a bundled environment's acceptance suite on raw model output.
    Env {
        /// smart_home, media_player, productivity, wearable, kitchen_appliance, data_capture
        name: String,
        #[arg(long)]
        checkpoint: PathBuf,
        /// LoRA adapter to merge first.
        #[arg(long)]
        lora: Option<PathBuf>,
        #[arg(long, default_value_t = 160)]
        max_len: usize,
        #[arg(long)]
        quiet: bool,
    },
    /// Exact-call accuracy of a model on JSONL examples (greedy decode).
    Eval {
        data: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        lora: Option<PathBuf>,
        /// Score the first N examples.
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 96)]
        max_len: usize,
    },
    /// Measure prefill and decode speed.
    Bench {
        #[arg(long, default_value = "models/needle3.cact")]
        model: PathBuf,
        /// Tools JSON; the prompt is the rendered tool-call prompt.
        #[arg(long)]
        tools: Option<PathBuf>,
        #[arg(long, default_value = "dim the living room lights to 30")]
        query: String,
        /// Tokens to decode (EOS ignored).
        #[arg(long, default_value_t = 128)]
        decode: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        /// Time only the last N prompt tokens, prefilled as one chunk after
        /// the rest (a conversation turn).
        #[arg(long)]
        chunk: Option<usize>,
        /// Measure agent turns instead (static prefix cached, envelope speeds).
        #[arg(long)]
        agent: bool,
        #[arg(long, default_value = "")]
        system: String,
    },
    /// Write templated training data for a bundled environment (JSONL).
    Synth {
        #[arg(long, default_value = "smart_home")]
        env: String,
        #[arg(long, default_value_t = 1000)]
        num_samples: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run { checkpoint, query, tools, max_len, seed, temperature, stats } => {
            let loaded = loader::load(&checkpoint)?;
            let mut prompt = query.unwrap_or_else(|| "The most surprising thing about".into());
            if let Some(tools) = tools {
                let text = std::fs::read_to_string(&tools).with_context(|| format!("read {}", tools.display()))?;
                let parsed: serde_json::Value = serde_json::from_str(&text)?;
                prompt = build_prompt(&prompt, Some(&parsed))?;
            }
            println!("prompt: {}", py_repr(&prompt));
            let opts = GenOptions { max_new_tokens: max_len, temperature, seed };
            let mut out = std::io::stdout();
            let (_, st) = generate(&loaded.model, &loaded.tokenizer, &prompt, &opts, |piece| {
                let _ = out.write_all(piece.as_bytes());
                let _ = out.flush();
            })?;
            println!();
            if stats {
                eprint!("{}", needle_engine::prof::report());
                eprintln!(
                    "prefill {} tokens {:.1} tok/s, decode {} tokens {:.1} tok/s",
                    st.prompt_tokens,
                    st.prefill_tps(),
                    st.new_tokens,
                    st.decode_tps()
                );
            }
        }
        Command::Finetune {
            jsonl_path,
            checkpoint,
            epochs,
            batch_size,
            lr,
            lora_rank,
            lora_alpha,
            max_len,
            val_split,
            seed,
            checkpoint_dir,
            out,
            no_score,
            max_steps,
            generate,
            model,
            workers,
        } => {
            let jsonl_path =
                if generate > 0 { generate::augment_jsonl(&jsonl_path, generate, &model, 25, None, workers)? } else { jsonl_path };
            let checkpoint = match checkpoint {
                Some(c) => Some(c),
                None if std::path::Path::new(needle_train::finetune::DEFAULT_BASE).exists() => None,
                None => {
                    println!("  {:<9} {}  downloading from Hugging Face", "fetch", needle_train::finetune::DEFAULT_BASE);
                    Some(fetch::fetch_checkpoint("needle3.safetensors", std::path::Path::new(fetch::CHECKPOINT_PREFIX), 3)?)
                }
            };
            let args = needle_train::finetune::FinetuneArgs {
                jsonl_path,
                checkpoint,
                epochs,
                batch_size,
                lr,
                lora_rank,
                lora_alpha,
                max_len,
                val_split,
                seed,
                checkpoint_dir,
                out,
                score: !no_score,
                max_steps,
            };
            needle_train::finetune::finetune(&args, None, |m| println!("{m}"))?;
        }
        Command::Build { checkpoint, lora, out, layers, base_archive, upload, platform } => {
            let base_archive = match base_archive {
                Some(b) => b,
                None => fetch::fetch_weights(3, None, true)?,
            };
            let mut folder = None;
            let out = match &platform {
                Some(p) => {
                    let dir = std::path::absolute(out.clone().unwrap_or_else(|| PathBuf::from(p)))?;
                    for path in fetch::download_platform(p, dir.parent().unwrap_or(std::path::Path::new(".")), 3, Some(&dir))? {
                        println!("  {:<9} {}  {:.2} MB", "engine", path.display(), net::megabytes(&path));
                    }
                    folder = Some(dir.clone());
                    Some(dir.join(fetch::base_weights(3)?))
                }
                None => out,
            };
            let written = build(checkpoint, lora, out, layers, &base_archive)?;
            match &folder {
                Some(dir) => {
                    if let Some(runner) = ["needle", "needle.exe"].iter().map(|n| dir.join(n)).find(|p| p.exists()) {
                        println!(
                            "  {:<9} {} --model {} --tools tools.json --serve",
                            "next",
                            runner.display(),
                            written.file_name().unwrap_or_default().to_string_lossy()
                        );
                    }
                }
                None => println!("  {:<9} needle.Needle(weights={}, tools=[...])", "next", py_repr(&written.display().to_string())),
            }
            if upload {
                let repo = std::env::var("NEEDLE_HF_REPO").map_err(|_| anyhow::anyhow!("set NEEDLE_HF_REPO=<you>/<model> to upload"))?;
                let name = written.file_name().unwrap_or_default().to_string_lossy().into_owned();
                net::hub_upload(&repo, &written, &name)?;
                println!("  {:<9} {name}  {repo}", "uploaded");
            }
        }
        Command::GenerateData { tools, augment, num_samples, batch_size, workers, model, output } => {
            generate::generate_main(tools.as_deref(), augment.as_deref(), num_samples, batch_size, workers, &model, output.as_deref())?;
        }
        Command::Download { spec, out, depth, generation } => download(&spec, &out, depth, generation)?,
        Command::Fetch { out, platform_tag, generation } => {
            let version = fetch::engine_version(generation)?;
            let dest = match out {
                Some(o) => o,
                None => fetch::cache_dir(generation)?,
            };
            let path = fetch::fetch_library(version, &dest, platform_tag.as_deref(), generation)?;
            println!("  {:<9} {}", "engine", path.display());
            println!(
                "  {:<9} copy to ~/.cache/cactus-needle/v{generation}/{version}/ on the device, or point NEEDLE{generation}_LIB_PATH at the file",
                "deploy"
            );
            println!("  {:<9} or build this repo's drop-in: cargo build --release -p needle-ffi (target/release/libneedle3.*)", "rust");
        }
        Command::Platform { verb } => {
            let verb = match verb {
                PlatformVerb::Finetune { train, validation, test, max_depth, suffix, out, depth, no_wait } => {
                    platform::Verb::Finetune { train, validation, test, max_depth, suffix, out, depth, no_wait }
                }
                PlatformVerb::Generate { tools, examples, description, message, suffix, out, no_wait } => {
                    platform::Verb::Generate { tools, examples, description, message, suffix, out, no_wait }
                }
                PlatformVerb::Jobs { job_id, wait, out, depth } => platform::Verb::Jobs { job_id, wait, out, depth },
                PlatformVerb::Models { model_id } => platform::Verb::Models { model_id },
                PlatformVerb::Files => platform::Verb::Files,
                PlatformVerb::Billing => platform::Verb::Billing,
            };
            platform::main(verb)?;
        }
        Command::Playground { weights, port, host } => {
            let downloads = std::env::temp_dir().join(format!("needle-playground-{}", std::process::id()));
            playground::playground(weights, &host, port, downloads)?;
        }
        Command::Serve { model, tools, system, port, host, max } => {
            let tools: Vec<serde_json::Value> = match tools {
                Some(t) => serde_json::from_str(&std::fs::read_to_string(t)?)?,
                None => vec![],
            };
            let system = match system {
                Some(p) => std::fs::read_to_string(p)?,
                None => String::new(),
            };
            playground::serve(&model, tools, system.trim(), &host, port, max)?;
        }
        Command::Complete { model, tools, system, max, queries } => {
            let loaded = loader::load(&model)?;
            let tools: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(&tools)?)?;
            let mut agent = needle_engine::agent::Agent::new(loaded.model.into(), loaded.tokenizer.into(), tools, &system)?;
            for q in queries {
                println!("{}", needle_engine::agent::envelope_json(&agent.complete(&q, max)?));
            }
        }
        Command::Env { name, checkpoint, lora, max_len, quiet } => {
            let env = needle_core::synth::environment(&name).with_context(|| format!("unknown environment {name}"))?;
            let loaded = load_with_lora(&checkpoint, lora.as_deref())?;
            let t0 = std::time::Instant::now();
            let res = needle_engine::harness::run_suite(&loaded.model, &loaded.tokenizer, &env, max_len)?;
            if !quiet {
                for f in &res.failures {
                    println!("{f}");
                }
            }
            println!(
                "{}/{} passed, {} critical failures ({:.1}s)",
                res.passed,
                res.total,
                res.critical_failures,
                t0.elapsed().as_secs_f64()
            );
            if !res.ok() {
                std::process::exit(1);
            }
        }
        Command::Eval { data, checkpoint, lora, limit, max_len } => {
            let loaded = load_with_lora(&checkpoint, lora.as_deref())?;
            let (examples, _) = needle_core::render::read_examples(&data)?;
            let n = limit.unwrap_or(examples.len()).min(examples.len());
            let refs: Vec<_> = examples[..n].iter().collect();
            let t0 = std::time::Instant::now();
            let c = needle_engine::harness::exact_calls(&loaded.model, &loaded.tokenizer, &refs, max_len)?;
            println!("{c}/{n} exact ({:.1}%)  {:.1}s", 100.0 * c as f64 / n.max(1) as f64, t0.elapsed().as_secs_f64());
        }
        Command::Bench { model, tools, query, decode, repeats, chunk, agent, system } => {
            let loaded = loader::load(&model)?;
            if agent {
                let tools: Vec<serde_json::Value> =
                    serde_json::from_str(&std::fs::read_to_string(tools.context("--agent needs --tools")?)?)?;
                let mut a = needle_engine::agent::Agent::new(loaded.model.into(), loaded.tokenizer.into(), tools, &system)?;
                let (mut bp, mut bd) = (0f64, 0f64);
                let t0 = std::time::Instant::now();
                for i in 0..repeats {
                    if i == 1 {
                        needle_engine::prof::reset();
                    }
                    a.reset();
                    let env = a.complete(&query, 512)?;
                    bp = bp.max(env["prefill_tps"].as_f64().unwrap_or(0.0));
                    bd = bd.max(env["decode_tps"].as_f64().unwrap_or(0.0));
                }
                eprint!("{}", needle_engine::prof::report());
                println!("mean turn {:.1} ms", t0.elapsed().as_secs_f64() * 1e3 / repeats as f64);
                println!("agent turn: prefill {bp:.1} tok/s   decode {bd:.1} tok/s");
                return Ok(());
            }
            let prompt = match tools {
                Some(t) => build_prompt(&query, Some(&serde_json::from_str(&std::fs::read_to_string(t)?)?))?,
                None => query,
            };
            let mut ids = vec![needle_core::tokenizer::BOS_ID];
            ids.extend(loaded.tokenizer.encode(&prompt));
            let m = &loaded.model;
            let (mut best_p, mut best_d) = (0f64, 0f64);
            for _ in 0..repeats {
                needle_engine::prof::reset();
                let mut s = m.session();
                let split = chunk.map_or(0, |c| ids.len().saturating_sub(c));
                m.forward(&mut s, &ids[..split], needle_engine::Outputs::None);
                needle_engine::prof::reset();
                let t0 = std::time::Instant::now();
                let mut logits = m.forward(&mut s, &ids[split..], needle_engine::Outputs::LastLogits).data;
                let p = (ids.len() - split) as f64 / t0.elapsed().as_secs_f64();
                if decode == 0 {
                    best_p = best_p.max(p);
                    continue;
                }
                needle_engine::prof::reset();
                let t1 = std::time::Instant::now();
                for _ in 0..decode {
                    let next = needle_engine::generate::argmax(&logits) as u32;
                    logits = m.forward(&mut s, &[next], needle_engine::Outputs::LastLogits).data;
                }
                let d = decode as f64 / t1.elapsed().as_secs_f64();
                best_p = best_p.max(p);
                best_d = best_d.max(d);
            }
            eprint!("{}", needle_engine::prof::report());
            println!("prefill {} tokens {:.1} tok/s   decode {} tokens {:.1} tok/s", ids.len(), best_p, decode, best_d);
        }
        Command::Synth { env, num_samples, seed, output } => {
            let e = needle_core::synth::environment(&env).with_context(|| format!("unknown environment {env}"))?;
            anyhow::ensure!(env == "smart_home", "templated data exists for smart_home only");
            let exclude: Vec<String> =
                e["test_cases"].as_array().into_iter().flatten().filter_map(|c| c["query"].as_str().map(str::to_string)).collect();
            let rows = needle_core::synth::smart_home(&e, num_samples, seed, &exclude);
            let mut f = std::io::BufWriter::new(std::fs::File::create(&output)?);
            for r in &rows {
                writeln!(f, "{}", serde_json::Value::Object(r.clone()))?;
            }
            println!("  {:<9} {} examples  {}", "wrote", rows.len(), output.display());
        }
    }
    Ok(())
}

/// `needle build`: merge, slice the rung, export CQ-W4 with the base
/// archive's tokenizer (or copy the base archive when nothing changes).
fn build(
    checkpoint: Option<PathBuf>,
    lora: Option<PathBuf>,
    out: Option<PathBuf>,
    layers: Option<usize>,
    base_archive: &std::path::Path,
) -> Result<PathBuf> {
    use needle_core::cact;
    use needle_core::checkpoint::{load_checkpoint, merge_lora, read_adapter};
    use needle_core::config::{effective_kv_window, ladder_config, ladder_slice};
    let base_layers = cact::read_layers(base_archive).with_context(|| format!("read the base archive {}", base_archive.display()))?;
    println!("  {:<9} {}  {:.2} MB", "base", base_archive.display(), std::fs::metadata(base_archive)?.len() as f64 / 1e6);
    if lora.is_none() && layers.is_none_or(|l| l == base_layers) {
        let out = out.context("pass --out <archive.cact>")?;
        std::fs::copy(base_archive, &out)?;
        println!("  {:<9} {}  {:.2} MB  the published base archive", "wrote", out.display(), std::fs::metadata(&out)?.len() as f64 / 1e6);
        return Ok(out);
    }
    let adapter = lora.as_deref().map(read_adapter).transpose()?;
    let checkpoint = checkpoint
        .or_else(|| adapter.as_ref().and_then(|a| a.base.clone()).map(PathBuf::from).filter(|p| p.exists()))
        .unwrap_or_else(|| PathBuf::from(needle_train::finetune::DEFAULT_BASE));
    let checkpoint = if checkpoint.exists() {
        checkpoint
    } else {
        println!("  {:<9} {}  downloading from Hugging Face", "fetch", checkpoint.display());
        fetch::fetch_checkpoint(&checkpoint.display().to_string(), checkpoint.parent().unwrap_or(std::path::Path::new(".")), 3)?
    };
    let (mut params, mut config, _) = load_checkpoint(&checkpoint)?;
    if let (Some(a), Some(path)) = (&adapter, &lora) {
        merge_lora(&mut params, a)?;
        println!("  {:<9} {} weight groups  {}", "merged", a.lora.len(), path.display());
        if params.keys().any(|k| k.starts_with("confidence_head/")) {
            params.retain(|k, _| !k.starts_with("confidence_head/"));
            println!("  {:<9} the confidence head; it is not trained locally, so confidence reports None", "dropped");
        }
    }
    config.ladder_depths.clear();
    config.ladder_sample = false;
    config.ladder_widths.clear();
    if let Some(l) = layers.filter(|&l| l != config.num_layers) {
        anyhow::ensure!((2..config.num_layers).contains(&l), "--layers must be in [2, {}], got {l}", config.num_layers);
        params = ladder_slice(&params, &config, l)?;
        config = ladder_config(&config, l)?;
    }
    println!("  {:<9} {} layers", "depth", config.num_layers);
    let out = out.unwrap_or_else(|| PathBuf::from(format!("{}.cact", checkpoint.file_stem().unwrap_or_default().to_string_lossy())));
    let archive = cact::Archive::open(base_archive)?;
    let info = cact::write_export(&out, &params, &config, 4, 128, Some(archive.tokenizer_blob()?), effective_kv_window(&config))?;
    println!("  {:<9} {}  {:.2} MB  {} tensors  W4A8", "wrote", out.display(), info.bytes as f64 / 1e6, info.tensors);
    Ok(out)
}

/// `needle download`: classify the spec the way the reference does.
fn download(spec: &str, out: &std::path::Path, depth: Option<u64>, generation: u32) -> Result<()> {
    let line = |label: &str, path: &std::path::Path| println!("  {label:<9} {}  {:.2} MB", path.display(), net::megabytes(path));
    if spec.contains('/') {
        fetch::register_download(generation);
        let parts: Vec<&str> = spec.split('/').filter(|p| !p.is_empty()).collect();
        anyhow::ensure!(parts.len() >= 2, "pass <org>/<repo>/<file>.cact or <org>/<repo>");
        let repo = parts[..2].join("/");
        let file = if parts.len() > 2 {
            parts[2..].join("/")
        } else {
            let cacts: Vec<String> = net::hub_list(&repo)?.into_iter().filter(|f| f.ends_with(".cact")).collect();
            anyhow::ensure!(
                cacts.len() == 1,
                "{repo} holds {} .cact files, name one: {}",
                cacts.len(),
                cacts.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
            );
            cacts[0].clone()
        };
        let cached = net::hub_download(&repo, &file, false)?;
        let dest = net::copy_to(&cached, &out.join(std::path::Path::new(&file).file_name().unwrap()))?;
        line("weights", &dest);
        println!("  {:<9} needle.Needle(weights={}, tools=[...])", "next", py_repr(&dest.display().to_string()));
    } else if spec.starts_with("model-") {
        for p in platform::Platform::new(None).download(spec, out, depth)? {
            line("weights", &p);
        }
        println!("  {:<9} needle.Needle(weights=<path>, tools=[...], auto_date=False)", "next");
    } else if fetch::PLATFORMS.contains(&spec) {
        let mut paths = fetch::download_platform(spec, out, generation, None)?;
        if generation >= 3 {
            paths.push(fetch::fetch_weights(generation, Some(&out.join(spec)), false)?);
        }
        for p in &paths {
            line("file", p);
        }
        if let Some(runner) = paths.iter().find(|p| matches!(p.file_name().and_then(|n| n.to_str()), Some("needle" | "needle.exe"))) {
            let weights = if generation >= 3 { format!(" --model {}", fetch::base_weights(generation)?) } else { String::new() };
            println!("  {:<9} {}{weights} --tools tools.json --serve", "next", runner.display());
        }
    } else if matches!(spec.trim_end_matches(".cact"), "needle2" | "needle3") {
        let generation = if spec.starts_with("needle2") { 2 } else { 3 };
        std::fs::create_dir_all(out)?;
        let p = fetch::fetch_weights(generation, Some(out), false)?;
        line("weights", &p);
        println!("  {:<9} needle.Needle(weights={}, tools=[...])", "next", py_repr(&p.display().to_string()));
    } else if spec.ends_with(".safetensors") {
        let generation = if spec.starts_with("needle2") { 2 } else { 3 };
        let p = fetch::fetch_checkpoint(spec, &out.join(fetch::CHECKPOINT_PREFIX), generation)?;
        line("file", &p);
        println!("  {:<9} needle finetune data.jsonl --checkpoint {} [--layers N]", "next", p.display());
    } else {
        anyhow::bail!(
            "unknown download {spec:?}: pass needle3 (base weights), needle3.safetensors (the checkpoint to fine-tune), a platform ({}), model-<id> from cactuscompute.com, or <org>/<repo>[/<file>.cact]",
            fetch::PLATFORMS.join(", ")
        );
    }
    Ok(())
}

fn load_with_lora(path: &std::path::Path, lora: Option<&std::path::Path>) -> Result<loader::Loaded> {
    match lora {
        None => loader::load(path),
        Some(adapter) => {
            let (mut params, config, _) = needle_core::checkpoint::load_checkpoint(path)?;
            let adapter = needle_core::checkpoint::read_adapter(adapter)?;
            needle_core::checkpoint::merge_lora(&mut params, &adapter)?;
            let tokenizer = loader::find_tokenizer(path)?;
            let model = needle_engine::Model::new(needle_engine::Weights::from_checkpoint(&params, &config)?);
            Ok(loader::Loaded { model, tokenizer })
        }
    }
}

/// Python `repr()` of a str, which the reference prints as `prompt: '...'`.
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}
