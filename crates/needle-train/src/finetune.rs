//! `needle finetune`: the reference's `finetune_local`, step for step.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, bail};
use needle_core::checkpoint::{Adapter, LoraPair, load_checkpoint, write_adapter};
use needle_core::quant::WEIGHT_BITS;
use needle_core::render::{encode_example, fit_max_len, read_examples};
use needle_core::{Tensor, Tokenizer};
use needle_engine::{Model, Weights};
use serde_json::Value;

use crate::graph::{Batch, Graph, LoraParams, TARGETS, target_dims};
use crate::optim::{AdamW, Schedule};
use crate::rand::{NpRng, normal, prng_key, split};

pub const DEFAULT_BASE: &str = "checkpoints/needle3.safetensors";

#[derive(Clone, Debug)]
pub struct FinetuneArgs {
    pub jsonl_path: PathBuf,
    pub checkpoint: Option<PathBuf>,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub max_len: usize,
    pub val_split: f64,
    pub seed: u64,
    pub checkpoint_dir: PathBuf,
    pub out: Option<PathBuf>,
    pub score: bool,
    /// Stop after this many optimizer steps (benchmarking); `None` runs all.
    pub max_steps: Option<usize>,
}

/// `init_lora`: `A ~ N(0, 1) / rank` per target from split keys, `B = 0`.
pub fn init_lora(graph: &Graph, rank: usize, seed: u64) -> LoraParams {
    let c = graph.config();
    let l = c.num_layers;
    let dims = target_dims(c);
    let mut key = prng_key(seed);
    let a = std::array::from_fn(|t| {
        let (k, sub) = split(key);
        key = k;
        normal(sub, l * dims[t].0 * rank).into_iter().map(|v| v / rank as f32).collect()
    });
    let b = std::array::from_fn(|t| vec![0.0; l * rank * dims[t].1]);
    LoraParams { rank, a, b }
}

pub fn target_path(t: usize) -> String {
    format!("stack/layers/block/self_attn/{}/kernel", TARGETS[t])
}

fn to_adapter(lora: &LoraParams, graph: &Graph, scale: f32, base: &Path, seed: u64) -> Adapter {
    let c = graph.config();
    let l = c.num_layers;
    let r = lora.rank;
    let mut map = BTreeMap::new();
    for (t, &(din, dout)) in target_dims(c).iter().enumerate() {
        map.insert(
            target_path(t),
            LoraPair { a: Tensor::new(vec![l, din, r], lora.a[t].clone()), b: Tensor::new(vec![l, r, dout], lora.b[t].clone()) },
        );
    }
    Adapter { lora: map, scale: scale as f64, base: Some(base.display().to_string()), rank: Some(r), seed: Some(seed as i64) }
}

struct Encoded {
    ids: Vec<Vec<u32>>,
    masks: Vec<Vec<f32>>,
}

fn batch_of(enc: &Encoded, idx: &[usize]) -> Batch {
    let rows: Vec<(&[u32], &[f32])> = idx.iter().map(|&i| (enc.ids[i].as_slice(), enc.masks[i].as_slice())).collect();
    Batch::from_padded(&rows)
}

/// `_score_quantised`: exact-call accuracy of the held-out split, decoded
/// greedily under the CQ-W4 weights the archive ships.
fn score(graph: &Graph, tok: &Tokenizer, examples: &[serde_json::Map<String, Value>], n_val: usize, seed: u64) -> Result<(usize, usize)> {
    if examples.is_empty() {
        return Ok((0, 0));
    }
    let order = NpRng::new(seed).permutation(examples.len());
    let model = Model::new(graph.w.clone());
    let held: Vec<_> = order[..n_val].iter().map(|&i| &examples[i]).collect();
    let correct = needle_engine::harness::exact_calls(&model, tok, &held, 96)?;
    Ok((correct, held.len()))
}

pub struct FinetuneReport {
    pub adapter: PathBuf,
    pub step_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub accuracy: Option<(usize, usize)>,
    pub secs_per_step: f64,
}

pub fn finetune(args: &FinetuneArgs, tok_hint: Option<Tokenizer>, mut emit: impl FnMut(&str)) -> Result<FinetuneReport> {
    let base_path = args.checkpoint.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_BASE));
    let (params, mut config, _) = load_checkpoint(&base_path)?;
    config.dtype = "float32".into();
    emit(&format!("  {:<9} {}  float32", "backend", backend()));
    emit(&format!("  {:<9} {} layers", "depth", config.num_layers));
    let tok = match tok_hint {
        Some(t) => t,
        None => needle_engine::loader::find_tokenizer(&base_path)?,
    };
    let (examples, skipped) = read_examples(&args.jsonl_path)?;
    if skipped > 0 {
        emit(&format!("  skipped {skipped} line(s) that are neither query/answers nor single-turn chat format"));
    }
    let max_len = fit_max_len(&examples, &tok, args.max_len)?;
    let mut enc = Encoded { ids: vec![], masks: vec![] };
    for e in &examples {
        let (ids, mask) = encode_example(&tok, e, max_len)?;
        enc.ids.push(ids);
        enc.masks.push(mask);
    }
    if enc.ids.is_empty() {
        bail!("no usable examples in {}", args.jsonl_path.display());
    }
    emit(&format!("  {:<9} {} examples  seq_len {}  cap {}", "data", enc.ids.len(), max_len, args.max_len));

    let scale = args.lora_alpha / args.lora_rank as f32;
    let t_load = Instant::now();
    let mut graph = Graph::new(Weights::from_checkpoint(&params, &config)?, scale);
    drop(params);
    emit(&format!("  {:<9} CQ W{WEIGHT_BITS} STE + A8 (matches export)  ({:.1}s to quantize)", "numerics", t_load.elapsed().as_secs_f64()));
    let mut rng = NpRng::new(args.seed);
    let mut lora = init_lora(&graph, args.lora_rank, args.seed);
    emit(&format!(
        "  {:<9} rank {}  alpha {}  {} weight groups",
        "lora",
        args.lora_rank,
        needle_core::pyjson::float_repr(args.lora_alpha as f64).trim_end_matches(".0"),
        TARGETS.len()
    ));

    let n = enc.ids.len();
    let n_val = ((n as f64 * args.val_split) as usize).min(n - 1);
    let (mut train, mut val) = (Encoded { ids: vec![], masks: vec![] }, Encoded { ids: vec![], masks: vec![] });
    if n_val > 0 {
        let order = rng.permutation(n);
        for (j, &i) in order.iter().enumerate() {
            let dst = if j < n_val { &mut val } else { &mut train };
            dst.ids.push(enc.ids[i].clone());
            dst.masks.push(enc.masks[i].clone());
        }
        emit(&format!("  {:<9} {} examples for validation", "holdout", n_val));
    } else {
        train = enc;
    }

    let (batch, count) = (args.batch_size, train.ids.len());
    let steps_per_epoch = count.div_ceil(batch);
    let total_steps = args.epochs * steps_per_epoch;
    let warmup = (total_steps / 20).max(1).min(total_steps.saturating_sub(1));
    let mut opt = AdamW::new(Schedule { peak: args.lr, warmup, decay_steps: total_steps }, &lora);
    emit(&format!("  {:<9} {} steps  warmup {}  cosine decay  clip 1.0", "schedule", total_steps, warmup));

    let every = (total_steps / 50).max(1);
    let mut step_i = 0;
    let mut step_losses = vec![];
    let mut val_losses = vec![];
    let t_train = Instant::now();
    'epochs: for epoch in 0..args.epochs {
        let order = rng.permutation(count);
        let mut last = 0f32;
        for start in (0..count).step_by(batch) {
            let idx = &order[start..(start + batch).min(count)];
            let b = batch_of(&train, idx);
            graph.merge(&lora);
            let out = graph.step(&b, &lora);
            opt.update(&mut lora, &out.grads);
            last = out.loss;
            step_losses.push(last);
            step_i += 1;
            if step_i % every == 0 {
                emit(&format!("  {:<9} {}/{}  loss {:.4}", "step", step_i, total_steps, last));
            }
            if args.max_steps.is_some_and(|m| step_i >= m) {
                break 'epochs;
            }
        }
        graph.merge(&lora);
        if n_val > 0 {
            let losses: Vec<f32> =
                (0..n_val).step_by(batch).map(|i| graph.loss(&batch_of(&val, &(i..(i + batch).min(n_val)).collect::<Vec<_>>()))).collect();
            let v = losses.iter().sum::<f32>() / losses.len() as f32;
            val_losses.push(v);
            emit(&format!("  {:<9} {}/{}  loss {:.4}  val {:.4}", "epoch", epoch + 1, args.epochs, last, v));
        } else {
            emit(&format!("  {:<9} {}/{}  loss {:.4}", "epoch", epoch + 1, args.epochs, last));
        }
    }
    let secs_per_step = t_train.elapsed().as_secs_f64() / step_i.max(1) as f64;
    emit(&format!("  {:<9} {:.2}s/step  {:.1}s total", "time", secs_per_step, t_train.elapsed().as_secs_f64()));
    graph.merge(&lora);

    let mut accuracy = None;
    if n_val > 0 && args.score {
        let (c, t) = score(&graph, &tok, &examples, n_val, args.seed)?;
        if t > 0 {
            emit(&format!("  {:<9} {c}/{t} held-out calls exact, scored on the W{WEIGHT_BITS} weights the archive ships", "accuracy"));
            accuracy = Some((c, t));
        }
    }

    std::fs::create_dir_all(&args.checkpoint_dir)?;
    let out = args.out.clone().unwrap_or_else(|| args.checkpoint_dir.join("needle_lora.safetensors"));
    write_adapter(&out, &to_adapter(&lora, &graph, scale, &base_path, args.seed))?;
    emit(&format!("  {:<9} {}", "adapter", out.display()));
    emit(&format!("  {:<9} needle build {} --lora {}", "next", base_path.display(), out.display()));
    emit(&format!("  {:<9} local tuning leaves the confidence head untrained; needle build drops it and confidence reports None", "note"));
    Ok(FinetuneReport { adapter: out, step_losses, val_losses, accuracy, secs_per_step })
}

fn backend() -> &'static str {
    if cfg!(target_os = "macos") { "cpu (accelerate)" } else { "cpu" }
}
