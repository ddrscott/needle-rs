//! Autoregressive decoding (`run.generate` of the reference, with a KV cache).

use anyhow::{Result, bail};
use needle_core::Tokenizer;
use needle_core::tokenizer::{BOS_ID, EOS_ID};

use crate::model::{Model, Outputs};

#[derive(Clone, Debug)]
pub struct GenOptions {
    pub max_new_tokens: usize,
    /// 0 is greedy.
    pub temperature: f32,
    pub seed: u64,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self { max_new_tokens: 256, temperature: 0.0, seed: 0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

impl GenStats {
    pub fn prefill_tps(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_secs.max(1e-9)
    }

    pub fn decode_tps(&self) -> f64 {
        self.new_tokens as f64 / self.decode_secs.max(1e-9)
    }
}

/// First index of the maximum (`jnp.argmax`).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// SplitMix64: a small deterministic sampler RNG.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sample from `softmax(logits / temperature)`.
pub fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = logits.iter().map(|&l| (((l - m) / temperature) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut r = rng.next_f64() * total;
    for (i, &p) in w.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    w.len() - 1
}

/// Decode a completion for `prompt`, calling `on_text` with each new piece
/// of text as it becomes printable.
pub fn generate(
    model: &Model,
    tok: &Tokenizer,
    prompt: &str,
    opts: &GenOptions,
    mut on_text: impl FnMut(&str),
) -> Result<(String, GenStats)> {
    let mut ids = vec![BOS_ID];
    ids.extend(tok.encode(prompt));
    let max_len = model.config().max_seq_len;
    let buf_len = max_len.min(ids.len() + opts.max_new_tokens);
    if ids.len() >= buf_len {
        bail!("Prompt ({} tokens) does not fit in max_seq_len={max_len}", ids.len());
    }
    let mut stats = GenStats { prompt_tokens: ids.len(), ..Default::default() };
    let mut rng = Rng::new(opts.seed);
    let mut s = model.session();
    let t0 = web_time::Instant::now();
    let mut logits = model.forward(&mut s, &ids, Outputs::LastLogits).data;
    stats.prefill_secs = t0.elapsed().as_secs_f64();
    let t1 = web_time::Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = String::new();
    for _ in ids.len() - 1..buf_len - 1 {
        let next = if opts.temperature <= 0.0 { argmax(&logits) } else { sample(&logits, opts.temperature, &mut rng) } as u32;
        if next == EOS_ID {
            break;
        }
        generated.push(next);
        let text = tok.decode(&generated);
        if text.len() > printed.len() && text.starts_with(printed.as_str()) {
            on_text(&text[printed.len()..]);
            printed = text;
        }
        if s.len() + 1 >= buf_len {
            break;
        }
        logits = model.forward(&mut s, &[next], Outputs::LastLogits).data;
    }
    stats.new_tokens = generated.len();
    stats.decode_secs = t1.elapsed().as_secs_f64();
    let text = tok.decode(&generated);
    if text.len() > printed.len() && text.starts_with(printed.as_str()) {
        on_text(&text[printed.len()..]);
    }
    Ok((text, stats))
}
