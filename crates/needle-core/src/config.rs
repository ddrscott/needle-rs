//! `TransformerConfig` and the geometry rules derived from it (head dims,
//! engram layout, Hadamard blocks, the depth ladder, the KV budget).

use anyhow::{Result, bail};
use serde_json::{Map, Value, json};

use crate::tensor::Params;

pub const ENGRAM_SUB_DIM: usize = 128;
pub const ENGRAM_CONV_TAPS: usize = 4;
pub const ENGRAM_SEED: u32 = 0x9E37_79B9;
pub const ENGRAM_PRIME: u32 = 0x0100_0193;
pub const HADA_COND_RANK: usize = 8;
pub const HADA_PERM_SEEDS: [u32; 2] = [11, 13];

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub pad_token_id: u32,
    pub embedding_dim: usize,
    pub embedding_probes: usize,
    pub embedding_queries: usize,
    pub confidence_probes: usize,
    pub confidence_queries: usize,
    pub router_probes: usize,
    pub router_queries: usize,
    pub rope_theta: f64,
    pub dtype: String,
    pub flash: bool,
    pub engram_orders: Vec<usize>,
    pub engram_heads: usize,
    pub engram_slots: usize,
    pub engram_seed_heads: usize,
    pub engram_layers: Vec<usize>,
    pub global_layers: Vec<usize>,
    pub sliding_window: usize,
    pub ladder_depths: Vec<usize>,
    pub ladder_sample: bool,
    pub ladder_widths: Vec<usize>,
    pub ladder_order: Vec<usize>,
    pub mhc_lanes: usize,
    pub qkv_conv_taps: usize,
    pub out_vocab: usize,
    pub kv_window: usize,
    pub kv_bits: usize,
    pub act_bits: usize,
    pub weight_bits: String,
    /// Keys the checkpoint carried that this port does not model; kept so a
    /// checkpoint written back out carries them unchanged.
    pub extra: Map<String, Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 16384,
            d_model: 768,
            num_heads: 12,
            num_kv_heads: 2,
            num_layers: 20,
            qk_head_dim: 48,
            v_head_dim: 64,
            max_seq_len: 4096,
            pad_token_id: 0,
            embedding_dim: 128,
            embedding_probes: 4,
            embedding_queries: 4,
            confidence_probes: 4,
            confidence_queries: 4,
            router_probes: 4,
            router_queries: 4,
            rope_theta: 100000.0,
            dtype: "bfloat16".into(),
            flash: true,
            engram_orders: vec![2, 3],
            engram_heads: 0,
            engram_slots: 18432,
            engram_seed_heads: 0,
            engram_layers: vec![3, 7, 11, 15, 19],
            global_layers: vec![4, 9, 14, 19],
            sliding_window: 1024,
            ladder_depths: vec![],
            ladder_sample: false,
            ladder_widths: vec![],
            ladder_order: vec![],
            mhc_lanes: 4,
            qkv_conv_taps: 3,
            out_vocab: 0,
            kv_window: 0,
            kv_bits: 8,
            act_bits: 8,
            weight_bits: String::new(),
            extra: Map::new(),
        }
    }
}

fn as_usize(v: &Value, key: &str) -> Result<usize> {
    match v {
        Value::Number(n) => n
            .as_u64()
            .map(|x| x as usize)
            .or_else(|| n.as_f64().map(|f| f as usize))
            .ok_or_else(|| anyhow::anyhow!("config {key} is not a count")),
        Value::Bool(b) => Ok(*b as usize),
        _ => bail!("config {key} is not a number: {v}"),
    }
}

fn as_list(v: &Value, key: &str) -> Result<Vec<usize>> {
    match v {
        Value::Array(items) => items.iter().map(|i| as_usize(i, key)).collect(),
        Value::Null => Ok(vec![]),
        _ => bail!("config {key} is not a list: {v}"),
    }
}

impl Config {
    /// `TransformerConfig.from_saved`: the keys a checkpoint omits fall back to
    /// the value that disables the feature, not the dataclass default, so an
    /// old checkpoint loads with the architecture it was trained with.
    pub fn from_saved(saved: &Map<String, Value>) -> Result<Self> {
        let mut map = saved.clone();
        let off: [(&str, Value); 11] = [
            ("qk_head_dim", json!(0)),
            ("v_head_dim", json!(0)),
            ("sliding_window", json!(0)),
            ("global_layers", json!([])),
            ("ladder_depths", json!([])),
            ("ladder_sample", json!(false)),
            ("ladder_widths", json!([])),
            ("ladder_order", json!([])),
            ("engram_seed_heads", json!(0)),
            ("qkv_conv_taps", json!(0)),
            ("out_vocab", json!(0)),
        ];
        for (k, v) in off {
            map.entry(k.to_string()).or_insert(v);
        }
        Self::from_kwargs(&map)
    }

    /// `TransformerConfig(**kwargs)`, including the legacy `attn_dim` rule.
    pub fn from_kwargs(kw: &Map<String, Value>) -> Result<Self> {
        let mut c = Config::default();
        for (k, v) in kw {
            match k.as_str() {
                "vocab_size" => c.vocab_size = as_usize(v, k)?,
                "d_model" => c.d_model = as_usize(v, k)?,
                "num_heads" => c.num_heads = as_usize(v, k)?,
                "num_kv_heads" => c.num_kv_heads = as_usize(v, k)?,
                "num_layers" => c.num_layers = as_usize(v, k)?,
                "qk_head_dim" => c.qk_head_dim = as_usize(v, k)?,
                "v_head_dim" => c.v_head_dim = as_usize(v, k)?,
                "max_seq_len" => c.max_seq_len = as_usize(v, k)?,
                "pad_token_id" => c.pad_token_id = as_usize(v, k)? as u32,
                "embedding_dim" => c.embedding_dim = as_usize(v, k)?,
                "embedding_probes" => c.embedding_probes = as_usize(v, k)?,
                "embedding_queries" => c.embedding_queries = as_usize(v, k)?,
                "confidence_probes" => c.confidence_probes = as_usize(v, k)?,
                "confidence_queries" => c.confidence_queries = as_usize(v, k)?,
                "router_probes" => c.router_probes = as_usize(v, k)?,
                "router_queries" => c.router_queries = as_usize(v, k)?,
                "rope_theta" => c.rope_theta = v.as_f64().ok_or_else(|| anyhow::anyhow!("rope_theta"))?,
                "dtype" => c.dtype = v.as_str().unwrap_or("float32").to_string(),
                "flash" => c.flash = v.as_bool().unwrap_or(true),
                "engram_orders" => c.engram_orders = as_list(v, k)?,
                "engram_heads" => c.engram_heads = as_usize(v, k)?,
                "engram_slots" => c.engram_slots = as_usize(v, k)?,
                "engram_seed_heads" => c.engram_seed_heads = as_usize(v, k)?,
                "engram_layers" => c.engram_layers = as_list(v, k)?,
                "global_layers" => c.global_layers = as_list(v, k)?,
                "sliding_window" => c.sliding_window = as_usize(v, k)?,
                "ladder_depths" => c.ladder_depths = as_list(v, k)?,
                "ladder_sample" => c.ladder_sample = v.as_bool().unwrap_or(false),
                "ladder_widths" => c.ladder_widths = as_list(v, k)?,
                "ladder_order" => c.ladder_order = as_list(v, k)?,
                "mhc_lanes" => c.mhc_lanes = as_usize(v, k)?,
                "qkv_conv_taps" => c.qkv_conv_taps = as_usize(v, k)?,
                "out_vocab" => c.out_vocab = as_usize(v, k)?,
                "kv_window" => c.kv_window = as_usize(v, k)?,
                "kv_bits" => c.kv_bits = as_usize(v, k)?,
                "act_bits" => c.act_bits = as_usize(v, k)?,
                "weight_bits" => c.weight_bits = v.as_str().unwrap_or("").to_string(),
                "remat" | "scan_unroll" | "attn_dim" => {}
                _ => {
                    c.extra.insert(k.clone(), v.clone());
                }
            }
        }
        if let Some(attn) = kw.get("attn_dim").and_then(Value::as_u64).filter(|&a| a > 0) {
            let head = attn as usize / c.num_heads;
            if !kw.contains_key("qk_head_dim") {
                c.qk_head_dim = head;
            }
            if !kw.contains_key("v_head_dim") {
                c.v_head_dim = head;
            }
            if !kw.contains_key("sliding_window") {
                c.sliding_window = 0;
            }
            if !kw.contains_key("global_layers") {
                c.global_layers.clear();
            }
        }
        Ok(c)
    }

    /// The dict a checkpoint stores as its `config` metadata.
    pub fn to_json(&self) -> Map<String, Value> {
        let mut m = Map::new();
        let mut put = |k: &str, v: Value| {
            m.insert(k.to_string(), v);
        };
        put("vocab_size", json!(self.vocab_size));
        put("d_model", json!(self.d_model));
        put("num_heads", json!(self.num_heads));
        put("num_kv_heads", json!(self.num_kv_heads));
        put("num_layers", json!(self.num_layers));
        put("qk_head_dim", json!(self.qk_head_dim));
        put("v_head_dim", json!(self.v_head_dim));
        put("max_seq_len", json!(self.max_seq_len));
        put("pad_token_id", json!(self.pad_token_id));
        put("embedding_dim", json!(self.embedding_dim));
        put("embedding_probes", json!(self.embedding_probes));
        put("embedding_queries", json!(self.embedding_queries));
        put("confidence_probes", json!(self.confidence_probes));
        put("confidence_queries", json!(self.confidence_queries));
        put("router_probes", json!(self.router_probes));
        put("router_queries", json!(self.router_queries));
        put("rope_theta", json!(self.rope_theta));
        put("dtype", json!(self.dtype));
        put("flash", json!(self.flash));
        put("engram_orders", json!(self.engram_orders));
        put("engram_heads", json!(self.engram_heads));
        put("engram_slots", json!(self.engram_slots));
        put("engram_seed_heads", json!(self.engram_seed_heads));
        put("engram_layers", json!(self.engram_layers));
        put("global_layers", json!(self.global_layers));
        put("sliding_window", json!(self.sliding_window));
        put("ladder_depths", json!(self.ladder_depths));
        put("ladder_sample", json!(self.ladder_sample));
        put("ladder_widths", json!(self.ladder_widths));
        put("ladder_order", json!(self.ladder_order));
        put("mhc_lanes", json!(self.mhc_lanes));
        put("qkv_conv_taps", json!(self.qkv_conv_taps));
        put("out_vocab", json!(self.out_vocab));
        put("kv_window", json!(self.kv_window));
        put("kv_bits", json!(self.kv_bits));
        put("act_bits", json!(self.act_bits));
        put("weight_bits", json!(self.weight_bits));
        for (k, v) in &self.extra {
            m.insert(k.clone(), v.clone());
        }
        m
    }

    /// `(qk_head_dim, v_head_dim)` with the legacy even split as fallback.
    pub fn head_dims(&self) -> (usize, usize) {
        let legacy = self.d_model / self.num_heads;
        let qk = if self.qk_head_dim > 0 { self.qk_head_dim } else { legacy };
        let v = if self.v_head_dim > 0 { self.v_head_dim } else { legacy };
        (qk, v)
    }

    /// `(orders, heads per order, sub_dim)` of the engram tables.
    pub fn engram_geometry(&self) -> (Vec<usize>, usize, usize) {
        let orders = self.engram_orders.clone();
        let heads = if self.engram_heads > 0 { self.engram_heads } else { (self.d_model / (orders.len() * ENGRAM_SUB_DIM)).max(1) };
        let sub_dim = self.d_model / (orders.len() * heads);
        (orders, heads, sub_dim)
    }

    pub fn num_engram_tables(&self) -> usize {
        let (orders, heads, _) = self.engram_geometry();
        orders.len() * heads
    }

    pub fn engram_dilation(&self) -> usize {
        self.engram_orders.iter().copied().max().unwrap_or(1)
    }

    /// The Hadamard MLP width: `d_model` rounded up to a power of two.
    pub fn hada_n(&self) -> usize {
        self.d_model.next_power_of_two()
    }

    pub fn hada_split(&self) -> bool {
        !self.ladder_widths.is_empty()
    }

    pub fn out_rows(&self) -> usize {
        if self.out_vocab > 0 { self.out_vocab } else { self.vocab_size }
    }

    pub fn is_global(&self, layer: usize) -> bool {
        self.global_layers.contains(&layer)
    }

    /// The attention window a layer sees, `None` for full causal attention.
    pub fn layer_window(&self, layer: usize) -> Option<usize> {
        (self.sliding_window > 0 && !self.is_global(layer)).then_some(self.sliding_window)
    }
}

/// `_hada_blocks`: the Kronecker split `n = a * b`, `a` the power of two at
/// half the bit length.
pub fn hada_blocks(n: usize) -> (usize, usize) {
    let bits = usize::BITS - (n - 1).leading_zeros();
    let a = 1usize << (bits / 2);
    (a, n / a)
}

fn ladder_bisection(num_layers: usize) -> Result<Vec<usize>> {
    if num_layers < 1 {
        bail!("models require at least one layer");
    }
    if num_layers == 1 {
        return Ok(vec![0]);
    }
    let mut selected = vec![0, num_layers - 1];
    let mut order = selected.clone();
    while order.len() < num_layers {
        selected.sort_unstable();
        // Largest gap wins, ties go to the leftmost gap.
        let (_, left, right) = selected
            .windows(2)
            .filter(|w| w[1] - w[0] > 1)
            .map(|w| (w[1] - w[0], w[0], w[1]))
            .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
            .expect("a gap remains while layers are unselected");
        let candidate = (left + right) / 2;
        selected.push(candidate);
        order.push(candidate);
    }
    Ok(order)
}

/// Block selection order: the saved order of a sliced rung, else bisection.
pub fn ladder_order(config: &Config) -> Result<Vec<usize>> {
    if !config.ladder_order.is_empty() {
        let mut sorted = config.ladder_order.clone();
        sorted.sort_unstable();
        if sorted != (0..config.num_layers).collect::<Vec<_>>() {
            bail!("ladder_order {:?} is not a permutation of {} blocks", config.ladder_order, config.num_layers);
        }
        return Ok(config.ladder_order.clone());
    }
    ladder_bisection(config.num_layers)
}

pub fn ladder_layer_order(num_layers: usize) -> Result<Vec<usize>> {
    ladder_bisection(num_layers)
}

/// Sorted original-layer indices kept by the `depth` rung.
pub fn ladder_layer_indices(config: &Config, depth: usize) -> Result<Vec<usize>> {
    if !(2..=config.num_layers).contains(&depth) {
        bail!("ladder depth must be in [2, {}], got {depth}", config.num_layers);
    }
    let mut kept = ladder_order(config)?[..depth].to_vec();
    kept.sort_unstable();
    Ok(kept)
}

pub fn ladder_layer_ranks(config: &Config) -> Result<Vec<usize>> {
    let order = ladder_order(config)?;
    let mut ranks = vec![0; order.len()];
    for (rank, &layer) in order.iter().enumerate() {
        ranks[layer] = rank;
    }
    Ok(ranks)
}

/// The config of the `depth` rung, layer indices remapped.
pub fn ladder_config(config: &Config, depth: usize) -> Result<Config> {
    assert!(depth < config.num_layers);
    let selected = ladder_layer_indices(config, depth)?;
    let remap = |l: &usize| selected.iter().position(|s| s == l);
    let mut c = config.clone();
    c.num_layers = depth;
    c.ladder_order = ladder_order(config)?.iter().filter_map(remap).collect();
    c.global_layers = config.global_layers.iter().filter_map(remap).collect();
    c.engram_layers = config.engram_layers.iter().filter_map(remap).collect();
    c.ladder_depths.clear();
    c.ladder_sample = false;
    Ok(c)
}

pub const HEAD_KEYS: [&str; 3] = ["embedding_head", "confidence_head", "router_head"];

/// `ladder_slice`: cut the stacked per-layer params down to the rung.
pub fn ladder_slice(params: &Params, config: &Config, depth: usize) -> Result<Params> {
    let selected = ladder_layer_indices(config, depth)?;
    let rows: Vec<usize> = std::iter::once(0).chain(selected.iter().map(|l| l + 1)).collect();
    let mut out = Params::new();
    for (k, v) in params {
        if let Some(rest) = k.strip_prefix("stack/") {
            if rest.starts_with("final_norm") {
                out.insert(k.clone(), v.clone());
            } else {
                out.insert(k.clone(), v.take0(&selected));
            }
        } else if let Some(head) = HEAD_KEYS.iter().find(|h| k.starts_with(&format!("{h}/"))) {
            let leaf = &k[head.len() + 1..];
            match leaf {
                "probes" | "gain" => {
                    out.insert(k.clone(), v.take0(&rows));
                }
                "row_bias" => {
                    // (q, L+1, k): take along axis 1.
                    let (q, l1, kk) = (v.shape[0], v.shape[1], v.shape[2]);
                    let mut data = Vec::with_capacity(q * rows.len() * kk);
                    for qi in 0..q {
                        for &r in &rows {
                            let base = (qi * l1 + r) * kk;
                            data.extend_from_slice(&v.data[base..base + kk]);
                        }
                    }
                    out.insert(k.clone(), crate::Tensor::new(vec![q, rows.len(), kk], data));
                }
                _ => {
                    out.insert(k.clone(), v.clone());
                }
            }
        } else if let Some(rest) = k.strip_prefix("engrams_") {
            let (site, leaf) = rest.split_once('/').unwrap_or((rest, ""));
            let site: usize = site.parse()?;
            let layer = config.engram_layers[site];
            if selected.contains(&layer) {
                let new_site = config.engram_layers[..site].iter().filter(|l| selected.contains(l)).count();
                out.insert(format!("engrams_{new_site}/{leaf}"), v.clone());
            }
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    Ok(out)
}

pub const KV_BUDGET_BYTES: usize = 11 * 1024 * 1024 + 512 * 1024;
pub const KV_GROUP: usize = 32;
pub const KV_WINDOW_MIN: usize = 160;

pub fn kv_budget_window(config: &Config) -> usize {
    let (qk, v) = config.head_dims();
    let kd = config.num_kv_heads * qk;
    let vd = config.num_kv_heads * v;
    let (d, l) = (config.d_model, config.num_layers);
    let sites = config.engram_layers.len();
    let per_layer = kd + vd + (kd / KV_GROUP + vd / KV_GROUP) * 4;
    let per_site = d + (d / KV_GROUP) * 4;
    let sw = config.sliding_window;
    if sw > 0 {
        let n_global = config.global_layers.len();
        let local = sw.min(config.max_seq_len);
        let fixed = ((l - n_global) * per_layer + sites * per_site) * local;
        if n_global == 0 {
            return if fixed <= KV_BUDGET_BYTES { config.max_seq_len } else { KV_WINDOW_MIN.max(local) };
        }
        // Python floor division on a possibly negative numerator.
        let num = KV_BUDGET_BYTES as i64 - fixed as i64;
        let den = (n_global * per_layer) as i64;
        let ctx = num.div_euclid(den).div_euclid(KV_GROUP as i64) * KV_GROUP as i64;
        return (KV_WINDOW_MIN as i64).max(ctx.min(config.max_seq_len as i64)) as usize;
    }
    let per_pos = l * per_layer + sites * per_site;
    let window = (KV_BUDGET_BYTES / per_pos) / KV_GROUP * KV_GROUP;
    KV_WINDOW_MIN.max(window.min(config.max_seq_len))
}

pub fn effective_kv_window(config: &Config) -> usize {
    let budget = kv_budget_window(config);
    if config.kv_window > 0 { budget.min(config.kv_window) } else { budget }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisection_order_matches_python() {
        // _ladder_layer_order(20) from the reference.
        assert_eq!(ladder_layer_order(20).unwrap(), vec![0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18]);
    }

    #[test]
    fn hada_blocks_split() {
        assert_eq!(hada_blocks(1024), (32, 32));
        assert_eq!(hada_blocks(64), (8, 8));
        assert_eq!(hada_blocks(512), (16, 32));
    }
}
