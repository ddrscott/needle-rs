//! The training graph: a batched forward pass over the quantized network and
//! its hand-derived reverse pass, producing gradients for the LoRA factors
//! only.
//!
//! Numerics follow `finetune_local` in the reference exactly: every matmul
//! weight is CQ-W4 fake-quantized (the LoRA targets after merging), every
//! matmul input is A8 fake-quantized, queries and KV are int8 per head, and
//! all quantizers are straight-through for the gradient.
//!
//! Three things make it fast without changing the math:
//!
//! * **Shared prefixes.** Tool-calling data repeats one system+tools prompt
//!   across every example. A batch is laid out as rows, and positions the
//!   sequences share (their longest common prefix) are one set of rows that
//!   every sequence's attention reads. Causality makes this exact; the
//!   reverse pass sums each sequence's gradient into the shared rows.
//! * **GEMM attention with recompute.** Scores and outputs are matrix
//!   products per (sequence, head); only a log-sum-exp per query row is kept,
//!   and the reverse recomputes probabilities from it.
//! * **Low-rank LoRA gradients.** `dA = s X^T (dY B^T)` and
//!   `dB = s (X A)^T dY`; the dense weight gradient is never formed.
//!
//! Memory follows the reference's `nn.remat`: the forward keeps each layer's
//! input stream, the reverse recomputes one layer with a tape at a time.

use needle_core::config::{Config, ENGRAM_CONV_TAPS, ENGRAM_PRIME, ENGRAM_SEED};
use needle_core::quant::{ACT_BITS, CQ_GROUP_SIZE, cq_quantize_rows, fake_quant_rows, ste};
use needle_engine::Weights;
use needle_engine::hada::{MlpTape, mlp_bwd, mlp_fwd};
use needle_engine::linalg::{dot, gemm_strided, matmul, matmul_into, matmul_t, matmul_tn_into, rms_unit, sigmoid, zc_rms_norm};
use needle_engine::prof::span;
use needle_engine::weights::LayerWeights;
use rayon::prelude::*;

/// LoRA targets in `flatten_dict` order of the (key-sorted) param tree,
/// which is the order the reference splits init keys in.
pub const TARGETS: [&str; 5] = ["gate_proj", "k_proj", "out_proj", "q_proj", "v_proj"];
pub const GATE: usize = 0;
pub const K: usize = 1;
pub const OUT: usize = 2;
pub const Q: usize = 3;
pub const V: usize = 4;

/// `(in, out)` of each target (Flax kernel orientation).
pub fn target_dims(c: &Config) -> [(usize, usize); 5] {
    let (qk, vd) = c.head_dims();
    let d = c.d_model;
    [(d, c.num_heads * vd), (d, c.num_kv_heads * qk), (c.num_heads * vd, d), (d, c.num_heads * qk), (d, c.num_kv_heads * vd)]
}

/// Stacked LoRA factors: `a[t]` is `[L, in, r]`, `b[t]` is `[L, r, out]`.
#[derive(Clone, Debug)]
pub struct LoraParams {
    pub rank: usize,
    pub a: [Vec<f32>; 5],
    pub b: [Vec<f32>; 5],
}

impl LoraParams {
    pub fn zeros_like(&self) -> Self {
        Self { rank: self.rank, a: self.a.clone().map(|v| vec![0.0; v.len()]), b: self.b.clone().map(|v| vec![0.0; v.len()]) }
    }

    /// Every tensor in `flatten_dict` order: (A, B) per target.
    pub fn tensors(&self) -> impl Iterator<Item = &Vec<f32>> {
        (0..5).flat_map(move |t| [&self.a[t], &self.b[t]])
    }

    pub fn tensors_mut(&mut self) -> impl Iterator<Item = &mut Vec<f32>> {
        self.a.iter_mut().zip(self.b.iter_mut()).flat_map(|(a, b)| [a, b])
    }
}

/// One sequence of a batch: the row of each position, and the first
/// position whose row it owns (earlier rows belong to the shared prefix).
#[derive(Clone, Debug)]
pub struct SeqRows {
    pub rows: Vec<usize>,
    pub owned_from: usize,
}

/// A training batch laid out as rows.
pub struct Batch {
    pub ids: Vec<u32>,
    /// Position of each row in its sequence(s).
    pub pos: Vec<usize>,
    /// Sequence that computes each row.
    pub owner: Vec<usize>,
    pub seqs: Vec<SeqRows>,
    /// `(row, target token, weight)`: logits at `row` predict `target`.
    pub targets: Vec<(usize, u32, f32)>,
    /// `max(sum(mask[:, 1:]), 1)`.
    pub denom: f32,
}

impl Batch {
    /// From padded `[B, T]` ids and loss masks (the reference's arrays).
    /// Each sequence is cut after its last scored position and the
    /// sequences' longest common prefix becomes shared rows.
    pub fn from_padded(rows: &[(&[u32], &[f32])]) -> Self {
        Self::build(rows, true)
    }

    /// As [`Batch::from_padded`] with every sequence on its own rows.
    pub fn unshared(rows: &[(&[u32], &[f32])]) -> Self {
        Self::build(rows, false)
    }

    fn build(rows: &[(&[u32], &[f32])], share: bool) -> Self {
        let denom = rows.iter().map(|(_, m)| m.iter().skip(1).sum::<f32>()).sum::<f32>().max(1.0);
        let live: Vec<(&[u32], &[f32], usize)> =
            rows.iter().filter_map(|(ids, mask)| (1..ids.len()).rev().find(|&j| mask[j] > 0.0).map(|last| (*ids, *mask, last))).collect();
        let mut prefix = 0;
        if share && live.len() > 1 {
            let first = &live[0].0[..live[0].2];
            prefix = first.len();
            for (ids, _, len) in &live[1..] {
                let n = first.iter().zip(&ids[..*len]).take_while(|(a, b)| a == b).count();
                prefix = prefix.min(n);
            }
        }
        let mut b = Batch { ids: vec![], pos: vec![], owner: vec![], seqs: vec![], targets: vec![], denom };
        for (s, (ids, mask, len)) in live.iter().enumerate() {
            let shared = if s == 0 { 0 } else { prefix };
            let mut seq_rows: Vec<usize> = (0..shared).collect();
            for p in shared..*len {
                seq_rows.push(b.ids.len());
                b.ids.push(ids[p]);
                b.pos.push(p);
                b.owner.push(s);
            }
            for p in 0..*len {
                if mask[p + 1] > 0.0 {
                    b.targets.push((seq_rows[p], ids[p + 1], mask[p + 1]));
                }
            }
            b.seqs.push(SeqRows { rows: seq_rows, owned_from: shared });
        }
        b
    }

    pub fn rows(&self) -> usize {
        self.ids.len()
    }

    /// Row of position `pos - back` in the sequence that owns `row`.
    #[inline]
    fn back(&self, row: usize, back: usize) -> Option<usize> {
        let p = self.pos[row];
        (p >= back).then(|| self.seqs[self.owner[row]].rows[p - back])
    }
}

/// Frozen weights plus the base copies of the LoRA targets.
pub struct Graph {
    pub w: Weights,
    /// Finetune numerics (CQ-W4 weights, A8 activations, int8 q/KV). Off only
    /// for exact float gradient checks.
    pub quant: bool,
    /// Raw (unquantized) target weights per layer, engine layout `[out, in]`.
    base: Vec<[Vec<f32>; 5]>,
    pub scale: f32,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    site_of_layer: Vec<Option<usize>>,
}

fn target_mut(l: &mut LayerWeights, t: usize) -> &mut Vec<f32> {
    match t {
        GATE => &mut l.wgate,
        K => &mut l.wk,
        OUT => &mut l.wout,
        Q => &mut l.wq,
        _ => &mut l.wv,
    }
}

fn target_ref(l: &LayerWeights, t: usize) -> &Vec<f32> {
    match t {
        GATE => &l.wgate,
        K => &l.wk,
        OUT => &l.wout,
        Q => &l.wq,
        _ => &l.wv,
    }
}

/// Tape of one layer's forward, enough to run its reverse.
#[derive(Default)]
struct Tape {
    p: Vec<f32>,
    /// Sinkhorn log-states per row: `(2 * iters + 1) * n * n`.
    sk: Vec<f32>,
    u: Vec<f32>,
    y: Vec<f32>,
    x1: Vec<f32>,
    alpha: Vec<f32>,
    hq: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    qq: Vec<f32>,
    kq: Vec<f32>,
    vq: Vec<f32>,
    lse: Vec<f32>,
    att: Vec<f32>,
    gp: Vec<f32>,
    oq: Vec<f32>,
    o: Vec<f32>,
    x2: Vec<f32>,
    mlp: MlpTape,
}

/// Engram keys and tapped values per site, `[rows, d]` each.
struct EngramKv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// Loss and LoRA gradients of one batch.
pub struct StepOut {
    pub loss: f32,
    pub grads: LoraParams,
}

const SINKHORN_ITERS: usize = 20;

fn quantize(on: bool, x: &mut [f32], width: usize, bits: u32) {
    if on {
        fake_quant_rows(x, width, bits);
    }
}

/// `ZCRMSNorm` backward: `dx` for `y = (1+s) x / rms(x)` (no scale: `rms_unit`).
fn zc_rms_norm_bwd(x: &[f32], scale: Option<&[f32]>, dy: &[f32], dx: &mut [f32], accumulate: bool) {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let r = (ms + 1e-6).sqrt();
    let g = |i: usize| scale.map_or(dy[i], |s| (1.0 + s[i]) * dy[i]);
    let gx: f32 = (0..x.len()).map(|i| g(i) * x[i]).sum();
    let c = gx / (n * r * r * r);
    for i in 0..x.len() {
        let v = g(i) / r - x[i] * c;
        if accumulate { dx[i] += v } else { dx[i] = v }
    }
}

fn lane_mean(x: &[f32], n: usize, d: usize, out: &mut [f32]) {
    out.copy_from_slice(&x[..d]);
    for lane in 1..n {
        for (o, v) in out.iter_mut().zip(&x[lane * d..(lane + 1) * d]) {
            *o += v;
        }
    }
    for o in out.iter_mut() {
        *o /= n as f32;
    }
}

/// Sinkhorn in log space; writes every half-step state into `states`
/// (`(2 * iters + 1) * n * n`) and returns `exp` of the last.
fn sinkhorn_fwd(r: &[f32], n: usize, states: &mut [f32]) -> Vec<f32> {
    let nn = n * n;
    let mut k = r.to_vec();
    states[..nn].copy_from_slice(&k);
    for it in 0..SINKHORN_ITERS {
        for row in 0..n {
            let rr = &mut k[row * n..(row + 1) * n];
            let m = rr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + rr.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            rr.iter_mut().for_each(|v| *v -= lse);
        }
        states[(2 * it + 1) * nn..(2 * it + 2) * nn].copy_from_slice(&k);
        for col in 0..n {
            let m = (0..n).map(|rw| k[rw * n + col]).fold(f32::NEG_INFINITY, f32::max);
            let lse = m + (0..n).map(|rw| (k[rw * n + col] - m).exp()).sum::<f32>().ln();
            for rw in 0..n {
                k[rw * n + col] -= lse;
            }
        }
        states[(2 * it + 2) * nn..(2 * it + 3) * nn].copy_from_slice(&k);
    }
    k.into_iter().map(f32::exp).collect()
}

/// Reverse of [`sinkhorn_fwd`] given `dH`, from the stored states.
fn sinkhorn_bwd(states: &[f32], dh: &[f32], n: usize) -> Vec<f32> {
    let nn = n * n;
    let last = &states[2 * SINKHORN_ITERS * nn..];
    let mut dl: Vec<f32> = dh.iter().zip(last).map(|(g, l)| g * l.exp()).collect();
    for it in (0..SINKHORN_ITERS).rev() {
        let inp = &states[(2 * it + 1) * nn..(2 * it + 2) * nn];
        for col in 0..n {
            let m = (0..n).map(|rw| inp[rw * n + col]).fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = (0..n).map(|rw| (inp[rw * n + col] - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let s: f32 = (0..n).map(|rw| dl[rw * n + col]).sum();
            for rw in 0..n {
                dl[rw * n + col] -= e[rw] / z * s;
            }
        }
        let inp = &states[2 * it * nn..(2 * it + 1) * nn];
        for rw in 0..n {
            let row = &inp[rw * n..(rw + 1) * n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let s: f32 = dl[rw * n..(rw + 1) * n].iter().sum();
            for c in 0..n {
                dl[rw * n + c] -= e[c] / z * s;
            }
        }
    }
    dl
}

/// Rows `rows` of `src` (width `w`, columns `c0..c0+cw`) gathered densely.
fn gather(src: &[f32], w: usize, rows: &[usize], c0: usize, cw: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * cw);
    for &r in rows {
        out.extend_from_slice(&src[r * w + c0..r * w + c0 + cw]);
    }
    out
}

impl Graph {
    /// Freeze `w` under CQ-W4 (keeping the raw LoRA targets aside).
    pub fn new(w: Weights, scale: f32) -> Self {
        Self::with_numerics(w, scale, true)
    }

    pub fn with_numerics(mut w: Weights, scale: f32, quant: bool) -> Self {
        let base: Vec<[Vec<f32>; 5]> = w.layers.iter().map(|l| std::array::from_fn(|t| target_ref(l, t).clone())).collect();
        if quant {
            w.cq_fake_quantize(needle_core::quant::WEIGHT_BITS);
        }
        let c = w.config.clone();
        let (qk, _) = c.head_dims();
        let half = qk / 2;
        let theta = c.rope_theta as f32;
        let freqs: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / qk as f32)).collect();
        let (mut rope_cos, mut rope_sin) = (Vec::new(), Vec::new());
        for t in 0..c.max_seq_len {
            for &f in &freqs {
                let a = t as f32 * f;
                rope_cos.push(a.cos());
                rope_sin.push(a.sin());
            }
        }
        let mut site_of_layer = vec![None; c.num_layers];
        for (s, &l) in c.engram_layers.iter().enumerate() {
            site_of_layer[l] = Some(s);
        }
        Self { w, quant, base, scale, rope_cos, rope_sin, site_of_layer }
    }

    pub fn config(&self) -> &Config {
        &self.w.config
    }

    /// `cq_ste(W + scale * A @ B)` into the live target weights.
    pub fn merge(&mut self, lora: &LoraParams) {
        let c = self.w.config.clone();
        let dims = target_dims(&c);
        let r = lora.rank;
        let scale = self.scale;
        let base = &self.base;
        let quant = self.quant;
        self.w.layers.par_iter_mut().enumerate().for_each(|(l, lw)| {
            for (t, &(din, dout)) in dims.iter().enumerate() {
                let a = &lora.a[t][l * din * r..(l + 1) * din * r];
                let b = &lora.b[t][l * r * dout..(l + 1) * r * dout];
                // (A @ B)^T = B^T A^T straight into the [out, in] layout.
                let mut delta = vec![0f32; dout * din];
                gemm_strided(dout, din, r, 1.0, b, dout, true, a, r, true, 0.0, &mut delta, din);
                let raw = &base[l][t];
                let merged: Vec<f32> = raw.iter().zip(&delta).map(|(w, dw)| w + scale * dw).collect();
                *target_mut(lw, t) = if quant {
                    let q = cq_quantize_rows(&merged, din, needle_core::quant::WEIGHT_BITS, CQ_GROUP_SIZE);
                    ste(&merged, &q)
                } else {
                    merged
                };
            }
        });
    }

    fn embed(&self, ids: &[u32]) -> Vec<f32> {
        let d = self.w.config.d_model;
        let scale = (d as f64).sqrt() as f32;
        let mut x0 = vec![0f32; ids.len() * d];
        x0.par_chunks_mut(d).zip(ids.par_iter()).for_each(|(o, &tok)| {
            for (a, b) in o.iter_mut().zip(&self.w.embedding[tok as usize * d..(tok as usize + 1) * d]) {
                *a = b * scale;
            }
        });
        x0
    }

    fn engram_kv(&self, batch: &Batch) -> Option<EngramKv> {
        let c = &self.w.config;
        if c.engram_layers.is_empty() {
            return None;
        }
        let d = c.d_model;
        let (orders, heads, sub) = c.engram_geometry();
        let tables = orders.len() * heads;
        let stride = if c.engram_seed_heads > 0 { c.engram_seed_heads } else { heads };
        let dil = c.engram_dilation();
        let n = batch.rows();
        let mut idx = vec![0usize; n * tables];
        let mut ok = vec![0f32; n * tables];
        for row in 0..n {
            let pos = batch.pos[row];
            for (oi, &order) in orders.iter().enumerate() {
                for h in 0..heads {
                    let mut acc = ENGRAM_SEED.wrapping_mul((oi * stride + h + 1) as u32);
                    for j in 0..order {
                        let tok = batch.back(row, j).map_or(0, |r| batch.ids[r]);
                        acc = (acc ^ tok).wrapping_mul(ENGRAM_PRIME);
                    }
                    acc ^= acc >> 15;
                    let tbl = oi * heads + h;
                    idx[row * tables + tbl] = (acc % c.engram_slots as u32) as usize;
                    ok[row * tables + tbl] = if pos + 1 >= order { 1.0 } else { 0.0 };
                }
            }
        }
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for ew in &self.w.engrams {
            let mut e = vec![0f32; n * d];
            e.par_chunks_mut(d).enumerate().for_each(|(i, row)| {
                for tbl in 0..tables {
                    let src = (tbl * c.engram_slots + idx[i * tables + tbl]) * sub;
                    let okv = ok[i * tables + tbl];
                    for (o, v) in row[tbl * sub..(tbl + 1) * sub].iter_mut().zip(&ew.tables[src..src + sub]) {
                        *o = v * okv;
                    }
                }
            });
            quantize(self.quant, &mut e, d, ACT_BITS);
            let k = matmul_t(&e, n, d, &ew.wk, d);
            let vpre = matmul_t(&e, n, d, &ew.wv, d);
            let mut v = vec![0f32; n * d];
            v.par_chunks_mut(d).enumerate().for_each(|(row, dst)| {
                for j in 0..ENGRAM_CONV_TAPS {
                    if let Some(r) = batch.back(row, j * dil) {
                        for ((o, a), b) in dst.iter_mut().zip(&ew.taps[j * d..(j + 1) * d]).zip(&vpre[r * d..(r + 1) * d]) {
                            *o += a * b;
                        }
                    }
                }
            });
            ks.push(k);
            vs.push(v);
        }
        Some(EngramKv { k: ks, v: vs })
    }

    /// One layer forward over the lane stream `xs [rows, n*d]`, in place.
    fn layer_fwd(&self, l: usize, batch: &Batch, xs: &mut [f32], engram: Option<&EngramKv>, mut tape: Option<&mut Tape>) {
        let c = &self.w.config;
        let lw = &self.w.layers[l];
        let (n, d) = (c.mhc_lanes, c.d_model);
        let nc = n * d;
        let ncols = 2 * n + n * n;
        let nn = n * n;
        let rows = batch.rows();
        let lane = l % n;

        let (p, u) = span("mhc_read", || {
            let mut nx = vec![0f32; rows * nc];
            nx.par_chunks_mut(nc).zip(xs.par_chunks(nc)).for_each(|(o, x)| {
                rms_unit(x, o);
                quantize(self.quant, o, nc, ACT_BITS);
            });
            let p = matmul_t(&nx, rows, nc, &lw.phi, ncols);
            drop(nx);
            let mut u = vec![0f32; rows * d];
            u.par_chunks_mut(d).enumerate().for_each(|(i, ui)| {
                let pi = &p[i * ncols..(i + 1) * ncols];
                let x = &xs[i * nc..(i + 1) * nc];
                for j in 0..n {
                    let off = if j == lane { 4.0 } else { -4.0 };
                    let hp = sigmoid(lw.a_pre * pi[j] + lw.b_pre[j] + off);
                    for (o, v) in ui.iter_mut().zip(&x[j * d..(j + 1) * d]) {
                        *o += hp * v;
                    }
                }
            });
            (p, u)
        });

        let y = self.block_fwd(l, batch, &u, engram, tape.as_deref_mut());

        let states_per = (2 * SINKHORN_ITERS + 1) * nn;
        let mut sk = if tape.is_some() { vec![0f32; rows * states_per] } else { vec![] };
        span("mhc_write", || {
            let write = |i: usize, x: &mut [f32], st: &mut [f32]| {
                let pi = &p[i * ncols..(i + 1) * ncols];
                let r: Vec<f32> = (0..nn).map(|kk| lw.a_res * pi[2 * n + kk] + lw.b_res[kk]).collect();
                let h = sinkhorn_fwd(&r, n, st);
                let old = x.to_vec();
                let yi = &y[i * d..(i + 1) * d];
                for a in 0..n {
                    let off = if a == lane { 0.0 } else { -4.0 };
                    let hp = 2.0 * sigmoid(lw.a_post * pi[n + a] + lw.b_post[a] + off);
                    let dst = &mut x[a * d..(a + 1) * d];
                    for (e, o) in dst.iter_mut().enumerate() {
                        let mut acc = 0f32;
                        for b in 0..n {
                            acc += h[a * n + b] * old[b * d + e];
                        }
                        *o = acc + hp * yi[e];
                    }
                }
            };
            if sk.is_empty() {
                xs.par_chunks_mut(nc).enumerate().for_each(|(i, x)| {
                    let mut st = vec![0f32; states_per];
                    write(i, x, &mut st);
                });
            } else {
                xs.par_chunks_mut(nc).zip(sk.par_chunks_mut(states_per)).enumerate().for_each(|(i, (x, st))| write(i, x, st));
            }
        });
        if let Some(t) = tape {
            t.p = p;
            t.sk = sk;
            t.u = u;
            t.y = y;
        }
    }

    fn block_fwd(&self, l: usize, batch: &Batch, u: &[f32], engram: Option<&EngramKv>, tape: Option<&mut Tape>) -> Vec<f32> {
        let c = &self.w.config;
        let lw = &self.w.layers[l];
        let d = c.d_model;
        let rows = batch.rows();
        let mut x = u.to_vec();
        let mut alpha = vec![];
        if let (Some(site), Some(e)) = (self.site_of_layer[l], engram) {
            let (ek, ev) = (&e.k[site], &e.v[site]);
            let inv = 1.0 / (d as f64).sqrt() as f32;
            alpha = vec![0f32; rows];
            x.par_chunks_mut(d).zip(alpha.par_iter_mut()).enumerate().for_each(|(i, (xi, al))| {
                let mut a = vec![0f32; d];
                let mut b = vec![0f32; d];
                rms_unit(xi, &mut a);
                rms_unit(&ek[i * d..(i + 1) * d], &mut b);
                *al = sigmoid(dot(&a, &b) * inv);
                for (o, v) in xi.iter_mut().zip(&ev[i * d..(i + 1) * d]) {
                    *o += *al * v;
                }
            });
        }
        let x1 = x;
        let mut hq = x1.clone();
        hq.par_chunks_mut(d).for_each(|r| {
            zc_rms_norm(r, &lw.norm_in);
            quantize(self.quant, r, d, ACT_BITS);
        });
        let att = span("attention_fwd", || self.attention_fwd(l, batch, &hq));
        let mut x2 = x1.clone();
        let gate = sigmoid(lw.attn_gate);
        x2.par_chunks_mut(d).zip(att.o.par_chunks(d)).for_each(|(xr, o)| {
            let mut a = o.to_vec();
            zc_rms_norm(&mut a, &lw.post_norm);
            for (xv, av) in xr.iter_mut().zip(&a) {
                *xv += gate * av;
            }
        });
        let mut h2 = x2.clone();
        h2.par_chunks_mut(d).for_each(|r| zc_rms_norm(r, &lw.pre_hada));
        let mut mtape = MlpTape::default();
        let keep = tape.is_some();
        let mlp = span("mlp_fwd", || mlp_fwd(lw, &self.w.perm1, &self.w.perm2, d, &h2, keep.then_some(&mut mtape)));
        let mut out = x2.clone();
        out.par_iter_mut().zip(mlp.par_iter()).zip(u.par_iter()).for_each(|((o, m), uu)| *o = (*o + m) - uu);
        if let Some(t) = tape {
            t.x1 = x1;
            t.alpha = alpha;
            t.hq = hq;
            t.q = att.q;
            t.k = att.k;
            t.qq = att.qq;
            t.kq = att.kq;
            t.vq = att.vq;
            t.lse = att.lse;
            t.att = att.att;
            t.gp = att.gp;
            t.oq = att.oq;
            t.o = att.o;
            t.x2 = x2;
            t.mlp = mtape;
        }
        out
    }

    fn rope(&self, row: &mut [f32], pos: usize) {
        let half = row.len() / 2;
        let (cs, sn) = (&self.rope_cos[pos * half..(pos + 1) * half], &self.rope_sin[pos * half..(pos + 1) * half]);
        for i in 0..half {
            let (x1, x2) = (row[i], row[i + half]);
            row[i] = x1 * cs[i] - x2 * sn[i];
            row[i + half] = x2 * cs[i] + x1 * sn[i];
        }
    }

    /// Causal depthwise conv over positions (`taps` wide), per owner sequence.
    fn conv(&self, batch: &Batch, pre: &[f32], w: &[f32], width: usize) -> Vec<f32> {
        let taps = self.w.config.qkv_conv_taps;
        if taps == 0 {
            return pre.to_vec();
        }
        let mut out = vec![0f32; pre.len()];
        out.par_chunks_mut(width).enumerate().for_each(|(row, dst)| {
            for j in 0..taps {
                if let Some(r) = batch.back(row, j) {
                    for ((o, a), b) in dst.iter_mut().zip(&w[j * width..(j + 1) * width]).zip(&pre[r * width..(r + 1) * width]) {
                        *o += a * b;
                    }
                }
            }
        });
        out
    }

    /// Transpose of [`Graph::conv`]: scatter each row's gradient back to the
    /// rows it read.
    fn conv_bwd(&self, batch: &Batch, dpost: &[f32], w: &[f32], width: usize) -> Vec<f32> {
        let taps = self.w.config.qkv_conv_taps;
        if taps == 0 {
            return dpost.to_vec();
        }
        let mut out = vec![0f32; dpost.len()];
        for row in 0..batch.rows() {
            for j in 0..taps {
                if let Some(r) = batch.back(row, j) {
                    let (src, wj) = (&dpost[row * width..(row + 1) * width], &w[j * width..(j + 1) * width]);
                    for ((o, a), b) in out[r * width..(r + 1) * width].iter_mut().zip(wj).zip(src) {
                        *o += a * b;
                    }
                }
            }
        }
        out
    }

    fn attention_fwd(&self, l: usize, batch: &Batch, hq: &[f32]) -> AttFwd {
        let c = &self.w.config;
        let lw = &self.w.layers[l];
        let d = c.d_model;
        let rows = batch.rows();
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        let qp = matmul_t(hq, rows, d, &lw.wq, qd);
        let kp = matmul_t(hq, rows, d, &lw.wk, kd);
        let vp = matmul_t(hq, rows, d, &lw.wv, vdim);
        let gp = matmul_t(hq, rows, d, &lw.wgate, nh * vd);
        let q = self.conv(batch, &qp, &lw.q_taps, qd);
        let k = self.conv(batch, &kp, &lw.k_taps, kd);
        let v = self.conv(batch, &vp, &lw.v_taps, vdim);
        drop((qp, kp, vp));
        let mut qq = q.clone();
        qq.par_chunks_mut(qd).enumerate().for_each(|(i, r)| {
            for h in r.chunks_mut(qk) {
                zc_rms_norm(h, &lw.q_norm);
                self.rope(h, batch.pos[i]);
            }
            quantize(self.quant, r, qk, 8);
        });
        let mut kq = k.clone();
        kq.par_chunks_mut(kd).enumerate().for_each(|(i, r)| {
            for h in r.chunks_mut(qk) {
                zc_rms_norm(h, &lw.k_norm);
                self.rope(h, batch.pos[i]);
            }
            quantize(self.quant, r, qk, 8);
        });
        let mut vq = v;
        quantize(self.quant, &mut vq, vd, 8);

        let (att, lse) = span("attn_core_fwd", || self.attn_core_fwd(l, batch, &qq, &kq, &vq));
        let mut oq = att.clone();
        oq.par_iter_mut().zip(gp.par_iter()).for_each(|(o, g)| *o *= sigmoid(*g));
        oq.par_chunks_mut(nh * vd).for_each(|r| quantize(self.quant, r, nh * vd, ACT_BITS));
        let o = matmul_t(&oq, rows, nh * vd, &lw.wout, d);
        AttFwd { q, k, qq, kq, vq, lse, att, gp, oq, o }
    }

    /// Probabilities for one (sequence, head): `[owned queries, keys]`,
    /// masked and softmaxed; with `lse_in`, normalised by the stored
    /// log-sum-exp instead (the reverse pass). Returns the log-sum-exps.
    fn attn_probs(&self, l: usize, seq: &SeqRows, qs: &[f32], kh: &[f32], lse_in: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let (qk, _) = self.w.config.head_dims();
        let len = seq.rows.len();
        let q0 = seq.owned_from;
        let lq = len - q0;
        let mut s = vec![0f32; lq * len];
        gemm_strided(lq, len, qk, 1.0, qs, qk, false, kh, qk, true, 0.0, &mut s, len);
        let window = self.w.config.layer_window(l);
        let mut lse = vec![0f32; lq];
        s.chunks_mut(len).zip(lse.iter_mut()).enumerate().for_each(|(i, (row, lo_out))| {
            let pos = q0 + i;
            let lo = window.map_or(0, |w| (pos + 1).saturating_sub(w));
            let live = &mut row[lo..=pos];
            let z = match lse_in {
                Some(prev) => prev[i],
                None => {
                    let m = live.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    m + live.iter().map(|v| (v - m).exp()).sum::<f32>().ln()
                }
            };
            for v in live.iter_mut() {
                *v = (*v - z).exp();
            }
            row[..lo].fill(0.0);
            row[pos + 1..].fill(0.0);
            *lo_out = z;
        });
        (s, lse)
    }

    fn attn_core_fwd(&self, l: usize, batch: &Batch, qq: &[f32], kq: &[f32], vq: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let c = &self.w.config;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        let group = nh / nkv;
        let fused = qk.max(vd);
        let qscale = ((fused as f64) / (qk as f64)).sqrt() as f32;
        let sscale = (1.0 / (fused as f64).sqrt()) as f32;
        let rows = batch.rows();
        let tasks: Vec<(usize, usize)> = (0..batch.seqs.len()).flat_map(|s| (0..nh).map(move |h| (s, h))).collect();
        let outs: Vec<(Vec<f32>, Vec<f32>)> = tasks
            .par_iter()
            .map(|&(s, h)| {
                let seq = &batch.seqs[s];
                let kvh = h / group;
                let kh = gather(kq, kd, &seq.rows, kvh * qk, qk);
                let vh = gather(vq, vdim, &seq.rows, kvh * vd, vd);
                let mut qs = gather(qq, qd, &seq.rows[seq.owned_from..], h * qk, qk);
                // score = dot(q * qscale, k) * sscale, the reference's fused path.
                qs.iter_mut().for_each(|v| *v = (*v * qscale) * sscale);
                let (p, lse) = self.attn_probs(l, seq, &qs, &kh, None);
                let lq = seq.rows.len() - seq.owned_from;
                let mut o = vec![0f32; lq * vd];
                gemm_strided(lq, vd, seq.rows.len(), 1.0, &p, seq.rows.len(), false, &vh, vd, false, 0.0, &mut o, vd);
                (o, lse)
            })
            .collect();
        let mut att = vec![0f32; rows * nh * vd];
        let mut lse_all = vec![0f32; rows * nh];
        for (&(s, h), (o, lse)) in tasks.iter().zip(outs) {
            let seq = &batch.seqs[s];
            for (i, &r) in seq.rows[seq.owned_from..].iter().enumerate() {
                att[r * nh * vd + h * vd..r * nh * vd + (h + 1) * vd].copy_from_slice(&o[i * vd..(i + 1) * vd]);
                lse_all[r * nh + h] = lse[i];
            }
        }
        (att, lse_all)
    }

    /// Loss of a batch (the reference's `eval_step`).
    pub fn loss(&self, batch: &Batch) -> f32 {
        let xs = self.forward_stream(batch, None);
        self.head(batch, &xs, false).0
    }

    fn forward_stream(&self, batch: &Batch, mut checkpoints: Option<&mut Vec<Vec<f32>>>) -> Vec<f32> {
        let c = &self.w.config;
        let (n, d) = (c.mhc_lanes, c.d_model);
        let x0 = self.embed(&batch.ids);
        let engram = self.engram_kv(batch);
        let mut xs = vec![0f32; batch.rows() * n * d];
        xs.par_chunks_mut(n * d).zip(x0.par_chunks(d)).for_each(|(o, x)| {
            for lane in 0..n {
                o[lane * d..(lane + 1) * d].copy_from_slice(x);
            }
        });
        for l in 0..c.num_layers {
            if let Some(ck) = checkpoints.as_deref_mut() {
                ck.push(xs.clone());
            }
            self.layer_fwd(l, batch, &mut xs, engram.as_ref(), None);
        }
        xs
    }

    /// Masked cross-entropy over the scored rows; with `grad`, also `dxs`.
    fn head(&self, batch: &Batch, xs: &[f32], grad: bool) -> (f32, Vec<f32>) {
        let c = &self.w.config;
        let (n, d) = (c.mhc_lanes, c.d_model);
        let vocab = c.out_rows();
        // Distinct scored rows (a shared row can carry several targets).
        let mut uniq: Vec<usize> = batch.targets.iter().map(|t| t.0).collect();
        uniq.sort_unstable();
        uniq.dedup();
        let slot = |row: usize| uniq.binary_search(&row).unwrap();
        let r = uniq.len();
        let mut hf = vec![0f32; r * d];
        let mut hm = vec![0f32; r * d];
        hf.par_chunks_mut(d).zip(hm.par_chunks_mut(d)).zip(uniq.par_iter()).for_each(|((f, m), &row)| {
            lane_mean(&xs[row * n * d..(row + 1) * n * d], n, d, m);
            f.copy_from_slice(m);
            zc_rms_norm(f, &self.w.final_norm);
            quantize(self.quant, f, d, ACT_BITS);
        });
        let emb = &self.w.embedding[..vocab * d];
        let logits = matmul_t(&hf, r, d, emb, vocab);
        let lse: Vec<f32> = logits
            .par_chunks(vocab)
            .map(|lg| {
                let m = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                m + lg.iter().map(|v| (v - m).exp()).sum::<f32>().ln()
            })
            .collect();
        let loss = batch
            .targets
            .iter()
            .map(|&(row, tgt, w)| {
                let s = slot(row);
                (lse[s] - logits[s * vocab + tgt as usize]) * w
            })
            .sum::<f32>()
            / batch.denom;
        if !grad {
            return (loss, vec![]);
        }
        // dlogits = sum over the row's targets of w/denom * (softmax - onehot).
        let mut wsum = vec![0f32; r];
        for &(row, _, w) in &batch.targets {
            wsum[slot(row)] += w / batch.denom;
        }
        let mut dl = vec![0f32; r * vocab];
        dl.par_chunks_mut(vocab).enumerate().for_each(|(s, g)| {
            let lg = &logits[s * vocab..(s + 1) * vocab];
            for (gv, lv) in g.iter_mut().zip(lg) {
                *gv = (lv - lse[s]).exp() * wsum[s];
            }
        });
        for &(row, tgt, w) in &batch.targets {
            dl[slot(row) * vocab + tgt as usize] -= w / batch.denom;
        }
        let mut dhf = vec![0f32; r * d];
        matmul_into(&dl, r, vocab, emb, d, &mut dhf, false);
        let mut dxs = vec![0f32; batch.rows() * n * d];
        for (s, &row) in uniq.iter().enumerate() {
            let mut dm = vec![0f32; d];
            zc_rms_norm_bwd(&hm[s * d..(s + 1) * d], Some(&self.w.final_norm), &dhf[s * d..(s + 1) * d], &mut dm, false);
            for lane in 0..n {
                for (o, v) in dxs[row * n * d + lane * d..row * n * d + (lane + 1) * d].iter_mut().zip(&dm) {
                    *o += v / n as f32;
                }
            }
        }
        (loss, dxs)
    }

    /// Loss and LoRA gradients (`jax.value_and_grad(loss_fn)`).
    pub fn step(&self, batch: &Batch, lora: &LoraParams) -> StepOut {
        let c = &self.w.config;
        let mut checkpoints = Vec::with_capacity(c.num_layers);
        let xs = span("forward", || self.forward_stream(batch, Some(&mut checkpoints)));
        let (loss, mut dxs) = span("head", || self.head(batch, &xs, true));
        drop(xs);
        let engram = self.engram_kv(batch);
        let mut grads = lora.zeros_like();
        for l in (0..c.num_layers).rev() {
            let mut xs = checkpoints.pop().unwrap();
            let x_in = xs.clone();
            let mut tape = Tape::default();
            span("recompute", || self.layer_fwd(l, batch, &mut xs, engram.as_ref(), Some(&mut tape)));
            drop(xs);
            dxs = span("layer_bwd", || self.layer_bwd(l, batch, &x_in, &tape, &dxs, engram.as_ref(), lora, &mut grads));
        }
        StepOut { loss, grads }
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_bwd(
        &self,
        l: usize,
        batch: &Batch,
        xs: &[f32],
        t: &Tape,
        dxo: &[f32],
        engram: Option<&EngramKv>,
        lora: &LoraParams,
        grads: &mut LoraParams,
    ) -> Vec<f32> {
        let c = &self.w.config;
        let lw = &self.w.layers[l];
        let (n, d) = (c.mhc_lanes, c.d_model);
        let nc = n * d;
        let nn = n * n;
        let ncols = 2 * n + nn;
        let rows = batch.rows();
        let lane = l % n;
        let states_per = (2 * SINKHORN_ITERS + 1) * nn;

        // mHC write reverse.
        let mut dxs = vec![0f32; rows * nc];
        let mut dy = vec![0f32; rows * d];
        let mut dp = vec![0f32; rows * ncols];
        span("mhc_write_bwd", || {
            dxs.par_chunks_mut(nc).zip(dy.par_chunks_mut(d)).zip(dp.par_chunks_mut(ncols)).enumerate().for_each(|(i, ((dx, dyi), dpi))| {
                let pi = &t.p[i * ncols..(i + 1) * ncols];
                let x = &xs[i * nc..(i + 1) * nc];
                let g = &dxo[i * nc..(i + 1) * nc];
                let yi = &t.y[i * d..(i + 1) * d];
                let st = &t.sk[i * states_per..(i + 1) * states_per];
                let h: Vec<f32> = st[2 * SINKHORN_ITERS * nn..].iter().map(|v| v.exp()).collect();
                let mut dh = vec![0f32; nn];
                for a in 0..n {
                    let ga = &g[a * d..(a + 1) * d];
                    for b in 0..n {
                        dh[a * n + b] = dot(ga, &x[b * d..(b + 1) * d]);
                        let hab = h[a * n + b];
                        for (o, v) in dx[b * d..(b + 1) * d].iter_mut().zip(ga) {
                            *o += hab * v;
                        }
                    }
                    let off = if a == lane { 0.0 } else { -4.0 };
                    let s = sigmoid(lw.a_post * pi[n + a] + lw.b_post[a] + off);
                    dpi[n + a] = dot(ga, yi) * 2.0 * s * (1.0 - s) * lw.a_post;
                    for (o, v) in dyi.iter_mut().zip(ga) {
                        *o += 2.0 * s * v;
                    }
                }
                let dr = sinkhorn_bwd(st, &dh, n);
                for kk in 0..nn {
                    dpi[2 * n + kk] = lw.a_res * dr[kk];
                }
            })
        });

        let du = span("block_bwd", || self.block_bwd(l, batch, t, &dy, engram, lora, grads));

        // mHC read reverse.
        span("mhc_read_bwd", || {
            dxs.par_chunks_mut(nc).zip(dp.par_chunks_mut(ncols)).enumerate().for_each(|(i, (dx, dpi))| {
                let pi = &t.p[i * ncols..(i + 1) * ncols];
                let x = &xs[i * nc..(i + 1) * nc];
                let dui = &du[i * d..(i + 1) * d];
                for j in 0..n {
                    let off = if j == lane { 4.0 } else { -4.0 };
                    let s = sigmoid(lw.a_pre * pi[j] + lw.b_pre[j] + off);
                    dpi[j] = dot(dui, &x[j * d..(j + 1) * d]) * s * (1.0 - s) * lw.a_pre;
                    for (o, v) in dx[j * d..(j + 1) * d].iter_mut().zip(dui) {
                        *o += s * v;
                    }
                }
            });
            let dnx = matmul(&dp, rows, ncols, &lw.phi, nc);
            dxs.par_chunks_mut(nc).enumerate().for_each(|(i, dx)| {
                zc_rms_norm_bwd(&xs[i * nc..(i + 1) * nc], None, &dnx[i * nc..(i + 1) * nc], dx, true);
            });
        });
        dxs
    }

    #[allow(clippy::too_many_arguments)]
    fn block_bwd(
        &self,
        l: usize,
        batch: &Batch,
        t: &Tape,
        dy: &[f32],
        engram: Option<&EngramKv>,
        lora: &LoraParams,
        grads: &mut LoraParams,
    ) -> Vec<f32> {
        let c = &self.w.config;
        let lw = &self.w.layers[l];
        let d = c.d_model;
        let rows = batch.rows();
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);

        // out = x2 + mlp(zc(x2)) - u.
        let dh2 = span("mlp_bwd", || mlp_bwd(lw, &self.w.perm1, &self.w.perm2, d, &t.mlp, dy));
        let mut dx2 = dy.to_vec();
        dx2.par_chunks_mut(d).enumerate().for_each(|(i, g)| {
            zc_rms_norm_bwd(&t.x2[i * d..(i + 1) * d], Some(&lw.pre_hada), &dh2[i * d..(i + 1) * d], g, true);
        });
        drop(dh2);
        // x2 = x1 + sigmoid(gate) * zc(o).
        let gate = sigmoid(lw.attn_gate);
        let mut dout = vec![0f32; rows * d];
        dout.par_chunks_mut(d).enumerate().for_each(|(i, g)| {
            let da: Vec<f32> = dx2[i * d..(i + 1) * d].iter().map(|v| gate * v).collect();
            zc_rms_norm_bwd(&t.o[i * d..(i + 1) * d], Some(&lw.post_norm), &da, g, false);
        });
        self.lora_grad(l, OUT, &t.oq, &dout, rows, lora, grads);
        let mut doq = vec![0f32; rows * nh * vd];
        matmul_into(&dout, rows, d, &lw.wout, nh * vd, &mut doq, false);
        drop(dout);
        let mut datt = vec![0f32; rows * nh * vd];
        let mut dgp = vec![0f32; rows * nh * vd];
        datt.par_iter_mut().zip(dgp.par_iter_mut()).enumerate().for_each(|(i, (da, dg))| {
            let s = sigmoid(t.gp[i]);
            *da = doq[i] * s;
            *dg = doq[i] * t.att[i] * s * (1.0 - s);
        });
        drop(doq);

        let (mut dqr, mut dkr, dv) = span("attn_core_bwd", || self.attn_core_bwd(l, batch, t, &datt));
        drop(datt);

        // Rope and per-head norm reverse.
        let half = qk / 2;
        let unrope_norm = |dr: &mut [f32], pre: &[f32], scale: &[f32], pos: usize| {
            let (cs, sn) = (&self.rope_cos[pos * half..(pos + 1) * half], &self.rope_sin[pos * half..(pos + 1) * half]);
            let mut dn = vec![0f32; qk];
            for i in 0..half {
                let (g1, g2) = (dr[i], dr[i + half]);
                dn[i] = g1 * cs[i] + g2 * sn[i];
                dn[i + half] = -g1 * sn[i] + g2 * cs[i];
            }
            zc_rms_norm_bwd(pre, Some(scale), &dn, dr, false);
        };
        dqr.par_chunks_mut(qd).enumerate().for_each(|(i, r)| {
            for (h, hr) in r.chunks_mut(qk).enumerate() {
                unrope_norm(hr, &t.q[i * qd + h * qk..i * qd + (h + 1) * qk], &lw.q_norm, batch.pos[i]);
            }
        });
        dkr.par_chunks_mut(kd).enumerate().for_each(|(i, r)| {
            for (h, hr) in r.chunks_mut(qk).enumerate() {
                unrope_norm(hr, &t.k[i * kd + h * qk..i * kd + (h + 1) * qk], &lw.k_norm, batch.pos[i]);
            }
        });

        let dqp = self.conv_bwd(batch, &dqr, &lw.q_taps, qd);
        let dkp = self.conv_bwd(batch, &dkr, &lw.k_taps, kd);
        let dvp = self.conv_bwd(batch, &dv, &lw.v_taps, vdim);
        drop((dqr, dkr, dv));

        span("lora_grad", || {
            self.lora_grad(l, Q, &t.hq, &dqp, rows, lora, grads);
            self.lora_grad(l, K, &t.hq, &dkp, rows, lora, grads);
            self.lora_grad(l, V, &t.hq, &dvp, rows, lora, grads);
            self.lora_grad(l, GATE, &t.hq, &dgp, rows, lora, grads);
        });
        let mut dh = vec![0f32; rows * d];
        matmul_into(&dqp, rows, qd, &lw.wq, d, &mut dh, false);
        matmul_into(&dkp, rows, kd, &lw.wk, d, &mut dh, true);
        matmul_into(&dvp, rows, vdim, &lw.wv, d, &mut dh, true);
        matmul_into(&dgp, rows, nh * vd, &lw.wgate, d, &mut dh, true);

        // h = zc(x1): dx1 = dx2 + zc'(dh); engram gate; then y = out - u.
        let mut du = dx2;
        du.par_chunks_mut(d).enumerate().for_each(|(i, g)| {
            zc_rms_norm_bwd(&t.x1[i * d..(i + 1) * d], Some(&lw.norm_in), &dh[i * d..(i + 1) * d], g, true);
        });
        if let (Some(site), Some(e)) = (self.site_of_layer[l], engram) {
            let (ek, ev) = (&e.k[site], &e.v[site]);
            let inv = 1.0 / (d as f64).sqrt() as f32;
            du.par_chunks_mut(d).enumerate().for_each(|(i, g)| {
                let al = t.alpha[i];
                let dz = dot(g, &ev[i * d..(i + 1) * d]) * al * (1.0 - al) * inv;
                let mut kh = vec![0f32; d];
                rms_unit(&ek[i * d..(i + 1) * d], &mut kh);
                let dhat: Vec<f32> = kh.iter().map(|v| v * dz).collect();
                zc_rms_norm_bwd(&t.u[i * d..(i + 1) * d], None, &dhat, g, true);
            });
        }
        du.par_iter_mut().zip(dy.par_iter()).for_each(|(a, b)| *a -= b);
        du
    }

    /// Reverse of the attention core: `(dq, dk, dv)` of the quantized,
    /// roped queries and keys and the values, summed over sequences.
    fn attn_core_bwd(&self, l: usize, batch: &Batch, t: &Tape, datt: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &self.w.config;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        let group = nh / nkv;
        let fused = qk.max(vd);
        let qscale = ((fused as f64) / (qk as f64)).sqrt() as f32;
        let sscale = (1.0 / (fused as f64).sqrt()) as f32;
        let rows = batch.rows();
        let tasks: Vec<(usize, usize)> = (0..batch.seqs.len()).flat_map(|s| (0..nkv).map(move |h| (s, h))).collect();
        type Part = (Vec<f32>, Vec<f32>, Vec<f32>);
        let parts: Vec<Part> = tasks
            .par_iter()
            .map(|&(s, kvh)| {
                let seq = &batch.seqs[s];
                let len = seq.rows.len();
                let q0 = seq.owned_from;
                let lq = len - q0;
                let owned = &seq.rows[q0..];
                let kh = gather(&t.kq, kd, &seq.rows, kvh * qk, qk);
                let vh = gather(&t.vq, vdim, &seq.rows, kvh * vd, vd);
                let mut dk = vec![0f32; len * qk];
                let mut dv = vec![0f32; len * vd];
                let mut dq = vec![0f32; lq * group * qk];
                for gi in 0..group {
                    let h = kvh * group + gi;
                    let mut qs = gather(&t.qq, qd, owned, h * qk, qk);
                    qs.iter_mut().for_each(|v| *v = (*v * qscale) * sscale);
                    let lse: Vec<f32> = owned.iter().map(|&r| t.lse[r * nh + h]).collect();
                    let (p, _) = self.attn_probs(l, seq, &qs, &kh, Some(&lse));
                    let go = gather(datt, nh * vd, owned, h * vd, vd);
                    // dV += P^T dO ; dP = dO V^T.
                    gemm_strided(len, vd, lq, 1.0, &p, len, true, &go, vd, false, 1.0, &mut dv, vd);
                    let mut ds = vec![0f32; lq * len];
                    gemm_strided(lq, len, vd, 1.0, &go, vd, false, &vh, vd, true, 0.0, &mut ds, len);
                    // dS = P * (dP - rowsum(P * dP)).
                    for (dsr, pr) in ds.chunks_mut(len).zip(p.chunks(len)) {
                        let sdot: f32 = dsr.iter().zip(pr).map(|(a, b)| a * b).sum();
                        for (a, b) in dsr.iter_mut().zip(pr) {
                            *a = *b * (*a - sdot);
                        }
                    }
                    // dq = dS K * (qscale*sscale); dK += dS^T (q * qscale*sscale).
                    let mut dqh = vec![0f32; lq * qk];
                    gemm_strided(lq, qk, len, qscale * sscale, &ds, len, false, &kh, qk, false, 0.0, &mut dqh, qk);
                    gemm_strided(len, qk, lq, 1.0, &ds, len, true, &qs, qk, false, 1.0, &mut dk, qk);
                    for i in 0..lq {
                        dq[(i * group + gi) * qk..(i * group + gi + 1) * qk].copy_from_slice(&dqh[i * qk..(i + 1) * qk]);
                    }
                }
                (dq, dk, dv)
            })
            .collect();
        let mut dq = vec![0f32; rows * qd];
        let mut dk = vec![0f32; rows * kd];
        let mut dv = vec![0f32; rows * vdim];
        for (&(s, kvh), (dqs, dks, dvs)) in tasks.iter().zip(parts) {
            let seq = &batch.seqs[s];
            for (i, &r) in seq.rows[seq.owned_from..].iter().enumerate() {
                for gi in 0..group {
                    let h = kvh * group + gi;
                    dq[r * qd + h * qk..r * qd + (h + 1) * qk].copy_from_slice(&dqs[(i * group + gi) * qk..(i * group + gi + 1) * qk]);
                }
            }
            for (j, &r) in seq.rows.iter().enumerate() {
                for (a, b) in dk[r * kd + kvh * qk..r * kd + (kvh + 1) * qk].iter_mut().zip(&dks[j * qk..(j + 1) * qk]) {
                    *a += b;
                }
                for (a, b) in dv[r * vdim + kvh * vd..r * vdim + (kvh + 1) * vd].iter_mut().zip(&dvs[j * vd..(j + 1) * vd]) {
                    *a += b;
                }
            }
        }
        (dq, dk, dv)
    }

    /// Accumulate `dA`, `dB` of target `tg` at layer `l` from the input `x`
    /// and output gradient `dy` of its matmul.
    #[allow(clippy::too_many_arguments)]
    fn lora_grad(&self, l: usize, tg: usize, x: &[f32], dy: &[f32], rows: usize, lora: &LoraParams, grads: &mut LoraParams) {
        let (din, dout) = target_dims(&self.w.config)[tg];
        let r = lora.rank;
        let a = &lora.a[tg][l * din * r..(l + 1) * din * r];
        let b = &lora.b[tg][l * r * dout..(l + 1) * r * dout];
        let s = self.scale;
        // t1 = dY B^T [rows, r]; dA += s X^T t1.
        let t1 = matmul_t(dy, rows, dout, b, r);
        let mut da = vec![0f32; din * r];
        matmul_tn_into(x, rows, din, &t1, r, &mut da, false);
        // t2 = X A [rows, r]; dB += s t2^T dY.
        let t2 = matmul(x, rows, din, a, r);
        let mut db = vec![0f32; r * dout];
        matmul_tn_into(&t2, rows, r, dy, dout, &mut db, false);
        for (g, v) in grads.a[tg][l * din * r..(l + 1) * din * r].iter_mut().zip(&da) {
            *g += s * v;
        }
        for (g, v) in grads.b[tg][l * r * dout..(l + 1) * r * dout].iter_mut().zip(&db) {
            *g += s * v;
        }
    }
}

struct AttFwd {
    q: Vec<f32>,
    k: Vec<f32>,
    qq: Vec<f32>,
    kq: Vec<f32>,
    vq: Vec<f32>,
    lse: Vec<f32>,
    att: Vec<f32>,
    gp: Vec<f32>,
    oq: Vec<f32>,
    o: Vec<f32>,
}
