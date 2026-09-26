//! The Needle 3 forward pass (Laddered Simple Attention Network).
//!
//! One code path serves prefill and decode: [`Model::forward`] advances a
//! [`Session`] by a chunk of tokens, reading and extending its caches (the
//! attention KV cache, the pre-tap conv histories, the engram value history).
//! Batch size is one; the batched training graph lives in `needle-train`.

use needle_core::config::{Config, ENGRAM_CONV_TAPS, ENGRAM_PRIME, ENGRAM_SEED, HADA_COND_RANK, hada_blocks};
use needle_core::quant::{ACT_BITS, fake_quant_rows};

use crate::linalg::{dot, matmul_t, rms_unit, sigmoid, silu, softmax_inplace, zc_rms_norm};
use crate::prof::span;
use crate::weights::{LayerWeights, QWeights, Weights};

/// Engram keys and tapped values per site, `[t * d]` each.
pub(crate) type EngramKv = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Deploy numerics (`quant=True` in the reference): A8 activations at every
/// matmul input, int8 per-head query and KV.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Numerics {
    pub quant: bool,
}

pub struct Model {
    pub w: Weights,
    /// Packed matmul weights (the CQ fast path); when present the f32
    /// matmul fields of `w` may be empty.
    pub q: Option<QWeights>,
    pub numerics: Numerics,
    pub(crate) rope_cos: Vec<f32>,
    pub(crate) rope_sin: Vec<f32>,
    rope_len: usize,
    /// Engram site index for each layer.
    pub(crate) site_of_layer: Vec<Option<usize>>,
    /// Tables of the native-exact packed path (present with packed weights).
    pub(crate) tables: Option<crate::infer::Tables>,
}

/// Per-layer attention cache.
#[derive(Clone, Default)]
pub(crate) struct LayerCache {
    /// Post-norm, post-rope keys `[pos, kv_heads * qk]`.
    pub(crate) k: Vec<f32>,
    /// Values `[pos, kv_heads * v]`.
    pub(crate) v: Vec<f32>,
    /// Pre-tap projections of the last `taps - 1` positions, newest last.
    pub(crate) q_hist: Vec<f32>,
    pub(crate) k_hist: Vec<f32>,
    pub(crate) v_hist: Vec<f32>,
}

/// Decoding state for one sequence.
#[derive(Clone)]
pub struct Session {
    pub tokens: Vec<u32>,
    pub(crate) layers: Vec<LayerCache>,
    /// Pre-tap engram values per site, the last `(taps - 1) * dilation` rows.
    pub(crate) engram_vhist: Vec<Vec<f32>>,
    /// Per-depth cells of every position, `[len, L+1, d]`, when tracked
    /// (the probe heads read them).
    pub cells: Option<Vec<f32>>,
    /// The native engine's attention span, `(sink, ring)`: every query sees
    /// the first `sink` positions (the static prefix) and the last `ring`,
    /// on top of its layer's band. `None` attends to everything.
    pub attend: Option<(usize, usize)>,
    /// Caches of the packed path (present with packed weights).
    pub nat: Option<crate::infer::NatState>,
}

impl Session {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Drop the last `n <= ROLLBACK` positions without recomputing.
    pub fn rollback(&mut self, model: &Model, n: usize) {
        assert!(n <= ROLLBACK, "rollback of {n} exceeds {ROLLBACK}");
        if n == 0 {
            return;
        }
        let c = model.config();
        let (qk, vd) = c.head_dims();
        let (qd, kd, vdim) = (c.num_heads * qk, c.num_kv_heads * qk, c.num_kv_heads * vd);
        let len = self.tokens.len() - n;
        self.tokens.truncate(len);
        if let Some(nat) = self.nat.as_mut() {
            nat.rollback(model, len, n);
        }
        let drop_rows = |v: &mut Vec<f32>, width: usize| {
            let rows = v.len() / width;
            v.truncate(rows.saturating_sub(n) * width);
        };
        for l in &mut self.layers {
            l.k.truncate(len * kd);
            l.v.truncate(len * vdim);
            drop_rows(&mut l.q_hist, qd);
            drop_rows(&mut l.k_hist, kd);
            drop_rows(&mut l.v_hist, vdim);
        }
        for h in &mut self.engram_vhist {
            drop_rows(h, c.d_model);
        }
        if let Some(cells) = self.cells.as_mut() {
            cells.truncate(len * (c.num_layers + 1) * c.d_model);
        }
    }

    /// Forget positions `len..` the way the engine's context-overflow
    /// re-feed does: the KV cache keeps what it holds, the conv history and
    /// RoPE state start over.
    pub fn rewind_zeroed(&mut self, model: &Model, len: usize) {
        self.tokens.truncate(len);
        if let Some(nat) = self.nat.as_mut() {
            nat.rewind_zeroed(model, len);
        }
        if let Some(cells) = self.cells.as_mut() {
            let c = model.config();
            cells.truncate(len * (c.num_layers + 1) * c.d_model);
        }
    }

    /// Drop everything past position `len` (rewinding a conversation).
    pub fn truncate(&mut self, model: &Model, len: usize) {
        if len >= self.tokens.len() {
            return;
        }
        if self.tokens.len() - len <= ROLLBACK {
            return self.rollback(model, self.tokens.len() - len);
        }
        // Conv histories cannot be rewound cheaply; recompute from scratch.
        let tokens = self.tokens[..len].to_vec();
        let tracked = self.cells.is_some();
        *self = model.session();
        if tracked {
            self.cells = Some(vec![]);
        }
        if !tokens.is_empty() {
            model.forward(self, &tokens, Outputs::None);
        }
    }
}

/// What a forward call returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outputs {
    None,
    /// Logits for the last position only.
    LastLogits,
    /// Logits for every new position.
    AllLogits,
    /// Final normed hidden state of every new position.
    Hidden,
    /// Final normed hidden state of the last position.
    LastHidden,
}

pub struct ForwardOut {
    /// `[rows, out_vocab]` or `[rows, d]` depending on [`Outputs`].
    pub data: Vec<f32>,
    pub rows: usize,
    /// Lane-mean stream after the embedding and after each layer,
    /// `[t, L+1, d]`, when cells were requested.
    pub cells: Option<Vec<f32>>,
}

pub(crate) fn shifted<'a>(hist: &'a [f32], cur: &'a [f32], width: usize, t: usize, back: usize) -> Option<&'a [f32]> {
    // Row `t - back` in the concatenation [hist rows | cur rows].
    let h = hist.len() / width;
    let idx = h + t;
    if back > idx {
        return None;
    }
    let r = idx - back;
    Some(if r < h { &hist[r * width..(r + 1) * width] } else { &cur[(r - h) * width..(r - h + 1) * width] })
}

/// Positions a session can drop cheaply (speculative drafts): the conv and
/// engram histories keep this many extra rows.
pub const ROLLBACK: usize = 32;

/// The keys a query at `pos` sees, as up to two ascending ranges: the
/// layer's band (`band` positions back), cut down in an engine session to
/// the sink and the last `ring` positions (`Session::attend`).
pub(crate) fn visible_keys(pos: usize, band: Option<usize>, attend: Option<(usize, usize)>) -> [std::ops::Range<usize>; 2] {
    let lo = band.map_or(0, |w| (pos + 1).saturating_sub(w));
    let Some((sink, ring)) = attend else { return [lo..pos + 1, 0..0] };
    let recent = (pos + 1).saturating_sub(ring).max(lo);
    let sink_end = sink.min(pos + 1);
    if lo >= sink_end || recent <= sink_end {
        // One stretch: the sink is outside the band, or meets the recent keys.
        return [lo.max(if lo >= sink_end { recent } else { lo })..pos + 1, 0..0];
    }
    [lo..sink_end, recent..pos + 1]
}

/// Virtual keys `v` (numbering the keys of `ranges` in order) as physical
/// ranges: at most one piece per range.
pub(crate) fn slice_ranges(ranges: &[std::ops::Range<usize>; 2], v: std::ops::Range<usize>) -> [std::ops::Range<usize>; 2] {
    let n0 = ranges[0].len();
    let a = v.start.min(n0)..v.end.min(n0);
    let b = v.start.max(n0) - n0..v.end.max(n0) - n0;
    [ranges[0].start + a.start..ranges[0].start + a.end, ranges[1].start + b.start..ranges[1].start + b.end]
}

pub(crate) fn keep_tail(hist: &mut Vec<f32>, cur: &[f32], width: usize, rows: usize) {
    hist.extend_from_slice(cur);
    // Readers only look at the newest rows, so trim lazily: shifting the
    // history every token costs more than the memory it saves.
    let n = hist.len() / width;
    if n > 2 * rows {
        hist.drain(..(n - rows) * width);
    }
}

impl Model {
    pub fn new(w: Weights) -> Self {
        crate::cpu::init();
        let c = &w.config;
        let mut site_of_layer = vec![None; c.num_layers];
        for (s, &l) in c.engram_layers.iter().enumerate() {
            site_of_layer[l] = Some(s);
        }
        let mut m = Self {
            w,
            q: None,
            numerics: Numerics::default(),
            rope_cos: vec![],
            rope_sin: vec![],
            rope_len: 0,
            site_of_layer,
            tables: None,
        };
        let max = m.w.config.max_seq_len;
        m.ensure_rope(max);
        m
    }

    pub fn config(&self) -> &Config {
        &self.w.config
    }

    /// A model running on packed weights.
    pub fn quantized(w: Weights, q: QWeights) -> Self {
        let mut m = Self::new(w);
        m.q = Some(q);
        m.tables = Some(crate::infer::Tables::new(&m));
        m
    }

    /// Row `tok` of the token embedding.
    pub(crate) fn embedding_row(&self, tok: u32) -> Vec<f32> {
        let d = self.w.config.d_model;
        match &self.q {
            Some(q) => q.embedding.row(tok as usize),
            None => self.w.embedding[tok as usize * d..(tok as usize + 1) * d].to_vec(),
        }
    }

    pub fn session(&self) -> Session {
        let c = self.config();
        Session {
            tokens: vec![],
            layers: vec![LayerCache::default(); c.num_layers],
            engram_vhist: vec![vec![]; c.engram_layers.len()],
            cells: None,
            attend: None,
            nat: self.tables.is_some().then(|| crate::infer::NatState::new(self)),
        }
    }

    /// `precompute_rope_freqs` up to `len` positions (f32, like the reference).
    fn ensure_rope(&mut self, len: usize) {
        if len <= self.rope_len {
            return;
        }
        let (qk, _) = self.w.config.head_dims();
        let half = qk / 2;
        let theta = self.w.config.rope_theta as f32;
        let freqs: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / qk as f32)).collect();
        self.rope_cos = Vec::with_capacity(len * half);
        self.rope_sin = Vec::with_capacity(len * half);
        for t in 0..len {
            for &f in &freqs {
                let a = t as f32 * f;
                self.rope_cos.push(a.cos());
                self.rope_sin.push(a.sin());
            }
        }
        self.rope_len = len;
    }

    fn aq(&self, x: &mut [f32], width: usize) {
        if self.numerics.quant {
            fake_quant_rows(x, width, ACT_BITS);
        }
    }

    /// Advance `s` by `new` tokens.
    pub fn forward(&self, s: &mut Session, new: &[u32], out: Outputs) -> ForwardOut {
        self.forward_impl(s, new, out, false)
    }

    /// As [`Model::forward`], fed the way the native engine prefills: in
    /// chunks of 32 tokens (`FUN_6c88`), each seeing only the keys its
    /// session's attention span allows. Returns the last position's logits
    /// (empty for [`Outputs::None`]).
    pub fn prefill(&self, s: &mut Session, new: &[u32], out: Outputs) -> Vec<f32> {
        if self.native() {
            return self.prefill_native(s, new, out);
        }
        let mut last = vec![];
        let chunks: Vec<&[u32]> = new.chunks(32).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let want = if i + 1 == chunks.len() { out } else { Outputs::None };
            last = self.forward(s, chunk, want).data;
        }
        last
    }

    /// Whether this model runs the native-exact packed path.
    pub fn native(&self) -> bool {
        self.q.is_some() && !self.numerics.quant && self.tables.as_ref().is_some_and(|t| t.exact)
    }

    /// As [`Model::forward`] and also collect the per-layer cells the probe
    /// heads read.
    pub fn forward_cells(&self, s: &mut Session, new: &[u32], out: Outputs) -> ForwardOut {
        self.forward_impl(s, new, out, true)
    }

    fn forward_impl(&self, s: &mut Session, new: &[u32], outputs: Outputs, want_cells: bool) -> ForwardOut {
        // Run on a pool worker so the many small parallel sections inside
        // start from a worker (cheap) instead of injecting from outside.
        self.forward_inner(s, new, outputs, want_cells)
    }

    fn forward_inner(&self, s: &mut Session, new: &[u32], outputs: Outputs, want_cells: bool) -> ForwardOut {
        let want_cells = want_cells || s.cells.is_some();
        if self.native() {
            return self.forward_native(s, new, outputs, want_cells);
        }
        let c = self.config();
        let t = new.len();
        let d = c.d_model;
        let p0 = s.tokens.len();
        assert!(p0 + t <= self.rope_len, "sequence of {} exceeds max_seq_len {}", p0 + t, self.rope_len);
        s.tokens.extend_from_slice(new);

        // Embedding.
        let scale = (d as f64).sqrt() as f32;
        let mut x0 = vec![0f32; t * d];
        for (i, &tok) in new.iter().enumerate() {
            let row = self.embedding_row(tok);
            for (o, v) in x0[i * d..(i + 1) * d].iter_mut().zip(&row) {
                *o = v * scale;
            }
        }

        let engram_kv = span("engram", || self.engram_kv(s, p0, t));

        let n = c.mhc_lanes;
        let l_total = c.num_layers;
        // Lane stream [t, n, d].
        let mut xs = vec![0f32; t * n * d];
        for i in 0..t {
            for lane in 0..n {
                xs[(i * n + lane) * d..(i * n + lane + 1) * d].copy_from_slice(&x0[i * d..(i + 1) * d]);
            }
        }
        let mut cells = want_cells.then(|| {
            let mut cells = vec![0f32; t * (l_total + 1) * d];
            for i in 0..t {
                cells[i * (l_total + 1) * d..i * (l_total + 1) * d + d].copy_from_slice(&x0[i * d..(i + 1) * d]);
            }
            cells
        });

        for l in 0..l_total {
            span("layer", || self.layer(l, s, p0, &mut xs, t, engram_kv.as_ref()));
            if let Some(cells) = cells.as_mut() {
                for i in 0..t {
                    let dst = &mut cells[(i * (l_total + 1) + l + 1) * d..(i * (l_total + 1) + l + 2) * d];
                    lane_mean(&xs[i * n * d..(i + 1) * n * d], n, d, dst);
                }
            }
        }

        // Lane mean, final norm, tied head.
        let rows: Vec<usize> = match outputs {
            Outputs::None => vec![],
            Outputs::LastLogits => vec![t - 1],
            Outputs::AllLogits | Outputs::Hidden => (0..t).collect(),
            Outputs::LastHidden => vec![t - 1],
        };
        let mut h = vec![0f32; rows.len() * d];
        for (r, &i) in rows.iter().enumerate() {
            let dst = &mut h[r * d..(r + 1) * d];
            lane_mean(&xs[i * n * d..(i + 1) * n * d], n, d, dst);
            zc_rms_norm(dst, &self.w.final_norm);
        }
        let data = match outputs {
            Outputs::None => vec![],
            Outputs::Hidden | Outputs::LastHidden => h,
            _ => {
                self.aq(&mut h, d);
                span("logits", || match &self.q {
                    Some(q) => {
                        let act = crate::qlinear::QAct::new(&h, rows.len(), d, q.embedding.act_bits());
                        let full = q.embedding.matmul(&act);
                        let (v, o) = (q.embedding.out, c.out_rows());
                        if v == o { full } else { (0..rows.len()).flat_map(|r| full[r * v..r * v + o].to_vec()).collect() }
                    }
                    None => matmul_t(&h, rows.len(), d, &self.w.embedding[..c.out_rows() * d], c.out_rows()),
                })
            }
        };
        if let (Some(store), Some(c)) = (s.cells.as_mut(), cells.as_ref()) {
            store.extend_from_slice(c);
        }
        ForwardOut { data, rows: rows.len(), cells }
    }

    /// Engram keys and (tapped) values for the new positions: `[sites][t * d]`.
    pub(crate) fn engram_kv(&self, s: &mut Session, p0: usize, t: usize) -> Option<EngramKv> {
        let c = self.config();
        if c.engram_layers.is_empty() {
            return None;
        }
        let d = c.d_model;
        let (orders, heads, sub) = c.engram_geometry();
        let tables = orders.len() * heads;
        let stride = if c.engram_seed_heads > 0 { c.engram_seed_heads } else { heads };
        let dil = c.engram_dilation();
        let slots = c.engram_slots as u32;
        // Hashed n-gram indices and validity for each new position.
        let mut idx = vec![0usize; t * tables];
        let mut ok = vec![0f32; t * tables];
        for i in 0..t {
            let pos = p0 + i;
            for (oi, &order) in orders.iter().enumerate() {
                for h in 0..heads {
                    let seed = ENGRAM_SEED.wrapping_mul((oi * stride + h + 1) as u32);
                    let mut acc = seed;
                    for j in 0..order {
                        let tok = if pos >= j { s.tokens[pos - j] } else { 0 };
                        acc = (acc ^ tok).wrapping_mul(ENGRAM_PRIME);
                    }
                    acc ^= acc >> 15;
                    let tbl = oi * heads + h;
                    idx[i * tables + tbl] = (acc % slots) as usize;
                    ok[i * tables + tbl] = if pos + 1 >= order { 1.0 } else { 0.0 };
                }
            }
        }
        let mut ks = Vec::with_capacity(c.engram_layers.len());
        let mut vs = Vec::with_capacity(c.engram_layers.len());
        for (site, ew) in self.w.engrams.iter().enumerate() {
            let mut e = vec![0f32; t * d];
            for i in 0..t {
                for tbl in 0..tables {
                    let okv = ok[i * tables + tbl];
                    if okv == 0.0 {
                        continue;
                    }
                    let slot = tbl * c.engram_slots + idx[i * tables + tbl];
                    let dst = &mut e[i * d + tbl * sub..i * d + (tbl + 1) * sub];
                    match &self.q {
                        Some(q) => dst.copy_from_slice(&q.engram_tables[site].row(slot)),
                        None => dst.copy_from_slice(&ew.tables[slot * sub..(slot + 1) * sub]),
                    }
                }
            }
            self.aq(&mut e, d);
            let (k, vpre) = match &self.q {
                Some(q) => {
                    let kv = q.engram_kv[site].apply(&e, t);
                    let mut k = vec![0f32; t * d];
                    let mut v = vec![0f32; t * d];
                    for i in 0..t {
                        k[i * d..(i + 1) * d].copy_from_slice(&kv[i * 2 * d..i * 2 * d + d]);
                        v[i * d..(i + 1) * d].copy_from_slice(&kv[i * 2 * d + d..(i + 1) * 2 * d]);
                    }
                    (k, v)
                }
                None => (matmul_t(&e, t, d, &ew.wk, d), matmul_t(&e, t, d, &ew.wv, d)),
            };
            let hist = &mut s.engram_vhist[site];
            let mut v = vec![0f32; t * d];
            for i in 0..t {
                let dst = &mut v[i * d..(i + 1) * d];
                for j in 0..ENGRAM_CONV_TAPS {
                    let tap = &ew.taps[j * d..(j + 1) * d];
                    if let Some(src) = shifted(hist, &vpre, d, i, j * dil) {
                        for ((o, a), b) in dst.iter_mut().zip(tap).zip(src) {
                            *o += a * b;
                        }
                    }
                }
            }
            keep_tail(hist, &vpre, d, (ENGRAM_CONV_TAPS - 1) * dil + ROLLBACK);
            ks.push(k);
            vs.push(v);
        }
        Some((ks, vs))
    }

    fn layer(&self, l: usize, s: &mut Session, p0: usize, xs: &mut [f32], t: usize, engram: Option<&EngramKv>) {
        let c = self.config();
        let lw = &self.w.layers[l];
        let (n, d) = (c.mhc_lanes, c.d_model);
        let nc = n * d;
        let ncols = 2 * n + n * n;

        // mHC read: normalise the whole lane stream, project to the pre/post/res gates.
        let mut nx = vec![0f32; t * nc];
        each_row(&mut nx, nc, |i, o| rms_unit(&xs[i * nc..(i + 1) * nc], o));
        self.aq(&mut nx, nc);
        let phi: &[f32] = match &self.q {
            Some(q) => &q.phi_f32[l],
            None => &lw.phi,
        };
        let proj = span("phi_proj", || {
            if t == 1 {
                let mut p = vec![0f32; ncols];
                let pp = crate::team::SyncPtr(p.as_mut_ptr());
                crate::team::team().run(&|tid, nt| {
                    for r in crate::team::share(ncols, tid, nt) {
                        // SAFETY: members write disjoint outputs.
                        unsafe { *pp.ptr().add(r) = dot(&nx, &phi[r * nc..(r + 1) * nc]) };
                    }
                });
                p
            } else {
                matmul_t(&nx, t, nc, phi, ncols)
            }
        });
        let lane = l % n;
        let mut u = vec![0f32; t * d];
        let mut hpost = vec![0f32; t * n];
        let mut hres = vec![0f32; t * n * n];
        // Per row: [u (d) | hpost (n) | hres (n*n)].
        let width = d + n + n * n;
        let mut packed = vec![0f32; t * width];
        each_row(&mut packed, width, |i, row| {
            let p = &proj[i * ncols..(i + 1) * ncols];
            let x = &xs[i * nc..(i + 1) * nc];
            let (ui, rest) = row.split_at_mut(d);
            let (hp_out, hr) = rest.split_at_mut(n);
            for j in 0..n {
                let pre_off = if j == lane { 4.0 } else { -4.0 };
                let post_off = if j == lane { 0.0 } else { -4.0 };
                let hp = sigmoid(lw.a_pre * p[j] + lw.b_pre[j] + pre_off);
                for (o, v) in ui.iter_mut().zip(&x[j * d..(j + 1) * d]) {
                    *o += hp * v;
                }
                hp_out[j] = 2.0 * sigmoid(lw.a_post * p[n + j] + lw.b_post[j] + post_off);
            }
            for k in 0..n * n {
                hr[k] = lw.a_res * p[2 * n + k] + lw.b_res[k];
            }
            sinkhorn(hr, n, 20);
        });
        for i in 0..t {
            let row = &packed[i * width..(i + 1) * width];
            u[i * d..(i + 1) * d].copy_from_slice(&row[..d]);
            hpost[i * n..(i + 1) * n].copy_from_slice(&row[d..d + n]);
            hres[i * n * n..(i + 1) * n * n].copy_from_slice(&row[d + n..]);
        }

        let y = span("block", || self.block(l, lw, s, p0, &u, t, engram));

        // mHC write: mix lanes with the doubly-stochastic residual, add the block delta.
        span("mhc_write", || {
            each_row(xs, nc, |i, x| {
                let hr = &hres[i * n * n..(i + 1) * n * n];
                let yi = &y[i * d..(i + 1) * d];
                let mut new = vec![0f32; nc];
                for a in 0..n {
                    let dst = &mut new[a * d..(a + 1) * d];
                    let hp = hpost[i * n + a];
                    for (o, v) in dst.iter_mut().zip(yi) {
                        *o = hp * v;
                    }
                    for b in 0..n {
                        let w = hr[a * n + b];
                        for (o, v) in dst.iter_mut().zip(&x[b * d..(b + 1) * d]) {
                            *o += w * v;
                        }
                    }
                }
                x.copy_from_slice(&new);
            })
        });
    }

    /// One block applied to the mHC read `u`; returns `block(u) - u`.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, l: usize, lw: &LayerWeights, s: &mut Session, p0: usize, u: &[f32], t: usize, engram: Option<&EngramKv>) -> Vec<f32> {
        let c = self.config();
        let d = c.d_model;
        let mut x = u.to_vec();
        if let (Some(site), Some((ks, vs))) = (self.site_of_layer[l], engram) {
            let (ek, ev) = (&ks[site], &vs[site]);
            let inv = 1.0 / (d as f64).sqrt() as f32;
            each_row(&mut x, d, |i, xi| {
                let mut a = vec![0f32; d];
                let mut b = vec![0f32; d];
                rms_unit(xi, &mut a);
                rms_unit(&ek[i * d..(i + 1) * d], &mut b);
                let alpha = sigmoid(dot(&a, &b) * inv);
                for (o, v) in xi.iter_mut().zip(&ev[i * d..(i + 1) * d]) {
                    *o += alpha * v;
                }
            });
        }
        let skip = x.clone();
        let mut h = x;
        each_row(&mut h, d, |_, r| zc_rms_norm(r, &lw.norm_in));
        let attn = span("attention", || self.attention(l, lw, s, p0, &mut h, t));
        let mut x = skip;
        let gate = sigmoid(lw.attn_gate);
        for (o, a) in x.iter_mut().zip(&attn) {
            *o += gate * a;
        }
        let mut h2 = x.clone();
        each_row(&mut h2, d, |_, r| zc_rms_norm(r, &lw.pre_hada));
        let mlp = span("mlp", || self.hadamard_mlp(lw, &h2, t));
        for (o, m) in x.iter_mut().zip(&mlp) {
            *o += m;
        }
        for (o, v) in x.iter_mut().zip(u) {
            *o -= v;
        }
        x
    }

    fn attention(&self, l: usize, lw: &LayerWeights, s: &mut Session, p0: usize, h: &mut [f32], t: usize) -> Vec<f32> {
        let c = self.config();
        let d = c.d_model;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        self.aq(h, d);
        let (qpre, kpre, vpre, gate) = span("qkvg_proj", || match &self.q {
            Some(q) => {
                let all = q.qkvg[l].apply(h, t);
                let w = qd + kd + vdim + nh * vd;
                let col = |a: usize, b: usize| -> Vec<f32> {
                    let mut v = Vec::with_capacity(t * (b - a));
                    for i in 0..t {
                        v.extend_from_slice(&all[i * w + a..i * w + b]);
                    }
                    v
                };
                (col(0, qd), col(qd, qd + kd), col(qd + kd, qd + kd + vdim), col(qd + kd + vdim, w))
            }
            None => (
                matmul_t(h, t, d, &lw.wq, qd),
                matmul_t(h, t, d, &lw.wk, kd),
                matmul_t(h, t, d, &lw.wv, vdim),
                matmul_t(h, t, d, &lw.wgate, nh * vd),
            ),
        });

        let cache = &mut s.layers[l];
        let taps = c.qkv_conv_taps;
        let conv = |pre: &[f32], hist: &[f32], w: &[f32], width: usize| -> Vec<f32> {
            if taps == 0 {
                return pre.to_vec();
            }
            let mut out = vec![0f32; t * width];
            for i in 0..t {
                let dst = &mut out[i * width..(i + 1) * width];
                for j in 0..taps {
                    if let Some(src) = shifted(hist, pre, width, i, j) {
                        for ((o, a), b) in dst.iter_mut().zip(&w[j * width..(j + 1) * width]).zip(src) {
                            *o += a * b;
                        }
                    }
                }
            }
            out
        };
        let mut q = conv(&qpre, &cache.q_hist, &lw.q_taps, qd);
        let mut k = conv(&kpre, &cache.k_hist, &lw.k_taps, kd);
        let mut v = conv(&vpre, &cache.v_hist, &lw.v_taps, vdim);
        if taps > 1 {
            keep_tail(&mut cache.q_hist, &qpre, qd, taps - 1 + ROLLBACK);
            keep_tail(&mut cache.k_hist, &kpre, kd, taps - 1 + ROLLBACK);
            keep_tail(&mut cache.v_hist, &vpre, vdim, taps - 1 + ROLLBACK);
        }

        // Per-head QK norm and rotary embedding at absolute positions.
        let half = qk / 2;
        let rope = |row: &mut [f32], pos: usize| {
            let (cs, sn) = (&self.rope_cos[pos * half..(pos + 1) * half], &self.rope_sin[pos * half..(pos + 1) * half]);
            for i in 0..half {
                let (x1, x2) = (row[i], row[i + half]);
                row[i] = x1 * cs[i] - x2 * sn[i];
                row[i + half] = x2 * cs[i] + x1 * sn[i];
            }
        };
        for i in 0..t {
            for hq in q[i * qd..(i + 1) * qd].chunks_mut(qk) {
                zc_rms_norm(hq, &lw.q_norm);
                rope(hq, p0 + i);
            }
            for hk in k[i * kd..(i + 1) * kd].chunks_mut(qk) {
                zc_rms_norm(hk, &lw.k_norm);
                rope(hk, p0 + i);
            }
        }
        if self.numerics.quant {
            fake_quant_rows(&mut q, qk, 8);
            fake_quant_rows(&mut k, qk, 8);
            fake_quant_rows(&mut v, vd, 8);
        }
        cache.k.extend_from_slice(&k);
        cache.v.extend_from_slice(&v);
        let (kc, vc) = (&cache.k, &cache.v);

        // Scaled dot-product attention, grouped queries, causal (+ band).
        let window = c.layer_window(l);
        let attend = s.attend;
        let fused = qk.max(vd);
        let qscale = ((fused as f64) / (qk as f64)).sqrt() as f32;
        let sscale = (1.0 / (fused as f64).sqrt()) as f32;
        let group = nh / nkv;
        let mut out = vec![0f32; t * nh * vd];
        if t == 1 {
            span("attn_heads", || self.decode_attention(&q, kc, vc, p0, window, s.attend, &mut out));
        } else if t <= 32 {
            // Short chunks (a conversation turn): per (row, KV head), the
            // decode attention kernel over that row's visible keys; small
            // GEMMs would queue on the shared matrix units.
            let tasks = t * nkv;
            let stride = vd + 2;
            let mut parts = vec![0f32; tasks * group * stride];
            let pp = crate::team::SyncPtr(parts.as_mut_ptr());
            let scale = qscale * sscale;
            crate::team::team().run(&|tid, n| {
                for task in crate::team::share(tasks, tid, n) {
                    let (i, kvh) = (task / nkv, task % nkv);
                    let ranges = visible_keys(p0 + i, window, attend);
                    // SAFETY: each task owns its block of `parts`.
                    let dst = unsafe { pp.slice(task * group * stride..(task + 1) * group * stride) };
                    let qrow = &q[i * qd + kvh * group * qk..i * qd + (kvh + 1) * group * qk];
                    crate::attn::attend_ranges(qrow, kc, vc, kd, vdim, kvh, &ranges, scale, group, qk, vd, dst, stride);
                }
            });
            for task in 0..tasks {
                let (i, kvh) = (task / nkv, task % nkv);
                for g in 0..group {
                    let b = &parts[(task * group + g) * stride..(task * group + g + 1) * stride];
                    let hh = kvh * group + g;
                    let inv = 1.0 / b[1];
                    for (o, v) in out[i * nh * vd + hh * vd..i * nh * vd + (hh + 1) * vd].iter_mut().zip(&b[2..]) {
                        *o = v * inv;
                    }
                }
            }
        } else {
            // Prefill: per head, S = Q K^T and O = P V as strided GEMMs over
            // the cache, with the causal/window mask applied between.
            let keys = p0 + t;
            let mut parts = vec![0f32; nh * t * vd];
            let pp = crate::team::SyncPtr(parts.as_mut_ptr());
            crate::team::team().run(&|tid, n| {
                for hh in crate::team::share(nh, tid, n) {
                    let kvh = hh / group;
                    let mut sc = vec![0f32; t * keys];
                    crate::linalg::gemm_strided(
                        t,
                        keys,
                        qk,
                        qscale * sscale,
                        &q[hh * qk..],
                        qd,
                        false,
                        &kc[kvh * qk..],
                        kd,
                        true,
                        0.0,
                        &mut sc,
                        keys,
                    );
                    for (i, row) in sc.chunks_mut(keys).enumerate() {
                        let [a, b] = visible_keys(p0 + i, window, attend);
                        row[..a.start].fill(f32::NEG_INFINITY);
                        row[a.end..b.start.max(a.end)].fill(f32::NEG_INFINITY);
                        row[b.end.max(a.end)..].fill(f32::NEG_INFINITY);
                        softmax_inplace(row);
                    }
                    // SAFETY: each member writes only its own heads' block.
                    let o = unsafe { pp.slice(hh * t * vd..(hh + 1) * t * vd) };
                    crate::linalg::gemm_strided(t, vd, keys, 1.0, &sc, keys, false, &vc[kvh * vd..], vdim, false, 0.0, o, vd);
                }
            });
            let parts: Vec<&[f32]> = parts.chunks(t * vd).collect();
            for (hh, o) in parts.iter().enumerate() {
                for i in 0..t {
                    out[i * nh * vd + hh * vd..i * nh * vd + (hh + 1) * vd].copy_from_slice(&o[i * vd..(i + 1) * vd]);
                }
            }
        }
        for (o, g) in out.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
        self.aq(&mut out, nh * vd);
        let mut o = span("out_proj", || match &self.q {
            Some(q) => q.out[l].apply(&out, t),
            None => matmul_t(&out, t, nh * vd, &lw.wout, d),
        });
        each_row(&mut o, d, |_, r| zc_rms_norm(r, &lw.post_norm));
        o
    }

    fn hadamard_mlp(&self, lw: &LayerWeights, x: &[f32], t: usize) -> Vec<f32> {
        let c = self.config();
        let d = c.d_model;
        let n = c.hada_n();
        let (ba, bb) = hada_blocks(n);
        if t == 1 && ba == 32 && bb == 32 {
            return self.hadamard_mlp_row(lw, x);
        }

        let r = HADA_COND_RANK;
        let mut out = vec![0f32; t * d];
        each_row(&mut out, d, |i, o| {
            let mut z = vec![0f32; n];
            let mut tmp = vec![0f32; n];
            let xi = &x[i * d..(i + 1) * d];
            // cond = 1 + softmax(x cond_v) cond_u
            let mut sm = [0f32; HADA_COND_RANK];
            for (j, xv) in xi.iter().enumerate() {
                let row = &lw.cond_v[j * r..(j + 1) * r];
                for k in 0..r {
                    sm[k] += xv * row[k];
                }
            }
            softmax_inplace(&mut sm);
            let mut cond = vec![0f32; n];
            for k in 0..r {
                let w = sm[k];
                for (c, u) in cond.iter_mut().zip(&lw.cond_u[k * n..(k + 1) * n]) {
                    *c += w * u;
                }
            }
            for j in 0..d {
                z[j] = lw.d1[j] * xi[j];
            }
            kron_apply(&z, &lw.kron[0], ba, &lw.kron[1], bb, &mut tmp);
            for j in 0..n {
                z[j] = tmp[self.w.perm1[j]];
            }
            for j in 0..n {
                z[j] = silu(lw.d2[j] * (1.0 + cond[j]) * z[j] + lw.b2[j]);
            }
            kron_apply(&z, &lw.kron[2], ba, &lw.kron[3], bb, &mut tmp);
            for j in 0..n {
                z[j] = lw.d3[j] * tmp[self.w.perm2[j]];
            }
            kron_apply(&z, &lw.kron[4], ba, &lw.kron[5], bb, &mut tmp);
            for j in 0..d {
                o[j] = lw.d4[j] * tmp[j];
            }
        });
        out
    }
}

impl Model {
    /// Single-token attention, flash-decoding style: each team member takes
    /// one KV head and a slice of the visible keys, reads every key and
    /// value row once for all the query heads that share it, and keeps a
    /// running max and sum per head; the partial results merge exactly.
    fn decode_attention(
        &self,
        q: &[f32],
        kc: &[f32],
        vc: &[f32],
        pos: usize,
        window: Option<usize>,
        attend: Option<(usize, usize)>,
        out: &mut [f32],
    ) {
        use crate::linalg::fast_exp;
        use crate::team::{SyncPtr, share, team};
        let c = self.config();
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (kd, vdim) = (nkv * qk, nkv * vd);
        let group = nh / nkv;
        let fused = qk.max(vd);
        let scale = (((fused as f64) / (qk as f64)).sqrt() as f32) * ((1.0 / (fused as f64).sqrt()) as f32);
        let ranges = visible_keys(pos, window, attend);
        let keys = ranges[0].len() + ranges[1].len();
        let tm = team();
        // A fixed split (as in the fused kernel), so results do not depend
        // on the team size.
        let chunks = 3.min(keys.div_ceil(16).max(1));
        let tasks = nkv * chunks;
        // Per task and head: [max, sum, o (vd)].
        let stride = vd + 2;
        let mut parts = vec![0f32; tasks * group * stride];
        let pp = SyncPtr(parts.as_mut_ptr());
        let qs: Vec<f32> = q.iter().map(|v| v * scale).collect();
        tm.run(&|tid, n| {
            for task in share(tasks, tid, n) {
                let (kvh, ci) = (task / chunks, task % chunks);
                let pieces = slice_ranges(&ranges, share(keys, ci, chunks));
                let positions = || pieces[0].clone().chain(pieces[1].clone());
                let len = pieces[0].len() + pieces[1].len();
                // SAFETY: each task owns its block of `parts`.
                let dst = unsafe { pp.slice(task * group * stride..(task + 1) * group * stride) };
                if len == 0 {
                    for g in 0..group {
                        dst[g * stride] = f32::NEG_INFINITY;
                    }
                    continue;
                }
                let mut scores = vec![0f32; group * len];
                for (j, sp) in positions().enumerate() {
                    let krow = &kc[sp * kd + kvh * qk..sp * kd + (kvh + 1) * qk];
                    for g in 0..group {
                        let hh = kvh * group + g;
                        scores[g * len + j] = dot(&qs[hh * qk..(hh + 1) * qk], krow);
                    }
                }
                for g in 0..group {
                    let sc = &mut scores[g * len..(g + 1) * len];
                    let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut l = 0f32;
                    for v in sc.iter_mut() {
                        *v = fast_exp(*v - m);
                        l += *v;
                    }
                    dst[g * stride] = m;
                    dst[g * stride + 1] = l;
                }
                for (j, sp) in positions().enumerate() {
                    let vrow = &vc[sp * vdim + kvh * vd..sp * vdim + (kvh + 1) * vd];
                    for g in 0..group {
                        let w = scores[g * len + j];
                        let o = &mut dst[g * stride + 2..(g + 1) * stride];
                        for (a, b) in o.iter_mut().zip(vrow) {
                            *a += w * b;
                        }
                    }
                }
            }
        });
        for kvh in 0..nkv {
            for g in 0..group {
                let hh = kvh * group + g;
                let blocks: Vec<&[f32]> = (0..chunks)
                    .map(|ci| &parts[((kvh * chunks + ci) * group + g) * stride..((kvh * chunks + ci) * group + g + 1) * stride])
                    .collect();
                let m = blocks.iter().map(|b| b[0]).fold(f32::NEG_INFINITY, f32::max);
                let o = &mut out[hh * vd..(hh + 1) * vd];
                o.fill(0.0);
                let mut l = 0f32;
                for b in &blocks {
                    if b[0] == f32::NEG_INFINITY {
                        continue;
                    }
                    let w = fast_exp(b[0] - m);
                    l += w * b[1];
                    for (a, v) in o.iter_mut().zip(&b[2..]) {
                        *a += w * v;
                    }
                }
                let inv = 1.0 / l;
                o.iter_mut().for_each(|v| *v *= inv);
            }
        }
    }

    /// One row of the Hadamard MLP split across the decode team: each
    /// Kronecker stage is two passes over 32-row blocks, and the permutation
    /// and elementwise steps fold into the pass that follows them.
    fn hadamard_mlp_row(&self, lw: &LayerWeights, x: &[f32]) -> Vec<f32> {
        use crate::team::{SyncPtr, share, team};
        const N: usize = 32;
        let d = self.config().d_model;
        let n = N * N;
        let r = HADA_COND_RANK;
        let mut sm = [0f32; HADA_COND_RANK];
        for (j, xv) in x.iter().enumerate() {
            let row = &lw.cond_v[j * r..(j + 1) * r];
            for k in 0..r {
                sm[k] += xv * row[k];
            }
        }
        softmax_inplace(&mut sm);
        let mut z = vec![0f32; n];
        for j in 0..d {
            z[j] = lw.d1[j] * x[j];
        }
        let mut m = vec![0f32; n];
        let mut tmat = vec![0f32; n];
        let (zp, mp, tp) = (SyncPtr(z.as_mut_ptr()), SyncPtr(m.as_mut_ptr()), SyncPtr(tmat.as_mut_ptr()));
        let (p1, p2) = (&self.w.perm1, &self.w.perm2);
        // M[i,:] = sum_j Z[i,j] B[j,:] for this member's rows i (pairs share loads).
        let zb = |rows: std::ops::Range<usize>, b: &[f32]| {
            // SAFETY: rows of Z are read-only here; rows of M are this member's.
            let zz = unsafe { std::slice::from_raw_parts(zp.ptr(), n) };
            let mut i = rows.start;
            while i < rows.end {
                if i + 1 < rows.end {
                    let (r0, r1) = mac2_32(|j| zz[i * N + j], |j| zz[(i + 1) * N + j], b);
                    unsafe { mp.slice(i * N..(i + 1) * N) }.copy_from_slice(&r0);
                    unsafe { mp.slice((i + 1) * N..(i + 2) * N) }.copy_from_slice(&r1);
                    i += 2;
                } else {
                    let (r0, _) = mac2_32(|j| zz[i * N + j], |_| 0.0, b);
                    unsafe { mp.slice(i * N..(i + 1) * N) }.copy_from_slice(&r0);
                    i += 1;
                }
            }
        };
        // T[k,:] = sum_i A[i,k] M[i,:] for this member's rows k.
        let atm = |rows: std::ops::Range<usize>, a: &[f32]| {
            // SAFETY: M is complete (previous pass) and read-only here.
            let mm = unsafe { std::slice::from_raw_parts(mp.ptr(), n) };
            let mut k = rows.start;
            while k < rows.end {
                if k + 1 < rows.end {
                    let (r0, r1) = mac2_32(|i| a[i * N + k], |i| a[i * N + k + 1], mm);
                    unsafe { tp.slice(k * N..(k + 1) * N) }.copy_from_slice(&r0);
                    unsafe { tp.slice((k + 1) * N..(k + 2) * N) }.copy_from_slice(&r1);
                    k += 2;
                } else {
                    let (r0, _) = mac2_32(|i| a[i * N + k], |_| 0.0, mm);
                    unsafe { tp.slice(k * N..(k + 1) * N) }.copy_from_slice(&r0);
                    k += 1;
                }
            }
        };
        let tm = team();
        tm.run(&|tid, nt| zb(share(N, tid, nt), &lw.kron[1]));
        tm.run(&|tid, nt| atm(share(N, tid, nt), &lw.kron[0]));
        tm.run(&|tid, nt| {
            let rows = share(N, tid, nt);
            // SAFETY: T is complete; this member rewrites only its rows of Z,
            // and the Z B pass that follows reads only those rows.
            let tt = unsafe { std::slice::from_raw_parts(tp.ptr(), n) };
            let zr = unsafe { zp.slice(rows.start * N..rows.end * N) };
            for (o, j) in zr.iter_mut().zip(rows.start * N..rows.end * N) {
                let mut cond = 1.0;
                for k in 0..r {
                    cond += sm[k] * lw.cond_u[k * n + j];
                }
                *o = silu(lw.d2[j] * cond * tt[p1[j]] + lw.b2[j]);
            }
            zb(rows, &lw.kron[3]);
        });
        tm.run(&|tid, nt| atm(share(N, tid, nt), &lw.kron[2]));
        tm.run(&|tid, nt| {
            let rows = share(N, tid, nt);
            let tt = unsafe { std::slice::from_raw_parts(tp.ptr(), n) };
            let zr = unsafe { zp.slice(rows.start * N..rows.end * N) };
            for (o, j) in zr.iter_mut().zip(rows.start * N..rows.end * N) {
                *o = lw.d3[j] * tt[p2[j]];
            }
            zb(rows, &lw.kron[5]);
        });
        tm.run(&|tid, nt| atm(share(N, tid, nt), &lw.kron[4]));
        (0..d).map(|j| lw.d4[j] * tmat[j]).collect()
    }
}

/// Apply `f(row_index, row)` over `buf` in rows of `width`: in parallel for
/// a prefill, inline for a single decode row (where waking the pool costs
/// more than the work).
fn each_row(buf: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    let rows = buf.len() / width.max(1);
    if rows <= 1 {
        buf.chunks_mut(width).enumerate().for_each(|(i, r)| f(i, r));
        return;
    }
    let p = crate::team::SyncPtr(buf.as_mut_ptr());
    crate::team::team().run(&|tid, n| {
        for i in crate::team::share(rows, tid, n) {
            // SAFETY: each member owns a disjoint range of rows.
            f(i, unsafe { p.slice(i * width..(i + 1) * width) });
        }
    });
}

fn lane_mean(x: &[f32], n: usize, d: usize, out: &mut [f32]) {
    out.copy_from_slice(&x[..d]);
    for lane in 1..n {
        for (o, v) in out.iter_mut().zip(&x[lane * d..(lane + 1) * d]) {
            *o += v;
        }
    }
    let inv = n as f32;
    for o in out.iter_mut() {
        *o /= inv;
    }
}

/// `_kron_apply`: `z (a x b) -> A^T Z B`, flattened.
fn kron_apply(z: &[f32], a: &[f32], na: usize, b: &[f32], nb: usize, out: &mut [f32]) {
    if na == 32 && nb == 32 {
        return kron_apply_32(z, a, b, out);
    }
    // M = Z B  (na x nb)
    let mut m = vec![0f32; na * nb];
    for i in 0..na {
        let mi = &mut m[i * nb..(i + 1) * nb];
        for j in 0..nb {
            let zij = z[i * nb + j];
            for (o, bv) in mi.iter_mut().zip(&b[j * nb..(j + 1) * nb]) {
                *o += zij * bv;
            }
        }
    }
    // out = A^T M
    out[..na * nb].fill(0.0);
    for i in 0..na {
        let mi = &m[i * nb..(i + 1) * nb];
        for k in 0..na {
            let aik = a[i * na + k];
            let ok = &mut out[k * nb..(k + 1) * nb];
            for (o, mv) in ok.iter_mut().zip(mi) {
                *o += aik * mv;
            }
        }
    }
}

/// `acc_r[l] += sum_j c(r, j) * rows[j][l]` for two output rows at once, so
/// each loaded row feeds 16 independent accumulators: 32-wide rows, 32 terms.
#[inline(always)]
pub(crate) fn mac2_32(c0: impl Fn(usize) -> f32, c1: impl Fn(usize) -> f32, rows: &[f32]) -> ([f32; 32], [f32; 32]) {
    const N: usize = 32;
    assert!(rows.len() >= N * N);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; `rows` holds 32 rows of 32.
    unsafe {
        use std::arch::aarch64::*;
        let mut a0 = [vdupq_n_f32(0.0); 8];
        let mut a1 = [vdupq_n_f32(0.0); 8];
        let p = rows.as_ptr();
        for j in 0..N {
            let (w0, w1) = (c0(j), c1(j));
            for q in 0..8 {
                let bq = vld1q_f32(p.add(j * N + q * 4));
                a0[q] = vfmaq_n_f32(a0[q], bq, w0);
                a1[q] = vfmaq_n_f32(a1[q], bq, w1);
            }
        }
        let mut o0 = [0f32; N];
        let mut o1 = [0f32; N];
        for q in 0..8 {
            vst1q_f32(o0.as_mut_ptr().add(q * 4), a0[q]);
            vst1q_f32(o1.as_mut_ptr().add(q * 4), a1[q]);
        }
        (o0, o1)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut a0 = [0f32; N];
        let mut a1 = [0f32; N];
        for j in 0..N {
            let row: &[f32; N] = rows[j * N..(j + 1) * N].try_into().unwrap();
            let (w0, w1) = (c0(j), c1(j));
            for l in 0..N {
                a0[l] += w0 * row[l];
                a1[l] += w1 * row[l];
            }
        }
        (a0, a1)
    }
}

/// [`kron_apply`] for the 32 x 32 split: `M = Z B`, then `A^T M`, two
/// output rows per pass.
pub(crate) fn kron_apply_32(z: &[f32], a: &[f32], b: &[f32], out: &mut [f32]) {
    const N: usize = 32;
    let mut m = [0f32; N * N];
    for i in (0..N).step_by(2) {
        let (r0, r1) = mac2_32(|j| z[i * N + j], |j| z[(i + 1) * N + j], b);
        m[i * N..(i + 1) * N].copy_from_slice(&r0);
        m[(i + 1) * N..(i + 2) * N].copy_from_slice(&r1);
    }
    for k in (0..N).step_by(2) {
        let (r0, r1) = mac2_32(|i| a[i * N + k], |i| a[i * N + k + 1], &m);
        out[k * N..(k + 1) * N].copy_from_slice(&r0);
        out[(k + 1) * N..(k + 2) * N].copy_from_slice(&r1);
    }
}

/// `_sinkhorn`, in place on an `n x n` block: alternate row and column
/// normalisation of `exp(R)`. The reference works in log space
/// (`L -= logsumexp`); dividing by the sums in linear space is the same map
/// with one exponential per entry instead of one per entry per step. A block
/// whose sums underflow falls back to the log-space form.
pub(crate) fn sinkhorn(k: &mut [f32], n: usize, iters: usize) {
    let mut lin = k.to_vec();
    // Per-row shifts cancel in the first row normalisation.
    for r in 0..n {
        let row = &mut lin[r * n..(r + 1) * n];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.iter_mut().for_each(|v| *v = (*v - m).exp());
    }
    let mut ok = true;
    'outer: for _ in 0..iters {
        for r in 0..n {
            let row = &mut lin[r * n..(r + 1) * n];
            let sum: f32 = row.iter().sum();
            if !(sum > 0.0 && sum.is_finite()) {
                ok = false;
                break 'outer;
            }
            let inv = 1.0 / sum;
            row.iter_mut().for_each(|v| *v *= inv);
        }
        for col in 0..n {
            let sum: f32 = (0..n).map(|r| lin[r * n + col]).sum();
            if !(sum > 1e-30 && sum.is_finite()) {
                ok = false;
                break 'outer;
            }
            let inv = 1.0 / sum;
            for r in 0..n {
                lin[r * n + col] *= inv;
            }
        }
    }
    if ok {
        k.copy_from_slice(&lin);
        return;
    }
    for _ in 0..iters {
        for r in 0..n {
            let row = &mut k[r * n..(r + 1) * n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
            row.iter_mut().for_each(|v| *v -= lse);
        }
        for col in 0..n {
            let m = (0..n).map(|r| k[r * n + col]).fold(f32::NEG_INFINITY, f32::max);
            let lse = m + (0..n).map(|r| (k[r * n + col] - m).exp()).sum::<f32>().ln();
            for r in 0..n {
                k[r * n + col] -= lse;
            }
        }
    }
    k.iter_mut().for_each(|v| *v = v.exp());
}

#[cfg(test)]
mod bench {
    #[test]
    #[ignore]
    fn bench_kron32() {
        let z: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.02).cos()).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.03).sin()).collect();
        let mut out = vec![0f32; 1024];
        let n = 100_000;
        let t = web_time::Instant::now();
        for _ in 0..n {
            super::kron_apply_32(std::hint::black_box(&z), &a, &b, &mut out);
            std::hint::black_box(&out);
        }
        let dt = t.elapsed().as_secs_f64() / n as f64;
        eprintln!("kron32: {:.2}us ({:.1} GMAC/s)", dt * 1e6, 65536.0 / dt / 1e9);
    }
}
