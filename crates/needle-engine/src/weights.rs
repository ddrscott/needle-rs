//! Model weights in the layout the engine runs: every projection stored
//! `[out, in]` so a row is contiguous along its reduction axis. Built from a
//! training checkpoint (Flax `[in, out]` kernels, transposed here) or from a
//! `.cact` archive (already `[out, in]`, CQ matrices dequantized).

use anyhow::{Context, Result, bail, ensure};
use needle_core::cact::{Archive, Dtype};
use needle_core::config::{Config, HADA_COND_RANK, hada_blocks};
use needle_core::rng::hada_perms;
use needle_core::tensor::{Params, Tensor, get};

use crate::qlinear::{QLinear, QMat};

#[derive(Clone)]
pub struct LayerWeights {
    pub norm_in: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub q_taps: Vec<f32>,
    pub k_taps: Vec<f32>,
    pub v_taps: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wgate: Vec<f32>,
    pub wout: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub attn_gate: f32,
    pub pre_hada: Vec<f32>,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub b2: Vec<f32>,
    pub d3: Vec<f32>,
    pub d4: Vec<f32>,
    /// `[w1a, w1b, w2a, w2b, w3a, w3b]`, each row-major square.
    pub kron: [Vec<f32>; 6],
    /// `[d_model, rank]`
    pub cond_v: Vec<f32>,
    /// `[rank, hada_n]`
    pub cond_u: Vec<f32>,
    /// mHC projections stacked `[pre (n) | post (n) | res (n*n)]` rows, each `[n*d]`.
    pub phi: Vec<f32>,
    pub b_pre: Vec<f32>,
    pub b_post: Vec<f32>,
    pub b_res: Vec<f32>,
    pub a_pre: f32,
    pub a_post: f32,
    pub a_res: f32,
}

#[derive(Clone)]
pub struct EngramWeights {
    /// `[num_tables * slots, sub_dim]`
    pub tables: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    /// `[ENGRAM_CONV_TAPS, d_model]`
    pub taps: Vec<f32>,
}

/// A probe-pooling head (confidence, embedding, router).
#[derive(Clone)]
pub struct ProbeHead {
    /// `[(L+1), k, d]`
    pub probes: Vec<f32>,
    /// `[(L+1), k]`
    pub gain: Vec<f32>,
    /// `[q, d]`
    pub query: Vec<f32>,
    /// `[q, (L+1), k]`
    pub row_bias: Vec<f32>,
    /// `[out, q*d]`
    pub proj: Vec<f32>,
    pub bias: Vec<f32>,
    pub k: usize,
    pub q: usize,
    pub out: usize,
}

#[derive(Clone)]
pub struct Weights {
    pub config: Config,
    /// `[vocab, d]`
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub engrams: Vec<EngramWeights>,
    pub final_norm: Vec<f32>,
    pub perm1: Vec<usize>,
    pub perm2: Vec<usize>,
    pub confidence: Option<ProbeHead>,
    pub embedding_head: Option<ProbeHead>,
    pub router: Option<ProbeHead>,
    pub router_calibration: Option<[f32; 3]>,
}

/// Walks an archive's tensors in directory order. In fast mode the large
/// CQ matrices are kept packed (collected in `packed`) and come back empty.
struct Cursor<'a> {
    a: &'a Archive,
    i: usize,
    fast: bool,
    packed: Vec<QMat>,
}

impl Cursor<'_> {
    fn next(&mut self) -> Result<Tensor> {
        let t = self.a.tensor(self.i).with_context(|| format!("archive tensor {}", self.i))?;
        self.i += 1;
        Ok(t)
    }

    /// A matmul weight: dequantized, or (fast mode) packed for the CQ path.
    fn big(&mut self) -> Result<Vec<f32>> {
        if self.fast && self.a.records[self.i].dtype == Dtype::Cq {
            let q = QMat::from_cq(&self.a.cq(self.i)?);
            self.i += 1;
            self.packed.push(q);
            Ok(vec![])
        } else {
            let t = self.next()?;
            if self.fast {
                let (o, i) = (t.shape[0], t.shape[1]);
                self.packed.push(QMat::from_f32(&t.data, o, i));
                return Ok(vec![]);
            }
            Ok(t.data)
        }
    }

    fn take(&mut self) -> QMat {
        self.packed.pop().expect("a packed matrix")
    }
}

/// The quantized fast path: every matmul weight packed at 2-4 bits.
pub struct QWeights {
    pub embedding: QMat,
    /// Fused `q | k | v | gate` per layer.
    pub qkvg: Vec<QLinear>,
    pub out: Vec<QLinear>,
    /// Fused mHC `pre | post | res` per layer.
    pub phi: Vec<QLinear>,
    /// The same, dequantized: 24 rows is too few to amortize activation prep.
    pub phi_f32: Vec<Vec<f32>>,
    pub engram_tables: Vec<QMat>,
    /// Fused engram `key | value` per site.
    pub engram_kv: Vec<QLinear>,
}

impl QWeights {
    pub fn bytes(&self) -> usize {
        self.embedding.bytes()
            + self.qkvg.iter().chain(&self.out).chain(&self.phi).chain(&self.engram_kv).map(QLinear::bytes).sum::<usize>()
            + self.engram_tables.iter().map(QMat::bytes).sum::<usize>()
    }

    /// Pack f32 weights (a checkpoint) at CQ-4 for the fast path.
    pub fn from_weights(w: &Weights) -> Self {
        use rayon::prelude::*;
        let c = &w.config;
        let d = c.d_model;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let nc = c.mhc_lanes * d;
        let ncols = 2 * c.mhc_lanes + c.mhc_lanes * c.mhc_lanes;
        let (_, _, sub) = c.engram_geometry();
        let layers: Vec<(QLinear, QLinear, QLinear)> = w
            .layers
            .par_iter()
            .map(|l| {
                (
                    QLinear::new(vec![
                        QMat::from_f32(&l.wq, nh * qk, d),
                        QMat::from_f32(&l.wk, nkv * qk, d),
                        QMat::from_f32(&l.wv, nkv * vd, d),
                        QMat::from_f32(&l.wgate, nh * vd, d),
                    ]),
                    QLinear::new(vec![QMat::from_f32(&l.wout, d, nh * vd)]),
                    QLinear::new(vec![QMat::from_f32(&l.phi, ncols, nc)]),
                )
            })
            .collect();
        let engrams: Vec<(QMat, QLinear)> = w
            .engrams
            .par_iter()
            .map(|e| {
                (
                    QMat::from_f32(&e.tables, e.tables.len() / sub, sub),
                    QLinear::new(vec![QMat::from_f32(&e.wk, d, d), QMat::from_f32(&e.wv, d, d)]),
                )
            })
            .collect();
        let (qkvg, rest): (Vec<_>, Vec<_>) = layers.into_iter().map(|(a, b, c)| (a, (b, c))).unzip();
        let (out, phi) = rest.into_iter().unzip();
        let (engram_tables, engram_kv) = engrams.into_iter().unzip();
        let phi_f32 = w.layers.iter().map(|l| l.phi.clone()).collect();
        Self { embedding: QMat::from_f32(&w.embedding, w.embedding.len() / d, d), qkvg, out, phi, phi_f32, engram_tables, engram_kv }
    }
}

fn layer_of(t: &Tensor, i: usize) -> Vec<f32> {
    t.slice0(i).to_vec()
}

/// Layer `i` of a stacked `[L, in, out]` kernel, transposed to `[out, in]`.
fn kernel_t(t: &Tensor, i: usize) -> Vec<f32> {
    t.index0(i).t_last2().data
}

impl Weights {
    pub fn from_checkpoint(params: &Params, config: &Config) -> Result<Self> {
        crate::cpu::init();
        let blk = |k: &str| get(params, &format!("stack/layers/block/{k}"));
        let st = |k: &str| get(params, &format!("stack/{k}"));
        let n = config.mhc_lanes;
        let nc = n * config.d_model;
        let mut layers = Vec::with_capacity(config.num_layers);
        let (phi_pre, phi_post, phi_res) = (st("mhc_phi_pre")?, st("mhc_phi_post")?, st("mhc_phi_res")?);
        for i in 0..config.num_layers {
            let taps = |k: &str| -> Result<Vec<f32>> { if config.qkv_conv_taps > 0 { Ok(layer_of(blk(k)?, i)) } else { Ok(vec![]) } };
            // mHC: (nC, lanes) per layer -> rows [lane][nC].
            let mut phi = Vec::with_capacity((2 * n + n * n) * nc);
            for p in [phi_pre, phi_post, phi_res] {
                let cols = p.shape[2];
                let l = p.slice0(i);
                for j in 0..cols {
                    phi.extend((0..nc).map(|c| l[c * cols + j]));
                }
            }
            let h = |k: &str| -> Result<Vec<f32>> { Ok(layer_of(blk(&format!("hadamard_mlp/{k}"))?, i)) };
            layers.push(LayerWeights {
                norm_in: layer_of(blk("ZCRMSNorm_0/scale")?, i),
                wq: kernel_t(blk("self_attn/q_proj/kernel")?, i),
                wk: kernel_t(blk("self_attn/k_proj/kernel")?, i),
                wv: kernel_t(blk("self_attn/v_proj/kernel")?, i),
                q_taps: taps("self_attn/q_taps")?,
                k_taps: taps("self_attn/k_taps")?,
                v_taps: taps("self_attn/v_taps")?,
                q_norm: layer_of(blk("self_attn/q_norm/scale")?, i),
                k_norm: layer_of(blk("self_attn/k_norm/scale")?, i),
                wgate: kernel_t(blk("self_attn/gate_proj/kernel")?, i),
                wout: kernel_t(blk("self_attn/out_proj/kernel")?, i),
                post_norm: layer_of(blk("post_attn_norm/scale")?, i),
                attn_gate: blk("attn_gate")?.data[i],
                pre_hada: layer_of(blk("pre_hada_norm/scale")?, i),
                d1: h("d1")?,
                d2: h("d2")?,
                b2: h("b2")?,
                d3: h("d3")?,
                d4: h("d4")?,
                kron: [h("w1a")?, h("w1b")?, h("w2a")?, h("w2b")?, h("w3a")?, h("w3b")?],
                cond_v: h("cond_v")?,
                cond_u: h("cond_u")?,
                phi,
                b_pre: layer_of(st("mhc_b_pre")?, i),
                b_post: layer_of(st("mhc_b_post")?, i),
                b_res: layer_of(st("mhc_b_res")?, i),
                a_pre: st("mhc_a_pre")?.data[i],
                a_post: st("mhc_a_post")?.data[i],
                a_res: st("mhc_a_res")?.data[i],
            });
        }
        let mut engrams = Vec::new();
        for s in 0..config.engram_layers.len() {
            let e = |k: &str| get(params, &format!("engrams_{s}/{k}"));
            engrams.push(EngramWeights {
                tables: e("embedding")?.data.clone(),
                wk: e("key_proj/kernel")?.t_last2().data,
                wv: e("value_proj/kernel")?.t_last2().data,
                taps: e("taps")?.data.clone(),
            });
        }
        let head = |key: &str, k: usize, q: usize| -> Result<Option<ProbeHead>> {
            let Some(probes) = params.get(&format!("{key}/probes")) else { return Ok(None) };
            let kernel = get(params, &format!("{key}/proj/kernel"))?;
            let out = kernel.shape[1];
            let bias = params.get(&format!("{key}/proj/bias")).map(|b| b.data.clone()).unwrap_or(vec![0.0; out]);
            ensure!(probes.shape[1] == k, "{key} has {} probes, config says {k}", probes.shape[1]);
            Ok(Some(ProbeHead {
                probes: probes.data.clone(),
                gain: get(params, &format!("{key}/gain"))?.data.clone(),
                query: get(params, &format!("{key}/query"))?.data.clone(),
                row_bias: get(params, &format!("{key}/row_bias"))?.data.clone(),
                proj: kernel.t_last2().data,
                bias,
                k,
                q,
                out,
            }))
        };
        let [perm1, perm2] = hada_perms(config.hada_n(), config.hada_split());
        Ok(Self {
            config: config.clone(),
            embedding: get(params, "embedding/embedding")?.data.clone(),
            layers,
            engrams,
            final_norm: get(params, "stack/final_norm/scale")?.data.clone(),
            perm1,
            perm2,
            confidence: head("confidence_head", config.confidence_probes, config.confidence_queries)?,
            embedding_head: head("embedding_head", config.embedding_probes, config.embedding_queries)?,
            router: head("router_head", config.router_probes, config.router_queries)?,
            router_calibration: params.get("router_head/calibration").map(|c| [c.data[0], c.data[1], c.data[2]]),
        })
    }

    /// Dequantize a `.cact` archive into f32 weights.
    pub fn from_archive(a: &Archive) -> Result<Self> {
        Ok(Self::from_archive_mode(a, false)?.0)
    }

    /// Read a `.cact` archive; with `fast`, keep the matmul weights packed
    /// (returned as [`QWeights`]) and leave their f32 fields empty.
    pub fn from_archive_mode(a: &Archive, fast: bool) -> Result<(Self, Option<QWeights>)> {
        crate::cpu::init();
        let config = a.header.to_config();
        let h = &a.header;
        let (qk, vd) = config.head_dims();
        let n = config.mhc_lanes;
        let mut cur = Cursor { a, i: 0, fast, packed: vec![] };
        let embedding = cur.big()?;
        let q_embedding = fast.then(|| cur.take());
        ensure!(fast || embedding.len() == config.vocab_size * config.d_model, "embedding shape does not match the header");
        let mut q_layers: Vec<[QMat; 5]> = vec![];
        let mut raw_layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let norm_in = cur.next()?.data;
            let (wq, wk, wv) = (cur.big()?, cur.big()?, cur.big()?);
            let (q_taps, k_taps, v_taps) =
                if h.qkv_conv_taps > 0 { (cur.next()?.data, cur.next()?.data, cur.next()?.data) } else { (vec![], vec![], vec![]) };
            let (q_norm, k_norm) = (cur.next()?.data, cur.next()?.data);
            let (wgate, wout) = (cur.big()?, cur.big()?);
            if fast {
                let (o, g, v, k, q) = (cur.take(), cur.take(), cur.take(), cur.take(), cur.take());
                q_layers.push([q, k, v, g, o]);
            }
            let post_norm = cur.next()?.data;
            let attn_gate = cur.next()?.data[0];
            let pre_hada = cur.next()?.data;
            let (d1, d2, b2, d3, d4) = (cur.next()?.data, cur.next()?.data, cur.next()?.data, cur.next()?.data, cur.next()?.data);
            let kron = [cur.next()?.data, cur.next()?.data, cur.next()?.data, cur.next()?.data, cur.next()?.data, cur.next()?.data];
            let (cond_v, cond_u) = (cur.next()?.data, cur.next()?.data);
            ensure!(fast || wq.len() == config.num_heads * qk * config.d_model, "q_proj shape");
            ensure!(fast || wout.len() == config.d_model * config.num_heads * vd, "out_proj shape");
            raw_layers.push(LayerWeights {
                norm_in,
                wq,
                wk,
                wv,
                q_taps,
                k_taps,
                v_taps,
                q_norm,
                k_norm,
                wgate,
                wout,
                post_norm,
                attn_gate,
                pre_hada,
                d1,
                d2,
                b2,
                d3,
                d4,
                kron,
                cond_v,
                cond_u,
                phi: vec![],
                b_pre: vec![],
                b_post: vec![],
                b_res: vec![],
                a_pre: 0.0,
                a_post: 0.0,
                a_res: 0.0,
            });
        }
        let (a_pre, a_post, a_res) = (cur.next()?.data, cur.next()?.data, cur.next()?.data);
        let (b_pre, b_post, b_res) = (cur.next()?.data, cur.next()?.data, cur.next()?.data);
        let (phi_pre, phi_post, phi_res) = (cur.big()?, cur.big()?, cur.big()?);
        let q_phi = fast.then(|| {
            let (res, post, pre) = (cur.take(), cur.take(), cur.take());
            (0..config.num_layers)
                .map(|l| {
                    QLinear::new(vec![
                        pre.slice_rows(l * n, (l + 1) * n),
                        post.slice_rows(l * n, (l + 1) * n),
                        res.slice_rows(l * n * n, (l + 1) * n * n),
                    ])
                })
                .collect::<Vec<_>>()
        });
        let nc = n * config.d_model;
        for (l, lw) in raw_layers.iter_mut().enumerate() {
            lw.a_pre = a_pre[l];
            lw.a_post = a_post[l];
            lw.a_res = a_res[l];
            lw.b_pre = b_pre[l * n..(l + 1) * n].to_vec();
            lw.b_post = b_post[l * n..(l + 1) * n].to_vec();
            lw.b_res = b_res[l * n * n..(l + 1) * n * n].to_vec();
            if !fast {
                let mut phi = Vec::with_capacity((2 * n + n * n) * nc);
                phi.extend_from_slice(&phi_pre[l * n * nc..(l + 1) * n * nc]);
                phi.extend_from_slice(&phi_post[l * n * nc..(l + 1) * n * nc]);
                phi.extend_from_slice(&phi_res[l * n * n * nc..(l + 1) * n * n * nc]);
                lw.phi = phi;
            }
        }
        let perm = |t: Tensor| -> Vec<usize> { t.data.iter().map(|&v| v as usize).collect() };
        let perm1 = perm(cur.next()?);
        let perm2 = perm(cur.next()?);
        let mut engrams = Vec::new();
        let mut q_engrams: Vec<(QMat, QLinear)> = vec![];
        for _ in 0..config.engram_layers.len() {
            let (tables, wk, wv) = (cur.big()?, cur.big()?, cur.big()?);
            engrams.push(EngramWeights { tables, wk, wv, taps: cur.next()?.data });
            if fast {
                let (v, k, t) = (cur.take(), cur.take(), cur.take());
                q_engrams.push((t, QLinear::new(vec![k, v])));
            }
        }
        let final_norm = cur.next()?.data;

        // Probe heads: everything between final_norm and the tokenizer.
        let tok_index = a.records.iter().position(|r| r.dtype == Dtype::Raw).unwrap_or(a.records.len());
        let mut confidence = None;
        let mut embedding_head = None;
        let mut router = None;
        let mut router_calibration = None;
        if cur.i < tok_index {
            let manifest = cur.next()?.data;
            let l1 = config.num_layers + 1;
            let d = config.d_model;
            for code in manifest {
                let probes = cur.next()?;
                let gain = cur.next()?;
                let query = cur.next()?;
                let row_bias = cur.next()?;
                let proj = cur.next()?;
                let bias = cur.next()?.data;
                let k = gain.shape[1];
                let q = query.shape[0];
                ensure!(probes.numel() == l1 * k * d, "probe head rows do not match the depth");
                let head = ProbeHead {
                    probes: probes.data,
                    gain: gain.data,
                    query: query.data,
                    row_bias: row_bias.data,
                    out: proj.shape[0],
                    proj: proj.data,
                    bias,
                    k,
                    q,
                };
                match code.round() as i32 {
                    1 => embedding_head = Some(head),
                    2 => confidence = Some(head),
                    3 => {
                        let c = cur.next()?.data;
                        router_calibration = Some([c[0], c[1], c[2]]);
                        router = Some(head);
                    }
                    other => bail!("unknown probe head code {other}"),
                }
            }
        }
        let expect = config.hada_n();
        ensure!(perm1.len() == expect && perm2.len() == expect, "Hadamard permutations do not match hada_n");
        let (ba, bb) = hada_blocks(expect);
        ensure!(raw_layers.iter().all(|l| l.kron[0].len() == ba * ba && l.kron[1].len() == bb * bb), "Hadamard factor shapes");
        ensure!(raw_layers.iter().all(|l| l.cond_u.len() == HADA_COND_RANK * expect), "cond_u shape");
        let w = Self {
            config,
            embedding,
            layers: raw_layers,
            engrams,
            final_norm,
            perm1,
            perm2,
            confidence,
            embedding_head,
            router,
            router_calibration,
        };
        let q = q_embedding.map(|embedding| {
            let mut qkvg = vec![];
            let mut out = vec![];
            for [q, k, v, g, o] in q_layers {
                qkvg.push(QLinear::new(vec![q, k, v, g]));
                out.push(QLinear::new(vec![o]));
            }
            let (engram_tables, engram_kv) = q_engrams.into_iter().unzip();
            let phi = q_phi.unwrap_or_default();
            let phi_f32 = phi.iter().map(|p: &QLinear| p.dequantize()).collect();
            QWeights { embedding, qkvg, out, phi, phi_f32, engram_tables, engram_kv }
        });
        Ok((w, q))
    }

    /// Fake-quantize every matmul weight the way `cq_ste_params` does (the
    /// numerics a `needle build` archive ships), reduction axis last.
    pub fn cq_fake_quantize(&mut self, bits: u32) {
        use needle_core::quant::{CQ_GROUP_SIZE, cq_quantize_rows, ste};
        use rayon::prelude::*;
        let q = |w: &mut Vec<f32>, d: usize| {
            let deq = cq_quantize_rows(w, d, bits, CQ_GROUP_SIZE);
            *w = ste(w, &deq);
        };
        let d = self.config.d_model;
        let (qk, vd) = self.config.head_dims();
        let nh = self.config.num_heads;
        q(&mut self.embedding, d);
        let nc = self.config.mhc_lanes * d;
        self.layers.par_iter_mut().for_each(|l| {
            q(&mut l.wq, d);
            q(&mut l.wk, d);
            q(&mut l.wv, d);
            q(&mut l.wgate, d);
            q(&mut l.wout, nh * vd);
            q(&mut l.phi, nc);
        });
        let (_, _, sub) = self.config.engram_geometry();
        self.engrams.par_iter_mut().for_each(|e| {
            q(&mut e.tables, sub);
            q(&mut e.wk, d);
            q(&mut e.wv, d);
        });
        let _ = qk;
    }
}
