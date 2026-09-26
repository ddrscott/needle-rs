//! Probe-pooling heads: calibrated confidence and the sentence embedding.
//!
//! A head attends over the lane-mean stream of every depth (`L + 1` rows:
//! the embedding and each block's output) with `k` learned probes per row,
//! then pools those `(L+1)*k` summaries with `q` learned queries.

use crate::linalg::{dot, rms_unit, softmax_inplace};
use crate::model::{Model, Outputs};
use crate::weights::ProbeHead;

/// `probe_pool` over cells `[t, l1, d]`: the pooled `[q * d]` vector.
pub fn probe_pool(cells: &[f32], t: usize, l1: usize, d: usize, h: &ProbeHead) -> Vec<f32> {
    let (k, q) = (h.k, h.q);
    let inv = 1.0 / (d as f64).sqrt() as f32;
    // r[l, k, :] = rms_unit(sum_t softmax_t(cells . probe) cells) * gain,
    // the (depth, probe) pairs split across the decode team.
    let mut r = vec![0f32; l1 * k * d];
    let rp = crate::team::SyncPtr(r.as_mut_ptr());
    crate::team::team().run(&|tid, nt| {
        let mut scores = vec![0f32; t];
        let mut acc = vec![0f32; d];
        for lk in crate::team::share(l1 * k, tid, nt) {
            let l = lk / k;
            let probe = &h.probes[lk * d..(lk + 1) * d];
            for (ti, s) in scores.iter_mut().enumerate() {
                *s = dot(&cells[(ti * l1 + l) * d..(ti * l1 + l + 1) * d], probe) * inv;
            }
            softmax_inplace(&mut scores);
            acc.fill(0.0);
            for (ti, &w) in scores.iter().enumerate() {
                for (a, c) in acc.iter_mut().zip(&cells[(ti * l1 + l) * d..(ti * l1 + l + 1) * d]) {
                    *a += w * c;
                }
            }
            // SAFETY: each member writes only its own (depth, probe) rows.
            let dst = unsafe { rp.slice(lk * d..(lk + 1) * d) };
            rms_unit(&acc, dst);
            let g = h.gain[lk];
            dst.iter_mut().for_each(|v| *v *= g);
        }
    });
    let m = l1 * k;
    let mut out = vec![0f32; q * d];
    for qi in 0..q {
        let query = &h.query[qi * d..(qi + 1) * d];
        let mut u: Vec<f32> = (0..m).map(|j| dot(&r[j * d..(j + 1) * d], query) * inv + h.row_bias[qi * m + j]).collect();
        softmax_inplace(&mut u);
        let dst = &mut out[qi * d..(qi + 1) * d];
        for (j, &w) in u.iter().enumerate() {
            for (a, v) in dst.iter_mut().zip(&r[j * d..(j + 1) * d]) {
                *a += w * v;
            }
        }
    }
    out
}

fn project(h: &ProbeHead, pooled: &[f32]) -> Vec<f32> {
    let n = pooled.len();
    (0..h.out).map(|o| dot(&h.proj[o * n..(o + 1) * n], pooled) + h.bias[o]).collect()
}

/// Cells of a token sequence (`hidden_cells`): `[t, L+1, d]`.
pub fn cells(model: &Model, tokens: &[u32]) -> Vec<f32> {
    let mut s = model.session();
    model.forward_cells(&mut s, tokens, Outputs::None).cells.expect("cells requested")
}

/// `forward_confidence`: the head's logit, `None` without a head.
pub fn confidence_logit(model: &Model, cells: &[f32], t: usize) -> Option<f32> {
    let h = model.w.confidence.as_ref()?;
    let c = model.config();
    let pooled = probe_pool(cells, t, c.num_layers + 1, c.d_model, h);
    Some(project(h, &pooled)[0])
}

/// The retrieval embedding: the pooled probe vector, unit norm. Uses the
/// embedding head when the archive has one, else the confidence head's pool.
pub fn embedding(model: &Model, cells: &[f32], t: usize) -> Option<Vec<f32>> {
    let c = model.config();
    let l1 = c.num_layers + 1;
    let mut v = match (&model.w.embedding_head, &model.w.confidence) {
        (Some(h), _) => project(h, &probe_pool(cells, t, l1, c.d_model, h)),
        (None, Some(h)) => probe_pool(cells, t, l1, c.d_model, h),
        _ => return None,
    };
    let norm = (v.iter().map(|x| x * x).sum::<f32>() + 1e-12).sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    Some(v)
}
