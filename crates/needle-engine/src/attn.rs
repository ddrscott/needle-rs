//! Flash attention over f32 keys for the general (unpacked) forward pass:
//! partial softmax sums per key range, merged exactly.

use crate::linalg::{dot, fast_exp};

/// [`attend_group`] over the keys of `ranges` (empty ranges allowed),
/// merged exactly into `dst`; no keys at all leaves a `-inf` max.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_ranges(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    kd: usize,
    vdim: usize,
    kvh: usize,
    ranges: &[std::ops::Range<usize>],
    scale: f32,
    group: usize,
    qk: usize,
    vd: usize,
    dst: &mut [f32],
    stride: usize,
) {
    let mut first = true;
    let mut tmp = vec![];
    for r in ranges.iter().filter(|r| !r.is_empty()) {
        if first {
            dst[..group * stride].fill(0.0);
            attend_group(q, kc, vc, kd, vdim, kvh, r.start, r.len(), scale, group, qk, vd, dst, stride);
            first = false;
            continue;
        }
        tmp.clear();
        tmp.resize(group * stride, 0.0);
        attend_group(q, kc, vc, kd, vdim, kvh, r.start, r.len(), scale, group, qk, vd, &mut tmp, stride);
        for g in 0..group {
            let (a, b) = (&mut dst[g * stride..(g + 1) * stride], &tmp[g * stride..(g + 1) * stride]);
            let m = a[0].max(b[0]);
            let (wa, wb) = (fast_exp(a[0] - m), fast_exp(b[0] - m));
            a[0] = m;
            a[1] = wa * a[1] + wb * b[1];
            for (x, y) in a[2..2 + vd].iter_mut().zip(&b[2..2 + vd]) {
                *x = wa * *x + wb * y;
            }
        }
    }
    if first {
        for g in 0..group {
            dst[g * stride..(g + 1) * stride].fill(0.0);
            dst[g * stride] = f32::NEG_INFINITY;
        }
    }
}

/// Partial attention of `group` query heads (sharing one KV head) over keys
/// `k0..k0+len`: per head, the max score, the sum of `exp(score - max)`, and
/// the unnormalised weighted value sum, into `dst` rows of `stride`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_group(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    kd: usize,
    vdim: usize,
    kvh: usize,
    k0: usize,
    len: usize,
    scale: f32,
    group: usize,
    qk: usize,
    vd: usize,
    dst: &mut [f32],
    stride: usize,
) {
    let mut scores = vec![0f32; group * len];
    #[cfg(target_arch = "aarch64")]
    if qk == 48 && vd == 64 {
        // SAFETY: NEON is baseline on aarch64; slice bounds are checked by
        // the indexing that builds each pointer.
        unsafe { neon_scores48(q, &kc[k0 * kd + kvh * qk..], kd, len, group, scale, &mut scores) };
    } else {
        scalar_scores(q, kc, kd, kvh, k0, len, group, qk, scale, &mut scores);
    }
    #[cfg(not(target_arch = "aarch64"))]
    scalar_scores(q, kc, kd, kvh, k0, len, group, qk, scale, &mut scores);
    for g in 0..group {
        let sc = &mut scores[g * len..(g + 1) * len];
        let m = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut l_sum = 0f32;
        for v in sc.iter_mut() {
            *v = fast_exp(*v - m);
            l_sum += *v;
        }
        dst[g * stride] = m;
        dst[g * stride + 1] = l_sum;
        let o = &mut dst[g * stride + 2..(g + 1) * stride];
        #[cfg(target_arch = "aarch64")]
        if vd == 64 {
            // SAFETY: as above.
            unsafe { neon_weighted64(sc, &vc[k0 * vdim + kvh * vd..], vdim, o) };
            continue;
        }
        o.fill(0.0);
        for (j, &w) in sc.iter().enumerate() {
            let vrow = &vc[(k0 + j) * vdim + kvh * vd..(k0 + j) * vdim + (kvh + 1) * vd];
            for (a, b) in o.iter_mut().zip(vrow) {
                *a += w * b;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scalar_scores(
    q: &[f32],
    kc: &[f32],
    kd: usize,
    kvh: usize,
    k0: usize,
    len: usize,
    group: usize,
    qk: usize,
    scale: f32,
    scores: &mut [f32],
) {
    for j in 0..len {
        let krow = &kc[(k0 + j) * kd + kvh * qk..(k0 + j) * kd + (kvh + 1) * qk];
        for g in 0..group {
            scores[g * len + j] = dot(&q[g * qk..(g + 1) * qk], krow) * scale;
        }
    }
}

/// Scores of `group` 48-wide query heads against `len` keys (rows `kd`
/// apart), two keys per pass so each query load feeds both.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_scores48(q: &[f32], k: &[f32], kd: usize, len: usize, group: usize, scale: f32, scores: &mut [f32]) {
    use std::arch::aarch64::*;
    assert!(q.len() >= group * 48 && (len == 0 || k.len() >= (len - 1) * kd + 48));
    unsafe {
        let qp = q.as_ptr();
        let mut j = 0;
        while j + 2 <= len {
            let (ka, kb) = (k.as_ptr().add(j * kd), k.as_ptr().add((j + 1) * kd));
            let mut kav = [vdupq_n_f32(0.0); 12];
            let mut kbv = [vdupq_n_f32(0.0); 12];
            for i in 0..12 {
                kav[i] = vld1q_f32(ka.add(i * 4));
                kbv[i] = vld1q_f32(kb.add(i * 4));
            }
            for g in 0..group {
                let (mut a0, mut a1, mut b0, mut b1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                for i in (0..12).step_by(2) {
                    let q0 = vld1q_f32(qp.add(g * 48 + i * 4));
                    let q1 = vld1q_f32(qp.add(g * 48 + i * 4 + 4));
                    a0 = vfmaq_f32(a0, q0, kav[i]);
                    a1 = vfmaq_f32(a1, q1, kav[i + 1]);
                    b0 = vfmaq_f32(b0, q0, kbv[i]);
                    b1 = vfmaq_f32(b1, q1, kbv[i + 1]);
                }
                *scores.get_unchecked_mut(g * len + j) = vaddvq_f32(vaddq_f32(a0, a1)) * scale;
                *scores.get_unchecked_mut(g * len + j + 1) = vaddvq_f32(vaddq_f32(b0, b1)) * scale;
            }
            j += 2;
        }
        if j < len {
            let ka = k.as_ptr().add(j * kd);
            for g in 0..group {
                let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
                for i in (0..12).step_by(2) {
                    a0 = vfmaq_f32(a0, vld1q_f32(qp.add(g * 48 + i * 4)), vld1q_f32(ka.add(i * 4)));
                    a1 = vfmaq_f32(a1, vld1q_f32(qp.add(g * 48 + i * 4 + 4)), vld1q_f32(ka.add(i * 4 + 4)));
                }
                *scores.get_unchecked_mut(g * len + j) = vaddvq_f32(vaddq_f32(a0, a1)) * scale;
            }
        }
    }
}

/// `o = sum_j w[j] * v_j` for 64-wide value rows `vdim` apart, the output
/// held in registers across the keys.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_weighted64(w: &[f32], v: &[f32], vdim: usize, o: &mut [f32]) {
    use std::arch::aarch64::*;
    assert!(o.len() >= 64 && (w.is_empty() || v.len() >= (w.len() - 1) * vdim + 64));
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        for (j, &wj) in w.iter().enumerate() {
            let vr = v.as_ptr().add(j * vdim);
            for (i, a) in acc.iter_mut().enumerate() {
                *a = vfmaq_n_f32(*a, vld1q_f32(vr.add(i * 4)), wj);
            }
        }
        for (i, a) in acc.iter().enumerate() {
            vst1q_f32(o.as_mut_ptr().add(i * 4), *a);
        }
    }
}
