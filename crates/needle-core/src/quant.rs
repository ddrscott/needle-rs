//! Cactus Quants (CQ): Hadamard-rotated, unit-normalised groups snapped to a
//! Lloyd-Max Gaussian codebook, plus the symmetric absmax quantizer used for
//! A8 activations and the int8 KV cache.

use half::f16;

pub const WEIGHT_BITS: u32 = 4;
pub const HEAD_BITS: u32 = 4;
pub const ACT_BITS: u32 = 8;
pub const CQ_GROUP_SIZE: usize = 128;
/// Record `bits` value that marks ternary crumbs.
pub const TERNARY_RECORD_BITS: u32 = 5;
pub const CB_BITS: [u32; 3] = [2, 3, 4];

const TERNARY_C: f64 = 1.2240064;

// `_lloyd_max_gaussian(bits)` from the reference: 200 Lloyd iterations over
// 400k seeded Gaussian draws. Stored as exact f64 bit patterns.
const LM2: [u64; 4] = [0xbff81b52e4028034, 0xbfdce467275b97c2, 0x3fdcfd177c54d348, 0x3ff828af1c856640];
const LM3: [u64; 8] = [
    0xc00121468f3c6b81,
    0xbff556e7125deaca,
    0xbfe7f2e0f9d592ab,
    0xbfce6e4509e1aa2c,
    0x3fd000bb25e4dfa5,
    0x3fe82dc469b936a5,
    0x3ff568607ac227c5,
    0x40012e5beccc8fda,
];
const LM4: [u64; 16] = [
    0xc005ae0f408928fb,
    0xc0005b0b8a516d54,
    0xbff9841ff0758fc8,
    0xbff3bbbebadac679,
    0xbfed82c6dd21f59e,
    0xbfe47717151708b5,
    0xbfd7d9b9e4b565fa,
    0xbfbd4ca474b8559b,
    0x3fc2683e9154fce2,
    0x3fd9daf49368793a,
    0x3fe586b43e57d29d,
    0x3feec3d50cbc3b33,
    0x3ff4688db93b85f1,
    0x3ffa3164906b428f,
    0x4000ace7bbde2442,
    0x4005ed98d06f2a4a,
];

/// The unit-variance Lloyd-Max centroids for a width (`bits == 5` is ternary).
pub fn lloyd_max(bits: u32) -> Vec<f64> {
    match bits {
        1 => {
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            vec![-c, c]
        }
        2 => LM2.iter().map(|&b| f64::from_bits(b)).collect(),
        3 => LM3.iter().map(|&b| f64::from_bits(b)).collect(),
        4 => LM4.iter().map(|&b| f64::from_bits(b)).collect(),
        TERNARY_RECORD_BITS => vec![-TERNARY_C, 0.0, TERNARY_C],
        _ => panic!("no CQ codebook for {bits} bits"),
    }
}

/// `_cq_codebook_np(bits, group)`: centroids scaled onto the unit sphere.
pub fn codebook(bits: u32, group: usize) -> Vec<f32> {
    let s = (group as f64).sqrt();
    lloyd_max(bits).into_iter().map(|c| (c / s) as f32).collect()
}

/// Scale of the normalised Walsh-Hadamard matrix of size `n`.
pub fn hadamard_scale(n: usize) -> f32 {
    (1.0f64 / (n as f64).sqrt()) as f32
}

/// Dense normalised Sylvester Walsh-Hadamard matrix (row-major).
pub fn hadamard(n: usize) -> Vec<f32> {
    assert!(n.is_power_of_two());
    let s = hadamard_scale(n);
    let mut h = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let sign = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
            h[i * n + j] = sign * s;
        }
    }
    h
}

/// In-place unnormalised fast Walsh-Hadamard transform (`x @ H_sylvester`).
#[inline]
pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// `x @ H` for the normalised Hadamard matrix, in place.
#[inline]
pub fn rotate(x: &mut [f32]) {
    fwht(x);
    let s = hadamard_scale(x.len());
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// `_cq_nearest`: nearest centroid, ties to the lower one.
#[inline]
pub fn nearest(x: f32, cb: &[f32]) -> usize {
    // searchsorted(side="left"): first index with cb[i] >= x.
    let pos = cb.partition_point(|&c| c < x).clamp(1, cb.len() - 1);
    let (l, r) = (cb[pos - 1], cb[pos]);
    if (x - l).abs() <= (x - r).abs() { pos - 1 } else { pos }
}

/// One quantized group: the centroid indices and the (unrounded) L2 norm.
fn quantize_group(g: &mut [f32], cb: &[f32], idx: &mut [u8]) -> f32 {
    rotate(g);
    let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = norm.max(1e-12);
    for (i, v) in g.iter().enumerate() {
        idx[i] = nearest(v / denom, cb) as u8;
    }
    norm
}

/// Fake-quantize rows of length `d` (reduction axis last) through CQ at
/// `bits`: what `cq_quantize` returns, with the norm stored at f16.
pub fn cq_quantize_rows(w: &[f32], d: usize, bits: u32, group: usize) -> Vec<f32> {
    let cb = codebook(bits, group);
    let rows = w.len() / d;
    let in_pad = d.div_ceil(group) * group;
    let mut out = vec![0f32; w.len()];
    let mut g = vec![0f32; group];
    let mut idx = vec![0u8; group];
    for r in 0..rows {
        let src = &w[r * d..(r + 1) * d];
        let dst = &mut out[r * d..(r + 1) * d];
        for start in (0..in_pad).step_by(group) {
            let take = group.min(d.saturating_sub(start));
            g[..take].copy_from_slice(&src[start..start + take]);
            g[take..].fill(0.0);
            let norm = quantize_group(&mut g, &cb, &mut idx);
            let norm = f16::from_f32(norm).to_f32();
            for (i, v) in g.iter_mut().enumerate() {
                *v = cb[idx[i] as usize] * norm;
            }
            rotate(&mut g);
            dst[start..start + take].copy_from_slice(&g[..take]);
        }
    }
    out
}

/// Straight-through value `w + (q - w)` as the reference computes it in f32.
pub fn ste(w: &[f32], q: &[f32]) -> Vec<f32> {
    w.iter().zip(q).map(|(&w, &q)| w + (q - w)).collect()
}

/// `fake_quant(x, group, bits)`: symmetric absmax integer quantization,
/// groups along the last axis, straight-through form.
pub fn fake_quant_rows(x: &mut [f32], group: usize, bits: u32) {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    for g in x.chunks_mut(group) {
        let absmax = g.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if absmax > 0.0 { absmax / qmax } else { 1.0 };
        for v in g.iter_mut() {
            let q = (*v / scale).round_ties_even().clamp(-qmax - 1.0, qmax) * scale;
            *v = *v + (q - *v);
        }
    }
}

/// `_pack_lsb`: indices of one row packed LSB-first, 8 per `bits` bytes.
pub fn pack_lsb(idx: &[u8], bits: u32) -> Vec<u8> {
    let bits = bits as usize;
    let mut out = Vec::with_capacity(idx.len() * bits / 8);
    for chunk in idx.chunks(8) {
        let mut word = 0u64;
        for (i, &v) in chunk.iter().enumerate() {
            word |= (v as u64) << (i * bits);
        }
        for b in 0..bits {
            out.push((word >> (8 * b)) as u8);
        }
    }
    out
}

pub fn unpack_lsb(packed: &[u8], bits: u32, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    unpack_lsb_into(packed, bits, &mut out);
    out
}

/// As [`unpack_lsb`], into `out` (`out.len()` indices).
pub fn unpack_lsb_into(packed: &[u8], bits: u32, out: &mut [u8]) {
    let bits = bits as usize;
    let mask = (1u64 << bits) - 1;
    let mut k = 0;
    for chunk in packed.chunks(bits) {
        let mut word = 0u64;
        for (b, &byte) in chunk.iter().enumerate() {
            word |= (byte as u64) << (8 * b);
        }
        for i in 0..8 {
            if k == out.len() {
                return;
            }
            out[k] = ((word >> (i * bits)) & mask) as u8;
            k += 1;
        }
    }
}

/// A CQ-packed `[out, in]` matrix: indices then f16 group norms.
pub struct CqPacked {
    pub packed: Vec<u8>,
    pub norms: Vec<f16>,
}

/// `_cq_pack(w, bits, group)` for a row-major `[out, d]` matrix.
pub fn cq_pack(w: &[f32], d: usize, bits: u32, group: usize) -> CqPacked {
    assert_eq!(bits, WEIGHT_BITS, "CQ packing supports bits={WEIGHT_BITS}");
    let cb = codebook(bits, group);
    let rows = w.len() / d;
    let in_pad = d.div_ceil(group) * group;
    let per_row = in_pad * bits as usize / 8;
    let mut packed = vec![0u8; rows * per_row];
    let mut norms = vec![f16::ZERO; rows * (in_pad / group)];
    use rayon::prelude::*;
    packed.par_chunks_mut(per_row).zip(norms.par_chunks_mut(in_pad / group)).enumerate().for_each(|(r, (prow, nrow))| {
        let src = &w[r * d..(r + 1) * d];
        let mut idx = vec![0u8; in_pad];
        let mut g = vec![0f32; group];
        for (gi, start) in (0..in_pad).step_by(group).enumerate() {
            let take = group.min(d.saturating_sub(start));
            g[..take].copy_from_slice(&src[start..start + take]);
            g[take..].fill(0.0);
            let norm = quantize_group(&mut g, &cb, &mut idx[start..start + group]);
            nrow[gi] = f16::from_f32(norm);
        }
        prow.copy_from_slice(&pack_lsb(&idx, bits));
    });
    CqPacked { packed, norms }
}

/// `_cq_unpack`: dequantize a CQ blob back to `[out, in_dim]` f32.
pub fn cq_unpack(packed: &[u8], norms: &[f16], out: usize, in_dim: usize, bits: u32, group: usize) -> Vec<f32> {
    let in_pad = in_dim.div_ceil(group) * group;
    let cb = codebook(bits, group);
    let row_bytes = packed_row_bytes(in_pad, bits);
    let mut w = vec![0f32; out * in_dim];
    let mut g = vec![0f32; group];
    for r in 0..out {
        let prow = &packed[r * row_bytes..(r + 1) * row_bytes];
        let idx = if bits == TERNARY_RECORD_BITS {
            unpack_lsb(prow, 2, in_pad).into_iter().map(|c| if c == 3 { 0 } else { c + 1 }).collect()
        } else {
            unpack_lsb(prow, bits, in_pad)
        };
        for (gi, start) in (0..in_pad).step_by(group).enumerate() {
            let norm = norms[r * (in_pad / group) + gi].to_f32();
            for i in 0..group {
                g[i] = cb[idx[start + i] as usize] * norm;
            }
            rotate(&mut g);
            let take = group.min(in_dim.saturating_sub(start));
            w[r * in_dim + start..r * in_dim + start + take].copy_from_slice(&g[..take]);
        }
    }
    w
}

pub fn packed_row_bytes(in_pad: usize, bits: u32) -> usize {
    if bits == TERNARY_RECORD_BITS { in_pad * 2 / 8 } else { in_pad * bits as usize / 8 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_matches_dense() {
        let n = 16;
        let h = hadamard(n);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let dense: Vec<f32> = (0..n).map(|j| (0..n).map(|i| x[i] * h[i * n + j]).sum()).collect();
        let mut y = x.clone();
        rotate(&mut y);
        for (a, b) in dense.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn pack_round_trips() {
        let idx: Vec<u8> = (0..256).map(|i| (i * 7 % 16) as u8).collect();
        assert_eq!(unpack_lsb(&pack_lsb(&idx, 4), 4, 256), idx);
        let idx3: Vec<u8> = (0..256).map(|i| (i * 5 % 8) as u8).collect();
        assert_eq!(unpack_lsb(&pack_lsb(&idx3, 3), 3, 256), idx3);
    }

    #[test]
    fn nearest_ties_low() {
        let cb = [-1.0, 0.0, 1.0];
        assert_eq!(nearest(0.5, &cb), 1);
        assert_eq!(nearest(-0.5, &cb), 0);
        assert_eq!(nearest(5.0, &cb), 2);
        assert_eq!(nearest(-5.0, &cb), 0);
    }
}
