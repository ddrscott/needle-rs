//! The Monarch-Hadamard MLP, batched over rows.
//!
//! Each Kronecker factor pair applies `z -> A^T Z B` to a row viewed as an
//! `a x b` block. Over many rows that is two GEMMs with a block transpose
//! between them, which keeps the work on the matrix units instead of in
//! scalar loops. The reverse (`dZ = A dY B^T`) has the same shape.

use needle_core::config::{HADA_COND_RANK, hada_blocks};
use rayon::prelude::*;

use crate::linalg::{gemm_strided, matmul, sigmoid, silu, softmax_inplace};
use crate::weights::LayerWeights;

/// Per-row values the reverse pass needs.
#[derive(Default)]
pub struct MlpTape {
    pub sm: Vec<f32>,
    pub cond: Vec<f32>,
    pub z3: Vec<f32>,
    pub a2: Vec<f32>,
}

/// Transpose each `r x c` block of `x` (rows blocks) into `c x r`.
fn block_transpose(x: &[f32], blocks: usize, r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(r * c).zip(x.par_chunks(r * c)).for_each(|(o, b)| {
        for i in 0..r {
            for j in 0..c {
                o[j * r + i] = b[i * c + j];
            }
        }
    });
    let _ = blocks;
    out
}

/// `A^T Z B` per row (or `A Z B^T` when `transpose`), rows of `na * nb`.
pub fn kron_rows(z: &[f32], rows: usize, a: &[f32], na: usize, b: &[f32], nb: usize, transpose: bool) -> Vec<f32> {
    let n = na * nb;
    debug_assert_eq!(z.len(), rows * n);
    // M = Z B (or Z B^T): [rows*na, nb] x [nb, nb].
    let mut m = vec![0f32; rows * n];
    gemm_strided(rows * na, nb, nb, 1.0, z, nb, false, b, nb, transpose, 0.0, &mut m, nb);
    // Per row: (A^T M)^T = M^T A, (A M)^T = M^T A^T.
    let mt = block_transpose(&m, rows, na, nb);
    let mut pt = vec![0f32; rows * n];
    gemm_strided(rows * nb, na, na, 1.0, &mt, na, false, a, na, transpose, 0.0, &mut pt, na);
    block_transpose(&pt, rows, nb, na)
}

pub fn mlp_fwd(lw: &LayerWeights, perm1: &[usize], perm2: &[usize], d: usize, x: &[f32], tape: Option<&mut MlpTape>) -> Vec<f32> {
    let rows = x.len() / d;
    let n = perm1.len();
    let (ba, bb) = hada_blocks(n);
    let rk = HADA_COND_RANK;
    let mut sm = matmul(x, rows, d, &lw.cond_v, rk);
    sm.par_chunks_mut(rk).for_each(softmax_inplace);
    let mut cond = matmul(&sm, rows, rk, &lw.cond_u, n);
    cond.par_iter_mut().for_each(|c| *c += 1.0);
    let mut z1 = vec![0f32; rows * n];
    z1.par_chunks_mut(n).zip(x.par_chunks(d)).for_each(|(z, xi)| {
        for j in 0..d {
            z[j] = lw.d1[j] * xi[j];
        }
    });
    let z2 = kron_rows(&z1, rows, &lw.kron[0], ba, &lw.kron[1], bb, false);
    drop(z1);
    let mut z3 = vec![0f32; rows * n];
    let mut a2 = vec![0f32; rows * n];
    let mut z4 = vec![0f32; rows * n];
    z3.par_chunks_mut(n).zip(a2.par_chunks_mut(n)).zip(z4.par_chunks_mut(n)).enumerate().for_each(|(i, ((z3r, a2r), z4r))| {
        let z2r = &z2[i * n..(i + 1) * n];
        let cr = &cond[i * n..(i + 1) * n];
        for j in 0..n {
            z3r[j] = z2r[perm1[j]];
            a2r[j] = lw.d2[j] * cr[j] * z3r[j] + lw.b2[j];
            z4r[j] = silu(a2r[j]);
        }
    });
    drop(z2);
    let z5 = kron_rows(&z4, rows, &lw.kron[2], ba, &lw.kron[3], bb, false);
    drop(z4);
    let mut z7 = vec![0f32; rows * n];
    z7.par_chunks_mut(n).zip(z5.par_chunks(n)).for_each(|(o, z)| {
        for j in 0..n {
            o[j] = lw.d3[j] * z[perm2[j]];
        }
    });
    drop(z5);
    let z8 = kron_rows(&z7, rows, &lw.kron[4], ba, &lw.kron[5], bb, false);
    let mut out = vec![0f32; rows * d];
    out.par_chunks_mut(d).zip(z8.par_chunks(n)).for_each(|(o, z)| {
        for j in 0..d {
            o[j] = lw.d4[j] * z[j];
        }
    });
    if let Some(t) = tape {
        t.sm = sm;
        t.cond = cond;
        t.z3 = z3;
        t.a2 = a2;
    }
    out
}

/// Gradient of the MLP input given the output gradient.
pub fn mlp_bwd(lw: &LayerWeights, perm1: &[usize], perm2: &[usize], d: usize, t: &MlpTape, dout: &[f32]) -> Vec<f32> {
    let rows = dout.len() / d;
    let n = perm1.len();
    let (ba, bb) = hada_blocks(n);
    let rk = HADA_COND_RANK;
    let mut dz8 = vec![0f32; rows * n];
    dz8.par_chunks_mut(n).zip(dout.par_chunks(d)).for_each(|(o, g)| {
        for j in 0..d {
            o[j] = lw.d4[j] * g[j];
        }
    });
    let dz7 = kron_rows(&dz8, rows, &lw.kron[4], ba, &lw.kron[5], bb, true);
    drop(dz8);
    let mut dz5 = vec![0f32; rows * n];
    dz5.par_chunks_mut(n).zip(dz7.par_chunks(n)).for_each(|(o, g)| {
        for j in 0..n {
            o[perm2[j]] += lw.d3[j] * g[j];
        }
    });
    drop(dz7);
    let dz4 = kron_rows(&dz5, rows, &lw.kron[2], ba, &lw.kron[3], bb, true);
    drop(dz5);
    let mut dz2 = vec![0f32; rows * n];
    let mut dcond = vec![0f32; rows * n];
    dz2.par_chunks_mut(n).zip(dcond.par_chunks_mut(n)).enumerate().for_each(|(i, (dz2r, dcr))| {
        let (a2, z3, cond, g) =
            (&t.a2[i * n..(i + 1) * n], &t.z3[i * n..(i + 1) * n], &t.cond[i * n..(i + 1) * n], &dz4[i * n..(i + 1) * n]);
        for j in 0..n {
            let s = sigmoid(a2[j]);
            let da = g[j] * s * (1.0 + a2[j] * (1.0 - s));
            dz2r[perm1[j]] += da * lw.d2[j] * cond[j];
            dcr[j] = da * lw.d2[j] * z3[j];
        }
    });
    drop(dz4);
    let dz1 = kron_rows(&dz2, rows, &lw.kron[0], ba, &lw.kron[1], bb, true);
    drop(dz2);
    // Conditioning path: dsm = dcond cond_u^T, softmax reverse, dx += dcl cond_v^T.
    let mut dsm = vec![0f32; rows * rk];
    gemm_strided(rows, rk, n, 1.0, &dcond, n, false, &lw.cond_u, n, true, 0.0, &mut dsm, rk);
    dsm.par_chunks_mut(rk).zip(t.sm.par_chunks(rk)).for_each(|(g, s)| {
        let sdot: f32 = s.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
        for (gi, si) in g.iter_mut().zip(s) {
            *gi = si * (*gi - sdot);
        }
    });
    let mut dx = vec![0f32; rows * d];
    dx.par_chunks_mut(d).zip(dz1.par_chunks(n)).for_each(|(o, g)| {
        for j in 0..d {
            o[j] = lw.d1[j] * g[j];
        }
    });
    gemm_strided(rows, d, rk, 1.0, &dsm, rk, false, &lw.cond_v, rk, true, 1.0, &mut dx, d);
    dx
}
