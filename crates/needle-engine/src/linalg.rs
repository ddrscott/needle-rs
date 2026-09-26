//! Dense kernels. Matrix products go to Accelerate (AMX) on macOS and to
//! `matrixmultiply` elsewhere; the rest are small fused loops.

/// `y[t, o] = sum_i x[t, i] * w[o, i]` — activations `[t, n_in]` against a
/// weight stored `[n_out, n_in]` (rows contiguous along the reduction).
pub fn matmul_t(x: &[f32], t: usize, n_in: usize, w: &[f32], n_out: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * n_out];
    matmul_t_into(x, t, n_in, w, n_out, &mut y, false);
    y
}

/// As [`matmul_t`], writing into `y` (accumulating when `accumulate`).
pub fn matmul_t_into(x: &[f32], t: usize, n_in: usize, w: &[f32], n_out: usize, y: &mut [f32], accumulate: bool) {
    debug_assert_eq!(x.len(), t * n_in);
    debug_assert_eq!(w.len(), n_out * n_in);
    gemm(Op { m: t, n: n_out, k: n_in, a: x, ta: false, b: w, tb: true }, y, accumulate);
}

/// `y[t, o] = sum_i x[t, i] * w[i, o]` — weight stored `[n_in, n_out]`.
pub fn matmul(x: &[f32], t: usize, n_in: usize, w: &[f32], n_out: usize) -> Vec<f32> {
    let mut y = vec![0f32; t * n_out];
    matmul_into(x, t, n_in, w, n_out, &mut y, false);
    y
}

pub fn matmul_into(x: &[f32], t: usize, n_in: usize, w: &[f32], n_out: usize, y: &mut [f32], accumulate: bool) {
    debug_assert_eq!(x.len(), t * n_in);
    debug_assert_eq!(w.len(), n_in * n_out);
    gemm(Op { m: t, n: n_out, k: n_in, a: x, ta: false, b: w, tb: false }, y, accumulate);
}

/// `a^T b` for `a [n, p]`, `b [n, q]` -> `[p, q]`.
pub fn matmul_tn_into(a: &[f32], n: usize, p: usize, b: &[f32], q: usize, c: &mut [f32], accumulate: bool) {
    debug_assert_eq!(a.len(), n * p);
    debug_assert_eq!(b.len(), n * q);
    gemm(Op { m: p, n: q, k: n, a, ta: true, b, tb: false }, c, accumulate);
}

/// `C = alpha op(A) op(B) + beta C` with explicit leading dimensions, for
/// operating on strided sub-matrices (a head's columns inside a row).
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    ta: bool,
    b: &[f32],
    ldb: usize,
    tb: bool,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for v in &mut c[i * ldc..i * ldc + n] {
                *v *= beta;
            }
        }
        return;
    }
    #[cfg(target_os = "macos")]
    accelerate::sgemm_ld(m, n, k, alpha, a, lda, ta, b, ldb, tb, beta, c, ldc);
    #[cfg(not(target_os = "macos"))]
    {
        let (a_rs, a_cs) = if ta { (1, lda) } else { (lda, 1) };
        let (b_rs, b_cs) = if tb { (1, ldb) } else { (ldb, 1) };
        // SAFETY: callers pass operands covering the strided extents.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a.as_ptr(),
                a_rs as isize,
                a_cs as isize,
                b.as_ptr(),
                b_rs as isize,
                b_cs as isize,
                beta,
                c.as_mut_ptr(),
                ldc as isize,
                1,
            );
        }
    }
}

/// `C[m, n] (+)= op(A) op(B)` with row-major operands; `op(A)` is `[m, k]`
/// (stored `[k, m]` when `ta`), `op(B)` is `[k, n]` (stored `[n, k]` when `tb`).
struct Op<'a> {
    m: usize,
    n: usize,
    k: usize,
    a: &'a [f32],
    ta: bool,
    b: &'a [f32],
    tb: bool,
}

fn gemm(op: Op<'_>, c: &mut [f32], accumulate: bool) {
    let Op { m, n, k, a, ta, b, tb } = op;
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            c.fill(0.0);
        }
        return;
    }
    #[cfg(target_os = "macos")]
    accelerate::sgemm(m, n, k, a, ta, b, tb, c, accumulate);
    #[cfg(not(target_os = "macos"))]
    {
        let (a_rs, a_cs) = if ta { (1, m) } else { (k, 1) };
        let (b_rs, b_cs) = if tb { (1, k) } else { (n, 1) };
        // SAFETY: operand lengths are checked against m, n, k above.
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                a_rs as isize,
                a_cs as isize,
                b.as_ptr(),
                b_rs as isize,
                b_cs as isize,
                if accumulate { 1.0 } else { 0.0 },
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

#[cfg(target_os = "macos")]
mod accelerate {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_ld(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        ta: bool,
        b: &[f32],
        ldb: usize,
        tb: bool,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) {
        let a_need = if ta { (k - 1) * lda + m } else { (m - 1) * lda + k };
        let b_need = if tb { (n - 1) * ldb + k } else { (k - 1) * ldb + n };
        assert!(a.len() >= a_need && b.len() >= b_need && c.len() >= (m - 1) * ldc + n);
        // SAFETY: extents asserted above.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                if ta { TRANS } else { NO_TRANS },
                if tb { TRANS } else { NO_TRANS },
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                ldc as i32,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(m: usize, n: usize, k: usize, a: &[f32], ta: bool, b: &[f32], tb: bool, c: &mut [f32], accumulate: bool) {
        assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
        let lda = if ta { m } else { k };
        let ldb = if tb { k } else { n };
        // SAFETY: operand lengths were asserted against m, n, k.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                if ta { TRANS } else { NO_TRANS },
                if tb { TRANS } else { NO_TRANS },
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                if accumulate { 1.0 } else { 0.0 },
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent accumulators keep the FP pipeline and NEON lanes busy.
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        let (x, y) = (&a[c * 8..c * 8 + 8], &b[c * 8..c * 8 + 8]);
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// `exp` to about 1 ulp (Cephes `expf`), branch-free so loops over slices
/// vectorize; libm's scalar call dominated decode's elementwise work.
#[inline(always)]
pub fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.33654, 88.72283);
    let k = (x * LOG2E).round();
    let r = x - k * C1 - k * C2;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5e-1;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((k as i32 + 127) as u32) << 23)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp(-x))
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `x * rsqrt(mean(x^2) + eps)` (the `_rms_unit` of the reference).
#[inline]
pub fn rms_unit(x: &[f32], out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + 1e-6).sqrt();
    for (o, v) in out.iter_mut().zip(x) {
        *o = v * r;
    }
}

/// `ZCRMSNorm`: `(1 + scale) * x / sqrt(mean(x^2) + eps)`, in place.
#[inline]
pub fn zc_rms_norm(x: &mut [f32], scale: &[f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + 1e-6).sqrt();
    for (v, s) in x.iter_mut().zip(scale) {
        *v = (1.0 + s) * *v * inv;
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if m == f32::NEG_INFINITY {
        x.fill(0.0);
        return;
    }
    for v in x.iter_mut() {
        *v = if *v == f32::NEG_INFINITY { 0.0 } else { fast_exp(*v - m) };
    }
    let s: f32 = x.iter().sum();
    let inv = 1.0 / s;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

pub fn logsumexp(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    m + x.iter().map(|v| (v - m).exp()).sum::<f32>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_exp_is_accurate() {
        let mut worst = 0f32;
        let mut x = -87.0f32;
        while x < 88.0 {
            let (a, b) = (fast_exp(x), x.exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst relative error {worst}");
    }

    #[test]
    fn matmul_variants_agree() {
        let (t, i, o) = (3, 5, 4);
        let x: Vec<f32> = (0..t * i).map(|v| v as f32 * 0.1).collect();
        let w: Vec<f32> = (0..o * i).map(|v| (v as f32 * 0.37).sin()).collect();
        let y = matmul_t(&x, t, i, &w, o);
        let mut wt = vec![0f32; i * o];
        for r in 0..o {
            for c in 0..i {
                wt[c * o + r] = w[r * i + c];
            }
        }
        let y2 = matmul(&x, t, i, &wt, o);
        for a in 0..t {
            for b in 0..o {
                let want: f32 = (0..i).map(|k| x[a * i + k] * w[b * i + k]).sum();
                assert!((y[a * o + b] - want).abs() < 1e-5);
                assert!((y2[a * o + b] - want).abs() < 1e-5);
            }
        }
    }
}
