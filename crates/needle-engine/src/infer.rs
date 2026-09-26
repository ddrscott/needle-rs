//! The packed-weight forward pass, reproducing libneedle3 bit for bit.
//!
//! Every float operation here follows the native engine's order, rounding
//! and fusion (verified stage by stage against buffers dumped from the
//! running library): int8 attention with 21-bit fixed-point softmax
//! weights, an int8 mHC gate projection, int8 engram value history, f64
//! Sinkhorn, the engine's inline `exp`, RoPE advanced by recurrence while
//! decoding, and four-accumulator reductions. The call pattern matters too:
//! feeds are cut into chunks of at most 32 tokens from the feed start, a
//! single token takes the decode kernels, and attention inside a chunk sees
//! the ring as of the chunk's end.
//!
//! Work is split across the decode team per operation (matmul rows,
//! attention heads, chunk rows); every value is still reduced by one fixed
//! code path, so results do not depend on the team size.

use half::f16;
use needle_core::config::{ENGRAM_CONV_TAPS, ENGRAM_PRIME, ENGRAM_SEED, HADA_COND_RANK};

use crate::model::{EngramKv, ForwardOut, Model, Outputs, ROLLBACK, Session};
use crate::qlinear::{GROUP, QAct, quant_i8};
use crate::team::{SyncPtr, share, team};

mod chunk;
mod step;

// // NUMERICS

/// `1/sqrtf(48)`, the query scale folded into its int8 scale.
const INV_SQRT_QK: f32 = f32::from_bits(0x3e13cd3a);
/// Softmax weights quantize to 21 bits: `w * (2^21 - 1) / max`.
const Q21: f32 = f32::from_bits(0x49fffff8);
/// `1 / (2^21 - 1)`, rounded as the engine has it.
const C21: f32 = f32::from_bits(0x35000004);
/// Masked logits.
pub const NEG: f32 = f32::from_bits(0xf149f2ca);
/// The engine's team size, which fixes decode attention's key chunking.
const NATIVE_TEAM: usize = 4;

/// The system libm the engine links against (Apple's).
#[cfg(target_vendor = "apple")]
mod sys {
    #[repr(C)]
    struct SinCos {
        sin: f32,
        cos: f32,
    }

    unsafe extern "C" {
        fn expf(x: f32) -> f32;
        fn exp(x: f64) -> f64;
        fn powf(x: f32, y: f32) -> f32;
        fn __sincosf_stret(x: f32) -> SinCos;
    }

    // SAFETY (all four): pure libm functions.
    #[inline]
    pub fn lexpf(x: f32) -> f32 {
        unsafe { expf(x) }
    }
    #[inline]
    pub fn lexp(x: f64) -> f64 {
        unsafe { exp(x) }
    }
    #[inline]
    pub fn lpowf(x: f32, y: f32) -> f32 {
        unsafe { powf(x, y) }
    }
    #[inline]
    pub fn sincosf(x: f32) -> (f32, f32) {
        let r = unsafe { __sincosf_stret(x) };
        (r.sin, r.cos)
    }
}

/// The musl port, for targets without Apple's libm (the browser among
/// them). A result can differ from Apple's in the last bit.
#[cfg(not(target_vendor = "apple"))]
mod sys {
    #[inline]
    pub fn lexpf(x: f32) -> f32 {
        libm::expf(x)
    }
    #[inline]
    pub fn lexp(x: f64) -> f64 {
        libm::exp(x)
    }
    #[inline]
    pub fn lpowf(x: f32, y: f32) -> f32 {
        libm::powf(x, y)
    }
    #[inline]
    pub fn sincosf(x: f32) -> (f32, f32) {
        libm::sincosf(x)
    }
}

/// libm `expf` (the engine's scalar exp).
use sys::lexpf;
use sys::{lexp, lpowf, sincosf};

/// WebAssembly SIMD helpers. Wasm has no fused multiply-add, and the engine's
/// arithmetic is fma throughout, so [`wsimd::fma4`] computes it exactly: an
/// f32 product is exact in f64, the f64 sum rounds once, and narrowing to
/// f32 rounds again, which differs from a single rounding only when the f64
/// sum sits exactly on an f32 midpoint (or narrows to an f32 subnormal).
/// Those lanes, a few in a billion, redo the fma in software.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wsimd {
    use std::arch::wasm32::*;

    #[inline(always)]
    pub fn splat(x: f32) -> v128 {
        f32x4_splat(x)
    }

    /// # Safety
    /// Four floats readable at `p`.
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> v128 {
        unsafe { v128_load(p.cast()) }
    }

    /// # Safety
    /// Four floats writable at `p`.
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: v128) {
        unsafe { v128_store(p.cast(), v) }
    }

    /// Four f16 bit patterns (finite) widened exactly: the magnitude bits
    /// moved into f32 position and scaled by 2^112, which also covers
    /// subnormals.
    /// # Safety
    /// Four halves readable at `p`.
    #[inline(always)]
    pub unsafe fn load_f16(p: *const u16) -> v128 {
        unsafe {
            let h = u32x4_extend_low_u16x8(v128_load64_zero(p.cast()));
            let sign = i32x4_shl(v128_and(h, u32x4_splat(0x8000)), 16);
            let mag = f32x4_mul(i32x4_shl(v128_and(h, u32x4_splat(0x7fff)), 13), f32x4_splat(f32::from_bits(0x7780_0000)));
            v128_or(mag, sign)
        }
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)]
    fn needs_software(r: v128) -> bool {
        let half = i64x2_eq(v128_and(r, i64x2_splat(0x1fff_ffff)), i64x2_splat(0x1000_0000));
        let e = u64x2_shr(v128_and(r, i64x2_splat(0x7ff0_0000_0000_0000)), 52);
        // f32 subnormal range, less f64 zero and subnormals (which narrow
        // to the same zero either way).
        let tiny = v128_and(i64x2_lt(e, i64x2_splat(1023 - 126)), i64x2_gt(e, i64x2_splat(0)));
        v128_any_true(v128_or(half, tiny))
    }

    /// `fma(a, b, c)` on four lanes, correctly rounded: the hardware's
    /// fused multiply-add. The page loads this build only after
    /// [`madd_is_fused`] says the browser really fuses it.
    #[cfg(target_feature = "relaxed-simd")]
    #[inline(always)]
    pub fn fma4(a: v128, b: v128, c: v128) -> v128 {
        f32x4_relaxed_madd(a, b, c)
    }

    /// Whether `relaxed_madd` rounds once: a sum that sits just under an
    /// f32 midpoint comes out right only when the product is not rounded
    /// first.
    #[cfg(target_feature = "relaxed-simd")]
    pub fn madd_is_fused() -> bool {
        let (a, b, c) = (f32::from_bits(0x3f80_0001), f32::from_bits(0x337f_fffe), f32::from_bits(0x3f80_0001));
        let got = f32x4_extract_lane::<0>(f32x4_relaxed_madd(f32x4_splat(a), f32x4_splat(b), f32x4_splat(c)));
        // `black_box` keeps the check from being folded at compile time.
        got.to_bits() == std::hint::black_box(a).mul_add(std::hint::black_box(b), std::hint::black_box(c)).to_bits()
    }

    /// `fma(a, b, c)` on four lanes, correctly rounded.
    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)]
    pub fn fma4(a: v128, b: v128, c: v128) -> v128 {
        let hi = |v: v128| i32x4_shuffle::<2, 3, 0, 1>(v, v);
        let lo_r = f64x2_add(f64x2_mul(f64x2_promote_low_f32x4(a), f64x2_promote_low_f32x4(b)), f64x2_promote_low_f32x4(c));
        let hi_r = f64x2_add(f64x2_mul(f64x2_promote_low_f32x4(hi(a)), f64x2_promote_low_f32x4(hi(b))), f64x2_promote_low_f32x4(hi(c)));
        if needs_software(lo_r) || needs_software(hi_r) {
            return slow(a, b, c);
        }
        i32x4_shuffle::<0, 1, 4, 5>(f32x4_demote_f64x2_zero(lo_r), f32x4_demote_f64x2_zero(hi_r))
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    #[cold]
    #[inline(never)]
    fn slow(a: v128, b: v128, c: v128) -> v128 {
        let (a, b, c) = (lanes(a), lanes(b), lanes(c));
        f32x4(a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1]), a[2].mul_add(b[2], c[2]), a[3].mul_add(b[3], c[3]))
    }

    #[inline(always)]
    pub fn lanes(v: v128) -> [f32; 4] {
        [f32x4_extract_lane::<0>(v), f32x4_extract_lane::<1>(v), f32x4_extract_lane::<2>(v), f32x4_extract_lane::<3>(v)]
    }

    /// Round half away from zero (NEON's `fcvtas`, `f32::round`): the
    /// fraction `x - trunc(x)` is exact, and a half or more steps out.
    #[inline(always)]
    pub fn round_away(x: v128) -> v128 {
        let t = f32x4_trunc(x);
        // `+-1` where the fraction is a half or more, else a zero of `x`'s
        // sign (so `-0.3` stays `-0.0`).
        let sign = v128_and(x, u32x4_splat(0x8000_0000));
        let step = v128_or(sign, v128_and(f32x4_ge(f32x4_abs(f32x4_sub(x, t)), f32x4_splat(0.5)), f32x4_splat(1.0)));
        f32x4_add(t, step)
    }

    /// [`super::nexp`] on four lanes (NEON's `nexp4`).
    #[inline(always)]
    pub fn nexp4(x: v128) -> v128 {
        let c = |b: u32| f32x4_splat(f32::from_bits(b));
        let x = f32x4_min(f32x4_max(x, c(0xc2b0c0a5)), c(0x42b0c0a5));
        let k = f32x4_nearest(f32x4_mul(x, c(0x3fb8aa3b)));
        let r = fma4(k, c(0xbf318000), x);
        let r = fma4(k, c(0x395e8083), r);
        let mut p = fma4(c(0x39506967), r, c(0x3ab743ce));
        p = fma4(r, p, c(0x3c088908));
        p = fma4(r, p, c(0x3d2aa9c1));
        p = fma4(r, p, c(0x3e2aaaaa));
        p = fma4(r, p, f32x4_splat(0.5));
        let y = fma4(f32x4_mul(r, r), p, r);
        let s = i32x4_add(i32x4_shl(i32x4_trunc_sat_f32x4(k), 23), i32x4_splat(0x3f80_0000));
        fma4(s, y, s)
    }
}

/// Check the wasm SIMD helpers against their scalar definitions (run from
/// the browser build; returns a report).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[doc(hidden)]
pub fn wasm_simd_selftest(n: usize) -> String {
    use std::arch::wasm32::*;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Random f32s with exponents spread over most of the range.
    let mut rnd = |spread: u32| {
        let r = next();
        let exp = 127u32.wrapping_add((r as u32 % (2 * spread + 1)).wrapping_sub(spread)) & 0xff;
        f32::from_bits(((r >> 32) as u32 & 0x807f_ffff) | (exp.clamp(1, 254) << 23))
    };
    let (mut bad_fma, mut slow_hits) = (0usize, 0usize);
    for i in 0..n {
        let spread = [3, 20, 60, 126][i % 4];
        let (a, b, c) = (
            [rnd(spread), rnd(spread), rnd(spread), rnd(spread)],
            [rnd(spread), rnd(spread), rnd(spread), rnd(spread)],
            [rnd(spread), rnd(spread), rnd(spread), rnd(spread)],
        );
        let v = wsimd::fma4(f32x4(a[0], a[1], a[2], a[3]), f32x4(b[0], b[1], b[2], b[3]), f32x4(c[0], c[1], c[2], c[3]));
        for (k, got) in wsimd::lanes(v).into_iter().enumerate() {
            let want = a[k].mul_add(b[k], c[k]);
            if got.to_bits() != want.to_bits() && !(got.is_nan() && want.is_nan()) {
                bad_fma += 1;
            }
        }
    }
    // Midpoints: `a * b` exact in f32 terms plus a `c` that lands the f64
    // sum on a halfway point, then nudged by a tiny opposite term.
    for i in 0..n.min(1 << 20) {
        // `(1 + 2^-23) * 2^-24 (1 - 2^-23) = 2^-24 - 2^-70`: added to an odd
        // `c` the exact sum sits just under a midpoint, and the f64 sum on it.
        let a = f32::from_bits(0x3f80_0001);
        let b = f32::from_bits(0x337f_fffe);
        let c = f32::from_bits(0x3f80_0001 | ((i as u32 & 0x003f_ffff) << 1));
        let (b, c) = if i % 2 == 0 { (b, c) } else { (-b, -c) };
        let v = wsimd::fma4(f32x4_splat(a), f32x4_splat(b), f32x4_splat(c));
        let want = a.mul_add(b, c);
        if wsimd::lanes(v)[0].to_bits() != want.to_bits() {
            bad_fma += 1;
        }
        let lo = f64::from(a) * f64::from(b) + f64::from(c);
        if (lo.to_bits() & 0x1fff_ffff) == 0x1000_0000 {
            slow_hits += 1;
        }
    }
    let mut bad_f16 = 0usize;
    for h in 0..=u16::MAX {
        let x = half::f16::from_bits(h);
        if !x.is_finite() {
            continue;
        }
        // SAFETY: four halves on the stack.
        let v = unsafe { wsimd::load_f16([h, h, h, h].as_ptr()) };
        if wsimd::lanes(v)[0].to_bits() != x.to_f32().to_bits() {
            bad_f16 += 1;
        }
    }
    let mut bad_exp = 0usize;
    for i in 0..n {
        let x = (i as f32 / n as f32) * 200.0 - 100.0 + f32::from_bits(next() as u32 & 0x3fff_ffff) * 1e-6;
        if wsimd::lanes(wsimd::nexp4(f32x4_splat(x)))[0].to_bits() != nexp(x).to_bits() {
            bad_exp += 1;
        }
    }
    let mut bad_round = 0usize;
    for i in 0..n {
        let x = match i % 3 {
            0 => (i as f32 - n as f32 / 2.0) * 0.5,
            1 => f32::from_bits(next() as u32 & 0xcfff_ffff),
            _ => (next() as u32 as f32 / u32::MAX as f32 - 0.5) * 4.0e6,
        };
        if wsimd::lanes(wsimd::round_away(f32x4_splat(x)))[0].to_bits() != x.round().to_bits() && !x.is_nan() {
            bad_round += 1;
        }
    }
    let mut bad_idot = 0usize;
    for len in [16usize, 48, 128, 37] {
        for _ in 0..(n / 64).max(1) {
            let a: Vec<i8> = (0..len).map(|_| next() as i8).collect();
            let b: Vec<i8> = (0..len).map(|_| next() as i8).collect();
            let want: i32 = a.iter().zip(&b).map(|(&x, &y)| x as i32 * y as i32).sum();
            if idot(&a, &b) != want {
                bad_idot += 1;
            }
        }
    }
    format!(
        "fma4 mismatches {bad_fma} of {} (midpoint lanes seen {slow_hits}); f16 mismatches {bad_f16}; nexp4 mismatches {bad_exp} of {n}; round mismatches {bad_round}; idot mismatches {bad_idot}",
        n * 4 + n.min(1 << 20)
    )
}

/// The engine's inline vector exp: clamp, `k = rint(x log2 e)`, a two-part
/// reduction, a degree-6 polynomial, all fused.
#[inline]
pub(crate) fn nexp(x: f32) -> f32 {
    let x = x.max(f32::from_bits(0xc2b0c0a5)).min(f32::from_bits(0x42b0c0a5));
    let k = (x * f32::from_bits(0x3fb8aa3b)).round_ties_even();
    let r = k.mul_add(f32::from_bits(0xbf318000), x);
    let r = k.mul_add(f32::from_bits(0x395e8083), r);
    let mut p = f32::from_bits(0x39506967).mul_add(r, f32::from_bits(0x3ab743ce));
    p = r.mul_add(p, f32::from_bits(0x3c088908));
    p = r.mul_add(p, f32::from_bits(0x3d2aa9c1));
    p = r.mul_add(p, f32::from_bits(0x3e2aaaaa));
    p = r.mul_add(p, 0.5);
    let y = (r * r).mul_add(p, r);
    let s = f32::from_bits(((k as i32) << 23).wrapping_add(0x3f80_0000) as u32);
    s.mul_add(y, s)
}

/// `sum x*y` over a multiple of 16 elements the engine's way: four 4-lane
/// fma accumulators over 16-element chunks, combined `((A2+A3)+A1)+A0`,
/// then `(l0+l1)+(l2+l3)`.
#[inline]
fn dot16(x: &[f32], y: &[f32]) -> f32 {
    debug_assert!(x.len().is_multiple_of(16) && x.len() == y.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; loads stay inside the slices.
    unsafe {
        use std::arch::aarch64::*;
        let mut a = [vdupq_n_f32(0.0); 4];
        for c in 0..x.len() / 16 {
            for (j, aj) in a.iter_mut().enumerate() {
                let i = c * 16 + j * 4;
                *aj = vfmaq_f32(*aj, vld1q_f32(x.as_ptr().add(i)), vld1q_f32(y.as_ptr().add(i)));
            }
        }
        let t = vaddq_f32(vaddq_f32(vaddq_f32(a[2], a[3]), a[1]), a[0]);
        let p = vpaddq_f32(t, t);
        vgetq_lane_f32(p, 0) + vgetq_lane_f32(p, 1)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: loads stay inside the slices (whole 16-element chunks).
    unsafe {
        use std::arch::wasm32::*;
        let mut a = [f32x4_splat(0.0); 4];
        for c in 0..x.len() / 16 {
            for (j, aj) in a.iter_mut().enumerate() {
                let i = c * 16 + j * 4;
                *aj = wsimd::fma4(wsimd::load(x.as_ptr().add(i)), wsimd::load(y.as_ptr().add(i)), *aj);
            }
        }
        let t = wsimd::lanes(f32x4_add(f32x4_add(f32x4_add(a[2], a[3]), a[1]), a[0]));
        (t[0] + t[1]) + (t[2] + t[3])
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let mut a = [[0f32; 4]; 4];
        for c in 0..x.len() / 16 {
            for (j, aj) in a.iter_mut().enumerate() {
                for (l, al) in aj.iter_mut().enumerate() {
                    let i = c * 16 + j * 4 + l;
                    *al = x[i].mul_add(y[i], *al);
                }
            }
        }
        let t: [f32; 4] = std::array::from_fn(|l| ((a[2][l] + a[3][l]) + a[1][l]) + a[0][l]);
        (t[0] + t[1]) + (t[2] + t[3])
    }
}

/// `1 / sqrtf(ss / n + 1e-6)`.
#[inline]
fn rinv(ss: f32, n: usize) -> f32 {
    1.0 / (ss / n as f32 + 1e-6).sqrt()
}

/// Zero-centered RMSNorm: `t = x * r`, `out = fma(t, s, t)`.
fn zcn(x: &[f32], s: &[f32], out: &mut [f32]) {
    let r = rinv(dot16(x, x), x.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len().is_multiple_of(4) {
        // SAFETY: whole 4-element blocks of equal-length slices.
        unsafe {
            for i in (0..x.len()).step_by(4) {
                let t = std::arch::wasm32::f32x4_mul(wsimd::load(x.as_ptr().add(i)), wsimd::splat(r));
                wsimd::store(out.as_mut_ptr().add(i), wsimd::fma4(t, wsimd::load(s.as_ptr().add(i)), t));
            }
        }
        return;
    }
    for ((o, &v), &s) in out.iter_mut().zip(x).zip(s) {
        let t = v * r;
        *o = t.mul_add(s, t);
    }
}

/// [`zcn`] in place.
fn zcn_inplace(x: &mut [f32], s: &[f32]) {
    let r = rinv(dot16(x, x), x.len());
    for (v, &s) in x.iter_mut().zip(s) {
        let t = *v * r;
        *v = t.mul_add(s, t);
    }
}

/// [`zcn`] with the scale as the archive stores it (f16, widened per use).
fn zcn16(x: &[f32], s: &[f16], out: &mut [f32]) {
    let r = rinv(dot16(x, x), x.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len().is_multiple_of(4) {
        // SAFETY: whole 4-element blocks of equal-length slices.
        unsafe {
            for i in (0..x.len()).step_by(4) {
                let t = std::arch::wasm32::f32x4_mul(wsimd::load(x.as_ptr().add(i)), wsimd::splat(r));
                wsimd::store(out.as_mut_ptr().add(i), wsimd::fma4(t, wsimd::load_f16(s.as_ptr().add(i).cast()), t));
            }
        }
        return;
    }
    for ((o, &v), &s) in out.iter_mut().zip(x).zip(s) {
        let t = v * r;
        *o = t.mul_add(s.to_f32(), t);
    }
}

/// [`zcn16`] in place.
fn zcn16_inplace(x: &mut [f32], s: &[f16]) {
    let r = rinv(dot16(x, x), x.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len().is_multiple_of(4) {
        // SAFETY: whole 4-element blocks of equal-length slices.
        unsafe {
            for i in (0..x.len()).step_by(4) {
                let t = std::arch::wasm32::f32x4_mul(wsimd::load(x.as_ptr().add(i)), wsimd::splat(r));
                wsimd::store(x.as_mut_ptr().add(i), wsimd::fma4(t, wsimd::load_f16(s.as_ptr().add(i).cast()), t));
            }
        }
        return;
    }
    for (v, &s) in x.iter_mut().zip(s) {
        let t = *v * r;
        *v = t.mul_add(s.to_f32(), t);
    }
}

/// `acc[i] = fma(u[i], w, acc[i])` with `u` stored f16.
#[inline]
fn axpy16(acc: &mut [f32], u: &[f16], w: f32) {
    debug_assert_eq!(acc.len(), u.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON with fp16 conversion is baseline on Apple silicon; the
    // loop stays inside the slices.
    unsafe {
        use std::arch::aarch64::*;
        let n = acc.len() / 4 * 4;
        for i in (0..n).step_by(4) {
            let a = vld1q_f32(acc.as_ptr().add(i));
            vst1q_f32(acc.as_mut_ptr().add(i), vfmaq_n_f32(a, u.load4(i), w));
        }
        for i in n..acc.len() {
            acc[i] = u[i].to_f32().mul_add(w, acc[i]);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: whole 4-element blocks, then a scalar tail.
    unsafe {
        let n = acc.len() / 4 * 4;
        for i in (0..n).step_by(4) {
            let a = wsimd::load(acc.as_ptr().add(i));
            wsimd::store(acc.as_mut_ptr().add(i), wsimd::fma4(wsimd::load_f16(u.as_ptr().add(i).cast()), wsimd::splat(w), a));
        }
        for i in n..acc.len() {
            acc[i] = u[i].to_f32().mul_add(w, acc[i]);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    for (a, &u) in acc.iter_mut().zip(u) {
        *a = u.to_f32().mul_add(w, *a);
    }
}

/// A row source for the kernels: f32, or the archive's f16 widened on
/// load (exact, so the arithmetic is the same).
pub(crate) trait Src: Sync {
    #[allow(dead_code)]
    fn at(&self, i: usize) -> f32;
    /// # Safety
    /// Elements `i..i + 4` exist.
    #[cfg(target_arch = "aarch64")]
    unsafe fn load4(&self, i: usize) -> std::arch::aarch64::float32x4_t;
    /// # Safety
    /// Elements `i..i + 4` exist.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe fn load4w(&self, i: usize) -> std::arch::wasm32::v128;
}

impl Src for [f32] {
    #[inline(always)]
    fn at(&self, i: usize) -> f32 {
        self[i]
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load4(&self, i: usize) -> std::arch::aarch64::float32x4_t {
        unsafe { std::arch::aarch64::vld1q_f32(self.as_ptr().add(i)) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline(always)]
    unsafe fn load4w(&self, i: usize) -> std::arch::wasm32::v128 {
        unsafe { wsimd::load(self.as_ptr().add(i)) }
    }
}

impl Src for [f16] {
    #[inline(always)]
    fn at(&self, i: usize) -> f32 {
        self[i].to_f32()
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn load4(&self, i: usize) -> std::arch::aarch64::float32x4_t {
        unsafe {
            use std::arch::aarch64::*;
            vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(self.as_ptr().add(i).cast())))
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline(always)]
    unsafe fn load4w(&self, i: usize) -> std::arch::wasm32::v128 {
        unsafe { wsimd::load_f16(self.as_ptr().add(i).cast()) }
    }
}

/// Integer dot of two int8 rows.
#[inline]
fn idot(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: dotprod detected; loads stay inside the slices.
        return unsafe { neon::idot(a, b) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use std::arch::wasm32::*;
        let n = a.len() / 16 * 16;
        let mut acc = i32x4_splat(0);
        // SAFETY: whole 16-byte blocks inside both slices.
        unsafe {
            for i in (0..n).step_by(16) {
                let (x, y) = (v128_load(a.as_ptr().add(i).cast()), v128_load(b.as_ptr().add(i).cast()));
                acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(x, y)));
                acc = i32x4_add(acc, i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(x, y)));
            }
        }
        let tail: i32 = a[n..].iter().zip(&b[n..]).map(|(&x, &y)| x as i32 * y as i32).sum();
        return i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc)
            + tail;
    }
    #[allow(unreachable_code)]
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Integer dots of the int8 query against the keys in `slots`.
#[inline]
fn key_dots(q8: &[i8], k8: &[i8], slots: &[usize], nkv: usize, kvh: usize, qk: usize, out: &mut [i32]) {
    #[cfg(target_arch = "aarch64")]
    if qk == 48 && std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: dotprod detected; every key row is `qk` bytes in `k8`.
        return unsafe { neon::key_dots48(q8, k8, slots, nkv, kvh, out) };
    }
    for (o, &slot) in out.iter_mut().zip(slots) {
        *o = idot(q8, &k8[(slot * nkv + kvh) * qk..(slot * nkv + kvh + 1) * qk]);
    }
}

/// The three digit dots of four value rows over one run of `l` slots from
/// `s0`: `[row][hi, mid, lo]`.
#[inline]
fn pv_dots4(hi: &[i8], mid: &[i8], lo: &[i8], rows: [&[i8]; 4], s0: usize) -> [[i32; 3]; 4] {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: dotprod detected; each row covers `s0 + hi.len()` slots.
        return unsafe { neon::pv_dots4(hi, mid, lo, rows, s0) };
    }
    rows.map(|r| {
        let (a, b, c) = idot3(hi, mid, lo, &r[s0..s0 + hi.len()]);
        [a, b, c]
    })
}

/// Three integer dots against one int8 row (the P.V digits).
#[inline]
fn idot3(a: &[i8], b: &[i8], c: &[i8], v: &[i8]) -> (i32, i32, i32) {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        // SAFETY: dotprod detected; loads stay inside the slices.
        return unsafe { neon::idot3(a, b, c, v) };
    }
    (idot(a, v), idot(b, v), idot(c, v))
}

/// Run `f(i)` for `i in 0..n` on the team (inline when `n == 1`).
fn par(n: usize, f: &(dyn Fn(usize) + Sync)) {
    if n <= 1 {
        (0..n).for_each(f);
    } else {
        team().run(&|tid, nt| share(n, tid, nt).for_each(f));
    }
}

// // STATE

/// The mHC gate projection as the engine keeps it: each dequantized row
/// requantized to int8 with its own scale.
pub(crate) struct Phi8 {
    rows: usize,
    width: usize,
    q: Vec<i8>,
    s: Vec<f32>,
}

impl Phi8 {
    pub(crate) fn new(w: &[f32], rows: usize) -> Self {
        let width = w.len() / rows;
        let mut q = vec![0i8; w.len()];
        let s = (0..rows).map(|r| quant_i8(&w[r * width..(r + 1) * width], &mut q[r * width..(r + 1) * width])).collect();
        Self { rows, width, q, s }
    }
}

/// Per-model tables of the native path.
pub(crate) struct Tables {
    /// RoPE frequencies `powf(theta, i * (-2/qk))` and their `sincosf`.
    freq: Vec<f32>,
    sd: Vec<f32>,
    cd: Vec<f32>,
    pub(crate) phi: Vec<Phi8>,
    /// The small per-layer weights as the archive stores them.
    pub(crate) layers: Vec<LayerF16>,
    /// Whether every one of them is exactly an f16 (an archive's are; a
    /// checkpoint's need not be, and then the packed path is not exact).
    pub(crate) exact: bool,
}

/// One layer's small weights in f16, widened where they are used: half the
/// bytes each member streams per layer, the same values.
pub(crate) struct LayerF16 {
    pub kron: [Vec<f16>; 6],
    pub cond_v: Vec<f16>,
    pub cond_u: Vec<f16>,
    pub d1: Vec<f16>,
    pub d2: Vec<f16>,
    pub b2: Vec<f16>,
    pub d3: Vec<f16>,
    pub d4: Vec<f16>,
    pub norm_in: Vec<f16>,
    pub post_norm: Vec<f16>,
    pub pre_hada: Vec<f16>,
    pub q_taps: Vec<f16>,
    pub k_taps: Vec<f16>,
    pub v_taps: Vec<f16>,
    pub q_norm: Vec<f16>,
    pub k_norm: Vec<f16>,
}

impl LayerF16 {
    /// The layer's weights narrowed, and whether nothing was lost.
    fn new(lw: &crate::weights::LayerWeights) -> (Self, bool) {
        let mut exact = true;
        let mut nar = |v: &[f32]| -> Vec<f16> {
            v.iter()
                .map(|&x| {
                    let h = f16::from_f32(x);
                    exact &= h.to_f32().to_bits() == x.to_bits();
                    h
                })
                .collect()
        };
        let t = Self {
            kron: std::array::from_fn(|k| nar(&lw.kron[k])),
            cond_v: nar(&lw.cond_v),
            cond_u: nar(&lw.cond_u),
            d1: nar(&lw.d1),
            d2: nar(&lw.d2),
            b2: nar(&lw.b2),
            d3: nar(&lw.d3),
            d4: nar(&lw.d4),
            norm_in: nar(&lw.norm_in),
            post_norm: nar(&lw.post_norm),
            pre_hada: nar(&lw.pre_hada),
            q_taps: nar(&lw.q_taps),
            k_taps: nar(&lw.k_taps),
            v_taps: nar(&lw.v_taps),
            q_norm: nar(&lw.q_norm),
            k_norm: nar(&lw.k_norm),
        };
        (t, exact)
    }
}

impl Tables {
    pub(crate) fn new(model: &Model) -> Self {
        let c = model.config();
        let (qk, _) = c.head_dims();
        let coef = -2.0f32 / qk as f32;
        let theta = c.rope_theta as f32;
        // SAFETY: a pure libm function.
        let freq: Vec<f32> = (0..qk / 2).map(|i| lpowf(theta, i as f32 * coef)).collect();
        let (sd, cd) = freq.iter().map(|&f| sincosf(f)).unzip();
        let ncols = 2 * c.mhc_lanes + c.mhc_lanes * c.mhc_lanes;
        let phi = model.q.as_ref().map_or(vec![], |q| q.phi_f32.iter().map(|w| Phi8::new(w, ncols)).collect());
        let mut exact = true;
        let layers = model
            .w
            .layers
            .iter()
            .map(|lw| {
                let (t, e) = LayerF16::new(lw);
                exact &= e;
                t
            })
            .collect();
        Self { freq, sd, cd, phi, layers, exact }
    }
}

/// One layer's caches.
#[derive(Clone, Default)]
struct LayerState {
    /// int8 keys `[pos][kv_head][qk]` and their scales `[pos][kv_head]`.
    k8: Vec<i8>,
    ks: Vec<f32>,
    /// int8 values transposed, `[kv_head][vd][cap]`, and scales `[pos][kv_head]`.
    v8t: Vec<i8>,
    vcap: usize,
    vs: Vec<f32>,
    /// Pre-tap `q | k | v` projections, newest last.
    hist: Vec<f32>,
}

impl LayerState {
    /// Make room for `slots` KV slots.
    fn reserve(&mut self, slots: usize, nkv: usize, qk: usize, vd: usize) {
        if slots > self.vcap {
            let cap = (slots * 2).max(64);
            // 16 bytes of slack: padded P.V blocks read past a row's end.
            let mut v8t = vec![0i8; nkv * vd * cap + 16];
            for r in 0..nkv * vd {
                v8t[r * cap..r * cap + self.vcap].copy_from_slice(&self.v8t[r * self.vcap..(r + 1) * self.vcap]);
            }
            self.v8t = v8t;
            self.vcap = cap;
        }
        if self.ks.len() < slots * nkv {
            self.k8.resize(slots * nkv * qk, 0);
            self.ks.resize(slots * nkv, 0.0);
            self.vs.resize(slots * nkv, 0.0);
        }
    }
}

/// One engram site's int8 value history: `[pos][d]` and `[pos][d / 32]`.
#[derive(Clone, Default)]
struct EngramState {
    v8: Vec<i8>,
    vs: Vec<f32>,
}

/// The native path's decoding state for one sequence.
#[derive(Clone, Default)]
pub struct NatState {
    layers: Vec<LayerState>,
    engram: Vec<EngramState>,
    /// RoPE `(pos, cos, sin)` of recent positions, newest last: a single
    /// token right after the newest advances it by recurrence.
    rope: Vec<(usize, Vec<f32>, Vec<f32>)>,
    /// Final normed hidden state of the last position a forward returned
    /// outputs for (the sparse logits start from it).
    pub last_hidden: Vec<f32>,
    /// Which position each KV slot holds (the engine's slot map): a rolled
    /// back feed leaves its entries behind, and attention skips a position
    /// whose slot another one has taken since.
    pub(crate) slot_pos: Vec<i64>,
}

impl NatState {
    pub(crate) fn new(model: &Model) -> Self {
        let c = model.config();
        Self {
            layers: vec![LayerState::default(); c.num_layers],
            engram: vec![EngramState::default(); c.engram_layers.len()],
            rope: vec![],
            last_hidden: vec![],
            slot_pos: vec![],
        }
    }

    /// Forget positions `len..` the way the engine's re-feed does: the KV
    /// slots and slot map stay, the conv history is emptied (the next chunk
    /// reads zeros) and the RoPE state is dropped (the next feed recomputes
    /// its angles directly).
    pub(crate) fn rewind_zeroed(&mut self, model: &Model, len: usize) {
        let d = model.config().d_model;
        for l in &mut self.layers {
            l.hist.clear();
        }
        for e in &mut self.engram {
            e.v8.truncate(len * d);
            e.vs.truncate(len * d / 32);
        }
        self.rope.clear();
    }

    /// Forget positions `len..` (at most [`ROLLBACK`] of them), as though
    /// they had never been fed.
    pub(crate) fn rollback(&mut self, model: &Model, len: usize, n: usize) {
        let c = model.config();
        let (qk, vd) = c.head_dims();
        let nkv = c.num_kv_heads;
        let width = (c.num_heads + nkv) * qk + nkv * vd;
        for l in &mut self.layers {
            let rows = l.hist.len() / width;
            l.hist.truncate(rows.saturating_sub(n) * width);
        }
        let d = c.d_model;
        for e in &mut self.engram {
            e.v8.truncate(len * d);
            e.vs.truncate(len * d / 32);
        }
        self.rope.retain(|(p, _, _)| *p < len);
    }
}

// // FORWARD

impl Model {
    /// A prompt feed (`FUN_6c88`): before the chunks go in, the engine looks
    /// for a position the feed will attend whose slot a rolled-back branch
    /// has since taken, and re-feeds from the first such position with a
    /// zeroed conv history.
    pub(crate) fn prefill_native(&self, s: &mut Session, new: &[u32], outputs: Outputs) -> Vec<f32> {
        let pos = s.tokens.len();
        let end = pos + new.len();
        let stale = match (s.attend, s.nat.as_ref()) {
            (Some((sink, ring)), Some(st)) if !st.slot_pos.is_empty() => {
                let lo = end.saturating_sub(ring).max(sink);
                (lo..pos).find(|&p| st.slot_pos.get(slot_of(p, s.attend)).copied() != Some(p as i64))
            }
            _ => None,
        };
        let Some(p) = stale else {
            return self.forward_native(s, new, outputs, false).data;
        };
        let refeed = [&s.tokens[p..pos], new].concat();
        s.tokens.truncate(p);
        if let Some(nat) = s.nat.as_mut() {
            nat.rewind_zeroed(self, p);
        }
        if let Some(cells) = s.cells.as_mut() {
            let c = self.config();
            cells.truncate(p * (c.num_layers + 1) * c.d_model);
        }
        self.forward_native(s, &refeed, outputs, false).data
    }

    /// Advance `s` by `new` tokens on the packed weights, in the engine's
    /// chunks of at most 32.
    pub(crate) fn forward_native(&self, s: &mut Session, new: &[u32], outputs: Outputs, want_cells: bool) -> ForwardOut {
        let chunks: Vec<&[u32]> = new.chunks(32).collect();
        let mut data = vec![];
        let mut rows = 0;
        let mut cells: Option<Vec<f32>> = want_cells.then(Vec::new);
        for (i, chunk) in chunks.iter().enumerate() {
            let last = i + 1 == chunks.len();
            let want = match outputs {
                Outputs::LastLogits | Outputs::LastHidden if !last => Outputs::None,
                o => o,
            };
            let (_, vd) = self.config().head_dims();
            let out = if chunk.len() == 1 && std::env::var_os("NEEDLE_CHUNK_STEP").is_none() {
                self.step_native(s, chunk[0], want, want_cells)
            } else if chunk.len() > 1 && GROUP.is_multiple_of(vd) && std::env::var_os("NEEDLE_CHUNK_OPS").is_none() {
                self.chunk_job(s, chunk, want, want_cells)
            } else {
                self.chunk_native(s, chunk, want, want_cells)
            };
            data.extend(out.data);
            rows += out.rows;
            if let (Some(all), Some(c)) = (cells.as_mut(), out.cells) {
                all.extend(c);
            }
        }
        ForwardOut { data, rows, cells }
    }

    /// Logits for the final normed hidden state `h` with the decode kernel
    /// (the engine's full logits).
    pub fn head_logits(&self, h: &[f32]) -> Vec<f32> {
        let q = self.q.as_ref().expect("packed weights");
        let act = QAct::new(h, 1, h.len(), q.embedding.act_bits());
        let mut full = q.embedding.matmul(&act);
        full.truncate(self.config().out_rows());
        full
    }

    /// The engine's sparse logits: `ids` scored with the single-row kernel,
    /// everything else [`NEG`].
    pub fn head_logits_sparse(&self, h: &[f32], ids: &[u32]) -> Vec<f32> {
        let q = self.q.as_ref().expect("packed weights");
        let act = QAct::new(h, 1, h.len(), q.embedding.act_bits());
        let mut out = vec![NEG; self.config().out_rows()];
        let mut vals = vec![0f32; ids.len()];
        let vp = SyncPtr(vals.as_mut_ptr());
        let work = |r: std::ops::Range<usize>| {
            for k in r {
                // SAFETY: entry `k` belongs to this member.
                unsafe { *vp.ptr().add(k) = q.embedding.single_row(ids[k] as usize, &act) };
            }
        };
        if ids.len() < 64 {
            work(0..ids.len());
        } else {
            team().run(&|tid, nt| work(share(ids.len(), tid, nt)));
        }
        for (&id, &v) in ids.iter().zip(&vals) {
            out[id as usize] = v;
        }
        out
    }

    fn chunk_native(&self, s: &mut Session, toks: &[u32], outputs: Outputs, want_cells: bool) -> ForwardOut {
        let c = self.config();
        let q = self.q.as_ref().expect("packed weights");
        let tables = self.tables.as_ref().expect("native tables");
        let t = toks.len();
        let (n, d) = (c.mhc_lanes, c.d_model);
        assert!(n == 4 && c.hada_n() == 1024 && HADA_COND_RANK == 8, "the native path covers the shipped geometry");
        let nc = n * d;
        let l_total = c.num_layers;
        let p0 = s.tokens.len();
        assert!(p0 + t <= c.max_seq_len, "sequence exceeds max_seq_len");
        s.tokens.extend_from_slice(toks);
        let attend = s.attend;
        let st = s.nat.as_mut().expect("native state");
        mark_slots(&mut st.slot_pos, p0, p0 + t, attend);

        // Embedding, the same row in every lane.
        let scale = (d as f32).sqrt();
        let mut xs = vec![0f32; t * nc];
        for (i, &tok) in toks.iter().enumerate() {
            let row = q.embedding.row(tok as usize);
            for lane in 0..n {
                for (o, v) in xs[i * nc + lane * d..i * nc + (lane + 1) * d].iter_mut().zip(&row) {
                    *o = v * scale;
                }
            }
        }
        let mut cells = want_cells.then(|| vec![0f32; t * (l_total + 1) * d]);
        let record = |cells: &mut Option<Vec<f32>>, xs: &[f32], depth: usize| {
            if let Some(cl) = cells.as_mut() {
                for i in 0..t {
                    let dst = &mut cl[(i * (l_total + 1) + depth) * d..(i * (l_total + 1) + depth + 1) * d];
                    lane_mean(&xs[i * nc..(i + 1) * nc], d, dst);
                }
            }
        };
        record(&mut cells, &xs, 0);

        let engram = crate::prof::span("engram", || self.engram_native(st, &s.tokens, p0, t));
        let (cos, sin) = rope_rows(tables, st, p0, t);

        for l in 0..l_total {
            self.layer_native(l, st, attend, p0, &mut xs, t, &cos, &sin, engram.as_ref());
            record(&mut cells, &xs, l + 1);
        }

        // Lane mean, final norm, tied head.
        let rows: Vec<usize> = match outputs {
            Outputs::None => vec![],
            Outputs::LastLogits | Outputs::LastHidden => vec![t - 1],
            Outputs::AllLogits | Outputs::Hidden => (0..t).collect(),
        };
        let mut h = vec![0f32; rows.len() * d];
        for (r, &i) in rows.iter().enumerate() {
            let mut mean = vec![0f32; d];
            lane_mean(&xs[i * nc..(i + 1) * nc], d, &mut mean);
            zcn(&mean, &self.w.final_norm, &mut h[r * d..(r + 1) * d]);
        }
        if let Some(r) = rows.len().checked_sub(1) {
            st.last_hidden = h[r * d..(r + 1) * d].to_vec();
        }
        let data = match outputs {
            Outputs::None => vec![],
            Outputs::Hidden | Outputs::LastHidden => h,
            _ => {
                crate::prof::span("head", || rows.iter().enumerate().flat_map(|(r, _)| self.head_logits(&h[r * d..(r + 1) * d])).collect())
            }
        };
        if let (Some(store), Some(cl)) = (s.cells.as_mut(), cells.as_ref()) {
            store.extend_from_slice(cl);
        }
        ForwardOut { data, rows: rows.len(), cells }
    }

    /// Engram keys (f32) and conv-tapped values for the chunk, `[sites][t * d]`,
    /// writing each site's int8 value history.
    fn engram_native(&self, st: &mut NatState, tokens: &[u32], p0: usize, t: usize) -> Option<EngramKv> {
        let c = self.config();
        if c.engram_layers.is_empty() {
            return None;
        }
        let q = self.q.as_ref().expect("packed weights");
        let d = c.d_model;
        let (orders, heads, sub) = c.engram_geometry();
        let stride = if c.engram_seed_heads > 0 { c.engram_seed_heads } else { heads };
        let dil = c.engram_dilation();
        let slots = c.engram_slots as u32;
        let mut keys = Vec::with_capacity(c.engram_layers.len());
        let mut vals = Vec::with_capacity(c.engram_layers.len());
        for site in 0..c.engram_layers.len() {
            let mut e = vec![0f32; t * d];
            for i in 0..t {
                let pos = p0 + i;
                for (oi, &order) in orders.iter().enumerate() {
                    for h in 0..heads {
                        if pos + 1 < order {
                            continue;
                        }
                        let mut acc = ENGRAM_SEED.wrapping_mul((oi * stride + h + 1) as u32);
                        for j in 0..order {
                            let tok = if pos >= j { tokens[pos - j] } else { 0 };
                            acc = (acc ^ tok).wrapping_mul(ENGRAM_PRIME);
                        }
                        acc ^= acc >> 15;
                        let tbl = oi * heads + h;
                        let row = tbl * c.engram_slots + (acc % slots) as usize;
                        e[i * d + tbl * sub..i * d + (tbl + 1) * sub].copy_from_slice(&q.engram_tables[site].row(row));
                    }
                }
            }
            let lin = &q.engram_kv[site];
            let act = QAct::new(&e, t, d, lin.act_bits());
            let mut kv = vec![0f32; t * 2 * d];
            let y = SyncPtr(kv.as_mut_ptr());
            par_rows(lin.out, &|r| lin.rows_into(r, &act, y, 2 * d));
            // int8 value history in 32-wide groups, then the dilated conv
            // reading every tap back from it (the current position too).
            let es = &mut st.engram[site];
            es.v8.resize((p0 + t) * d, 0);
            es.vs.resize((p0 + t) * d / 32, 0.0);
            {
                let (v8, vs) = (SyncPtr(es.v8.as_mut_ptr()), SyncPtr(es.vs.as_mut_ptr()));
                let kv = &kv;
                par(t, &|i| {
                    let pos = p0 + i;
                    for g in 0..d / 32 {
                        // SAFETY: position `pos` belongs to row `i` alone.
                        unsafe {
                            *vs.ptr().add(pos * d / 32 + g) = quant_i8(
                                &kv[i * 2 * d + d + g * 32..i * 2 * d + d + (g + 1) * 32],
                                v8.slice(pos * d + g * 32..pos * d + (g + 1) * 32),
                            );
                        }
                    }
                });
            }
            let taps = &self.w.engrams[site].taps;
            let mut v = vec![0f32; t * d];
            let mut k = vec![0f32; t * d];
            for i in 0..t {
                k[i * d..(i + 1) * d].copy_from_slice(&kv[i * 2 * d..i * 2 * d + d]);
            }
            {
                let vp = SyncPtr(v.as_mut_ptr());
                let es = &*es;
                par(t, &|i| {
                    let pos = p0 + i;
                    // SAFETY: row `i` of `v` belongs to this item.
                    let dst = unsafe { vp.slice(i * d..(i + 1) * d) };
                    for j in 0..ENGRAM_CONV_TAPS {
                        let Some(p) = pos.checked_sub(j * dil) else { break };
                        let (q8, qs) = (&es.v8[p * d..(p + 1) * d], &es.vs[p * d / 32..(p + 1) * d / 32]);
                        for (ch, o) in dst.iter_mut().enumerate() {
                            let deq = qs[ch / 32] * q8[ch] as f32;
                            *o = taps[j * d + ch].mul_add(deq, *o);
                        }
                    }
                });
            }
            keys.push(k);
            vals.push(v);
        }
        Some((keys, vals))
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_native(
        &self,
        l: usize,
        st: &mut NatState,
        attend: Option<(usize, usize)>,
        p0: usize,
        xs: &mut [f32],
        t: usize,
        cos: &[f32],
        sin: &[f32],
        engram: Option<&EngramKv>,
    ) {
        let c = self.config();
        let q = self.q.as_ref().expect("packed weights");
        let tables = self.tables.as_ref().expect("native tables");
        let lw = &self.w.layers[l];
        let (n, d) = (c.mhc_lanes, c.d_model);
        let nc = n * d;
        let lane = l % n;
        let (qk, vd) = c.head_dims();
        let (nh, nkv) = (c.num_heads, c.num_kv_heads);
        let (qd, kd, vdim) = (nh * qk, nkv * qk, nkv * vd);
        let width = qd + kd + vdim;
        let taps = c.qkv_conv_taps;
        let half = qk / 2;
        let phi = &tables.phi[l];
        let ncols = phi.rows;

        // (a) mHC read: normalize the lanes, int8 them (one scale per token,
        // no rotation), project to the gates, mix the block input.
        let mut u = vec![0f32; t * d];
        let mut p = vec![0f32; t * ncols];
        let _sp = crate::prof::Span::new("mhc_read");
        {
            let (up, pp) = (SyncPtr(u.as_mut_ptr()), SyncPtr(p.as_mut_ptr()));
            let xs: &[f32] = xs;
            par(t, &|i| {
                let x = &xs[i * nc..(i + 1) * nc];
                let r = rinv(dot16(x, x), nc);
                let nx: Vec<f32> = x.iter().map(|v| v * r).collect();
                let mut q8 = vec![0i8; nc];
                let sx = quant_i8(&nx, &mut q8);
                // SAFETY: row `i` of `p` and `u` belongs to this item.
                let (pr, ur) = unsafe { (pp.slice(i * ncols..(i + 1) * ncols), up.slice(i * d..(i + 1) * d)) };
                for (k, o) in pr.iter_mut().enumerate() {
                    *o = (sx * phi.s[k]) * idot(&phi.q[k * phi.width..(k + 1) * phi.width], &q8) as f32;
                }
                let mut hp = [0f32; 4];
                for (j, g) in hp.iter_mut().enumerate() {
                    let off = if j == lane { 4.0 } else { -4.0 };
                    let z = ((pr[j] * lw.a_pre) + off) + lw.b_pre[j];
                    *g = 1.0 / (lexpf(-z) + 1.0);
                }
                for (o, &v) in ur.iter_mut().zip(&x[..d]) {
                    *o = v * hp[0];
                }
                for (b, &g) in hp.iter().enumerate().skip(1) {
                    for (o, &v) in ur.iter_mut().zip(&x[b * d..(b + 1) * d]) {
                        *o = v.mul_add(g, *o);
                    }
                }
            });
        }

        drop(_sp);
        let _sp = crate::prof::Span::new("norm_qkvg");
        // (b) engram gate, (c) input norm, (d) A8 and the q | k | v | gate
        // projection.
        let mut x1 = u.clone();
        let mut h = vec![0f32; t * d];
        {
            let (xp, hp) = (SyncPtr(x1.as_mut_ptr()), SyncPtr(h.as_mut_ptr()));
            let site = self.site_of_layer[l].zip(engram);
            par(t, &|i| {
                // SAFETY: row `i` of `x1` and `h` belongs to this item.
                let (xr, hr) = unsafe { (xp.slice(i * d..(i + 1) * d), hp.slice(i * d..(i + 1) * d)) };
                if let Some((site, (keys, vals))) = site {
                    let kr = &keys[site][i * d..(i + 1) * d];
                    let (rx, rk) = (rinv(dot16(xr, xr), d), rinv(dot16(kr, kr), d));
                    let a: Vec<f32> = xr.iter().map(|v| v * rx).collect();
                    let b: Vec<f32> = kr.iter().map(|v| v * rk).collect();
                    let alpha = 1.0 / (lexpf((-dot16(&a, &b)) / (d as f32).sqrt()) + 1.0);
                    for (o, &v) in xr.iter_mut().zip(&vals[site][i * d..(i + 1) * d]) {
                        *o = v.mul_add(alpha, *o);
                    }
                }
                zcn(xr, &lw.norm_in, hr);
            });
        }
        let lin = &q.qkvg[l];
        let act = QAct::new(&h, t, d, lin.act_bits());
        let mut proj = vec![0f32; t * lin.out];
        {
            let y = SyncPtr(proj.as_mut_ptr());
            par_rows(lin.out, &|r| lin.rows_into(r, &act, y, lin.out));
        }

        drop(_sp);
        let _sp = crate::prof::Span::new("conv_rope_kv");
        // (e) conv taps over the pre-tap history, (f) per-head norm and
        // RoPE, (g) the int8 KV cache.
        let ls = &mut st.layers[l];
        let hrows = ls.hist.len() / width;
        let mut qkv = vec![0f32; t * width];
        let qp = SyncPtr(qkv.as_mut_ptr());
        let hist: &[f32] = &ls.hist;
        par(t, &|i| {
            // Row `i - j` of the chunk, else of the history; zeros before
            // position 0.
            let prev = |j: usize| -> Option<&[f32]> {
                if j <= i {
                    Some(&proj[(i - j) * lin.out..(i - j) * lin.out + width])
                } else if hrows + i >= j {
                    Some(&hist[(hrows + i - j) * width..(hrows + i - j + 1) * width])
                } else {
                    None
                }
            };
            // SAFETY: row `i` of `qkv` belongs to this item.
            let out = unsafe { qp.slice(i * width..(i + 1) * width) };
            for (start, cw, w) in [(0, qd, &lw.q_taps), (qd, kd, &lw.k_taps), (qd + kd, vdim, &lw.v_taps)] {
                let o = &mut out[start..start + cw];
                let cur = &prev(0).expect("current row")[start..start + cw];
                for ((o, &wv), &x) in o.iter_mut().zip(&w[..cw]).zip(cur) {
                    *o = wv * x;
                }
                for j in 1..taps {
                    let wj = &w[j * cw..(j + 1) * cw];
                    match prev(j) {
                        Some(r) => {
                            for ((o, &wv), &x) in o.iter_mut().zip(wj).zip(&r[start..start + cw]) {
                                *o = wv.mul_add(x, *o);
                            }
                        }
                        None => {
                            for (o, &wv) in o.iter_mut().zip(wj) {
                                *o = wv.mul_add(0.0, *o);
                            }
                        }
                    }
                }
            }
            let row = out;
            let (cs, sn) = (&cos[i * half..(i + 1) * half], &sin[i * half..(i + 1) * half]);
            for hh in 0..nh + nkv {
                let (off, scale) = if hh < nh { (hh * qk, &lw.q_norm) } else { (qd + (hh - nh) * qk, &lw.k_norm) };
                let x = &mut row[off..off + qk];
                zcn_inplace(x, scale);
                for f in 0..half {
                    let (a, b) = (x[f], x[f + half]);
                    x[f] = (-sn[f]).mul_add(b, cs[f] * a);
                    x[f + half] = sn[f].mul_add(a, cs[f] * b);
                }
            }
        });
        for i in 0..t {
            ls.hist.extend_from_slice(&proj[i * lin.out..i * lin.out + width]);
        }
        let keep = taps - 1 + ROLLBACK;
        let rows = ls.hist.len() / width;
        if rows > 2 * keep {
            ls.hist.drain(..(rows - keep) * width);
        }
        let end = p0 + t;
        // The KV cache is the engine's slot ring: a rolled-back position
        // leaves what it wrote in place (the engine never restores KV).
        ls.reserve(slot_of(end - 1, attend) + 1, nkv, qk, vd);
        {
            // Rows of a chunk (at most 32) never share a slot.
            let (k8, ks, vs, v8t, vcap) = (
                SyncPtr(ls.k8.as_mut_ptr()),
                SyncPtr(ls.ks.as_mut_ptr()),
                SyncPtr(ls.vs.as_mut_ptr()),
                SyncPtr(ls.v8t.as_mut_ptr()),
                ls.vcap,
            );
            let qkv = &qkv;
            par(t, &|i| {
                let pos = slot_of(p0 + i, attend);
                let row = &qkv[i * width..(i + 1) * width];
                let mut vq = [0i8; 256];
                // SAFETY: slot `pos` belongs to row `i` alone.
                unsafe {
                    for kvh in 0..nkv {
                        *ks.ptr().add(pos * nkv + kvh) = quant_i8(
                            &row[qd + kvh * qk..qd + (kvh + 1) * qk],
                            k8.slice((pos * nkv + kvh) * qk..(pos * nkv + kvh + 1) * qk),
                        );
                        *vs.ptr().add(pos * nkv + kvh) = quant_i8(&row[qd + kd + kvh * vd..qd + kd + (kvh + 1) * vd], &mut vq[..vd]);
                        for (dm, &v) in vq[..vd].iter().enumerate() {
                            *v8t.ptr().add((kvh * vd + dm) * vcap + pos) = v;
                        }
                    }
                }
            });
        }

        drop(_sp);
        let _sp = crate::prof::Span::new("attention");
        // (h) attention, (i) the gate, (j) A8 and the out projection.
        let ls = &st.layers[l];
        let sp: &[i64] = &st.slot_pos;
        let window = c.layer_window(l);
        let nested = t > 1;
        let mut att = vec![0f32; t * nh * vd];
        {
            let ap = SyncPtr(att.as_mut_ptr());
            let group = nh / nkv;
            par(t * nh, &|task| {
                let (i, hh) = (task / nh, task % nh);
                let pos = p0 + i;
                let keys = key_ranges(pos, end, window, attend, sp);
                // SAFETY: head `hh` of row `i` belongs to this task.
                let out = unsafe { ap.slice((i * nh + hh) * vd..(i * nh + hh + 1) * vd) };
                attend_head(ls, &qkv[i * width + hh * qk..i * width + (hh + 1) * qk], hh / group, nkv, qk, vd, &keys, attend, nested, out);
            });
        }
        for i in 0..t {
            for (a, &g) in att[i * nh * vd..(i + 1) * nh * vd].iter_mut().zip(&proj[i * lin.out + width..(i + 1) * lin.out]) {
                *a /= nexp(-g) + 1.0;
            }
        }
        drop(_sp);
        let _sp = crate::prof::Span::new("out_proj");
        let lin_o = &q.out[l];
        let act = QAct::new(&att, t, nh * vd, lin_o.act_bits());
        let mut o = vec![0f32; t * d];
        {
            let y = SyncPtr(o.as_mut_ptr());
            par_rows(lin_o.out, &|r| lin_o.rows_into(r, &act, y, d));
        }

        drop(_sp);
        let _sp = crate::prof::Span::new("post_mlp");
        // (k) post block: residual, pre-MLP norm, the Hadamard MLP, the post
        // and residual gates, Sinkhorn, and the mHC write.
        let ga = 1.0 / (lexpf(-lw.attn_gate) + 1.0);
        let xp = SyncPtr(xs.as_mut_ptr());
        par(t, &|i| {
            let mut x2 = vec![0f32; d];
            let mut on = vec![0f32; d];
            zcn(&o[i * d..(i + 1) * d], &lw.post_norm, &mut on);
            for ((x2, &on), &x1) in x2.iter_mut().zip(&on).zip(&x1[i * d..(i + 1) * d]) {
                *x2 = on.mul_add(ga, x1);
            }
            let mut h2 = vec![0f32; d];
            zcn(&x2, &lw.pre_hada, &mut h2);
            let mlp = self.mlp_native(&tables.layers[l], &h2);
            let pr = &p[i * ncols..(i + 1) * ncols];
            let hpost: [f32; 4] = std::array::from_fn(|j| {
                let off = if j == lane { 0.0 } else { -4.0 };
                let z = ((pr[n + j] * lw.a_post) + off) + lw.b_post[j];
                2.0 / (lexpf(-z) + 1.0)
            });
            let zres: [f32; 16] = std::array::from_fn(|k| pr[2 * n + k].mul_add(lw.a_res, lw.b_res[k]));
            let hres = sinkhorn64(&zres);
            // SAFETY: row `i` of the lane stream belongs to this item, and
            // nothing else reads it during the phase.
            let x = unsafe { xp.slice(i * nc..(i + 1) * nc) };
            let old: Vec<f32> = x.to_vec();
            let ur = &u[i * d..(i + 1) * d];
            let y: Vec<f32> = (0..d).map(|ch| (x2[ch] + mlp[ch]) - ur[ch]).collect();
            for a in 0..n {
                let xa = &mut x[a * d..(a + 1) * d];
                for (o, &yv) in xa.iter_mut().zip(&y) {
                    *o = yv * hpost[a];
                }
                for b in 0..n {
                    let h = hres[a * n + b];
                    for (o, &v) in xa.iter_mut().zip(&old[b * d..(b + 1) * d]) {
                        *o = v.mul_add(h, *o);
                    }
                }
            }
        });
    }

    /// The Hadamard MLP of one row: conditioning softmax, three Kronecker
    /// stages `(A^T Z) B`, SiLU after the first. Returns `mlp[..d]`.
    fn mlp_native(&self, lw: &LayerF16, h2: &[f32]) -> Vec<f32> {
        const NB: usize = 32;
        const HN: usize = NB * NB;
        let d = h2.len();
        let w = cond_weights(h2, &lw.cond_v);
        let mut z = [0f32; HN];
        for (j, zz) in z.iter_mut().enumerate().take(d) {
            *zz = h2[j] * lw.d1[j].to_f32();
        }
        let mut t = [0f32; HN];
        kron_native(&z, &lw.kron[0], &lw.kron[1], &mut t);
        // cond = 1 + sum_k w_k u_k, an fma chain over k per element.
        let mut cond = [1f32; HN];
        for (k, &wk) in w.iter().enumerate() {
            axpy16(&mut cond, &lw.cond_u[k * HN..(k + 1) * HN], wk);
        }
        for (j, zz) in z.iter_mut().enumerate() {
            *zz = t[self.w.perm1[j]].mul_add(cond[j] * lw.d2[j].to_f32(), lw.b2[j].to_f32());
        }
        silu_native(&mut z);
        kron_native(&z, &lw.kron[2], &lw.kron[3], &mut t);
        for (j, zz) in z.iter_mut().enumerate() {
            *zz = t[self.w.perm2[j]] * lw.d3[j].to_f32();
        }
        kron_native(&z, &lw.kron[4], &lw.kron[5], &mut t);
        (0..d).map(|j| t[j] * lw.d4[j].to_f32()).collect()
    }
}

/// The MLP conditioning softmax weights of the pre-MLP row `h2`: logits
/// accumulated in four sets by `j % 4` (fma chains over `j`), combined
/// `((C + D) + B) + A`, then a libm softmax with the engine's sum tree.
fn cond_weights(h2: &[f32], cond_v: &[f16]) -> [f32; 8] {
    let mut acc = [[0f32; 8]; 4];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; `cond_v` holds 8 per element of
    // `h2`.
    unsafe {
        use std::arch::aarch64::*;
        debug_assert!(h2.len().is_multiple_of(4));
        let z = vdupq_n_f32(0.0);
        let (mut a0, mut a1, mut b0, mut b1, mut c0, mut c1, mut d0, mut d1) = (z, z, z, z, z, z, z, z);
        for j in (0..h2.len()).step_by(4) {
            a0 = vfmaq_n_f32(a0, cond_v.load4(j * 8), h2[j]);
            a1 = vfmaq_n_f32(a1, cond_v.load4(j * 8 + 4), h2[j]);
            b0 = vfmaq_n_f32(b0, cond_v.load4((j + 1) * 8), h2[j + 1]);
            b1 = vfmaq_n_f32(b1, cond_v.load4((j + 1) * 8 + 4), h2[j + 1]);
            c0 = vfmaq_n_f32(c0, cond_v.load4((j + 2) * 8), h2[j + 2]);
            c1 = vfmaq_n_f32(c1, cond_v.load4((j + 2) * 8 + 4), h2[j + 2]);
            d0 = vfmaq_n_f32(d0, cond_v.load4((j + 3) * 8), h2[j + 3]);
            d1 = vfmaq_n_f32(d1, cond_v.load4((j + 3) * 8 + 4), h2[j + 3]);
        }
        for (o, (lo, hi)) in acc.iter_mut().zip([(a0, a1), (b0, b1), (c0, c1), (d0, d1)]) {
            vst1q_f32(o.as_mut_ptr(), lo);
            vst1q_f32(o.as_mut_ptr().add(4), hi);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: `cond_v` holds 8 per element of `h2`.
    unsafe {
        use std::arch::wasm32::*;
        let mut v = [f32x4_splat(0.0); 8];
        for (j, &hv) in h2.iter().enumerate() {
            let set = (j % 4) * 2;
            let h = f32x4_splat(hv);
            v[set] = wsimd::fma4(h, cond_v.load4w(j * 8), v[set]);
            v[set + 1] = wsimd::fma4(h, cond_v.load4w(j * 8 + 4), v[set + 1]);
        }
        for (k, a) in acc.iter_mut().enumerate() {
            a[..4].copy_from_slice(&wsimd::lanes(v[2 * k]));
            a[4..].copy_from_slice(&wsimd::lanes(v[2 * k + 1]));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    for (j, &hv) in h2.iter().enumerate() {
        let a = &mut acc[j % 4];
        for (k, ak) in a.iter_mut().enumerate() {
            *ak = hv.mul_add(cond_v[j * 8 + k].to_f32(), *ak);
        }
    }
    let lg: [f32; 8] = std::array::from_fn(|k| ((acc[2][k] + acc[3][k]) + acc[1][k]) + acc[0][k]);
    let m = lg.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: [f32; 8] = std::array::from_fn(|k| lexpf(lg[k] - m));
    let sum = e[7] + ((e[6] + (e[5] + e[4])) + ((e[3] + e[2]) + (e[1] + e[0])));
    let inv = 1.0 / sum;
    std::array::from_fn(|k| e[k] * inv)
}

/// Split a matmul's output rows across the team.
fn par_rows(out: usize, f: &(dyn Fn(std::ops::Range<usize>) + Sync)) {
    team().run(&|tid, nt| {
        let r = share(out, tid, nt);
        if !r.is_empty() {
            f(r);
        }
    });
}

/// `((x0 + x1) + x2 + x3) * (1/4)` per element.
fn lane_mean(x: &[f32], d: usize, out: &mut [f32]) {
    for (ch, o) in out.iter_mut().enumerate() {
        *o = (((x[ch] + x[d + ch]) + x[2 * d + ch]) + x[3 * d + ch]) * 0.25;
    }
}

/// RoPE `cos` and `sin` for the chunk's rows, `[t][qk/2]` each: `sincosf` of
/// `freq * pos` for chunk rows, and for a single token right after the
/// newest known position, the angle-addition recurrence from it.
fn rope_rows(tb: &Tables, st: &mut NatState, p0: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let half = tb.freq.len();
    let mut cos = vec![0f32; t * half];
    let mut sin = vec![0f32; t * half];
    let prev = st.rope.last().filter(|(p, _, _)| t == 1 && *p + 1 == p0);
    if let Some((_, c0, s0)) = prev {
        for f in 0..half {
            cos[f] = (-tb.sd[f]).mul_add(s0[f], tb.cd[f] * c0[f]);
            sin[f] = tb.sd[f].mul_add(c0[f], tb.cd[f] * s0[f]);
        }
    } else {
        for i in 0..t {
            for f in 0..half {
                let (s, c) = sincosf(tb.freq[f] * (p0 + i) as f32);
                cos[i * half + f] = c;
                sin[i * half + f] = s;
            }
        }
    }
    st.rope.retain(|(p, _, _)| *p < p0);
    for i in 0..t {
        st.rope.push((p0 + i, cos[i * half..(i + 1) * half].to_vec(), sin[i * half..(i + 1) * half].to_vec()));
    }
    let keep = ROLLBACK + 1;
    if st.rope.len() > 2 * keep {
        st.rope.drain(..st.rope.len() - keep);
    }
    (cos, sin)
}

/// The KV slot of position `p`: the sink keeps its positions, the rest
/// share a ring (unbounded without an attention span).
#[inline]
fn slot_of(p: usize, attend: Option<(usize, usize)>) -> usize {
    match attend {
        Some((sink, ring)) if p >= sink => (p - sink) % ring + sink,
        _ => p,
    }
}

/// The keys a query sees, as ascending position ranges (empty ones at the
/// end).
pub(crate) type Keys = [std::ops::Range<usize>; KEY_RUNS];
const KEY_RUNS: usize = 16;

/// How many keys `keys` holds.
#[inline]
pub(crate) fn keys_total(keys: &Keys) -> usize {
    keys.iter().map(|r| r.len()).sum()
}

/// Record that positions `p0..end` now hold their slots.
pub(crate) fn mark_slots(sp: &mut Vec<i64>, p0: usize, end: usize, attend: Option<(usize, usize)>) {
    for p in p0..end {
        let s = slot_of(p, attend);
        if sp.len() <= s {
            sp.resize(s + 1, -1);
        }
        sp[s] = p as i64;
    }
}

/// The keys a query at `pos` sees, in the engine's order: the sink (within
/// the layer's band), then the ring as of the chunk's end, less any
/// position whose slot holds another position now (`sp`, the slot map).
fn key_ranges(pos: usize, chunk_end: usize, window: Option<usize>, attend: Option<(usize, usize)>, sp: &[i64]) -> Keys {
    let lo = window.map_or(0, |w| (pos + 1).saturating_sub(w));
    let coarse = match attend {
        None => [lo..pos + 1, 0..0],
        Some((sink, ring)) => {
            let a = lo..sink.min(pos + 1).max(lo);
            let b0 = chunk_end.saturating_sub(ring).max(sink).min(pos + 1);
            [a, b0..pos + 1]
        }
    };
    let mut out: Keys = Default::default();
    let mut k = 0;
    let mut push = |r: std::ops::Range<usize>| {
        if r.is_empty() {
            return;
        }
        // A run that would not fit joins the previous one only if adjacent;
        // sixteen runs cover any rollback pattern the agent produces.
        assert!(k < KEY_RUNS, "too many key runs");
        out[k] = r;
        k += 1;
    };
    for r in coarse {
        if attend.is_none() {
            push(r);
            continue;
        }
        let mut start = None;
        for p in r.clone() {
            let ok = sp.get(slot_of(p, attend)).copied() == Some(p as i64);
            match (ok, start) {
                (true, None) => start = Some(p),
                (false, Some(a)) => {
                    push(a..p);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(a) = start {
            push(a..r.end);
        }
    }
    out
}

/// Two ranges as a key list (tests and the general path).
#[cfg(test)]
fn keys_of(a: std::ops::Range<usize>, b: std::ops::Range<usize>) -> Keys {
    let mut k: Keys = Default::default();
    k[0] = a;
    k[1] = b;
    k
}

/// Per-thread scratch for [`attend_head`].
#[derive(Default)]
struct AttnScratch {
    slots: Vec<usize>,
    s: Vec<f32>,
    scale: Vec<f32>,
    dots: Vec<i32>,
    digits: Vec<i8>,
    runs: Vec<(usize, usize, usize)>,
}

thread_local! {
    static ATTN: std::cell::RefCell<AttnScratch> = std::cell::RefCell::new(AttnScratch::default());
}

/// One head's attention over `keys`, in the engine's key chunks, into `out`
/// (`vd` values).
#[allow(clippy::too_many_arguments)]
fn attend_head(
    ls: &LayerState,
    qh: &[f32],
    kvh: usize,
    nkv: usize,
    qk: usize,
    vd: usize,
    keys: &Keys,
    attend: Option<(usize, usize)>,
    nested: bool,
    out: &mut [f32],
) {
    ATTN.with_borrow_mut(|sc| attend_head_fast(sc, ls, qh, kvh, nkv, qk, vd, keys, attend, nested, out));
}

/// The straightforward form of [`attend_head_fast`] (tests compare them).
#[allow(clippy::too_many_arguments, dead_code)]
fn attend_head_ref(
    sc: &mut AttnScratch,
    ls: &LayerState,
    qh: &[f32],
    kvh: usize,
    nkv: usize,
    qk: usize,
    vd: usize,
    keys: &Keys,
    attend: Option<(usize, usize)>,
    nested: bool,
    out: &mut [f32],
) {
    let mut q8 = [0i8; 64];
    let qs = quant_i8(qh, &mut q8[..qk]) * INV_SQRT_QK;
    let q8 = &q8[..qk];
    let n = keys_total(keys);
    let k = if nested { 1 } else { (2 * NATIVE_TEAM / nkv).max(1) };
    let mut c = n.div_ceil(256);
    if k * 64 <= n && c < k {
        c = k;
    }
    let chunk = n.div_ceil(c.max(1)).div_ceil(16) * 16;
    // Each key's slot, and the runs of consecutive slots (the values are
    // transposed, so a run is contiguous per value dimension).
    sc.slots.clear();
    for r in keys.iter().filter(|r| !r.is_empty()) {
        // Consecutive positions take consecutive slots until the ring wraps.
        let mut slot = slot_of(r.start, attend);
        for p in r.clone() {
            sc.slots.push(slot);
            slot += 1;
            if let Some((sink, ring)) = attend
                && p + 1 >= sink
                && slot == sink + ring
            {
                slot = sink;
            }
        }
    }
    debug_assert!(sc.slots.iter().zip(keys.iter().flat_map(|r| r.clone())).all(|(&s, p)| s == slot_of(p, attend)));
    for v in [&mut sc.s, &mut sc.scale] {
        v.resize(chunk, 0.0);
    }
    sc.dots.resize(chunk, 0);
    sc.digits.resize(3 * chunk, 0);
    let mut parts: [(f32, f32, [f32; 64]); 8] = [(0.0, 0.0, [0.0; 64]); 8];
    let mut np = 0;
    for c0 in (0..n).step_by(chunk) {
        let len = chunk.min(n - c0);
        let sl = &sc.slots[c0..c0 + len];
        let blocked = len / 4 * 4;
        key_dots(q8, &ls.k8, sl, nkv, kvh, qk, &mut sc.dots[..len]);
        // Scores: `(ks * dot) * qs` in blocks of four, `(ks * qs) * dot`
        // for the last `len % 4`.
        for (o, &slot) in sc.scale.iter_mut().zip(sl) {
            *o = ls.ks[slot * nkv + kvh];
        }
        let s = &mut sc.s[..len];
        for i in 0..blocked {
            s[i] = (sc.scale[i] * sc.dots[i] as f32) * qs;
        }
        for i in blocked..len {
            s[i] = (sc.scale[i] * qs) * sc.dots[i] as f32;
        }
        let m = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum = chunk_exp(s, m);
        for (o, &slot) in sc.scale.iter_mut().zip(sl) {
            *o = ls.vs[slot * nkv + kvh];
        }
        let mut wmax = 0f32;
        for (w, &v) in s.iter_mut().zip(&sc.scale) {
            *w *= v;
            wmax = wmax.max(*w);
        }
        let qw = Q21 / wmax;
        let (hi, rest) = sc.digits.split_at_mut(chunk);
        let (mid, lo) = rest.split_at_mut(chunk);
        digits21(&s[..len], qw, &mut hi[..len], &mut mid[..len], &mut lo[..len]);
        sc.runs.clear();
        for (i, &slot) in sl.iter().enumerate() {
            match sc.runs.last_mut() {
                Some((_, s0, l)) if *s0 + *l == slot => *l += 1,
                _ => sc.runs.push((i, slot, 1)),
            }
        }
        let cw = wmax * C21;
        let o = &mut parts[np].2;
        for dm0 in (0..vd).step_by(4) {
            let rows: [&[i8]; 4] = std::array::from_fn(|k| {
                let r = kvh * vd + (dm0 + k).min(vd - 1);
                &ls.v8t[r * ls.vcap..(r + 1) * ls.vcap]
            });
            let mut sums = [[0i32; 3]; 4];
            for &(i0, s0, l) in &sc.runs {
                let part = pv_dots4(&hi[i0..i0 + l], &mid[i0..i0 + l], &lo[i0..i0 + l], rows, s0);
                for (a, b) in sums.iter_mut().zip(part) {
                    a[0] += b[0];
                    a[1] += b[1];
                    a[2] += b[2];
                }
            }
            for (k, sm) in sums.iter().enumerate().take((vd - dm0).min(4)) {
                o[dm0 + k] = 16384f32.mul_add(sm[0] as f32, 128f32.mul_add(sm[1] as f32, sm[2] as f32)) * cw;
            }
        }
        parts[np].0 = m;
        parts[np].1 = sum;
        np += 1;
    }
    let parts = &parts[..np];
    if parts.len() == 1 {
        let (_, sum, o) = &parts[0];
        let inv = 1.0 / sum;
        for (x, &v) in out.iter_mut().zip(o) {
            *x = v * inv;
        }
        return;
    }
    let big = parts.iter().fold(-1e30f32, |a, p| a.max(p.0));
    let mut acc = [0f32; 64];
    let mut total = 0f32;
    for (m, sum, o) in parts {
        let w = lexpf(m - big);
        for (a, &v) in acc.iter_mut().zip(o) {
            *a = v.mul_add(w, *a);
        }
        total = w.mul_add(*sum, total);
    }
    let inv = 1.0 / total;
    for (x, &a) in out.iter_mut().zip(&acc) {
        *x = a * inv;
    }
}

/// Runs of consecutive slots covering `keys` in order: `(first key index,
/// first slot, length)`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn slot_runs(keys: &Keys, attend: Option<(usize, usize)>, runs: &mut Vec<(usize, usize, usize)>) {
    runs.clear();
    let mut idx = 0;
    for r in keys.iter().filter(|r| !r.is_empty()) {
        let mut p = r.start;
        while p < r.end {
            let slot = slot_of(p, attend);
            // A run ends where the ring wraps.
            let room = match attend {
                Some((sink, ring)) if p >= sink => sink + ring - slot,
                _ => usize::MAX,
            };
            let len = room.min(r.end - p);
            match runs.last_mut() {
                Some((_, s0, l)) if *s0 + *l == slot => *l += len,
                _ => runs.push((idx, slot, len)),
            }
            idx += len;
            p += len;
        }
    }
}

/// [`attend_head_ref`] organised around runs of consecutive slots: scales
/// load as vectors, the softmax digits are zero-padded to whole 16-key
/// blocks, and each run's P.V takes one pass per four value rows.
#[allow(clippy::too_many_arguments)]
fn attend_head_fast(
    sc: &mut AttnScratch,
    ls: &LayerState,
    qh: &[f32],
    kvh: usize,
    nkv: usize,
    qk: usize,
    vd: usize,
    keys: &Keys,
    attend: Option<(usize, usize)>,
    nested: bool,
    out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    if fast_attention(qk, vd) {
        let n = keys_total(keys);
        let (count, _) = key_chunks(n, nested, nkv);
        let mut parts = [(0f32, 0f32, [0f32; 64]); 8];
        // SAFETY: `fast_attention` checked the CPU; the value cache has 16
        // bytes of slack past every row for the padded blocks.
        unsafe { neon::attend_parts(sc, ls, qh, kvh, nkv, vd, keys, attend, nested, 0..count, &mut parts) };
        return merge_parts(&parts[..count], out);
    }
    attend_head_ref(sc, ls, qh, kvh, nkv, qk, vd, keys, attend, nested, out)
}

/// One key chunk's partial attention: its score max, its softmax sum, and
/// its unnormalised output.
pub(crate) type Part = (f32, f32, [f32; 64]);

/// Whether [`attend_chunk`] and [`attend_head_fast`] take the NEON path.
#[inline]
fn fast_attention(qk: usize, vd: usize) -> bool {
    #[cfg(target_arch = "aarch64")]
    return qk == 48 && vd <= 64 && vd.is_multiple_of(4) && std::arch::is_aarch64_feature_detected!("dotprod");
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (qk, vd);
        false
    }
}

/// The engine's key chunking for `n` keys: how many chunks, and their size
/// (a multiple of 16).
#[inline]
fn key_chunks(n: usize, nested: bool, nkv: usize) -> (usize, usize) {
    let k = if nested { 1 } else { (2 * NATIVE_TEAM / nkv).max(1) };
    let mut c = n.div_ceil(256);
    if k * 64 <= n && c < k {
        c = k;
    }
    let chunk = n.div_ceil(c.max(1)).div_ceil(16) * 16;
    (n.div_ceil(chunk.max(1)), chunk)
}

/// Chunk `ci` of one head's attention (`None` past the last chunk), for
/// splitting a head's chunks across members; [`merge_parts`] combines them.
#[allow(clippy::too_many_arguments)]
fn attend_chunk(
    ls: &LayerState,
    qh: &[f32],
    kvh: usize,
    nkv: usize,
    qk: usize,
    vd: usize,
    keys: &Keys,
    attend: Option<(usize, usize)>,
    nested: bool,
    ci: usize,
) -> Option<Part> {
    let n = keys_total(keys);
    if ci >= key_chunks(n, nested, nkv).0 {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    if fast_attention(qk, vd) {
        let mut part = [(0f32, 0f32, [0f32; 64])];
        // SAFETY: as in `attend_head_fast`.
        ATTN.with_borrow_mut(|sc| unsafe { neon::attend_parts(sc, ls, qh, kvh, nkv, vd, keys, attend, nested, ci..ci + 1, &mut part) });
        return Some(part[0]);
    }
    // Callers split heads by chunk only when `fast_attention` holds.
    let _ = (ls, qh, kvh, qk, vd, attend);
    unreachable!("chunked attention runs on the NEON path only")
}

/// Combine a head's chunk parts: one chunk normalises by its sum; several
/// merge with libm `expf` weights against the largest max.
fn merge_parts(parts: &[Part], out: &mut [f32]) {
    if parts.len() == 1 {
        let (_, sum, o) = &parts[0];
        let inv = 1.0 / sum;
        for (x, &v) in out.iter_mut().zip(o) {
            *x = v * inv;
        }
        return;
    }
    let big = parts.iter().fold(-1e30f32, |a, p| a.max(p.0));
    let mut acc = [0f32; 64];
    let mut total = 0f32;
    for (m, sum, o) in parts {
        let w = lexpf(m - big);
        for (a, &v) in acc.iter_mut().zip(o) {
            *a = v.mul_add(w, *a);
        }
        total = w.mul_add(*sum, total);
    }
    let inv = 1.0 / total;
    for (x, &a) in out.iter_mut().zip(&acc) {
        *x = a * inv;
    }
}

/// Softmax weights to 21-bit integers (`round(w * qw)`, half away from
/// zero) split into three 7-bit digits.
#[inline]
fn digits21(w: &[f32], qw: f32, hi: &mut [i8], mid: &mut [i8], lo: &mut [i8]) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; loads and stores stay in bounds.
    unsafe {
        use std::arch::aarch64::*;
        let qv = vdupq_n_f32(qw);
        let m7 = vdupq_n_s32(127);
        while i + 4 <= w.len() {
            let q = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(w.as_ptr().add(i)), qv));
            let h = vshrq_n_s32::<14>(q);
            let md = vandq_s32(vshrq_n_s32::<7>(q), m7);
            let l = vandq_s32(q, m7);
            for (dst, v) in [(hi.as_mut_ptr(), h), (mid.as_mut_ptr(), md), (lo.as_mut_ptr(), l)] {
                let b = vmovn_s16(vcombine_s16(vmovn_s32(v), vdup_n_s16(0)));
                let lanes: [i8; 8] = std::mem::transmute(b);
                std::ptr::copy_nonoverlapping(lanes.as_ptr(), dst.add(i), 4);
            }
            i += 4;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: loads and stores stay in bounds (whole 4-element blocks).
    unsafe {
        use std::arch::wasm32::*;
        let qv = f32x4_splat(qw);
        let m7 = i32x4_splat(127);
        while i + 4 <= w.len() {
            let q = i32x4_trunc_sat_f32x4(wsimd::round_away(f32x4_mul(wsimd::load(w.as_ptr().add(i)), qv)));
            let lanes = [i32x4_shr(q, 14), v128_and(i32x4_shr(q, 7), m7), v128_and(q, m7)];
            for (dst, v) in [hi.as_mut_ptr(), mid.as_mut_ptr(), lo.as_mut_ptr()].into_iter().zip(lanes) {
                *dst.add(i) = i32x4_extract_lane::<0>(v) as i8;
                *dst.add(i + 1) = i32x4_extract_lane::<1>(v) as i8;
                *dst.add(i + 2) = i32x4_extract_lane::<2>(v) as i8;
                *dst.add(i + 3) = i32x4_extract_lane::<3>(v) as i8;
            }
            i += 4;
        }
    }
    while i < w.len() {
        let wq = (w[i] * qw).round() as i32;
        hi[i] = (wq >> 14) as i8;
        mid[i] = ((wq >> 7) & 127) as i8;
        lo[i] = (wq & 127) as i8;
        i += 1;
    }
}

/// Softmax numerators of a key chunk in place (the engine's inline exp over
/// blocks of eight and four, libm `expf` for the last one to three), and
/// their sum in the engine's order.
fn chunk_exp(s: &mut [f32], m: f32) -> f32 {
    let l = s.len();
    let mut i = 0;
    let mut sum;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; every block stays inside `s`.
    unsafe {
        use std::arch::aarch64::*;
        let mv = vdupq_n_f32(m);
        let (mut a, mut b) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let p = s.as_mut_ptr();
        while i + 8 <= l {
            let e0 = neon::nexp4(vsubq_f32(vld1q_f32(p.add(i)), mv));
            let e1 = neon::nexp4(vsubq_f32(vld1q_f32(p.add(i + 4)), mv));
            vst1q_f32(p.add(i), e0);
            vst1q_f32(p.add(i + 4), e1);
            a = vaddq_f32(e0, a);
            b = vaddq_f32(e1, b);
            i += 8;
        }
        while i + 4 <= l {
            let e0 = neon::nexp4(vsubq_f32(vld1q_f32(p.add(i)), mv));
            vst1q_f32(p.add(i), e0);
            a = vaddq_f32(e0, a);
            i += 4;
        }
        let t = vaddq_f32(a, b);
        let q = vpaddq_f32(t, t);
        sum = vgetq_lane_f32(q, 0) + vgetq_lane_f32(q, 1);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: every block stays inside `s`.
    unsafe {
        use std::arch::wasm32::*;
        let mv = f32x4_splat(m);
        let (mut a, mut b) = (f32x4_splat(0.0), f32x4_splat(0.0));
        let p = s.as_mut_ptr();
        while i + 8 <= l {
            let e0 = wsimd::nexp4(f32x4_sub(wsimd::load(p.add(i)), mv));
            let e1 = wsimd::nexp4(f32x4_sub(wsimd::load(p.add(i + 4)), mv));
            wsimd::store(p.add(i), e0);
            wsimd::store(p.add(i + 4), e1);
            a = f32x4_add(a, e0);
            b = f32x4_add(b, e1);
            i += 8;
        }
        while i + 4 <= l {
            let e0 = wsimd::nexp4(f32x4_sub(wsimd::load(p.add(i)), mv));
            wsimd::store(p.add(i), e0);
            a = f32x4_add(a, e0);
            i += 4;
        }
        let t = wsimd::lanes(f32x4_add(a, b));
        sum = (t[0] + t[1]) + (t[2] + t[3]);
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let mut a = [0f32; 4];
        let mut b = [0f32; 4];
        while i + 8 <= l {
            for k in 0..4 {
                let e = nexp(s[i + k] - m);
                s[i + k] = e;
                a[k] += e;
            }
            for k in 0..4 {
                let e = nexp(s[i + 4 + k] - m);
                s[i + 4 + k] = e;
                b[k] += e;
            }
            i += 8;
        }
        while i + 4 <= l {
            for k in 0..4 {
                let e = nexp(s[i + k] - m);
                s[i + k] = e;
                a[k] += e;
            }
            i += 4;
        }
        let t: [f32; 4] = std::array::from_fn(|k| a[k] + b[k]);
        sum = (t[0] + t[1]) + (t[2] + t[3]);
    }
    if l - i >= 2 {
        let (e0, e1) = (lexpf(s[i] - m), lexpf(s[i + 1] - m));
        s[i] = e0;
        s[i + 1] = e1;
        sum = (e0 + sum) + e1;
        i += 2;
    }
    if l - i == 1 {
        let e = lexpf(s[i] - m);
        s[i] = e;
        sum += e;
    }
    sum
}

/// One Kronecker stage `T = (A^T Z) B` on 32x32 blocks, each sum an fma
/// chain from `j = 0`.
fn kron_native(z: &[f32], a: &[f16], b: &[f16], t: &mut [f32]) {
    const NB: usize = 32;
    let mut mt = [0f32; NB * NB];
    // SAFETY: whole 32x32 blocks.
    unsafe {
        mix8::<true, _, _>(a, z, SyncPtr(mt.as_mut_ptr()), 0..MIX_TILES);
        mix8::<false, _, _>(mt.as_slice(), b, SyncPtr(t.as_mut_ptr()), 0..MIX_TILES);
    }
}

/// Tiles of a [`mix8`] product: 8 rows by 8 columns.
const MIX_TILES: usize = 16;

/// `out[r][c] = sum_j ct[j][r] * rows[j][c]` over the 8x8 tiles `tiles` (of
/// [`MIX_TILES`]) of a 32x32 output, each element an fma chain over `j`
/// from zero (`ct` holds the coefficients by `j` first). With `TRANSPOSE`
/// the tile lands at `out[c][r]`.
///
/// # Safety
/// `out` points at 1024 floats and no one else writes those tiles.
#[inline]
unsafe fn mix8<const TRANSPOSE: bool, C: Src + ?Sized, R: Src + ?Sized>(
    ct: &C,
    rows: &R,
    out: SyncPtr<f32>,
    tiles: std::ops::Range<usize>,
) {
    const NB: usize = 32;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; every access is inside the
    // 32x32 blocks.
    unsafe {
        use std::arch::aarch64::*;
        for t in tiles {
            let r0 = (t / 4) * 8;
            // Two 4-column blocks at once: sixteen independent fma chains
            // hide the fma latency.
            {
                let cb = (t % 4) * 2;
                let z = vdupq_n_f32(0.0);
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (z, z, z, z, z, z, z, z);
                let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6, mut b7) = (z, z, z, z, z, z, z, z);
                for j in 0..NB {
                    let (v, w) = (rows.load4(j * NB + 4 * cb), rows.load4(j * NB + 4 * cb + 4));
                    let c0 = ct.load4(j * NB + r0);
                    let c1 = ct.load4(j * NB + r0 + 4);
                    a0 = vfmaq_laneq_f32::<0>(a0, v, c0);
                    b0 = vfmaq_laneq_f32::<0>(b0, w, c0);
                    a1 = vfmaq_laneq_f32::<1>(a1, v, c0);
                    b1 = vfmaq_laneq_f32::<1>(b1, w, c0);
                    a2 = vfmaq_laneq_f32::<2>(a2, v, c0);
                    b2 = vfmaq_laneq_f32::<2>(b2, w, c0);
                    a3 = vfmaq_laneq_f32::<3>(a3, v, c0);
                    b3 = vfmaq_laneq_f32::<3>(b3, w, c0);
                    a4 = vfmaq_laneq_f32::<0>(a4, v, c1);
                    b4 = vfmaq_laneq_f32::<0>(b4, w, c1);
                    a5 = vfmaq_laneq_f32::<1>(a5, v, c1);
                    b5 = vfmaq_laneq_f32::<1>(b5, w, c1);
                    a6 = vfmaq_laneq_f32::<2>(a6, v, c1);
                    b6 = vfmaq_laneq_f32::<2>(b6, w, c1);
                    a7 = vfmaq_laneq_f32::<3>(a7, v, c1);
                    b7 = vfmaq_laneq_f32::<3>(b7, w, c1);
                }
                if TRANSPOSE {
                    // out[c][r]: 4x4 transposes of the (row, column) tiles.
                    let tile = |p: [float32x4_t; 4], c0: usize, rr: usize| {
                        let t0 = vtrn1q_f32(p[0], p[1]);
                        let t1 = vtrn2q_f32(p[0], p[1]);
                        let t2 = vtrn1q_f32(p[2], p[3]);
                        let t3 = vtrn2q_f32(p[2], p[3]);
                        let cols = [
                            vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))),
                            vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))),
                            vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2))),
                            vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3))),
                        ];
                        for (i, v) in cols.into_iter().enumerate() {
                            vst1q_f32(out.ptr().add((c0 + i) * NB + rr), v);
                        }
                    };
                    tile([a0, a1, a2, a3], 4 * cb, r0);
                    tile([a4, a5, a6, a7], 4 * cb, r0 + 4);
                    tile([b0, b1, b2, b3], 4 * cb + 4, r0);
                    tile([b4, b5, b6, b7], 4 * cb + 4, r0 + 4);
                } else {
                    for (k, (x, y)) in
                        [(a0, b0), (a1, b1), (a2, b2), (a3, b3), (a4, b4), (a5, b5), (a6, b6), (a7, b7)].into_iter().enumerate()
                    {
                        vst1q_f32(out.ptr().add((r0 + k) * NB + 4 * cb), x);
                        vst1q_f32(out.ptr().add((r0 + k) * NB + 4 * cb + 4), y);
                    }
                }
            }
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: every access is inside the 32x32 blocks, and the caller owns
    // the tiles written.
    unsafe {
        use std::arch::wasm32::*;
        for t in tiles {
            let (r0, c0) = ((t / 4) * 8, (t % 4) * 8);
            for r in r0..r0 + 8 {
                let (mut o0, mut o1) = (f32x4_splat(0.0), f32x4_splat(0.0));
                for j in 0..NB {
                    let c = f32x4_splat(ct.at(j * NB + r));
                    o0 = wsimd::fma4(c, rows.load4w(j * NB + c0), o0);
                    o1 = wsimd::fma4(c, rows.load4w(j * NB + c0 + 4), o1);
                }
                if TRANSPOSE {
                    for (k, v) in wsimd::lanes(o0).into_iter().chain(wsimd::lanes(o1)).enumerate() {
                        *out.ptr().add((c0 + k) * NB + r) = v;
                    }
                } else {
                    wsimd::store(out.ptr().add(r * NB + c0), o0);
                    wsimd::store(out.ptr().add(r * NB + c0 + 4), o1);
                }
            }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    for t in tiles {
        let (r0, c0) = ((t / 4) * 8, (t % 4) * 8);
        for r in r0..r0 + 8 {
            let mut o = [0f32; 8];
            for j in 0..NB {
                let c = ct.at(j * NB + r);
                for (k, x) in o.iter_mut().enumerate() {
                    *x = c.mul_add(rows.at(j * NB + c0 + k), *x);
                }
            }
            for (k, &v) in o.iter().enumerate() {
                let cc = c0 + k;
                // SAFETY: the caller owns this tile.
                unsafe { *out.ptr().add(if TRANSPOSE { cc * NB + r } else { r * NB + cc }) = v };
            }
        }
    }
}

/// SiLU the engine's way, `z / (nexp(-z) + 1)`, in place.
fn silu_native(z: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; loads and stores stay in `z`.
    unsafe {
        use std::arch::aarch64::*;
        let (head, tail) = z.as_chunks_mut::<4>();
        for c in head {
            let v = vld1q_f32(c.as_ptr());
            let e = neon::nexp4(vnegq_f32(v));
            vst1q_f32(c.as_mut_ptr(), vdivq_f32(v, vaddq_f32(e, vdupq_n_f32(1.0))));
        }
        for v in tail {
            *v /= nexp(-*v) + 1.0;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // SAFETY: loads and stores stay in `z`.
    unsafe {
        use std::arch::wasm32::*;
        let (head, tail) = z.as_chunks_mut::<4>();
        for c in head {
            let v = wsimd::load(c.as_ptr());
            let e = wsimd::nexp4(f32x4_neg(v));
            wsimd::store(c.as_mut_ptr(), f32x4_div(v, f32x4_add(e, f32x4_splat(1.0))));
        }
        for v in tail {
            *v /= nexp(-*v) + 1.0;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    for v in z {
        *v /= nexp(-*v) + 1.0;
    }
}

/// The engine's 4x4 Sinkhorn: `exp` of each row less its max in f64, then
/// 20 rounds of row and column normalization (pairwise sums, times the
/// reciprocal).
fn sinkhorn64(z: &[f32; 16]) -> [f32; 16] {
    let mut e = [0f64; 16];
    for r in 0..4 {
        let m = z[r * 4].max(z[r * 4 + 1]).max(z[r * 4 + 2].max(z[r * 4 + 3])) as f64;
        for c in 0..4 {
            // SAFETY: a pure libm function.
            e[r * 4 + c] = lexp(z[r * 4 + c] as f64 - m);
        }
    }
    for _ in 0..20 {
        for r in 0..4 {
            let inv = 1.0 / ((e[r * 4] + e[r * 4 + 1]) + (e[r * 4 + 2] + e[r * 4 + 3]));
            for c in 0..4 {
                e[r * 4 + c] *= inv;
            }
        }
        for c in 0..4 {
            let inv = 1.0 / ((e[c] + e[4 + c]) + (e[8 + c] + e[12 + c]));
            for r in 0..4 {
                e[r * 4 + c] *= inv;
            }
        }
    }
    e.map(|v| v as f32)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// [`super::nexp`] on four lanes, op for op.
    #[inline(always)]
    pub unsafe fn nexp4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let c = |b: u32| vdupq_n_f32(f32::from_bits(b));
            let x = vminq_f32(vmaxq_f32(x, c(0xc2b0c0a5)), c(0x42b0c0a5));
            let k = vrndnq_f32(vmulq_f32(x, c(0x3fb8aa3b)));
            let r = vfmaq_f32(x, k, c(0xbf318000));
            let r = vfmaq_f32(r, k, c(0x395e8083));
            let mut p = vfmaq_f32(c(0x3ab743ce), c(0x39506967), r);
            p = vfmaq_f32(c(0x3c088908), r, p);
            p = vfmaq_f32(c(0x3d2aa9c1), r, p);
            p = vfmaq_f32(c(0x3e2aaaaa), r, p);
            p = vfmaq_f32(vdupq_n_f32(0.5), r, p);
            let y = vfmaq_f32(r, vmulq_f32(r, r), p);
            let s = vreinterpretq_f32_s32(vaddq_s32(vshlq_n_s32::<23>(vcvtq_s32_f32(k)), vdupq_n_s32(0x3f80_0000)));
            vfmaq_f32(s, s, y)
        }
    }

    /// See [`super::attend_head_fast`].
    #[target_feature(enable = "neon,dotprod")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn attend_parts(
        sc: &mut super::AttnScratch,
        ls: &super::LayerState,
        qh: &[f32],
        kvh: usize,
        nkv: usize,
        vd: usize,
        keys: &super::Keys,
        attend: Option<(usize, usize)>,
        nested: bool,
        which: std::ops::Range<usize>,
        parts: &mut [super::Part],
    ) {
        use super::{C21, INV_SQRT_QK, Q21, chunk_exp, quant_i8};
        unsafe {
            const QK: usize = 48;
            let mut q8 = [0i8; QK];
            let qs = quant_i8(qh, &mut q8) * INV_SQRT_QK;
            let qp = q8.as_ptr();
            let (q0, q1, q2) = (vld1q_s8(qp), vld1q_s8(qp.add(16)), vld1q_s8(qp.add(32)));
            let n = super::keys_total(keys);
            let (_, chunk) = super::key_chunks(n, nested, nkv);
            super::slot_runs(keys, attend, &mut sc.runs);
            sc.s.resize(chunk + 16, 0.0);
            sc.scale.resize(chunk + 16, 0.0);
            sc.dots.resize(chunk + 16, 0);
            let pad = chunk + 16 * sc.runs.len().max(1);
            sc.digits.resize(3 * pad, 0);
            let k8 = ls.k8.as_ptr();
            let dot = |slot: usize| {
                let r = k8.add((slot * nkv + kvh) * QK);
                let a = vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(r));
                let a = vdotq_s32(a, q1, vld1q_s8(r.add(16)));
                vdotq_s32(a, q2, vld1q_s8(r.add(32)))
            };
            let mut crun: Vec<(usize, usize, usize)> = Vec::with_capacity(4);
            let mut blocks: Vec<(usize, usize, usize)> = Vec::with_capacity(4);
            for ci in which.clone() {
                let c0 = ci * chunk;
                if c0 >= n {
                    break;
                }
                let np = ci - which.start;
                let len = chunk.min(n - c0);
                // This chunk's pieces of the runs, as (chunk index, slot, len).
                crun.clear();
                for &(i0, s0, l) in &sc.runs {
                    let (a, b) = (i0.max(c0), (i0 + l).min(c0 + len));
                    if a < b {
                        crun.push((a - c0, s0 + (a - i0), b - a));
                    }
                }
                // Integer dots and the key and value scales, run by run.
                for &(j0, s0, l) in &crun {
                    let mut i = 0;
                    while i + 4 <= l {
                        let (a, b, cc, d) = (dot(s0 + i), dot(s0 + i + 1), dot(s0 + i + 2), dot(s0 + i + 3));
                        vst1q_s32(sc.dots.as_mut_ptr().add(j0 + i), vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(cc, d)));
                        i += 4;
                    }
                    while i < l {
                        sc.dots[j0 + i] = vaddvq_s32(dot(s0 + i));
                        i += 1;
                    }
                    for i in 0..l {
                        sc.scale[j0 + i] = *ls.ks.get_unchecked((s0 + i) * nkv + kvh);
                    }
                }
                let blocked = len / 4 * 4;
                let s = &mut sc.s[..len];
                let qsv = vdupq_n_f32(qs);
                let mut mv = vdupq_n_f32(f32::NEG_INFINITY);
                let mut i = 0;
                while i < blocked {
                    let d = vcvtq_f32_s32(vld1q_s32(sc.dots.as_ptr().add(i)));
                    let v = vmulq_f32(vmulq_f32(vld1q_f32(sc.scale.as_ptr().add(i)), d), qsv);
                    vst1q_f32(s.as_mut_ptr().add(i), v);
                    mv = vmaxq_f32(mv, v);
                    i += 4;
                }
                let mut m = vmaxvq_f32(mv);
                for i in blocked..len {
                    s[i] = (sc.scale[i] * qs) * sc.dots[i] as f32;
                    m = m.max(s[i]);
                }
                let sum = chunk_exp(s, m);
                for &(j0, s0, l) in &crun {
                    for i in 0..l {
                        sc.scale[j0 + i] = *ls.vs.get_unchecked((s0 + i) * nkv + kvh);
                    }
                }
                let mut wv = vdupq_n_f32(0.0);
                let mut i = 0;
                while i + 4 <= len {
                    let w = vmulq_f32(vld1q_f32(s.as_ptr().add(i)), vld1q_f32(sc.scale.as_ptr().add(i)));
                    vst1q_f32(s.as_mut_ptr().add(i), w);
                    wv = vmaxq_f32(wv, w);
                    i += 4;
                }
                let mut wmax = vmaxvq_f32(wv);
                while i < len {
                    s[i] *= sc.scale[i];
                    wmax = wmax.max(s[i]);
                    i += 1;
                }
                let qw = Q21 / wmax;
                // Digits, each run's padded with zeros to whole blocks, packed
                // run after run.
                let (hi, rest) = sc.digits.split_at_mut(pad);
                let (mid, lo) = rest.split_at_mut(pad);
                let mut at = 0;
                blocks.clear();
                for &(j0, s0, l) in &crun {
                    let padded = l.div_ceil(16) * 16;
                    super::digits21(&s[j0..j0 + l], qw, &mut hi[at..at + l], &mut mid[at..at + l], &mut lo[at..at + l]);
                    for v in [&mut hi[at + l..at + padded], &mut mid[at + l..at + padded], &mut lo[at + l..at + padded]] {
                        v.fill(0);
                    }
                    blocks.push((at, s0, padded / 16));
                    at += padded;
                }
                let cw = wmax * C21;
                let o = &mut parts[np].2;
                let v8t = ls.v8t.as_ptr();
                let base = kvh * vd;
                for dm0 in (0..vd).step_by(4) {
                    let rows = [0, 1, 2, 3].map(|k| v8t.add((base + dm0 + k) * ls.vcap));
                    let mut acc = [[vdupq_n_s32(0); 3]; 4];
                    for &(d0, s0, nb) in &blocks {
                        for b in 0..nb {
                            let off = d0 + 16 * b;
                            let (h, m, l) =
                                (vld1q_s8(hi.as_ptr().add(off)), vld1q_s8(mid.as_ptr().add(off)), vld1q_s8(lo.as_ptr().add(off)));
                            for (a, r) in acc.iter_mut().zip(rows) {
                                let v = vld1q_s8(r.add(s0 + 16 * b));
                                a[0] = vdotq_s32(a[0], h, v);
                                a[1] = vdotq_s32(a[1], m, v);
                                a[2] = vdotq_s32(a[2], l, v);
                            }
                        }
                    }
                    let sum4 = |t: usize| vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(acc[0][t], acc[1][t]), vpaddq_s32(acc[2][t], acc[3][t])));
                    let (sh, sm, sl) = (sum4(0), sum4(1), sum4(2));
                    let v = vmulq_n_f32(vfmaq_n_f32(vfmaq_n_f32(sl, sm, 128.0), sh, 16384.0), cw);
                    vst1q_f32(o.as_mut_ptr().add(dm0), v);
                }
                parts[np].0 = m;
                parts[np].1 = sum;
            }
        }
    }

    /// Dots of a 48-byte query against key rows, four keys per reduction.
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn key_dots48(q8: &[i8], k8: &[i8], slots: &[usize], nkv: usize, kvh: usize, out: &mut [i32]) {
        unsafe {
            let q = q8.as_ptr();
            let (q0, q1, q2) = (vld1q_s8(q), vld1q_s8(q.add(16)), vld1q_s8(q.add(32)));
            let row = |slot: usize| k8.as_ptr().add((slot * nkv + kvh) * 48);
            let dot = |slot: usize| {
                let r = row(slot);
                let a = vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(r));
                let a = vdotq_s32(a, q1, vld1q_s8(r.add(16)));
                vdotq_s32(a, q2, vld1q_s8(r.add(32)))
            };
            let n = slots.len();
            let mut i = 0;
            while i + 4 <= n {
                let (a, b, c, d) = (dot(slots[i]), dot(slots[i + 1]), dot(slots[i + 2]), dot(slots[i + 3]));
                vst1q_s32(out.as_mut_ptr().add(i), vpaddq_s32(vpaddq_s32(a, b), vpaddq_s32(c, d)));
                i += 4;
            }
            while i < n {
                out[i] = vaddvq_s32(dot(slots[i]));
                i += 1;
            }
        }
    }

    /// See [`super::pv_dots4`].
    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn pv_dots4(hi: &[i8], mid: &[i8], lo: &[i8], rows: [&[i8]; 4], s0: usize) -> [[i32; 3]; 4] {
        unsafe {
            let n = hi.len();
            let mut acc = [[vdupq_n_s32(0); 3]; 4];
            let mut i = 0;
            while i + 16 <= n {
                let (h, m, l) = (vld1q_s8(hi.as_ptr().add(i)), vld1q_s8(mid.as_ptr().add(i)), vld1q_s8(lo.as_ptr().add(i)));
                for (a, r) in acc.iter_mut().zip(rows) {
                    let v = vld1q_s8(r.as_ptr().add(s0 + i));
                    a[0] = vdotq_s32(a[0], h, v);
                    a[1] = vdotq_s32(a[1], m, v);
                    a[2] = vdotq_s32(a[2], l, v);
                }
                i += 16;
            }
            let mut out = acc.map(|a| a.map(|x| vaddvq_s32(x)));
            while i < n {
                for (o, r) in out.iter_mut().zip(rows) {
                    let v = *r.get_unchecked(s0 + i) as i32;
                    o[0] += *hi.get_unchecked(i) as i32 * v;
                    o[1] += *mid.get_unchecked(i) as i32 * v;
                    o[2] += *lo.get_unchecked(i) as i32 * v;
                }
                i += 1;
            }
            out
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn idot3(a: &[i8], b: &[i8], c: &[i8], v: &[i8]) -> (i32, i32, i32) {
        unsafe {
            let n = v.len();
            let (mut x, mut y, mut z) = (vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0));
            let mut i = 0;
            while i + 16 <= n {
                let vv = vld1q_s8(v.as_ptr().add(i));
                x = vdotq_s32(x, vld1q_s8(a.as_ptr().add(i)), vv);
                y = vdotq_s32(y, vld1q_s8(b.as_ptr().add(i)), vv);
                z = vdotq_s32(z, vld1q_s8(c.as_ptr().add(i)), vv);
                i += 16;
            }
            let (mut sx, mut sy, mut sz) = (vaddvq_s32(x), vaddvq_s32(y), vaddvq_s32(z));
            while i < n {
                let vv = *v.get_unchecked(i) as i32;
                sx += *a.get_unchecked(i) as i32 * vv;
                sy += *b.get_unchecked(i) as i32 * vv;
                sz += *c.get_unchecked(i) as i32 * vv;
                i += 1;
            }
            (sx, sy, sz)
        }
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn idot(a: &[i8], b: &[i8]) -> i32 {
        unsafe {
            let n = a.len();
            let mut acc = vdupq_n_s32(0);
            let mut i = 0;
            while i + 16 <= n {
                acc = vdotq_s32(acc, vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i)));
                i += 16;
            }
            let mut s = vaddvq_s32(acc);
            while i < n {
                s += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
                i += 1;
            }
            s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot16_matches_the_scalar_order() {
        let x: Vec<f32> = (0..768).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let y: Vec<f32> = (0..768).map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.021).collect();
        let mut a = [[0f32; 4]; 4];
        for c in 0..48 {
            for j in 0..4 {
                for l in 0..4 {
                    let i = c * 16 + j * 4 + l;
                    a[j][l] = x[i].mul_add(y[i], a[j][l]);
                }
            }
        }
        let t: [f32; 4] = std::array::from_fn(|l| ((a[2][l] + a[3][l]) + a[1][l]) + a[0][l]);
        assert_eq!(dot16(&x, &y).to_bits(), ((t[0] + t[1]) + (t[2] + t[3])).to_bits());
    }

    #[test]
    fn fast_attention_matches_reference() {
        let (nkv, qk, vd) = (2usize, 48usize, 64usize);
        let mut ls = LayerState::default();
        ls.reserve(800, nkv, qk, vd);
        for (i, v) in ls.k8.iter_mut().enumerate() {
            *v = ((i * 37 % 251) as i32 - 125) as i8;
        }
        for (i, v) in ls.v8t.iter_mut().enumerate() {
            *v = ((i * 53 % 249) as i32 - 124) as i8;
        }
        for (i, v) in ls.ks.iter_mut().enumerate() {
            *v = 0.01 + (i % 7) as f32 * 0.003;
        }
        for (i, v) in ls.vs.iter_mut().enumerate() {
            *v = 0.02 + (i % 5) as f32 * 0.002;
        }
        let q: Vec<f32> = (0..qk).map(|i| (i as f32 * 0.3).sin() * 3.0).collect();
        let mut sc = AttnScratch::default();
        type Case = (Keys, Option<(usize, usize)>);
        let cases: Vec<Case> = vec![
            (keys_of(0..1, 0..0), None),
            (keys_of(0..37, 0..0), None),
            (keys_of(0..300, 0..0), None),
            (keys_of(0..416, 416..460), Some((416, 256))),
            (keys_of(0..41, 250..297), Some((41, 256))),
            (keys_of(0..41, 290..560), Some((41, 256))),
            (keys_of(0..20, 20..20), Some((41, 256))),
        ];
        for (keys, attend) in cases {
            for nested in [false, true] {
                for kvh in 0..nkv {
                    let (mut a, mut b) = (vec![0f32; vd], vec![0f32; vd]);
                    attend_head_ref(&mut sc, &ls, &q, kvh, nkv, qk, vd, &keys, attend, nested, &mut a);
                    attend_head_fast(&mut sc, &ls, &q, kvh, nkv, qk, vd, &keys, attend, nested, &mut b);
                    for (x, y) in a.iter().zip(&b) {
                        assert_eq!(x.to_bits(), y.to_bits(), "{keys:?} {attend:?} nested {nested}");
                    }
                }
            }
        }
    }

    #[test]
    fn nexp_is_close_to_exp() {
        for i in -200..200 {
            let x = i as f32 * 0.37;
            let (a, b) = (nexp(x), x.exp());
            assert!((a - b).abs() <= 2e-7 * b.max(1e-30), "{x}: {a} vs {b}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// `cargo test --release -p needle-engine infer::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_attend_head() {
        let (nkv, qk, vd) = (2usize, 48usize, 64usize);
        let n = 460usize;
        let mut ls = LayerState::default();
        ls.reserve(n, nkv, qk, vd);
        for (i, v) in ls.k8.iter_mut().enumerate() {
            *v = ((i * 37 % 255) as i32 - 127) as i8;
        }
        for (i, v) in ls.v8t.iter_mut().enumerate() {
            *v = ((i * 53 % 255) as i32 - 127) as i8;
        }
        for (i, v) in ls.ks.iter_mut().enumerate() {
            *v = 0.01 + (i % 7) as f32 * 0.001;
        }
        for (i, v) in ls.vs.iter_mut().enumerate() {
            *v = 0.02 + (i % 5) as f32 * 0.001;
        }
        let q: Vec<f32> = (0..qk).map(|i| (i as f32 * 0.3).sin()).collect();
        let keys = keys_of(0..416, 416..n);
        let mut out = vec![0f32; vd];
        for name in ["full"] {
            let reps = std::env::var("NX_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20000);
            let t = web_time::Instant::now();
            for _ in 0..reps {
                attend_head(&ls, std::hint::black_box(&q), 0, nkv, qk, vd, &keys, Some((416, 256)), false, &mut out);
                std::hint::black_box(&mut out);
            }
            eprintln!("attend_head {name}: {:.2} us", t.elapsed().as_secs_f64() / reps as f64 * 1e6);
        }
        for nested in [false, true] {
            let reps = std::env::var("NX_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20000);
            let t = web_time::Instant::now();
            for _ in 0..reps {
                attend_head(&ls, std::hint::black_box(&q), 0, nkv, qk, vd, &keys, Some((416, 256)), nested, &mut out);
                std::hint::black_box(&mut out);
            }
            eprintln!("attend_head nested={nested}: {:.2} us", t.elapsed().as_secs_f64() / reps as f64 * 1e6);
        }
    }

    #[test]
    #[ignore]
    fn bench_post_pieces() {
        let z: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.37).sin()).collect();
        let a: Vec<f16> = (0..1024).map(|i| f16::from_f32((i as f32 * 0.11).cos() * 0.1)).collect();
        let mut t = [0f32; 1024];
        let n = 20000;
        let t0 = web_time::Instant::now();
        for _ in 0..n {
            kron_native(std::hint::black_box(&z), &a, &a, &mut t);
            std::hint::black_box(&mut t);
        }
        eprintln!("kron stage {:.2} us", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
        let mut zz = z.clone();
        let t0 = web_time::Instant::now();
        for _ in 0..n {
            silu_native(std::hint::black_box(&mut zz));
        }
        eprintln!("silu 1024 {:.2} us", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
        let zr: [f32; 16] = std::array::from_fn(|i| i as f32 * 0.3 - 2.0);
        let t0 = web_time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(sinkhorn64(std::hint::black_box(&zr)));
        }
        eprintln!("sinkhorn {:.2} us", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
        let x: Vec<f32> = (0..768).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut o = vec![0f32; 768];
        let t0 = web_time::Instant::now();
        for _ in 0..n {
            zcn(std::hint::black_box(&x), &x, &mut o);
            std::hint::black_box(&mut o);
        }
        eprintln!("zcn 768 {:.2} us", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
    }
}

#[cfg(test)]
mod bench_f16 {
    /// `cargo test --release -p needle-engine bench_f16 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn widen_cost() {
        use half::f16;
        let src: Vec<f16> = (0..1024).map(|i| f16::from_f32((i as f32 * 0.37).sin())).collect();
        let srcf: Vec<f32> = src.iter().map(|v| v.to_f32()).collect();
        let x: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.11).cos()).collect();
        let n = 200_000;
        let t = web_time::Instant::now();
        let mut acc = 0f32;
        for _ in 0..n {
            let mut a = [0f32; 4];
            for (k, (&w, &v)) in src.iter().zip(&x).enumerate() {
                a[k % 4] = w.to_f32().mul_add(v, a[k % 4]);
            }
            acc += a[0] + a[1] + a[2] + a[3];
        }
        eprintln!("f16 scalar to_f32 dot 1024: {:.0} ns ({acc})", t.elapsed().as_secs_f64() / n as f64 * 1e9);
        let t = web_time::Instant::now();
        let mut acc = 0f32;
        for _ in 0..n {
            let mut a = [0f32; 4];
            for (k, (&w, &v)) in srcf.iter().zip(&x).enumerate() {
                a[k % 4] = w.mul_add(v, a[k % 4]);
            }
            acc += a[0] + a[1] + a[2] + a[3];
        }
        eprintln!("f32 dot 1024: {:.0} ns ({acc})", t.elapsed().as_secs_f64() / n as f64 * 1e9);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON with fp16 conversions is baseline on Apple silicon.
        unsafe {
            use std::arch::aarch64::*;
            let t = web_time::Instant::now();
            let mut acc = 0f32;
            for _ in 0..n {
                let mut a = vdupq_n_f32(0.0);
                for i in (0..1024).step_by(4) {
                    let w = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(src.as_ptr().add(i).cast())));
                    a = vfmaq_f32(a, w, vld1q_f32(x.as_ptr().add(i)));
                }
                acc += vaddvq_f32(a);
            }
            eprintln!("f16 neon widen dot 1024: {:.0} ns ({acc})", t.elapsed().as_secs_f64() / n as f64 * 1e9);
        }
    }
}
