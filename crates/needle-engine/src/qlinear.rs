//! Matmuls straight off CQ-packed weights, with the native engine's exact
//! arithmetic.
//!
//! A CQ group reconstructs as `w = (cb[idx] * norm) H` with `H` the
//! normalised Walsh-Hadamard matrix, and `H` is symmetric, so
//! `x . w = norm * sum_i cb[idx_i] * (H x)_i`. The input is rotated once per
//! group (a 128-point FWHT shared by every output row) and quantized to int8
//! per group (A8); each weight row is then a codebook lookup (`tbl`) and
//! int8 dot products (`sdot`), scaled by the group norm. Weights stay at 2-4
//! bits in memory, which is what makes decode fast: it is bandwidth-bound.
//!
//! The integer sums are exact, so only the float combine decides the result.
//! libneedle3 has three combines, reproduced here bit for bit:
//!
//! * decode (one token): four float lanes per row, each fed by a fixed
//!   quarter of every group, scale `(sx * cbs) * norm`, lanes summed
//!   `(l0 + l1) + (l2 + l3)`;
//! * prefill (a chunk of tokens): one int sum per group, an `fma` chain over
//!   groups, scale `(norm * cbs) * sx`, except the chunk's last `T % 4`
//!   tokens, which use `norm * (cbs * sx)`;
//! * single row (the logits of a few ids): as prefill's block tokens.
//!
//! Codes stay in the archive's layout (LSB-first, consecutive weights), as
//! in the engine. One 16-byte load holds 64 2-bit or 32 4-bit weights; a
//! nibble indexes a 16-entry table (two of them for 2-bit codes, one per
//! code of the pair), so the expanded vectors hold every fourth (or second)
//! weight. The activations are stored in the matching order instead
//! ([`QAct`]), which also fixes the decode kernel's lane membership: 2-bit
//! lane `(p % 64) / 16`, 4-bit lane `(p % 32) / 8`.

use half::f16;
use needle_core::cact::CqView;
use needle_core::quant::{TERNARY_RECORD_BITS, codebook, cq_pack, rotate, unpack_lsb_into};

use crate::team::{SyncPtr, share, team};

pub const GROUP: usize = 128;

/// `1/127` as the engine rounds it; every int8 scale is `amax * K127`.
pub const K127: f32 = f32::from_bits(0x3c010204);

#[derive(Clone)]
pub struct QMat {
    pub out: usize,
    pub in_dim: usize,
    groups: usize,
    /// Bits per code (2 or 4); ternary records pack at 2.
    bits: u8,
    /// Codes as the archive packs them, `out * groups * bytes_per_group`.
    codes: Vec<u8>,
    /// Group L2 norms as stored (f16), `[out, groups]`.
    norms: Vec<f16>,
    /// Nibble tables: the first (or only) and second code of a nibble.
    tables: [[i8; 16]; 2],
    /// Code to int8 codebook value.
    vals: [i8; 4 * 4],
    /// Scale of the int8 codebook.
    cbs: f32,
    /// Code to f32 codebook value (for dequantizing rows).
    fvals: [f32; 16],
}

/// Rotated, int8-quantized activations for one or more input rows, each
/// group stored in the code layout of the matrices it feeds (`bits`).
pub struct QAct {
    pub rows: usize,
    groups: usize,
    bits: u8,
    xq: Vec<i8>,
    /// Per (row, group) scale.
    sx: Vec<f32>,
}

/// The engine's int8 quantizer: `s = amax * K127` (1 when all zero), then
/// `x / s` rounded half away from zero and saturated. Returns `s`.
#[inline]
pub fn quant_i8(x: &[f32], q: &mut [i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if x.len().is_multiple_of(16) {
        // SAFETY: NEON is baseline on aarch64; the loops stay inside the
        // slices (whole 16-element blocks).
        return unsafe { neon::quant_i8(x, q) };
    }
    let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = if m > 0.0 { m * K127 } else { 1.0 };
    quant_with(x, 1.0 / s, q);
    s
}

/// The scale [`quant_i8`] would pick for `x`.
#[inline]
pub fn quant_scale(x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if x.len().is_multiple_of(4) {
        // SAFETY: NEON is baseline on aarch64; whole 4-element blocks.
        let m = unsafe {
            use std::arch::aarch64::*;
            let mut m = vdupq_n_f32(0.0);
            for i in (0..x.len()).step_by(4) {
                m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x.as_ptr().add(i))));
            }
            vmaxvq_f32(m)
        };
        return if m > 0.0 { m * K127 } else { 1.0 };
    }
    let m = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if m > 0.0 { m * K127 } else { 1.0 }
}

/// Quantize `x` with a given scale (`inv = 1 / scale`), as [`quant_i8`] does
/// once it has its scale.
#[inline]
pub fn quant_with(x: &[f32], inv: f32, q: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    if x.len().is_multiple_of(16) {
        // SAFETY: NEON is baseline on aarch64; whole 16-element blocks.
        return unsafe { neon::quant_with(x, inv, q) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if x.len().is_multiple_of(16) {
        // SAFETY: whole 16-element blocks: round half away, then saturating
        // narrows, as NEON does.
        unsafe {
            use std::arch::wasm32::*;
            let inv = f32x4_splat(inv);
            for i in (0..x.len()).step_by(16) {
                let c = |k: usize| {
                    i32x4_trunc_sat_f32x4(crate::infer::wsimd::round_away(f32x4_mul(v128_load(x.as_ptr().add(i + 4 * k).cast()), inv)))
                };
                let lo = i16x8_narrow_i32x4(c(0), c(1));
                let hi = i16x8_narrow_i32x4(c(2), c(3));
                v128_store(q.as_mut_ptr().add(i).cast(), i8x16_narrow_i16x8(lo, hi));
            }
        }
        return;
    }
    for (o, v) in q.iter_mut().zip(x) {
        *o = (v * inv).round().clamp(-128.0, 127.0) as i8;
    }
}

/// Position of input `p` (within a group) in the activation layout for
/// `bits`-bit codes.
#[inline]
fn act_pos(p: usize, bits: u8) -> usize {
    if bits == 2 { (p % 4) * 32 + p / 4 } else { (p % 2) * 64 + p / 2 }
}

/// Rotate and int8-quantize group `g` of `x` (one row) into `q`, in the
/// layout of `bits`-bit codes, returning its scale: the unit of work of
/// [`QAct::new`].
pub fn prep_group(x: &[f32], g: usize, q: &mut [i8], bits: u8) -> f32 {
    let mut buf = [0f32; GROUP];
    let start = g * GROUP;
    let take = GROUP.min(x.len() - start);
    buf[..take].copy_from_slice(&x[start..start + take]);
    rotate(&mut buf);
    let mut nat = [0i8; GROUP];
    let s = quant_i8(&buf, &mut nat);
    for (p, &v) in nat.iter().enumerate() {
        q[act_pos(p, bits)] = v;
    }
    s
}

impl QAct {
    /// A single-row activation whose groups the caller fills with
    /// [`prep_group`] (in parallel).
    pub fn empty(in_dim: usize, bits: u8) -> Self {
        Self::empty_rows(1, in_dim, bits)
    }

    /// As [`QAct::empty`] for `rows` rows (row `r`, group `g` at unit
    /// `r * groups + g`).
    pub fn empty_rows(rows: usize, in_dim: usize, bits: u8) -> Self {
        let groups = in_dim.div_ceil(GROUP);
        Self { rows, groups, bits, xq: vec![0; rows * groups * GROUP], sx: vec![0.0; rows * groups] }
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Raw group buffers for [`prep_group`].
    pub fn parts_mut(&mut self) -> (&mut [i8], &mut [f32]) {
        (&mut self.xq, &mut self.sx)
    }

    /// Prepare `x [rows, in_dim]` for `bits`-bit matrices, on the team when
    /// there is enough of it.
    pub fn new(x: &[f32], rows: usize, in_dim: usize, bits: u8) -> Self {
        let mut a = Self::empty_rows(rows, in_dim, bits);
        let groups = a.groups;
        let units = rows * groups;
        let (qp, sp) = (SyncPtr(a.xq.as_mut_ptr()), SyncPtr(a.sx.as_mut_ptr()));
        let unit = |u: usize| {
            let (r, g) = (u / groups, u % groups);
            // SAFETY: each unit owns its own group of `xq` and slot of `sx`.
            unsafe { *sp.ptr().add(u) = prep_group(&x[r * in_dim..(r + 1) * in_dim], g, qp.slice(u * GROUP..(u + 1) * GROUP), bits) };
        };
        if units >= 24 {
            team().run(&|tid, n| share(units, tid, n).for_each(unit));
        } else {
            (0..units).for_each(unit);
        }
        a
    }
}

/// The int8 codebook and its scale, as the engine derives them.
fn int8_codebook(cb: &[f32]) -> (Vec<i8>, f32) {
    let m = cb.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = if m > 0.0 { m * K127 } else { 1.0 };
    let inv = 1.0 / scale;
    (cb.iter().map(|v| (v * inv).round().clamp(-128.0, 127.0) as i8).collect(), scale)
}

impl QMat {
    /// From an archive CQ tensor (`[out, in]`, group 128).
    pub fn from_cq(v: &CqView<'_>) -> Self {
        assert_eq!(v.group, GROUP, "CQ group size {} is not supported", v.group);
        let groups = v.in_pad() / GROUP;
        let ternary = v.bits == TERNARY_RECORD_BITS;
        let (cb, bits) = if ternary { (codebook(TERNARY_RECORD_BITS, GROUP), 2) } else { (codebook(v.bits, GROUP), v.bits as u8) };
        // Ternary records pack code 3 for the first value and 0..2 for the
        // rest.
        let code_of = |c: usize| if ternary { if c == 3 { 0 } else { c + 1 } } else { c };
        let norms = (0..v.out * groups).map(|i| f16::from_f32(v.norm(i / groups, i % groups))).collect();
        let row_bytes = v.row_bytes();
        let bpg = GROUP * bits as usize / 8;
        let mut codes = vec![0u8; v.out * groups * bpg];
        for r in 0..v.out {
            codes[r * groups * bpg..(r + 1) * groups * bpg].copy_from_slice(&v.packed[r * row_bytes..r * row_bytes + groups * bpg]);
        }
        Self::build(v.out, v.in_dim, bits, &cb, code_of, codes, norms)
    }

    /// Quantize an f32 `[out, in]` matrix to CQ-4 for the fast path.
    pub fn from_f32(w: &[f32], out: usize, in_dim: usize) -> Self {
        let p = cq_pack(w, in_dim, 4, GROUP);
        Self::build(out, in_dim, 4, &codebook(4, GROUP), |c| c, p.packed, p.norms)
    }

    fn build(out: usize, in_dim: usize, bits: u8, cb: &[f32], code_of: impl Fn(usize) -> usize, codes: Vec<u8>, norms: Vec<f16>) -> Self {
        let groups = in_dim.div_ceil(GROUP);
        let (cb8, cbs) = int8_codebook(cb);
        let ncodes = 1usize << bits;
        let mut vals = [0i8; 16];
        let mut fvals = [0f32; 16];
        for c in 0..ncodes {
            let k = code_of(c);
            vals[c] = cb8.get(k).copied().unwrap_or(0);
            fvals[c] = cb.get(k).copied().unwrap_or(0.0);
        }
        let mut tables = [[0i8; 16]; 2];
        for n in 0..16 {
            if bits == 2 {
                tables[0][n] = vals[n & 3];
                tables[1][n] = vals[n >> 2];
            } else {
                tables[0][n] = vals[n];
            }
        }
        Self { out, in_dim, groups, bits, codes, norms, tables, vals, cbs, fvals }
    }

    /// Rows `r0..r1` as their own matrix.
    pub fn slice_rows(&self, r0: usize, r1: usize) -> QMat {
        let (g, bpg) = (self.groups, self.bytes_per_group());
        QMat {
            out: r1 - r0,
            codes: self.codes[r0 * g * bpg..r1 * g * bpg].to_vec(),
            norms: self.norms[r0 * g..r1 * g].to_vec(),
            ..self.clone_meta()
        }
    }

    fn clone_meta(&self) -> QMat {
        QMat { codes: vec![], norms: vec![], ..*self }
    }

    fn compatible(&self, o: &QMat) -> bool {
        self.in_dim == o.in_dim && self.bits == o.bits && self.tables == o.tables && self.fvals == o.fvals && self.cbs == o.cbs
    }

    /// Bytes of packed codes and norms.
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.norms.len() * 2
    }

    /// The activation layout this matrix reads (its code width).
    pub fn act_bits(&self) -> u8 {
        self.bits
    }

    fn bytes_per_group(&self) -> usize {
        GROUP * self.bits as usize / 8
    }

    /// The codes of one group of one row, in natural order.
    fn group_codes(&self, r: usize, g: usize, out: &mut [u8; GROUP]) {
        let bpg = self.bytes_per_group();
        let at = (r * self.groups + g) * bpg;
        unpack_lsb_into(&self.codes[at..at + bpg], self.bits as u32, out);
    }

    /// Row `r` dequantized to f32 (embedding and engram lookups): each group
    /// `cb[code] * norm`, rotated.
    pub fn row(&self, r: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.groups * GROUP];
        let mut codes = [0u8; GROUP];
        for g in 0..self.groups {
            self.group_codes(r, g, &mut codes);
            let norm = self.norms[r * self.groups + g].to_f32();
            let dst = &mut out[g * GROUP..(g + 1) * GROUP];
            for (o, &c) in dst.iter_mut().zip(codes.iter()) {
                *o = self.fvals[c as usize] * norm;
            }
            rotate(dst);
        }
        out.truncate(self.in_dim);
        out
    }

    /// Integer sums of row `r`, group `g` against token row `t` of `a`, in
    /// the decode kernel's four lanes.
    fn lane_sums(&self, r: usize, g: usize, a: &QAct, t: usize) -> [i32; 4] {
        let mut codes = [0u8; GROUP];
        self.group_codes(r, g, &mut codes);
        let x = &a.xq[(t * a.groups + g) * GROUP..(t * a.groups + g + 1) * GROUP];
        let mut s = [0i32; 4];
        for (p, &c) in codes.iter().enumerate() {
            let lane = if self.bits == 2 { (p % 64) / 16 } else { (p % 32) / 8 };
            s[lane] += self.vals[c as usize] as i32 * x[act_pos(p, self.bits)] as i32;
        }
        s
    }

    /// Decode combine for row `r` (portable reference).
    fn decode_row_ref(&self, r: usize, a: &QAct) -> f32 {
        let mut lanes = [0f32; 4];
        for g in 0..self.groups {
            let s = self.lane_sums(r, g, a, 0);
            let sc = (a.sx[g] * self.cbs) * self.norms[r * self.groups + g].to_f32();
            for (l, v) in lanes.iter_mut().zip(s) {
                *l = (v as f32).mul_add(sc, *l);
            }
        }
        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    }

    /// Prefill combine for row `r`, token `t` (portable reference).
    fn prefill_row_ref(&self, r: usize, a: &QAct, t: usize, block: bool) -> f32 {
        let mut acc = 0f32;
        for g in 0..self.groups {
            let s: i32 = self.lane_sums(r, g, a, t).iter().sum();
            let sx = a.sx[t * a.groups + g];
            let norm = self.norms[r * self.groups + g].to_f32();
            let sc = if block { (norm * self.cbs) * sx } else { norm * (self.cbs * sx) };
            acc = (s as f32).mul_add(sc, acc);
        }
        acc
    }

    /// Rows `rows` against the single token of `a` (the decode kernel).
    pub fn decode_rows(&self, rows: std::ops::Range<usize>, a: &QAct, out: &mut [f32]) {
        debug_assert!(a.groups == self.groups && a.bits == self.bits);
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let (g, bpg) = (self.groups, self.bytes_per_group());
            let codes = &self.codes[rows.start * g * bpg..rows.end * g * bpg];
            let norms = &self.norms[rows.start * g..rows.end * g];
            // SAFETY: dotprod detected; slices cover the rows and one token.
            unsafe { neon::decode(self.bits, codes, norms, g, &a.xq, &a.sx, &self.tables, self.cbs, out) };
            return;
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let (g, bpg) = (self.groups, self.bytes_per_group());
            let codes = &self.codes[rows.start * g * bpg..rows.end * g * bpg];
            let norms = &self.norms[rows.start * g..rows.end * g];
            // SAFETY: simd128 is enabled at build time; slices cover the rows
            // and one token.
            unsafe { wasm::decode(self.bits, codes, norms, g, &a.xq, &a.sx, &self.tables, self.cbs, out) };
            return;
        }
        #[allow(unreachable_code)]
        for (o, r) in out.iter_mut().zip(rows) {
            *o = self.decode_row_ref(r, a);
        }
    }

    /// Rows `rows` against every token of `a` (the prefill kernel), written
    /// to `y[t * ldy + col0 + r]`.
    pub fn prefill_rows(&self, rows: std::ops::Range<usize>, a: &QAct, y: SyncPtr<f32>, ldy: usize, col0: usize) {
        debug_assert!(a.groups == self.groups && a.bits == self.bits);
        let blocked = a.rows / 4 * 4;
        let put = |r: usize, t: usize, v: &[f32]| {
            for (k, vk) in v.iter().enumerate() {
                // SAFETY: column `col0 + r` belongs to the caller's row range.
                unsafe { *y.ptr().add((t + k) * ldy + col0 + r) = *vk };
            }
        };
        for r in rows {
            let mut t = 0;
            while t + 8 <= blocked {
                put(r, t, &self.prefill_rown::<8>(r, a, t, true));
                t += 8;
            }
            while t + 4 <= blocked {
                put(r, t, &self.prefill_rown::<4>(r, a, t, true));
                t += 4;
            }
            while t < a.rows {
                put(r, t, &self.prefill_rown::<1>(r, a, t, false));
                t += 1;
            }
        }
    }

    /// Row `r` against `N` token rows from `t`, every one a block token or
    /// every one a remainder token.
    #[inline]
    fn prefill_rown<const N: usize>(&self, r: usize, a: &QAct, t: usize, block: bool) -> [f32; N] {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            let (g, bpg) = (self.groups, self.bytes_per_group());
            let codes = &self.codes[r * g * bpg..(r + 1) * g * bpg];
            let norms = &self.norms[r * g..(r + 1) * g];
            let stride = g * GROUP;
            let xq = &a.xq[t * stride..(t + N) * stride];
            let sx = &a.sx[t * g..(t + N) * g];
            // SAFETY: dotprod detected; slices cover `N` token rows.
            return unsafe { neon::prefill_n::<N>(self.bits, codes, &self.tables, norms, self.cbs, xq, sx, stride, block) };
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let (g, bpg) = (self.groups, self.bytes_per_group());
            let codes = &self.codes[r * g * bpg..(r + 1) * g * bpg];
            let norms = &self.norms[r * g..(r + 1) * g];
            let stride = g * GROUP;
            let xq = &a.xq[t * stride..(t + N) * stride];
            let sx = &a.sx[t * g..(t + N) * g];
            // SAFETY: simd128 is enabled at build time; slices cover `N`
            // token rows.
            return unsafe { wasm::prefill_n::<N>(self.bits, codes, &self.tables, norms, self.cbs, xq, sx, stride, block) };
        }
        #[allow(unreachable_code)]
        std::array::from_fn(|k| self.prefill_row_ref(r, a, t + k, block))
    }

    /// Row `r` against token 0 of `a` with the single-row kernel (the
    /// engine's sparse logits).
    pub fn single_row(&self, r: usize, a: &QAct) -> f32 {
        self.prefill_rown::<1>(r, a, 0, true)[0]
    }

    /// `y[t, o] = x[t] . w[o]` for prepared activations, with the decode
    /// kernel for one token and the prefill kernel for more.
    pub fn matmul(&self, a: &QAct) -> Vec<f32> {
        let mut y = vec![0f32; a.rows * self.out];
        self.matmul_into(a, &mut y, self.out, 0);
        y
    }

    /// As [`QMat::matmul`], writing column block `[col0, col0 + out)` of a
    /// `[rows, ldy]` output (so fused projections share one buffer).
    pub fn matmul_into(&self, a: &QAct, y: &mut [f32], ldy: usize, col0: usize) {
        assert!(a.groups == self.groups && a.bits == self.bits);
        let yp = SyncPtr(y.as_mut_ptr());
        let work = a.rows * self.out * self.groups;
        let run = |rows: std::ops::Range<usize>| {
            if a.rows == 1 {
                // SAFETY: callers split disjoint row ranges of the output.
                let dst = unsafe { std::slice::from_raw_parts_mut(yp.ptr().add(col0 + rows.start), rows.len()) };
                self.decode_rows(rows, a, dst);
            } else {
                self.prefill_rows(rows, a, yp, ldy, col0);
            }
        };
        // Below this many 128-wide groups a single core beats waking the team.
        if work < 1024 {
            run(0..self.out);
        } else {
            team().run(&|tid, n| run(share(self.out, tid, n)));
        }
    }
}

/// Several matrices over one input, output side by side (fused
/// projections). Neighbours with the same width and codebook are merged so
/// one parallel pass covers them.
#[derive(Clone)]
pub struct QLinear {
    parts: Vec<QMat>,
    pub out: usize,
    pub in_dim: usize,
}

impl QLinear {
    pub fn new(parts: Vec<QMat>) -> Self {
        assert!(!parts.is_empty());
        let in_dim = parts[0].in_dim;
        let out = parts.iter().map(|p| p.out).sum();
        let mut merged: Vec<QMat> = vec![];
        for p in parts {
            assert_eq!(p.in_dim, in_dim);
            match merged.last_mut() {
                Some(last) if last.compatible(&p) => {
                    last.codes.extend_from_slice(&p.codes);
                    last.norms.extend_from_slice(&p.norms);
                    last.out += p.out;
                }
                _ => merged.push(p),
            }
        }
        assert!(merged.iter().all(|p| p.bits == merged[0].bits), "fused parts must share a code width");
        Self { parts: merged, out, in_dim }
    }

    /// The activation layout these matrices read.
    pub fn act_bits(&self) -> u8 {
        self.parts[0].bits
    }

    pub fn matmul(&self, a: &QAct) -> Vec<f32> {
        let mut y = vec![0f32; a.rows * self.out];
        let mut col = 0;
        for p in &self.parts {
            p.matmul_into(a, &mut y, self.out, col);
            col += p.out;
        }
        y
    }

    /// Output rows `rows` (across fused parts) for every token of `a`, with
    /// the kernel the token count calls for, into `y[t * ldy + r]`.
    pub fn rows_into(&self, rows: std::ops::Range<usize>, a: &QAct, y: SyncPtr<f32>, ldy: usize) {
        let mut base = 0;
        for p in &self.parts {
            let (lo, hi) = (rows.start.max(base), rows.end.min(base + p.out));
            if lo < hi {
                if a.rows == 1 {
                    // SAFETY: the caller owns output rows `rows`.
                    let dst = unsafe { std::slice::from_raw_parts_mut(y.ptr().add(lo), hi - lo) };
                    p.decode_rows(lo - base..hi - base, a, dst);
                } else {
                    p.prefill_rows(lo - base..hi - base, a, y, ldy, base);
                }
            }
            base += p.out;
        }
    }

    /// Prepare `x [rows, in]` and multiply.
    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.matmul(&QAct::new(x, rows, self.in_dim, self.act_bits()))
    }

    pub fn bytes(&self) -> usize {
        self.parts.iter().map(QMat::bytes).sum()
    }

    /// All rows as a dense `[out, in]` f32 matrix.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(self.out * self.in_dim);
        for p in &self.parts {
            for r in 0..p.out {
                w.extend(p.row(r));
            }
        }
        w
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use half::f16;

    use super::{GROUP, K127};

    /// [`super::quant_i8`] on whole 16-element blocks: `fcvtas` (round half
    /// away) then saturating narrows, as the engine does.
    #[inline]
    pub unsafe fn quant_i8(x: &[f32], q: &mut [i8]) -> f32 {
        unsafe {
            let n = x.len();
            let p = x.as_ptr();
            let mut m = vdupq_n_f32(0.0);
            for i in (0..n).step_by(4) {
                m = vmaxq_f32(m, vabsq_f32(vld1q_f32(p.add(i))));
            }
            let m = vmaxvq_f32(m);
            let s = if m > 0.0 { m * K127 } else { 1.0 };
            quant_with(x, 1.0 / s, q);
            s
        }
    }

    #[inline]
    pub unsafe fn quant_with(x: &[f32], inv: f32, q: &mut [i8]) {
        unsafe {
            let p = x.as_ptr();
            let inv = vdupq_n_f32(inv);
            for i in (0..x.len()).step_by(16) {
                let c = |k: usize| vcvtaq_s32_f32(vmulq_f32(vld1q_f32(p.add(i + 4 * k)), inv));
                let lo = vcombine_s16(vqmovn_s32(c(0)), vqmovn_s32(c(1)));
                let hi = vcombine_s16(vqmovn_s32(c(2)), vqmovn_s32(c(3)));
                vst1q_s8(q.as_mut_ptr().add(i), vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
            }
        }
    }

    /// A group's 128 weights as eight 16-lane vectors, in the activation
    /// layout's order (so vector `v` meets activation bytes `16v..16v+16`).
    #[inline(always)]
    unsafe fn expand(bits: u8, c: *const u8, t0: int8x16_t, t1: int8x16_t) -> [int8x16_t; 8] {
        unsafe {
            let mask = vdupq_n_u8(15);
            let mut w = [vdupq_n_s8(0); 8];
            if bits == 2 {
                // Byte j holds codes 4j..4j+3: low nibble the first pair,
                // high nibble the second; `t0`/`t1` read a pair's codes.
                for m in 0..2 {
                    let b = vld1q_u8(c.add(16 * m));
                    let (lo, hi) = (vandq_u8(b, mask), vshrq_n_u8(b, 4));
                    w[m] = vqtbl1q_s8(t0, lo);
                    w[2 + m] = vqtbl1q_s8(t1, lo);
                    w[4 + m] = vqtbl1q_s8(t0, hi);
                    w[6 + m] = vqtbl1q_s8(t1, hi);
                }
            } else {
                // Byte j holds codes 2j (low nibble) and 2j+1 (high).
                for m in 0..4 {
                    let b = vld1q_u8(c.add(16 * m));
                    w[m] = vqtbl1q_s8(t0, vandq_u8(b, mask));
                    w[4 + m] = vqtbl1q_s8(t0, vshrq_n_u8(b, 4));
                }
            }
            w
        }
    }

    /// `(l0 + l1) + (l2 + l3)`.
    #[inline(always)]
    unsafe fn lane_total(v: float32x4_t) -> f32 {
        unsafe {
            let p = vpaddq_f32(v, v);
            vgetq_lane_f32(p, 0) + vgetq_lane_f32(p, 1)
        }
    }

    /// The decode kernel, four rows per pass so each group's activation
    /// loads serve four rows. A row's lanes are the four `sdot` lanes of its
    /// group sum, scaled `(sx * cbs) * norm`.
    #[target_feature(enable = "neon,dotprod")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode(
        bits: u8,
        codes: &[u8],
        norms: &[f16],
        groups: usize,
        xq: &[i8],
        sx: &[f32],
        tables: &[[i8; 16]; 2],
        cbs: f32,
        out: &mut [f32],
    ) {
        unsafe {
            let (t0, t1) = (vld1q_s8(tables[0].as_ptr()), vld1q_s8(tables[1].as_ptr()));
            let bpg = GROUP * bits as usize / 8;
            let n = out.len();
            let load = |g: usize| -> [int8x16_t; 8] {
                let x = xq.as_ptr().add(g * GROUP);
                std::array::from_fn(|k| vld1q_s8(x.add(16 * k)))
            };
            let sums = |r: usize, g: usize, xv: &[int8x16_t; 8]| {
                let w = expand(bits, codes.as_ptr().add((r * groups + g) * bpg), t0, t1);
                let (mut a, mut b) = (vdupq_n_s32(0), vdupq_n_s32(0));
                for k in 0..4 {
                    a = vdotq_s32(a, w[2 * k], xv[2 * k]);
                    b = vdotq_s32(b, w[2 * k + 1], xv[2 * k + 1]);
                }
                vcvtq_f32_s32(vaddq_s32(a, b))
            };
            let mut r = 0;
            while r + 4 <= n {
                let mut lanes = [vdupq_n_f32(0.0); 4];
                for g in 0..groups {
                    let xv = load(g);
                    let sxc = *sx.get_unchecked(g) * cbs;
                    for (k, l) in lanes.iter_mut().enumerate() {
                        let norm = norms.get_unchecked((r + k) * groups + g).to_f32();
                        *l = vfmaq_n_f32(*l, sums(r + k, g, &xv), sxc * norm);
                    }
                }
                for (k, l) in lanes.iter().enumerate() {
                    out[r + k] = lane_total(*l);
                }
                r += 4;
            }
            while r < n {
                let mut lanes = vdupq_n_f32(0.0);
                for g in 0..groups {
                    let xv = load(g);
                    let norm = norms.get_unchecked(r * groups + g).to_f32();
                    lanes = vfmaq_n_f32(lanes, sums(r, g, &xv), (*sx.get_unchecked(g) * cbs) * norm);
                }
                out[r] = lane_total(lanes);
                r += 1;
            }
        }
    }

    /// The prefill kernel: one weight row against `N` token rows (`stride`
    /// apart), expanding each weight group once for all of them. Each token
    /// sums its group exactly, then `acc = fma(sum, scale, acc)`.
    #[target_feature(enable = "neon,dotprod")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prefill_n<const N: usize>(
        bits: u8,
        codes: &[u8],
        tables: &[[i8; 16]; 2],
        norms: &[f16],
        cbs: f32,
        xq: &[i8],
        sx: &[f32],
        stride: usize,
        block: bool,
    ) -> [f32; N] {
        unsafe {
            let (t0, t1) = (vld1q_s8(tables[0].as_ptr()), vld1q_s8(tables[1].as_ptr()));
            let bpg = GROUP * bits as usize / 8;
            let groups = norms.len();
            let mut accf = [0f32; N];
            for g in 0..groups {
                let w = expand(bits, codes.as_ptr().add(g * bpg), t0, t1);
                let norm = norms.get_unchecked(g).to_f32();
                for (k, af) in accf.iter_mut().enumerate() {
                    let x = xq.as_ptr().add(k * stride + g * GROUP);
                    let mut acc = vdupq_n_s32(0);
                    for (j, wj) in w.iter().enumerate() {
                        acc = vdotq_s32(acc, *wj, vld1q_s8(x.add(j * 16)));
                    }
                    let s = *sx.get_unchecked(k * groups + g);
                    let sc = if block { (norm * cbs) * s } else { norm * (cbs * s) };
                    *af = (vaddvq_s32(acc) as f32).mul_add(sc, *af);
                }
            }
            accf
        }
    }
}

/// The NEON kernels on WebAssembly SIMD: `i8x16.swizzle` is the same
/// 16-entry table lookup, and widening multiplies with pairwise adds give
/// `sdot`'s four-byte lane sums exactly. The float combine is the scalar
/// `fma` in the same order, so results match the NEON kernels bit for bit.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;

    use half::f16;

    use super::GROUP;

    #[inline(always)]
    unsafe fn load(p: *const u8) -> v128 {
        // SAFETY: the caller keeps 16 bytes readable at `p`; wasm loads need
        // no alignment.
        unsafe { v128_load(p.cast()) }
    }

    /// As `neon::expand`: a group's 128 weights as eight 16-lane vectors in
    /// the activation layout's order.
    #[inline(always)]
    unsafe fn expand(bits: u8, c: *const u8, t0: v128, t1: v128) -> [v128; 8] {
        unsafe {
            let mask = u8x16_splat(15);
            let mut w = [i8x16_splat(0); 8];
            if bits == 2 {
                for m in 0..2 {
                    let b = load(c.add(16 * m));
                    let (lo, hi) = (v128_and(b, mask), u8x16_shr(b, 4));
                    w[m] = i8x16_swizzle(t0, lo);
                    w[2 + m] = i8x16_swizzle(t1, lo);
                    w[4 + m] = i8x16_swizzle(t0, hi);
                    w[6 + m] = i8x16_swizzle(t1, hi);
                }
            } else {
                for m in 0..4 {
                    let b = load(c.add(16 * m));
                    w[m] = i8x16_swizzle(t0, v128_and(b, mask));
                    w[4 + m] = i8x16_swizzle(t0, u8x16_shr(b, 4));
                }
            }
            w
        }
    }

    /// Byte-pair sums of `w * x`: bytes `0-1, 2-3, 4-5, 6-7` and `8-9, ...,
    /// 14-15`, added into `lo` and `hi`.
    #[inline(always)]
    fn dot_pairs(w: v128, x: v128, lo: &mut v128, hi: &mut v128) {
        *lo = i32x4_add(*lo, i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(w, x)));
        *hi = i32x4_add(*hi, i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(w, x)));
    }

    /// Pair sums to `sdot` lanes (bytes `4i..4i+4`).
    #[inline(always)]
    fn lanes(lo: v128, hi: v128) -> v128 {
        i32x4_add(i32x4_shuffle::<0, 2, 4, 6>(lo, hi), i32x4_shuffle::<1, 3, 5, 7>(lo, hi))
    }

    /// As `neon::decode`, a row at a time.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode(
        bits: u8,
        codes: &[u8],
        norms: &[f16],
        groups: usize,
        xq: &[i8],
        sx: &[f32],
        tables: &[[i8; 16]; 2],
        cbs: f32,
        out: &mut [f32],
    ) {
        unsafe {
            let (t0, t1) = (load(tables[0].as_ptr().cast()), load(tables[1].as_ptr().cast()));
            let bpg = GROUP * bits as usize / 8;
            for (r, o) in out.iter_mut().enumerate() {
                let mut acc = [0f32; 4];
                for g in 0..groups {
                    let w = expand(bits, codes.as_ptr().add((r * groups + g) * bpg), t0, t1);
                    let x = xq.as_ptr().add(g * GROUP).cast::<u8>();
                    let (mut lo, mut hi) = (i32x4_splat(0), i32x4_splat(0));
                    for (k, wk) in w.iter().enumerate() {
                        dot_pairs(*wk, load(x.add(16 * k)), &mut lo, &mut hi);
                    }
                    let s = lanes(lo, hi);
                    let sc = (*sx.get_unchecked(g) * cbs) * norms.get_unchecked(r * groups + g).to_f32();
                    let sv =
                        [i32x4_extract_lane::<0>(s), i32x4_extract_lane::<1>(s), i32x4_extract_lane::<2>(s), i32x4_extract_lane::<3>(s)];
                    for (a, v) in acc.iter_mut().zip(sv) {
                        *a = (v as f32).mul_add(sc, *a);
                    }
                }
                *o = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
        }
    }

    /// As `neon::prefill_n`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prefill_n<const N: usize>(
        bits: u8,
        codes: &[u8],
        tables: &[[i8; 16]; 2],
        norms: &[f16],
        cbs: f32,
        xq: &[i8],
        sx: &[f32],
        stride: usize,
        block: bool,
    ) -> [f32; N] {
        unsafe {
            let (t0, t1) = (load(tables[0].as_ptr().cast()), load(tables[1].as_ptr().cast()));
            let bpg = GROUP * bits as usize / 8;
            let groups = norms.len();
            let mut accf = [0f32; N];
            for g in 0..groups {
                let w = expand(bits, codes.as_ptr().add(g * bpg), t0, t1);
                let norm = norms.get_unchecked(g).to_f32();
                for (k, af) in accf.iter_mut().enumerate() {
                    let x = xq.as_ptr().add(k * stride + g * GROUP).cast::<u8>();
                    let (mut lo, mut hi) = (i32x4_splat(0), i32x4_splat(0));
                    for (j, wj) in w.iter().enumerate() {
                        dot_pairs(*wj, load(x.add(16 * j)), &mut lo, &mut hi);
                    }
                    let t = i32x4_add(lo, hi);
                    let total =
                        i32x4_extract_lane::<0>(t) + i32x4_extract_lane::<1>(t) + i32x4_extract_lane::<2>(t) + i32x4_extract_lane::<3>(t);
                    let s = *sx.get_unchecked(k * groups + g);
                    let sc = if block { (norm * cbs) * s } else { norm * (cbs * s) };
                    *af = (total as f32).mul_add(sc, *af);
                }
            }
            accf
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use needle_core::quant::cq_unpack;

    #[test]
    fn kernels_match_reference() {
        let (out, din) = (37, 384);
        let w: Vec<f32> = (0..out * din).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.05).collect();
        let q = QMat::from_f32(&w, out, din);
        let p = cq_pack(&w, din, 4, GROUP);
        let deq = cq_unpack(&p.packed, &p.norms, out, din, 4, GROUP);
        let x: Vec<f32> = (0..din).map(|i| (i * 31 % 97) as f32 / 48.0 - 1.0).collect();
        let a = QAct::new(&x, 1, din, q.act_bits());
        let y = q.matmul(&a);
        for o in 0..out {
            let want: f32 = (0..din).map(|i| x[i] * deq[o * din + i]).sum();
            assert!((y[o] - want).abs() < 0.02 * (1.0 + want.abs()), "{} vs {want}", y[o]);
            assert_eq!(y[o].to_bits(), q.decode_row_ref(o, &a).to_bits(), "decode row {o}");
        }
        for rows in [2, 3, 5, 9, 13, 17] {
            let xs: Vec<f32> = (0..rows * din).map(|i| (i as f32 * 0.37).sin()).collect();
            let a = QAct::new(&xs, rows, din, q.act_bits());
            let y = q.matmul(&a);
            for t in 0..rows {
                for o in 0..out {
                    let want = q.prefill_row_ref(o, &a, t, t < rows / 4 * 4);
                    assert_eq!(y[t * out + o].to_bits(), want.to_bits(), "rows {rows} t {t} o {o}");
                }
            }
        }
        let row = q.row(5);
        for i in 0..din {
            assert!((row[i] - deq[5 * din + i]).abs() < 1e-5);
        }
    }

    #[test]
    fn two_bit_kernels_match_reference() {
        // A 2-bit matrix built by hand: codes cycle through the four values.
        let (out, din) = (9usize, 256usize);
        let groups = din / GROUP;
        let codes: Vec<u8> = (0..out * din / 4).map(|i| (i * 37 % 251) as u8).collect();
        let norms: Vec<f16> = (0..out * groups).map(|i| f16::from_f32(0.5 + i as f32 * 0.01)).collect();
        let q = QMat::build(out, din, 2, &codebook(2, GROUP), |c| c, codes, norms);
        let xs: Vec<f32> = (0..5 * din).map(|i| (i as f32 * 0.21).cos()).collect();
        let a = QAct::new(&xs[..din], 1, din, 2);
        let y = q.matmul(&a);
        for (o, v) in y.iter().enumerate() {
            assert_eq!(v.to_bits(), q.decode_row_ref(o, &a).to_bits(), "decode row {o}");
        }
        let a = QAct::new(&xs, 5, din, 2);
        let y = q.matmul(&a);
        for t in 0..5 {
            for o in 0..out {
                assert_eq!(y[t * out + o].to_bits(), q.prefill_row_ref(o, &a, t, t < 4).to_bits(), "t {t} o {o}");
            }
        }
    }
}

#[cfg(test)]
mod bench_scaling {
    /// `NEEDLE_THREADS=6 cargo test --release -p needle-engine bench_scaling -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_scaling() {
        use crate::team::{SyncPtr, share, team};
        let path = &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/needle3.cact");
        if !path.exists() {
            return;
        }
        let l = crate::loader::load(path).unwrap();
        let q = l.model.q.as_ref().unwrap();
        let x: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
        let a = super::QAct::new(&x, 1, 768, q.qkvg[0].act_bits());
        let tm = team();
        for nt in 1..=tm.threads() {
            let mut y = vec![0f32; q.qkvg[0].out];
            let yp = SyncPtr(y.as_mut_ptr());
            let reps = 400;
            let t = web_time::Instant::now();
            for rep in 0..reps {
                // Walk every layer so the weights stream as in a real step.
                let lin = &q.qkvg[rep % 20];
                tm.run_n(nt, &|tid, n| lin.rows_into(share(lin.out, tid, n), &a, yp, lin.out));
            }
            let per = t.elapsed().as_secs_f64() / reps as f64;
            let bytes = q.qkvg[0].bytes() as f64;
            eprintln!("qkvg on {nt}: {:.1} us  {:.1} GB/s", per * 1e6, bytes / per / 1e9);
        }
        // The same with the weights hot in cache: one layer over and over.
        for nt in [1, tm.threads()] {
            let mut y = vec![0f32; q.qkvg[0].out];
            let yp = SyncPtr(y.as_mut_ptr());
            let lin = &q.qkvg[0];
            let reps = 400;
            let t = web_time::Instant::now();
            for _ in 0..reps {
                tm.run_n(nt, &|tid, n| lin.rows_into(share(lin.out, tid, n), &a, yp, lin.out));
            }
            let per = t.elapsed().as_secs_f64() / reps as f64;
            eprintln!("qkvg on {nt}, one layer hot: {:.1} us", per * 1e6);
        }
    }
}
