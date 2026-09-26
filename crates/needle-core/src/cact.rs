//! The `.cact` deployment archive (format tag `0x05E12A84`).
//!
//! A 196-byte header carrying the architecture geometry, the shared CQ
//! codebooks, a nameless 44-byte-per-record tensor directory, then 64-byte
//! aligned blobs in the fixed canonical order the runtime indexes by
//! position. See `export.py` in the reference for the full layout.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use half::f16;

use crate::config::{Config, ENGRAM_CONV_TAPS, HEAD_KEYS};
use crate::quant::{self, CB_BITS, HEAD_BITS};
use crate::rng::hada_perms;
use crate::tensor::{Params, Tensor, get};

pub const TAG: u32 = 0x05E1_2A84;
pub const TAG_V2: u32 = 0x05E1_2A83;
pub const ALIGN: usize = 64;
pub const HEADER_BYTES: usize = 49 * 4;
pub const REC_SIZE: usize = 44;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Dtype {
    F16 = 1,
    F32 = 2,
    Cq = 3,
    Raw = 4,
}

impl Dtype {
    fn from_u8(v: u8) -> Result<Self> {
        Ok(match v {
            1 => Dtype::F16,
            2 => Dtype::F32,
            3 => Dtype::Cq,
            4 => Dtype::Raw,
            _ => bail!("unknown .cact tensor dtype {v}"),
        })
    }
}

/// The geometry the header carries.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub tag: u32,
    pub num_tensors: u32,
    pub codebook_len: u32,
    pub kv_window: u32,
    pub kv_bits: u32,
    pub vocab: u32,
    pub out_vocab: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub max_seq_len: u32,
    pub hada_n: u32,
    pub mhc_lanes: u32,
    pub sliding_window: u32,
    pub global_mask: u64,
    pub qkv_conv_taps: u32,
    pub engram_slots: u32,
    pub engram_sub_dim: u32,
    pub num_engram_tables: u32,
    pub engram_conv_taps: u32,
    pub engram_conv_dilation: u32,
    pub engram_seed_heads: u32,
    pub engram_orders: Vec<u32>,
    pub engram_sites: Vec<u32>,
    pub rope_theta: f32,
}

impl Header {
    fn to_bytes(&self) -> Vec<u8> {
        let mut orders4 = self.engram_orders.clone();
        orders4.resize(4, 0);
        let mut sites16 = self.engram_sites.clone();
        sites16.resize(16, 0);
        let mut f: Vec<u32> = vec![
            self.tag,
            self.num_tensors,
            self.codebook_len,
            self.kv_window,
            self.kv_bits,
            self.vocab,
            self.out_vocab,
            self.d_model,
            self.num_heads,
            self.num_kv_heads,
            self.num_layers,
            self.qk_head_dim,
            self.v_head_dim,
            self.max_seq_len,
            self.hada_n,
            self.mhc_lanes,
            self.sliding_window,
            (self.global_mask & 0xFFFF_FFFF) as u32,
            (self.global_mask >> 32) as u32,
            self.qkv_conv_taps,
            self.engram_slots,
            self.engram_sub_dim,
            self.num_engram_tables,
            self.engram_conv_taps,
            self.engram_conv_dilation,
            self.engram_seed_heads,
            self.engram_orders.len() as u32,
        ];
        f.extend(&orders4);
        f.push(self.engram_sites.len() as u32);
        f.extend(&sites16);
        let mut out: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.extend(self.rope_theta.to_le_bytes());
        debug_assert_eq!(out.len(), HEADER_BYTES);
        out
    }

    fn parse(b: &[u8]) -> Result<Self> {
        ensure!(b.len() >= HEADER_BYTES, "not a complete .cact archive");
        let u = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
        let tag = u(0);
        if tag != TAG {
            if tag == TAG_V2 {
                bail!("this is a Needle 2 archive; Needle 3 (tag 0x{TAG:08x}) is required");
            }
            bail!("unknown .cact format tag 0x{tag:08x}");
        }
        let num_orders = u(26) as usize;
        let num_sites = u(31) as usize;
        ensure!(num_orders <= 4 && num_sites <= 16, "corrupt .cact header");
        Ok(Header {
            tag,
            num_tensors: u(1),
            codebook_len: u(2),
            kv_window: u(3),
            kv_bits: u(4),
            vocab: u(5),
            out_vocab: u(6),
            d_model: u(7),
            num_heads: u(8),
            num_kv_heads: u(9),
            num_layers: u(10),
            qk_head_dim: u(11),
            v_head_dim: u(12),
            max_seq_len: u(13),
            hada_n: u(14),
            mhc_lanes: u(15),
            sliding_window: u(16),
            global_mask: u(17) as u64 | ((u(18) as u64) << 32),
            qkv_conv_taps: u(19),
            engram_slots: u(20),
            engram_sub_dim: u(21),
            num_engram_tables: u(22),
            engram_conv_taps: u(23),
            engram_conv_dilation: u(24),
            engram_seed_heads: u(25),
            engram_orders: (0..num_orders).map(|i| u(27 + i)).collect(),
            engram_sites: (0..num_sites).map(|i| u(32 + i)).collect(),
            rope_theta: f32::from_le_bytes(b[48 * 4..49 * 4].try_into().unwrap()),
        })
    }

    pub fn global_layers(&self) -> Vec<usize> {
        (0..self.num_layers as usize).filter(|i| self.global_mask >> i & 1 == 1).collect()
    }

    /// The model config this header describes (fields the archive does not
    /// carry keep the Needle 3 defaults).
    pub fn to_config(&self) -> Config {
        let tables = self.num_engram_tables as usize;
        let orders: Vec<usize> = self.engram_orders.iter().map(|&o| o as usize).collect();
        Config {
            vocab_size: self.vocab as usize,
            out_vocab: self.out_vocab as usize,
            d_model: self.d_model as usize,
            num_heads: self.num_heads as usize,
            num_kv_heads: self.num_kv_heads as usize,
            num_layers: self.num_layers as usize,
            qk_head_dim: self.qk_head_dim as usize,
            v_head_dim: self.v_head_dim as usize,
            max_seq_len: self.max_seq_len as usize,
            mhc_lanes: self.mhc_lanes as usize,
            sliding_window: self.sliding_window as usize,
            global_layers: self.global_layers(),
            qkv_conv_taps: self.qkv_conv_taps as usize,
            engram_slots: self.engram_slots as usize,
            engram_heads: if orders.is_empty() { 0 } else { tables / orders.len() },
            engram_seed_heads: self.engram_seed_heads as usize,
            engram_orders: orders,
            engram_layers: self.engram_sites.iter().map(|&s| s as usize).collect(),
            rope_theta: self.rope_theta as f64,
            kv_window: self.kv_window as usize,
            kv_bits: self.kv_bits as usize,
            dtype: "float32".into(),
            ..Config::default()
        }
    }
}

/// One directory record.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub nbytes: u64,
    pub group: u32,
    pub bits: u32,
}

impl Record {
    fn to_bytes(&self) -> [u8; REC_SIZE] {
        let mut b = [0u8; REC_SIZE];
        b[0] = self.dtype as u8;
        b[1] = self.shape.len() as u8;
        for (i, &d) in self.shape.iter().take(4).enumerate() {
            b[4 + i * 4..8 + i * 4].copy_from_slice(&(d as u32).to_le_bytes());
        }
        b[20..28].copy_from_slice(&self.offset.to_le_bytes());
        b[28..36].copy_from_slice(&self.nbytes.to_le_bytes());
        b[36..40].copy_from_slice(&self.group.to_le_bytes());
        b[40..44].copy_from_slice(&self.bits.to_le_bytes());
        b
    }

    fn parse(b: &[u8]) -> Result<Self> {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let ndim = b[1] as usize;
        ensure!(ndim <= 4, "corrupt .cact record");
        Ok(Record {
            dtype: Dtype::from_u8(b[0])?,
            shape: (0..ndim).map(|i| u32_at(4 + i * 4) as usize).collect(),
            offset: u64_at(20),
            nbytes: u64_at(28),
            group: u32_at(36),
            bits: u32_at(40),
        })
    }
}

/// A tensor about to be written.
struct Out {
    dtype: Dtype,
    shape: Vec<usize>,
    blob: Vec<u8>,
    group: u32,
    bits: u32,
}

fn fp16(t: &[f32], shape: Vec<usize>) -> Out {
    let blob = t.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
    Out { dtype: Dtype::F16, shape, blob, group: 0, bits: 0 }
}

fn fp16_t(t: &Tensor) -> Out {
    fp16(&t.data, t.shape.clone())
}

/// CQ-pack a row-major `[out, in]` matrix.
fn cq(mat: &[f32], out: usize, d: usize, bits: u32, group: usize) -> Out {
    let p = quant::cq_pack(mat, d, bits, group);
    let mut blob = p.packed;
    blob.extend(p.norms.iter().flat_map(|n| n.to_le_bytes()));
    Out { dtype: Dtype::Cq, shape: vec![out, d], blob, group: group as u32, bits }
}

/// A stacked Flax kernel `[L, in, out]`: layer `i` transposed to `[out, in]`.
fn kernel_t(t: &Tensor, i: usize) -> (Vec<f32>, usize, usize) {
    let layer = t.index0(i);
    let (din, dout) = (layer.shape[0], layer.shape[1]);
    (layer.t_last2().data, dout, din)
}

fn tensors(params: &Params, config: &Config, bits: u32, group: usize) -> Result<Vec<Out>> {
    let (_, _, sub_dim) = config.engram_geometry();
    let num_tables = config.num_engram_tables();
    let emb = get(params, "embedding/embedding")?;
    let mut ts = vec![cq(&emb.data, emb.shape[0], emb.shape[1], bits, group)];

    let b = |k: &str| get(params, &format!("stack/layers/block/{k}"));
    let taps = config.qkv_conv_taps > 0;
    for i in 0..config.num_layers {
        let proj = |k: &str| -> Result<Out> {
            let (m, o, d) = kernel_t(b(&format!("self_attn/{k}/kernel"))?, i);
            Ok(cq(&m, o, d, bits, group))
        };
        let row = |k: &str| -> Result<Out> { Ok(fp16_t(&b(k)?.index0(i))) };
        ts.push(row("ZCRMSNorm_0/scale")?);
        ts.push(proj("q_proj")?);
        ts.push(proj("k_proj")?);
        ts.push(proj("v_proj")?);
        if taps {
            ts.push(row("self_attn/q_taps")?);
            ts.push(row("self_attn/k_taps")?);
            ts.push(row("self_attn/v_taps")?);
        }
        ts.push(row("self_attn/q_norm/scale")?);
        ts.push(row("self_attn/k_norm/scale")?);
        ts.push(proj("gate_proj")?);
        ts.push(proj("out_proj")?);
        ts.push(row("post_attn_norm/scale")?);
        ts.push(fp16(b("attn_gate")?.slice0(i), vec![1]));
        ts.push(row("pre_hada_norm/scale")?);
        for k in ["d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u"] {
            ts.push(row(&format!("hadamard_mlp/{k}"))?);
        }
    }

    for name in ["mhc_a_pre", "mhc_a_post", "mhc_a_res", "mhc_b_pre", "mhc_b_post", "mhc_b_res"] {
        ts.push(fp16_t(get(params, &format!("stack/{name}"))?));
    }
    for name in ["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"] {
        let phi = get(params, &format!("stack/{name}"))?;
        let (l, nc, lanes) = (phi.shape[0], phi.shape[1], phi.shape[2]);
        // (L, nC, lanes) -> (L, lanes, nC) -> (L*lanes, nC)
        let mut m = Vec::with_capacity(phi.numel());
        for li in 0..l {
            for j in 0..lanes {
                for c in 0..nc {
                    m.push(phi.data[(li * nc + c) * lanes + j]);
                }
            }
        }
        ts.push(cq(&m, l * lanes, nc, bits, group));
    }

    let n = config.hada_n();
    for p in hada_perms(n, config.hada_split()) {
        let blob = p.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect();
        ts.push(Out { dtype: Dtype::F32, shape: vec![n], blob, group: 0, bits: 0 });
    }

    for s in 0..config.engram_layers.len() {
        let e = |k: &str| get(params, &format!("engrams_{s}/{k}"));
        let tables = e("embedding")?;
        ts.push(cq(&tables.data, num_tables * config.engram_slots, sub_dim, bits, group));
        for k in ["key_proj", "value_proj"] {
            let w = e(&format!("{k}/kernel"))?;
            ts.push(cq(&w.t_last2().data, w.shape[1], w.shape[0], bits, group));
        }
        ts.push(fp16_t(e("taps")?));
    }

    ts.push(fp16_t(get(params, "stack/final_norm/scale")?));
    ts.extend(head_tensors(params, group)?);
    Ok(ts)
}

fn head_code(key: &str) -> f32 {
    match key {
        "embedding_head" => 1.0,
        "confidence_head" => 2.0,
        "router_head" => 3.0,
        _ => unreachable!(),
    }
}

fn head_tensors(params: &Params, group: usize) -> Result<Vec<Out>> {
    let present: Vec<&str> = HEAD_KEYS.iter().copied().filter(|h| params.keys().any(|k| k.starts_with(&format!("{h}/")))).collect();
    if present.is_empty() {
        return Ok(vec![]);
    }
    let codes: Vec<f32> = present.iter().map(|h| head_code(h)).collect();
    let mut ts = vec![fp16(&codes, vec![codes.len()])];
    for h in present {
        let g = |k: &str| get(params, &format!("{h}/{k}"));
        let matrix = |t: &Tensor| {
            let d = *t.shape.last().unwrap();
            cq(&t.data, t.numel() / d, d, HEAD_BITS, group)
        };
        let kernel = g("proj/kernel")?.t_last2();
        ts.push(matrix(g("probes")?));
        ts.push(fp16_t(g("gain")?));
        ts.push(matrix(g("query")?));
        ts.push(fp16_t(g("row_bias")?));
        ts.push(matrix(&kernel));
        match params.get(&format!("{h}/proj/bias")) {
            Some(bias) => ts.push(fp16_t(bias)),
            None => ts.push(fp16(&vec![0.0; kernel.shape[0]], vec![kernel.shape[0]])),
        }
        if h == "router_head" {
            let cal = params.get("router_head/calibration").map(|t| t.data.clone()).unwrap_or(vec![0.90, 0.00, 0.60]);
            ts.push(fp16(&cal, vec![3]));
        }
    }
    Ok(ts)
}

fn align(n: usize) -> usize {
    (n + ALIGN - 1) & !(ALIGN - 1)
}

/// `_pack_cact`: the archive bytes for a checkpoint.
pub fn pack(params: &Params, config: &Config, bits: u32, group: usize, tokenizer_blob: Option<&[u8]>, kv_window: usize) -> Result<Vec<u8>> {
    let mut ts = tensors(params, config, bits, group)?;
    let vocab = config.vocab_size;
    if config.out_vocab > vocab {
        bail!("out_vocab {} exceeds the exported vocab {vocab}", config.out_vocab);
    }
    if let Some(blob) = tokenizer_blob {
        let pieces = u32::from_le_bytes(blob[..4].try_into()?) as usize;
        if pieces > vocab {
            bail!("tokenizer vocab {pieces} > exported vocab {vocab}");
        }
        ts.push(Out { dtype: Dtype::Raw, shape: vec![], blob: blob.to_vec(), group: 0, bits: 0 });
    }
    let cb: Vec<f32> = CB_BITS.iter().flat_map(|&b| quant::codebook(b, group)).collect();
    let (qk, v) = config.head_dims();
    let (orders, heads, sub_dim) = config.engram_geometry();
    if config.num_layers > 64 {
        bail!("cact global_mask holds 64 layers; deeper stacks need another bump");
    }
    if config.engram_layers.len() > 16 {
        bail!("cact header holds 16 engram sites");
    }
    let header = Header {
        tag: TAG,
        num_tensors: ts.len() as u32,
        codebook_len: cb.len() as u32,
        kv_window: kv_window as u32,
        kv_bits: if config.kv_bits > 0 { config.kv_bits as u32 } else { 8 },
        vocab: vocab as u32,
        out_vocab: config.out_vocab as u32,
        d_model: config.d_model as u32,
        num_heads: config.num_heads as u32,
        num_kv_heads: config.num_kv_heads as u32,
        num_layers: config.num_layers as u32,
        qk_head_dim: qk as u32,
        v_head_dim: v as u32,
        max_seq_len: config.max_seq_len as u32,
        hada_n: config.hada_n() as u32,
        mhc_lanes: config.mhc_lanes as u32,
        sliding_window: config.sliding_window as u32,
        global_mask: config.global_layers.iter().fold(0u64, |m, &g| m | (1 << g)),
        qkv_conv_taps: config.qkv_conv_taps as u32,
        engram_slots: config.engram_slots as u32,
        engram_sub_dim: sub_dim as u32,
        num_engram_tables: (orders.len() * heads) as u32,
        engram_conv_taps: ENGRAM_CONV_TAPS as u32,
        engram_conv_dilation: orders.iter().copied().max().unwrap_or(0) as u32,
        engram_seed_heads: config.engram_seed_heads as u32,
        engram_orders: orders.iter().map(|&o| o as u32).collect(),
        engram_sites: config.engram_layers.iter().map(|&s| s as u32).collect(),
        rope_theta: config.rope_theta as f32,
    };
    let mut buf = header.to_bytes();
    buf.extend(cb.iter().flat_map(|v| v.to_le_bytes()));
    let mut pos = buf.len() + ts.len() * REC_SIZE;
    let mut records = Vec::with_capacity(ts.len());
    for t in &ts {
        pos = align(pos);
        records.push(Record {
            dtype: t.dtype,
            shape: t.shape.clone(),
            offset: pos as u64,
            nbytes: t.blob.len() as u64,
            group: t.group,
            bits: t.bits,
        });
        pos += t.blob.len();
    }
    for r in &records {
        buf.extend(r.to_bytes());
    }
    for (t, r) in ts.iter().zip(&records) {
        buf.resize(r.offset as usize, 0);
        buf.extend(&t.blob);
    }
    Ok(buf)
}

pub struct ExportInfo {
    pub bytes: usize,
    pub tensors: usize,
}

pub fn write_export(
    path: &Path,
    params: &Params,
    config: &Config,
    bits: u32,
    group: usize,
    tokenizer_blob: Option<&[u8]>,
    kv_window: usize,
) -> Result<ExportInfo> {
    let buf = pack(params, config, bits, group, tokenizer_blob, kv_window)?;
    std::fs::write(path, &buf).with_context(|| format!("write {}", path.display()))?;
    let tensors = u32::from_le_bytes(buf[4..8].try_into()?) as usize;
    Ok(ExportInfo { bytes: buf.len(), tensors })
}

/// A CQ matrix as stored: `[out, in]` logical, rows padded to the group.
pub struct CqView<'a> {
    pub out: usize,
    pub in_dim: usize,
    pub group: usize,
    pub bits: u32,
    pub packed: &'a [u8],
    pub norms: &'a [u8],
}

impl CqView<'_> {
    pub fn in_pad(&self) -> usize {
        self.in_dim.div_ceil(self.group) * self.group
    }

    pub fn row_bytes(&self) -> usize {
        quant::packed_row_bytes(self.in_pad(), self.bits)
    }

    pub fn norm(&self, row: usize, g: usize) -> f32 {
        let i = (row * (self.in_pad() / self.group) + g) * 2;
        f16::from_le_bytes([self.norms[i], self.norms[i + 1]]).to_f32()
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let norms: Vec<f16> = self.norms.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c)).collect();
        quant::cq_unpack(self.packed, &norms, self.out, self.in_dim, self.bits, self.group)
    }
}

/// A read `.cact` archive, backed by an owned or mapped byte buffer.
pub struct Archive {
    pub header: Header,
    pub codebook: Vec<f32>,
    pub records: Vec<Record>,
    data: ArchiveBytes,
}

enum ArchiveBytes {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
    /// Someone else's buffer; see [`Archive::from_raw`].
    Borrowed(*const u8, usize),
}

// SAFETY: every variant is read-only; a borrowed buffer's owner promises
// it outlives the archive (`Archive::from_raw`).
unsafe impl Send for ArchiveBytes {}
unsafe impl Sync for ArchiveBytes {}

impl std::ops::Deref for ArchiveBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            ArchiveBytes::Owned(v) => v,
            ArchiveBytes::Mapped(m) => m,
            // SAFETY: `from_raw`'s caller keeps these bytes alive and unchanged.
            ArchiveBytes::Borrowed(p, n) => unsafe { std::slice::from_raw_parts(*p, *n) },
        }
    }
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only mapping of a file we do not modify.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::parse(ArchiveBytes::Mapped(mmap)).with_context(|| format!("read {}", path.display()))
    }

    /// An archive over a caller's buffer, without copying it.
    ///
    /// # Safety
    /// `ptr` must point at `len` readable bytes that stay alive and unchanged
    /// for as long as the returned archive does.
    pub unsafe fn from_raw(ptr: *const u8, len: usize) -> Result<Self> {
        Self::parse(ArchiveBytes::Borrowed(ptr, len))
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(ArchiveBytes::Owned(bytes))
    }

    fn parse(data: ArchiveBytes) -> Result<Self> {
        let header = Header::parse(&data)?;
        let mut off = HEADER_BYTES;
        let cb_n = header.codebook_len as usize;
        let codebook = data
            .get(off..off + cb_n * 4)
            .context("truncated codebook")?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        off += cb_n * 4;
        let mut records = Vec::with_capacity(header.num_tensors as usize);
        for _ in 0..header.num_tensors {
            let r = Record::parse(data.get(off..off + REC_SIZE).context("truncated directory")?)?;
            ensure!((r.offset + r.nbytes) as usize <= data.len(), "tensor past the end of the archive");
            records.push(r);
            off += REC_SIZE;
        }
        Ok(Self { header, codebook, records, data })
    }

    pub fn bytes(&self, i: usize) -> &[u8] {
        let r = &self.records[i];
        &self.data[r.offset as usize..(r.offset + r.nbytes) as usize]
    }

    pub fn cq(&self, i: usize) -> Result<CqView<'_>> {
        let r = &self.records[i];
        ensure!(r.dtype == Dtype::Cq, "tensor {i} is {:?}, not CQ", r.dtype);
        let (out, in_dim) = (r.shape[0], r.shape[1]);
        let group = r.group as usize;
        let in_pad = in_dim.div_ceil(group) * group;
        let n_packed = out * quant::packed_row_bytes(in_pad, r.bits);
        let blob = self.bytes(i);
        Ok(CqView { out, in_dim, group, bits: r.bits, packed: &blob[..n_packed], norms: &blob[n_packed..] })
    }

    /// Tensor `i` as f32 (CQ matrices dequantized).
    pub fn tensor(&self, i: usize) -> Result<Tensor> {
        let r = &self.records[i];
        let blob = self.bytes(i);
        let shape = r.shape.clone();
        Ok(match r.dtype {
            Dtype::F16 => Tensor::new(shape, blob.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c).to_f32()).collect()),
            Dtype::F32 => Tensor::new(shape, blob.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
            Dtype::Cq => Tensor::new(shape, self.cq(i)?.dequantize()),
            Dtype::Raw => bail!("tensor {i} is a raw attachment"),
        })
    }

    /// The tokenizer attachment.
    pub fn tokenizer_blob(&self) -> Result<&[u8]> {
        let i = self.records.iter().position(|r| r.dtype == Dtype::Raw).context("archive carries no tokenizer")?;
        Ok(self.bytes(i))
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// `read_layers`: the depth of an archive without reading its weights.
pub fn read_layers(path: &Path) -> Result<usize> {
    let mut f = std::fs::File::open(path)?;
    let mut b = vec![0u8; HEADER_BYTES];
    std::io::Read::read_exact(&mut f, &mut b)?;
    Ok(Header::parse(&b)?.num_layers as usize)
}
