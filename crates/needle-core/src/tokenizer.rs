//! SentencePiece BPE, bit-compatible with `sentencepiece.SentencePieceProcessor`
//! for the Needle tokenizer (identity normalizer, byte fallback).
//!
//! Loads either the `tokenizer.model` protobuf or the tokenizer blob a
//! `.cact` archive carries, and writes that blob for export.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::render::CHAT_MARKERS;

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
pub const BOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const IM_START_ID: u32 = 4;
pub const IM_END_ID: u32 = 5;
pub const THINK_START_ID: u32 = 6;
pub const THINK_END_ID: u32 = 7;
pub const TOOLS_START_ID: u32 = 8;
pub const TOOLS_END_ID: u32 = 9;
pub const TOOL_CALL_START_ID: u32 = 10;
pub const TOOL_CALL_END_ID: u32 = 11;

const META_SPACE: char = '\u{2581}';
const UNK_SURFACE: &str = " \u{2047} ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceKind {
    /// Code in the `.cact` tokenizer blob.
    fn blob_code(self) -> u8 {
        match self {
            PieceKind::Normal | PieceKind::Unused => 0,
            PieceKind::Unknown => 1,
            PieceKind::Control => 2,
            PieceKind::UserDefined => 3,
            PieceKind::Byte => 4,
        }
    }

    fn from_blob(code: u8) -> Result<Self> {
        Ok(match code {
            0 => PieceKind::Normal,
            1 => PieceKind::Unknown,
            2 => PieceKind::Control,
            3 => PieceKind::UserDefined,
            4 => PieceKind::Byte,
            _ => bail!("unknown tokenizer piece type {code}"),
        })
    }

    /// `SentencePiece.Type` in the model protobuf.
    fn from_proto(code: u64) -> Result<Self> {
        Ok(match code {
            1 => PieceKind::Normal,
            2 => PieceKind::Unknown,
            3 => PieceKind::Control,
            4 => PieceKind::UserDefined,
            5 => PieceKind::Unused,
            6 => PieceKind::Byte,
            _ => bail!("unknown sentencepiece type {code}"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    kinds: Vec<PieceKind>,
    /// Pieces BPE may merge into or emit (NORMAL, USER_DEFINED, UNUSED).
    vocab: HashMap<String, u32>,
    /// Every piece, reserved ones included (`PieceToId`).
    all: HashMap<String, u32>,
    /// User-defined symbols, longest first, matched greedily and never merged.
    user_defined: Vec<String>,
    ud_first: [bool; 256],
    byte_ids: [u32; 256],
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    byte_fallback: bool,
    unk_id: u32,
}

/// Minimal protobuf reader for the SentencePiece `ModelProto`.
struct Proto<'a> {
    buf: &'a [u8],
    pos: usize,
}

enum Field<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
}

impl<'a> Proto<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).context("truncated varint")?;
            self.pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("varint too long")
    }

    fn next(&mut self) -> Result<Option<(u32, Field<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let (num, wire) = ((key >> 3) as u32, key & 7);
        let field = match wire {
            0 => Field::Varint(self.varint()?),
            1 => {
                let b = self.buf.get(self.pos..self.pos + 8).context("truncated fixed64")?;
                self.pos += 8;
                {
                    let _ = b;
                    Field::Fixed64
                }
            }
            2 => {
                let n = self.varint()? as usize;
                let b = self.buf.get(self.pos..self.pos + n).context("truncated bytes")?;
                self.pos += n;
                Field::Bytes(b)
            }
            5 => {
                let b = self.buf.get(self.pos..self.pos + 4).context("truncated fixed32")?;
                self.pos += 4;
                Field::Fixed32(u32::from_le_bytes(b.try_into()?))
            }
            w => bail!("unsupported protobuf wire type {w}"),
        };
        Ok(Some((num, field)))
    }
}

impl Tokenizer {
    fn build(
        pieces: Vec<String>,
        scores: Vec<f32>,
        kinds: Vec<PieceKind>,
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        byte_fallback: bool,
    ) -> Result<Self> {
        let mut vocab = HashMap::new();
        let mut all = HashMap::new();
        let mut user_defined = Vec::new();
        let mut byte_ids = [u32::MAX; 256];
        let mut unk_id = UNK_ID;
        for (i, (p, k)) in pieces.iter().zip(&kinds).enumerate() {
            let id = i as u32;
            all.entry(p.clone()).or_insert(id);
            match k {
                PieceKind::Normal | PieceKind::UserDefined | PieceKind::Unused => {
                    vocab.entry(p.clone()).or_insert(id);
                }
                PieceKind::Byte => {
                    let hex = p.strip_prefix("<0x").and_then(|s| s.strip_suffix('>'));
                    let b = hex.and_then(|h| u8::from_str_radix(h, 16).ok()).with_context(|| format!("bad byte piece {p}"))?;
                    byte_ids[b as usize] = id;
                }
                PieceKind::Unknown => unk_id = id,
                PieceKind::Control => {}
            }
            if *k == PieceKind::UserDefined {
                user_defined.push(p.clone());
            }
        }
        user_defined.sort_by_key(|p| std::cmp::Reverse(p.len()));
        let mut ud_first = [false; 256];
        for u in &user_defined {
            if let Some(&b) = u.as_bytes().first() {
                ud_first[b as usize] = true;
            }
        }
        Ok(Self {
            pieces,
            scores,
            kinds,
            vocab,
            all,
            user_defined,
            ud_first,
            byte_ids,
            add_dummy_prefix,
            remove_extra_whitespaces,
            byte_fallback,
            unk_id,
        })
    }

    /// Parse a SentencePiece `tokenizer.model`.
    pub fn from_model_bytes(buf: &[u8]) -> Result<Self> {
        let (mut pieces, mut scores, mut kinds) = (Vec::new(), Vec::new(), Vec::new());
        let (mut add_dummy, mut remove_ws, mut byte_fallback) = (true, true, false);
        let mut normalizer = String::from("nmt_nfkc");
        let mut charsmap = false;
        let mut model_type = 1u64;
        let mut top = Proto::new(buf);
        while let Some((num, field)) = top.next()? {
            match (num, field) {
                (1, Field::Bytes(b)) => {
                    let (mut piece, mut score, mut kind) = (String::new(), 0f32, PieceKind::Normal);
                    let mut p = Proto::new(b);
                    while let Some((n, f)) = p.next()? {
                        match (n, f) {
                            (1, Field::Bytes(s)) => piece = String::from_utf8(s.to_vec())?,
                            (2, Field::Fixed32(v)) => score = f32::from_bits(v),
                            (3, Field::Varint(v)) => kind = PieceKind::from_proto(v)?,
                            _ => {}
                        }
                    }
                    pieces.push(piece);
                    scores.push(score);
                    kinds.push(kind);
                }
                (2, Field::Bytes(b)) => {
                    let mut p = Proto::new(b);
                    while let Some((n, f)) = p.next()? {
                        match (n, f) {
                            (3, Field::Varint(v)) => model_type = v,
                            (35, Field::Varint(v)) => byte_fallback = v != 0,
                            _ => {}
                        }
                    }
                }
                (3, Field::Bytes(b)) => {
                    let mut p = Proto::new(b);
                    while let Some((n, f)) = p.next()? {
                        match (n, f) {
                            (1, Field::Bytes(s)) => normalizer = String::from_utf8(s.to_vec())?,
                            (2, Field::Bytes(s)) => charsmap = !s.is_empty(),
                            (3, Field::Varint(v)) => add_dummy = v != 0,
                            (4, Field::Varint(v)) => remove_ws = v != 0,
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if model_type != 2 {
            bail!("tokenizer model type {model_type} is not BPE");
        }
        if charsmap || normalizer != "identity" {
            bail!("tokenizer normalizer {normalizer:?} is not supported (identity only)");
        }
        Self::build(pieces, scores, kinds, add_dummy, remove_ws, byte_fallback)
    }

    pub fn from_model_file(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_model_bytes(&buf)
    }

    /// Parse the tokenizer blob of a `.cact` archive.
    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        let rd_u32 =
            |o: usize| -> Result<u32> { Ok(u32::from_le_bytes(blob.get(o..o + 4).context("truncated tokenizer blob")?.try_into()?)) };
        let n = rd_u32(0)? as usize;
        let add_dummy = *blob.get(20).context("truncated tokenizer blob")? != 0;
        let byte_fb = *blob.get(21).context("truncated tokenizer blob")? != 0;
        let mut off = 24;
        let (mut pieces, mut scores, mut kinds) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let score = f32::from_bits(rd_u32(off)?);
            let kind = PieceKind::from_blob(*blob.get(off + 4).context("truncated tokenizer blob")?)?;
            let len = u16::from_le_bytes(blob.get(off + 5..off + 7).context("truncated")?.try_into()?) as usize;
            off += 7;
            let s = std::str::from_utf8(blob.get(off..off + len).context("truncated piece")?)?;
            off += len;
            pieces.push(s.to_string());
            scores.push(score);
            kinds.push(kind);
        }
        Self::build(pieces, scores, kinds, add_dummy, false, byte_fb)
    }

    /// The `.cact` tokenizer blob (`_tokenizer_blob`): chat markers are the
    /// only user-defined pieces it records.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for v in [self.pieces.len() as u32, PAD_ID, EOS_ID, BOS_ID, UNK_ID] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(self.add_dummy_prefix as u8);
        out.push(1);
        out.extend_from_slice(&0u16.to_le_bytes());
        for (i, p) in self.pieces.iter().enumerate() {
            let kind = match self.kinds[i] {
                k @ (PieceKind::Control | PieceKind::Unknown | PieceKind::Byte) => k,
                _ if CHAT_MARKERS.contains(&p.as_str()) => PieceKind::UserDefined,
                _ => PieceKind::Normal,
            };
            out.extend_from_slice(&self.scores[i].to_le_bytes());
            out.push(kind.blob_code());
            out.extend_from_slice(&(p.len() as u16).to_le_bytes());
            out.extend_from_slice(p.as_bytes());
        }
        out
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> &str {
        &self.pieces[id as usize]
    }

    pub fn kind(&self, id: u32) -> PieceKind {
        self.kinds[id as usize]
    }

    pub fn piece_id(&self, piece: &str) -> Option<u32> {
        self.all.get(piece).copied().filter(|&i| i > 0)
    }

    fn normalize(&self, text: &str) -> String {
        let mut s: String = if self.remove_extra_whitespaces {
            let mut out = String::with_capacity(text.len());
            for w in text.split(' ').filter(|w| !w.is_empty()) {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(w);
            }
            out
        } else {
            text.to_string()
        };
        if self.add_dummy_prefix && !s.is_empty() {
            s.insert(0, ' ');
        }
        s.replace(' ', "\u{2581}")
    }

    fn match_user_defined(&self, s: &str) -> Option<usize> {
        let b = *s.as_bytes().first()?;
        if !self.ud_first[b as usize] {
            return None;
        }
        self.user_defined.iter().find(|u| s.starts_with(u.as_str())).map(|u| u.len())
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }
        let norm = self.normalize(text);
        let bytes = norm.as_str();

        #[derive(Clone, Copy)]
        struct Sym {
            start: usize,
            end: usize,
            prev: isize,
            next: isize,
            freeze: bool,
        }
        let mut syms: Vec<Sym> = Vec::with_capacity(norm.len());
        let mut pos = 0;
        while pos < bytes.len() {
            let rest = &bytes[pos..];
            let (len, freeze) = match self.match_user_defined(rest) {
                Some(l) => (l, true),
                None => (rest.chars().next().map_or(1, char::len_utf8), false),
            };
            let i = syms.len() as isize;
            syms.push(Sym { start: pos, end: pos + len, prev: i - 1, next: i + 1, freeze });
            pos += len;
        }
        if let Some(last) = syms.last_mut() {
            last.next = -1;
        }

        #[derive(PartialEq)]
        struct Pair {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl Eq for Pair {}
        impl Ord for Pair {
            fn cmp(&self, o: &Self) -> Ordering {
                self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal).then(o.left.cmp(&self.left))
            }
        }
        impl PartialOrd for Pair {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }

        let mut heap = BinaryHeap::new();
        let try_pair = |heap: &mut BinaryHeap<Pair>, syms: &[Sym], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (a, b) = (syms[l as usize], syms[r as usize]);
            if a.freeze || b.freeze {
                return;
            }
            let piece = &bytes[a.start..b.end];
            if let Some(&id) = self.vocab.get(piece) {
                if self.kinds[id as usize] == PieceKind::Unused {
                    return;
                }
                heap.push(Pair { score: self.scores[id as usize], left: l as usize, right: r as usize, size: piece.len() });
            }
        };
        for i in 1..syms.len() {
            try_pair(&mut heap, &syms, i as isize - 1, i as isize);
        }
        while let Some(top) = heap.pop() {
            let (l, r) = (syms[top.left], syms[top.right]);
            let (llen, rlen) = (l.end - l.start, r.end - r.start);
            if llen == 0 || rlen == 0 || llen + rlen != top.size {
                continue;
            }
            // Merge right into left.
            syms[top.left].end = r.end;
            syms[top.left].next = r.next;
            if r.next >= 0 {
                syms[r.next as usize].prev = top.left as isize;
            }
            syms[top.right].start = syms[top.right].end;
            let left = syms[top.left];
            try_pair(&mut heap, &syms, left.prev, top.left as isize);
            try_pair(&mut heap, &syms, top.left as isize, left.next);
        }

        let mut ids = Vec::with_capacity(syms.len());
        let mut i: isize = if syms.is_empty() { -1 } else { 0 };
        while i >= 0 {
            let s = syms[i as usize];
            let piece = &bytes[s.start..s.end];
            match self.vocab.get(piece) {
                Some(&id) => ids.push(id),
                None if self.byte_fallback => {
                    ids.extend(piece.bytes().map(|b| self.byte_ids[b as usize]));
                }
                None => ids.push(self.unk_id),
            }
            i = s.next;
        }
        ids
    }

    /// As [`Tokenizer::decode`], but the raw bytes: byte-fallback pieces
    /// that split a UTF-8 character stay as they are instead of becoming
    /// U+FFFD, so a prefix of the output is a prefix of every extension.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(kind) = self.kinds.get(id as usize).copied() else { continue };
            match kind {
                PieceKind::Byte => out.push(u8::from_str_radix(&self.pieces[id as usize][3..5], 16).unwrap_or(b'?')),
                PieceKind::Control => {}
                PieceKind::Unknown => out.extend_from_slice(UNK_SURFACE.as_bytes()),
                _ => out.extend_from_slice(self.pieces[id as usize].replace(META_SPACE, " ").as_bytes()),
            }
        }
        if self.add_dummy_prefix && out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }

    /// `sp.Decode(ids)`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |out: &mut String, bytes: &mut Vec<u8>| {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };
        for &id in ids {
            let Some(kind) = self.kinds.get(id as usize).copied() else { continue };
            match kind {
                PieceKind::Byte => {
                    let p = &self.pieces[id as usize];
                    bytes.push(u8::from_str_radix(&p[3..5], 16).unwrap_or(b'?'));
                }
                PieceKind::Control => {}
                PieceKind::Unknown => {
                    flush(&mut out, &mut bytes);
                    out.push_str(UNK_SURFACE);
                }
                _ => {
                    flush(&mut out, &mut bytes);
                    out.push_str(&self.pieces[id as usize]);
                }
            }
        }
        flush(&mut out, &mut bytes);
        let mut text = out.replace(META_SPACE, " ");
        if self.add_dummy_prefix && text.starts_with(' ') {
            text.remove(0);
        }
        text
    }
}
