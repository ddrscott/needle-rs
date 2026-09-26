//! The engine's own JSON value: byte strings, numbers kept as their source
//! text, object members in source order with duplicate keys kept. Lookups
//! take the first member with a key. The parser and the compact writer
//! follow the library's (`FUN_0001b87c`, `FUN_00022180`), quirks included.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// The number's source text.
    Num(Vec<u8>),
    Str(Vec<u8>),
    Arr(Vec<Json>),
    Obj(Vec<(Vec<u8>, Json)>),
}

impl Json {
    /// The first member named `key`.
    pub fn get(&self, key: &[u8]) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &[u8]) -> Option<&mut Json> {
        match self {
            Json::Obj(m) => m.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&[u8]> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The scalar text the library keeps for a value: a string's bytes, a
    /// number's source, `true`/`false`; empty otherwise.
    pub fn text(&self) -> &[u8] {
        match self {
            Json::Str(s) | Json::Num(s) => s,
            Json::Bool(true) => b"true",
            Json::Bool(false) => b"false",
            _ => b"",
        }
    }
}

/// A call's name (the first `name` member, when it is a string).
pub fn call_name(call: &Json) -> Option<&[u8]> {
    call.get(b"name").and_then(Json::as_str)
}

/// A call's argument members (the first `arguments` member, when it is an
/// object).
pub fn call_args(call: &Json) -> Option<&Vec<(Vec<u8>, Json)>> {
    match call.get(b"arguments") {
        Some(Json::Obj(m)) => Some(m),
        _ => None,
    }
}

pub fn call_args_mut(call: &mut Json) -> Option<&mut Vec<(Vec<u8>, Json)>> {
    match call.get_mut(b"arguments") {
        Some(Json::Obj(m)) => Some(m),
        _ => None,
    }
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// Parse a whole text: one value, then only whitespace.
pub fn parse(s: &[u8]) -> Option<Json> {
    let mut p = Parser { s, pos: 0, depth: 0 };
    let v = p.value()?;
    p.ws();
    (p.pos == s.len()).then_some(v)
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
    depth: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.pos < self.s.len() && is_ws(self.s[self.pos]) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn value(&mut self) -> Option<Json> {
        self.ws();
        if self.pos >= self.s.len() || self.depth > 64 {
            return None;
        }
        match self.s[self.pos] {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Json::Str),
            _ if self.s[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Some(Json::Bool(true))
            }
            _ if self.s[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Some(Json::Bool(false))
            }
            _ if self.s[self.pos..].starts_with(b"null") => {
                self.pos += 4;
                Some(Json::Null)
            }
            _ => self.number(),
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.pos;
        let s = self.s;
        let mut i = start;
        if s.get(i) == Some(&b'-') {
            i += 1;
        }
        match s.get(i) {
            Some(b'0') => i += 1,
            Some(b'1'..=b'9') => {
                while s.get(i).is_some_and(u8::is_ascii_digit) {
                    i += 1;
                }
            }
            _ => return None,
        }
        if s.get(i) == Some(&b'.') {
            i += 1;
            if !s.get(i).is_some_and(u8::is_ascii_digit) {
                return None;
            }
            while s.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        if matches!(s.get(i), Some(b'e' | b'E')) {
            i += 1;
            if matches!(s.get(i), Some(b'+' | b'-')) {
                i += 1;
            }
            if !s.get(i).is_some_and(u8::is_ascii_digit) {
                return None;
            }
            while s.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        self.pos = i;
        Some(Json::Num(s[start..i].to_vec()))
    }

    /// A string at `"`: the decoded bytes. The library's escape decoding
    /// passes an unknown escape through as its character and keeps an
    /// invalid `\u` as a literal `u`.
    fn string(&mut self) -> Option<Vec<u8>> {
        let s = self.s;
        let mut i = self.pos + 1;
        let mut out = vec![];
        while i < s.len() && s[i] != b'"' {
            if s[i] != b'\\' {
                out.push(s[i]);
                i += 1;
                continue;
            }
            if i + 1 >= s.len() {
                out.push(b'\\');
                i += 1;
                continue;
            }
            let e = s[i + 1];
            i += 2;
            match e {
                b'b' => out.push(0x08),
                b'f' => out.push(0x0c),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => match hex4(&s[i..]) {
                    Some(mut cp) => {
                        i += 4;
                        if (0xd800..0xdc00).contains(&cp)
                            && s[i..].starts_with(b"\\u")
                            && let Some(lo) = hex4(&s[i + 2..])
                            && (0xdc00..0xe000).contains(&lo)
                        {
                            cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                            i += 6;
                        }
                        push_utf8(&mut out, cp);
                    }
                    None => out.push(b'u'),
                },
                c => out.push(c),
            }
        }
        if i >= s.len() {
            return None;
        }
        self.pos = i + 1;
        Some(out)
    }

    fn object(&mut self) -> Option<Json> {
        self.pos += 1;
        self.depth += 1;
        let mut members = vec![];
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Some(Json::Obj(members));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return None;
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return None;
            }
            self.pos += 1;
            let v = self.value()?;
            members.push((key, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Some(Json::Obj(members));
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self) -> Option<Json> {
        self.pos += 1;
        self.depth += 1;
        let mut items = vec![];
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Some(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Some(Json::Arr(items));
                }
                _ => return None,
            }
        }
    }
}

fn hex4(s: &[u8]) -> Option<u32> {
    if s.len() < 4 || !s[..4].iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    u32::from_str_radix(std::str::from_utf8(&s[..4]).ok()?, 16).ok()
}

/// UTF-8 bytes of a code point; a lone surrogate gets its 3-byte form.
fn push_utf8(out: &mut Vec<u8>, cp: u32) {
    match cp {
        0..0x80 => out.push(cp as u8),
        0x80..0x800 => out.extend([0xc0 | (cp >> 6) as u8, 0x80 | (cp & 0x3f) as u8]),
        0x800..0x10000 => out.extend([0xe0 | (cp >> 12) as u8, 0x80 | ((cp >> 6) & 0x3f) as u8, 0x80 | (cp & 0x3f) as u8]),
        _ => out.extend([
            0xf0 | (cp >> 18) as u8,
            0x80 | ((cp >> 12) & 0x3f) as u8,
            0x80 | ((cp >> 6) & 0x3f) as u8,
            0x80 | (cp & 0x3f) as u8,
        ]),
    }
}

/// Compact JSON: no spaces, members in order, numbers as their source text,
/// strings escaped with the short forms and `\u00xx` for other controls.
pub fn write(v: &Json) -> Vec<u8> {
    let mut out = vec![];
    write_into(&mut out, v);
    out
}

fn write_into(out: &mut Vec<u8>, v: &Json) {
    match v {
        Json::Null => out.extend_from_slice(b"null"),
        Json::Bool(_) | Json::Num(_) => out.extend_from_slice(v.text()),
        Json::Str(s) => write_str(out, s),
        Json::Arr(items) => {
            out.push(b'[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_into(out, x);
            }
            out.push(b']');
        }
        Json::Obj(members) => {
            out.push(b'{');
            for (i, (k, x)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write_into(out, x);
            }
            out.push(b'}');
        }
    }
}

fn write_str(out: &mut Vec<u8>, s: &[u8]) {
    out.push(b'"');
    for &c in s {
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            c if c < 0x20 => out.extend_from_slice(format!("\\u{c:04x}").as_bytes()),
            c => out.push(c),
        }
    }
    out.push(b'"');
}

/// The value as `serde_json`, for the envelope. Numbers go through
/// `serde_json`'s own number parsing, so a source text it would print
/// differently (`1e5`, `0.50`) is canonicalized there.
pub fn to_serde(v: &Json) -> serde_json::Value {
    use serde_json::Value;
    let text = |s: &[u8]| String::from_utf8_lossy(s).into_owned();
    match v {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Num(n) => serde_json::from_slice::<serde_json::Number>(n).map_or(Value::Null, Value::Number),
        Json::Str(s) => Value::String(text(s)),
        Json::Arr(items) => Value::Array(items.iter().map(to_serde).collect()),
        Json::Obj(members) => Value::Object(members.iter().map(|(k, x)| (text(k), to_serde(x))).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_order_duplicates_and_number_text() {
        let src = " [{\"b\":1.50,\"a\":\"x\\u00e9\\/\",\"b\":true,\"n\":null,\"e\":-2e+3}] ".as_bytes();
        let v = parse(src).unwrap();
        assert_eq!(write(&v), "[{\"b\":1.50,\"a\":\"x\u{e9}/\",\"b\":true,\"n\":null,\"e\":-2e+3}]".as_bytes());
        assert_eq!(v.clone(), parse(&write(&v)).unwrap());
    }

    #[test]
    fn parser_quirks() {
        assert!(parse(b"[1,]").is_none());
        assert!(parse(b"01").is_none());
        assert!(parse(b"1.").is_none());
        assert_eq!(parse(br#""\q\uZZ""#), Some(Json::Str(b"quZZ".to_vec())));
        assert_eq!(write(&Json::Str(vec![1, b'\t'])), br#""\u0001\t""#.to_vec());
        assert_eq!(parse(b"\"\\ud83d\\ude00\""), Some(Json::Str("\u{1f600}".as_bytes().to_vec())));
    }
}
