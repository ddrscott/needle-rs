//! `json.dumps` as CPython writes it. Prompts embed serialized tool schemas
//! and calls, so the bytes (float repr, escapes, separators) must match the
//! reference or the model sees a different token stream.

use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub struct DumpOpts {
    /// `separators=(",", ":")` when true, else `(", ", ": ")`.
    pub compact: bool,
    pub sort_keys: bool,
    pub ensure_ascii: bool,
}

impl DumpOpts {
    /// `json.dumps(x, separators=(",", ":"), ensure_ascii=False)`
    pub const PROMPT: DumpOpts = DumpOpts { compact: true, sort_keys: false, ensure_ascii: false };
    /// `json.dumps(x)`
    pub const DEFAULT: DumpOpts = DumpOpts { compact: false, sort_keys: false, ensure_ascii: true };
    /// `json.dumps(x, sort_keys=True)`
    pub const SORTED: DumpOpts = DumpOpts { compact: false, sort_keys: true, ensure_ascii: true };
}

pub fn dumps(v: &Value, opts: DumpOpts) -> String {
    let mut out = String::new();
    write_value(&mut out, v, opts);
    out
}

/// `json.dumps(x, separators=(",", ":"), ensure_ascii=False)`.
pub fn dumps_compact(v: &Value) -> String {
    dumps(v, DumpOpts::PROMPT)
}

fn write_value(out: &mut String, v: &Value, o: DumpOpts) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(out, s, o.ensure_ascii),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(if o.compact { "," } else { ", " });
                }
                write_value(out, item, o);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if o.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (k, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(if o.compact { "," } else { ", " });
                }
                write_str(out, k, o.ensure_ascii);
                out.push_str(if o.compact { ":" } else { ": " });
                write_value(out, item, o);
            }
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// CPython `float.__repr__`: shortest round-trip digits, fixed notation for
/// decimal exponents in [-4, 16), scientific (`1e-05`, `1e+16`) otherwise.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` is the shortest round-trip representation.
    let sci = format!("{:e}", f);
    let (mant, exp) = sci.split_once('e').expect("scientific form");
    let exp: i32 = exp.parse().expect("exponent");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp >= 0 {
            if n > exp + 1 {
                format!("{}.{}", &digits[..(exp + 1) as usize], &digits[(exp + 1) as usize..])
            } else {
                format!("{}{}.0", digits, "0".repeat((exp + 1 - n) as usize))
            }
        } else {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{esign}{:02}", exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_match_cpython() {
        for (f, s) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456.789, "123456.789"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (-2.5e-7, "-2.5e-07"),
            (1.23456, "1.23456"),
            (1.5e300, "1.5e+300"),
        ] {
            assert_eq!(float_repr(f), s);
        }
    }

    #[test]
    fn separators_and_escapes() {
        let v = json!({"a": [1, 2.5, "é\n\"x\""], "b": null});
        assert_eq!(dumps_compact(&v), "{\"a\":[1,2.5,\"é\\n\\\"x\\\"\"],\"b\":null}");
        assert_eq!(dumps(&v, DumpOpts::DEFAULT), "{\"a\": [1, 2.5, \"\\u00e9\\n\\\"x\\\"\"], \"b\": null}");
    }
}
