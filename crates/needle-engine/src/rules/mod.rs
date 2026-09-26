//! The engine's deterministic post-processing of a decoded call list, a
//! port of the native library's `FUN_0006998c`: argument repairs, the
//! gates that withhold the calls, date resolution and the duplicate drop.
//! See `docs/engine-rules.md` for each rule and its evidence.
//!
//! Everything works on bytes, as the library does: only ASCII is ever
//! lowercased, and numbers keep their source text.

pub mod dates;
pub mod gates;
pub mod json;
pub mod repairs;
pub mod schema;
pub mod tail;
pub mod text;
pub mod validation;
pub mod verbs;

use json::{Json, call_args, call_args_mut, call_name};
use schema::Tool;
use text::{lower, squash, trim};

pub use validation::{negation_cue, ungrounded_fields};

/// What the post-processor reads besides the calls.
pub struct Context<'a> {
    /// The current user request.
    pub request: &'a [u8],
    /// The system text as stored at init: the `date:` fact and timestamps
    /// blanked.
    pub system: &'a [u8],
    /// The running conversation: the stored system text, then `"\n"` plus
    /// every input so far, this one included.
    pub conversation: &'a [u8],
    /// The declared tools (all of them, not a retrieved subset).
    pub tools: &'a [Tool],
}

/// The post-processed call list and the two withhold flags. `withhold`
/// comes from any gate; `withhold_all` only from the cross-call gates (the
/// caller uses it instead of `withhold` when a trigger forced the call).
#[derive(Debug)]
pub struct Outcome {
    pub calls: Vec<u8>,
    pub withhold: bool,
    pub withhold_all: bool,
}

/// `FUN_0006998c`: repair and gate a decoded call list (compact JSON).
/// Text that is not a non-empty JSON array comes back unchanged.
pub fn postprocess(calls_json: &[u8], ctx: &Context) -> Outcome {
    let mut out = Outcome { calls: calls_json.to_vec(), withhold: false, withhold_all: false };
    let Some(Json::Arr(calls)) = json::parse(calls_json) else { return out };
    if calls.is_empty() {
        return out;
    }
    let tools = ctx.tools;
    let request = ctx.request;
    let quotes = repairs::quoted_spans(request);
    let base = if ctx.conversation.is_empty() { ctx.system } else { ctx.conversation };
    let squashed = squash(&[base, b"\n", request].concat());
    let ok = !trim(request, b" \t\n").is_empty();

    let mut calls = repairs::merge_split_place(calls, request, tools);
    for call in calls.iter_mut() {
        if !matches!(call, Json::Obj(_)) || call_name(call).is_none() || call_args(call).is_none() {
            continue;
        }
        if let Some(args) = call_args_mut(call) {
            repairs::split_name(args, request);
        }
        if ok
            && let Some(new) = verbs::polarity_rename(request, tools, call_name(call).unwrap_or_default())
            && let Some(Json::Str(n)) = call.get_mut(b"name")
        {
            *n = new;
        }
        let name = call_name(call).unwrap_or_default().to_vec();
        if let Some(args) = call_args_mut(call) {
            verbs::flip(args, request, tools, &name);
            if gates::origin_copies_destination(args, tools, &name) {
                out.withhold = true;
            }
        }
        if ok
            && (gates::unmentioned_enum(request, call, tools, &name)
                | gates::control_or_place(&squashed, call, tools, &name)
                | gates::no_quantity(&squashed, request, call, tools, &name))
        {
            out.withhold = true;
        }
        let Some(args) = call_args_mut(call) else { continue };
        rebuild_args(args, request, &quotes, tools, &name);
        if ok {
            tail::drop_ungrounded(args, &squashed, request, tools, &name);
            tail::reground_name(args, &squashed, request, tools, &name);
            tail::apply_defaults(args, &squashed, request, tools, &name);
            tail::snap_enum(args, request, tools, &name);
            tail::route_from_request(args, &squashed, request, tools, &name);
            tail::amount_from_by(args, request, tools, &name);
        }
        tail::floats_for_numbers(args, tools, &name);
    }
    if ok {
        let hits = [gates::excluded(&mut calls, request), gates::negated(&mut calls, request), gates::reported(request)];
        if hits.iter().any(|h| *h) {
            out.withhold = true;
            out.withhold_all = true;
        }
    }
    if let Some(today) = dates::today(ctx.system) {
        dates::resolve(&mut calls, request, today);
    }
    let mut seen: Vec<Vec<u8>> = vec![];
    calls.retain(|c| {
        let s = json::write(c);
        if seen.contains(&s) {
            return false;
        }
        seen.push(s);
        true
    });
    out.calls = json::write(&Json::Arr(calls));
    out
}

/// The per-argument rebuild: MIME types, phone completion, quoted text,
/// place recovery, casing and capitals for free-text strings; temperature
/// conversion for numbers. Empty optional strings (and optional `*mail*`
/// strings with no `@`) are dropped.
fn rebuild_args(args: &mut Vec<(Vec<u8>, Json)>, request: &[u8], quotes: &[Vec<u8>], tools: &[Tool], name: &[u8]) {
    let tooltext = repairs::tool_text(tools, name);
    args.retain_mut(|(key, value)| {
        if matches!(value, Json::Str(_) | Json::Arr(_)) {
            let info = schema::info(tools, name, key);
            if !info.has_enum && repairs::wants_mime(key, &info.description) {
                repairs::to_mime(value);
            }
        }
        match value {
            Json::Str(s) => {
                let info = schema::info(tools, name, key);
                let mut v = repairs::complete_phone(key, s, request);
                if !info.has_enum {
                    v = repairs::restore_quoted(&v, quotes);
                    if repairs::is_place_param(key, &info.description, &tooltext) {
                        v = repairs::recover_place(&v, request);
                    }
                    v = repairs::restore_casing(&v, request, &info.description);
                    v = repairs::capitalize(key, v);
                }
                let keep = info.required || (!v.is_empty() && (!text::contains(&lower(key), b"mail") || v.contains(&b'@')));
                *s = v;
                keep
            }
            Json::Num(t) => {
                let info = schema::info(tools, name, key);
                if !info.has_enum {
                    *t = repairs::convert_temperature(key, &info.description, t, request);
                }
                true
            }
            _ => true,
        }
    });
}

/// The system text as the library stores it for grounding: every
/// `YYYY-MM-DD[T ]HH:MM` timestamp blanked to spaces, then the first
/// `date: ` fact (up to the next `;`) replaced by one space.
pub fn grounding_text(system: &str) -> Vec<u8> {
    let mut s = system.as_bytes().to_vec();
    let d = |c: u8| c.is_ascii_digit();
    let mut p = 0;
    while p + 16 <= s.len() {
        let w = &s[p..p + 16];
        let stamp = w[..4].iter().all(|c| d(*c))
            && w[4] == b'-'
            && d(w[5])
            && d(w[6])
            && w[7] == b'-'
            && d(w[8])
            && d(w[9])
            && matches!(w[10], b'T' | b' ')
            && d(w[11])
            && d(w[12])
            && w[13] == b':'
            && d(w[14])
            && d(w[15]);
        if stamp {
            s[p..p + 16].fill(b' ');
            p += 16;
        } else {
            p += 1;
        }
    }
    if let Some(at) = text::find(&s, b"date: ") {
        let end = text::find_from(&s, b";", at).unwrap_or(s.len());
        s.splice(at..end, *b" ");
    }
    s
}

/// Whether the request names an enum option: lowercase letters and digits
/// with everything else as word breaks, a short option (under five
/// characters) as a whole word, a longer one anywhere.
pub fn names_option(text: &str, option: &str) -> bool {
    let norm = |s: &str| {
        let mut out = String::from(" ");
        for c in s.chars() {
            if c.is_ascii_alphanumeric() {
                out.push(c.to_ascii_lowercase());
            } else if !out.ends_with(' ') {
                out.push(' ');
            }
        }
        if !out.ends_with(' ') {
            out.push(' ');
        }
        out
    };
    let (hay, opt) = (norm(text), norm(option));
    let bare = opt.trim();
    if bare.is_empty() {
        return false;
    }
    if bare.len() < 5 { hay.contains(&opt) } else { hay.contains(bare) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tools: &str, system: &str, request: &str, calls: &str) -> Outcome {
        let tools = schema::parse_tools(tools.as_bytes());
        let conv = [system.as_bytes(), b"\n", request.as_bytes()].concat();
        let ctx = Context { request: request.as_bytes(), system: system.as_bytes(), conversation: &conv, tools: &tools };
        postprocess(calls.as_bytes(), &ctx)
    }

    const TIMER: &str = r#"[{"name":"set_timer","parameters":{"type":"object","properties":{"minutes":{"type":"number"},"label":{"type":"string"}},"required":["minutes"]}}]"#;

    #[test]
    fn floats_dedupe_and_passthrough() {
        let o = run(
            TIMER,
            " ",
            "set a timer for 20 minutes",
            r#"[{"name":"set_timer","arguments":{"minutes":20}},{"name":"set_timer","arguments":{"minutes":20}}]"#,
        );
        assert_eq!(String::from_utf8(o.calls).unwrap(), r#"[{"name":"set_timer","arguments":{"minutes":20.0}}]"#);
        assert!(!o.withhold);
        assert_eq!(run(TIMER, "", "x", "[]").calls, b"[]");
        assert_eq!(run(TIMER, "", "x", " {} ").calls, b" {} ");
    }

    #[test]
    fn negation_withholds_everything() {
        let o = run(TIMER, " ", "don't set a timer for 20 minutes", r#"[{"name":"set_timer","arguments":{"minutes":20}}]"#);
        assert!(o.withhold && o.withhold_all);
        let o = run(TIMER, " ", "set a timer", r#"[{"name":"set_timer","arguments":{"minutes":20}}]"#);
        assert!(o.withhold && !o.withhold_all);
    }

    #[test]
    fn grounding_blanks_the_date_fact() {
        assert_eq!(grounding_text("date: 2026-09-25 Fri 10:00; locale: en-US"), b" ; locale: en-US");
        assert_eq!(grounding_text("date: 2026-09-25 Fri 10:00"), b" ");
        assert_eq!(grounding_text("Today is 2026-09-25."), b"Today is 2026-09-25.");
    }
}
