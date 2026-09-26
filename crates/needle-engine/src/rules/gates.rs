//! The gates that withhold the whole call list. Per call: origin copied
//! into a required destination (`FUN_00080a50`), an unmentioned required
//! enum option (`FUN_00081a48`), a control word or unmentioned place in a
//! required slot (`FUN_00082a48`), no quantity for a required number
//! (`FUN_00083cbc`). Across calls: only what the request excludes
//! (`FUN_000961e8`), a negated verb (`FUN_000977e4`), a reported command
//! (`FUN_00097f24`). The two cross-call gates that look at single calls
//! drop the hit calls instead when some calls survive.

use super::json::{Json, call_args, call_name};
use super::schema::{self, Tool};
use super::text::{
    NON_PLACE, PLACE_SLOT, contains, find_all, find_from, has_quantity, has_word, is_in, lower, name_words, normq, pad, split_name_any,
    squash, trim, ws_tokens,
};
use super::verbs::verb_value;

const ORIGIN_KEYS: [&str; 10] =
    ["from", "start", "origin", "source", "pickup", "departure", "start_city", "pickup_city", "from_location", "start_location"];
const DEST_KEYS: [&str; 10] =
    ["to", "end", "target", "arrival", "dropoff", "end_city", "destination", "to_location", "end_location", "dropoff_city"];

/// `FUN_00080a50`: an origin equal to the destination. A required origin
/// withholds (true); an optional one is erased.
pub fn origin_copies_destination(args: &mut Vec<(Vec<u8>, Json)>, tools: &[Tool], name: &[u8]) -> bool {
    let o = args.iter().position(|(k, _)| is_in(&lower(k), &ORIGIN_KEYS));
    let d = args.iter().position(|(k, _)| is_in(&lower(k), &DEST_KEYS));
    let (Some(o), Some(d)) = (o, d) else { return false };
    let (Json::Str(ov), Json::Str(dv)) = (&args[o].1, &args[d].1) else { return false };
    let ov = lower(trim(ov, b" \t"));
    if ov.is_empty() || ov != lower(trim(dv, b" \t")) {
        return false;
    }
    if schema::info(tools, name, &args[o].0).required {
        return true;
    }
    args.remove(o);
    false
}

/// Argument names exempt from the enum gate (substring match).
const ENUM_CONTROL: [&str; 13] =
    ["category", "type", "kind", "priority", "sentiment", "intent", "format", "unit", "metric", "status", "action", "mode", "level"];

/// `FUN_00081a48`: a required enum (two or more options) set to an option
/// the request never names.
pub fn unmentioned_enum(request: &[u8], call: &Json, tools: &[Tool], name: &[u8]) -> bool {
    let Some(args) = call_args(call) else { return false };
    let n = normq(request);
    let sqq = squash(request);
    for (key, value) in args {
        let Json::Str(v) = value else { continue };
        let info = schema::info(tools, name, key);
        if !info.required || info.options.len() < 2 {
            continue;
        }
        let kl = lower(key);
        if ENUM_CONTROL.iter().any(|w| contains(&kl, w.as_bytes())) {
            continue;
        }
        let words = name_words(v);
        if words.is_empty() && v.iter().filter(|c| c.is_ascii_alphabetic()).count() < 4 {
            continue;
        }
        let sqv = squash(v);
        let mut mentioned = if sqv.len() <= 4 && !words.is_empty() {
            has_word(&n, &sqv) || verb_value(&n, v)
        } else {
            (!sqv.is_empty() && contains(&sqq, &sqv)) || verb_value(&n, v)
        };
        if !mentioned {
            mentioned = words.iter().any(|w| {
                has_word(&n, w)
                    || (w.len() >= 4 && {
                        let mut stem = vec![b' '];
                        stem.extend_from_slice(&w[..w.len() - 1]);
                        contains(&n, &stem)
                    })
            });
        }
        if !mentioned {
            return true;
        }
    }
    false
}

const CONTROL_SLOT: [&str; 14] = [
    "action",
    "command",
    "state",
    "mode",
    "operation",
    "op",
    "setting",
    "status",
    "direction",
    "toggle",
    "switch",
    "power",
    "verb",
    "intent",
];
const CONTROL_WORDS: [&str; 19] = [
    "on", "off", "toggle", "up", "down", "dim", "brighten", "open", "close", "start", "stop", "play", "pause", "mute", "unmute", "lock",
    "unlock", "raise", "lower",
];

/// `FUN_00082a48`: a required free-text slot holding a control word, or a
/// required place slot holding a number, a non-place word, or a place the
/// conversation never mentions.
pub fn control_or_place(squashed: &[u8], call: &Json, tools: &[Tool], name: &[u8]) -> bool {
    let Some(args) = call_args(call) else { return false };
    for (key, value) in args {
        let Json::Str(val) = value else { continue };
        let info = schema::info(tools, name, key);
        if !info.required || info.has_enum || info.ty != b"string" {
            continue;
        }
        let n = lower(key);
        let v = lower(val);
        if split_name_any(&n, |w| is_in(w, &CONTROL_SLOT)) {
            continue;
        }
        if contains(&normq(&info.description), &pad(&v)) {
            continue;
        }
        if is_in(&v, &CONTROL_WORDS) {
            return true;
        }
        if info.has_default && squash(&info.default) == squash(val) {
            continue;
        }
        if split_name_any(&n, |w| is_in(w, &PLACE_SLOT)) {
            let head = val.split(|c| *c == b',').next().unwrap_or_default();
            let sqp = squash(head);
            if !sqp.is_empty() && sqp.iter().all(u8::is_ascii_digit) {
                return true;
            }
            if is_in(&v, &NON_PLACE) {
                return true;
            }
            if sqp.len() >= 2 && !contains(squashed, &sqp) {
                return true;
            }
        }
    }
    false
}

/// `FUN_00083cbc`: the first required number with no enum and no default,
/// when the conversation names no quantity at all.
pub fn no_quantity(squashed: &[u8], request: &[u8], call: &Json, tools: &[Tool], name: &[u8]) -> bool {
    let Some(args) = call_args(call) else { return false };
    let first = args.iter().find(|(k, v)| {
        let info = schema::info(tools, name, k);
        matches!(v, Json::Num(_)) && info.required && !info.has_enum && !info.has_default
    });
    first.is_some() && !has_quantity(squashed, request)
}

/// Clause terminators after an exclusion or a negation (table `0xbdf20`).
const TERMINATORS: [&str; 12] = [",", ".", ";", "!", "?", " and ", " but ", " then ", " instead", " just ", " only ", " rather"];

/// The first terminator at or after `from`, or the end.
fn terminator(s: &[u8], from: usize) -> Option<usize> {
    TERMINATORS.iter().filter_map(|t| find_from(s, t.as_bytes(), from)).min()
}

const EXCLUSION_CUES: [&str; 10] =
    ["except for", "except", "but not", "but leave", "leave out", "apart from", "other than", "excluding", "but skip", "skip"];

/// The excluded phrases of a request, squashed: up to four words after each
/// exclusion cue, leading articles skipped.
pub fn excluded_phrases(request: &[u8]) -> Vec<Vec<u8>> {
    let p = pad(&lower(request));
    let mut out = vec![];
    for cue in EXCLUSION_CUES {
        let needle = pad(cue.as_bytes());
        for at in find_all(&p, &needle) {
            let start = at + needle.len();
            let end = terminator(&p, start).unwrap_or(p.len());
            let mut words: Vec<&[u8]> = vec![];
            for t in ws_tokens(&p[start..end.max(start)]) {
                if words.is_empty() && matches!(t, b"the" | b"my" | b"a" | b"an") {
                    continue;
                }
                words.push(trim(t, b"!,.:;?"));
                if words.len() == 4 {
                    break;
                }
            }
            if !words.is_empty() {
                out.push(words.concat());
            }
        }
    }
    out
}

/// Mark calls, then: all hit withholds, some hit are dropped.
fn all_or_drop(calls: &mut Vec<Json>, hit: impl Fn(&Json) -> bool) -> bool {
    let marks: Vec<bool> = calls.iter().map(&hit).collect();
    if !marks.iter().any(|m| *m) {
        return false;
    }
    if marks.iter().all(|m| *m) {
        return true;
    }
    let mut it = marks.iter();
    calls.retain(|_| !it.next().copied().unwrap_or(false));
    false
}

/// `FUN_000961e8`: calls whose short string arguments name only what the
/// request excludes ("everything except the kitchen").
pub fn excluded(calls: &mut Vec<Json>, request: &[u8]) -> bool {
    let excl = excluded_phrases(request);
    if excl.is_empty() {
        return false;
    }
    all_or_drop(calls, |call| {
        call_args(call).is_some_and(|args| {
            args.iter().any(|(_, v)| {
                let Json::Str(s) = v else { return false };
                if ws_tokens(s).len() >= 5 {
                    return false;
                }
                let v = squash(s);
                v.len() >= 3 && excl.iter().any(|e| e == &v || contains(e, &v) || e.is_empty() || contains(&v, e))
            })
        })
    })
}

const NEGATION_CUES: [&str; 11] =
    ["don't", "do not", "dont", "never", "no longer", "must not", "mustn't", "shouldn't", "should not", "make sure not", "no need"];

/// `FUN_000ac408`: a negation is forgiven when a terminator follows it with
/// more text after, and none of the call's values sit in the negated span.
fn negation_elsewhere(lq: &[u8], p: usize, call: &Json) -> bool {
    let Some(t) = terminator(lq, p.min(lq.len())) else { return false };
    let clause = squash(&lq[p.min(t)..t]);
    if !lq[t..].iter().any(u8::is_ascii_alphanumeric) {
        return false;
    }
    let Some(args) = call_args(call) else { return false };
    let mut seen = false;
    for (_, v) in args {
        if !matches!(v, Json::Num(_) | Json::Str(_)) {
            continue;
        }
        let v = squash(v.text());
        if v.len() < 2 {
            continue;
        }
        if contains(&clause, &v) {
            return false;
        }
        seen = true;
    }
    seen
}

/// `FUN_000abd1c`: the request negates a word of the call's tool name
/// within the three words after a negation cue.
pub fn call_negated(lq: &[u8], call: &Json) -> bool {
    let Some(name) = call_name(call) else { return false };
    let words = name_words(name);
    let p = pad(lq);
    for w in &words {
        for cue in NEGATION_CUES {
            let needle = pad(cue.as_bytes());
            for at in find_all(&p, &needle) {
                let pos = at + needle.len();
                if ["forget", "hesitate", "worry", "mind", "disturb"].iter().any(|x| p[pos..].starts_with(x.as_bytes())) {
                    continue;
                }
                let mut cands = vec![pos];
                if let Some(s) = find_from(&p, b" ", pos) {
                    cands.push(s + 1);
                    if let Some(s2) = find_from(&p, b" ", s + 1) {
                        cands.push(s2 + 1);
                    }
                }
                for pk in cands {
                    // The padded text is one byte ahead of `lq`.
                    if p[pk..].starts_with(w) && !negation_elsewhere(lq, pk - 1, call) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// `FUN_000977e4`: the request negates the called tool's verb.
pub fn negated(calls: &mut Vec<Json>, request: &[u8]) -> bool {
    let lq = lower(request);
    all_or_drop(calls, |c| call_negated(&lq, c))
}

const REPORTING: [&str; 9] = ["texted", "said", "wrote", "emailed", "messaged", "posted", "complained", "mentioned", "told me"];

/// `FUN_00097f24`: a reporting verb followed within 40 bytes by a quote
/// mark (an apostrophe counts).
pub fn reported(request: &[u8]) -> bool {
    let lq = lower(request);
    REPORTING.iter().any(|verb| {
        find_all(&lq, verb.as_bytes()).into_iter().any(|i| {
            let from = i + verb.len();
            let to = (from + 40).min(lq.len());
            (from..to).any(|j| {
                matches!(lq[j], b'\'' | b'"')
                    || (j + 2 < lq.len() && lq[j] == 0xe2 && lq[j + 1] == 0x80 && matches!(lq[j + 2], 0x9c | 0x98))
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::json::parse;
    use crate::rules::schema::parse_tools;

    fn call(s: &str) -> Json {
        parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn exclusion_phrases_overlap() {
        let e = excluded_phrases(b"turn off every light except for the kitchen, thanks");
        assert!(e.contains(&b"kitchen".to_vec()) && e.contains(&b"forthekitchen".to_vec()));
        let mut calls = vec![call(r#"{"name":"l","arguments":{"room":"kitchen"}}"#), call(r#"{"name":"l","arguments":{"room":"hall"}}"#)];
        assert!(!excluded(&mut calls, b"turn off every light except for the kitchen"));
        assert_eq!(calls.len(), 1);
        let mut one = vec![call(r#"{"name":"l","arguments":{"room":"kitchen"}}"#)];
        assert!(excluded(&mut one, b"turn off every light except the kitchen"));
    }

    #[test]
    fn negation_cases() {
        let timer = call(r#"{"name":"set_timer","arguments":{"minutes":20}}"#);
        assert!(call_negated(b"don't set a timer for 20 minutes", &timer));
        assert!(!call_negated(b"don't ever set a thing", &timer));
        let vac = call(r#"{"name":"start_robot_vacuum","arguments":{}}"#);
        assert!(call_negated(b"never run the vacuum", &vac));
        let lights = call(r#"{"name":"control_lights","arguments":{"room":"study"}}"#);
        assert!(!call_negated(b"don't turn on the study lights", &lights));
        assert!(call_negated(b"don't turn lights on", &lights));
        assert!(!call_negated(b"don't forget to set the timer", &timer));
        // A terminator, more text, and no value inside the negated span.
        let t2 = call(r#"{"name":"set_timer","arguments":{"label":"eggs"}}"#);
        assert!(!call_negated(b"don't set the timer yet, set one for eggs", &t2));
    }

    #[test]
    fn reported_command() {
        assert!(reported(b"mom said 'turn off the lights'"));
        assert!(reported(b"mom said she'll call"));
        assert!(!reported(b"turn off the lights"));
    }

    #[test]
    fn required_slots() {
        let tools = parse_tools(
            br#"[{"name":"t","parameters":{"properties":{"room":{"type":"string"},"n":{"type":"number"},
                 "door":{"type":"string","enum":["front","back door"]}},"required":["room","n","door"]}}]"#,
        );
        let c = |room: &str| call(&format!(r#"{{"name":"t","arguments":{{"room":"{room}"}}}}"#));
        assert!(control_or_place(b"heatthekitchen", &c("off"), &tools, b"t"));
        assert!(control_or_place(b"heatthekitchen", &c("garage"), &tools, b"t"));
        assert!(!control_or_place(b"heatthekitchen", &c("Kitchen"), &tools, b"t"));
        assert!(control_or_place(b"heatthekitchen", &c("friday"), &tools, b"t"));
        let n = call(r#"{"name":"t","arguments":{"n":3}}"#);
        assert!(no_quantity(b"heat", b"heat", &n, &tools, b"t"));
        assert!(!no_quantity(b"heat", b"heat it twenty", &n, &tools, b"t"));
        let d = call(r#"{"name":"t","arguments":{"door":"back door"}}"#);
        assert!(unmentioned_enum(b"lock the front one", &d, &tools, b"t"));
        assert!(!unmentioned_enum(b"lock the back one", &d, &tools, b"t"));
    }

    #[test]
    fn origin_destination() {
        let tools = parse_tools(br#"[{"name":"r","parameters":{"properties":{"from":{"type":"string"},"to":{"type":"string"}}}}]"#);
        let mut args = vec![(b"from".to_vec(), Json::Str(b"Boston ".to_vec())), (b"to".to_vec(), Json::Str(b"boston".to_vec()))];
        assert!(!origin_copies_destination(&mut args, &tools, b"r"));
        assert_eq!(args.len(), 1);
    }
}
