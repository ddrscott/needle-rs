//! Repairs that run after the per-argument rebuild: dropping ungrounded
//! optionals (`FUN_0008d284`), re-grounding a required name slot
//! (`FUN_0008e128`), default substitutions (`FUN_0008ec38`), snapping an
//! enum to the option the request names (`FUN_0008fe64`), origin and
//! destination from `from`/`to` phrases (`FUN_000911d4`), a missing number
//! from "by N" (`FUN_000951a8`), and `.0` on number-typed integers
//! (`FUN_00095d08`).

use super::json::Json;
use super::schema::{self, Tool, find_tool};
use super::text::{
    PLACE_SLOT, contains, date_key, find_all, has_quantity, id_key, is_in, lower, name_words, normq, pad, split_name_any, squash, trim,
    word_tokens, ws_tokens,
};

type Args = Vec<(Vec<u8>, Json)>;

/// `FUN_000a7ca8`: `23.0` as `23`, `2.50` as `2.5`.
fn canonical_number(text: &[u8]) -> Vec<u8> {
    let mut c = text.to_vec();
    if c.contains(&b'.') {
        while c.last() == Some(&b'0') {
            c.pop();
        }
        if c.last() == Some(&b'.') {
            c.pop();
        }
    }
    c
}

/// `FUN_000a7e50`: `c` occurs in `hay` not glued to other digits.
fn standalone(c: &[u8], hay: &[u8]) -> bool {
    let d = |i: usize| hay.get(i).is_some_and(u8::is_ascii_digit);
    find_all(hay, c).into_iter().any(|p| {
        let before = (p >= 1 && d(p - 1)) || (p >= 2 && hay[p - 1] == b'.' && d(p - 2));
        let e = p + c.len();
        let after = d(e) || (hay.get(e) == Some(&b'.') && d(e + 1));
        !before && !after
    })
}

const NUMBER_WORDS: [(&str, &str); 39] = [
    ("0", "zero"),
    ("1", "one"),
    ("1", "single"),
    ("2", "two"),
    ("2", "couple"),
    ("2", "double"),
    ("2", "twice"),
    ("3", "three"),
    ("3", "triple"),
    ("4", "four"),
    ("5", "five"),
    ("6", "six"),
    ("7", "seven"),
    ("8", "eight"),
    ("9", "nine"),
    ("10", "ten"),
    ("11", "eleven"),
    ("12", "twelve"),
    ("12", "dozen"),
    ("13", "thirteen"),
    ("14", "fourteen"),
    ("15", "fifteen"),
    ("16", "sixteen"),
    ("17", "seventeen"),
    ("18", "eighteen"),
    ("19", "nineteen"),
    ("20", "twenty"),
    ("30", "thirty"),
    ("40", "forty"),
    ("50", "fifty"),
    ("60", "sixty"),
    ("70", "seventy"),
    ("80", "eighty"),
    ("90", "ninety"),
    ("100", "hundred"),
    ("1000", "thousand"),
    ("1000000", "million"),
    ("0.5", "half"),
    ("0.25", "quarter"),
];

/// `FUN_000a7184`: the squashed conversation states this number (as
/// digits, with thousands commas, or as a number word).
pub fn number_grounded(hay: &[u8], text: &[u8]) -> bool {
    let c = canonical_number(text);
    if c.is_empty() || standalone(&c, hay) {
        return true;
    }
    let mut hay2 = hay.first().map(|b| vec![*b]).unwrap_or_default();
    let mut i = 0;
    while i + 2 <= hay.len() {
        let drop = hay[i + 1] == b',' && hay[i].is_ascii_digit() && i + 4 < hay.len() && hay[i + 2..=i + 4].iter().all(u8::is_ascii_digit);
        if !drop {
            hay2.push(hay[i + 1]);
        }
        i += 1;
    }
    if hay2.len() != hay.len() && standalone(&c, &hay2) {
        return true;
    }
    NUMBER_WORDS.iter().any(|(d, w)| d.as_bytes() == c.as_slice() && contains(hay, w.as_bytes()))
}

/// A standalone 19xx/20xx year in `v`.
fn has_year(v: &[u8]) -> bool {
    (0..v.len().saturating_sub(3)).any(|p| {
        (v[p..].starts_with(b"19") || v[p..].starts_with(b"20"))
            && v[p..p + 4].iter().all(u8::is_ascii_digit)
            && (p == 0 || !v[p - 1].is_ascii_digit())
            && v.get(p + 4).is_none_or(|c| !c.is_ascii_digit())
    })
}

/// `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM` or `YYYY-MM-DDTHH:MM:SS`.
pub fn iso_shape(v: &[u8]) -> bool {
    matches!(v.len(), 10 | 16 | 19)
        && v.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            10 => *c == b'T',
            13 | 16 => *c == b':',
            _ => c.is_ascii_digit(),
        })
}

fn date_like(key: &[u8], v: &[u8]) -> bool {
    iso_shape(v)
        || (v.len() >= 10 && v[4] == b'-' && v[7] == b'-' && v[..4].iter().all(u8::is_ascii_digit))
        || (v.len() >= 4 && v[1] == b':' && v[0].is_ascii_digit())
        || (v.len() >= 5 && v[2] == b':' && v[0].is_ascii_digit() && v[1].is_ascii_digit())
        || (date_key(key) && has_year(v))
}

/// `FUN_0008d284`: optional, non-enum strings the conversation never
/// mentions are dropped, and so are optional numbers it never states
/// (unless the request has a quantity word).
pub fn drop_ungrounded(args: &mut Args, hay: &[u8], request: &[u8], tools: &[Tool], name: &[u8]) {
    let qty = super::text::has_quantity_word(request);
    args.retain(|(key, v)| {
        let info = schema::info(tools, name, key);
        if info.required || info.has_enum {
            return true;
        }
        match v {
            Json::Num(t) => qty || number_grounded(hay, t),
            Json::Str(s) => {
                if id_key(&lower(key)) || date_like(key, s) {
                    return true;
                }
                let sq = squash(s);
                sq.len() < 3 || contains(hay, &sq)
            }
            _ => true,
        }
    });
}

const NAME_WORDS: [&str; 15] = [
    "device",
    "appliance",
    "name",
    "item",
    "target",
    "entity",
    "object",
    "thing",
    "product",
    "app",
    "contact",
    "title",
    "recipient",
    "person",
    "who",
];

/// `FUN_000a8028`: a slot that names a thing or person.
fn nameish(kl: &[u8], desc: &[u8]) -> bool {
    if split_name_any(kl, |w| is_in(w, &NAME_WORDS)) {
        return true;
    }
    contains(&lower(desc), b"name") && !split_name_any(kl, |w| is_in(w, &PLACE_SLOT))
}

const LEFTOVER_FILLER: [&str; 49] = [
    "can",
    "could",
    "would",
    "will",
    "you",
    "please",
    "kindly",
    "hey",
    "hi",
    "ok",
    "okay",
    "i",
    "i'd",
    "want",
    "need",
    "would",
    "like",
    "to",
    "me",
    "my",
    "our",
    "us",
    "it",
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "some",
    "now",
    "right",
    "just",
    "go",
    "ahead",
    "and",
    "for",
    "of",
    "with",
    "so",
    "then",
    "also",
    "again",
    "quickly",
    "asap",
    "immediately",
    "thanks",
    "thank",
];
const LEFTOVER_CONTROL: [&str; 24] = [
    "turn",
    "switch",
    "toggle",
    "power",
    "put",
    "set",
    "make",
    "flip",
    "start",
    "stop",
    "open",
    "close",
    "activate",
    "deactivate",
    "enable",
    "disable",
    "shut",
    "run",
    "kill",
    "fire",
    "up",
    "down",
    "off",
    "on",
];

/// `FUN_000a85d0`: the request's one leftover phrase once filler, control
/// words, the tool's enum options and its name words are taken out; empty
/// unless there is exactly one run of fewer than five words.
pub fn leftover(request: &[u8], tool: Option<&Tool>) -> Vec<u8> {
    let ident = tool.map(|t| name_words(&t.name)).unwrap_or_default();
    let mut runs: Vec<Vec<&[u8]>> = vec![vec![]];
    for tok in word_tokens(request) {
        let t = lower(tok);
        let breaker = is_in(&t, &LEFTOVER_FILLER)
            || is_in(&t, &LEFTOVER_CONTROL)
            || tool.is_some_and(|tl| tl.params.iter().any(|p| p.info.options.iter().any(|o| lower(o) == t)))
            || ident.contains(&t);
        if breaker {
            if runs.last().is_some_and(|r| !r.is_empty()) {
                runs.push(vec![]);
            }
        } else if let Some(r) = runs.last_mut() {
            r.push(tok);
        }
    }
    let full: Vec<&Vec<&[u8]>> = runs.iter().filter(|r| !r.is_empty()).collect();
    match full[..] {
        [r] if r.len() < 5 => r.join(&b' '),
        _ => vec![],
    }
}

/// `FUN_0008e128`: a required name-like string the conversation never
/// mentions becomes the request's leftover phrase.
pub fn reground_name(args: &mut Args, hay: &[u8], request: &[u8], tools: &[Tool], name: &[u8]) {
    let tool = find_tool(tools, name);
    for (key, v) in args.iter_mut() {
        let Json::Str(s) = v else { continue };
        let info = schema::info(tools, name, key);
        let kl = lower(key);
        if !info.required || info.has_enum || info.ty != b"string" || id_key(&kl) || !nameish(&kl, &info.description) {
            continue;
        }
        let sv = squash(s);
        if sv.len() < 2 || contains(hay, &sv) {
            continue;
        }
        let cand = leftover(request, tool);
        if !cand.is_empty() && contains(hay, &squash(&cand)) {
            *s = cand;
        }
    }
}

/// `FUN_0008ec38`: defaults stand in for a string that is just a tool-name
/// word or the bare leftover phrase, and for a required number the
/// conversation gives no quantity for; an optional number equal to its
/// default is dropped when its digits appear nowhere.
pub fn apply_defaults(args: &mut Args, hay: &[u8], request: &[u8], tools: &[Tool], name: &[u8]) {
    let tool = find_tool(tools, name);
    let cand = leftover(request, tool);
    let ident = name_words(name);
    args.retain_mut(|(key, v)| {
        let info = schema::info(tools, name, key);
        if !info.has_default {
            return true;
        }
        let d = &info.default;
        match v {
            Json::Str(s) => {
                if !info.has_enum && ident.contains(&lower(s)) && squash(s) != squash(d) {
                    *s = d.clone();
                    return true;
                }
                if info.required && !info.has_enum && !cand.is_empty() && squash(s) == squash(&cand) && squash(&cand) != squash(d) {
                    *s = d.clone();
                }
                true
            }
            Json::Num(t) => {
                if info.required {
                    if !info.has_enum && t != d && !has_quantity(hay, request) {
                        *t = d.clone();
                    }
                    return true;
                }
                if t == d {
                    let lead: Vec<u8> = t.iter().copied().take_while(u8::is_ascii_digit).collect();
                    return lead.is_empty() || contains(hay, &lead);
                }
                true
            }
            _ => true,
        }
    });
}

/// `FUN_0008fe64`: an enum string snaps to the one option (three or more
/// letters, not all digits) the request names as a whole word.
pub fn snap_enum(args: &mut Args, request: &[u8], tools: &[Tool], name: &[u8]) {
    let r = normq(request);
    for (key, v) in args.iter_mut() {
        let Json::Str(s) = v else { continue };
        let info = schema::info(tools, name, key);
        if info.options.len() < 2 {
            continue;
        }
        let vl = lower(s);
        let mut hits = 0;
        let mut found: Option<&Vec<u8>> = None;
        for o in &info.options {
            let ol = lower(o);
            if ol.len() < 3 || ol.iter().all(u8::is_ascii_digit) {
                continue;
            }
            if contains(&r, &pad(&ol)) {
                hits += 1;
                found = Some(o);
            }
        }
        if hits != 1 {
            continue;
        }
        let m = found.cloned().unwrap_or_default();
        if lower(&m) == vl || contains(&r, &pad(&vl)) {
            continue;
        }
        *s = m;
    }
}

const ORIGIN_NAMES: [&str; 21] = [
    "pickup",
    "pickup_location",
    "pickup_address",
    "pickup_point",
    "pickup_city",
    "origin",
    "origin_location",
    "origin_city",
    "origin_address",
    "from_location",
    "from_address",
    "from_city",
    "start_location",
    "start_address",
    "start_point",
    "departure",
    "departure_location",
    "departure_city",
    "departure_airport",
    "starting_point",
    "start_place",
];
const DEST_NAMES: [&str; 23] = [
    "destination",
    "destination_location",
    "destination_city",
    "destination_address",
    "dropoff",
    "dropoff_location",
    "dropoff_address",
    "dropoff_point",
    "dropoff_city",
    "drop_off",
    "drop_off_location",
    "to_location",
    "to_address",
    "to_city",
    "end_location",
    "end_address",
    "end_point",
    "arrival",
    "arrival_location",
    "arrival_city",
    "arrival_airport",
    "ending_point",
    "end_place",
];
const PHRASE_STOP: [&str; 25] = [
    "to", "from", "for", "with", "at", "by", "and", "then", "please", "asap", "now", "today", "tomorrow", "tonight", "around", "before",
    "after", "via", "using", "in", "on", "picking", "pick", "drop", "dropping",
];
const NOT_A_PLACE: [&str; 53] = [
    "get", "go", "be", "see", "catch", "meet", "make", "have", "do", "take", "visit", "attend", "buy", "watch", "eat", "grab", "book",
    "order", "find", "check", "call", "send", "reach", "arrive", "leave", "come", "bring", "stay", "wait", "help", "know", "use", "try",
    "start", "stop", "play", "set", "turn", "put", "give", "tell", "let", "keep", "pay", "ride", "drive", "walk", "fly", "travel", "head",
    "return", "pick", "drop",
];
const EDGE: &[u8] = b",.?!;:";

/// `FUN_000a9f68`: up to five words from `k`, a leading article skipped,
/// stopping at a stop word or after a word with edge punctuation.
fn place_phrase(words: &[&[u8]], k: usize) -> Vec<u8> {
    let mut out: Vec<&[u8]> = vec![];
    for w in words.iter().skip(k) {
        if out.len() > 4 {
            break;
        }
        let lw = lower(w);
        let t = trim(&lw, EDGE);
        if t.is_empty() || is_in(t, &PHRASE_STOP) {
            break;
        }
        let clipped = t.len() != lw.len();
        if out.is_empty() && matches!(t, b"a" | b"an" | b"the") {
            if clipped {
                break;
            }
            continue;
        }
        out.push(trim(w, EDGE));
        if clipped {
            break;
        }
    }
    out.join(&b' ')
}

/// `FUN_000aa8e4`: a phrase that starts with a verb or a clock time is no
/// place.
fn valid_place(p: &[u8]) -> bool {
    let Some(f) = ws_tokens(p).first().map(|f| lower(f)) else { return false };
    if is_in(&f, &NOT_A_PLACE) {
        return false;
    }
    !(f.iter().any(u8::is_ascii_digit) && (f.contains(&b':') || contains(&f, b"am") || contains(&f, b"pm")))
}

/// `FUN_000ab0e0`: write a phrase into a slot unless the model's value is
/// grounded; add the slot when missing.
fn apply_place(args: &mut Args, field: &[u8], phrase: &[u8], hay: &[u8]) {
    if field.is_empty() || phrase.is_empty() {
        return;
    }
    let sp = squash(phrase);
    if sp.len() < 3 || sp.iter().all(u8::is_ascii_digit) {
        return;
    }
    match args.iter_mut().find(|(k, _)| k == field) {
        Some((_, Json::Str(s))) => {
            let sv = squash(s);
            if sv.is_empty() || !contains(hay, &sv) {
                *s = phrase.to_vec();
            }
        }
        Some(_) => {}
        None => args.push((field.to_vec(), Json::Str(phrase.to_vec()))),
    }
}

/// `FUN_000911d4`: origin and destination slots from `from X`, `to Y`,
/// `pick (me) up at/from X` and `drop (me) off at Y`.
pub fn route_from_request(args: &mut Args, hay: &[u8], request: &[u8], tools: &[Tool], name: &[u8]) {
    let Some(tool) = find_tool(tools, name) else { return };
    let slot = |names: &[&str]| {
        tool.params.iter().find(|p| p.info.ty == b"string" && !p.info.has_enum && is_in(&lower(&p.key), names)).map(|p| p.key.clone())
    };
    let (of, df) = (slot(&ORIGIN_NAMES).unwrap_or_default(), slot(&DEST_NAMES).unwrap_or_default());
    if of.is_empty() && df.is_empty() {
        return;
    }
    let words = ws_tokens(request);
    let n = words.len();
    let (mut origin, mut dest) = (vec![], vec![]);
    for i in 0..n.saturating_sub(1) {
        let w = lower(trim(words[i], EDGE));
        let low = |j: usize| lower(words[j]);
        if origin.is_empty() && w == b"from" {
            let c = place_phrase(&words, i + 1);
            if valid_place(&c) {
                origin = c;
            }
        }
        if origin.is_empty() && matches!(w.as_slice(), b"pick" | b"pickup" | b"picking") {
            let mut j = i + 1;
            while j < n && matches!(low(j).as_slice(), b"up" | b"me" | b"us") {
                j += 1;
            }
            if j < n && matches!(low(j).as_slice(), b"at" | b"from") {
                let c = place_phrase(&words, j + 1);
                if valid_place(&c) {
                    origin = c;
                }
            }
        }
        if w == b"to" {
            let c = place_phrase(&words, i + 1);
            if valid_place(&c) {
                dest = c;
            }
        }
        if matches!(w.as_slice(), b"drop" | b"dropoff" | b"dropping") {
            let mut j = i + 1;
            while j < n && matches!(low(j).as_slice(), b"off" | b"me" | b"us") {
                j += 1;
            }
            if j < n && low(j) == b"at" {
                let c = place_phrase(&words, j + 1);
                if valid_place(&c) {
                    dest = c;
                }
            }
        }
    }
    if !origin.is_empty() && !dest.is_empty() && squash(&origin) == squash(&dest) {
        return;
    }
    apply_place(args, &of, &origin, hay);
    apply_place(args, &df, &dest, hay);
}

/// `FUN_000951a8`: "by N" fills the tool's only numeric parameter when the
/// call left it out.
pub fn amount_from_by(args: &mut Args, request: &[u8], tools: &[Tool], name: &[u8]) {
    let q = lower(request);
    let toks = ws_tokens(&q);
    if toks.len() < 2 {
        return;
    }
    let found = (0..toks.len() - 1).find_map(|i| {
        let w = trim(toks[i + 1], b",.?!;:%");
        (toks[i] == b"by" && !w.is_empty() && w.iter().all(u8::is_ascii_digit)).then(|| w.to_vec())
    });
    let Some(n) = found else { return };
    let Some(tool) = find_tool(tools, name) else { return };
    let numeric: Vec<&Vec<u8>> =
        tool.params.iter().filter(|p| matches!(p.info.ty.as_slice(), b"integer" | b"number")).map(|p| &p.key).collect();
    let [p] = numeric[..] else { return };
    if args.iter().any(|(k, _)| k == p) || schema::info(tools, name, p).has_enum {
        return;
    }
    args.push((p.clone(), Json::Num(n)));
}

fn float_type(t: &[u8]) -> bool {
    matches!(t, b"float" | b"number" | b"double")
}

fn add_point_zero(v: &mut Json) {
    if let Json::Num(t) = v
        && !t.iter().any(|c| matches!(c, b'.' | b'e' | b'E'))
    {
        t.extend_from_slice(b".0");
    }
}

/// `FUN_00095d08`: integers under a number/float/double type (or in an
/// array of them) get `.0`.
pub fn floats_for_numbers(args: &mut Args, tools: &[Tool], name: &[u8]) {
    for (key, v) in args.iter_mut() {
        let info = schema::info(tools, name, key);
        if float_type(&info.ty) {
            add_point_zero(v);
        } else if info.ty == b"array"
            && float_type(&info.items_ty)
            && let Json::Arr(items) = v
        {
            items.iter_mut().for_each(add_point_zero);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::schema::parse_tools;

    #[test]
    fn number_grounding() {
        assert!(number_grounded(b"logit23please", b"23.0") && !number_grounded(b"logit23.5please", b"23"));
        assert!(number_grounded(b"set1,500steps", b"1500"));
        assert!(!number_grounded(b"setto140", b"14"));
        assert!(number_grounded(b"setfortwentyminutes", b"20.0"));
        assert!(number_grounded(b"someone", b"1"));
    }

    #[test]
    fn leftover_phrase() {
        let tools =
            parse_tools(br#"[{"name":"set_lights","parameters":{"properties":{"mode":{"type":"string","enum":["warm","cool"]}}}}]"#);
        assert_eq!(leftover(b"turn on the kitchen lamp", tools.first()), b"kitchen lamp");
        assert_eq!(leftover(b"set the lights to warm", tools.first()), b"");
        assert_eq!(leftover(b"kitchen lamp and hall light", tools.first()), b"");
    }

    #[test]
    fn route_phrases() {
        let tools = parse_tools(
            br#"[{"name":"ride","parameters":{"properties":{"pickup_location":{"type":"string"},"destination":{"type":"string"}}}}]"#,
        );
        let mut args: Args = vec![(b"destination".to_vec(), Json::Str(b"Bostonia".to_vec()))];
        let req = b"pick me up at 5pm from the Hilton and take me to Union Station, then home";
        route_from_request(&mut args, &squash(req), req, &tools, b"ride");
        assert_eq!(args[0].1, Json::Str(b"Union Station".to_vec()));
        assert_eq!(args[1], (b"pickup_location".to_vec(), Json::Str(b"Hilton".to_vec())));
    }

    #[test]
    fn by_amount_and_floats() {
        let tools = parse_tools(
            br#"[{"name":"vol","parameters":{"properties":{"step":{"type":"number"},"dir":{"type":"string"},"xs":{"type":"array","items":{"type":"number"}}}}}]"#,
        );
        let mut args: Args =
            vec![(b"dir".to_vec(), Json::Str(b"up".to_vec())), (b"xs".to_vec(), Json::Arr(vec![Json::Num(b"2".to_vec())]))];
        amount_from_by(&mut args, b"raise the volume by 10%", &tools, b"vol");
        floats_for_numbers(&mut args, &tools, b"vol");
        assert_eq!(args[1].1, Json::Arr(vec![Json::Num(b"2.0".to_vec())]));
        assert_eq!(args[2], (b"step".to_vec(), Json::Num(b"10.0".to_vec())));
    }

    #[test]
    fn enum_snap() {
        let tools = parse_tools(br#"[{"name":"m","parameters":{"properties":{"cycle":{"type":"string","enum":["eco","heavy","on"]}}}}]"#);
        let mut args: Args = vec![(b"cycle".to_vec(), Json::Str(b"eco".to_vec()))];
        snap_enum(&mut args, b"run it on Heavy", &tools, b"m");
        assert_eq!(args[0].1, Json::Str(b"heavy".to_vec()));
    }
}
