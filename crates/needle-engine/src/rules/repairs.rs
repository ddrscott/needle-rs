//! Argument repairs that run before and inside the per-argument rebuild:
//! the first/last name split (`FUN_0007943c`), merging a place query split
//! over two calls (`FUN_00078b38`), file extensions to MIME types
//! (`FUN_000a2d9c`), phone completion (`FUN_00085c5c`), quoted text
//! restore (`FUN_00086bd0`), place query recovery (`FUN_00088a0c`),
//! request casing (`FUN_0008b26c`), title capitalization (`FUN_0008c7b0`)
//! and celsius/fahrenheit conversion (`FUN_00084c80`).

use super::json::{Json, call_args, call_name};
use super::schema::{self, Tool};
use super::text::{NON_PLACE, Token, contains, find, find_all, is_in, lower, tokens, trim, ws_tokens};

// First and last names.

const FIRST_KEYS: [&str; 6] = ["firstname", "givenname", "first_name", "given_name", "first", "forename"];
const LAST_KEYS: [&str; 6] = ["surname", "last_name", "family_name", "familyname", "last", "lastname"];
const HONORIFICS: [&str; 15] = ["dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "prof", "prof.", "sir", "madam", "mx", "mx."];
/// Words before a surname that are not a first name (`FUN_0009e148`).
const NOT_FIRST: [&str; 12] = ["save", "named", "contact", "create", "please", "call", "name", "my", "add", "for", "the", "new"];
/// Words after a first name that are not a surname (`FUN_0009e7dc`).
const NOT_LAST: [&str; 12] = ["i", "as", "to", "his", "her", "and", "the", "with", "their", "phone", "email", "please"];

/// Words of a name value with leading honorifics dropped.
fn name_parts(v: &[u8]) -> Vec<Vec<u8>> {
    let words = ws_tokens(trim(v, b" \t"));
    let skip = words.iter().take_while(|w| is_in(&lower(w), &HONORIFICS)).count();
    // The library rejoins with single spaces and splits again.
    words[skip..].iter().map(|w| w.to_vec()).collect()
}

fn capitalish(t: &[u8]) -> bool {
    t.first().is_some_and(|c| *c >= 0x80 || c.is_ascii_uppercase())
}

fn strip_possessive(t: &[u8]) -> Vec<u8> {
    if t.len() > 2 && t.ends_with(b"'s") {
        return t[..t.len() - 2].to_vec();
    }
    if t.len() >= 5 && t.ends_with("’s".as_bytes()) {
        return t[..t.len() - 4].to_vec();
    }
    t.to_vec()
}

/// The token after `word` in the request that can be its surname.
fn following_surname(request: &[u8], toks: &[Token], word: &[u8]) -> Option<Vec<u8>> {
    let wl = lower(word);
    (0..toks.len().saturating_sub(1)).find_map(|i| {
        let q = &toks[i + 1];
        (lower(&toks[i].text) == wl
            && capitalish(&q.text)
            && !is_in(&lower(&q.text), &NOT_LAST)
            && trim(&request[toks[i].end..q.start], b" \t\n").is_empty())
        .then(|| strip_possessive(&q.text))
    })
}

/// `FUN_0007943c`: split a full name that sits in one of a first/last
/// pair, fill a missing half from the request, and drop honorifics.
pub fn split_name(args: &mut [(Vec<u8>, Json)], request: &[u8]) {
    let fi = args.iter().position(|(k, _)| is_in(&lower(k), &FIRST_KEYS));
    let li = args.iter().position(|(k, _)| is_in(&lower(k), &LAST_KEYS));
    if fi.is_none() && li.is_none() {
        return;
    }
    let value = |i: Option<usize>| -> Option<Vec<u8>> {
        match i {
            None => Some(vec![]),
            Some(i) => args[i].1.as_str().map(<[u8]>::to_vec),
        }
    };
    let (Some(f0), Some(l0)) = (value(fi), value(li)) else { return };
    let mut fw = name_parts(&f0);
    let mut lw = name_parts(&l0);
    let prefix = |a: &[Vec<u8>], b: &[Vec<u8>]| a.len() <= b.len() && b[..a.len()] == *a;
    let suffix = |a: &[Vec<u8>], b: &[Vec<u8>]| a.len() <= b.len() && b[b.len() - a.len()..] == *a;
    if lw.len() >= 2 && (fw.is_empty() || prefix(&fw, &lw)) {
        fw = vec![lw.remove(0)];
    } else if fw.len() >= 2 && (lw.is_empty() || suffix(&lw, &fw)) {
        lw = fw.split_off(1);
    } else if lw.len() == 1 && fw.is_empty() {
        let toks = tokens(request);
        if toks.len() >= 2 {
            let ll = lower(&lw[0]);
            let first = (1..toks.len()).find_map(|i| {
                let p = &toks[i - 1];
                let pl = lower(&p.text);
                (lower(&toks[i].text) == ll
                    && capitalish(&p.text)
                    && !is_in(&pl, &HONORIFICS)
                    && !is_in(&pl, &NOT_FIRST)
                    && trim(&request[p.end..toks[i].start], b" \t\n").is_empty())
                .then(|| p.text.clone())
            });
            if let Some(p) = first {
                fw = vec![p];
            } else if let Some(q) = following_surname(request, &toks, &lw[0]) {
                fw = vec![lw[0].clone()];
                lw = vec![q];
            }
        }
    } else if fw.len() == 1 && lw.is_empty() {
        let toks = tokens(request);
        if toks.len() >= 2
            && let Some(q) = following_surname(request, &toks, &fw[0])
        {
            lw = vec![q];
        }
    }
    let first = fw.join(&b' ');
    let last = lw.join(&b' ');
    if let Some(i) = fi
        && first != f0
    {
        args[i].1 = Json::Str(first);
    }
    if let Some(i) = li
        && last != l0
    {
        args[i].1 = Json::Str(last);
    }
}

// Places.

/// Key substrings that are never places.
const NOT_PLACE_KEY: [&str; 12] = ["mail", "phone", "url", "link", "site", "type", "kind", "mode", "key", "format", "unit", "category"];
const PLACE_KEYS: [&str; 7] = ["query", "location", "place", "venue", "where", "address", "destination"];
const PLACE_TEXT: [&str; 9] = ["address", "location", "navigate", "directions", "maps", "map", "place", "venue", "geo"];

/// `FUN_000a45ec`: a description or tool text that talks about places.
fn place_text(t: &[u8]) -> bool {
    let mut t = lower(t);
    for phrase in [&b"email address"[..], b"e-mail address"] {
        while let Some(p) = find(&t, phrase) {
            t.drain(p..p + phrase.len());
        }
    }
    tokens(&t).iter().any(|w| is_in(&w.text, &PLACE_TEXT))
}

/// `FUN_00088848`: the tool's name and description.
pub fn tool_text(tools: &[Tool], name: &[u8]) -> Vec<u8> {
    match tools.iter().find(|t| t.name == name) {
        Some(t) => [&t.name[..], b" ", &t.description].concat(),
        None => name.to_vec(),
    }
}

/// `FUN_000881f0`: is this argument a place?
pub fn is_place_param(key: &[u8], desc: &[u8], tooltext: &[u8]) -> bool {
    let k = lower(key);
    if NOT_PLACE_KEY.iter().any(|w| contains(&k, w.as_bytes())) || k.ends_with(b"id") {
        return false;
    }
    if k == b"query" {
        return place_text(desc) || place_text(tooltext);
    }
    is_in(&k, &PLACE_KEYS) || place_text(desc)
}

/// Words the value may skip over in the request.
const FILLER: &[&str] = &[
    "a", "an", "the", "in", "at", "on", "of", "de", "la", "le", "du", "el", "is", "les", "del", "des", "los", "las", "near", "which",
    "located", "that",
];
const STOP_WORDS: &[&str] = &[
    "and",
    "or",
    "also",
    "then",
    "while",
    "so",
    "but",
    "please",
    "can",
    "could",
    "right",
    "now",
    "immediately",
    "simultaneously",
    "after",
    "before",
    "because",
    "map",
    "maps",
];
const PREPS: [&str; 4] = ["at", "in", "on", "near"];
const PLACE_NOUNS: [&str; 60] = [
    "bakery",
    "cafe",
    "restaurant",
    "bookstore",
    "gallery",
    "hotel",
    "museum",
    "store",
    "shop",
    "market",
    "library",
    "venue",
    "park",
    "garden",
    "kitchen",
    "bistro",
    "pub",
    "bar",
    "cinema",
    "theater",
    "theatre",
    "stadium",
    "arena",
    "school",
    "university",
    "hospital",
    "clinic",
    "pharmacy",
    "station",
    "airport",
    "mall",
    "plaza",
    "office",
    "building",
    "tower",
    "center",
    "centre",
    "branch",
    "exhibition",
    "studio",
    "salon",
    "gym",
    "church",
    "temple",
    "mosque",
    "downtown",
    "uptown",
    "campus",
    "hall",
    "club",
    "lounge",
    "diner",
    "eatery",
    "pizzeria",
    "trattoria",
    "brasserie",
    "boutique",
    "supermarket",
    "grocery",
    "deli",
];
const SUFFIXES: &[&str] = &[
    "street",
    "st",
    "road",
    "rd",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "lane",
    "ln",
    "drive",
    "rue",
    "via",
    "plaza",
    "plac",
    "highway",
    "hwy",
    "calle",
    "strasse",
    "platz",
    "square",
    "piazza",
    "avenida",
];
const ADDR_CONN: &[&str] = &[
    "de", "del", "la", "le", "les", "des", "du", "los", "las", "el", "of", "the", "and", "n", "s", "e", "w", "ne", "nw", "se", "sw",
    "north", "south", "east", "west", "downtown", "uptown", "suite", "ste", "floor", "unit", "apt",
];

fn word(w: &[u8], list: &[&str]) -> bool {
    !w.is_empty() && is_in(w, list)
}

/// `FUN_000a5748`: a year, or a clock time like `5pm` / `10:`.
fn timeish(t: &[u8]) -> bool {
    if t.len() == 4 && t.iter().all(u8::is_ascii_digit) && (t.starts_with(b"19") || t.starts_with(b"20")) {
        return true;
    }
    if !t.first().is_some_and(u8::is_ascii_digit) {
        return false;
    }
    let skip = if t.get(1).is_some_and(u8::is_ascii_digit) { 2 } else { 1 };
    let rest = lower(&t[skip..]);
    rest.starts_with(b"am") || rest.starts_with(b"pm") || rest.starts_with(b":")
}

fn lead_ok(t: &[u8]) -> bool {
    t.first().is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c >= 0x80)
}

struct Recovery<'a> {
    req: &'a [u8],
    rt: Vec<Token>,
    rl: Vec<Vec<u8>>,
}

impl Recovery<'_> {
    fn gap(&self, a: usize, b: usize) -> &[u8] {
        &self.req[self.rt[a].end..self.rt[b].start]
    }

    fn rl(&self, i: usize) -> &[u8] {
        self.rl.get(i).map_or(b"", |w| w.as_slice())
    }

    /// `FUN_000a5198`: the place query ends after token `j`.
    fn stop(&self, j: usize) -> bool {
        if j + 1 >= self.rt.len() {
            return true;
        }
        if self.gap(j, j + 1).iter().any(|c| matches!(c, b'.' | b'!' | b'?' | b';')) {
            return true;
        }
        let n1 = self.rl(j + 1);
        if word(n1, STOP_WORDS) {
            return true;
        }
        let (w2, w3) = (self.rl(j + 2), self.rl(j + 3));
        n1 == b"on" && (matches!(w2, b"map" | b"maps") || (w2 == b"the" && matches!(w3, b"map" | b"maps")) || (w2 == b"a" && w3 == b"map"))
    }

    /// `FUN_000a6048`: a street-address run after token `p`. Returns
    /// (last token, count, has a street suffix, a capitalized word after a
    /// comma).
    fn address_run(&self, mut p: usize) -> (usize, usize, bool, bool) {
        let (mut last, mut n, mut suffix, mut comma_cap) = (p, 0, false, false);
        loop {
            let q = p + 1;
            if q >= self.rt.len() {
                break;
            }
            let gap = self.gap(p, q);
            if gap.iter().any(|c| matches!(c, b'.' | b'!' | b'?' | b';')) || word(self.rl(q), STOP_WORDS) {
                break;
            }
            let g = trim(gap, b" \t\n");
            let comma = g == b",";
            if !g.is_empty() && !comma {
                break;
            }
            let t = &self.rt[q].text;
            let w = self.rl(q);
            let first = lead_ok(t) && !is_in(w, &NON_PLACE) && (t.len() >= 2 || t[0].is_ascii_digit());
            let second = (t[0].is_ascii_digit() && !timeish(t)) || word(w, SUFFIXES) || (word(w, ADDR_CONN) && n > 0);
            if !(first || second) {
                break;
            }
            if word(w, SUFFIXES) {
                suffix = true;
            }
            if comma && lead_ok(t) {
                comma_cap = true;
            }
            n += 1;
            last = q;
            p = q;
        }
        (last, n, suffix, comma_cap)
    }
}

/// `FUN_00088a0c`: widen a place query to its full span in the request,
/// then on to a following `, City`, an `at/in/on/near <address>` run, a
/// `which is located at ...` clause, or a place noun after a closing quote.
pub fn recover_place(value: &[u8], request: &[u8]) -> Vec<u8> {
    let vt: Vec<Vec<u8>> = tokens(value).iter().map(|t| lower(&t.text)).collect();
    if vt.is_empty() {
        return value.to_vec();
    }
    let rt = tokens(request);
    let rl: Vec<Vec<u8>> = rt.iter().map(|t| lower(&t.text)).collect();
    let r = Recovery { req: request, rt, rl };
    let n = r.rt.len();
    let matched = (0..n).filter(|&i| r.rl[i] == vt[0]).find_map(|i| {
        let (mut j, mut k) = (i, 0);
        while k < vt.len() && j < n {
            if r.rl[j] == vt[k] {
                k += 1;
                j += 1;
            } else if is_in(&r.rl[j], FILLER) {
                j += 1;
            } else {
                return None;
            }
        }
        (k == vt.len()).then_some((i, j - 1))
    });
    let Some((start, mut cur)) = matched else { return value.to_vec() };
    let mut quoted = false;
    loop {
        if r.stop(cur) {
            break;
        }
        let gap = r.gap(cur, cur + 1);
        if gap.contains(&b'\'') || gap.contains(&b'"') || contains(gap, &[0xe2, 0x80, 0x9d]) || contains(gap, &[0xe2, 0x80, 0x99]) {
            quoted = true;
        }
        let g = trim(gap, b" \t\n");
        let n1 = r.rl(cur + 1).to_vec();
        if g == b"," {
            let t = &r.rt[cur + 1].text;
            if !lead_ok(t) || is_in(&n1, &NON_PLACE) || t.len() < 2 {
                break;
            }
            let mut c = cur + 1;
            while c + 1 < n
                && trim(r.gap(c, c + 1), b" \t\n").is_empty()
                && lead_ok(&r.rt[c + 1].text)
                && !is_in(r.rl(c + 1), &NON_PLACE)
                && !timeish(&r.rt[c + 1].text)
                && !r.stop(c)
            {
                c += 1;
            }
            cur = c;
        } else if is_in(&n1, &PREPS) {
            let Some(w) = r.rt.get(cur + 2).map(|t| t.text.clone()) else { break };
            if w.is_empty() || timeish(&w) || is_in(&lower(&w), &NON_PLACE) {
                break;
            }
            let (last, count, suffix, comma_cap) = r.address_run(cur + 1);
            let d = w[0].is_ascii_digit();
            if count == 0 || (count == 1 && d) {
                break;
            }
            if count == 1 || d || suffix || comma_cap {
                cur = last;
            } else {
                break;
            }
        } else if matches!(n1.as_slice(), b"is" | b"which" | b"located" | b"that") {
            let (w1, w2) = (r.rl(cur + 2), r.rl(cur + 3));
            if w1 == b"located" || is_in(w1, &PREPS) || (is_in(w2, &PREPS) && matches!(w1, b"is" | b"located")) {
                cur += 1;
            } else {
                break;
            }
        } else {
            if !quoted || !only_quotes(g) || !is_in(&n1, &PLACE_NOUNS) {
                break;
            }
            cur += 1;
        }
    }
    let cleaned = clean_place(&request[r.rt[start].start..r.rt[cur].end]);
    if cleaned.is_empty() { value.to_vec() } else { cleaned }
}

/// Every byte of `g` is a quote mark (`'`, `"`, curly quotes).
fn only_quotes(g: &[u8]) -> bool {
    let mut i = 0;
    while i < g.len() {
        if matches!(g[i], b'\'' | b'"') {
            i += 1;
        } else if g[i..].len() >= 3 && g[i] == 0xe2 && g[i + 1] == 0x80 && matches!(g[i + 2], 0x98 | 0x99 | 0x9c | 0x9d) {
            i += 3;
        } else {
            return false;
        }
    }
    true
}

fn word_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c >= 0x80
}

const CLEAN_PATTERNS: [(&str, &str); 10] = [
    (" which is located at ", " "),
    (" that is located at ", " "),
    (" is located at ", " at "),
    (" located at ", " "),
    (" which is at ", " "),
    (" that is at ", " "),
    (" which is ", " "),
    (" that is ", " "),
    (" is located ", " "),
    (" located ", " "),
];
const TRAILING: [&str; 18] =
    ["a", "an", "and", "in", "at", "on", "is", "of", "by", "to", "for", "or", "with", "the", "that", "near", "which", "located"];

/// `FUN_0009a90c`: tidy a recovered place span: quotes out, relative
/// clauses collapsed, a trailing `on the map` and dangling function words
/// dropped.
pub fn clean_place(span: &[u8]) -> Vec<u8> {
    let mut s = vec![b' '];
    let mut i = 0;
    while i < span.len() {
        let c = span[i];
        if c == b'"' {
            i += 1;
        } else if c == 0xe2 && span.get(i + 1) == Some(&0x80) && matches!(span.get(i + 2), Some(0x98 | 0x99 | 0x9c | 0x9d)) {
            i += 3;
        } else if c == b'\'' {
            if i > 0 && word_byte(span[i - 1]) && span.get(i + 1).is_some_and(|n| word_byte(*n)) {
                s.push(c);
            }
            i += 1;
        } else {
            s.push(c);
            i += 1;
        }
    }
    s.push(b' ');
    for (pat, rep) in CLEAN_PATTERNS {
        let mut from = 0;
        while let Some(p) = super::text::find_from(&lower(&s), pat.as_bytes(), from) {
            s.splice(p..p + pat.len(), rep.bytes());
            from = p + rep.len();
        }
    }
    let mut words: Vec<&[u8]> = ws_tokens(&s);
    let norm = |w: &[u8]| lower(trim(w, b",."));
    if words.len() >= 3 {
        let k = words.len();
        let (a, b, c) = (norm(words[k - 3]), norm(words[k - 2]), norm(words[k - 1]));
        if a == b"on" && matches!(b.as_slice(), b"the" | b"a") && matches!(c.as_slice(), b"map" | b"maps") {
            words.truncate(k - 3);
        }
    }
    // Case-sensitive here, unlike the map check above.
    while words.last().is_some_and(|w| is_in(trim(w, b",."), &TRAILING)) {
        words.pop();
    }
    trim(&words.join(&b' '), b" ,.;:!?").to_vec()
}

/// Words the merge may find between two halves of a place query.
const MERGE_GAP: [&str; 16] = ["in", "at", "on", "near", "of", "the", "a", "an", "and", "by", "to", "is", "located", "which", "that", "s"];

/// `FUN_00078b38`: two consecutive calls to the same tool, each with one
/// place argument under the same key, whose values sit close together in
/// the request, become one call with the joined span.
pub fn merge_split_place(calls: Vec<Json>, request: &[u8], tools: &[Tool]) -> Vec<Json> {
    let mut out: Vec<Json> = vec![];
    for cur in calls {
        if let Some(prev) = out.last_mut()
            && let Some(merged) = merged_value(prev, &cur, request, tools)
            && let Some(Json::Obj(args)) = prev.get_mut(b"arguments")
        {
            args[0].1 = Json::Str(merged);
            continue;
        }
        out.push(cur);
    }
    out
}

fn merged_value(prev: &Json, cur: &Json, request: &[u8], tools: &[Tool]) -> Option<Vec<u8>> {
    let (pn, cn) = (call_name(prev)?, call_name(cur)?);
    if pn != cn {
        return None;
    }
    let (pa, ca) = (call_args(prev)?, call_args(cur)?);
    let ([(pk, Json::Str(pv))], [(ck, Json::Str(cv))]) = (pa.as_slice(), ca.as_slice()) else { return None };
    if pk != ck {
        return None;
    }
    let info = schema::info(tools, pn, pk);
    if !is_place_param(pk, &info.description, &tool_text(tools, pn)) {
        return None;
    }
    let rl = lower(request);
    let (p1, p2) = (find(&rl, &lower(pv))?, find(&rl, &lower(cv))?);
    if p1 >= p2 || pv.is_empty() || cv.is_empty() {
        return None;
    }
    let after = p1 + pv.len();
    let between = if p2 >= after { &request[after..p2] } else { request.get(after..).unwrap_or_default() };
    if between.len() >= 25 || !tokens(between).iter().all(|t| is_in(&lower(&t.text), &MERGE_GAP)) {
        return None;
    }
    let end = (p2 + cv.len()).min(request.len());
    Some(clean_place(&request[p1..end]))
}

// File types.

const MIME: &[(&str, &str)] = &[
    ("csv", "text/csv"),
    ("txt", "text/plain"),
    ("text", "text/plain"),
    ("html", "text/html"),
    ("htm", "text/html"),
    ("md", "text/markdown"),
    ("markdown", "text/markdown"),
    ("xml", "application/xml"),
    ("json", "application/json"),
    ("pdf", "application/pdf"),
    ("zip", "application/zip"),
    ("gz", "application/gzip"),
    ("tar", "application/x-tar"),
    ("doc", "application/msword"),
    ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("xls", "application/vnd.ms-excel"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("ppt", "application/vnd.ms-powerpoint"),
    ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("png", "image/png"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("svg", "image/svg+xml"),
    ("bmp", "image/bmp"),
    ("heic", "image/heic"),
    ("mp3", "audio/mpeg"),
    ("wav", "audio/wav"),
    ("ogg", "audio/ogg"),
    ("flac", "audio/flac"),
    ("m4a", "audio/mp4"),
    ("aac", "audio/aac"),
    ("mp4", "video/mp4"),
    ("mov", "video/quicktime"),
    ("avi", "video/x-msvideo"),
    ("mkv", "video/x-matroska"),
    ("webm", "video/webm"),
    ("py", "text/x-python"),
    ("js", "application/javascript"),
    ("apk", "application/vnd.android.package-archive"),
    ("exe", "application/x-msdownload"),
    ("epub", "application/epub+zip"),
    ("rtf", "application/rtf"),
    ("ics", "text/calendar"),
    ("vcf", "text/vcard"),
];

/// `FUN_000841cc`: the key or description mentions MIME.
pub fn wants_mime(key: &[u8], desc: &[u8]) -> bool {
    contains(&lower(key), b"mime") || contains(&lower(desc), b"mime")
}

/// `FUN_000a2d9c`: a bare file extension as its MIME type.
fn mime(s: &[u8]) -> Vec<u8> {
    let e = lower(trim(s, b" \t."));
    if e.contains(&b'/') || e.is_empty() {
        return s.to_vec();
    }
    MIME.iter().find(|(ext, _)| ext.as_bytes() == e.as_slice()).map_or_else(|| s.to_vec(), |(_, m)| m.as_bytes().to_vec())
}

/// `FUN_00084bf8`: MIME rewrite of a string, or of every string in an
/// array (recursively).
pub fn to_mime(v: &mut Json) {
    match v {
        Json::Str(s) => *s = mime(s),
        Json::Arr(items) => items.iter_mut().for_each(to_mime),
        _ => {}
    }
}

// Phone numbers.

fn digits(s: &[u8]) -> Vec<u8> {
    s.iter().copied().filter(u8::is_ascii_digit).collect()
}

/// `FUN_000a41c0`: phone-number-like spans of the request.
fn phone_candidates(req: &[u8]) -> Vec<&[u8]> {
    let mut out = vec![];
    let mut p = 0;
    while p < req.len() {
        let c = req[p];
        if !(c == b'(' || c == b'+' || c.is_ascii_digit()) {
            p += 1;
            continue;
        }
        let mut j = if c == b'+' { p + 1 } else { p };
        if req.get(j) == Some(&b'(') {
            j += 1;
        }
        if !req.get(j).is_some_and(u8::is_ascii_digit) {
            p = j.max(p + 1);
            continue;
        }
        while j < req.len() && matches!(req[j], 0 | b' ' | b'(' | b')' | b'-' | b'.' | b'0'..=b'9') {
            j += 1;
        }
        let mut end = j;
        while end > p && !req[end - 1].is_ascii_digit() {
            end -= 1;
        }
        let span = &req[p..end];
        if digits(span).len() >= 6 && span.len() >= 7 {
            out.push(span);
        }
        p = end.max(p + 1);
    }
    out
}

/// `FUN_00085c5c`: complete a truncated phone number from the request, or
/// take the request's formatting of the same digits.
pub fn complete_phone(key: &[u8], value: &[u8], request: &[u8]) -> Vec<u8> {
    let t = trim(value, b" \t");
    if t.is_empty() {
        return value.to_vec();
    }
    let k = lower(key);
    if !["phone", "mobile", "tel", "number", "fax"].iter().any(|w| contains(&k, w.as_bytes())) {
        return value.to_vec();
    }
    let vd = digits(value);
    if vd.is_empty() {
        return value.to_vec();
    }
    let i = usize::from(matches!(t[0], b'+' | b'('));
    if !t.get(i).is_some_and(u8::is_ascii_digit)
        || !t[i..].iter().all(|c| matches!(c, 0 | b' ' | b'\'' | b'(' | b')' | b'-' | b'.' | b'0'..=b'9'))
    {
        return value.to_vec();
    }
    let cands = phone_candidates(request);
    let mut best = value.to_vec();
    for c in &cands {
        let cd = digits(c);
        if cd.len() > vd.len() && contains(&cd, &vd) {
            let t2 = trim(c, b" .,;:");
            if digits(t2).len() > digits(&best).len() {
                best = t2.to_vec();
            }
        }
    }
    if best != value {
        return best;
    }
    for c in &cands {
        let o = trim(c, b" .,;:");
        if digits(c) == vd && o != value {
            return o.to_vec();
        }
    }
    value.to_vec()
}

// Quoted text.

fn alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

/// `FUN_00078360`: quoted spans of the request, double quotes first.
pub fn quoted_spans(req: &[u8]) -> Vec<Vec<u8>> {
    let mut out = vec![];
    let dq = |i: usize| -> usize {
        if req[i] == b'"' {
            1
        } else if i + 2 < req.len() && req[i] == 0xe2 && req[i + 1] == 0x80 && matches!(req[i + 2], 0x9c | 0x9d) {
            3
        } else {
            0
        }
    };
    let mut i = 0;
    while i < req.len() {
        let n = dq(i);
        if n == 0 {
            i += 1;
            continue;
        }
        let start = i + n;
        if start >= req.len() {
            break;
        }
        let Some((j, m)) = (start..req.len()).find_map(|j| {
            let m = dq(j);
            (m > 0).then_some((j, m))
        }) else {
            break;
        };
        if j == start {
            break;
        }
        out.push(req[start..j].to_vec());
        i = j + m;
    }
    let sq_open = |i: usize| -> usize {
        if req[i] == b'\'' {
            1
        } else if req[i..].starts_with(&[0xe2, 0x80, 0x98]) {
            3
        } else {
            0
        }
    };
    let sq_close = |j: usize| -> usize {
        if req[j] == b'\'' {
            1
        } else if req[j..].starts_with(&[0xe2, 0x80, 0x99]) {
            3
        } else {
            0
        }
    };
    let mut i = 0;
    'outer: while i < req.len() {
        let n = sq_open(i);
        if n == 0 || (i > 0 && alpha(req[i - 1])) {
            i += 1;
            continue;
        }
        let start = i + n;
        if start >= req.len() {
            return out;
        }
        let mut j = start;
        while j < req.len() {
            let m = sq_close(j);
            if m == 0 {
                j += 1;
                continue;
            }
            if req.get(j + m).is_some_and(|c| alpha(*c)) {
                i += 1;
                continue 'outer;
            }
            if j > start {
                out.push(req[start..j].to_vec());
                i = j + m;
            } else {
                i += 1;
            }
            continue 'outer;
        }
        return out;
    }
    out
}

/// `FUN_00086bd0`: a value that matches a quoted span (or is a truncated
/// prefix of one) becomes the span verbatim.
pub fn restore_quoted(value: &[u8], spans: &[Vec<u8>]) -> Vec<u8> {
    if trim(value, b" \t").is_empty() {
        return value.to_vec();
    }
    let punct = b" \t\n.,!?:;";
    let vn = lower(trim(value, punct));
    for q in spans {
        if lower(trim(q, punct)) == vn && q.as_slice() != value {
            return trim(q, b",; ").to_vec();
        }
    }
    let mut p = lower(value);
    while p.last().is_some_and(|c| matches!(c, b' ' | b'.')) {
        p.pop();
    }
    for q in spans {
        if q.len() > value.len() + 8 && lower(q).starts_with(&p) && q.len() <= 2 * value.len() {
            return trim(q, b",; ").to_vec();
        }
    }
    value.to_vec()
}

// Casing and capitals.

/// `FUN_0008b26c`: take the request's casing of a free-text value when it
/// has more capitals.
pub fn restore_casing(value: &[u8], request: &[u8], desc: &[u8]) -> Vec<u8> {
    if value.len() < 4 {
        return value.to_vec();
    }
    let lo = value.iter().any(u8::is_ascii_lowercase);
    let up = value.iter().any(u8::is_ascii_uppercase);
    let space = value.contains(&b' ');
    let at = value.contains(&b'@');
    if !(lo || up) || !((lo && up) || space || at) {
        return value.to_vec();
    }
    if !desc.is_empty() && contains(desc, value) {
        return value.to_vec();
    }
    let Some(pos) = find(&lower(request), &lower(value)) else { return value.to_vec() };
    let span = &request[pos..pos + value.len()];
    let caps = |s: &[u8]| s.iter().filter(|c| c.is_ascii_uppercase()).count();
    if span != value && caps(span) > caps(value) { span.to_vec() } else { value.to_vec() }
}

const TITLE_KEYS: [&str; 8] = ["title", "subject", "event_name", "event_title", "extra_message", "message", "note", "label"];

/// `FUN_0008c7b0`: a title-like field starts with a capital (first byte
/// only).
pub fn capitalize(key: &[u8], mut value: Vec<u8>) -> Vec<u8> {
    if is_in(&lower(key), &TITLE_KEYS) && value.first().is_some_and(u8::is_ascii_lowercase) {
        value[0] -= 0x20;
    }
    value
}

// Temperatures.

/// `FUN_000a3568`: the request states `s` in the source unit.
fn stated_in_unit(request: &[u8], s: &[u8], celsius_source: bool) -> bool {
    let r = lower(request);
    let units: &[&str] = if celsius_source { &["c", "celsius", "centigrade"] } else { &["f", "fahrenheit"] };
    let skip_spaces = |mut q: usize| {
        while r.get(q) == Some(&b' ') {
            q += 1;
        }
        q
    };
    find_all(&r, s).into_iter().any(|pos| {
        if pos > 0 && (r[pos - 1].is_ascii_digit() || r[pos - 1] == b'.') {
            return false;
        }
        let mut q = skip_spaces(pos + s.len());
        let rest = &r[q.min(r.len())..];
        let adv = if rest.starts_with(&[0xc2, 0xb0]) {
            2
        } else if rest.starts_with(b"degrees") {
            7
        } else if rest.starts_with(b"degree") {
            6
        } else {
            0
        };
        if adv > 0 {
            q = skip_spaces(q + adv);
        }
        let u = &r[q.min(r.len())..(q + 12).min(r.len())];
        units.iter().any(|w| u.starts_with(w.as_bytes()) && !u.get(w.len()).is_some_and(u8::is_ascii_alphanumeric))
    })
}

/// `FUN_00084c80`: a temperature the request gives in the other unit is
/// converted to the parameter's unit (a `_f`/fahrenheit or `_c`/celsius
/// key or description).
pub fn convert_temperature(key: &[u8], desc: &[u8], text: &[u8], request: &[u8]) -> Vec<u8> {
    let (k, d) = (lower(key), lower(desc));
    let f = k.ends_with(b"_f") || contains(&k, b"fahrenheit") || contains(&d, b"fahrenheit");
    let c = k.ends_with(b"_c") || contains(&k, b"celsius") || contains(&d, b"celsius");
    if f == c {
        return text.to_vec();
    }
    let v: f64 = if text.is_empty() {
        0.0
    } else {
        match std::str::from_utf8(text).ok().and_then(|t| t.parse().ok()) {
            Some(v) => v,
            None => return text.to_vec(),
        }
    };
    let mut s = text.to_vec();
    if s.contains(&b'.') {
        while s.last() == Some(&b'0') {
            s.pop();
        }
        if s.last() == Some(&b'.') {
            s.pop();
        }
    }
    let is_int = !s.contains(&b'.');
    if !stated_in_unit(request, &s, f) {
        return text.to_vec();
    }
    let x = if f { v * 1.8 + 32.0 } else { v * 0.5555555555555556 + -17.77777777777778 };
    if is_int { format!("{}", x.round() as i64).into_bytes() } else { format!("{x:.1}").into_bytes() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: Vec<u8>) -> String {
        String::from_utf8(v).unwrap()
    }

    #[test]
    fn name_split_cases() {
        let mut a = vec![(b"first_name".to_vec(), Json::Str(b"Dr. Jane Smith".to_vec())), (b"last_name".to_vec(), Json::Str(vec![]))];
        split_name(&mut a, b"add Dr. Jane Smith");
        assert_eq!(a[0].1, Json::Str(b"Jane".to_vec()));
        assert_eq!(a[1].1, Json::Str(b"Smith".to_vec()));
        let mut b = vec![(b"last_name".to_vec(), Json::Str(b"Smith".to_vec()))];
        split_name(&mut b, b"save Jane Smith's number");
        assert_eq!(b.len(), 1);
        let mut c = vec![(b"first_name".to_vec(), Json::Str(b"".to_vec())), (b"last_name".to_vec(), Json::Str(b"Smith".to_vec()))];
        split_name(&mut c, b"call Jane Smith");
        assert_eq!(c[0].1, Json::Str(b"Jane".to_vec()));
        let mut d = vec![(b"first".to_vec(), Json::Str(b"Jane".to_vec())), (b"last".to_vec(), Json::Str(b"".to_vec()))];
        split_name(&mut d, b"text Jane O'Neil's mom");
        assert_eq!(d[1].1, Json::Str(b"O'Neil".to_vec()));
    }

    #[test]
    fn place_recovery() {
        let r = b"find Joe's Pizza on Main Street, Springfield and book it";
        assert_eq!(s(recover_place(b"joe's pizza", r)), "Joe's Pizza on Main Street, Springfield");
        assert_eq!(s(recover_place(b"museum modern art", b"navigate to the museum of modern art on the map")), "museum of modern art");
        assert_eq!(s(clean_place(b"cafe which is located at 5 Elm St on the map")), "cafe 5 Elm St");
    }

    #[test]
    fn phone_and_quotes_and_mime() {
        assert_eq!(s(complete_phone(b"phone", b"555-1234", b"call +1 (415) 555-1234 now")), "+1 (415) 555-1234");
        assert_eq!(s(complete_phone(b"phone", b"4155551234", b"call 415.555.1234")), "415.555.1234");
        assert_eq!(s(complete_phone(b"name", b"555-1234", b"call +1 (415) 555-1234")), "555-1234");
        let spans = quoted_spans("titled \"Packing List\" and 'don't' ‘Big Day’".as_bytes());
        assert_eq!(spans, vec![b"Packing List".to_vec(), "Big Day".as_bytes().to_vec()]);
        assert_eq!(s(restore_quoted(b"packing list.", &spans)), "Packing List");
        let mut v = Json::Arr(vec![Json::Str(b".PDF ".to_vec()), Json::Str(b"image/png".to_vec())]);
        to_mime(&mut v);
        assert_eq!(v, Json::Arr(vec![Json::Str(b"application/pdf".to_vec()), Json::Str(b"image/png".to_vec())]));
    }

    #[test]
    fn casing_capitals_temperature() {
        assert_eq!(s(restore_casing(b"packing list", b"save Packing List now", b"")), "Packing List");
        assert_eq!(s(restore_casing(b"jazz", b"play Jazz", b"")), "jazz");
        assert_eq!(s(capitalize(b"Title", b"groceries".to_vec())), "Groceries");
        assert_eq!(s(capitalize(b"title", b" x".to_vec())), " x");
        assert_eq!(s(convert_temperature(b"temp_f", b"", b"20", b"set it to 20 degrees c")), "68");
        assert_eq!(s(convert_temperature(b"temp_f", b"", b"20.5", b"set it to 20.5\xc2\xb0C")), "68.9");
        assert_eq!(s(convert_temperature(b"temp_c", b"", b"70", b"make it 70 f")), "21");
        assert_eq!(s(convert_temperature(b"temp_f", b"", b"20", b"set it to 20")), "20");
    }
}
