//! Byte-string helpers shared by the rules. The library works on raw bytes
//! and only ever lowercases ASCII, so these do the same.

/// ASCII-lowercased copy.
pub fn lower(s: &[u8]) -> Vec<u8> {
    s.to_ascii_lowercase()
}

/// First index of `needle` in `hay` at or after `from`. An empty needle is
/// found at `from` (while `from <= len`).
pub fn find_from(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    if needle.is_empty() {
        return Some(from);
    }
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    find_from(hay, needle, 0)
}

pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
    find(hay, needle).is_some()
}

/// Every non-overlapping occurrence, left to right.
pub fn find_all(hay: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = vec![];
    let mut from = 0;
    while let Some(p) = find_from(hay, needle, from) {
        out.push(p);
        from = p + needle.len().max(1);
    }
    out
}

/// `FUN_000789d4`: strip bytes in `set` from both ends (NUL always counts,
/// as `strchr` matches the terminator).
pub fn trim<'a>(s: &'a [u8], set: &[u8]) -> &'a [u8] {
    let hit = |c: &u8| *c == 0 || set.contains(c);
    let start = s.iter().position(|c| !hit(c)).unwrap_or(s.len());
    let end = s.iter().rposition(|c| !hit(c)).map_or(start, |e| e + 1);
    &s[start..end.max(start)]
}

/// ASCII-lowercased with every space, `-` and `_` removed.
pub fn squash(s: &[u8]) -> Vec<u8> {
    s.iter().filter(|c| !matches!(c, b' ' | b'-' | b'_')).map(u8::to_ascii_lowercase).collect()
}

/// `" " + map(s) + " "`: `a-z`, `0-9`, space and `'` kept, `A-Z` lowercased,
/// every other byte a space.
pub fn normq(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 2);
    out.push(b' ');
    for &c in s {
        out.push(match c {
            b'a'..=b'z' | b'0'..=b'9' | b' ' | b'\'' => c,
            b'A'..=b'Z' => c | 0x20,
            _ => b' ',
        });
    }
    out.push(b' ');
    out
}

/// `" " + s + " "`.
pub fn pad(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() + 2);
    out.push(b' ');
    out.extend_from_slice(s);
    out.push(b' ');
    out
}

/// Whether `" " + word + " "` occurs in `hay`.
pub fn has_word(hay: &[u8], word: &[u8]) -> bool {
    contains(hay, &pad(word))
}

/// `FUN_0009f0b8`: lowercase alphanumeric words of an identifier. An
/// uppercase letter after a lowercase one starts a new word (camelCase);
/// every other byte separates. Words under three bytes and `get`, `set`,
/// `the`, `and`, `for`, `with` are dropped.
pub fn name_words(s: &[u8]) -> Vec<Vec<u8>> {
    let mut out = vec![];
    let mut cur: Vec<u8> = vec![];
    let flush = |cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>| {
        let keep = cur.len() >= 3 && !matches!(cur.as_slice(), b"get" | b"set" | b"the" | b"and" | b"for" | b"with");
        if keep {
            out.push(cur.clone());
        }
        cur.clear();
    };
    for &c in s {
        match c {
            b'a'..=b'z' | b'0'..=b'9' => cur.push(c),
            b'A'..=b'Z' => {
                if cur.last().is_some_and(u8::is_ascii_lowercase) {
                    flush(&mut cur, &mut out);
                }
                cur.push(c | 0x20);
            }
            _ => flush(&mut cur, &mut out),
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// `FUN_000a1c90`: split `name` on space, `-` and `_`; whether any
/// non-empty piece satisfies `pred`.
pub fn split_name_any(name: &[u8], pred: impl Fn(&[u8]) -> bool) -> bool {
    name.split(|c| matches!(c, b' ' | b'-' | b'_')).any(|w| !w.is_empty() && pred(w))
}

/// `FUN_0009cff8`: split on space, tab and newline, dropping empty pieces.
pub fn ws_tokens(s: &[u8]) -> Vec<&[u8]> {
    s.split(|c| matches!(c, b' ' | b'\t' | b'\n')).filter(|w| !w.is_empty()).collect()
}

/// A token of `FUN_0009a078` with its byte span.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    pub text: Vec<u8>,
    pub start: usize,
    pub end: usize,
}

/// Length of a separator sequence at `i` (`C2 A0..BF`, `E2 80 xx`,
/// `E2 81 xx`), or 0.
fn separator_len(s: &[u8], i: usize) -> usize {
    match s[i] {
        0xc2 if s.get(i + 1).is_some_and(|b| (0xa0..=0xbf).contains(b)) => 2,
        0xe2 if matches!(s.get(i + 1), Some(0x80 | 0x81)) && i + 2 < s.len() => 3,
        _ => 0,
    }
}

fn is_word_at(s: &[u8], i: usize) -> bool {
    i < s.len() && (s[i].is_ascii_alphanumeric() || (s[i] >= 0x80 && separator_len(s, i) == 0))
}

/// `FUN_0009a078`: runs of ASCII alphanumerics and non-ASCII bytes (minus
/// the Latin-1 punctuation and general-punctuation blocks). An apostrophe
/// (`'` or `’`) followed by a word byte stays inside the token.
pub fn tokens(s: &[u8]) -> Vec<Token> {
    let mut out = vec![];
    let mut i = 0;
    while i < s.len() {
        if !is_word_at(s, i) {
            let n = if s[i] >= 0x80 { separator_len(s, i).max(1) } else { 1 };
            i += n;
            continue;
        }
        let start = i;
        loop {
            if is_word_at(s, i) || (i < s.len() && s[i] == b'\'' && is_word_at(s, i + 1)) {
                i += 1;
            } else if s[i..].starts_with(&[0xe2, 0x80, 0x99]) && is_word_at(s, i + 3) {
                i += 3;
            } else {
                break;
            }
        }
        out.push(Token { text: s[start..i].to_vec(), start, end: i });
    }
    out
}

/// `FUN_000a2b94`: maximal runs of ASCII letters, digits, `'` and bytes
/// at or above 0x80.
pub fn word_tokens(s: &[u8]) -> Vec<&[u8]> {
    s.split(|c| !(c.is_ascii_alphanumeric() || *c == b'\'' || *c >= 0x80)).filter(|w| !w.is_empty()).collect()
}

/// `FUN_0009faa4`: whether some phrase occurs as `" " + p + " "` in `text`
/// without a negation in the 12 bytes before it.
pub fn phrase_hit(text: &[u8], phrases: &[&str]) -> bool {
    phrases.iter().any(|p| {
        let needle = pad(p.as_bytes());
        find_all(text, &needle).into_iter().any(|pos| {
            let win = &text[pos.saturating_sub(12)..pos];
            !["don't", "do not", "dont", "never", "not"].iter().any(|n| contains(win, n.as_bytes()))
        })
    })
}

/// The 92 words that state a quantity or a time (table `0xbcbf8`).
pub const QUANTITY_WORDS: [&str; 92] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "half",
    "quarter",
    "third",
    "double",
    "triple",
    "dozen",
    "couple",
    "few",
    "several",
    "first",
    "second",
    "last",
    "next",
    "previous",
    "past",
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "now",
    "noon",
    "midnight",
    "morning",
    "evening",
    "afternoon",
    "week",
    "weekly",
    "weekend",
    "month",
    "monthly",
    "year",
    "yearly",
    "annual",
    "annually",
    "day",
    "daily",
    "hour",
    "hourly",
    "minute",
    "decade",
    "century",
    "max",
    "maximum",
    "min",
    "minimum",
    "full",
    "fully",
    "highest",
    "lowest",
    "all",
    "everything",
    "halfway",
    "middle",
    "mute",
    "muted",
    "silent",
    "silence",
    "none",
    "nothing",
    "empty",
];

/// `FUN_000a6bb0`: the request has a quantity word.
pub fn has_quantity_word(request: &[u8]) -> bool {
    word_tokens(request).iter().any(|t| {
        let t = lower(t);
        QUANTITY_WORDS.iter().any(|q| q.as_bytes() == t.as_slice())
    })
}

/// `FUN_000a2580`: a digit anywhere in the squashed conversation, or a
/// quantity word in the request.
pub fn has_quantity(squashed: &[u8], request: &[u8]) -> bool {
    squashed.iter().any(u8::is_ascii_digit) || has_quantity_word(request)
}

/// Words that are not places: months, weekdays, units, currencies,
/// pronouns and `am`/`pm` (table `0xbcae8`).
pub const NON_PLACE: [&str; 34] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "celsius",
    "fahrenheit",
    "kelvin",
    "usd",
    "eur",
    "gbp",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "it",
    "am",
    "pm",
];

/// Exact membership of a byte string in a word list.
pub fn is_in(w: &[u8], list: &[&str]) -> bool {
    list.iter().any(|x| x.as_bytes() == w)
}

/// Place words for slot names (`FUN_000a2214`).
pub const PLACE_SLOT: [&str; 9] = ["room", "place", "venue", "area", "zone", "city", "site", "location", "region"];

/// The key contains `date`, `time`, `when`, `start`, `end`, `due` or
/// `schedule` (`FUN_000a752c`).
pub fn date_key(key: &[u8]) -> bool {
    let k = lower(key);
    ["date", "time", "when", "start", "end", "due", "schedule"].iter().any(|w| contains(&k, w.as_bytes()))
}

/// Key is `id` or ends in `_id`.
pub fn id_key(kl: &[u8]) -> bool {
    kl == b"id" || kl.ends_with(b"_id")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: Vec<Vec<u8>>) -> Vec<String> {
        v.into_iter().map(|w| String::from_utf8(w).unwrap()).collect()
    }

    #[test]
    fn name_words_drop_short_and_stop_words() {
        assert_eq!(strs(name_words(b"set_timer")), ["timer"]);
        assert_eq!(strs(name_words(b"start_robot_vacuum")), ["start", "robot", "vacuum"]);
        assert_eq!(strs(name_words(b"playMusicNow")), ["play", "music", "now"]);
        assert_eq!(strs(name_words(b"turn_on_lights")), ["turn", "lights"]);
    }

    #[test]
    fn tokens_keep_apostrophes_and_spans() {
        let t = tokens("Joe’s cafe, don't '90s".as_bytes());
        let words: Vec<_> = t.iter().map(|t| String::from_utf8(t.text.clone()).unwrap()).collect();
        assert_eq!(words, ["Joe’s", "cafe", "don't", "90s"]);
        assert_eq!((t[1].start, t[1].end), (8, 12));
    }

    #[test]
    fn phrase_hit_skips_negated() {
        assert!(!phrase_hit(&normq(b"don't turn on the lights"), &["turn on"]));
        assert!(phrase_hit(&normq(b"please turn on the lights"), &["turn on"]));
        assert!(!phrase_hit(&normq(b"I cannot open it"), &["open"]));
    }

    #[test]
    fn trim_and_squash() {
        assert_eq!(trim(b"  .a b. ", b" ."), b"a b");
        assert_eq!(trim(b"...", b"."), b"");
        assert_eq!(squash(b"Living-Room_A b"), b"livingroomab");
    }
}
