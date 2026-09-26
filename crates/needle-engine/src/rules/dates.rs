//! Date resolution (`FUN_000986e8`): the date part of ISO-shaped,
//! date-keyed arguments is rewritten from the dates the request mentions
//! (`next friday`, `March 4th`, `tomorrow`, an ISO date), counted from the
//! first ISO date in the stored system text.

use super::json::{Json, call_args_mut};
use super::tail::iso_shape;
use super::text::{date_key, lower, tokens};

/// Days since 1970-01-01 (Hinnant's `days_from_civil`).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// (year, month, day) of a day number.
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// `FUN_000acd3c`: a real calendar date.
pub fn valid(y: i64, m: i64, d: i64) -> bool {
    (1..=12).contains(&m) && (1..=31).contains(&d) && civil_from_days(days_from_civil(y, m, d)) == (y, m, d)
}

fn atoi(s: &[u8]) -> i64 {
    s.iter().fold(0, |a, c| a * 10 + i64::from(c - b'0'))
}

/// `FUN_000aca64`: the first standalone `YYYY-MM-DD` at or after `from`:
/// (position, y, m, d).
pub fn find_iso(s: &[u8], from: usize) -> Option<(usize, i64, i64, i64)> {
    let digit = |i: usize| s.get(i).is_some_and(u8::is_ascii_digit);
    let mut p = from;
    while p + 10 <= s.len() {
        let w = &s[p..p + 10];
        let shape = w.iter().enumerate().all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() });
        if shape && !(p > 0 && digit(p - 1)) && !digit(p + 10) {
            return Some((p, atoi(&w[..4]), atoi(&w[5..7]), atoi(&w[8..10])));
        }
        p += 1;
    }
    None
}

/// Today's day number from the stored system text, when it has a valid
/// ISO date.
pub fn today(system: &[u8]) -> Option<i64> {
    let (_, y, m, d) = find_iso(system, 0)?;
    valid(y, m, d).then(|| days_from_civil(y, m, d))
}

const MONTHS: [&str; 12] =
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];
const WEEKDAYS: [&str; 7] = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];

fn month(w: &[u8]) -> i64 {
    MONTHS.iter().position(|m| m.as_bytes() == w).map_or(0, |i| i as i64 + 1)
}

fn weekday(w: &[u8]) -> Option<i64> {
    WEEKDAYS.iter().position(|d| d.as_bytes() == w).map(|i| i as i64)
}

/// `FUN_000aef64`: `4`, `04`, `4th`, `21st` as a day of the month.
fn day_ordinal(t: &[u8]) -> i64 {
    let k = t.iter().take_while(|c| c.is_ascii_digit()).count();
    if !(1..=2).contains(&k) {
        return 0;
    }
    let suffix = lower(&t[k..]);
    if !matches!(suffix.as_slice(), b"" | b"st" | b"nd" | b"rd" | b"th") {
        return 0;
    }
    let v = atoi(&t[..k]);
    if (1..=31).contains(&v) { v } else { 0 }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Iso(i64, i64, i64),
    /// year (0 when not given), month, day
    MonthDay(i64, i64, i64),
    /// weekday, modifier (1 `next`, 2 `this`/`coming`, 0 otherwise)
    Weekday(i64, u8),
    Relative(i64),
}

impl Kind {
    fn rank(&self) -> u8 {
        match self {
            Kind::Iso(..) => 0,
            Kind::MonthDay(..) => 1,
            Kind::Weekday(..) => 2,
            Kind::Relative(_) => 3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Mention {
    pos: usize,
    tok: usize,
    kind: Kind,
}

fn four_digits(t: &[u8]) -> Option<i64> {
    (t.len() == 4 && t.iter().all(u8::is_ascii_digit)).then(|| atoi(t))
}

/// `FUN_000acfa8`: day numbers of the dates the request mentions, in order.
pub fn mentioned_days(request: &[u8], today: i64) -> Vec<i64> {
    let toks = tokens(request);
    let lw: Vec<Vec<u8>> = toks.iter().map(|t| lower(&t.text)).collect();
    let n = toks.len();
    let tok_at = |pos: usize| toks.iter().position(|t| t.start >= pos).unwrap_or(n);
    let mut ms = vec![];
    let mut from = 0;
    while let Some((p, y, m, d)) = find_iso(request, from) {
        ms.push(Mention { pos: p, tok: tok_at(p), kind: Kind::Iso(y, m, d) });
        from = p + 10;
    }
    for i in 0..n {
        let pos = toks[i].start;
        let m = month(&lw[i]);
        if m != 0 && i + 1 < n && day_ordinal(&toks[i + 1].text) != 0 {
            let year = toks.get(i + 2).and_then(|t| four_digits(&t.text)).unwrap_or(0);
            ms.push(Mention { pos, tok: i, kind: Kind::MonthDay(year, m, day_ordinal(&toks[i + 1].text)) });
            continue;
        }
        let dy = day_ordinal(&toks[i].text);
        if dy != 0 && i + 1 < n {
            let j = if lw[i + 1] == b"of" && i + 2 < n { i + 2 } else { i + 1 };
            let m = month(&lw[j]);
            if m != 0 {
                let year = toks.get(j + 1).and_then(|t| four_digits(&t.text)).unwrap_or(0);
                ms.push(Mention { pos, tok: i, kind: Kind::MonthDay(year, m, dy) });
                continue;
            }
        }
        if let Some(w) = weekday(&lw[i]) {
            let md = match i.checked_sub(1).map(|j| lw[j].as_slice()) {
                Some(b"next") => 1,
                Some(b"this" | b"coming") => 2,
                _ => 0,
            };
            ms.push(Mention { pos, tok: i, kind: Kind::Weekday(w, md) });
            continue;
        }
        let off = match lw[i].as_slice() {
            b"today" | b"tonight" => Some(0),
            b"tomorrow" => Some(if i >= 2 && lw[i - 1] == b"after" && lw[i - 2] == b"day" { 2 } else { 1 }),
            _ => None,
        };
        if let Some(o) = off {
            ms.push(Mention { pos, tok: i, kind: Kind::Relative(o) });
        }
    }
    // Stable insertion sort by position, then prune near duplicates.
    ms.sort_by_key(|m| m.pos);
    let mut kept: Vec<Mention> = vec![];
    for c in ms {
        let Some(l) = kept.last().copied() else {
            kept.push(c);
            continue;
        };
        if c.tok <= l.tok + 3 {
            if l.kind.rank() == 2 && c.kind.rank() <= 1 {
                if let Some(last) = kept.last_mut() {
                    *last = c;
                }
                continue;
            }
            if l.kind.rank() <= 1 && c.kind.rank() == 2 {
                continue;
            }
        }
        if c.pos <= l.pos && c.kind.rank() == l.kind.rank() {
            continue;
        }
        kept.push(c);
    }
    let wd = (today + 3).rem_euclid(7);
    let (year, _, _) = civil_from_days(today);
    kept.iter()
        .filter_map(|m| match m.kind {
            Kind::Iso(y, mo, d) => valid(y, mo, d).then(|| days_from_civil(y, mo, d)),
            Kind::MonthDay(y, mo, d) if y != 0 => valid(y, mo, d).then(|| days_from_civil(y, mo, d)),
            Kind::MonthDay(_, mo, d) => {
                if !valid(year, mo, d) {
                    return None;
                }
                let y = year + i64::from(days_from_civil(year, mo, d) < today);
                valid(y, mo, d).then(|| days_from_civil(y, mo, d))
            }
            Kind::Weekday(w, 1) => Some(today - wd + 7 + w),
            Kind::Weekday(w, _) => {
                let k = (w - wd).rem_euclid(7);
                Some(today + if k == 0 { 7 } else { k })
            }
            Kind::Relative(o) => Some(today + o),
        })
        .collect()
}

fn format_day(z: i64) -> Vec<u8> {
    let (y, m, d) = civil_from_days(z);
    format!("{y:04}-{m:02}-{d:02}").into_bytes()
}

/// `FUN_000986e8`: rewrite the date part of every ISO-shaped string argument
/// under a date-ish key: all of them with a single mentioned date, or
/// pairwise when the counts match.
pub fn resolve(calls: &mut [Json], request: &[u8], today: i64) {
    let mut days = mentioned_days(request, today);
    if days.is_empty() {
        return;
    }
    let mut targets: Vec<&mut Vec<u8>> = calls
        .iter_mut()
        .filter_map(call_args_mut)
        .flat_map(|args| args.iter_mut())
        .filter_map(|(k, v)| match v {
            Json::Str(s) if iso_shape(s) && date_key(k) => Some(s),
            _ => None,
        })
        .collect();
    if targets.is_empty() {
        return;
    }
    if days.len() > targets.len() && days.iter().all(|d| *d == days[0]) {
        days.truncate(1);
    }
    let assign = |t: &mut Vec<u8>, day: i64| {
        let mut v = format_day(day);
        v.extend_from_slice(&t[10..]);
        *t = v;
    };
    if days.len() == 1 {
        targets.iter_mut().for_each(|t| assign(t, days[0]));
    } else if days.len() == targets.len() {
        targets.iter_mut().zip(&days).for_each(|(t, d)| assign(t, *d));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(s: &str) -> i64 {
        let (_, y, m, d) = find_iso(s.as_bytes(), 0).unwrap();
        days_from_civil(y, m, d)
    }

    fn show(v: Vec<i64>) -> Vec<String> {
        v.into_iter().map(|z| String::from_utf8(format_day(z)).unwrap()).collect()
    }

    #[test]
    fn civil_round_trip() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(day("2026-09-25")), (2026, 9, 25));
        assert!(!valid(2026, 2, 29) && valid(2028, 2, 29) && !valid(2026, 4, 31));
    }

    #[test]
    fn spec_examples_from_a_friday() {
        let t = day("2026-09-25");
        let one = |q: &str| show(mentioned_days(q.as_bytes(), t));
        assert_eq!(one("on friday"), ["2026-10-02"]);
        assert_eq!(one("next friday"), ["2026-10-02"]);
        assert_eq!(one("next monday"), ["2026-09-28"]);
        assert_eq!(one("saturday"), ["2026-09-26"]);
        assert_eq!(one("next sunday"), ["2026-10-04"]);
        assert_eq!(one("on March 4th"), ["2027-03-04"]);
        assert_eq!(one("September 25"), ["2026-09-25"]);
        assert_eq!(one("the 4th of July"), ["2027-07-04"]);
        assert_eq!(one("the day after tomorrow"), ["2026-09-27"]);
        assert_eq!(one("Friday, March 4th"), ["2027-03-04"]);
        assert!(one("Dec 25").is_empty());
    }

    #[test]
    fn rewrite_keeps_time() {
        let mut calls =
            vec![super::super::json::parse(br#"{"name":"e","arguments":{"start_time":"2026-09-28T10:00","title":"x"}}"#).unwrap()];
        resolve(&mut calls, b"meet on friday at 10", day("2026-09-25"));
        assert_eq!(calls[0].get(b"arguments").unwrap().get(b"start_time"), Some(&Json::Str(b"2026-10-02T10:00".to_vec())));
    }
}
