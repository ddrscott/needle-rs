//! The request-verb table (`FUN_0009f220`) and the two repairs driven by
//! it: renaming a call to its opposite-polarity sibling tool
//! (`FUN_0007d8fc`) and flipping a bool or on/off enum to the verb
//! (`FUN_0007dfc8`).

use super::json::Json;
use super::schema::{self, Tool};
use super::text::{is_in, lower, name_words, normq, phrase_hit};

/// One row of the verb table.
pub struct VerbDir {
    /// Request phrases for the positive direction.
    pub a: &'static [&'static str],
    /// Request phrases for the negative direction.
    pub b: &'static [&'static str],
    /// Words of a bool parameter name the row applies to.
    pub c: &'static [&'static str],
    /// Positive values (and tool-name words).
    pub d: &'static [&'static str],
    /// Negative values (and tool-name words).
    pub e: &'static [&'static str],
}

pub const VERB_DIRS: [VerbDir; 5] = [
    VerbDir { a: &["lock"], b: &["unlock"], c: &["lock"], d: &["lock", "locked"], e: &["unlock", "unlocked"] },
    VerbDir {
        a: &["turn on", "switch on", "enable", "power on", "activate", "turn back on", "turn it on", "back on"],
        b: &["turn off", "switch off", "disable", "power off", "deactivate", "turn back off", "turn it off", "back off"],
        c: &["state", "on", "power", "enabled", "active", "status"],
        d: &["on", "enable", "enabled"],
        e: &["off", "disable", "disabled"],
    },
    VerbDir { a: &["open"], b: &["close", "shut"], c: &["open", "position", "state"], d: &["open", "opened"], e: &["close", "closed"] },
    VerbDir { a: &["mute"], b: &["unmute"], c: &["mute"], d: &["mute", "muted"], e: &["unmute", "unmuted"] },
    VerbDir {
        a: &[
            "increase", "raise", "turn up", "crank up", "pump up", "bump up", "louder", "higher", "boost", "brighter", "warmer", "hotter",
            "faster", "bigger", "up", "more",
        ],
        b: &[
            "decrease",
            "lower",
            "turn down",
            "bring down",
            "quieter",
            "softer",
            "reduce",
            "dimmer",
            "cooler",
            "colder",
            "slower",
            "smaller",
            "down",
            "quiet",
            "dim",
            "less",
            "soften",
        ],
        c: &["increase", "raise"],
        d: &[
            "increase",
            "up",
            "raise",
            "higher",
            "more",
            "plus",
            "increment",
            "max",
            "maximum",
            "high",
            "highest",
            "loud",
            "fast",
            "bright",
        ],
        e: &[
            "decrease",
            "down",
            "lower",
            "reduce",
            "less",
            "minus",
            "decrement",
            "min",
            "minimum",
            "low",
            "lowest",
            "quiet",
            "slow",
            "dim",
        ],
    },
];

/// A row that fires on `n` (a `normq` text): exactly one direction is
/// named. Returns (positive fired, target values).
pub fn fires(row: &VerbDir, n: &[u8]) -> Option<(bool, &'static [&'static str])> {
    let a = phrase_hit(n, row.a);
    let b = phrase_hit(n, row.b);
    (a != b).then_some(if a { (true, row.d) } else { (false, row.e) })
}

/// `FUN_000a1680`: some firing row targets exactly this value.
pub fn verb_value(n: &[u8], value: &[u8]) -> bool {
    let v = lower(value);
    VERB_DIRS.iter().any(|row| fires(row, n).is_some_and(|(_, t)| is_in(&v, t)))
}

/// `FUN_0007d8fc`: when the request's verb contradicts a polarity word of
/// the tool name (`lock_door` for "unlock the door"), the sibling tool
/// whose name differs only in that word.
pub fn polarity_rename(request: &[u8], tools: &[Tool], name: &[u8]) -> Option<Vec<u8>> {
    let n = normq(request);
    let toks = name_words(name);
    for row in &VERB_DIRS {
        let Some((pos, _)) = fires(row, &n) else { continue };
        let (want, wrong) = if pos { (row.d, row.e) } else { (row.e, row.d) };
        for (i, t) in toks.iter().enumerate() {
            if !is_in(t, wrong) {
                continue;
            }
            for tool in tools {
                if tool.name == name {
                    continue;
                }
                let other = name_words(&tool.name);
                if other.len() == toks.len()
                    && is_in(&other[i], want)
                    && other.iter().zip(&toks).enumerate().all(|(j, (x, y))| j == i || x == y)
                {
                    return Some(tool.name.clone());
                }
            }
        }
    }
    None
}

/// Start/stop/pause/resume options and the request words for each
/// (table `0xc0478`).
const TRANSPORT: [(&str, &[&str]); 4] = [
    ("resume", &["resume", "carry on", "continue", "keep going", "keep cleaning", "go on"]),
    ("pause", &["pause", "hold on"]),
    ("stop", &["stop", "halt", "cancel"]),
    ("start", &["start", "begin"]),
];

/// `FUN_0007dfc8`: set a bool or on/off-style enum argument to the
/// direction the request names, and a start/stop/pause/resume enum to the
/// one transport verb the request uses.
pub fn flip(args: &mut [(Vec<u8>, Json)], request: &[u8], tools: &[Tool], name: &[u8]) {
    let n = normq(request);
    for row in &VERB_DIRS {
        let Some((pos, target)) = fires(row, &n) else { continue };
        for (key, value) in args.iter_mut() {
            let info = schema::info(tools, name, key);
            match value {
                Json::Str(v) if !info.options.is_empty() => {
                    let vl = lower(v);
                    if (is_in(&vl, row.d) || is_in(&vl, row.e))
                        && !is_in(&vl, target)
                        && let Some(o) = info.options.iter().find(|o| is_in(&lower(o), target))
                    {
                        *v = o.clone();
                    }
                }
                Json::Bool(b) => {
                    let kl = lower(key);
                    if row.c.iter().any(|w| super::text::contains(&kl, w.as_bytes())) && !super::text::contains(&kl, b"off") {
                        *b = pos;
                    }
                }
                _ => {}
            }
        }
    }
    for (key, value) in args.iter_mut() {
        let Json::Str(v) = value else { continue };
        let info = schema::info(tools, name, key);
        if info.options.is_empty() {
            continue;
        }
        let opts: Vec<Vec<u8>> = info.options.iter().map(|o| lower(o)).collect();
        let vl = lower(v);
        if !TRANSPORT.iter().any(|(k, _)| k.as_bytes() == vl.as_slice()) {
            continue;
        }
        let named: Vec<&str> =
            TRANSPORT.iter().filter(|(k, syn)| opts.iter().any(|o| o == k.as_bytes()) && phrase_hit(&n, syn)).map(|(k, _)| *k).collect();
        if let [k] = named[..]
            && k.as_bytes() != vl.as_slice()
            && let Some(i) = opts.iter().position(|o| o == k.as_bytes())
        {
            *v = info.options[i].clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::schema::parse_tools;

    fn tools() -> Vec<Tool> {
        parse_tools(
            br#"[{"name":"lock_door","parameters":{"properties":{"door":{"type":"string"}}}},
                 {"name":"unlock_door","parameters":{"properties":{"door":{"type":"string"}}}},
                 {"name":"vacuum","parameters":{"properties":{"action":{"type":"string","enum":["Start","Pause","Resume","Stop"]},
                   "power":{"type":"boolean"},"state":{"type":"string","enum":["On","Off"]}}}}]"#,
        )
    }

    #[test]
    fn rename_to_opposite_tool() {
        let t = tools();
        assert_eq!(polarity_rename(b"unlock the front door", &t, b"lock_door"), Some(b"unlock_door".to_vec()));
        assert_eq!(polarity_rename(b"lock the front door", &t, b"lock_door"), None);
        assert_eq!(polarity_rename(b"don't unlock it, lock it", &t, b"unlock_door"), Some(b"lock_door".to_vec()));
    }

    #[test]
    fn flip_bool_enum_and_transport() {
        let t = tools();
        let mut args = vec![
            (b"power".to_vec(), Json::Bool(true)),
            (b"state".to_vec(), Json::Str(b"on".to_vec())),
            (b"action".to_vec(), Json::Str(b"start".to_vec())),
        ];
        flip(&mut args, b"turn off the vacuum and pause it", &t, b"vacuum");
        assert_eq!(args[0].1, Json::Bool(false));
        assert_eq!(args[1].1, Json::Str(b"Off".to_vec()));
        assert_eq!(args[2].1, Json::Str(b"Pause".to_vec()));
    }
}
