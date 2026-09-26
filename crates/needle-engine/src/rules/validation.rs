//! The envelope's `validation` block (turn loop `0x608fc`): string and
//! number arguments the conversation never states, and whether the request
//! carries a negation cue (`FUN_0006b138`).

use super::json::{Json, call_args, call_name};
use super::schema::{Tool, find_tool};
use super::tail::number_grounded;
use super::text::{contains, find_all, lower, squash};

/// `tool.key` for every string or number argument the conversation does
/// not ground, in call and argument order. Parameters with an enum, a
/// `const` or `"grounded": true` are exempt; unknown ones are not.
pub fn ungrounded_fields(calls: &[Json], tools: &[Tool], conversation: &[u8]) -> Vec<String> {
    let hay = squash(conversation);
    let mut out = vec![];
    for call in calls {
        let (Some(name), Some(args)) = (call_name(call), call_args(call)) else { continue };
        let tool = find_tool(tools, name);
        for (key, v) in args {
            let exempt = tool.and_then(|t| t.params.iter().find(|p| &p.key == key)).is_some_and(|p| p.closed);
            let grounded = match v {
                Json::Str(s) => {
                    let s = squash(s);
                    s.len() <= 2 || contains(&hay, &s)
                }
                Json::Num(t) => number_grounded(&hay, t),
                _ => true,
            };
            if !exempt && !grounded {
                out.push(format!("{}.{}", String::from_utf8_lossy(name), String::from_utf8_lossy(key)));
            }
        }
    }
    out
}

const CUES: [&str; 10] =
    ["don't", "dont ", "do not", "never", "no longer", "must not", "mustn't", "shouldn't", "should not", "make sure not"];

/// `FUN_0006b138`: a negation cue anywhere in the request (substring, so
/// `whenever` counts), unless it is `don't mind/forget/hesitate/...`.
pub fn negation_cue(request: &[u8]) -> bool {
    let q = lower(request);
    CUES.iter().any(|cue| {
        find_all(&q, cue.as_bytes()).into_iter().any(|at| {
            let mut p = at + cue.len();
            while q.get(p) == Some(&b' ') {
                p += 1;
            }
            !["mind", "forget", "hesitate", "worry", "disturb"].iter().any(|w| q[p..].starts_with(w.as_bytes()))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::json::parse;
    use crate::rules::schema::parse_tools;

    #[test]
    fn cues() {
        assert!(negation_cue(b"Don't pause the song"));
        assert!(negation_cue(b"call me whenever"));
        assert!(!negation_cue(b"don't forget the milk"));
        assert!(!negation_cue(b"dont"));
        assert!(negation_cue(b"I never"));
    }

    #[test]
    fn ungrounded() {
        let tools = parse_tools(
            br#"[{"name":"e","parameters":{"properties":{"cat":{"type":"string","enum":["a"]},"amount":{"type":"number"},"note":{"type":"string"}}}}]"#,
        );
        let calls = [parse(br#"{"name":"e","arguments":{"cat":"zzz","amount":14,"note":"Lunch-Box","x":"hi"}}"#).unwrap()];
        assert_eq!(ungrounded_fields(&calls, &tools, b" \nset the volume to 140 for my lunchbox"), ["e.amount"]);
    }
}
