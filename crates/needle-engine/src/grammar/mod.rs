//! The decode grammar: the library's byte automaton for the call list,
//! compiled from the (normalized) tool schemas.
//!
//! The language is `[{"name":"<tool>","arguments":{...}},...]` with no
//! whitespace anywhere, `name` then `arguments` and nothing else. Argument
//! keys come in any order, each at most once, only declared ones, and
//! required ones before `}`. Values follow their schema (see
//! [`compile`]); ids are pinned to the conversation text; an identical
//! repeat of a call is refused.
//!
//! [`State`] steps one byte at a time, like the engine does for each
//! candidate token.

pub mod compile;
pub mod regex;
pub mod runtime;

use serde_json::Value;

use crate::rules::json::{self, Json};
pub use compile::Schemas;
pub use runtime::State;

#[derive(Clone, Debug, Default)]
pub struct Grammar {
    pub schemas: Schemas,
    /// The text grounded values must come from (the conversation so far);
    /// `None` switches grounding off.
    ctx: Option<Vec<u8>>,
    /// The list may not be empty (`[]` is refused).
    force: bool,
    /// When non-empty, only these tools may be called.
    allowed: Vec<Vec<u8>>,
}

impl Grammar {
    /// Compile tools that are already normalized (see
    /// [`crate::toolset::Toolset`]).
    pub fn from_json(tools: &[Json]) -> Grammar {
        Grammar { schemas: Schemas::compile(tools), ..Grammar::default() }
    }

    /// Compile a tools JSON array as the prompt carries it.
    pub fn from_text(tools_json: &[u8]) -> Grammar {
        match json::parse(tools_json) {
            Some(Json::Arr(items)) => Grammar::from_json(&items),
            _ => Grammar::default(),
        }
    }

    pub fn from_tools(tools: &[Value]) -> Grammar {
        let parsed: Vec<Json> = tools.iter().filter_map(|t| json::parse(needle_core::pyjson::dumps_compact(t).as_bytes())).collect();
        Grammar::from_json(&parsed)
    }

    /// Ground `id`/`*_id`/`grounded` values in this text.
    pub fn with_context(mut self, ctx: &[u8]) -> Grammar {
        self.ctx = Some(ctx.to_vec());
        self
    }

    /// Refuse an empty call list (a tool result's turn, or a trigger).
    pub fn must_call(mut self, force: bool) -> Grammar {
        self.force = force;
        self
    }

    /// Only these tools may be called (trigger matches).
    pub fn allow_only(mut self, names: Vec<Vec<u8>>) -> Grammar {
        self.allowed = names;
        self
    }

    pub fn start(&self) -> State<'_> {
        State::new(self)
    }

    /// The state after `b`, if every byte fits.
    pub fn state(&self, b: &[u8]) -> Option<State<'_>> {
        let mut s = self.start();
        s.feed(b).then_some(s)
    }

    /// Whether every byte of `b` fits, and whether it is a complete list.
    pub fn viable(&self, b: &[u8]) -> (bool, bool) {
        match self.state(b) {
            Some(s) => (true, s.complete()),
            None => (false, false),
        }
    }

    /// When `b` ends with the opening quote of a value whose schema is a
    /// set of string options (and every live reading agrees on it), those
    /// options.
    pub fn enum_open(&self, b: &[u8]) -> Option<Vec<String>> {
        let (last, head) = b.split_last()?;
        if *last != b'"' {
            return None;
        }
        let s = self.state(head)?;
        let opts = s.literal_options()?;
        if !s.accepts(b"\"") {
            return None;
        }
        Some(opts.iter().map(|o| String::from_utf8_lossy(o).into_owned()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn g() -> Grammar {
        Grammar::from_tools(&[json!({"name": "set_lights", "parameters": {"type": "object",
            "properties": {"room": {"type": "string", "enum": ["kitchen", "living_room"]},
                           "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                           "on": {"type": "boolean"}},
            "required": ["room"]}})])
    }

    fn ok(g: &Grammar, s: &str) -> bool {
        g.viable(s.as_bytes()).0
    }

    #[test]
    fn accepts_valid_calls_and_prefixes() {
        let g = g();
        let full = r#"[{"name":"set_lights","arguments":{"room":"kitchen","brightness":30}}]"#;
        assert_eq!(g.viable(full.as_bytes()), (true, true));
        for n in 0..full.len() {
            assert!(ok(&g, &full[..n]), "prefix {:?}", &full[..n]);
        }
        assert_eq!(g.viable(b"[]"), (true, true));
        assert!(!g.clone().must_call(true).viable(b"[]").0);
    }

    #[test]
    fn rejects_invalid() {
        let g = g();
        assert!(!ok(&g, r#"[{"name":"get_weather""#));
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"room":"garage""#));
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","brightness":150"#));
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"brightness":3}}"#), "missing required room");
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","room""#), "duplicate key");
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","brightness":1.5"#), "integer");
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","brightness":01"#), "leading zero");
        assert!(!ok(&g, r#"[{"name":"set_lights","arguments":{"room":"kitchen"}}, "#), "whitespace");
        assert!(ok(&g, r#"[{"name":"set_lights","arguments":{"room":"kitchen","brightness":10"#));
        // A lone 0 passes under a minimum until the number ends.
        let g2 = Grammar::from_tools(&[json!({"name": "t", "parameters": {"properties": {"m": {"type": "integer", "minimum": 1}}}})]);
        assert!(ok(&g2, r#"[{"name":"t","arguments":{"m":0"#));
        assert!(!ok(&g2, r#"[{"name":"t","arguments":{"m":0}"#));
    }

    #[test]
    fn duplicate_calls_are_refused() {
        let g = g();
        let one = r#"{"name":"set_lights","arguments":{"room":"kitchen"}}"#;
        assert!(!ok(&g, &format!("[{one},{one}")));
        assert!(ok(&g, &format!(r#"[{one},{{"name":"set_lights","arguments":{{"room":"living_room"}}}}]"#)));
    }

    #[test]
    fn grounded_ids_come_from_the_context() {
        let tools =
            [json!({"name": "get", "parameters": {"properties": {"order_id": {"type": "string"}, "user_id": {"type": "integer"}}}})];
        let g = Grammar::from_tools(&tools).with_context(b"track order AB-77 please");
        assert!(ok(&g, r#"[{"name":"get","arguments":{"order_id":"AB-77"}}]"#));
        assert!(!ok(&g, r#"[{"name":"get","arguments":{"order_id":"AB-78"#));
        // A numeric id needs a digit somewhere in the context.
        let g = Grammar::from_tools(&tools).with_context(b"no digits here");
        assert!(!ok(&g, r#"[{"name":"get","arguments":{"user_id""#));
    }

    #[test]
    fn arrays_patterns_and_variants() {
        let tools = [json!({"name": "t", "parameters": {"properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": true, "minItems": 1, "maxItems": 2},
            "code": {"type": "string", "pattern": "^[A-Z]{3}$"},
            "when": {"type": "string", "format": "date"},
            "v": {"anyOf": [{"type": "integer"}, {"type": "string", "enum": ["auto"]}]},
            "xs": {"type": "array"}}}})];
        let g = Grammar::from_tools(&tools);
        let call = |a: &str| format!(r#"[{{"name":"t","arguments":{{{a}}}}}]"#);
        assert!(ok(&g, &call(r#""tags":["a","b"]"#)));
        assert!(!ok(&g, &call(r#""tags":["a","a"]"#)));
        assert!(!ok(&g, &call(r#""tags":[]"#)));
        assert!(!ok(&g, &call(r#""tags":["a","b","c"]"#)));
        assert!(ok(&g, &call(r#""code":"ABC""#)));
        assert!(!ok(&g, &call(r#""code":"ABCD""#)));
        assert!(ok(&g, &call(r#""when":"2026-09-26""#)));
        assert!(!ok(&g, &call(r#""when":"tomorrow""#)));
        assert!(ok(&g, &call(r#""v":3"#)));
        assert!(ok(&g, &call(r#""v":"auto""#)));
        assert!(!ok(&g, &call(r#""v":"manual""#)));
        // Seven identical items in a row, or 64 items, end an array; bare
        // numbers are not counted.
        let seven = ["\"a\""; 7].join(",");
        assert!(ok(&g, &call(&format!(r#""xs":[{seven}]"#))));
        assert!(!ok(&g, &call(&format!(r#""xs":[{seven},"a"]"#))));
        let nums = ["1"; 70].join(",");
        assert!(ok(&g, &call(&format!(r#""xs":[{nums}]"#))));
    }

    #[test]
    fn enum_open_reports_string_options() {
        let g = g();
        let opts = g.enum_open(br#"[{"name":"set_lights","arguments":{"room":""#).unwrap();
        assert_eq!(opts, vec!["kitchen", "living_room"]);
        assert!(g.enum_open(br#"[{"name":"set_lights","arguments":{""#).is_none());
    }
}
