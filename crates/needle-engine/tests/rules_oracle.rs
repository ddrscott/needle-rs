//! Replays call lists through the post-processor and checks each against
//! the native library's own output, recorded by calling its
//! `FUN_0006998c` directly (`tests/parity/rules_oracle.py`). Set
//! `NEEDLE_RULES_CASES` to check another recording.

use needle_engine::rules::{Context, postprocess, schema};
use serde_json::Value;

#[test]
fn post_processor_matches_native() {
    let path = std::env::var("NEEDLE_RULES_CASES")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/parity/rules_cases.jsonl").to_string());
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("no recorded cases at {path}");
        return;
    };
    let mut bad = vec![];
    let mut n = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let c: Value = serde_json::from_str(line).expect("case is JSON");
        let s = |k: &str| c[k].as_str().unwrap_or_default().to_string();
        let tools_text = match &c["tools"] {
            Value::String(t) => t.clone(),
            t => needle_core::pyjson::dumps_compact(t),
        };
        let tools = schema::parse_tools(tools_text.as_bytes());
        let (request, system) = (s("request"), c["system"].as_str().unwrap_or(" ").to_string());
        let conversation = c["conversation"].as_str().map_or_else(|| format!("{system}\n{request}"), str::to_string);
        let ctx = Context { request: request.as_bytes(), system: system.as_bytes(), conversation: conversation.as_bytes(), tools: &tools };
        let out = postprocess(s("calls").as_bytes(), &ctx);
        let native = &c["native"];
        let got = (String::from_utf8_lossy(&out.calls).into_owned(), out.withhold, out.withhold_all);
        let want = (native["out"].as_str().unwrap_or_default().to_string(), native["withhold"] == true, native["withhold2"] == true);
        n += 1;
        if got != want {
            bad.push(format!("{request}\n  calls  {}\n  native {want:?}\n  ours   {got:?}", s("calls")));
        }
    }
    assert!(bad.is_empty(), "{} of {n} cases differ:\n{}", bad.len(), bad.join("\n"));
}
