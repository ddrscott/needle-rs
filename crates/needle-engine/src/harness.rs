//! Environment acceptance suites (`needle.environments._harness`), scored on
//! the model's own decoded calls.

use anyhow::Result;
use needle_core::Tokenizer;
use needle_core::pyjson::{DumpOpts, dumps};
use needle_core::render::{calls_of, render_example};
use serde_json::{Map, Value};

use crate::generate::{GenOptions, generate};
use crate::model::Model;

/// `_fold`: casefold strings, integral floats as ints.
fn fold(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(s.to_lowercase()),
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().unwrap();
            if f.fract() == 0.0 && f.abs() < 9e15 { Value::from(f as i64) } else { v.clone() }
        }
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), fold(x))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(fold).collect()),
        other => other.clone(),
    }
}

fn call_keys(calls: &[Value]) -> Vec<String> {
    let mut keys: Vec<String> = calls.iter().map(|c| dumps(&fold(c), DumpOpts::SORTED)).collect();
    keys.sort();
    keys
}

pub struct SuiteResult {
    pub passed: usize,
    pub total: usize,
    pub critical_failures: usize,
    pub failures: Vec<String>,
}

impl SuiteResult {
    /// `passed >= round(0.9 * total)` with no critical failure.
    pub fn ok(&self) -> bool {
        self.passed as f64 >= (0.9 * self.total as f64).round() && self.critical_failures == 0
    }
}

/// The calls a completion makes, as `{"name", "arguments"}` values.
pub fn decoded_calls(text: &str) -> Vec<Value> {
    let Some(body) = text.split_once(needle_core::render::TOOL_CALL_START).map(|(_, b)| b) else { return vec![] };
    let body = body.split(needle_core::render::TOOL_CALL_END).next().unwrap_or("");
    match serde_json::from_str::<Value>(body) {
        Ok(Value::Array(calls)) => calls.into_iter().filter(|c| c.is_object()).collect(),
        _ => vec![],
    }
}

pub fn run_suite(model: &Model, tok: &Tokenizer, env: &Value, max_new_tokens: usize) -> Result<SuiteResult> {
    let cases = env["test_cases"].as_array().cloned().unwrap_or_default();
    let mut res = SuiteResult { passed: 0, total: cases.len(), critical_failures: 0, failures: vec![] };
    for case in &cases {
        let mut ex = Map::new();
        ex.insert("system".into(), env["system"].clone());
        ex.insert("tools".into(), env["tools"].clone());
        ex.insert("query".into(), case["query"].clone());
        let (prompt, _) = render_example(&ex)?;
        let opts = GenOptions { max_new_tokens, ..Default::default() };
        let (text, _) = generate(model, tok, &prompt, &opts, |_| {})?;
        let got = decoded_calls(&text);
        let want = case["calls"].as_array().cloned().unwrap_or_default();
        let _ = calls_of(&text);
        if call_keys(&got) == call_keys(&want) {
            res.passed += 1;
        } else {
            if case.get("critical").and_then(Value::as_bool).unwrap_or(false) {
                res.critical_failures += 1;
            }
            res.failures.push(format!(
                "FAIL [{}] {}\n  want {}\n  got  {}",
                case["category"].as_str().unwrap_or(""),
                case["query"].as_str().unwrap_or(""),
                dumps(&Value::Array(want), DumpOpts::DEFAULT),
                dumps(&Value::Array(got), DumpOpts::DEFAULT)
            ));
        }
    }
    Ok(res)
}

/// Exact-call accuracy of greedy decodes over examples (`_score_quantised`):
/// the prompt rendered without answers, `max_new_tokens` of decode, the
/// calls compared by name and sorted-key arguments.
pub fn exact_calls(model: &Model, tok: &Tokenizer, examples: &[&Map<String, Value>], max_new_tokens: usize) -> Result<usize> {
    use needle_core::pyjson::dumps_compact;
    use needle_core::render::{TOOL_CALL_END, TOOL_CALL_START};
    let mut correct = 0;
    for ex in examples {
        let mut e = (*ex).clone();
        e.insert("answers".into(), Value::Array(vec![]));
        let (prompt, _) = render_example(&e)?;
        let (text, _) = generate(model, tok, &prompt, &GenOptions { max_new_tokens, ..Default::default() }, |_| {})?;
        let answers = ex.get("answers").or_else(|| ex.get("function_calls")).cloned().unwrap_or(Value::Array(vec![]));
        let want = calls_of(&format!("{TOOL_CALL_START}{}{TOOL_CALL_END}", dumps_compact(&answers)));
        correct += (calls_of(&text) == want) as usize;
    }
    Ok(correct)
}
