//! The chat wire format: how an example (tools, query, answers) becomes the
//! prompt and target the model trains and decodes on.

use std::io::BufRead;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::pyjson::{DumpOpts, dumps, dumps_compact, float_repr};
use crate::tokenizer::{BOS_ID, EOS_ID, PAD_ID, Tokenizer};

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const THINK_START: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const TOOLS_START: &str = "<tools>";
pub const TOOLS_END: &str = "</tools>";
pub const TOOL_CALL_START: &str = "<tool_call>";
pub const TOOL_CALL_END: &str = "</tool_call>";
pub const TOOL_RESULT_START: &str = "<tool_result>";
pub const TOOL_RESULT_END: &str = "</tool_result>";
pub const CONTEXT_START: &str = "<context>";
pub const CONTEXT_END: &str = "</context>";
pub const EXTRACT_START: &str = "<extract>";
pub const EXTRACT_END: &str = "</extract>";
pub const SCHEMA_START: &str = "<schema>";
pub const SCHEMA_END: &str = "</schema>";

pub const CHAT_MARKERS: [&str; 16] = [
    IM_START,
    IM_END,
    THINK_START,
    THINK_END,
    TOOLS_START,
    TOOLS_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESULT_START,
    TOOL_RESULT_END,
    CONTEXT_START,
    CONTEXT_END,
    EXTRACT_START,
    EXTRACT_END,
    SCHEMA_START,
    SCHEMA_END,
];

/// Python `str.isspace` per character (Unicode whitespace plus the ASCII
/// information separators `\x1c`-`\x1f`).
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `str(x)` for a decoded JSON value.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if !n.is_i64() && !n.is_u64() => float_repr(f),
            _ => n.to_string(),
        },
        other => other.to_string(),
    }
}

fn text_field(example: &Map<String, Value>, key: &str) -> String {
    match example.get(key) {
        Some(v) if truthy(v) => py_strip(&py_str(v)).to_string(),
        _ => String::new(),
    }
}

/// `render_example(example) -> (prompt, target)`.
pub fn render_example(example: &Map<String, Value>) -> Result<(String, String)> {
    let tools = example.get("tools").cloned().unwrap_or(Value::Array(vec![]));
    let tools_json = match &tools {
        Value::String(s) => s.clone(),
        v => dumps_compact(v),
    };
    let answers = example.get("answers").or_else(|| example.get("function_calls")).cloned().unwrap_or(Value::Array(vec![]));
    let answers_json = match &answers {
        Value::String(s) => s.clone(),
        v => dumps_compact(v),
    };
    let reasoning = text_field(example, "reasoning");
    let system = text_field(example, "system");
    let query = match example.get("query") {
        Some(Value::String(q)) => q.clone(),
        Some(v) => bail!("example query must be a string, got {v}"),
        None => bail!("example has no query"),
    };
    let prefix = if system.is_empty() { String::new() } else { format!("{IM_START}system\n{system}{IM_END}\n") };
    let prompt = format!("{prefix}{IM_START}user\n{TOOLS_START}{tools_json}{TOOLS_END}\n{query}{IM_END}\n{IM_START}assistant\n");
    let think = if reasoning.is_empty() { String::new() } else { format!("{THINK_START}\n{reasoning}\n{THINK_END}\n") };
    let target = format!("{think}{TOOL_CALL_START}{answers_json}{TOOL_CALL_END}{IM_END}");
    Ok((prompt, target))
}

/// The prompt alone for a query over a toolset (`run.build_prompt`).
pub fn build_prompt(query: &str, tools: Option<&Value>) -> Result<String> {
    match tools {
        Some(t) if truthy(t) => {
            let mut ex = Map::new();
            ex.insert("query".into(), Value::String(query.into()));
            ex.insert("tools".into(), t.clone());
            Ok(render_example(&ex)?.0)
        }
        _ => Ok(query.to_string()),
    }
}

/// `_encode`: BOS + prompt + target + EOS, loss mask over target and EOS,
/// truncated and padded to `max_len`.
pub fn encode_example(tok: &Tokenizer, example: &Map<String, Value>, max_len: usize) -> Result<(Vec<u32>, Vec<f32>)> {
    let (prompt, target) = render_example(example)?;
    let p = tok.encode(&prompt);
    let t = tok.encode(&target);
    let mut ids = Vec::with_capacity(p.len() + t.len() + 2);
    ids.push(BOS_ID);
    ids.extend(&p);
    ids.extend(&t);
    ids.push(EOS_ID);
    let mut mask = vec![0f32; 1 + p.len()];
    mask.extend(std::iter::repeat_n(1f32, t.len() + 1));
    ids.truncate(max_len);
    mask.truncate(max_len);
    ids.resize(max_len, PAD_ID);
    mask.resize(max_len, 0.0);
    Ok((ids, mask))
}

/// `from_chat`: a single-turn chat-format line as a query/answers example.
pub fn from_chat(example: &Map<String, Value>) -> Result<Option<Map<String, Value>>> {
    let Some(Value::Array(messages)) = example.get("messages") else { return Ok(None) };
    if example.contains_key("query") {
        return Ok(None);
    }
    let role = |m: &Value| m.get("role").and_then(Value::as_str).map(str::to_string);
    let turns: Vec<&Value> = messages.iter().filter(|m| role(m).as_deref() != Some("system")).collect();
    if turns.len() != 2 || role(turns[0]).as_deref() != Some("user") || role(turns[1]).as_deref() != Some("assistant") {
        return Ok(None);
    }
    let mut answers = Vec::new();
    if let Some(Value::Array(calls)) = turns[1].get("tool_calls") {
        for call in calls {
            let function = match call.get("function") {
                Some(f) if truthy(f) => f,
                _ => call,
            };
            let mut arguments = function.get("arguments").cloned().unwrap_or(Value::Object(Map::new()));
            if let Value::String(s) = &arguments {
                arguments = if py_strip(s).is_empty() {
                    Value::Object(Map::new())
                } else {
                    serde_json::from_str(s).context("tool call arguments are not JSON")?
                };
            }
            let mut a = Map::new();
            a.insert("name".into(), function.get("name").cloned().unwrap_or(Value::Null));
            a.insert("arguments".into(), arguments);
            answers.push(Value::Object(a));
        }
    }
    let tools: Vec<Value> = match example.get("tools") {
        Some(Value::Array(ts)) => ts
            .iter()
            .map(|t| match t {
                Value::Object(o) => o.get("function").cloned().unwrap_or_else(|| t.clone()),
                other => other.clone(),
            })
            .collect(),
        _ => vec![],
    };
    let content = match turns[0].get("content") {
        Some(v) if truthy(v) => v.clone(),
        _ => Value::String(String::new()),
    };
    let mut out = Map::new();
    out.insert("query".into(), content);
    out.insert("tools".into(), Value::Array(tools));
    out.insert("answers".into(), Value::Array(answers));
    let system = messages.iter().find(|m| role(m).as_deref() == Some("system"));
    if let Some(content) = system.and_then(|m| m.get("content")).filter(|c| truthy(c)) {
        out.insert("system".into(), content.clone());
    }
    Ok(Some(out))
}

/// `read_examples`: query/answers lines, plus convertible chat lines.
/// Returns the examples and how many lines were skipped.
pub fn read_examples(path: &std::path::Path) -> Result<(Vec<Map<String, Value>>, usize)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    let mut skipped = 0;
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        let line = py_strip(&line);
        if line.is_empty() {
            continue;
        }
        let example: Map<String, Value> = serde_json::from_str(line).context("JSONL line is not an object")?;
        let example = if example.contains_key("query") { Some(example) } else { from_chat(&example)? };
        match example {
            Some(e) => out.push(e),
            None => skipped += 1,
        }
    }
    Ok((out, skipped))
}

/// `fit_max_len`: the smallest power-of-two bucket (from 128) holding the
/// longest example, capped.
pub fn fit_max_len(examples: &[Map<String, Value>], tok: &Tokenizer, cap: usize) -> Result<usize> {
    let mut longest = 0;
    for e in examples {
        let (p, t) = render_example(e)?;
        longest = longest.max(tok.encode(&p).len() + tok.encode(&t).len() + 2);
    }
    let mut bucket = 128;
    while bucket < longest.min(cap) {
        bucket *= 2;
    }
    Ok(bucket.min(cap))
}

/// One parsed call: `(name, json.dumps(arguments, sort_keys=True))`.
pub type CallKey = (String, String);

/// `_calls_of`: the calls in a completion; `None` when the call block does
/// not parse as a JSON list, empty when there is no call block.
pub fn calls_of(text: &str) -> Option<Vec<CallKey>> {
    let Some((_, after)) = text.split_once(TOOL_CALL_START) else { return Some(vec![]) };
    let body = after.split(TOOL_CALL_END).next().unwrap_or("");
    let calls: Value = serde_json::from_str(body).ok()?;
    let Value::Array(calls) = calls else { return None };
    Some(
        calls
            .iter()
            .filter_map(|c| c.as_object())
            .map(|c| {
                let name = py_str(c.get("name").unwrap_or(&Value::Null));
                let args = match c.get("arguments") {
                    Some(a) if truthy(a) => a.clone(),
                    _ => Value::Object(Map::new()),
                };
                (name, dumps(&args, DumpOpts::SORTED))
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_like_reference() {
        let ex = json!({"query": "dim the kitchen to 10",
            "tools": [{"name": "set_lights", "parameters": {"type": "object"}}],
            "reasoning": "  room from query ",
            "answers": [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 10}}]});
        let (p, t) = render_example(ex.as_object().unwrap()).unwrap();
        assert_eq!(
            p,
            "<|im_start|>user\n<tools>[{\"name\":\"set_lights\",\"parameters\":{\"type\":\"object\"}}]</tools>\ndim the kitchen to 10<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            t,
            "<think>\nroom from query\n</think>\n<tool_call>[{\"name\":\"set_lights\",\"arguments\":{\"room\":\"kitchen\",\"brightness\":10}}]</tool_call><|im_end|>"
        );
    }

    #[test]
    fn calls_compare_by_sorted_arguments() {
        let a = calls_of("<tool_call>[{\"name\":\"f\",\"arguments\":{\"b\":1,\"a\":2}}]</tool_call>").unwrap();
        let b = calls_of("x<tool_call>[{\"arguments\":{\"a\":2,\"b\":1},\"name\":\"f\"}]").unwrap();
        assert_eq!(a, b);
        assert_eq!(calls_of("no call"), Some(vec![]));
        assert_eq!(calls_of("<tool_call>{oops"), None);
    }
}
