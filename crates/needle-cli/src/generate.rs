//! `needle generate-data`: synthesize training examples from tool schemas
//! through an OpenRouter chat model (the reference's `generate_dataset`).

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result, bail};
use needle_core::pyjson::{DumpOpts, dumps};
use serde_json::{Map, Value, json};

pub const DEFAULT_MODEL: &str = "deepseek/deepseek-flash-latest";

const GEN_SYSTEM: &str = "You generate training data for a tool-calling and extraction model. Given a set of tool or record schemas, produce realistic, diverse inputs paired with the exact calls that satisfy them. Return only JSON.";

fn gen_prompt(tools_json: &str, n: usize, refusals: usize) -> String {
    format!(
        r#"Schemas available (JSON):
{tools_json}

Produce {n} varied examples as a JSON array. Each element is an object:
  {{"query": "<a natural user request to act on, or a passage of text to extract from>",
    "reasoning": "<one short line deriving each argument from its source span in the query>",
    "answers": [{{"name": "<schema name>", "arguments": {{...}}}}]}}

Rules:
- Use only the schemas above; arguments must match them exactly and contain only
  values evidenced in the query.
- For an action tool the query is a command. For a record/extraction schema (its
  fields describe an entity), the query is a natural passage that contains those
  fields and the call extracts them.
- Cover single-call, multi-call, and about {refusals} off-topic inputs that no
  schema can serve (for those, "answers" is []).
- Vary phrasing, values, and which schemas are used. Return ONLY the JSON array."#
    )
}

fn openrouter_url() -> String {
    std::env::var("OPENROUTER_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1/chat/completions".into())
}

fn api_key(given: Option<&str>) -> Result<String> {
    match given.map(str::to_string).or_else(|| std::env::var("OPENROUTER_API_KEY").ok()) {
        Some(k) if !k.is_empty() => Ok(k),
        _ => bail!("set OPENROUTER_API_KEY to generate data"),
    }
}

fn chat(system: &str, user: &str, model: &str, key: &str) -> Result<String> {
    let body =
        json!({"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0.9});
    let resp: Value = ureq::post(&openrouter_url())
        .timeout(std::time::Duration::from_secs(180))
        .set("Authorization", &format!("Bearer {key}"))
        .set("HTTP-Referer", "https://github.com/cactus-compute/needle")
        .set("X-Title", "needle")
        .send_json(body)
        .map_err(|e| anyhow::anyhow!("openrouter: {e}"))?
        .into_json()?;
    resp["choices"][0]["message"]["content"].as_str().map(str::to_string).context("openrouter reply has no content")
}

/// `_parse_array`: the first-to-last bracket span, rows with query and answers.
fn parse_array(text: &str) -> Vec<Map<String, Value>> {
    let (Some(start), Some(end)) = (text.find('['), text.rfind(']')) else { return vec![] };
    if end <= start {
        return vec![];
    }
    match serde_json::from_str::<Value>(&text[start..=end]) {
        Ok(Value::Array(rows)) => rows
            .into_iter()
            .filter_map(|r| match r {
                Value::Object(o) if o.contains_key("query") && o.contains_key("answers") => Some(o),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

pub fn generate_examples(tools: &Value, n: usize, model: &str, key: &str, refusals: usize) -> Result<Vec<Map<String, Value>>> {
    let tools_json = match tools {
        Value::String(s) => s.clone(),
        v => serde_json::to_string_pretty(v)?,
    };
    let text = chat(GEN_SYSTEM, &gen_prompt(&tools_json, n, refusals), model, key)?;
    let mut rows = parse_array(&text);
    for r in &mut rows {
        r.entry("tools").or_insert_with(|| tools.clone());
    }
    Ok(rows)
}

fn dedup_key(e: &Map<String, Value>) -> (String, String) {
    let answers = e.get("answers").or_else(|| e.get("function_calls")).cloned().unwrap_or(Value::Array(vec![]));
    (e.get("query").and_then(Value::as_str).unwrap_or("").trim().to_lowercase(), dumps(&answers, DumpOpts::SORTED))
}

/// `generate_dataset`: parallel batches until `num_samples` unique rows.
pub fn generate_dataset(
    tools: &Value,
    num_samples: usize,
    model: &str,
    batch_size: usize,
    key: Option<&str>,
    workers: usize,
    mut progress: impl FnMut(usize, usize, usize),
) -> Result<Vec<Map<String, Value>>> {
    let key = api_key(key)?;
    let target = num_samples * 13 / 10;
    let max_submissions = (target / batch_size * 3).max(1);
    let (tx, rx) = mpsc::channel::<Result<Vec<Map<String, Value>>>>();
    let submit = |tx: mpsc::Sender<_>| {
        let (tools, model, key) = (tools.clone(), model.to_string(), key.clone());
        std::thread::spawn(move || {
            let _ = tx.send(generate_examples(&tools, batch_size, &model, &key, 3));
        });
    };
    let mut submitted = 0;
    let mut pending = 0;
    for _ in 0..workers.min(target.div_ceil(batch_size).max(1)) {
        submit(tx.clone());
        submitted += 1;
        pending += 1;
    }
    let (mut seen, mut rows, mut failed) = (HashSet::new(), vec![], 0);
    while pending > 0 && rows.len() < num_samples {
        let res = rx.recv()?;
        pending -= 1;
        match res {
            Ok(batch) => {
                for e in batch {
                    if seen.insert(dedup_key(&e)) {
                        rows.push(e);
                    }
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("  {:<9} {e}", "failed");
            }
        }
        if rows.len() < num_samples && submitted < max_submissions {
            submit(tx.clone());
            submitted += 1;
            pending += 1;
        }
        progress(rows.len().min(num_samples), num_samples, failed);
    }
    rows.truncate(num_samples);
    Ok(rows)
}

fn write_jsonl(path: &Path, rows: &[Map<String, Value>]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?);
    for r in rows {
        writeln!(f, "{}", dumps(&Value::Object(r.clone()), DumpOpts::DEFAULT))?;
    }
    Ok(())
}

fn print_progress(done: usize, total: usize, failed: usize) {
    println!("  {:<9} {done}/{total}  failed {failed}", "generated");
}

/// `augment_jsonl`: existing examples plus generated ones over their tools.
pub fn augment_jsonl(
    path: &Path,
    num_samples: usize,
    model: &str,
    batch_size: usize,
    out: Option<&Path>,
    workers: usize,
) -> Result<PathBuf> {
    let file = std::fs::File::open(path)?;
    let mut examples = vec![];
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            examples.push(serde_json::from_str::<Map<String, Value>>(&line)?);
        }
    }
    let mut names = HashSet::new();
    let mut tools = vec![];
    for e in &examples {
        for t in e.get("tools").and_then(Value::as_array).into_iter().flatten() {
            if let Some(n) = t.get("name").and_then(Value::as_str)
                && names.insert(n.to_string())
            {
                tools.push(t.clone());
            }
        }
    }
    if tools.is_empty() {
        bail!("no tool schemas found in {}", path.display());
    }
    let out = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(format!("{}.augmented.jsonl", path.display().to_string().trim_end_matches(".jsonl"))));
    let generated = generate_dataset(&Value::Array(tools), num_samples, model, batch_size, None, workers, print_progress)?;
    let all: Vec<_> = examples.into_iter().chain(generated).collect();
    write_jsonl(&out, &all)?;
    println!("  {:<9} {} examples  {}", "wrote", all.len(), out.display());
    Ok(out)
}

pub fn generate_main(
    tools: Option<&Path>,
    augment: Option<&Path>,
    num_samples: usize,
    batch_size: usize,
    workers: usize,
    model: &str,
    output: Option<&Path>,
) -> Result<()> {
    if let Some(t) = tools {
        let tools: Value = serde_json::from_str(&std::fs::read_to_string(t)?)?;
        let out = output.map(Path::to_path_buf).unwrap_or_else(|| "needle_data.jsonl".into());
        let rows = generate_dataset(&tools, num_samples, model, batch_size, None, workers, print_progress)?;
        write_jsonl(&out, &rows)?;
        println!("  {:<9} {} examples  {}", "wrote", rows.len(), out.display());
        Ok(())
    } else if let Some(a) = augment {
        augment_jsonl(a, num_samples, model, batch_size, output, workers).map(|_| ())
    } else {
        bail!("pass --tools <schemas.json> or --augment <data.jsonl>")
    }
}
