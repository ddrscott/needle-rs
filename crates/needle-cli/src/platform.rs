//! The hosted fine-tuning platform at cactuscompute.com (`needle.platform`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};

pub const BASE_URL: &str = "https://cactuscompute.com/v1";
pub const KEYS_URL: &str = "https://cactuscompute.com/dashboard/api-keys";
pub const JOBS_URL: &str = "https://cactuscompute.com/dashboard/jobs";
pub const BASE_MODEL: &str = "needle-3";
const TERMINAL: [&str; 3] = ["succeeded", "failed", "cancelled"];

/// An API failure with the platform's stable error code.
#[derive(Debug)]
pub struct PlatformError {
    pub code: String,
    pub message: String,
    pub status: u16,
    pub param: Option<String>,
    pub url: Option<String>,
}

impl std::fmt::Display for PlatformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)?;
        if let Some(p) = &self.param {
            write!(f, " (param: {p})")?;
        }
        if let Some(u) = &self.url {
            write!(f, " -> {u}")?;
        }
        Ok(())
    }
}

impl std::error::Error for PlatformError {}

fn perr(code: &str, message: impl Into<String>, status: u16) -> PlatformError {
    PlatformError { code: code.into(), message: message.into(), status, param: None, url: None }
}

/// `openai_tools`: flat tools to the OpenAI form the platform takes.
pub fn openai_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| if t.get("type") == Some(&json!("function")) { t.clone() } else { json!({"type": "function", "function": t}) })
        .collect()
}

pub struct Platform {
    api_key: Option<String>,
    base_url: String,
    retries: usize,
    agent: ureq::Agent,
    follower: ureq::Agent,
}

impl Platform {
    pub fn new(api_key: Option<String>) -> Self {
        let api_key = api_key.or_else(|| std::env::var("NEEDLE_API_KEY").ok()).filter(|k| !k.is_empty());
        let timeout = Duration::from_secs(60);
        Self {
            api_key,
            base_url: std::env::var("NEEDLE_PLATFORM_URL").unwrap_or_else(|_| BASE_URL.into()).trim_end_matches('/').into(),
            retries: 3,
            agent: ureq::AgentBuilder::new().redirects(0).timeout(timeout).build(),
            follower: ureq::AgentBuilder::new().timeout(timeout).build(),
        }
    }

    fn send(
        &self,
        method: &str,
        url: &str,
        body: Option<&Value>,
        raw: Option<&[u8]>,
        auth: bool,
    ) -> Result<(u16, Option<String>, Vec<u8>), PlatformError> {
        for attempt in 0..=self.retries {
            let mut req = self.agent.request(method, url);
            if auth && let Some(k) = &self.api_key {
                req = req.set("Authorization", &format!("Bearer {k}"));
            }
            let res = match (body, raw) {
                (Some(b), _) => req.send_json(b.clone()),
                (None, Some(r)) => req.set("Content-Type", "application/octet-stream").send_bytes(r),
                _ => req.call(),
            };
            match res {
                Ok(resp) => {
                    let status = resp.status();
                    let location = resp.header("Location").map(str::to_string);
                    let mut bytes = vec![];
                    std::io::Read::read_to_end(&mut resp.into_reader(), &mut bytes).map_err(|e| perr("network_error", e.to_string(), 0))?;
                    return Ok((status, location, bytes));
                }
                Err(ureq::Error::Status(code, resp)) => {
                    if (301..=308).contains(&code) {
                        return Ok((code, resp.header("Location").map(str::to_string), vec![]));
                    }
                    let retry_after = resp.header("Retry-After").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) as u64;
                    let payload: Value = resp.into_json().unwrap_or(Value::Null);
                    let e = payload.get("error").cloned().unwrap_or(Value::Null);
                    let s = |k: &str| e.get(k).and_then(Value::as_str).map(str::to_string);
                    let mut err = PlatformError {
                        code: s("code").unwrap_or_else(|| format!("http_{code}")),
                        message: s("message").unwrap_or_else(|| format!("HTTP {code}")),
                        status: code,
                        param: s("param"),
                        url: s("url"),
                    };
                    if code == 401 && self.api_key.is_none() {
                        err.message = "set NEEDLE_API_KEY to a key from the console".into();
                        err.url = err.url.or(Some(KEYS_URL.into()));
                    }
                    if code == 429 && err.code == "rate_limited" && attempt < self.retries {
                        std::thread::sleep(Duration::from_secs(retry_after.max(1)));
                        continue;
                    }
                    return Err(err);
                }
                Err(e) => return Err(perr("network_error", e.to_string(), 0)),
            }
        }
        Err(perr("rate_limited", "the API kept answering 429", 429))
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>, params: &[(&str, Option<String>)]) -> Result<Value, PlatformError> {
        let mut url = format!("{}/{path}", self.base_url);
        let q: Vec<String> = params.iter().filter_map(|(k, v)| v.as_ref().map(|v| format!("{k}={}", urlencode(v)))).collect();
        if !q.is_empty() {
            url = format!("{url}?{}", q.join("&"));
        }
        let (_, _, bytes) = self.send(method, &url, body.as_ref(), None, true)?;
        if bytes.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(&bytes).map_err(|e| perr("unexpected_response", e.to_string(), 0))
    }

    fn signed_link(&self, path: &str) -> Result<String, PlatformError> {
        let (status, loc, _) = self.send("GET", &format!("{}/{path}", self.base_url), None, None, true)?;
        match loc {
            Some(l) if (301..=308).contains(&status) => Ok(l),
            _ => Err(perr("unexpected_response", format!("expected a redirect from {path}, got {status}"), status)),
        }
    }

    fn fetch_to(&self, link: &str, dest: &Path) -> Result<PathBuf> {
        crate::net::download_url(self.follower.get(link), dest)?;
        Ok(dest.to_path_buf())
    }

    pub fn billing(&self) -> Result<Value, PlatformError> {
        self.call("GET", "billing", None, &[])
    }

    pub fn files(&self) -> Result<Vec<Value>, PlatformError> {
        Ok(self.call("GET", "files", None, &[("limit", Some("100".into()))])?["data"].as_array().cloned().unwrap_or_default())
    }

    /// Upload a `.jsonl` in the platform's three steps.
    pub fn upload(&self, path: &Path) -> Result<Value> {
        let bytes = std::fs::read(path)?;
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("data.jsonl");
        let reservation = self.call("POST", "files", Some(json!({"name": name, "bytes": bytes.len()})), &[])?;
        let url = reservation["url"].as_str().unwrap_or_default().to_string();
        self.send("PUT", &url, None, Some(&bytes), false)?;
        Ok(self.call("POST", &format!("files/{}/complete", reservation["id"].as_str().unwrap_or_default()), None, &[])?)
    }

    pub fn download_file(&self, file_id: &str, dest: &Path) -> Result<PathBuf> {
        let link = self.signed_link(&format!("files/{file_id}/content"))?;
        self.fetch_to(&link, dest)
    }

    pub fn generate(
        &self,
        tools: &[Value],
        examples: usize,
        description: Option<&str>,
        messages: Option<&[String]>,
        suffix: Option<&str>,
    ) -> Result<Value, PlatformError> {
        let mut body = json!({"tools": openai_tools(tools), "examples": examples});
        if let Some(d) = description {
            body["description"] = json!(d);
        }
        if let Some(m) = messages {
            body["messages"] = json!(m);
        }
        if let Some(s) = suffix {
            body["suffix"] = json!(s);
        }
        self.call("POST", "generations", Some(body), &[])
    }

    pub fn generation(&self, id: &str) -> Result<Value, PlatformError> {
        self.call("GET", &format!("generations/{id}"), None, &[])
    }

    pub fn finetune(
        &self,
        train: &[String],
        validation: &[String],
        test: &[String],
        max_depth: Option<u64>,
        suffix: Option<&str>,
    ) -> Result<Value, PlatformError> {
        let max_depth = match max_depth {
            Some(d) => d,
            None => self.model(BASE_MODEL)?["depth"].as_u64().unwrap_or(20),
        };
        let mut body = json!({"model": BASE_MODEL, "training_files": train, "validation_files": validation, "test_files": test, "max_depth": max_depth});
        if let Some(s) = suffix {
            body["suffix"] = json!(s);
        }
        self.call("POST", "fine_tuning/jobs", Some(body), &[])
    }

    pub fn job(&self, id: &str) -> Result<Value, PlatformError> {
        self.call("GET", &format!("fine_tuning/jobs/{id}"), None, &[])
    }

    pub fn jobs(&self) -> Result<Vec<Value>, PlatformError> {
        Ok(self.call("GET", "fine_tuning/jobs", None, &[("limit", Some("20".into()))])?["data"].as_array().cloned().unwrap_or_default())
    }

    /// Poll a job or generation until it ends; errors if it failed.
    pub fn wait(&self, job: &Value, poll: Duration, mut on_update: impl FnMut(&Value)) -> Result<Value, PlatformError> {
        let id = job["id"].as_str().unwrap_or_default().to_string();
        let is_gen = job["object"] == "generation";
        loop {
            let rec = if is_gen { self.generation(&id)? } else { self.job(&id)? };
            on_update(&rec);
            let status = rec["status"].as_str().unwrap_or("").to_string();
            if TERMINAL.contains(&status.as_str()) {
                if status == "succeeded" {
                    return Ok(rec);
                }
                let e = &rec["error"];
                return Err(perr(
                    e["code"].as_str().unwrap_or(&status),
                    e["message"].as_str().map(str::to_string).unwrap_or(format!("job {id} {status}")),
                    0,
                ));
            }
            std::thread::sleep(poll);
        }
    }

    pub fn models(&self) -> Result<Vec<Value>, PlatformError> {
        Ok(self.call("GET", "models", None, &[("limit", Some("100".into()))])?["data"].as_array().cloned().unwrap_or_default())
    }

    pub fn model(&self, id: &str) -> Result<Value, PlatformError> {
        self.call("GET", &format!("models/{id}"), None, &[])
    }

    /// A model's `.cact` files into `out`: every size, or one `depth`.
    pub fn download(&self, model_id: &str, out: &Path, depth: Option<u64>) -> Result<Vec<PathBuf>> {
        let rec = self.model(model_id)?;
        let stem = rec["name"].as_str().filter(|s| !s.is_empty()).or(rec["id"].as_str()).unwrap_or(model_id).to_string();
        let mut variants = rec["variants"].as_array().cloned().unwrap_or_else(|| vec![json!({"id": rec["id"], "depth": rec["depth"]})]);
        if let Some(d) = depth {
            variants.retain(|v| v["depth"].as_u64() == Some(d));
            if variants.is_empty() {
                return Err(perr("not_found", format!("{model_id} has no {d}-layer size"), 404).into());
            }
        }
        let mut paths = vec![];
        for v in variants {
            let dest = out.join(format!("{stem}-{}L.cact", v["depth"]));
            let link = self.signed_link(&format!("models/{}/content", v["id"].as_str().unwrap_or_default()))?;
            paths.push(self.fetch_to(&link, &dest)?);
        }
        Ok(paths)
    }
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn line(label: &str, value: impl std::fmt::Display) {
    println!("  {label:<9} {value}");
}

fn print_usage(client: &Platform) -> Result<()> {
    let b = client.billing()?;
    let (limits, usage) = (&b["limits"], &b["usage"]);
    let num = |v: &Value, d: &str| v.as_u64().map(|n| n.to_string()).unwrap_or_else(|| d.into());
    line(
        "plan",
        format!(
            "{}: {}/{} fine-tunes, {}/{} generated examples this period",
            b["plan"].as_str().unwrap_or("None"),
            num(&usage["jobs"], "0"),
            num(&limits["jobs"], "?"),
            num(&usage["examples"], "0"),
            num(&limits["examples"], "?")
        ),
    );
    Ok(())
}

fn print_job(job: &Value) {
    let mut l = format!("{}  {}", job["id"].as_str().unwrap_or(""), job["status"].as_str().unwrap_or("None"));
    if let Some(n) = job["name"].as_str().filter(|s| !s.is_empty()) {
        l += &format!("  {n}");
    }
    if let Some(d) = job["max_depth"].as_u64() {
        l += &format!("  max_depth {d}");
    }
    if job["error"].is_object() {
        l += &format!("  error {}", job["error"]["code"].as_str().unwrap_or("None"));
    }
    println!("  {l}");
}

fn print_evaluations(job: &Value) {
    let Some(rows) = job["evaluations"].as_array().filter(|r| !r.is_empty()) else { return };
    println!("  {:>5}  {:>12}  {:>12}", "depth", "validation", "test");
    let mut rows = rows.clone();
    rows.sort_by_key(|r| std::cmp::Reverse(r["depth"].as_u64().unwrap_or(0)));
    for r in rows {
        let (v, t) = (&r["validation"], &r["test"]);
        println!("  {:>4}L  {:>5}/{:<6}  {:>5}/{:<6}", r["depth"], v["correct"], v["total"], t["correct"], t["total"]);
    }
}

fn progress() -> impl FnMut(&Value) {
    let mut seen = std::collections::HashSet::new();
    move |rec: &Value| {
        let key = (rec["status"].to_string(), rec["completed_steps"].to_string());
        if !seen.insert(key) {
            return;
        }
        let status = rec["status"].as_str().unwrap_or("None");
        match rec["total_steps"].as_u64() {
            Some(total) if status == "running" => {
                line("job", format!("running, step {}/{total}", rec["completed_steps"].as_u64().unwrap_or(0)))
            }
            _ => line("job", status),
        }
    }
}

pub enum Verb {
    Finetune {
        train: String,
        validation: String,
        test: String,
        max_depth: Option<u64>,
        suffix: Option<String>,
        out: PathBuf,
        depth: Option<u64>,
        no_wait: bool,
    },
    Generate {
        tools: PathBuf,
        examples: usize,
        description: Option<String>,
        message: Option<Vec<String>>,
        suffix: Option<String>,
        out: PathBuf,
        no_wait: bool,
    },
    Jobs {
        job_id: Option<String>,
        wait: bool,
        out: Option<PathBuf>,
        depth: Option<u64>,
    },
    Models {
        model_id: Option<String>,
    },
    Files,
    Billing,
}

pub fn main(verb: Verb) -> Result<()> {
    let client = Platform::new(None);
    let poll = Duration::from_secs(10);
    let res: Result<()> = (|| {
        match verb {
            Verb::Finetune { train, validation, test, max_depth, suffix, out, depth, no_wait } => {
                print_usage(&client)?;
                let mut ids = vec![];
                for item in [&train, &validation, &test] {
                    if item.starts_with("file-") {
                        ids.push(item.clone());
                    } else {
                        let rec = client.upload(Path::new(item))?;
                        line(
                            "upload",
                            format!(
                                "{}  {}  {:.2} MB",
                                rec["id"].as_str().unwrap_or(""),
                                rec["filename"].as_str().unwrap_or(""),
                                rec["bytes"].as_f64().unwrap_or(0.0) / 1e6
                            ),
                        );
                        ids.push(rec["id"].as_str().unwrap_or("").to_string());
                    }
                }
                let job = client.finetune(&ids[..1], &ids[1..2], &ids[2..3], max_depth, suffix.as_deref())?;
                let id = job["id"].as_str().unwrap_or("").to_string();
                line("job", format!("{id}  max_depth {}", job["max_depth"]));
                line(
                    "watch",
                    format!("{JOBS_URL}/{id}  (Ctrl-C leaves the job running; needle platform jobs {id} --wait --out DIR resumes)"),
                );
                if no_wait {
                    return Ok(());
                }
                let job = client.wait(&job, poll, progress())?;
                print_evaluations(&job);
                let paths = client.download(job["fine_tuned_model"].as_str().unwrap_or(""), &out, depth)?;
                for p in &paths {
                    line("weights", format!("{}  {:.2} MB", p.display(), crate::net::megabytes(p)));
                }
                if let Some(p) = paths.last() {
                    line("next", format!("needle.Needle(weights={:?}, tools=[...], auto_date=False)", p.display().to_string()));
                }
            }
            Verb::Generate { tools, examples, description, message, suffix, out, no_wait } => {
                print_usage(&client)?;
                let tools: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(&tools)?)?;
                let job = client.generate(&tools, examples, description.as_deref(), message.as_deref(), suffix.as_deref())?;
                let id = job["id"].as_str().unwrap_or("").to_string();
                line("job", format!("{id}  {examples} examples"));
                line(
                    "watch",
                    format!("{JOBS_URL}/{id}  (Ctrl-C leaves the job running; needle platform jobs {id} --wait --out DIR resumes)"),
                );
                if no_wait {
                    return Ok(());
                }
                let job = client.wait(&job, poll, progress())?;
                for rec in job["files"].as_array().into_iter().flatten() {
                    let dest = client
                        .download_file(rec["id"].as_str().unwrap_or(""), &out.join(rec["filename"].as_str().unwrap_or("data.jsonl")))?;
                    line(rec["group"].as_str().unwrap_or("file"), format!("{}  {:.2} MB", dest.display(), crate::net::megabytes(&dest)));
                }
                line("next", "needle platform finetune <train.jsonl> <validation.jsonl> <test.jsonl>");
            }
            Verb::Jobs { job_id, wait, out, depth } => {
                let Some(id) = job_id else {
                    for j in client.jobs()? {
                        print_job(&j);
                    }
                    return Ok(());
                };
                let mut job = match client.job(&id) {
                    Ok(j) => j,
                    Err(e) if e.status == 404 => client.generation(&id)?,
                    Err(e) => return Err(e.into()),
                };
                if wait && !TERMINAL.contains(&job["status"].as_str().unwrap_or("")) {
                    job = client.wait(&job, poll, progress())?;
                }
                print_job(&job);
                print_evaluations(&job);
                if let Some(model) = job["fine_tuned_model"].as_str() {
                    line("model", model);
                    if let Some(out) = out {
                        for p in client.download(model, &out, depth)? {
                            line("weights", format!("{}  {:.2} MB", p.display(), crate::net::megabytes(&p)));
                        }
                    }
                    return Ok(());
                }
                for rec in job["files"].as_array().into_iter().flatten() {
                    let group = rec["group"].as_str().unwrap_or("file");
                    match &out {
                        Some(out) if job["object"] == "generation" && job["status"] == "succeeded" => {
                            let dest = client.download_file(
                                rec["id"].as_str().unwrap_or(""),
                                &out.join(rec["filename"].as_str().unwrap_or("data.jsonl")),
                            )?;
                            line(group, format!("{}  {:.2} MB", dest.display(), crate::net::megabytes(&dest)));
                        }
                        _ => line(group, format!("{}  {}", rec["id"].as_str().unwrap_or(""), rec["filename"].as_str().unwrap_or(""))),
                    }
                }
            }
            Verb::Models { model_id } => match model_id {
                Some(id) => {
                    let rec = client.model(&id)?;
                    line(
                        "model",
                        format!("{}  {}  {} layers", rec["id"].as_str().unwrap_or(""), rec["name"].as_str().unwrap_or(""), rec["depth"]),
                    );
                    for v in rec["variants"].as_array().into_iter().flatten() {
                        let size = v["bytes"].as_f64().map(|b| format!("  {:.2} MB", b / 1e6)).unwrap_or_default();
                        line(&format!("{}L", v["depth"]), format!("{}{size}", v["id"].as_str().unwrap_or("")));
                    }
                    line("next", format!("needle download {} [--depth N]", rec["id"].as_str().unwrap_or("")));
                }
                None => {
                    for rec in client.models()? {
                        line(
                            "model",
                            format!(
                                "{}  {}  {} layers",
                                rec["id"].as_str().unwrap_or(""),
                                rec["name"].as_str().unwrap_or(""),
                                rec["depth"]
                            ),
                        );
                    }
                }
            },
            Verb::Files => {
                for rec in client.files()? {
                    line(
                        "file",
                        format!(
                            "{}  {}  {:.2} MB",
                            rec["id"].as_str().unwrap_or(""),
                            rec["filename"].as_str().unwrap_or(""),
                            rec["bytes"].as_f64().unwrap_or(0.0) / 1e6
                        ),
                    );
                }
            }
            Verb::Billing => {
                let b = client.billing()?;
                line("plan", b["plan"].as_str().unwrap_or("None"));
                if let Some(limits) = b["limits"].as_object() {
                    for (k, limit) in limits {
                        line(k, format!("{} / {limit}", b["usage"][k].as_u64().unwrap_or(0)));
                    }
                }
                if let Some(end) = b["period_end"].as_i64() {
                    let dt = chrono::DateTime::from_timestamp(end, 0).map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default();
                    line("resets", dt);
                }
            }
        }
        Ok(())
    })();
    res.map_err(|e| match e.downcast::<PlatformError>() {
        Ok(pe) => anyhow::anyhow!("platform error {pe}"),
        Err(e) => e,
    })
}
