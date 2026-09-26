//! `needle playground` (the browser UI of the reference, served locally) and
//! `needle serve` (the native runner's HTTP mode).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use needle_engine::agent::Agent;
use needle_engine::loader;
use serde_json::{Value, json};
use tiny_http::{Header, Method, Request, Response, Server};

const INDEX: &str = include_str!("../playground/index.html");
const APP_JS: &str = include_str!("../playground/app.js");
const STYLE: &str = include_str!("../playground/style.css");

struct Engine {
    weights: PathBuf,
    name: String,
    model: Option<(Arc<needle_engine::Model>, Arc<needle_core::Tokenizer>)>,
    tools_json: Option<String>,
    agent: Option<Agent>,
}

impl Engine {
    fn load(&mut self) -> Result<()> {
        let l = loader::load(&self.weights)?;
        self.model = Some((Arc::new(l.model), Arc::new(l.tokenizer)));
        self.agent = None;
        self.tools_json = None;
        Ok(())
    }

    fn complete(&mut self, tools_json: &str, query: &str) -> Result<Value> {
        if self.model.is_none() {
            self.load()?;
        }
        if self.agent.is_none() || self.tools_json.as_deref() != Some(tools_json) {
            let (m, t) = self.model.clone().unwrap();
            let tools: Vec<Value> = serde_json::from_str(tools_json)?;
            let system = needle_engine::agent::with_date_fact("");
            self.agent = Some(Agent::new(m, t, tools, &system)?);
            self.tools_json = Some(tools_json.to_string());
        } else if let Some(a) = self.agent.as_mut() {
            a.reset();
        }
        self.agent.as_mut().unwrap().complete(query, 512)
    }
}

#[derive(Default)]
struct FtState {
    running: bool,
    step: String,
    log: Vec<String>,
    checkpoint: Option<String>,
    error: Option<String>,
}

impl FtState {
    fn to_json(&self) -> Value {
        json!({"running": self.running, "step": self.step, "log": self.log, "checkpoint": self.checkpoint, "error": self.error})
    }

    fn log(&mut self, m: impl Into<String>) {
        self.log.push(m.into());
        if self.log.len() > 100 {
            let drop = self.log.len() - 100;
            self.log.drain(..drop);
        }
    }
}

fn respond(req: Request, code: u16, body: Vec<u8>, ctype: &str) {
    let header = Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes()).unwrap();
    let _ = req.respond(Response::from_data(body).with_status_code(code).with_header(header));
}

fn respond_json(req: Request, v: &Value) {
    respond(req, 200, v.to_string().into_bytes(), "application/json");
}

fn body_json(req: &mut Request) -> Value {
    let mut s = String::new();
    let _ = req.as_reader().read_to_string(&mut s);
    serde_json::from_str(&s).unwrap_or(json!({}))
}

fn tools_json(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "[]".into(),
        other => other.to_string(),
    }
}

pub fn playground(weights: Option<PathBuf>, host: &str, port: u16, downloads: PathBuf) -> Result<()> {
    std::fs::create_dir_all(&downloads)?;
    let weights = match weights {
        Some(w) => w,
        None => crate::fetch::fetch_weights(3, None, false)?,
    };
    println!("needle playground: downloading and initializing the model...");
    let name = weights.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut engine = Engine { weights, name, model: None, tools_json: None, agent: None };
    engine.load()?;
    let engine = Arc::new(Mutex::new(engine));
    let ft = Arc::new(Mutex::new(FtState::default()));
    let server = Server::http(format!("{host}:{port}")).map_err(|e| anyhow::anyhow!("bind {host}:{port}: {e}"))?;
    println!("needle playground ready: http://{host}:{port}  ({})", engine.lock().unwrap().name);
    for mut req in server.incoming_requests() {
        let path = req.url().split('?').next().unwrap_or("/").to_string();
        match (req.method().clone(), path.as_str()) {
            (Method::Get, "/" | "/index.html") => respond(req, 200, INDEX.into(), "text/html; charset=utf-8"),
            (Method::Get, "/app.js") => respond(req, 200, APP_JS.into(), "application/javascript; charset=utf-8"),
            (Method::Get, "/style.css") => respond(req, 200, STYLE.into(), "text/css; charset=utf-8"),
            (Method::Get, "/model") => {
                let n = engine.lock().unwrap().name.clone();
                respond_json(req, &json!({"name": n}));
            }
            (Method::Get, "/finetune/status") => {
                let s = ft.lock().unwrap().to_json();
                respond_json(req, &s);
            }
            (Method::Get, p) if p.starts_with("/download/") => {
                let name = Path::new(&p["/download/".len()..]).file_name().map(|n| n.to_owned());
                match name.map(|n| downloads.join(n)).filter(|f| f.exists()) {
                    Some(f) => respond(req, 200, std::fs::read(f).unwrap_or_default(), "application/octet-stream"),
                    None => respond(req, 404, b"not found".to_vec(), "text/plain"),
                }
            }
            (Method::Post, "/complete") => {
                let body = body_json(&mut req);
                let res = engine.lock().unwrap().complete(&tools_json(&body["tools"]), body["query"].as_str().unwrap_or(""));
                respond_json(req, &res.unwrap_or_else(|e| json!({"error": e.to_string()})));
            }
            (Method::Post, "/reset") => {
                let mut e = engine.lock().unwrap();
                e.agent = None;
                e.tools_json = None;
                drop(e);
                respond_json(req, &json!({"ok": true}));
            }
            (Method::Post, "/load-model") => {
                let name = req
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("X-Filename"))
                    .map(|h| h.value.to_string())
                    .and_then(|v| Path::new(&v).file_name().map(|n| n.to_string_lossy().into_owned()))
                    .unwrap_or_else(|| "model.cact".into());
                let mut bytes = vec![];
                let _ = req.as_reader().read_to_end(&mut bytes);
                let dest = downloads.join(&name);
                let res = std::fs::write(&dest, bytes).map_err(anyhow::Error::from).and_then(|_| {
                    let mut e = engine.lock().unwrap();
                    e.weights = dest.clone();
                    e.name = name.clone();
                    e.load()
                });
                match res {
                    Ok(()) => respond_json(req, &json!({"name": name})),
                    Err(e) => respond_json(req, &json!({"error": e.to_string()})),
                }
            }
            (Method::Post, "/finetune") => {
                if ft.lock().unwrap().running {
                    respond_json(req, &json!({"error": "a finetune is already running"}));
                    continue;
                }
                let body = body_json(&mut req);
                let key = body["api_key"].as_str().unwrap_or("").trim().to_string();
                if key.is_empty() {
                    respond_json(req, &json!({"error": "OpenRouter API key is required"}));
                    continue;
                }
                let tools = tools_json(&body["tools"]);
                let samples = body["samples"].as_u64().unwrap_or(200) as usize;
                let (ft2, engine2, dl) = (ft.clone(), engine.clone(), downloads.clone());
                std::thread::spawn(move || finetune_worker(tools, key, samples, engine2, ft2, dl));
                respond_json(req, &json!({"ok": true}));
            }
            _ => respond(req, 404, b"not found".to_vec(), "text/plain"),
        }
    }
    Ok(())
}

fn finetune_worker(tools: String, key: String, samples: usize, engine: Arc<Mutex<Engine>>, ft: Arc<Mutex<FtState>>, dl: PathBuf) {
    let run = || -> Result<PathBuf> {
        {
            let mut s = ft.lock().unwrap();
            *s = FtState { running: true, step: "generating data".into(), ..Default::default() };
        }
        let tools_v: Value = serde_json::from_str(&tools)?;
        let ftc = ft.clone();
        let rows =
            crate::generate::generate_dataset(&tools_v, samples, crate::generate::DEFAULT_MODEL, 25, Some(&key), 8, move |d, t, _| {
                ftc.lock().unwrap().log(format!("generated {d}/{t}"));
            })?;
        let data = dl.join("needle_playground_data.jsonl");
        let text: String = rows
            .iter()
            .map(|r| format!("{}\n", needle_core::pyjson::dumps(&Value::Object(r.clone()), needle_core::pyjson::DumpOpts::DEFAULT)))
            .collect();
        std::fs::write(&data, text)?;
        ft.lock().unwrap().step = "training".into();
        let adapter = dl.join("needle_playground_lora.safetensors");
        let base = PathBuf::from(needle_train::finetune::DEFAULT_BASE);
        let base = if base.exists() {
            base
        } else {
            crate::fetch::fetch_checkpoint("needle3.safetensors", Path::new(crate::fetch::CHECKPOINT_PREFIX), 3)?
        };
        let args = needle_train::finetune::FinetuneArgs {
            jsonl_path: data,
            checkpoint: Some(base.clone()),
            epochs: 3,
            batch_size: 16,
            lr: 1e-4,
            lora_rank: 16,
            lora_alpha: 32.0,
            max_len: 1024,
            val_split: 0.1,
            seed: 0,
            checkpoint_dir: dl.clone(),
            out: Some(adapter.clone()),
            score: true,
            max_steps: None,
        };
        let ftc = ft.clone();
        needle_train::finetune::finetune(&args, None, move |m| ftc.lock().unwrap().log(m))?;
        ft.lock().unwrap().step = "building".into();
        let out = dl.join("needle_tuned.cact");
        let base_archive = crate::fetch::fetch_weights(3, None, false)?;
        crate::build(Some(base), Some(adapter), Some(out.clone()), None, &base_archive)?;
        Ok(out)
    };
    match run() {
        Ok(out) => {
            let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            {
                let mut e = engine.lock().unwrap();
                e.weights = out;
                e.name = name.clone();
                let _ = e.load();
            }
            let mut s = ft.lock().unwrap();
            s.running = false;
            s.step = "done".into();
            s.checkpoint = Some(name);
        }
        Err(e) => {
            let mut s = ft.lock().unwrap();
            s.running = false;
            s.step = "failed".into();
            s.error = Some(e.to_string());
            s.log(e.to_string());
        }
    }
}

/// The native runner's `--serve`: POST /complete {"input"}, POST /reset.
pub fn serve(model: &Path, tools: Vec<Value>, system: &str, host: &str, port: u16, max: usize) -> Result<()> {
    let l = loader::load(model)?;
    let mut agent = Agent::new(Arc::new(l.model), Arc::new(l.tokenizer), tools, system)?;
    let server = Server::http(format!("{host}:{port}")).map_err(|e| anyhow::anyhow!("bind {host}:{port}: {e}"))?;
    eprintln!("needle serving on http://{host}:{port}  POST /complete {{\"input\": \"...\"}}, POST /reset");
    for mut req in server.incoming_requests() {
        match (req.method().clone(), req.url()) {
            (Method::Post, "/complete") => {
                let body = body_json(&mut req);
                let res = agent.complete(body["input"].as_str().unwrap_or(""), body["max_new_tokens"].as_u64().map_or(max, |v| v as usize));
                let res = res.map_or_else(
                    |e| json!({"success": false, "error": e.to_string()}).to_string(),
                    |env| needle_engine::agent::envelope_json(&env),
                );
                respond(req, 200, res.into_bytes(), "application/json");
            }
            (Method::Post, "/reset") => {
                agent.reset();
                respond_json(req, &json!({"ok": true}));
            }
            _ => respond(req, 404, b"not found".to_vec(), "text/plain"),
        }
    }
    Ok(())
}
