//! Needle 3 in the browser.
//!
//! One [`Needle`] holds the model (loaded from the bytes of a `.cact`
//! archive) and an agent per system text and tools. It runs on the calling
//! thread, so call it from a Web Worker to keep the page live.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use needle_core::cact::Archive;
use needle_core::tokenizer::Tokenizer;
use needle_engine::agent::{Agent, envelope_json};
use needle_engine::model::Model;
use needle_engine::weights::Weights;
use wasm_bindgen::prelude::*;

/// The model and the agents started on it, one per system text and tools
/// (the page switches between them without reading the tools again).
#[wasm_bindgen]
pub struct Needle {
    model: Arc<Model>,
    tok: Arc<Tokenizer>,
    agents: HashMap<String, Agent>,
    current: Option<String>,
}

fn js(e: anyhow::Error) -> JsError {
    JsError::new(&format!("{e:#}"))
}

#[wasm_bindgen]
impl Needle {
    /// Load a `.cact` archive (its bytes; the archive is dropped once the
    /// weights are repacked).
    #[wasm_bindgen(constructor)]
    pub fn new(cact: Vec<u8>) -> Result<Needle, JsError> {
        let load = || -> Result<Needle> {
            let archive = Archive::from_bytes(cact)?;
            let tok = Tokenizer::from_blob(archive.tokenizer_blob()?)?;
            let (weights, q) = Weights::from_archive_mode(&archive, true)?;
            let model = match q {
                Some(q) => Model::quantized(weights, q),
                None => Model::new(weights),
            };
            Ok(Needle { model: Arc::new(model), tok: Arc::new(tok), agents: HashMap::new(), current: None })
        };
        load().map_err(js)
    }

    /// Switch to the agent for `key`, starting it on the system text and
    /// tools (a JSON list of OpenAI-style function schemas) the first time.
    /// Returns the prompt prefix length in tokens, or 0 when the agent
    /// already existed.
    pub fn init(&mut self, key: &str, system: &str, tools_json: &str) -> Result<usize, JsError> {
        let mut started = 0;
        if !self.agents.contains_key(key) {
            let agent = Agent::from_json(self.model.clone(), self.tok.clone(), tools_json.as_bytes(), system).map_err(js)?;
            started = agent.prefix_tokens();
            self.agents.insert(key.to_string(), agent);
        }
        self.current = Some(key.to_string());
        Ok(started)
    }

    fn agent(&mut self) -> Result<&mut Agent, JsError> {
        let key = self.current.as_ref().context("call init first").map_err(js)?;
        self.agents.get_mut(key).context("no agent for the current key").map_err(js)
    }

    /// One turn: a user message, or a tool result (JSON) after a call.
    /// Returns the result envelope as JSON.
    pub fn complete(&mut self, input: &str, max_new_tokens: usize) -> Result<String, JsError> {
        self.agent()?.complete(input, max_new_tokens).map(|v| envelope_json(&v)).map_err(js)
    }

    /// A typed decision: the turn must call a tool, and every enum argument
    /// it fills comes back with the probability of each option
    /// (`decisions` in the envelope).
    pub fn decide(&mut self, input: &str, max_new_tokens: usize) -> Result<String, JsError> {
        self.agent()?.decide(input, max_new_tokens).map(|v| envelope_json(&v)).map_err(js)
    }

    /// Forget the current agent's conversation, keeping its tools.
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.agent()?.reset();
        Ok(())
    }
}

/// Check the SIMD arithmetic against its scalar definition (a report).
#[doc(hidden)]
#[wasm_bindgen(js_name = simdSelftest)]
pub fn simd_selftest(n: usize) -> String {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return needle_engine::infer::wasm_simd_selftest(n);
    #[allow(unreachable_code)]
    {
        let _ = n;
        "no SIMD in this build".into()
    }
}

/// Which arithmetic this build uses: `"relaxed"` (the hardware's fused
/// multiply-add, when [`madd_fused`] holds) or `"exact"` (fused multiply-add
/// computed exactly in software).
#[wasm_bindgen(js_name = simdMode)]
pub fn simd_mode() -> String {
    if cfg!(target_feature = "relaxed-simd") { "relaxed".into() } else { "exact".into() }
}

/// Whether this build's arithmetic is exact in this browser: always for the
/// exact build; for the relaxed build, whether the browser fuses
/// `relaxed_madd` (it may not, on a CPU without FMA).
#[wasm_bindgen(js_name = maddFused)]
pub fn madd_fused() -> bool {
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    return needle_engine::infer::wsimd::madd_is_fused();
    #[allow(unreachable_code)]
    true
}
