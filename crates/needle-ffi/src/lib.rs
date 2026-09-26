//! The engine's C API, as declared in the published `needle.h`.
//!
//! One process-global model, like the reference engine: `needle_load` maps
//! an archive, `needle_init` binds a toolset (returning the static-prefix
//! token count), `needle_complete` runs a turn into a caller buffer,
//! `needle_embed` returns the retrieval vector, `needle_reset` rewinds the
//! conversation. Negative returns are failures; `needle_last_error` says why.

use std::ffi::{CStr, CString, c_char, c_float, c_int, c_uchar, c_ulonglong};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Context, Result, anyhow};
use needle_core::Tokenizer;
use needle_core::cact::Archive;
use needle_engine::agent::Agent;
use needle_engine::{Model, Weights};

struct Engine {
    model: Option<(Arc<Model>, Arc<Tokenizer>)>,
    agent: Option<Agent>,
    last_error: CString,
}

static ENGINE: LazyLock<Mutex<Engine>> = LazyLock::new(|| Mutex::new(Engine { model: None, agent: None, last_error: CString::default() }));

fn engine() -> std::sync::MutexGuard<'static, Engine> {
    ENGINE.lock().unwrap_or_else(|p| p.into_inner())
}

fn fail(e: &mut Engine, err: anyhow::Error, code: c_int) -> c_int {
    e.last_error = CString::new(format!("{err:#}").replace('\0', " ")).unwrap_or_default();
    code
}

/// # Safety
/// `p` is null or a valid NUL-terminated string.
unsafe fn cstr<'a>(p: *const c_char) -> Result<Option<&'a str>> {
    if p.is_null() {
        return Ok(None);
    }
    // SAFETY: the caller promises a NUL-terminated string.
    Ok(Some(unsafe { CStr::from_ptr(p) }.to_str().context("input is not UTF-8")?))
}

fn load_archive(archive: Archive) -> Result<(Arc<Model>, Arc<Tokenizer>)> {
    let tok = Tokenizer::from_blob(archive.tokenizer_blob()?)?;
    let (weights, q) = Weights::from_archive_mode(&archive, std::env::var_os("NEEDLE_F32").is_none())?;
    let model = match q {
        Some(q) => Model::quantized(weights, q),
        None => Model::new(weights),
    };
    drop(archive);
    needle_engine::cpu::release_cached_memory();
    Ok((Arc::new(model), Arc::new(tok)))
}

/// # Safety
/// `cact` points at `n` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn needle_load(cact: *const c_uchar, n: c_ulonglong) -> c_int {
    let mut e = engine();
    if cact.is_null() {
        return fail(&mut e, anyhow!("needle_load: null archive"), -1);
    }
    // SAFETY: the caller promises `n` readable bytes at `cact` for this call;
    // every weight is repacked into the model's own buffers and the archive
    // is dropped before we return.
    let loaded = unsafe { Archive::from_raw(cact, n as usize) }.and_then(load_archive);
    match loaded {
        Ok(m) => {
            e.model = Some(m);
            e.agent = None;
            0
        }
        Err(err) => fail(&mut e, err, -1),
    }
}

/// # Safety
/// Each argument is null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn needle_init(system_prompt: *const c_char, tools_json: *const c_char, tool_index_path: *const c_char) -> c_int {
    let mut e = engine();
    let run = |e: &mut Engine| -> Result<c_int> {
        // SAFETY: forwarded caller guarantees.
        let system = unsafe { cstr(system_prompt) }?.unwrap_or("");
        let tools = unsafe { cstr(tools_json) }?.unwrap_or("");
        let _index = unsafe { cstr(tool_index_path) }?;
        let (model, tok) = e.model.clone().context("no model loaded: call needle_load first")?;
        let agent = Agent::from_json(model, tok, tools.as_bytes(), system)?;
        let prefix = agent.prefix_tokens() as c_int;
        e.agent = Some(agent);
        Ok(prefix)
    };
    match run(&mut e) {
        Ok(v) => v,
        Err(err) => fail(&mut e, err, -1),
    }
}

fn write_out(out: *mut c_char, cap: c_int, text: &str) -> Option<c_int> {
    let bytes = text.as_bytes();
    if out.is_null() || cap <= 0 || bytes.len() + 1 > cap as usize {
        return None;
    }
    // SAFETY: `out` has `cap` writable bytes and we write len + 1 <= cap.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, bytes.len());
        *out.add(bytes.len()) = 0;
    }
    Some(bytes.len() as c_int)
}

/// # Safety
/// `input` is a NUL-terminated string; `out` has `out_capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn needle_complete(input: *const c_char, max_new_tokens: c_int, out: *mut c_char, out_capacity: c_int) -> c_int {
    let mut e = engine();
    let run = |e: &mut Engine| -> Result<String> {
        // SAFETY: forwarded caller guarantees.
        let text = unsafe { cstr(input) }?.unwrap_or("");
        let agent = e.agent.as_mut().context("no toolset bound: call needle_init first")?;
        // This thread leads the decode team for the turn.
        let qos = needle_engine::cpu::prefer_performance_cores();
        let out = agent.complete(text, max_new_tokens.max(1) as usize);
        needle_engine::cpu::restore_qos(qos);
        Ok(needle_engine::agent::envelope_json(&out?))
    };
    match run(&mut e) {
        Ok(json) => match write_out(out, out_capacity, &json) {
            Some(n) => n,
            None => fail(&mut e, anyhow!("output buffer of {out_capacity} bytes is too small"), -2),
        },
        Err(err) => {
            let code = fail(&mut e, err, -1);
            let msg = e.last_error.to_string_lossy().into_owned();
            let _ = write_out(out, out_capacity, &msg);
            code
        }
    }
}

/// # Safety
/// `input` is a NUL-terminated string; `out` is null or has
/// `out_capacity` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn needle_embed(input: *const c_char, out: *mut c_float, out_capacity: c_int) -> c_int {
    let mut e = engine();
    let run = |e: &mut Engine| -> Result<c_int> {
        let (model, _) = e.model.as_ref().context("no model loaded")?;
        let c = model.config();
        let dim = if model.w.embedding_head.is_some() { c.embedding_dim } else { c.confidence_queries * c.d_model };
        if out.is_null() {
            return Ok(dim as c_int);
        }
        // SAFETY: forwarded caller guarantees.
        let text = unsafe { cstr(input) }?.unwrap_or("");
        let agent = e.agent.as_ref().context("no toolset bound: call needle_init first")?;
        let v = agent.embed(text)?;
        if (out_capacity as usize) < v.len() {
            anyhow::bail!("embedding needs {} floats, buffer holds {out_capacity}", v.len());
        }
        // SAFETY: `out` holds at least `v.len()` floats (checked above).
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len()) };
        Ok(v.len() as c_int)
    };
    match run(&mut e) {
        Ok(n) => n,
        Err(err) => fail(&mut e, err, -1),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn needle_reset() {
    if let Some(a) = engine().agent.as_mut() {
        a.reset();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn needle_last_error() -> *const c_char {
    engine().last_error.as_ptr()
}
