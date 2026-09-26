//! Opt-in wall-clock profiling (`NEEDLE_PROFILE=1`): named spans summed
//! across calls, printed on demand.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::OnceLock;
use web_time::Instant;

static TOTALS: Mutex<BTreeMap<&'static str, (f64, usize)>> = Mutex::new(BTreeMap::new());

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("NEEDLE_PROFILE").is_some())
}

/// Time `f` under `name` when profiling is on.
#[inline]
pub fn span<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let t = Instant::now();
    let out = f();
    let dt = t.elapsed().as_secs_f64();
    let mut m = TOTALS.lock().unwrap();
    let e = m.entry(name).or_insert((0.0, 0));
    e.0 += dt;
    e.1 += 1;
    out
}

pub fn report() -> String {
    let m = TOTALS.lock().unwrap();
    let total: f64 = m.values().map(|v| v.0).sum();
    let mut rows: Vec<_> = m.iter().collect();
    rows.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let mut s = String::new();
    for (k, (t, n)) in rows {
        s += &format!("  {k:<22} {t:>8.3}s {:>5.1}%  x{n}\n", 100.0 * t / total.max(1e-12));
    }
    s
}

pub fn reset() {
    TOTALS.lock().unwrap().clear();
}

/// A span timed from creation to drop (when profiling is on).
pub struct Span(&'static str, Option<Instant>);

impl Span {
    pub fn new(name: &'static str) -> Self {
        Self(name, enabled().then(Instant::now))
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(t) = self.1 {
            let dt = t.elapsed().as_secs_f64();
            let mut m = TOTALS.lock().unwrap();
            let e = m.entry(self.0).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
        }
    }
}

/// Add `secs` to span `name` (when profiling is on).
pub fn add(name: &'static str, secs: f64) {
    if !enabled() {
        return;
    }
    let mut m = TOTALS.lock().unwrap();
    let e = m.entry(name).or_insert((0.0, 0));
    e.0 += secs;
    e.1 += 1;
}
