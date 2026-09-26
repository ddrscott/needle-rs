// The engine lives here, off the page's thread: it fetches the model once
// (then serves it from Cache Storage), picks the fastest exact build this
// browser can run, and answers one request at a time.

// Cactus Compute's published weights, pinned to the revision every number on
// the page and in the README was measured with, and checked on download.
const MODEL_REV = "b274efcb211a9eef48c9a88da4b43bd569696a39";
const MODEL_URL = `https://huggingface.co/Cactus-Compute/needle3/resolve/${MODEL_REV}/needle3.cact`;
const MODEL_SHA256 = "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38";
const MODEL_CACHE = "needle-rs-model-v2";

let needle = null;
let queue = Promise.resolve();

// The relaxed build uses the hardware's fused multiply-add. WebAssembly lets
// a browser run `relaxed_madd` unfused, which would change results, so it
// is used only when the check says this browser fuses it; otherwise the
// build that computes fused multiply-add exactly in software takes over.
async function loadEngine() {
  try {
    const m = await import("./pkg-relaxed/needle_wasm.js");
    await m.default();
    if (m.maddFused()) return { m, mode: "relaxed" };
  } catch (_) {
    // No relaxed SIMD in this browser.
  }
  const m = await import("./pkg/needle_wasm.js");
  await m.default();
  return { m, mode: "exact" };
}

async function sha256(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchModel(post) {
  // Drop copies cached under older names (the unpinned download).
  for (const name of await caches.keys().catch(() => [])) {
    if (name.startsWith("needle-rs-model") && name !== MODEL_CACHE) await caches.delete(name).catch(() => {});
  }
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  const hit = cache && (await cache.match(MODEL_URL));
  if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true };
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 35335380;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    post({ type: "progress", got, total });
  }
  const bytes = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    bytes.set(c, at);
    at += c.length;
  }
  if ((await sha256(bytes)) !== MODEL_SHA256) throw new Error("the downloaded model does not match the pinned revision");
  if (cache) {
    await cache
      .put(MODEL_URL, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }))
      .catch(() => {});
  }
  return { bytes, cached: false };
}

async function load(post) {
  const t0 = performance.now();
  const [{ m, mode }, model] = await Promise.all([loadEngine(), fetchModel(post)]);
  const t1 = performance.now();
  needle = new m.Needle(model.bytes);
  const t2 = performance.now();
  post({ type: "ready", mode, cached: model.cached, bytes: model.bytes.length, fetchMs: t1 - t0, loadMs: t2 - t1 });
}

function run({ key, system, tools, input, decide, maxTokens }) {
  const t0 = performance.now();
  const prefixTokens = needle.init(key, system, tools);
  const t1 = performance.now();
  needle.reset();
  const raw = decide ? needle.decide(input, maxTokens) : needle.complete(input, maxTokens);
  const t2 = performance.now();
  return { envelope: JSON.parse(raw), prefixTokens, prefixMs: prefixTokens ? t1 - t0 : 0, turnMs: t2 - t1 };
}

self.onmessage = (e) => {
  const msg = e.data;
  const post = (m) => self.postMessage({ id: msg.id, ...m });
  queue = queue.then(async () => {
    try {
      if (msg.type === "load") await load(post);
      else if (msg.type === "run") post({ type: "result", ...run(msg) });
    } catch (err) {
      post({ type: "error", message: String(err && err.message ? err.message : err) });
    }
  });
};
