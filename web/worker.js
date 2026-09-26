// The engine lives here, off the page's thread: it fetches the model once
// (then serves it from Cache Storage), picks the fastest exact build this
// browser can run, and answers one request at a time.

const MODEL_URL = "https://huggingface.co/Cactus-Compute/needle3/resolve/main/needle3.cact";
const MODEL_CACHE = "needle-rs-model-v1";

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

async function fetchModel(post) {
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
