// Cactus Compute's own browser engine (needle.wasm, Apache-2.0, from their
// Hugging Face repo, pinned to the revision benchmarked in the README), run
// through its C API so the page can race it against needle-rs. A classic
// worker: the Emscripten loader is a plain script.

const REV = "b274efcb211a9eef48c9a88da4b43bd569696a39";
const BASE = `https://huggingface.co/Cactus-Compute/needle3/resolve/${REV}/wasm/`;
// The same pinned weights the page's own worker caches.
const MODEL_URL = `https://huggingface.co/Cactus-Compute/needle3/resolve/${REV}/needle3.cact`;
const MODEL_CACHE = "needle-rs-model-v2";

let M = null;
let out = 0;
const CAP = 1 << 16;

async function modelBytes() {
  const cache = await caches.open(MODEL_CACHE).catch(() => null);
  const hit = cache && (await cache.match(MODEL_URL));
  const res = hit || (await fetch(MODEL_URL));
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function fetchOk(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

async function load() {
  const t0 = performance.now();
  // Hugging Face serves the loader without a JavaScript MIME type, which
  // importScripts refuses; run it from a Blob instead, and hand the module
  // its wasm bytes directly.
  const [js, wasm] = await Promise.all([fetchOk(`${BASE}needle.js`).then((r) => r.text()), fetchOk(`${BASE}needle.wasm`).then((r) => r.arrayBuffer())]);
  importScripts(URL.createObjectURL(new Blob([js], { type: "text/javascript" })));
  // eslint-disable-next-line no-undef
  M = await createNeedle({ wasmBinary: wasm, locateFile: (p) => `${BASE}${p}` });
  const bytes = await modelBytes();
  const t1 = performance.now();
  const p = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, p);
  const rc = M.ccall("needle_load", "number", ["number", "bigint"], [p, BigInt(bytes.length)]);
  M._free(p);
  if (rc !== 0) throw new Error(`needle_load failed (${rc})`);
  out = M._malloc(CAP);
  return { fetchMs: t1 - t0, loadMs: performance.now() - t1 };
}

function init(system, tools) {
  const t = performance.now();
  const n = M.ccall("needle_init", "number", ["string", "string", "number"], [system, tools, 0]);
  if (n < 0) throw new Error("needle_init failed");
  return { prefixTokens: n, prefixMs: performance.now() - t };
}

function run(input) {
  M.ccall("needle_reset", null, [], []);
  const t = performance.now();
  M.ccall("needle_complete", "number", ["string", "number", "number", "number"], [input, 512, out, CAP]);
  const turnMs = performance.now() - t;
  return { envelope: JSON.parse(M.UTF8ToString(out)), turnMs };
}

let queue = Promise.resolve();
self.onmessage = (e) => {
  const msg = e.data;
  const post = (m) => self.postMessage({ id: msg.id, ...m });
  queue = queue.then(async () => {
    try {
      if (msg.type === "load") post({ type: "ready", ...(await load()) });
      else if (msg.type === "init") post({ type: "inited", ...init(msg.system, msg.tools) });
      else if (msg.type === "run") post({ type: "result", ...run(msg.input) });
    } catch (err) {
      post({ type: "error", message: String(err && err.message ? err.message : err) });
    }
  });
};
