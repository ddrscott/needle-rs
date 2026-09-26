// The page: example tabs, the command box, the scenes the calls act on,
// and the race against Cactus's engine. needle-rs runs in worker.js, Cactus's
// engine in race-worker.js.

import { sceneFor } from "./scenes.js";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

// // WORKER

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let nextId = 1;
const pending = new Map();

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") return onProgress(m);
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  if (m.type === "error") p.reject(new Error(m.message));
  else p.resolve(m);
};

function call(msg) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...msg });
  });
}

// // STATUS

function status(text, ready = false) {
  $("status-text").textContent = text;
  $("status").classList.toggle("ready", ready);
}

function onProgress({ got, total }) {
  $("meter").hidden = false;
  $("meter-fill").style.width = `${Math.min(100, (100 * got) / total).toFixed(1)}%`;
  status(`downloading the model from Hugging Face: ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`);
}

// // EXAMPLES

let examples = [];
let current = null;
let busy = false;
let engineReady = false;

const CUSTOM = { id: "custom", label: "CUSTOM", blurb: "Paste your own tools (OpenAI-style function schemas) and try commands against them.", queries: [] };

function signature(tool) {
  const f = tool.function || tool;
  const props = (f.parameters && f.parameters.properties) || {};
  const req = new Set((f.parameters && f.parameters.required) || []);
  const frag = document.createDocumentFragment();
  frag.append(el("span", "fn", f.name), "(");
  Object.entries(props).forEach(([name, p], i) => {
    if (i) frag.append(", ");
    frag.append(name + (req.has(name) ? "" : "?") + ": ");
    const ty = p.enum ? p.enum.join("|") : p.type || "any";
    frag.append(el("span", "ty", ty));
  });
  frag.append(")");
  return frag;
}

function showTools(tools) {
  const pre = $("tools");
  pre.replaceChildren();
  tools.forEach((t, i) => {
    if (i) pre.append("\n");
    pre.append(signature(t));
  });
}

function customTools() {
  try {
    const tools = JSON.parse($("custom-tools").value);
    if (!Array.isArray(tools)) throw new Error("expected a JSON list of tools");
    return { tools, error: null };
  } catch (e) {
    return { tools: null, error: e.message };
  }
}

const scenes = new Map();

function sceneOf(id) {
  if (!scenes.has(id)) scenes.set(id, sceneFor(id));
  return scenes.get(id);
}

function select(ex) {
  current = ex;
  sceneOf(ex.id).mount($("scene"));
  $("race-set").textContent = ex.id === "custom" ? "custom" : ex.title.toLowerCase();
  for (const b of $("tabs").children) b.setAttribute("aria-selected", String(b.dataset.id === ex.id));
  $("blurb").textContent = ex.blurb;
  $("custom").hidden = ex.id !== "custom";
  $("tools").hidden = ex.id === "custom";
  if (ex.id !== "custom") showTools(ex.tools);
  const chips = $("chips");
  chips.replaceChildren();
  for (const q of ex.queries) {
    const b = el("button", "", q);
    b.type = "button";
    b.onclick = () => {
      $("input").value = q;
      run();
    };
    chips.append(b);
  }
  if (!ex.queries.length) chips.append(el("span", "empty", "type a command below"));
}

function buildTabs() {
  const tabs = $("tabs");
  for (const ex of [...examples, CUSTOM]) {
    const b = el("button", "", ex.label);
    b.type = "button";
    b.role = "tab";
    b.dataset.id = ex.id;
    b.onclick = () => select(ex);
    tabs.append(b);
  }
}

// // RUN

function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(16);
}

async function run() {
  const input = $("input").value.trim();
  if (!input || busy || !engineReady) return;
  let tools = current.tools;
  let system = current.system || "";
  let key = current.id;
  if (current.id === "custom") {
    const c = customTools();
    if (c.error) return status(`custom tools: ${c.error}`, true);
    tools = c.tools;
    system = $("custom-system").value;
    key = `custom:${hash(system + JSON.stringify(tools))}`;
  }
  busy = true;
  $("run").disabled = true;
  status(`thinking about "${input}"`);
  try {
    const r = await call({ type: "run", key, system, tools: JSON.stringify(tools), input, decide: $("decide").checked, maxTokens: 256 });
    render(input, r);
    const setup = r.prefixTokens ? ` (read ${r.prefixTokens} tokens of tools first, ${Math.round(r.prefixMs)} ms)` : "";
    status(`done in ${Math.round(r.turnMs)} ms${setup}`, true);
  } catch (e) {
    status(`error: ${e.message}`, true);
  } finally {
    busy = false;
    $("run").disabled = false;
  }
}

// // RENDER

function jsonView(value) {
  const pre = document.createDocumentFragment();
  const text = JSON.stringify(value, null, 2);
  // Colour keys, strings and numbers; everything is text, never HTML.
  const re = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([^"\d-]+|-)/g;
  let m;
  while ((m = re.exec(text))) {
    if (m[1] && m[2]) pre.append(el("span", "k", m[1]), m[2]);
    else if (m[1]) pre.append(el("span", "s", m[1]));
    else if (m[3]) pre.append(el("span", "n", m[3]));
    else pre.append(m[0]);
  }
  return pre;
}

function render(input, r) {
  const env = r.envelope;
  const threshold = Number($("threshold").value) || 0;
  const calls = env.function_calls || [];
  const withheld = env.suppressed_calls || [];
  const conf = Number(env.confidence) || 0;
  const act = calls.length > 0 && conf >= threshold;

  $("empty").hidden = true;
  $("result").hidden = false;
  const v = $("verdict");
  if (act) {
    v.className = "verdict act";
    v.textContent = "ACT: confident enough to run";
  } else if (calls.length || withheld.length) {
    v.className = "verdict escalate";
    v.textContent = calls.length ? "ESCALATE: below your threshold" : "ESCALATE: the engine withheld this call";
  } else {
    v.className = "verdict none";
    v.textContent = "NO CALL";
  }

  const scene = sceneOf(current.id);
  if (act) for (const c of calls) scene.note(scene.apply(c));
  else for (const c of calls.length ? calls : withheld) scene.hold(c);

  const shown = calls.length ? calls : withheld;
  $("call").replaceChildren(jsonView(shown.length === 1 ? shown[0] : shown));

  const box = $("decisions");
  box.replaceChildren();
  for (const d of env.decisions || []) {
    box.append(el("div", "label", `${d.tool}.${d.argument}`));
    const wrap = el("div", "decision");
    const entries = Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]);
    for (const [name, p] of entries) {
      const row = el("div", "opt" + (name === d.choice ? " chosen" : ""));
      const track = el("div", "track");
      const fill = el("div", "fill");
      fill.style.width = `${(100 * p).toFixed(1)}%`;
      track.append(fill);
      row.append(el("span", "name", name), track, el("span", "pct", p.toFixed(3)));
      wrap.append(row);
    }
    box.append(wrap);
  }

  $("conf-fill").style.width = `${(100 * conf).toFixed(1)}%`;
  $("conf-mark").style.left = `calc(${(100 * threshold).toFixed(1)}% - 1px)`;
  $("conf-fill").parentElement.classList.toggle("low", conf < threshold);
  $("conf-num").textContent = conf.toFixed(4);

  let why = env.reasoning || "";
  const ungrounded = env.validation && env.validation.ungrounded;
  if (!calls.length && withheld.length && ungrounded && ungrounded.length) {
    why += `${why ? " " : ""}(withheld: ${ungrounded.join(", ")} not found in the request)`;
  }
  $("reasoning").textContent = why || "none";
  $("timing").textContent =
    `${Math.round(r.turnMs)} ms  ·  prefill ${env.prefill_tps} tok/s  ·  decode ${env.decode_tps} tok/s  ·  ${engineMode}`;

  const li = el("li");
  const left = el("div");
  left.append(el("div", "q", `$ ${input}`));
  const summary = shown.map((c) => `${c.name}(${Object.entries(c.arguments || {}).map(([k, x]) => `${k}=${JSON.stringify(x)}`).join(", ")})`).join("; ");
  left.append(el("div", "c", summary || "no call"));
  const right = el("div", "t" + (act ? "" : " esc"), `${conf.toFixed(2)} · ${Math.round(r.turnMs)} ms`);
  li.append(left, right);
  $("history").prepend(li);
}

// // RACE

let cactus = null;
let raceId = 0;

function cactusCall(msg) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    const onMsg = (e) => {
      if (e.data.id !== id) return;
      cactus.removeEventListener("message", onMsg);
      e.data.type === "error" ? reject(new Error(e.data.message)) : resolve(e.data);
    };
    cactus.addEventListener("message", onMsg);
    cactus.postMessage({ id, ...msg });
  });
}

const median = (a) => {
  const b = [...a].sort((x, y) => x - y);
  return b.length ? (b.length % 2 ? b[b.length >> 1] : (b[b.length / 2 - 1] + b[b.length / 2]) / 2) : 0;
};

async function race() {
  if (busy || !engineReady) return;
  let tools = current.tools;
  let system = current.system || "";
  let queries = current.queries;
  if (current.id === "custom") {
    const c = customTools();
    if (c.error) return status(`custom tools: ${c.error}`, true);
    tools = c.tools;
    system = $("custom-system").value;
    queries = [$("input").value.trim()].filter(Boolean);
    if (!queries.length) return status("type a command to race on", true);
  }
  busy = true;
  $("race-go").disabled = true;
  $("run").disabled = true;
  $("race").hidden = false;
  const lanes = { ours: { turns: [], calls: [] }, cactus: { turns: [], calls: [] } };
  for (const who of ["ours", "cactus"]) {
    $(`chips-${who}`).replaceChildren();
    $(`fill-${who}`).style.width = "0";
    $(`stat-${who}`).textContent = "";
  }
  $("race-verdict").replaceChildren();
  try {
    if (!cactus) {
      $("race-note").textContent = "loading Cactus's engine from Hugging Face";
      cactus = new Worker(new URL("./race-worker.js", import.meta.url));
      await cactusCall({ type: "load" });
    }
    $("race-note").textContent = "racing: each command goes to both engines, alternating which goes first";
    const toolsJson = JSON.stringify(tools);
    // Both read the tools fresh (a new agent on our side).
    const key = `race:${++raceId}`;
    const [ourInit, theirInit] = [await call({ type: "run", key, system, tools: toolsJson, input: queries[0], decide: false, maxTokens: 512 }), await cactusCall({ type: "init", system, tools: toolsJson })];
    lanes.ours.initMs = ourInit.prefixMs;
    lanes.cactus.initMs = theirInit.prefixMs;
    const total = { ours: 0, cactus: 0 };
    const scale = () => Math.max(total.ours, total.cactus, 1);
    const draw = () => {
      for (const who of ["ours", "cactus"]) {
        const l = lanes[who];
        $(`fill-${who}`).style.width = `${(100 * total[who]) / scale()}%`;
        $(`stat-${who}`).textContent = `tools ${Math.round(l.initMs)} ms · median ${Math.round(median(l.turns))} ms · total ${(total[who] / 1000).toFixed(2)} s`;
      }
    };
    const key2 = (c) => JSON.stringify(c.function_calls || []);
    for (let i = 0; i < queries.length; i++) {
      const order = i % 2 ? ["cactus", "ours"] : ["ours", "cactus"];
      for (const who of order) {
        const r = who === "ours"
          ? await call({ type: "run", key, system, tools: toolsJson, input: queries[i], decide: false, maxTokens: 512 })
          : await cactusCall({ type: "run", input: queries[i] });
        lanes[who].turns.push(r.turnMs);
        lanes[who].calls.push(key2(r.envelope));
        total[who] += r.turnMs;
      }
      const same = lanes.ours.calls[i] === lanes.cactus.calls[i];
      for (const who of ["ours", "cactus"]) {
        const chip = el("span", same ? "" : "diff", `${Math.round(lanes[who].turns[i])} ms`);
        chip.title = `${queries[i]}\n${lanes[who].calls[i]}`;
        $(`chips-${who}`).append(chip);
      }
      draw();
    }
    const a = median(lanes.ours.turns), b = median(lanes.cactus.turns);
    const same = lanes.ours.calls.filter((c, i) => c === lanes.cactus.calls[i]).length;
    const v = $("race-verdict");
    if (a < b) v.append(el("span", "win", `needle-rs is ${(b / a).toFixed(2)}× faster per command`), ` in this browser (median ${Math.round(a)} vs ${Math.round(b)} ms).`);
    else v.append(`Cactus's engine won this one: median ${Math.round(b)} vs ${Math.round(a)} ms.`);
    v.append(el("div", "race-note", `Same calls on ${same} of ${queries.length} commands (differences outlined). Reading the tools: ${Math.round(lanes.ours.initMs)} vs ${Math.round(lanes.cactus.initMs)} ms.`));
    $("race-note").textContent = "Run it again, or pick another example above and race that.";
  } catch (e) {
    $("race-note").textContent = `race failed: ${e.message}`;
  } finally {
    busy = false;
    $("race-go").disabled = false;
    $("run").disabled = false;
  }
}

// // START

let engineMode = "";

async function start() {
  examples = await (await fetch("examples.json")).json();
  $("custom-tools").value = JSON.stringify(examples[0].tools, null, 2);
  buildTabs();
  select(examples[0]);
  $("scene-reset").onclick = () => sceneOf(current.id).reset();
  $("race-go").onclick = race;
  $("ask").onsubmit = (e) => {
    e.preventDefault();
    run();
  };
  status("loading the engine and the model");
  try {
    const r = await call({ type: "load" });
    engineMode = r.mode === "relaxed" ? "hardware fma (relaxed SIMD)" : "exact software fma (SIMD)";
    $("meter").hidden = true;
    engineReady = true;
    $("run").disabled = false;
    $("race-go").disabled = false;
    const src = r.cached ? "from cache" : `${(r.bytes / 1e6).toFixed(1)} MB downloaded`;
    status(`ready: model ${src}, loaded in ${Math.round(r.loadMs)} ms, ${engineMode}. Pick a command.`, true);
  } catch (e) {
    status(`could not start: ${e.message}. This page needs a browser with WebAssembly SIMD (Chrome 91, Firefox 89, Safari 16.4 or newer).`, true);
  }
}

start();
