"""Run a probe file through both engines and print the raw envelopes side by
side. A probe file is JSON: {"tools": [...], "system": "...", "queries": [...]}.

    ~/git/needle/.venv/bin/python tests/parity/probe.py probe.json
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINES = {
    "native": os.path.join(ROOT, "models/native/needle/libneedle3.dylib"),
    "ours": os.path.join(ROOT, "target/release/libneedle3.dylib"),
}
CHILD = r'''
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/git/needle"))
os.environ["NEEDLE_TELEMETRY"] = "0"
_loads = json.loads
raw = {}
def spy(s, *a, **k):
    t = s if isinstance(s, str) else s.decode() if isinstance(s, (bytes, bytearray)) else None
    if t is not None and '"function_calls"' in t:
        raw["v"] = t
    return _loads(s, *a, **k)
json.loads = spy
import needle
spec = _loads(open(sys.argv[1]).read())
if "env" in spec:
    import importlib
    m = importlib.import_module("needle.environments." + spec["env"])
    spec["tools"], spec["system"] = m.TOOLS, m.SYSTEM
agent = needle.Needle(tools=spec["tools"], system=spec.get("system", ""), auto_date=False)
for q in spec["queries"]:
    agent.reset(); raw.clear()
    agent.complete(q)
    print(raw.get("v", "null"), flush=True)
'''


def run(engine, path):
    env = dict(os.environ, NEEDLE3_LIB_PATH=ENGINES[engine])
    out = subprocess.run([sys.executable, "-c", CHILD, path], env=env, capture_output=True, text=True)
    return [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{") or l == "null"]


def brief(e):
    if e is None:
        return "null"
    v = e.get("validation")
    return (f"calls={json.dumps(e['function_calls'])} supp={json.dumps(e['suppressed_calls'])} "
            f"conf={e['confidence']} val={json.dumps(v) if v is not None else '-'}")


def short(e):
    if e is None:
        return "null"
    calls = e["function_calls"] + e["suppressed_calls"]
    what = ",".join(c["name"] for c in calls) or "[]"
    return ("SUPP " if e["suppressed_calls"] else "KEEP ") + what


args = [a for a in sys.argv[1:] if not a.startswith("--")]
spec = json.load(open(args[0]))
native, ours = run("native", args[0]), run("ours", args[0])
if "--short" in sys.argv:
    for q, a, b in zip(spec["queries"], native, ours):
        mark = "  " if short(a) == short(b) else "!!"
        print(f"{mark} native {short(a):28} ours {short(b):28} {q}")
    sys.exit()
for q, a, b in zip(spec["queries"], native, ours):
    same = brief(a) == brief(b)
    print(("  " if same else "!!") + f" {q}")
    print(f"     native {brief(a)}")
    if not same:
        print(f"     ours   {brief(b)}")
