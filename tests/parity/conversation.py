"""Run multi-turn conversations through both engines and print each turn's
raw envelope side by side. A conversation file is JSON:
{"tools": [...], "system": "...", "conversations": [["turn 1", "turn 2", ...], ...]}
(or {"env": "smart_home", ...}); a turn that parses as JSON is a tool result.

    ~/git/needle/.venv/bin/python tests/parity/conversation.py convo.json
"""
import json
import os
import re
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
for convo in spec["conversations"]:
    agent.reset()
    for turn in convo:
        raw.clear()
        agent.complete(turn)
        print(raw.get("v", "null"), flush=True)
'''


def run(engine, path):
    env = dict(os.environ, NEEDLE3_LIB_PATH=ENGINES[engine])
    out = subprocess.run([sys.executable, "-c", CHILD, path], env=env, capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l.startswith("{") or l == "null"]


def strip(s):
    return re.sub(r',"(prefill_tps|decode_tps|peak_ram_mb)":[0-9.]+', "", s)


spec = json.load(open(sys.argv[1]))
native, ours = run("native", sys.argv[1]), run("ours", sys.argv[1])
turns = [t for c in spec["conversations"] for t in c]
same = 0
for t, a, b in zip(turns, native, ours):
    ok = strip(a) == strip(b)
    same += ok
    print(("   " if ok else "!! ") + t[:70])
    if not ok:
        print("   native", strip(a)[:400])
        print("   ours  ", strip(b)[:400])
print(f"{same}/{len(turns)} turns identical")
