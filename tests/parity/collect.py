"""Every bundled environment case through NEEDLE3_LIB_PATH; one line per case
with the engine's raw envelope text (before the Python package touches it)."""
import importlib, json, os, sys
sys.path.insert(0, os.path.expanduser("~/git/needle"))
os.environ["NEEDLE_TELEMETRY"] = "0"
_orig = json.loads
last = {}
def spy(s, *a, **k):
    t = s if isinstance(s, str) else s.decode() if isinstance(s, (bytes, bytearray)) else None
    if t is not None and '"function_calls"' in t:
        last["raw"] = t
    return _orig(s, *a, **k)
json.loads = spy
import needle
for name in ["smart_home", "productivity", "data_capture", "wearable", "kitchen_appliance", "media_player"]:
    m = importlib.import_module(f"needle.environments.{name}")
    agent = needle.Needle(tools=m.TOOLS, system=m.SYSTEM, auto_date=False)
    for i, case in enumerate(m.TEST_CASES):
        agent.reset(); last.clear()
        r = agent.complete(case["query"])
        print(json.dumps({"env": name, "i": i, "query": case["query"], "raw": last.get("raw"), "final": r}), flush=True)
