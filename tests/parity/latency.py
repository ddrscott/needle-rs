"""End-to-end turn latency through the Python package, for whichever engine
NEEDLE3_LIB_PATH points at."""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.expanduser("~/git/needle"))
os.environ["NEEDLE_TELEMETRY"] = "0"
import needle

tools = json.load(open(sys.argv[1]))
queries = ["dim the living room lights to 30", "what's it like in Lagos right now?",
           "turn on the kitchen lights", "weather in Paris please"]
agent = needle.Needle(tools=tools, system="date: 2026-09-25 Fri 10:00", auto_date=False)
for q in queries:
    agent.reset(); agent.complete(q)
lat = {q: [] for q in queries}
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 10):
    for q in queries:
        agent.reset()
        t = time.perf_counter()
        r = agent.complete(q)
        lat[q].append(time.perf_counter() - t)
for q in queries:
    agent.reset()
    r = agent.complete(q)
    print(f"{min(lat[q])*1000:7.1f} ms min  {statistics.median(lat[q])*1000:7.1f} ms median  {q!r:45} -> {json.dumps(r['function_calls'])}")
print(f"total median {sum(statistics.median(v) for v in lat.values())*1000:.1f} ms")
