"""Interleaved latency A/B: short latency.py runs of each engine in turn, many
rounds, so background load lands on every variant alike.

    python3 tests/parity/ab.py ROUNDS name=lib[:ENV=VAL,...] ...

Prints each variant's median over rounds of the per-run total median.
"""
import os, re, statistics, subprocess, sys

PY = os.path.expanduser("~/git/needle/.venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
rounds = int(sys.argv[1])
variants = []
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    lib, _, env = rest.partition(":")
    extra = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    variants.append((name, lib, extra))
times = {name: [] for name, _, _ in variants}
for r in range(rounds):
    for name, lib, extra in variants:
        env = {**os.environ, "NEEDLE_TELEMETRY": "0", "NEEDLE3_LIB_PATH": lib, **extra}
        out = subprocess.run([PY, f"{HERE}/latency.py", f"{HERE}/tools.json", "6"], env=env,
                             capture_output=True, text=True).stdout
        m = re.search(r"total median ([\d.]+)", out)
        times[name].append(float(m.group(1)) if m else float("nan"))
    print(f"round {r + 1}: " + "  ".join(f"{n} {times[n][-1]:.1f}" for n in times), file=sys.stderr)
for name, ts in times.items():
    print(f"{name:10} median {statistics.median(ts):7.1f} ms   min {min(ts):7.1f} ms")
