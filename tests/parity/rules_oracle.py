"""Run call lists through the native library's post-processor directly.

The post-processor (`FUN_0006998c` at image offset 0x6998c) isn't exported,
so this finds it from `needle_init`'s address and calls it through a tiny C
shim (it returns a `std::string` by value, which arrives through x8).

Reads JSONL cases from stdin and writes each back with a "native" member:

    {"tools": [...] or "compact text", "system": "grounding text",
     "conversation": "...", "request": "...", "calls": "[...]"}
    -> "native": {"out": "...", "withhold": bool, "withhold2": bool}

`system` is the stored grounding text (see `grounding`); `conversation`
defaults to system + "\\n" + request.

    python3 tests/parity/rules_oracle.py < cases.jsonl > tests/parity/rules_cases.jsonl
"""
import ctypes
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "models/native/needle/libneedle3.dylib")
SHIM = os.path.join(ROOT, "target/rules_oracle_shim.dylib")
SHIM_C = r"""
#include <string.h>
typedef struct { char b[24]; } S24;
typedef S24 (*F)(void *, void *, void *, void *);
void call_post(void *f, void *out, void *calls, void *ctx, void *w1, void *w2) {
    S24 r = ((F)f)(calls, ctx, w1, w2);
    memcpy(out, &r, 24);
}
"""


def grounding(system):
    """The system text as needle_init stores it for grounding: timestamps
    blanked, then the first `date: ` fact (to the next `;`) cut to one space."""
    s = re.sub(r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d", lambda m: " " * 16, system)
    at = s.find("date: ")
    if at >= 0:
        end = s.find(";", at)
        s = s[:at] + " " + (s[end:] if end >= 0 else "")
    return s


class Str(ctypes.Structure):
    """libc++ std::string, long form."""
    _fields_ = [("data", ctypes.c_void_p), ("size", ctypes.c_uint64), ("cap", ctypes.c_uint64)]


class Ctx(ctypes.Structure):
    _fields_ = [("request", Str), ("system", Str), ("conv", Str), ("tools", Str),
                ("facts", ctypes.c_void_p), ("flag", ctypes.c_uint8), ("pad", ctypes.c_uint8 * 7)]


_keep = []


def _make(b):
    buf = ctypes.create_string_buffer(b, len(b) + 1)
    _keep.append(buf)
    return Str(ctypes.cast(buf, ctypes.c_void_p).value, len(b), (len(b) + 1) | (1 << 63))


def _read(s):
    raw = bytes(ctypes.string_at(ctypes.addressof(s), 24))
    return ctypes.string_at(s.data, s.size) if raw[23] & 0x80 else raw[:raw[23]]


def load():
    if not os.path.exists(SHIM):
        src = SHIM + ".c"
        with open(src, "w") as f:
            f.write(SHIM_C)
        subprocess.run(["cc", "-O2", "-shared", "-o", SHIM, src], check=True)
    lib = ctypes.CDLL(LIB)
    shim = ctypes.CDLL(SHIM)
    shim.call_post.argtypes = [ctypes.c_void_p] * 6
    fn = ctypes.cast(lib.needle_init, ctypes.c_void_p).value - 0x1AAC + 0x6998C
    return lambda *a: shim.call_post(fn, *a)


def run(post, case):
    tools = case["tools"]
    if not isinstance(tools, str):
        tools = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    system = case.get("system", " ")
    conv = case.get("conversation", system + "\n" + case["request"])
    ctx = Ctx(_make(case["request"].encode()), _make(system.encode()), _make(conv.encode()), _make(tools.encode()), None, 1)
    out = Str(0, 0, 0)
    w1, w2 = ctypes.c_uint8(0), ctypes.c_uint8(0)
    calls = _make(case["calls"].encode())
    post(ctypes.addressof(out), ctypes.addressof(calls), ctypes.addressof(ctx), ctypes.addressof(w1), ctypes.addressof(w2))
    return {"out": _read(out).decode("utf-8", "replace"), "withhold": bool(w1.value), "withhold2": bool(w2.value)}


if __name__ == "__main__":
    post = load()
    for line in sys.stdin:
        if line.strip():
            case = json.loads(line)
            case["native"] = run(post, case)
            print(json.dumps(case, ensure_ascii=False), flush=True)
