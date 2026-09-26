# Parity against the native engine

These scripts drive the reference Python package (`~/git/needle`) against
either engine library, so both sides do exactly the same work.

```sh
PY=~/git/needle/.venv/bin/python
NATIVE=$PWD/models/native/needle/libneedle3.dylib   # needle fetch
OURS=$PWD/target/release/libneedle3.dylib          # cargo build --release -p needle-ffi

# Every bundled environment case, one raw envelope per line
NEEDLE3_LIB_PATH=$NATIVE $PY tests/parity/collect.py > native.jsonl
NEEDLE3_LIB_PATH=$OURS   $PY tests/parity/collect.py > ours.jsonl
python3 tests/parity/compare.py native.jsonl ours.jsonl 40

# Single queries, raw envelopes side by side (tools + system, or a bundled env)
$PY tests/parity/probe.py probe.json --short

# Multi-turn conversations (a turn that parses as JSON is a tool result);
# tests/parity/conversations/ holds the ones checked: three short ones on a
# bundled environment, a 10-tool set (convo_retr), and one whose tool results
# overflow the context (convo_over, slow: the engine re-prefills every step
# once it is full)
$PY tests/parity/conversation.py tests/parity/conversations/convo1.json

# Four agent turns, end to end
NEEDLE3_LIB_PATH=$OURS $PY tests/parity/latency.py tests/parity/tools.json 20

# The same, both engines interleaved over many short rounds (background load
# lands on both alike); name=lib[:ENV=VAL,...]
python3 tests/parity/ab.py 8 native=$NATIVE ours=$OURS four=$OURS:NEEDLE_THREADS=4
```

`compare.py` ignores the timing fields and reports how many raw envelopes
match byte for byte, then the cases with the same decode that still differ
in a gate or in confidence.

`rules_oracle.py` calls the native library's call post-processor directly
(it is not exported; the script finds it from `needle_init`'s address) and
records its output for JSONL cases. `rules_cases.jsonl` holds 1,099 such
cases, which `cargo test` replays against `rules::postprocess`.
