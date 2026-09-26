# needle-rs

Live at https://askscottpierce.com/needle-rs

A Rust port of [Needle 3](https://github.com/cactus-compute/needle), Cactus Compute's 121M-parameter
tool-calling model. It covers everything the Python package (`cactus-needle` 3.0.1) does: the model
spec, inference, LoRA fine-tuning, `.cact` export, the hosted-platform client, the playground, and a
drop-in replacement for the native engine library the package ships as a binary. No Python, no JAX.

It's checked against the Python package at every layer (tokens, logits, gradients, archive bytes), and
it's 25-180x faster than the Python code paths it replaces.

**Try it in your browser: [askscottpierce.com/needle-rs](https://askscottpierce.com/needle-rs/).** The
same engine, compiled to WebAssembly, turns commands into typed tool calls right in the tab, with a
probability for every choice. It's a local, free alternative to cloud "typed decision" models like
TypeSafe's Jev for the kinds of calls a 121M model handles well (see `// NEEDLE_VS_JEV`).

## // QUICK_START

```sh
. ./env.sh                      # puts rustup's stable toolchain (1.98+) ahead of Homebrew's rustc
cargo build --release
./scripts/fetch-models.sh       # needle3.cact, needle3.safetensors, tokenizer.model -> models/

# One agent turn: grammar-constrained calls, reasoning, confidence, the engine's JSON envelope
./target/release/needle complete --tools tools.json --system "date: 2026-09-25 Fri 10:00" \
    "dim the living room lights to 30"

# The reference's dev path: greedy decode of the raw completion
./target/release/needle run --checkpoint models/needle3.cact --query "dim the lights" --tools tools.json
```

Use it from the Python package instead of the prebuilt engine:

```sh
cargo build --release -p needle-ffi
NEEDLE3_LIB_PATH=$PWD/target/release/libneedle3.dylib python -c 'import needle; ...'
```

## // COMMANDS

Everything the reference CLI has, with the same flags and defaults, plus a few dev tools.

| Command | What it does |
| --- | --- |
| `needle run --checkpoint X [--query Q --tools T]` | Greedy decode of the raw completion from a checkpoint or `.cact` |
| `needle finetune data.jsonl [--checkpoint X] [--generate N]` | LoRA fine-tune (query/answers or single-turn chat JSONL) |
| `needle build [ckpt] --lora A [--layers N] [--platform P] [--upload]` | Merge, slice the ladder rung, export CQ-W4 `.cact` |
| `needle generate-data --tools T \| --augment D` | Synthesize training data through OpenRouter |
| `needle download needle3 \| needle3.safetensors \| <platform> \| model-<id> \| org/repo[/f.cact]` | Pull published artifacts |
| `needle fetch [--platform-tag T]` | Fetch the native engine library from its wheel |
| `needle platform finetune \| generate \| jobs \| models \| files \| billing` | The hosted platform at cactuscompute.com |
| `needle playground [--weights X]` | The browser UI, served locally |
| `needle serve --tools T` | The native runner's HTTP mode (`POST /complete`, `POST /reset`) |
| `needle complete --tools T Q...` | Agent turns, one JSON envelope each (`--decide` for typed decisions, `--fresh` to reset between queries) |
| `needle eval data.jsonl --checkpoint X [--lora A]` | Exact-call accuracy of greedy decodes |
| `needle env smart_home --checkpoint X` | A bundled environment's frozen acceptance suite |
| `needle synth --output data.jsonl` | Templated smart-home training data (offline, no LLM) |
| `needle bench [--agent] [--chunk N]` | Prefill and decode speed |

## // LAYOUT

| Crate | Role |
| --- | --- |
| `needle-core` | The spec: config and ladder rules, SentencePiece BPE, safetensors checkpoints and adapters, the CQ quantizer, `.cact` reader/writer, prompt rendering, CPython-exact `json.dumps`, numpy's MT19937 |
| `needle-engine` | The forward pass with a KV cache, the packed CQ runtime, probe heads (confidence, embeddings), the decode grammar, the agent |
| `needle-train` | The hand-derived reverse pass, LoRA, optax-exact AdamW and schedule, numpy/JAX-exact random streams, the `finetune` loop |
| `needle-ffi` | `libneedle3`: the `needle.h` C API (`needle_load`, `needle_init`, `needle_complete`, `needle_embed`, `needle_reset`, `needle_last_error`) |
| `needle-cli` | The `needle` binary |
| `needle-wasm` | The browser build: the agent over `wasm-bindgen` (`web/` is the demo page) |

## // SPEED

Apple M3 Pro, same machine, same inputs. The Python numbers are the reference package's own code
paths (JAX 0.11 on CPU, numpy).

| Task | Python package | needle-rs | Speedup |
| --- | --- | --- | --- |
| `needle run`, tool-call prompt (118 tokens in, ~45 out) | 26.9 s | 0.69 s from the checkpoint, 0.15 s from the `.cact` | 39x / 179x |
| `needle finetune`, 160 examples, 1 epoch, batch 16, scored holdout | 13 min 41 s | 27 s | 30x |
| One optimizer step (batch 16, ~700-token examples) | ~60 s | 1.7-2.2 s | ~30x |
| `write_export` of the full model (63 MB CQ-W4 archive) | 6.9 s | 0.27 s | 25x |

Against the prebuilt native C++ engine (`libneedle3` 3.0.1), driven through the Python package's own
`Needle.complete()` so both sides do the same work (`tests/parity/latency.py`; `tests/parity/ab.py`
interleaves runs so background load lands on both alike):

| | Native engine | needle-rs |
| --- | --- | --- |
| Four agent turns, median total, other programs running (load average 6-8) | 113-118 ms | 97-101 ms |
| The same, six utility-QoS burners | 120 ms | 106 ms |
| The same, six full-speed burners on every core | 262-267 ms | 264-267 ms |
| Decode speed, one turn (tokens/s) | 1,440-1,770 | 1,550-1,950 |
| Prefill, a 20-token turn (tokens/s) | 3,560-3,890 | 3,460-4,090 |
| Resident memory in the Python process | 126 MB | 181 MB |

A single-token step runs as one team job: members advance through each layer's phases and meet at a
flat-combining barrier (about 200 ns for six members). Wide pieces are split (gate rows, projection
rows, attention heads by key chunk, Kronecker tiles, output columns); the small reductions every later
piece needs are worked out by every member for itself, which costs less than a barrier. A chunk of
prompt tokens forks and joins the team per operation instead. Work splits never change a value, so
the output doesn't depend on the member count. The team is one thread fewer than the performance
cores: a step waits on its slowest member, and leaving a core to the rest of the system keeps members
from being preempted. Only when every core is saturated does the native engine's four-thread pool
come out ahead; there, fewer members (`NEEDLE_THREADS=3`) match it.

The matmul kernels read the codes in the archive's own layout, as the engine does: a nibble indexes a
16-entry table (two for 2-bit pairs), one activation load serves four rows, and the activations are
stored in the matching de-interleaved order. The small per-layer weights (Kronecker factors, MLP
conditioning, norms, conv taps) stay in the archive's f16 and widen on load, which halves what every
member streams per layer without changing a value. A prompt chunk runs as one job too: each member
carries its rows through every row-local piece and meets the others only at the projections and the
attention, five barriers a layer instead of a fork and join per operation. Neither job allocates
per step: the scratch buffers live with the calling thread and each member's thread.

How many members a step uses follows what the machine is doing. Every member adds up its barrier
waits over the step; a step where someone waited more than 60% of it means a member was taken off
its core (an uneven split leaves a member waiting up to a third), and three such steps in a window
of 32 drop a member (down to three), while a thousand stall-free steps add one back on trial. The
count applies to every job (the head, the prompt chunks, activation prep), and workers left out stay
parked. `NEEDLE_THREADS` pins the count; timing-based tuning was tried first and lost to noise.

## // PARITY

Every piece is checked against the Python package as the oracle (`tests/oracle/fixtures.py` writes
the fixtures; the Rust tests skip when a fixture is missing).

| What | Result |
| --- | --- |
| Tokenizer, 3,010 fuzzed strings (markers, UTF-8, bytes) | identical ids and decodes |
| Float32 logits vs JAX, 127-token prompt | max diff 3e-4 of 162; same argmax everywhere |
| Quantized numerics (A8 + int8 KV + CQ-W4) | within JAX's own noise floor: JAX vs itself with weights nudged 1e-7 differs by median 0.36; Rust vs JAX by median 0.35 |
| Confidence head logit | 0.58330 vs 0.58329 |
| LoRA gradients, float numerics | relative error 3e-5 on all ten tensors |
| LoRA init (JAX threefry + erfinv) | relative error 2.6e-7 |
| Holdout split and epoch shuffles (numpy PCG64) | identical permutations |
| Fine-tune loss curve, 9 steps | tracks JAX to ~0.003 per step; val 0.9228 vs 0.9182; held-out 9/16 vs 9/16 |
| `.cact` export of the full model | header, directory, codebooks, tokenizer and every non-CQ tensor byte-identical; 123 of 63M bytes differ, all CQ rounding ties (numpy's own BLAS order differs across platforms) |
| The Python package's test suite on `libneedle3` from this repo | 170 passed |

Against the native engine, through the Python package (`tests/parity/`), on every case of the six
bundled environments (192 requests):

| What | Result |
| --- | --- |
| Whole raw envelope, timing fields aside | byte-identical in 192 of 192 |
| Multi-turn conversations with tool results (`tests/parity/conversations/`), including a 10-tool one and one whose tool results overflow the context | byte-identical in every turn |
| Per-turn tool retrieval | the engine only enables it for a model with a dedicated embedding readout, which `needle3.cact` lacks; with it every toolset runs in static mode, as here |
| Logits after a decode step, all 8,192 | bit-identical (dumped from the running library with lldb) |
| The call post-processor (repairs, gates, dates) against the library's own function | identical on 36,000 generated cases; 1,099 replay in `cargo test` |
| Prompt tokens (static prefix and turns) | identical |

The packed forward pass (`crates/needle-engine/src/infer.rs`) reproduces libneedle3's arithmetic
operation for operation: int8 attention with 21-bit fixed-point softmax weights and native's key
chunking, an int8 mHC gate projection, int8 engram value history, f64 Sinkhorn, the engine's inline
`exp`, RoPE advanced by recurrence while decoding, four-accumulator norms, fused multiply-adds where
the engine fuses them, three matmul float combines (decode, prefill with its remainder-token rule,
single-row sparse logits), and the engine's feed pattern (chunks of at most 32, then one token per
call, KV kept in the engine's slot ring). `docs/engine-rules.md` has the details and evidence.

Against JAX the quantized network is chaotic: an int8 rounding that flips on a 1e-7 difference moves
logits by a few tenths, so parity with the reference Python graph means staying inside the band JAX
shows against itself (the general forward pass is tested to cosine above 0.999, same top choice).

## // WHAT_CHANGED_UNDER_THE_HOOD

Same math, different machinery:

- **Training shares prompts.** Tool-calling datasets repeat one system+tools prompt across every
  example. Batches lay out the sequences' common prefix once; causality makes that exact, and the reverse
  pass sums each sequence's gradient into the shared rows. That's 2,070 rows down to 790 on a real batch.
- **LoRA gradients stay low-rank.** `dA = s X^T (dY B^T)`, `dB = s (X A)^T dY`. The dense weight
  gradient is never formed, and frozen weights are CQ-quantized once instead of every step.
- **The engine runs the packed weights.** A CQ group is `(codebook[idx] * norm) H`, and `H` is
  symmetric, so the rotation moves onto the activation. Each weight row is then a NEON table lookup
  plus int8 dot products, reading about 13 MB per token instead of 200.
- **Decode is one team job per token.** A small spin-synchronized team steps through each layer's
  phases in lockstep instead of forking and joining per operation. Attention is flash-decoding across
  KV-head and key slices.
- **Barriers that do not contend.** Each team member announces its arrival on its own cache line and
  one member gathers them, so a phase boundary costs a cache-line hop instead of a pile-up on one
  counter.

Call decoding follows the native engine token for token rather than taking shortcuts: structural
text the grammar forces is still decoded one token at a time, because the tokens the model sees
change what it writes next and what `confidence` reports.

## // TRAINING_A_WORKING_MODEL

`needle synth` writes templated smart-home data (no LLM needed). With the reference defaults
(3 epochs, rank 16, lr 1e-4) on 2,000 examples:

```sh
needle synth --num-samples 2000 --output train.jsonl
needle finetune train.jsonl --checkpoint models/needle3.safetensors --out runs/smart_home_lora.safetensors
needle build models/needle3.safetensors --lora runs/smart_home_lora.safetensors --out runs/smart_home.cact
needle eval val.jsonl --checkpoint runs/smart_home.cact
```

Validation loss went from 0.386 after epoch 1 to 0.200 after epoch 3 at 2.1 s per step. That's 339
steps in 12 minutes, against about 5.6 hours for the same run in JAX. Exact calls on a separate
validation set went from 31/60 (base) to 37/60 (tuned checkpoint) and 39/60 (the shipped W4 archive).

## // TYPED_DECISIONS

`Agent::decide` (`needle complete --decide`, `needle.decide()` in the browser) runs a turn that must
call a tool and scores every option of every enum argument it fills, whole option by whole option
(teacher-forced, token by token), instead of sampling one. The envelope gains a `decisions` list:

```json
{"tool":"attack","argument":"weapon","choice":"fireball",
 "probabilities":{"sword":0.0457,"bow":0.4162,"fireball":0.538}}
```

The chosen call is the one `complete` would pick, but scoring options it would not have scored leaves
their keys in the attention ring, so later tokens can differ from `complete`'s. It's an extension, not
part of the native engine's behavior, and the parity claims don't cover it.

## // IN_THE_BROWSER

`crates/needle-wasm` wraps the agent for JavaScript, and `web/` is the demo page
([live](https://askscottpierce.com/needle-rs/)). `scripts/build-web.sh` builds it (needs `wasm-pack`);
serve `web/` with any static server. The page fetches `needle3.cact` (35 MB) from Cactus Compute's
Hugging Face repo on the first visit and keeps it in Cache Storage.

- The int8 matmuls, attention dots, quantizer and vector exp have WebAssembly SIMD kernels that mirror
  the NEON ones exactly (`i8x16.swizzle` is the same table lookup; widening multiplies with pairwise
  adds give `sdot`'s lane sums).
- WebAssembly has no fused multiply-add, and the engine's arithmetic is fma throughout. Two builds cover
  it: `pkg-relaxed` uses the CPU's fma through relaxed SIMD, and the page loads it only after checking
  that the browser really fuses `relaxed_madd` (the spec lets it not); `pkg` computes each fma in f64
  and redoes in software the rare lane whose f64 sum sits exactly on an f32 rounding midpoint. A
  self-test (`simdSelftest`) checks both against scalar `fmaf` on 17M random inputs and a million
  constructed midpoint cases. The two builds give byte-identical envelopes on all 192 bundled cases.
- The relaxed build also uses the hardware int8 dot (`i32x4.relaxed_dot_i8x16_i7x16_add`), which is
  exact whenever the second operand is in 0..=127: the P.V digit planes already are, and each weight or
  query vector splits into positive and negative parts that are (`x.w = x.w+ - x.w-`). The same runtime
  check probes it.
- Against the native engine, the browser build makes the same calls on 188 of the 192 cases, and
  confidence matches on the median case (90% within 0.005). The drift is the math library: Apple's
  `expf`, `exp` and `powf` aren't correctly rounded and the musl port used here rounds differently
  (musl's `expf` disagrees with Apple's on 0.9% of inputs; 4 of the 24 RoPE frequencies differ).
- Against Cactus Compute's own browser engine (`wasm/needle.wasm` from their Hugging Face repo, which
  powers cactuscompute.com/needle), both loaded in the same page on an M3 Pro, the smart-home tool set
  (639 prompt tokens, 32 queries):

  | Browser | Turn, median (Cactus / ours) | Reading the tools (Cactus / ours) | Model load (Cactus / ours) |
  | --- | --- | --- | --- |
  | Chrome 154 | 373 / **239** ms | 2.64 / **1.65** s | 28 / 31 ms |
  | Firefox 155 | 419 / **359** ms | 2.96 / **2.52** s | 44 / 39 ms |
  | WebKit 26.6 (Safari) | 389 / **255** ms | 2.68 / **1.81** s | 41 / 32 ms |

  On all 192 bundled cases in Node, ours makes the same calls as the native engine on 188 and is
  byte-identical on 86; Cactus's browser engine makes the same calls on 191 and is byte-identical on 5.
  The exact build (no relaxed SIMD, or a CPU without fma) is slower, about 0.95 s a turn.
- Switching tool sets costs one read of the tools, and the page keeps an agent per tool set.

## // NEEDLE_VS_JEV

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (TypeSafe AI) answers typed
questions with probabilities instead of text, from a cloud API. Needle returns the same shape (typed,
schema-valid values plus how sure it is) on the device.

| | Jev | Needle 3 on needle-rs |
| --- | --- | --- |
| Returns | Typed answers (Choice, Score, Noul) with probabilities and confidence | Schema-valid tool calls, a confidence, and per-option probabilities for enum arguments |
| Runs | TypeSafe's API | The browser, the native library, the CLI |
| Latency | 70-500 ms end to end (TypeSafe's figure) | About 25 ms a turn native on an M3 Pro, about 0.25 s in the browser (measured) |
| Price | $0.042 per million input tokens, output free | Free |
| Weights | Closed, early access | Open (Apache-2.0); this engine is MIT |
| Breadth | Frontier-scale judgment | 121M parameters: strong at commands to calls, weak at open-ended judgment |

The honest read: they're two tiers, not rivals. Needle takes the frequent, well-formed commands on the
device, and its confidence (or its grounding check withholding a call) says when to hand a request to
Jev or a bigger model. In our tests it routed game, robot-arm and car commands well, and sorted
support tickets into teams whose names never appear in the text poorly (2 of 4); that kind of
judgment is Jev's pitch, not Needle's.

## // DEVELOPMENT

- `cargo test --release` runs the unit tests plus every oracle test whose fixture exists.
- Fixtures come from the reference's venv:
  `~/git/needle/.venv/bin/python tests/oracle/fixtures.py tokenizer|export|logits|split_quant|cells|grads|float_grads`.
- `NEEDLE_PROFILE=1` prints a span breakdown. `NEEDLE_THREADS` pins the team size (default: the
  performance cores, with the decode step choosing how many to use from measured stalls). `NEEDLE_F32=1` loads a `.cact` dequantized instead of packed.
  `NEEDLE_CHUNK_STEP=1` runs single tokens through the chunk path instead of the fused step, and
  `NEEDLE_CHUNK_OPS=1` runs prompt chunks as one operation at a time instead of one job (all three
  agree bit for bit). `NX_MEMBERS_LOG=1` prints the member-count decisions. `NEEDLE_TURN_TIMES=1` prints each agent
  turn's stages, and `NEEDLE_CALL_PROBS=1` prints every call token with the probability that feeds
  `confidence`.
- `tests/parity/` drives the reference Python package against the native engine and this one:
  whole-suite envelope comparisons, single probes, and end-to-end latency. See its README.
- `docs/engine-rules.md` records what the native engine does around the model (decoding, confidence,
  grounding gates, repairs), the evidence for each rule, and which ones this port matches.
- `scripts/build-web.sh` builds the browser engines into `web/pkg` and `web/pkg-relaxed`.
  `scripts/deploy-asp.sh` publishes the demo to askscottpierce.com/needle-rs: an assets-only Cloudflare
  Worker on two zone routes (`askscottpierce.com/needle-rs` and `/needle-rs/*`), with `web/` staged at
  `site/needle-rs/` so file paths match URLs. The path is hard-coded in `wrangler.jsonc` and the deploy
  script only; the page itself uses relative URLs.
- `cargo clippy --all-targets` and `cargo fmt --check` are clean. The MSRV is 1.98 (NEON `sdot`).

## // LICENSE_AND_THANKS

needle-rs is MIT-licensed (`LICENSE`).

A big thank-you to [Cactus Compute](https://github.com/cactus-compute/needle) for Needle 3: the model,
its weights ([Hugging Face](https://huggingface.co/Cactus-Compute/needle3)), and the reference package
and engine this port was built and checked against. All of that is theirs, under Apache-2.0, and none
of it is redistributed here: the weights are downloaded from their repo. The playground UI in
`crates/needle-cli/playground/` is copied from the reference package and keeps its Apache-2.0 license
(`crates/needle-cli/playground/LICENSE`).
