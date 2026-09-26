# Engine rules

What the native engine (`libneedle3` 3.0.1) does around the model, and how
this port matches it. The spec is the "Behaviour" section of the
[Needle 3 model card](https://huggingface.co/Cactus-Compute/needle3/blob/0420f0327acd15792cfe1b664ebf82ec186fc870/README.md)
and the [confidence guide](https://cactuscompute.com/blog/needle-confidence).
The details below were pinned by reading the stripped library's
disassembly and by probing it (`tests/parity/probe.py`). Every item says
where its evidence came from.

Status: **done** means the port behaves the same on every probe tried;
**partial** means the common case matches and the notes say what does not;
**todo** is not implemented yet.

## // DECODING

Code: `toolset.rs` (what `needle_init` does to its inputs), `grammar/`
(schema compiler, regex dialect, byte automaton), and the call loop in
`agent.rs`. Evidence labels: "tokens" means the prompt ids were captured
from the library with lldb (a breakpoint on the prefill at `0x6c88`) and
compared id for id; "trace" means the library's per-token commits and
candidate visits were captured the same way (`0x5f3b4`, `0x6905c`, the
feed at `0x684dc`).

- **done** The tools JSON is compacted, OpenAI `{"type":"function",
  "function":{...}}` wrappers are unwrapped, `triggers` is dropped, and tool
  and property names are snake_cased the library's way (`HTTPServer` ->
  `http_server`, a leading `_` dropped, non-ASCII names kept). Tools whose
  names collide all get `__1`, `__2`, ... by the sorted original names;
  colliding properties keep the first and number the rest from `__2`. Type
  aliases (`int`, `double`, `List[str]`, `Dict<K,V>`, `any`, ...) map to
  JSON-Schema types. Numbers keep their source text. Calls are written back
  in the caller's names. (tokens; `FUN_00004294`, `FUN_0001a7f0`,
  `FUN_00020708`, `FUN_0006b6e4`)
- **done** The system text: a JSON object becomes `key: value` facts over
  the library's key list (`date, locale, device, battery, network, location,
  user, assistant, thermostat, volume`), a `date` timestamp reformatted as
  `YYYY-MM-DD Www HH:MM`; plain text is kept verbatim (no trimming), except
  that text with a `YYYY-MM-DD[T ]HH:MM` timestamp and no `date: ` fact is
  replaced by `date: YYYY-MM-DD Www HH:MM` alone. The static prefix is
  `system block + <|im_start|>user + \n<tools>TOOLS</tools>`, with no newline
  after `user`; the first turn starts with it. A tool result is a `tool`
  turn, compacted, with an object wrapped in a list. (tokens: JSON-object,
  timestamp, blank and empty systems, collisions, a tool-result turn)
- **partial** Retrieval (six or more tools): each tool's canonical text
  (Python float repr) is embedded, the top five are rendered per turn in
  declaration order, and the grammar is compiled for them. The library
  keeps the conversation and splices its KV cache when the selection
  changes; this port re-renders the prompt each turn, so later turns of a
  retrieval conversation differ.
- **done** The turn is the engine's step machine (`0x5b35c`; `spec_turn`
  and lldb). Every step masks ids 4, 8, 9, 12 and 13. A user turn is forced
  to `<think>`; reasoning is greedy with `<think> <tool_call> </tool_call>`
  also masked, so `<eos>` or `<|im_end|>` can end the turn with no call
  (`"type":"respond"`). With 32 steps or fewer left, `</think>` is forced.
  After `</think>` the newline (when it is one token) and `<tool_call>` are
  forced. After a tool result the model opens freely: `<think>`,
  `<tool_call>` or prose, and once prose starts it can no longer call.
- **done** Attention in an engine session spans the static prefix (the
  sink) plus the last `kv_window` positions (256 in `needle3.cact`'s
  header), on top of each layer's own band (`spec_turn` §1.1 step 8). The
  reference's Python inference never windows; the engine does. Evidence: a
  761-byte tool result diverged at its first reasoning token without the
  window and matches native token for token with it. In retrieval mode the
  sink is the system block alone.
- **done** `triggers`: each pattern is an ECMAScript regex, matched case
  insensitively against the raw request (patterns with inline flags,
  lookbehind or named groups are dropped with the engine's stderr message).
  On a user turn without a negation cue, the tools with a match become the
  only names the grammar allows and the list cannot be `[]`; such a call is
  withheld only by the strong gates (negation, exclusion, reported speech),
  never by the confidence floor. (`spec_turn` §3.1 and §5.2; probes with
  `\bhelp\b`, `emergency`, `^lights? (on|off)$` and an invalid pattern)
- **partial** Retrieval (six or more tools; `spec_turn` §2.4): the top five
  tools by embedding similarity, a tool result keeping the previous
  selection. With the same selection the conversation continues; otherwise
  the prompt is rebuilt as the tools block, the history without the old
  block (at most one ring, 256 tokens), then the turn, and the tools block
  joins the attention sink. The engine sometimes splices the history back
  with its cached keys re-rotated to their new positions instead of
  re-prefilling (at most three splices in a row); this port always
  re-prefills, which is numerically close but not bit-identical on those
  turns. (lldb capture of the rebuild prefill; conversation probes)
- **done** Prefill runs in chunks of 32 tokens, as the engine's `FUN_6c88`
  does (the static prefix, every turn, retrieval rebuilds).
- **done** Input and context overflow (`spec_turn` §2.5): a turn longer than
  `ctx - (max_new + prefix)` keeps its last tokens; before any feed that
  would pass the context, the static prefix and the retrieval tools block
  stay and only the most recent conversation that fits is re-prefilled after
  them. Like the engine, once a conversation is that long every decode step
  re-prefills it, so such turns are slow (the native engine took over ten
  minutes on a 24-turn probe).
- **done** With four steps or fewer left, a complete list is closed with
  `</tool_call>` and an open one with `]` when that completes it. A grammar
  dead end ends the turn with `<|im_end|>`. A list the budget cut short
  keeps the calls whose arguments closed; with none the envelope reports
  `"error_code":"truncated"`. A string rollout or a named-option commit
  costs one budget step.
- **done** A free token's probability (for the geometric-mean confidence) is
  `1 / sum exp(l_j - l_tok)` computed like the engine: four f32 lanes of its
  Cephes `expf` (clamped at 88.376, ties-even rounding, fused multiply-adds),
  then an f64 tail. With no reasoning and no call, prose is the reasoning.
- **done** The call sampler's probabilities are `exp((double)l - max) / Z`
  with `Z` the four-lane f32 sum over the step's logits (base bans in
  place, before the call-step bans below). (asm `0x5e950`, lldb dump of the
  per-token probabilities)
- **done** Call tokens are chosen one at a time under the byte grammar, each
  candidate's bytes stepped on a copy of the grammar state. Nothing is
  forced; structural tokens are sampled and score 1 because nothing else
  fits. The call step also bans `<|im_end|> <think> </think> <tool_call>`
  and the token that would start a seventh repeat of a cycle of one to four
  tokens (`0x5e5d8`). (trace)
- **done** Sparse logits: after a sampled call token (and after
  `<tool_call>`), the library computes logits only for tokens whose first
  byte its state table allows next (plus `</tool_call>` and `<|im_end|>`)
  when that is under half the vocabulary; everything else is -1e30, which
  changes the visited mass and so the confidence. A whole enum option or a
  lookahead is fed with full logits. The table is coarse (a number admits
  `,}]` whatever encloses it). (trace; this is how a colliding name like
  `get_weather__1` is reached at all)
- **done** Lookahead at an argument value's start (`0x5ce7c`): the first
  four grammar-valid tokens are each continued greedily (grammar-valid
  argmax over full logits) until the top-level value ends or 24 tokens,
  scored by mean log-probability (2 less if unfinished), and the best is
  committed with its full-softmax probabilities, which feed the confidence
  unrenormalized. It runs when the top candidate is under 0.95 or the value
  is grounded, and never for a string-option value. (trace: `temperature`
  40 against a range, where it commits `20` at probability ~1e-9. Ghidra
  shows the branch comparison against the previous branch; the asm is a
  running `fmaxnm`, so it is the plain maximum.)
- **partial** An enum value whose options the conversation names is decoded
  as a whole: right after the value's opening quote, every option that the
  system text or any input so far mentions (even a lone one) is
  teacher-forced as `option` + `"`, every token fed on its own with full
  logits (the last one too), and the likeliest is committed the same way.
  Its first token's probability comes from the step's logits as they
  stand (sparse after a sampled token). (lldb: a later turn scored
  `dishwasher` because an earlier turn had asked for it)
  The library also enters this path at the value's start (sequences with a
  leading `"`), anywhere inside the string, and for numeric options; this
  port only does the first case, which is the one `":"` tokens produce.
- **done** The byte grammar (`FUN_00068610`, `FUN_00070510`,
  `FUN_00072248`), exactly: no whitespace; `name` then `arguments`; an
  identical repeat of a call is refused; argument keys in any order, each
  once, only declared ones, `}` only when the required ones are in; strict
  JSON numbers; a constrained number (integer, bounded, `multipleOf`) at
  most 20 bytes, its maximum checked on each digit (minimum only at the
  end, so a lone `0` passes under `minimum: 1` until it ends); string
  options and `maxLength` checked per character, exact match, `minLength`
  and every regex when the string closes; `id`/`*_id`/`grounded` values
  must be substrings of the conversation text, and a numeric one needs a
  digit in it; `minItems`/`maxItems` at `]`; `uniqueItems` on raw element
  text; `anyOf` readings forked (up to 16) and pruned per byte. Arrays stop
  taking items after seven identical ones in a row or 64 items, except
  that bare numbers are never counted (the library's quirk). The lexer
  takes a trailing comma. (unit tests; probes)
- **done** The schema compiler (`FUN_0002fd50`): `type` string or array (an
  array expands into variants), `enum` (replaced only when every option is a
  scalar), `const`, bounds of both drafts, `multipleOf` (also filtering
  number options), `pattern`, `format` (`uuid date time date-time email
  ipv4 ipv6 duration hostname` as fixed regexes), `items` (object form),
  `allOf` (merged), `anyOf`/`oneOf` (one object is inlined; variants are
  dropped when any is unconstrained or when the node has other
  constraints), `$ref` by last path segment into one `$defs`/`definitions`
  object, a node budget of 8192 and depth 32. Values mined from
  descriptions (`'a', 'b'` quoted tokens, or double-quoted after "one of",
  "options", "can be", ...; nothing after "e.g.", "such as", "for
  example"; not for required-by-flag or grounded fields). Unbounded numbers
  named `level`, `volume`, `*brightness*`, `*percent*` get [0, 100],
  `rating` [1, 5], `*minutes*`, `*seconds*`, `*hours*`, `*duration*`,
  `*count*` a minimum of 0. The root only reads `properties` and
  `required`. (unit tests; probes)
- **done** The regex dialect: ECMAScript-like, search semantics on code
  points, `\d \w \s` and negations, classes, `{n,m}` up to 64, `(?:)`;
  lookarounds, backreferences, `\b` and other escaped letters are errors and
  the pattern is dropped, as is one over 512 program states. (unit tests)
- **todo** `triggers`: a tool's trigger regexes that match the request force
  a call to one of the triggered tools and bypass the confidence gate. The
  grammar supports the allowlist and the forced call; matching is not
  wired.

## // NUMERICS

Code: `infer.rs` (the packed forward pass), `infer/step.rs` (the same for
one token as a single team job), `qlinear.rs` (the matmul kernels). Every
item was checked bit for bit against buffers dumped from the running
library with lldb (layer 0 of a prefill, a decode step with all 8,192
logits, the engram path, 1-, 2- and 4-chunk attention, sparse logits).

- **done** Quantizers: every int8 scale is `amax * K127` (`K127` = 1/127
  rounded to f32, the engine never divides), round half away from zero,
  saturate to -128..127. Activations: per 128-group FWHT, times
  `1/sqrt(128)`, then that quantizer.
- **done** Matmuls: integer sums are exact; three float combines. Decode
  (one token): four float lanes per row, lane `(p % 64) / 16` for 2-bit
  and `(p % 32) / 8` for 4-bit, scale `(sx * cbs) * norm`, lanes summed
  `(l0 + l1) + (l2 + l3)`. Prefill: one fma chain over groups, scale
  `(norm * cbs) * sx`, except the chunk's last `T % 4` tokens, which use
  `norm * (cbs * sx)`. Sparse logits: the prefill form for one token.
  Norms are the stored f16. Codes stay in the archive's LSB-first layout
  and the activations are de-interleaved to match (2-bit
  `act'[k*32 + j] = act[4j + k]`, 4-bit `act'[k*64 + j] = act[2j + k]`),
  which is what fixes the decode lanes. (asm `0x48aac`, `0x48f28`)
- **done** Norms: `dot16` (four 4-lane fma accumulators, combined
  `((A2+A3)+A1)+A0`, then `(l0+l1)+(l2+l3)`), `1 / sqrtf(ss / n + 1e-6)`,
  output `fma(x*r, s, x*r)`.
- **done** mHC: the gate projection is int8 rows (each dequantized row
  requantized) against one int8 scale per token over all 3,072 inputs, no
  rotation; gates `((p*a) + off) + b` through libm `expf`; residual logits
  fused; Sinkhorn in f64 with libm `exp`, 20 rounds of rows then columns;
  every mix an fma chain.
- **done** Attention: int8 q (per head, `1/sqrtf(48)` folded into its
  scale), int8 K and V per (position, kv head); scores `(ks*dot)*qs` in
  blocks of four and `(ks*qs)*dot` for the rest; softmax over key chunks
  with the engine's inline exp and libm `expf` for the last one to three;
  weights quantized to 21 bits and applied as three 7-bit digit sums;
  chunks merged with libm `expf`. Chunking follows the engine's team of
  four: a decode step with 256 or more keys uses four chunks, a prefill
  row one or two.
- **done** RoPE: frequencies `powf(theta, i * (-2/48))`; chunk rows use
  `sincosf(f * p)`; a single token right after the newest position
  advances by the angle-addition recurrence (so decode angles drift from
  `sincosf`, as in the engine).
- **done** Engram: values kept int8 per 32-wide group and read back for
  the dilated conv, the current position included; the gate normalizes
  both vectors and uses `(-dot)/sqrtf(768)` through libm `expf`.
- **done** MLP: the conditioning logits accumulate in four sets by
  `j % 4`; Kronecker stages are `(A^T Z) B`; SiLU `z / (exp(-z) + 1)`.
- **done** Slot map: the engine records which position each KV slot holds
  and attention skips a position whose slot another one has taken since.
  A rolled-back feed (option scoring, a lookahead) leaves its keys in the
  ring, so the replayed positions see fewer keys, and a later prompt feed
  first re-feeds from the first stale position it would attend, with a
  zeroed conv history (`FUN_6c88`). Context overflow (`FUN_67a50`) keeps
  the prefix and tools block in place and re-feeds the newest history
  after them, also with a zeroed conv history and fresh RoPE angles.
- **done** Per-turn retrieval is enabled only when the model ships a
  dedicated embedding readout (`0x2468`); `needle3.cact` does not, so its
  toolsets of any size run in static mode. The KV splice the engine does
  on a selection change is specified (`scratchpad/ghidra/spec_retrieval.md`)
  but not ported, since nothing reaches it.
- **done** Feed pattern: prompt feeds go in chunks of at most 32 from the
  feed start; everything the step loop commits goes in one token at a
  time: the forced `<think>`, `</think>`, the newline and `<tool_call>`,
  each enum option token (scoring and commit), each rollout token, and
  the stop tokens (fed at once, without logits). KV lives in the engine's
  slot ring (sink, then ring positions modulo 256), which the engine never
  restores after scoring or a rollout; the conv history and RoPE state are
  snapshotted and restored.

## // CONFIDENCE

- **done** The confidence head is not consulted: the library never loads it
  (`+0xb68` is null at the gate). Confidence is the smallest call-token
  probability. (lldb)
- **done** A call token's probability is the engine's sampler rule: visit
  candidates in logit order, summing full-softmax mass; stop once a
  candidate is chosen and the visited mass reaches 0.99 or 24 more have
  been visited; the chosen token's probability is renormalized over the
  visited candidates that fit. (disassembly of the visitor at `0x6905c`)
- **done** With no call tokens, confidence is the geometric mean over every
  generated token (0 when there are none). (disassembly)
- **done** Written as `%.4f`.

## // POST-PROCESSING

After a user turn decodes a non-empty call list, the library runs it through
one function, `FUN_0006998c`, before building the envelope. The port is
`crates/needle-engine/src/rules/`: `mod.rs` is the pipeline, with `gates.rs`,
`repairs.rs`, `tail.rs`, `dates.rs`, `verbs.rs` and `validation.rs` for the
rules, `json.rs` and `schema.rs` for the library's own JSON value and tool
table, and `text.rs` for the shared byte helpers. It works on bytes the way
the library does: only ASCII is ever lowercased, numbers keep their source
text, and object members keep their order and duplicates.

Evidence: the Ghidra decompile and the disassembly of `FUN_0006998c`, its
callees and the turn loop around it, then an oracle. `tests/parity/rules_oracle.py` calls the native function directly
(it isn't exported, so it's found from `needle_init`'s address and called
through a small C shim, because it returns a `std::string` through `x8`).
Every rule below matches the oracle byte for byte, flags included, on 237
real decodes from the bundled environments and probes, 218 hand-written
cases aimed at each rule, 24,000 random cases and 12,000 focused ones
(split-place merges, names, dates, routes). `tests/rules_oracle.rs` replays
1,099 of those recorded cases (`tests/parity/rules_cases.jsonl`) on every
`cargo test`.

### Order

1. Parse. Anything but a non-empty array comes back verbatim.
2. Merge a place query split over two consecutive calls (`FUN_00078b38`).
3. Per call with a string `name` and an object `arguments`:
   first/last name split; polarity rename; bool/enum flip; origin copied to
   destination; the enum, control/place and quantity gates; the
   per-argument rebuild; the tail repairs; `.0` on number-typed integers.
4. The cross-call gates: exclusion, negation, reported command.
5. Date resolution.
6. Drop duplicate calls (compact JSON text, first one wins), then write
   compact JSON.

"Repairs on" means the request isn't blank. The rename, the three per-call
gates after the origin check, the tail repairs and the cross-call gates need
it; everything else always runs.

### Context

- **done** The request, the stored system text, the running conversation
  and the tools. The stored system text has every `YYYY-MM-DD[T ]HH:MM`
  timestamp blanked and the `date: ...` fact (up to the next `;`) cut to one
  space, so `date: 2026-09-25 Fri 10:00` leaves `" "`. The conversation is
  that text plus `"\n" + input` for every turn, tool results included,
  trimmed to the last 12 KiB once it passes 16 KiB. (`needle_init` 0x3848,
  `FUN_0000790c`, lldb)
- **done** "Squashed" text for grounding checks: conversation (or system,
  when the conversation is empty) + `"\n"` + request, ASCII-lowercased, with
  spaces, `-` and `_` removed.

### Gates (calls withheld into `suppressed_calls`)

Two flags come back. Any gate sets the first; the three cross-call gates
also set the second. The turn withholds when there's a call list and
`confidence < 0.1` or the first flag is set. (A trigger-forced call would
use only the second flag; triggers aren't ported.) (`0x61c60`)

- **done** Origin equals destination (trimmed, lowercased): a required
  origin withholds, an optional one is erased. Always on.
- **done** A required enum with two or more options set to one the request
  never names. Names containing `category type kind priority sentiment
  intent format unit metric status action mode level` are exempt; stems and
  verb-table values count as named.
- **done** A required non-enum `string` slot holding a control word (`on
  off toggle up down dim ...`), or a place slot (`room place venue area zone
  city site location region` as a name word) holding a number, a month,
  weekday, unit, currency or pronoun, or text the squashed conversation
  doesn't contain. Control-slot names are exempt.
- **done** The first required number with no enum and no default, when the
  squashed conversation has no digit and the request has no quantity word
  (92 words: numbers, `half`, `today`, `week`, `max`, `all`, `mute`, ...).
- **done** Exclusion: up to four words after `except`, `but not`, `other
  than`, `skip`, ... (to the next `, . ; ! ?` or ` and `/` but `/...). Calls
  whose short string values match an excluded phrase are dropped; if every
  call matches, the list is withheld.
- **done** Negation: a cue (`don't`, `do not`, `dont`, `never`, `no longer`,
  `must not`, `mustn't`, `shouldn't`, `should not`, `make sure not`, `no
  need`; not after `forget`, `hesitate`, `worry`, `mind`, `disturb`)
  followed within three words by a word of the tool's name. Name words drop
  anything under three letters and `get set the and for with`, so
  `set_timer` is just `timer`. A negation is forgiven when a clause break
  follows with more text and none of the call's values sit inside the
  negated span. Same drop-or-withhold rule as exclusion.
- **done** Reported command: `said`, `texted`, `wrote`, `told me`, ...
  followed within 40 bytes by any quote mark, apostrophes included.

### Repairs

- **done** Split a full name in a first/last pair, fill a missing half from
  the request (`save Jane Smith's number`), and drop leading honorifics.
- **done** Rename a call to its opposite-polarity sibling tool when the
  request's verb says so (`lock_door` for "unlock the door").
- **done** Flip a bool or on/off-style enum to the request's verb (lock,
  on/off, open/close, mute, increase/decrease rows), and a
  start/stop/pause/resume enum to the one transport verb the request uses.
- **done** File extensions to MIME types (46 extensions) when the key or
  description says `mime`.
- **done** Complete a truncated phone number from the request, or take the
  request's formatting of the same digits. Any key containing `phone`,
  `mobile`, `tel`, `number` or `fax`, enums included.
- **done** Restore a quoted title or body verbatim (straight and curly
  quotes; a truncated prefix counts).
- **done** Recover a place query: the request span from its first to last
  matched word (filler words allowed in between), then a following `, City`,
  an `at/in/on/near` address run, a `which is located at` clause or a place
  noun after a closing quote, tidied up. Also merges a query the model split
  over two calls.
- **done** Restore request casing for free text when the request's span has
  more capitals.
- **done** Capitalize the first byte of `title`, `subject`, `event_name`,
  `event_title`, `extra_message`, `message`, `note`, `label`.
- **done** Celsius to fahrenheit and back, when the key or description
  names the unit and the request states the value in the other one.
- **done** Drop empty optional strings, and optional `*mail*` strings with
  no `@`.
- **done** Drop optional non-enum strings the conversation never mentions
  (dates, times, ids and anything under three letters exempt) and optional
  numbers it never states (unless the request has a quantity word).
- **done** A required name-like string that isn't grounded becomes the
  request's single leftover phrase.
- **done** Defaults stand in for a string that's just a tool-name word, for
  a required number with no quantity anywhere, and an optional number equal
  to its default is dropped when its digits appear nowhere.
- **done** Snap an enum to the one option (three letters or more) the
  request names as a whole word.
- **done** Origin and destination slots filled from `from X`, `to Y`,
  `pick me up at X`, `drop me off at Y` when the model's value isn't
  grounded.
- **done** A missing sole numeric parameter filled from `by N`.
- **done** `number`/`float`/`double`-typed integers (and arrays of them) get
  `.0`.
- **done** Dates: the date part of ISO-shaped values under date-ish keys is
  rewritten from `next/this/coming <weekday>`, bare weekdays, `March 4th`,
  `4th of July`, `today`, `tomorrow`, `day after tomorrow` and ISO dates in
  the request. Today comes from the first ISO date in the stored system
  text, so the standard `date:` fact never drives it: only an ISO date in
  free text does. (probe: `Today is 2026-09-25.` resolves `on Friday` to
  2026-10-02; `date: 2026-09-25 Fri 10:00` leaves the model's date alone)
- **done** Drop duplicate calls.

### Validation

- **done** `ungrounded`: `tool.key` for each string (squashed, over two
  bytes) or number the squashed conversation doesn't contain, in call and
  argument order. Numbers match as standalone digits, with thousands commas,
  or as a number word (`dozen`, `half`). Parameters with an enum, a `const`
  or `"grounded": true` are exempt. (`0x608fc`)
- **done** `negation`: a cue anywhere in the request as a substring (so
  `whenever` counts), unless `mind`/`forget`/... follows. (`FUN_0006b138`)
- **done** `validation` is present exactly when a call list is delivered or
  suppressed, and left out otherwise.

### Where the specs were wrong (decided by the oracle)

- Tool-name words always drop words under three letters and `get set the
  and for with` (`FUN_0009fffc` runs inside `FUN_0009f0b8`), for the rename
  and the leftover phrase too. `set_lights` is just `lights`.
- The exclusion phrase skips `the/my/a/an` only before its first kept word:
  `except for the kitchen` gives `forthekitchen`, not `forkitchen`.
- The place cleanup's trailing-word drop is case-sensitive (`1 ON` stays);
  only the `on the map` check lowercases.
- "Enumerated parameters are never touched" (model card) only holds for the
  rebuild's free-text repairs. Phone completion, the empty drop, the flip
  and the enum snap all change enums.

### Not ported

- **todo** Enum values mined from descriptions (the context's `+0x60`
  list). Our grammar doesn't mine them yet, so such a parameter isn't
  treated as an enum here either.
- **todo** Triggers (a trigger-forced call ignores confidence and the first
  flag).
- **partial** The envelope carries numbers through `serde_json`, so a
  non-canonical number the model wrote (`0.50`, `1e3`) comes out canonical
  where the library copies the text.
