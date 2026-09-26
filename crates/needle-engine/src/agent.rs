//! The tool-calling agent: the engine behind `needle.Needle`.
//!
//! One agent holds one toolset and one conversation. The static prefix
//! (system facts and the rendered tools) is prefilled once; each turn
//! appends the user text (or a tool result) in the chat wire format, lets
//! the model reason freely inside `<think>`, then decodes the call list under
//! the schema grammar. The reply is the JSON envelope the reference engine
//! returns.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use needle_core::Tokenizer;
use needle_core::pyjson::dumps_compact;
use needle_core::render::{IM_END, IM_START, TOOL_RESULT_END, TOOL_RESULT_START, TOOLS_END, TOOLS_START};
use needle_core::tokenizer::{BOS_ID, EOS_ID, IM_END_ID, PieceKind, THINK_END_ID, THINK_START_ID, TOOL_CALL_END_ID, TOOL_CALL_START_ID};
use serde_json::{Map, Value, json};

use crate::rules::json::Json;

use crate::grammar::{Grammar, State};
use crate::heads;
use crate::model::{Model, Outputs, Session};
use crate::rules::{self, Context, names_option};
use crate::toolset::{self, Toolset};

/// Calls below this confidence are withheld into `suppressed_calls`.
pub const SUPPRESS_BELOW: f64 = 0.1;
/// Declared tools beyond this count are retrieved per turn.
pub const RETRIEVE_TOP: usize = 5;

pub struct Agent {
    model: Arc<Model>,
    tok: Arc<Tokenizer>,
    /// The normalized tools (what the prompt and the post-processor see).
    tools: Vec<Value>,
    toolset: Toolset,
    /// Each token's text as the engine steps it through the grammar.
    pieces: Vec<Vec<u8>>,
    /// Token ids by the first byte of their text, for the sparse logits of
    /// call decoding.
    by_first_byte: Vec<Vec<u32>>,
    /// The system text as the prompt carries it (`needle_init`'s `S`).
    system: String,
    session: Session,
    prefix: Session,
    /// Tool embeddings for retrieval (large catalogues only).
    tool_vecs: Vec<Vec<f32>>,
    /// Retrieval: the tools the current tools block holds, and its length
    /// in tokens at the start of the history.
    selection: Vec<usize>,
    tools_block: usize,
    last_was_call: bool,
    turns: usize,
    /// Tokens that close the previous turn, fed with the next turn's
    /// prompt instead of as decode steps of their own.
    /// The tools as the post-processor reads them.
    rule_tools: Vec<rules::schema::Tool>,
    /// The system text with the `date:` fact blanked (the library's
    /// grounding base), and the running conversation built on it.
    grounding: Vec<u8>,
    conversation: Vec<u8>,
    /// Tools that must be called when a pattern matches the request
    /// (case-insensitive), by normalized name.
    triggers: Vec<(Vec<u8>, Vec<fancy_regex::Regex>)>,
}

/// The running conversation text is cut back to its last 12 KiB once it
/// passes 16 KiB.
const CONVERSATION_MAX: usize = 0x4000;
const CONVERSATION_KEEP: usize = 0x3000;

/// Python-compatible rounding for the envelope's floats.
fn round(v: f64, digits: i32) -> f64 {
    let p = 10f64.powi(digits);
    (v * p).round() / p
}

impl Agent {
    pub fn new(model: Arc<Model>, tok: Arc<Tokenizer>, tools: Vec<Value>, system: &str) -> Result<Self> {
        Self::from_json(model, tok, dumps_compact(&Value::Array(tools)).as_bytes(), system)
    }

    /// `needle_init`: the tools JSON and system text exactly as the caller
    /// passed them.
    pub fn from_json(model: Arc<Model>, tok: Arc<Tokenizer>, tools_json: &[u8], system: &str) -> Result<Self> {
        let session = model.session();
        let toolset = Toolset::normalize(tools_json);
        let system = String::from_utf8_lossy(&toolset::normalize_system(system.as_bytes())).into_owned();
        let rule_tools = rules::schema::parse_tools(&toolset.text);
        let grounding = rules::grounding_text(&system);
        let tools: Vec<Value> = toolset.tools.iter().map(rules::json::to_serde).collect();
        // A token's text: control and unknown pieces have none, a byte piece
        // is its byte, others are their text with `▁` as a space.
        let pieces: Vec<Vec<u8>> = (0..tok.vocab_size() as u32)
            .map(|id| match tok.kind(id) {
                PieceKind::Unknown | PieceKind::Control => vec![],
                PieceKind::Byte => tok.decode_bytes(&[id]),
                _ => tok.piece(id).replace('\u{2581}', " ").into_bytes(),
            })
            .collect();
        let mut by_first_byte = vec![vec![]; 256];
        for (id, piece) in pieces.iter().enumerate() {
            if let Some(&b) = piece.first() {
                by_first_byte[b as usize].push(id as u32);
            }
        }
        let triggers = toolset
            .triggers
            .iter()
            .map(|(name, pats)| {
                let compiled = pats
                    .iter()
                    .filter_map(|p| {
                        let compiled = if ecmascript_group_ok(p) {
                            fancy_regex::RegexBuilder::new(p).case_insensitive(true).build().ok()
                        } else {
                            None
                        };
                        if compiled.is_none() {
                            eprintln!("needle: ignoring invalid trigger for {}: {p}", String::from_utf8_lossy(name));
                        }
                        compiled
                    })
                    .collect();
                (name.clone(), compiled)
            })
            .collect();
        let mut agent = Self {
            model: model.clone(),
            tok,
            tools,
            toolset,
            triggers,
            pieces,
            by_first_byte,
            system,
            session: session.clone(),
            prefix: session,
            tool_vecs: vec![],
            selection: vec![],
            tools_block: 0,
            last_was_call: false,
            turns: 0,
            rule_tools,
            conversation: grounding.clone(),
            grounding,
        };
        // The engine ranks tools per turn only when the model ships a
        // dedicated embedding readout (`0x2468`); the confidence head's pool
        // serves `needle_embed`, but not retrieval.
        if agent.tools.len() > RETRIEVE_TOP && model.w.embedding_head.is_some() {
            let texts: Vec<String> =
                agent.toolset.tools.iter().map(|t| String::from_utf8_lossy(&toolset::canonical(t)).into_owned()).collect();
            agent.tool_vecs = texts.iter().map(|t| agent.embed(t)).collect::<Result<_>>()?;
        }
        let text = agent.prefix_text();
        let ids = agent.encode_with_bos(&text);
        if ids.len() >= model.config().max_seq_len {
            bail!(
                "needle_init context budget exceeded: static prefix is {} tokens, model context is {} tokens; reduce system/tools JSON or declare more than {RETRIEVE_TOP} tools to enable per-turn retrieval",
                ids.len(),
                model.config().max_seq_len
            );
        }
        if std::env::var_os("NEEDLE_PROMPT_IDS").is_some() {
            eprintln!("PREFIX {ids:?}");
        }
        let mut s = model.session();
        // The engine attends to the static prefix plus a ring of recent
        // positions (the archive's `kv_window`).
        s.attend = agent.attention_span(ids.len());
        model.prefill(&mut s, &ids, Outputs::None);
        agent.prefix = s.clone();
        agent.session = s;
        Ok(agent)
    }

    /// Tokens of the static prefix (what `needle_init` returns).
    pub fn prefix_tokens(&self) -> usize {
        self.prefix.len()
    }

    /// The system block: empty for an empty system text, which is otherwise
    /// kept as it is (no trimming).
    fn system_block(&self) -> String {
        if self.system.is_empty() { String::new() } else { format!("{IM_START}system\n{}{IM_END}\n", self.system) }
    }

    /// The static prefix. With retrieval the tools move into each turn and
    /// the prefix is the system block alone; otherwise the user header
    /// (with no newline: the first turn supplies it) and the tools block.
    fn prefix_text(&self) -> String {
        if !self.tool_vecs.is_empty() {
            return self.system_block();
        }
        let tools = if self.toolset.text.is_empty() {
            String::new()
        } else {
            format!("\n{TOOLS_START}{}{TOOLS_END}", String::from_utf8_lossy(&self.toolset.text))
        };
        format!("{}{IM_START}user{tools}", self.system_block())
    }

    fn encode_with_bos(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![BOS_ID];
        ids.extend(self.tok.encode(text));
        ids
    }

    /// Rewind to the static prefix, keeping the tools.
    pub fn reset(&mut self) {
        self.session = self.prefix.clone();
        self.last_was_call = false;
        self.turns = 0;
        self.conversation = self.grounding.clone();
        self.selection.clear();
        self.tools_block = 0;
    }

    /// `needle_embed`: the unit-norm probe-pool vector of a text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.encode_with_bos(text);
        let cells = heads::cells(&self.model, &ids);
        heads::embedding(&self.model, &cells, ids.len()).ok_or_else(|| anyhow::anyhow!("this archive carries no probe head"))
    }

    /// The tools rendered for a turn: all of them, or the top five by
    /// embedding similarity, in declaration order.
    fn select_tools(&self, text: &str) -> Result<Vec<usize>> {
        if self.tool_vecs.is_empty() {
            return Ok((0..self.tools.len()).collect());
        }
        let q = self.embed(text)?;
        let mut scored: Vec<(f32, usize)> =
            self.tool_vecs.iter().enumerate().map(|(i, v)| (v.iter().zip(&q).map(|(a, b)| a * b).sum(), i)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<usize> = scored.iter().take(RETRIEVE_TOP).map(|s| s.1).collect();
        keep.sort_unstable();
        Ok(keep)
    }

    /// The grammar-constrained greedy choice after `call_ids` and its
    /// probability: the highest logit whose text keeps the call list
    /// completable (`</tool_call>` once it is complete); `None` when nothing
    /// fits.
    ///
    /// The probability is the engine's: candidates are visited in logit order
    /// until one fits, then on until the visited softmax mass reaches 0.99 or
    /// 24 more have been seen, and the choice is renormalized over the
    /// visited candidates that fit. A choice with no fitting rival in view
    /// scores exactly 1.
    ///
    /// `sparse` is whether the engine fed the previous token with sparse
    /// logits (a sampled call token or `<tool_call>`; not a whole enum
    /// option, which is fed with full logits).
    fn constrained(&self, grammar: &Grammar, call_ids: &[u32], logits: &[f32], sparse: bool) -> Option<(u32, f64)> {
        let body = self.tok.decode_bytes(call_ids);
        // Each candidate's bytes are stepped on a copy of the grammar state.
        let state = crate::prof::span("c_state", || grammar.state(&body))?;
        let complete = state.complete();
        let mut masked = crate::prof::span("c_step_logits", || self.step_logits(&state, logits, sparse));
        // The softmax behind each visited candidate's mass is over the
        // step's logits (base bans in place); the call-step bans below only
        // touch the sampler's working copy.
        let (m, z) = crate::prof::span("c_softmax", || softmax_stats(&masked));
        // The ordinary call step also bans `<|im_end|>`, `<think>`,
        // `</think>` and `<tool_call>`, and the token that would start a
        // seventh repeat of a cycle of one to four tokens.
        for id in [IM_END_ID, THINK_START_ID, THINK_END_ID, TOOL_CALL_START_ID] {
            masked[id as usize] = -1e30;
        }
        for id in cycle_bans(call_ids) {
            if (id as usize) < masked.len() {
                masked[id as usize] = -1e30;
            }
        }
        let logits = masked.as_slice();
        let accepts = |cand: u32| -> bool {
            if cand == TOOL_CALL_END_ID {
                return complete;
            }
            if matches!(self.tok.kind(cand), PieceKind::Control | PieceKind::UserDefined | PieceKind::Unknown) {
                return false;
            }
            if !body.is_empty() {
                // A token's bytes follow the body's unchanged.
                let piece = &self.pieces[cand as usize];
                return !piece.is_empty() && state.accepts(piece);
            }
            let mut ids = call_ids.to_vec();
            ids.push(cand);
            let text = self.tok.decode_bytes(&ids);
            text.len() > body.len() && text.starts_with(&body) && state.accepts(&text[body.len()..])
        };
        let prob = |c: u32| (logits[c as usize] as f64 - m as f64).exp() / z;
        // Candidates in logit order: the top 128 by partial selection, then
        // the rest only if the scan runs past them.
        // Unmasked candidates by logit (lowest id first on ties), then the
        // masked ones by id, sorted only if the scan gets that far.
        let mut order: Vec<u32> = (0..logits.len() as u32).filter(|&i| logits[i as usize] > -1e30).collect();
        let live = order.len();
        let by_logit =
            |a: &u32, b: &u32| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b));
        let head = 128.min(live);
        if live > head {
            order.select_nth_unstable_by(head - 1, by_logit);
        }
        order[..head].sort_unstable_by(by_logit);
        let (mut mass, mut valid, mut extra) = (0f64, 0f64, 0usize);
        let mut chosen: Option<(u32, f64)> = None;
        let mut i = 0;
        let mut masked_in = false;
        loop {
            if i == order.len() {
                // Past every live candidate: the masked ones follow by id.
                if masked_in {
                    break;
                }
                masked_in = true;
                order.extend((0..logits.len() as u32).filter(|&c| logits[c as usize] <= -1e30));
                continue;
            }
            if i == head && live > head {
                order[head..live].sort_unstable_by(by_logit);
            }
            if chosen.is_some() {
                if mass >= 0.99 || extra >= 24 {
                    break;
                }
                extra += 1;
            } else if i >= head + 512 {
                break;
            }
            let c = order[i];
            let p = prob(c);
            mass += p;
            if debug_flag(&CALL_VISIT, "NEEDLE_CALL_VISIT") {
                eprintln!("    visit {c} {:?} p {p:.6} ok {}", self.tok.decode(&[c]), accepts(c));
            }
            if accepts(c) {
                valid += p;
                chosen = chosen.or(Some((c, p)));
            }
            i += 1;
        }
        match chosen {
            Some((c, p)) => Some((c, if valid > 0.0 { p / valid } else { 1.0 })),
            None => complete.then_some((TOOL_CALL_END_ID, 1.0)),
        }
    }

    /// The logits the engine holds after feeding the previous token: when
    /// that feed was sparse, only the tokens whose first byte the grammar
    /// can take next (plus `</tool_call>` and `<|im_end|>`), if that is
    /// under half the vocabulary, scored from the token's hidden state with
    /// the single-row kernel; the rest sit at -1e30.
    ///
    /// Empty `logits` means the feed returned only its hidden state; the full
    /// logits are then computed only if the step needs them.
    fn engine_logits(&self, state: &State, logits: &[f32], sparse: bool) -> Vec<f32> {
        let hidden = self.session.nat.as_ref().map(|n| &n.last_hidden).filter(|h| !h.is_empty());
        let vocab = self.model.config().out_rows();
        if let Some(next) = state.next_bytes().filter(|_| sparse) {
            let n: usize = (0..256).filter(|&b| next[b]).map(|b| self.by_first_byte[b].len()).sum();
            if 2 * n < vocab {
                let mut ids: Vec<u32> = (0..256).filter(|&b| next[b]).flat_map(|b| self.by_first_byte[b].iter().copied()).collect();
                ids.extend([TOOL_CALL_END_ID, IM_END_ID]);
                if let Some(h) = hidden {
                    return self.model.head_logits_sparse(h, &ids);
                }
                let mut masked = vec![-1e30; logits.len()];
                for &id in &ids {
                    masked[id as usize] = logits[id as usize];
                }
                return masked;
            }
        }
        match hidden {
            Some(h) if logits.is_empty() => self.model.head_logits(h),
            _ => logits.to_vec(),
        }
    }

    /// The logits a call step chooses from: [`Agent::engine_logits`] with
    /// the always-banned ids at -1e30.
    fn step_logits(&self, state: &State, logits: &[f32], sparse: bool) -> Vec<f32> {
        let mut masked = self.engine_logits(state, logits, sparse);
        for id in [4usize, 8, 9, 12, 13] {
            if id < masked.len() {
                masked[id] = -1e30;
            }
        }
        masked
    }

    /// The engine's lookahead at the start of an argument value
    /// (`0x5ce7c`): the first four grammar-valid tokens in logit order are
    /// each continued greedily (grammar-valid argmax, full logits) until
    /// the top-level value ends or 24 tokens, and scored by their mean
    /// log-probability, 2 less when unfinished; the best wins, the first on
    /// ties. It runs when the top candidate is under 0.95 or the value is
    /// grounded.
    ///
    /// Returns the winner's tokens, their probabilities (full softmax, not
    /// renormalized), and the logits after them; the session is left after
    /// the winner.
    fn rollout(
        &mut self,
        grammar: &Grammar,
        call_ids: &[u32],
        logits: &[f32],
        sparse: bool,
        left: usize,
    ) -> Option<(Vec<u32>, Vec<f64>, Vec<f32>)> {
        let body = self.tok.decode_bytes(call_ids);
        let state = grammar.state(&body)?;
        if !state.at_value_start()
            || state.pending_string_options()
            || left < 29
            || self.session.len() + 32 >= self.model.config().max_seq_len
        {
            return None;
        }
        let banned = [IM_END_ID, THINK_START_ID, THINK_END_ID, TOOL_CALL_START_ID, TOOL_CALL_END_ID];
        let prob_of = |l: &[f32], c: u32| -> f64 {
            let lc = l[c as usize];
            1.0 / l.iter().filter(|&&v| v > -1e30).map(|&v| ((v - lc) as f64).exp()).sum::<f64>()
        };
        // Up to four grammar-valid first tokens, in logit order.
        let masked = self.step_logits(&state, logits, sparse);
        let mut work = masked.clone();
        for id in banned {
            work[id as usize] = -1e30;
        }
        let mut cands: Vec<(u32, f64)> = vec![];
        while cands.len() < 4 {
            let c = argmax_excluding(&work, &[]);
            if work[c as usize] <= -1e30 {
                break;
            }
            work[c as usize] = -1e30;
            let piece = &self.pieces[c as usize];
            if !piece.is_empty() && state.accepts(piece) {
                cands.push((c, prob_of(&masked, c)));
            }
        }
        if cands.len() < 2 || !(cands[0].1 < 0.95 || state.pending_grounded()) {
            return None;
        }
        // Each branch (at most 24 tokens) is rolled back afterwards, except
        // the last one when it wins.
        let (mut best, mut top) = (None, -1e30f64);
        let mut branches: Vec<(Vec<u32>, Vec<f64>)> = vec![];
        for (i, &(c, p)) in cands.iter().enumerate() {
            let mut st = state.clone();
            st.feed(&self.pieces[c as usize]);
            let (mut toks, mut probs) = (vec![c], vec![p]);
            let mut sum = p.ln();
            let mut l = self.model.forward(&mut self.session, &[c], Outputs::LastLogits).data;
            while !st.value_closed() && toks.len() < 24 {
                let mut w = l.clone();
                for id in banned {
                    w[id as usize] = -1e30;
                }
                let next = loop {
                    let t = argmax_excluding(&w, &[]);
                    if w[t as usize] <= -1e30 {
                        break None;
                    }
                    w[t as usize] = -1e30;
                    let piece = &self.pieces[t as usize];
                    if !piece.is_empty() && st.accepts(piece) {
                        break Some(t);
                    }
                };
                let Some(t) = next else { break };
                st.feed(&self.pieces[t as usize]);
                let pt = prob_of(&l, t);
                toks.push(t);
                probs.push(pt);
                sum += pt.ln();
                l = self.model.forward(&mut self.session, &[t], Outputs::LastLogits).data;
            }
            let mut score = sum / toks.len() as f64;
            if !st.value_closed() {
                score -= 2.0;
            }
            if debug_flag(&CALL_PROBS, "NEEDLE_CALL_PROBS") {
                eprintln!("  branch {i} {toks:?} {:?} score {score:.4} closed {}", self.tok.decode(&toks), st.value_closed());
            }
            if score > top {
                best = Some(i);
                top = score;
            }
            let last = i + 1 == cands.len();
            if last && best == Some(i) {
                // The winner's state is already in place; `l` holds the
                // logits after its last token.
                branches.push((toks, probs));
                let (toks, probs) = branches.swap_remove(i);
                return Some((toks, probs, l));
            }
            self.session.rollback(&self.model, toks.len());
            branches.push((toks, probs));
        }
        let (toks, probs) = branches.swap_remove(best?);
        let next = self.feed_each(&toks);
        if debug_flag(&CALL_PROBS, "NEEDLE_CALL_PROBS") {
            eprintln!("  rollout {:?} {:?} p {probs:?}", toks, self.tok.decode(&toks));
        }
        Some((toks, probs, next))
    }

    /// Teacher-force each option sequence from the current state and return
    /// the likeliest (first on ties) with its share of the options' total
    /// likelihood. The session is left as it was.
    fn score_options(&mut self, seqs: &[Vec<u32>], logits: &[f32]) -> (usize, f64) {
        let mut scores = Vec::with_capacity(seqs.len());
        for seq in seqs {
            let mut score = token_prob_excluding(logits, seq[0], &[]).ln();
            // The engine feeds every option token on its own (the last one
            // too, though nothing reads its logits), then restores its
            // state, but not the KV ring: those slots keep the option's keys.
            for (k, &t) in seq.iter().enumerate() {
                let want = if k + 1 < seq.len() { Outputs::LastLogits } else { Outputs::None };
                let row = self.model.forward(&mut self.session, &[t], want).data;
                if k + 1 < seq.len() {
                    score += token_prob_excluding(&row, seq[k + 1], &[]).ln();
                }
            }
            self.session.rollback(&self.model, seq.len());
            scores.push(score);
        }
        let mut best = 0;
        for (i, &s) in scores.iter().enumerate() {
            if s > scores[best] {
                best = i;
            }
        }
        let z: f64 = scores.iter().filter(|&&s| s > -1e29).map(|s| (s - scores[best]).exp()).sum();
        (best, if z > 0.0 { 1.0 / z } else { 1.0 })
    }

    /// The engine's `(sink, ring)` attention span for a sink of `sink`
    /// positions, when the model declares a ring (`kv_window`).
    fn attention_span(&self, sink: usize) -> Option<(usize, usize)> {
        let ring = self.model.config().kv_window;
        (ring > 0).then_some((sink, ring))
    }

    /// The engine's context overflow (`FUN_67a50`): before `n` more tokens
    /// would pass the context, keep the static prefix and the retrieval
    /// tools block, and of the conversation only the most recent tokens that
    /// fit, re-prefilled right after them.
    fn make_room(&mut self, n: usize) {
        let ctx = self.model.config().max_seq_len;
        if self.session.len() + n <= ctx {
            return;
        }
        let p = self.prefix.len();
        let history = self.session.tokens[p.min(self.session.len())..].to_vec();
        let block = self.tools_block.min(history.len());
        let keep = ctx.saturating_sub(n + p + block).min(history.len() - block);
        // The engine keeps the prefix and tools block where they are and
        // re-feeds the newest history after them, with a zeroed conv history
        // (`FUN_67a50`).
        let kept = history[history.len() - keep..].to_vec();
        self.session.rewind_zeroed(&self.model, p + block);
        self.model.prefill(&mut self.session, &kept, Outputs::None);
    }

    /// Feed `toks` one at a time, as the engine's step loop does, and
    /// return the logits after the last.
    fn feed_each(&mut self, toks: &[u32]) -> Vec<f32> {
        let mut logits = vec![];
        for (k, &t) in toks.iter().enumerate() {
            self.make_room(1);
            // Only the last token's logits are read.
            let want = if k + 1 == toks.len() { Outputs::LastLogits } else { Outputs::None };
            logits = self.model.forward(&mut self.session, &[t], want).data;
        }
        logits
    }

    /// The id of `text` when it is a single token.
    fn single_token(&self, text: &str) -> Option<u32> {
        match self.tok.encode(text).as_slice() {
            [t] => Some(*t),
            _ => None,
        }
    }

    /// One turn: the envelope as JSON.
    pub fn complete(&mut self, text: &str, max_new_tokens: usize) -> Result<Value> {
        let is_result = self.last_was_call && looks_like_result(text);
        self.conversation.push(b'\n');
        self.conversation.extend_from_slice(text.as_bytes());
        if self.conversation.len() > CONVERSATION_MAX {
            self.conversation.drain(..self.conversation.len() - CONVERSATION_KEEP);
        }
        let retrieval = !self.tool_vecs.is_empty();
        // A tool result keeps the tools the call came from.
        let keep = if retrieval && is_result && !self.selection.is_empty() { self.selection.clone() } else { self.select_tools(text)? };
        let tools_text = if retrieval { self.toolset.subset_text(&keep) } else { self.toolset.text.clone() };
        // Triggers (user turns without a negation cue): the matching tools
        // become the only calls allowed, and the list cannot be empty.
        let triggered: Vec<Vec<u8>> = if is_result || text.is_empty() || rules::validation::negation_cue(text.as_bytes()) {
            vec![]
        } else {
            self.triggers
                .iter()
                .filter(|(_, pats)| pats.iter().any(|r| r.is_match(text).unwrap_or(false)))
                .map(|(n, _)| n.clone())
                .collect()
        };
        let mut grammar = Grammar::from_text(&tools_text).with_context(&self.conversation).must_call(is_result || !triggered.is_empty());
        if !triggered.is_empty() {
            grammar = grammar.allow_only(triggered.clone());
        }

        // The turn as the engine renders it. A tool result goes in a `tool`
        // turn; outside retrieval it is compacted and an object is wrapped
        // in a list.
        let turn_text = if is_result {
            let result = if retrieval {
                text.to_string()
            } else {
                let j = String::from_utf8_lossy(&toolset::compact_ws(text.as_bytes())).into_owned();
                if j.starts_with('{') { format!("[{j}]") } else { j }
            };
            format!("\n{IM_START}tool\n{TOOL_RESULT_START}{result}{TOOL_RESULT_END}{IM_END}\n{IM_START}assistant\n")
        } else if self.turns == 0 {
            format!("\n{text}{IM_END}\n{IM_START}assistant\n")
        } else {
            format!("\n{IM_START}user\n{text}{IM_END}\n{IM_START}assistant\n")
        };
        let t0 = Instant::now();
        let mut turn_ids = vec![];
        if retrieval {
            let p = self.prefix.len();
            let t_ids = self.tok.encode(&turn_text);
            let ctx = self.model.config().max_seq_len;
            let fits = self.session.len() + turn_ids.len() + max_new_tokens + t_ids.len() <= ctx;
            if self.session.len() > p && keep == self.selection && fits {
                // Same tools as last turn: continue the conversation.
                turn_ids.extend(t_ids);
            } else {
                // Rebuild: the tools block for this selection, the history
                // without the old block (at most one ring of it), then the
                // turn. The tools block joins the attention sink.
                let block = self.tok.encode(&format!("{IM_START}user\n{TOOLS_START}{}{TOOLS_END}", String::from_utf8_lossy(&tools_text)));
                let mut history = self.session.tokens[p.min(self.session.len())..].to_vec();
                history.append(&mut turn_ids);
                let tail = &history[self.tools_block.min(history.len())..];
                let fixed = p + max_new_tokens + block.len();
                let avail = ctx.saturating_sub(fixed);
                let t_ids = &t_ids[t_ids.len().saturating_sub(avail)..];
                let ring = self.model.config().kv_window.max(1);
                let cap = ctx.saturating_sub(t_ids.len() + fixed).min(ring);
                let tail = &tail[tail.len().saturating_sub(cap)..];
                self.session = self.prefix.clone();
                self.session.attend = self.attention_span(p + block.len());
                self.tools_block = block.len();
                self.selection = keep.clone();
                turn_ids = [block.as_slice(), tail, t_ids].concat();
            }
        } else {
            turn_ids.extend(self.tok.encode(&turn_text));
        }
        if std::env::var_os("NEEDLE_PROMPT_IDS").is_some() {
            eprintln!("TURN {turn_ids:?} at {} attend {:?}", self.session.len(), self.session.attend);
        }
        // A user turn always opens with `<think>` (the engine forces it as
        // its first step); after a tool result the model picks its own first
        // token.
        let user_turn = !is_result;
        // An input longer than the context allows keeps its end (the engine
        // truncates silently), and old history makes room for it.
        let limit = self.model.config().max_seq_len.saturating_sub(max_new_tokens + self.prefix.len()).max(1);
        if turn_ids.len() > limit {
            turn_ids.drain(..turn_ids.len() - limit);
        }
        self.make_room(turn_ids.len());
        let mut logits = crate::prof::span("turn_prefill", || self.model.prefill(&mut self.session, &turn_ids, Outputs::LastLogits));
        let prefill_secs = t0.elapsed().as_secs_f64();
        let prompt_tokens = turn_ids.len();
        if user_turn {
            logits = self.feed_each(&[THINK_START_ID]);
        }
        self.turns += 1;

        let t1 = Instant::now();
        // The engine's step machine (`0x5b35c`): `step` counts committed
        // decisions against `max_new_tokens`; `generated` counts tokens.
        let max_new = max_new_tokens;
        let mut step = usize::from(user_turn);
        let mut generated = step;
        // Every generated token's log-probability, for the confidence of a
        // turn without a call list.
        let (mut log_sum, mut log_count) = (0f64, 0usize);
        let nl = match self.tok.encode("\n").as_slice() {
            [t] => Some(*t),
            _ => None,
        };

        // Reasoning, and after a tool result the model's own opening.
        let mut reasoning_ids = vec![];
        let mut prose_ids = vec![];
        let t_think = Instant::now();
        let mut in_think = user_turn;
        let mut in_call = false;
        let mut ended = false;
        // Stop tokens the engine feeds (without logits) as the turn ends.
        let mut stop: Vec<u32> = vec![];
        let mut last: Option<u32> = None;
        while step < max_new {
            let mut work = logits.clone();
            ban(&mut work, &BASE_BANS);
            let tok = if in_think {
                ban(&mut work, &[THINK_START_ID, TOOL_CALL_START_ID, TOOL_CALL_END_ID]);
                if max_new - step <= 32 { THINK_END_ID } else { argmax(&work) }
            } else {
                ban(&mut work, &[THINK_END_ID, TOOL_CALL_END_ID]);
                if generated > 0 {
                    ban(&mut work, &[THINK_START_ID]);
                }
                if !prose_ids.is_empty() {
                    ban(&mut work, &[TOOL_CALL_START_ID]);
                }
                argmax(&work)
            };
            if tok == EOS_ID || tok == IM_END_ID {
                // The turn ends here with no call.
                stop = vec![tok];
                ended = true;
                break;
            }
            // A user turn's `</think>` brings the forced newline and
            // `<tool_call>`, each its own step.
            if tok == THINK_END_ID && user_turn {
                let mut closing = vec![THINK_END_ID];
                closing.extend(nl);
                closing.push(TOOL_CALL_START_ID);
                step += closing.len();
                generated += closing.len();
                logits = self.feed_each(&closing);
                in_call = true;
                break;
            }
            match tok {
                THINK_START_ID => in_think = true,
                THINK_END_ID => in_think = false,
                TOOL_CALL_START_ID => in_call = true,
                _ if last == Some(THINK_END_ID) && Some(tok) == nl => {}
                _ => {
                    log_sum += turn_token_prob(&work, tok).ln();
                    log_count += 1;
                    if in_think { reasoning_ids.push(tok) } else { prose_ids.push(tok) }
                }
            }
            step += 1;
            generated += 1;
            last = Some(tok);
            self.make_room(1);
            logits = self.model.forward(&mut self.session, &[tok], Outputs::LastLogits).data;
            if in_call {
                break;
            }
        }
        let think_secs = t_think.elapsed().as_secs_f64();
        let t_call = Instant::now();

        // The call list under the grammar.
        let mut call_ids: Vec<u32> = vec![];
        let mut call_probs: Vec<f64> = vec![];
        let mut closed = false;
        let mut sparse = true;
        while in_call && !ended && step < max_new {
            // Near the end of the budget the engine closes what it can.
            if max_new - step <= 4 {
                let body = self.tok.decode_bytes(&call_ids);
                if let Some(state) = grammar.state(&body) {
                    if state.complete() {
                        closed = true;
                        break;
                    }
                    if let Some(rb) = self.single_token("]")
                        && state.accepts(b"]")
                        && grammar.state(&[body.as_slice(), b"]"].concat()).is_some_and(|s| s.complete())
                    {
                        call_ids.push(rb);
                        call_probs.push(1.0);
                        log_sum += 0.0;
                        log_count += 1;
                        step += 1;
                        generated += 1;
                        self.make_room(1);
                        logits = self.model.forward(&mut self.session, &[rb], Outputs::LastLogits).data;
                        continue;
                    }
                }
            }
            // An enum value whose options the request names: score each named
            // option as a whole (teacher-forced) and commit the likeliest,
            // as the engine does.
            if let Some(options) = grammar.enum_open(&self.tok.decode_bytes(&call_ids)) {
                // An option counts as named when the conversation so far (the
                // system text and every input) mentions it, not just this
                // turn.
                let hay = String::from_utf8_lossy(&self.conversation);
                let named: Vec<&String> = options.iter().filter(|o| names_option(&hay, o)).collect();
                if !named.is_empty() {
                    let seqs: Vec<Vec<u32>> = named
                        .iter()
                        .map(|o| {
                            let mut t = self.tok.encode(o);
                            t.extend(self.tok.encode("\""));
                            t
                        })
                        .collect();
                    // Even a lone option is scored: its feeds overwrite KV
                    // ring slots the engine never restores.
                    let base = match grammar.state(&self.tok.decode_bytes(&call_ids)) {
                        Some(st) => self.step_logits(&st, &logits, sparse),
                        None => logits.clone(),
                    };
                    let (best, p) = self.score_options(&seqs, &base);
                    let ids = &seqs[best];
                    if debug_flag(&CALL_PROBS, "NEEDLE_CALL_PROBS") {
                        eprintln!("  named option {ids:?} {:?} p {p:.6}", named[best]);
                    }
                    call_probs.push(p);
                    call_probs.extend(std::iter::repeat_n(1.0, ids.len() - 1));
                    log_sum += p.ln();
                    log_count += ids.len();
                    step += 1;
                    generated += ids.len();
                    call_ids.extend_from_slice(ids);
                    let ids = ids.clone();
                    logits = self.feed_each(&ids);
                    sparse = false;
                    continue;
                }
            }
            if let Some((toks, probs, next)) = self.rollout(&grammar, &call_ids, &logits, sparse, max_new.saturating_sub(generated)) {
                for &p in &probs {
                    log_sum += p.ln();
                }
                call_probs.extend_from_slice(&probs);
                log_count += toks.len();
                // A whole rollout costs one budget step.
                step += 1;
                generated += toks.len();
                call_ids.extend_from_slice(&toks);
                logits = next;
                sparse = false;
                continue;
            }
            let Some((next, p)) = crate::prof::span("sampler", || self.constrained(&grammar, &call_ids, &logits, sparse)) else {
                // Nothing fits: the engine ends the turn with `<|im_end|>`,
                // leaving the list unterminated.
                ended = true;
                break;
            };
            sparse = true;
            if debug_flag(&CALL_PROBS, "NEEDLE_CALL_PROBS") {
                eprintln!("  call token {next} {:?} p {p:.6}", self.tok.decode(&[next]));
            }
            if next == TOOL_CALL_END_ID {
                closed = true;
                break;
            }
            call_probs.push(p);
            log_sum += p.ln();
            log_count += 1;
            call_ids.push(next);
            step += 1;
            generated += 1;
            self.make_room(1);
            // Fed sparsely: the next step computes the logits it needs from
            // the hidden state.
            let out = if self.model.native() { Outputs::LastHidden } else { Outputs::LastLogits };
            logits = self.model.forward(&mut self.session, &[next], out).data;
            if out == Outputs::LastHidden {
                logits.clear();
            }
        }
        let min_prob = call_probs.iter().copied().fold(1f64, f64::min);
        let call_tokens = call_probs.len();
        // A list the budget or a dead end cut short keeps the calls whose
        // arguments closed; with none, the turn reports truncation.
        let mut truncated = false;
        let mut body = self.tok.decode(&call_ids);
        if in_call && !closed {
            let bytes = self.tok.decode_bytes(&call_ids);
            let done: Vec<Vec<u8>> = grammar.state(&bytes).map(|s| s.emitted().to_vec()).unwrap_or_default();
            if done.is_empty() {
                truncated = true;
            } else {
                let parts: Vec<String> = done
                    .iter()
                    .map(|k| {
                        let cut = k.iter().position(|&c| c == b'|').unwrap_or(k.len());
                        let (name, args) = (&k[..cut], &k[(cut + 1).min(k.len())..]);
                        format!("{{\"name\":\"{}\",\"arguments\":{}}}", String::from_utf8_lossy(name), String::from_utf8_lossy(args))
                    })
                    .collect();
                body = format!("[{}]", parts.join(","));
            }
        }
        if ended && in_call {
            stop = vec![IM_END_ID];
        } else if !ended && in_call && closed {
            stop = vec![TOOL_CALL_END_ID, IM_END_ID];
        }
        for &t in &stop {
            self.make_room(1);
            self.model.forward(&mut self.session, &[t], Outputs::None);
        }
        let decode_secs = t1.elapsed().as_secs_f64();
        let call_secs = t_call.elapsed().as_secs_f64();
        if std::env::var_os("NEEDLE_TURN_TIMES").is_some() {
            eprintln!(
                "turn: prefill {:.1}ms ({} tok)  think {:.1}ms ({} tok)  call {:.1}ms ({} tok)",
                prefill_secs * 1e3,
                prompt_tokens,
                think_secs * 1e3,
                reasoning_ids.len() + 3,
                call_secs * 1e3,
                call_ids.len() + 2
            );
        }
        let has_call = !call_ids.is_empty();

        let body = body.trim_start_matches(['\t', '\n', '\r', ' ']).trim_end_matches(['\t', '\n', '\r', ' ']);
        // `ok`: nothing to parse, or the call text parses as JSON.
        let ok = !truncated && (!has_call || rules::json::parse(body.as_bytes()).is_some());
        let valid = ok && has_call;
        // The post-processor: repairs, gates, dates and the duplicate drop.
        let (post, withhold, withhold_strong) = if valid && body != "[]" && !is_result {
            let ctx =
                Context { request: text.as_bytes(), system: &self.grounding, conversation: &self.conversation, tools: &self.rule_tools };
            let out = rules::postprocess(body.as_bytes(), &ctx);
            if std::env::var_os("NEEDLE_RULES_TRACE").is_some() {
                eprintln!("rules: {body}\n    -> {} withhold={}", String::from_utf8_lossy(&out.calls), out.withhold);
            }
            (out.calls, out.withhold, out.withhold_all)
        } else {
            (body.as_bytes().to_vec(), false, false)
        };
        let post_calls = match rules::json::parse(&post) {
            Some(Json::Arr(items)) => items,
            _ => vec![],
        };
        // The calls in the caller's own tool and property names.
        let calls: Vec<Value> = match rules::json::parse(&self.toolset.renames.restore(&post)) {
            Some(Json::Arr(items)) => items.iter().map(rules::json::to_serde).collect(),
            _ => vec![],
        };
        // With no reasoning and no call, a prose answer is the reasoning.
        let mut reasoning = self.tok.decode(&reasoning_ids).trim().to_string();
        if reasoning.is_empty() && !has_call {
            reasoning = self.tok.decode(&prose_ids).trim().to_string();
        }

        // Confidence: the weakest call-token probability (the engine's
        // definition), or the geometric mean over the turn without a call list.
        let confidence = if call_tokens > 0 {
            min_prob
        } else if log_count > 0 {
            (log_sum / log_count as f64).exp()
        } else {
            0.0
        };
        // Calls are delivered unless a gate fired or confidence is under 0.1.
        let deliver = valid && post.as_slice() != b"[]";
        // A triggered call is withheld only by the strong gates (negation,
        // exclusion, reported speech); otherwise also under the confidence
        // floor or a weak gate.
        let withhold = deliver && if triggered.is_empty() { confidence < SUPPRESS_BELOW || withhold } else { withhold_strong };
        let validation = if deliver {
            let ungrounded = rules::validation::ungrounded_fields(&post_calls, &self.rule_tools, &self.conversation);
            json!({"ungrounded": ungrounded, "negation": rules::validation::negation_cue(text.as_bytes())})
        } else {
            Value::Null
        };
        let (function_calls, suppressed) = if withhold { (vec![], calls.clone()) } else { (calls.clone(), vec![]) };
        let kind = if has_call { "call" } else { "respond" };
        self.last_was_call = !function_calls.is_empty();

        let mut env = Map::new();
        env.insert("type".into(), json!(kind));
        env.insert("success".into(), json!(ok));
        let (error, code) = if truncated {
            (json!("tool call truncated: token budget exhausted"), json!("truncated"))
        } else if !ok {
            (json!(format!("malformed function_calls JSON: {body}")), json!("malformed"))
        } else {
            (Value::Null, Value::Null)
        };
        env.insert("error".into(), error);
        env.insert("error_code".into(), code);
        env.insert("reason".into(), if ok { Value::Null } else { json!("runtime_failure") });
        env.insert("function_calls".into(), Value::Array(function_calls));
        env.insert("suppressed_calls".into(), Value::Array(suppressed));
        env.insert("reasoning".into(), if reasoning.is_empty() { Value::Null } else { json!(reasoning) });
        env.insert("confidence".into(), json!(round(confidence, 4)));
        env.insert("prefill_tps".into(), json!(round(prompt_tokens as f64 / prefill_secs.max(1e-9), 1)));
        env.insert("decode_tps".into(), json!(round(generated as f64 / decode_secs.max(1e-9), 1)));
        env.insert("peak_ram_mb".into(), json!(round(peak_ram_mb(), 1)));
        if !validation.is_null() {
            env.insert("validation".into(), validation);
        }
        Ok(Value::Object(env))
    }
}

/// Whether every `(?` group is one ECMAScript `std::regex` knows (`(?:`,
/// `(?=`, `(?!`): inline flags, lookbehind and named groups make the
/// engine drop the pattern.
fn ecmascript_group_ok(pattern: &str) -> bool {
    let b = pattern.as_bytes();
    let mut i = 0;
    while i + 1 < b.len() {
        if b[i] == b'\\' {
            i += 2;
            continue;
        }
        if b[i] == b'(' && b[i + 1] == b'?' && !matches!(b.get(i + 2), Some(b':' | b'=' | b'!')) {
            return false;
        }
        i += 1;
    }
    true
}

/// Ids the engine masks on every step (`<|im_start|>` and the tools and
/// tool-result markers).
const BASE_BANS: [u32; 5] = [4, 8, 9, 12, 13];

fn ban(logits: &mut [f32], ids: &[u32]) {
    for &id in ids {
        if let Some(v) = logits.get_mut(id as usize) {
            *v = -1e30;
        }
    }
}

/// The first index of the maximum (`fmaxnm`, then the first equal value).
fn argmax(x: &[f32]) -> u32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x.iter().position(|&v| v == m).unwrap_or(0) as u32
}

/// The engine's `expf` (`0x5f2e8`): Cephes, clamped at ±88.376, ties-even
/// rounding and fused multiply-adds.
fn engine_expf(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    let x = x.clamp(-88.37626, 88.37626);
    let n = (x * LOG2E).round_ties_even();
    let x = n.mul_add(-0.693_359_4, x);
    let x = n.mul_add(2.121_944_4e-4, x);
    let mut p = 1.987_569_1e-4f32;
    p = p.mul_add(x, 1.398_199_9e-3);
    p = p.mul_add(x, 8.333_452e-3);
    p = p.mul_add(x, 4.166_579_6e-2);
    p = p.mul_add(x, 1.666_666_5e-1);
    p = p.mul_add(x, 0.5);
    let y = (x * x).mul_add(p, x);
    let two_n = f32::from_bits(((n as i32 + 127) as u32) << 23);
    two_n.mul_add(y, two_n)
}

/// Per-lane sums of `engine_expf(v - lt)` over groups of four.
fn lane_exp_sums(v: &[f32], lt: f32) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; loads stay inside `v`.
    unsafe {
        use std::arch::aarch64::*;
        let (lo, hi) = (vdupq_n_f32(-88.37626), vdupq_n_f32(88.37626));
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let (c1, c2) = (vdupq_n_f32(-0.693_359_4), vdupq_n_f32(2.121_944_4e-4));
        let p = [1.987_569_1e-4f32, 1.398_199_9e-3, 8.333_452e-3, 4.166_579_6e-2, 1.666_666_5e-1, 0.5].map(|c| vdupq_n_f32(c));
        let one_bits = vdupq_n_s32(0x3f80_0000);
        let base = vdupq_n_f32(lt);
        let exp4 = |ptr: *const f32| {
            let x = vminq_f32(vmaxq_f32(vsubq_f32(vld1q_f32(ptr), base), lo), hi);
            let n = vrndnq_f32(vmulq_f32(x, log2e));
            let x = vfmaq_f32(x, n, c1);
            let x = vfmaq_f32(x, n, c2);
            let mut q = vfmaq_f32(p[1], p[0], x);
            q = vfmaq_f32(p[2], q, x);
            q = vfmaq_f32(p[3], q, x);
            q = vfmaq_f32(p[4], q, x);
            q = vfmaq_f32(p[5], q, x);
            let y = vfmaq_f32(x, vmulq_f32(x, x), q);
            let two_n = vreinterpretq_f32_s32(vaddq_s32(vshlq_n_s32::<23>(vcvtq_s32_f32(n)), one_bits));
            vfmaq_f32(two_n, two_n, y)
        };
        // Four groups' exponentials at a time for parallelism, added in
        // order so each lane sums exactly as one accumulator would.
        let mut acc = vdupq_n_f32(0.0);
        let (quads, rest) = v.as_chunks::<16>();
        for q16 in quads {
            let p0 = q16.as_ptr();
            let (e0, e1, e2, e3) = (exp4(p0), exp4(p0.add(4)), exp4(p0.add(8)), exp4(p0.add(12)));
            acc = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(acc, e0), e1), e2), e3);
        }
        for chunk in rest.as_chunks::<4>().0 {
            acc = vaddq_f32(acc, exp4(chunk.as_ptr()));
        }
        let mut out = [0f32; 4];
        vst1q_f32(out.as_mut_ptr(), acc);
        return out;
    }
    #[allow(unreachable_code)]
    {
        let mut lanes = [0f32; 4];
        for chunk in v.as_chunks::<4>().0 {
            for (a, &x) in lanes.iter_mut().zip(chunk) {
                *a += engine_expf(x - lt);
            }
        }
        lanes
    }
}

/// A free token's probability as the engine commits it (`0x5f278`):
/// `1 / sum exp(l_j - l_tok)`, four lanes in f32 over whole groups of four,
/// the rest in f64 skipping masked logits.
fn turn_token_prob(logits: &[f32], tok: u32) -> f64 {
    let lt = logits[tok as usize];
    let main = logits.len() / 4 * 4;
    let lanes = lane_exp_sums(&logits[..main], lt);
    let mut z = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) as f64;
    for &v in &logits[main..] {
        if v > -1e30 {
            z += (v as f64 - lt as f64).exp();
        }
    }
    if z > 0.0 { 1.0 / z } else { 1.0 }
}

static CALL_VISIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static CALL_PROBS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether debug variable `name` is set (read once).
fn debug_flag(cell: &std::sync::OnceLock<bool>, name: &str) -> bool {
    *cell.get_or_init(|| std::env::var_os(name).is_some())
}

fn looks_like_result(text: &str) -> bool {
    let t = text.trim_start();
    t.starts_with('{') || t.starts_with('[')
}

/// The tokens that would extend a cycle of period 1 to 4 that has already
/// repeated six times at the end of `h` (`0x5e5d8`).
fn cycle_bans(h: &[u32]) -> Vec<u32> {
    let n = h.len();
    (1..=4usize).filter(|&p| n >= 6 * p && (n - 5 * p..n).all(|k| h[k] == h[k - p])).map(|p| h[n - p]).collect()
}

/// `softmax(logits)[id]` without materialising the distribution.
/// The max logit and `sum exp(l - max)` as the call sampler has them: four
/// f32 lanes of the engine's exp, then an f64 libm tail skipping masked
/// logits.
fn softmax_stats(logits: &[f32]) -> (f32, f64) {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let main = logits.len() / 4 * 4;
    let lanes = lane_exp_sums(&logits[..main], m);
    let mut z = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) as f64;
    for &v in &logits[main..] {
        if v > -1e30 {
            z += (v as f64 - m as f64).exp();
        }
    }
    (m, z)
}

/// `softmax(logits)[id]` with `banned` masked out.
fn token_prob_excluding(logits: &[f32], id: u32, banned: &[u32]) -> f64 {
    let l = logits[id as usize];
    let z: f64 = logits.iter().enumerate().filter(|(i, _)| !banned.contains(&(*i as u32))).map(|(_, &v)| ((v - l) as f64).exp()).sum();
    1.0 / z
}

fn argmax_excluding(x: &[f32], banned: &[u32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        if v > bv && !banned.contains(&(i as u32)) {
            bv = v;
            best = i as u32;
        }
    }
    best
}

/// `_with_date_fact`: prefix the local `date:` fact unless the system text
/// already carries a date (or is a JSON object).
pub fn with_date_fact(system: &str) -> String {
    let has_iso = system
        .as_bytes()
        .windows(10)
        .any(|w| w[4] == b'-' && w[7] == b'-' && w.iter().enumerate().all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit()));
    if system.contains("date:") || has_iso {
        return system.to_string();
    }
    let fact = chrono::Local::now().format("date: %Y-%m-%d %a %H:%M").to_string();
    if system.trim().is_empty() {
        return fact;
    }
    if system.trim_start().starts_with('{') {
        return system.to_string();
    }
    format!("{fact}; {system}")
}

/// The envelope as the engine writes it: compact JSON with the confidence
/// at four decimals (`"confidence":1.0000`).
pub fn envelope_json(env: &Value) -> String {
    let s = env.to_string();
    let key = ",\"confidence\":";
    let (Some(at), Some(c)) = (s.rfind(key), env.get("confidence").and_then(Value::as_f64)) else { return s };
    let start = at + key.len();
    let end = s[start..].find([',', '}']).map_or(s.len(), |e| start + e);
    format!("{}{c:.4}{}", &s[..start], &s[end..])
}

fn peak_ram_mb() -> f64 {
    crate::cpu::rusage().map_or(0.0, |r| {
        let kb = if cfg!(target_os = "macos") { r.maxrss as f64 / 1024.0 } else { r.maxrss as f64 };
        kb / 1024.0
    })
}

#[cfg(test)]
mod bench_prob {
    #[test]
    #[ignore]
    fn bench_turn_token_prob() {
        let logits: Vec<f32> = (0..8192).map(|i| ((i * 7919 % 1000) as f32) / 100.0 - 5.0).collect();
        let n = 2000;
        let t = std::time::Instant::now();
        let mut acc = 0f64;
        for i in 0..n {
            acc += super::turn_token_prob(std::hint::black_box(&logits), (i % 8192) as u32);
        }
        // The NEON lanes agree with the scalar engine_expf exactly.
        let lt = logits[17];
        let mut lanes = [0f32; 4];
        for chunk in logits.as_chunks::<4>().0 {
            for (a, &x) in lanes.iter_mut().zip(chunk) {
                *a += super::engine_expf(x - lt);
            }
        }
        assert_eq!(super::lane_exp_sums(&logits, lt), lanes);
        eprintln!("turn_token_prob {:.1} us ({acc:.3})", t.elapsed().as_secs_f64() / n as f64 * 1e6);
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(super::argmax(std::hint::black_box(&logits)));
        }
        eprintln!("argmax {:.1} us", t.elapsed().as_secs_f64() / n as f64 * 1e6);
    }
}
