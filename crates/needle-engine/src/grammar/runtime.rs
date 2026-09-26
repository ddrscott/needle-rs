//! The byte automaton (`FUN_00068610` and what it calls): the call-list
//! state machine, the JSON lexer for `arguments` (`FUN_00070510`), the
//! decoded scalar buffer (`FUN_00070d3c`, `FUN_00071eb8`), the schema stacks
//! (`FUN_00070ebc`, `FUN_00072248`, `FUN_00072a1c`, `FUN_00072d34`,
//! `FUN_00073410`), and the array run limits.
//!
//! A [`State`] takes one byte at a time and says whether it fits; a token
//! fits when all its bytes do. Nothing looks ahead: a value that can no
//! longer end validly is only refused at the byte that would end it.

use super::Grammar;
use super::compile::{Node, NodeId, ty};

/// Call-list states (`G+8`).
pub mod kind {
    pub const START: u8 = 0;
    pub const LIST: u8 = 1;
    pub const NAME_KEY: u8 = 2;
    pub const NAME: u8 = 3;
    pub const ARGS_KEY: u8 = 4;
    pub const ARGS: u8 = 5;
    pub const CALL_END: u8 = 6;
    pub const AFTER_CALL: u8 = 7;
    pub const NEXT_CALL: u8 = 8;
    pub const DONE: u8 = 9;
}

// Lexer token states.
const TOK_NONE: u8 = 0;
const TOK_STR: u8 = 1;
const TOK_ESC: u8 = 2;
const TOK_HEX: u8 = 3;
const TOK_NUM: u8 = 4;
const TOK_LIT: u8 = 5;

// Lexer structural states.
const ST_VALUE: u8 = 0;
const ST_KEY: u8 = 1;
const ST_COLON: u8 = 2;
const ST_OBJ_NEXT: u8 = 3;
const ST_ARR_NEXT: u8 = 4;

// Frame scalar modes.
const MODE_NONE: u8 = 0;
const MODE_KEY: u8 = 1;
const MODE_STRING: u8 = 2;
const MODE_NUMBER: u8 = 3;
const MODE_GROUNDED_STRING: u8 = 4;
const MODE_GROUNDED_NUMBER: u8 = 5;

/// A strict JSON lexer with no whitespace. After `,` it goes back to
/// "key or `}`" (objects) or "value or `]`" (arrays), so on its own it
/// takes a trailing comma.
#[derive(Clone, Debug, Default, PartialEq)]
struct Lexer {
    /// `o` and `a` for each open object and array.
    stack: Vec<u8>,
    tok: u8,
    st: u8,
    num: u8,
    hex: u8,
    lit: &'static [u8],
    lit_pos: usize,
    is_key: bool,
    done: bool,
}

impl Lexer {
    fn after_value(&mut self) -> bool {
        match self.stack.last() {
            None => self.done = true,
            Some(&top) => self.st = if top == b'o' { ST_OBJ_NEXT } else { ST_ARR_NEXT },
        }
        true
    }

    fn step(&mut self, c: u8) -> bool {
        if self.done {
            return false;
        }
        match self.tok {
            TOK_STR => {
                if c == b'"' {
                    self.tok = TOK_NONE;
                    if self.is_key {
                        self.is_key = false;
                        self.st = ST_COLON;
                        return true;
                    }
                    return self.after_value();
                }
                if c == b'\\' {
                    self.tok = TOK_ESC;
                    return true;
                }
                c > 0x1f
            }
            TOK_ESC => match c {
                b'b' | b'f' | b'n' | b'r' | b't' | b'"' | b'/' | b'\\' => {
                    self.tok = TOK_STR;
                    true
                }
                b'u' => {
                    self.tok = TOK_HEX;
                    self.hex = 4;
                    true
                }
                _ => false,
            },
            TOK_HEX => {
                if !c.is_ascii_hexdigit() {
                    return false;
                }
                self.hex -= 1;
                if self.hex == 0 {
                    self.tok = TOK_STR;
                }
                true
            }
            TOK_LIT => {
                if self.lit.get(self.lit_pos) != Some(&c) {
                    return false;
                }
                self.lit_pos += 1;
                if self.lit_pos < self.lit.len() {
                    return true;
                }
                self.tok = TOK_NONE;
                self.after_value()
            }
            TOK_NUM => {
                // Number states: 0 after `-`, 1 integer digits, 2 a leading
                // `0`, 3 after `.`, 4 fraction digits, 5 after `e`, 6 after
                // the exponent sign, 7 exponent digits.
                let d = c.is_ascii_digit();
                match self.num {
                    0 | 3 | 6 => {
                        if !d {
                            return false;
                        }
                        self.num = match self.num {
                            0 if c == b'0' => 2,
                            0 => 1,
                            3 => 4,
                            _ => 7,
                        };
                        return true;
                    }
                    1 | 2 if c == b'.' => {
                        self.num = 3;
                        return true;
                    }
                    1 | 4 | 7 if d => return true,
                    2 if d => return false,
                    5 => {
                        if c == b'-' || c == b'+' {
                            self.num = 6;
                        } else if d {
                            self.num = 7;
                        } else {
                            return false;
                        }
                        return true;
                    }
                    _ => {}
                }
                if matches!(self.num, 1 | 2 | 4) && (c == b'e' || c == b'E') {
                    self.num = 5;
                    return true;
                }
                if !matches!(self.num, 1 | 2 | 4 | 7) {
                    return false;
                }
                // The byte ends the number and is read as structure.
                self.tok = TOK_NONE;
                match self.stack.last() {
                    None => self.done = true,
                    Some(&top) => self.st = if top == b'o' { ST_OBJ_NEXT } else { ST_ARR_NEXT },
                }
                self.structural(c)
            }
            _ => self.structural(c),
        }
    }

    fn structural(&mut self, c: u8) -> bool {
        match self.st {
            ST_VALUE => match c {
                b'"' => {
                    self.tok = TOK_STR;
                    self.is_key = false;
                    true
                }
                b'-' | b'0'..=b'9' => {
                    self.tok = TOK_NUM;
                    self.num = match c {
                        b'-' => 0,
                        b'0' => 2,
                        _ => 1,
                    };
                    true
                }
                b'n' | b't' | b'f' => {
                    self.tok = TOK_LIT;
                    self.lit = match c {
                        b'n' => b"null",
                        b't' => b"true",
                        _ => b"false",
                    };
                    self.lit_pos = 1;
                    true
                }
                b'{' => {
                    self.stack.push(b'o');
                    self.st = ST_KEY;
                    true
                }
                b'[' => {
                    self.stack.push(b'a');
                    self.st = ST_VALUE;
                    true
                }
                b']' if self.stack.last() == Some(&b'a') => {
                    self.stack.pop();
                    self.after_value()
                }
                _ => false,
            },
            ST_KEY => match c {
                b'}' if self.stack.last() == Some(&b'o') => {
                    self.stack.pop();
                    self.after_value()
                }
                b'"' => {
                    self.tok = TOK_STR;
                    self.is_key = true;
                    true
                }
                _ => false,
            },
            ST_COLON => {
                if c != b':' {
                    return false;
                }
                self.st = ST_VALUE;
                true
            }
            ST_OBJ_NEXT | ST_ARR_NEXT => {
                let (close, next) = if self.st == ST_OBJ_NEXT { (b'}', ST_KEY) } else { (b']', ST_VALUE) };
                if c == close {
                    self.stack.pop();
                    return self.after_value();
                }
                if c != b',' {
                    return false;
                }
                self.st = next;
                true
            }
            _ => false,
        }
    }
}

/// The current scalar, decoded: string contents with escapes resolved (a
/// lone surrogate is written as its own three bytes), or a number's text.
#[derive(Clone, Debug, Default, PartialEq)]
struct Scalar {
    buf: Vec<u8>,
    /// Whether the last byte finished a character (not mid-escape).
    complete: bool,
    /// Code points in `buf`.
    cp: usize,
    esc: u8,
    hex: Vec<u8>,
    high: u32,
}

fn push_utf8(out: &mut Vec<u8>, cp: u32) {
    match cp {
        0..=0x7f => out.push(cp as u8),
        0x80..=0x7ff => out.extend([0xc0 | (cp >> 6) as u8, 0x80 | (cp & 0x3f) as u8]),
        0x800..=0xffff => out.extend([0xe0 | (cp >> 12) as u8, 0x80 | ((cp >> 6) & 0x3f) as u8, 0x80 | (cp & 0x3f) as u8]),
        _ => out.extend([
            0xf0 | (cp >> 18) as u8,
            0x80 | ((cp >> 12) & 0x3f) as u8,
            0x80 | ((cp >> 6) & 0x3f) as u8,
            0x80 | (cp & 0x3f) as u8,
        ]),
    }
}

impl Scalar {
    fn flush_high(&mut self) {
        if self.high != 0 {
            push_utf8(&mut self.buf, self.high);
            self.high = 0;
        }
    }

    /// One byte inside a string; whether it completed a character.
    fn decode(&mut self, c: u8) -> bool {
        match self.esc {
            0 => {
                if c == b'\\' {
                    self.esc = 1;
                    return false;
                }
                self.flush_high();
                self.buf.push(c);
                true
            }
            1 => {
                self.esc = 0;
                if c == b'u' {
                    self.esc = 2;
                    self.hex.clear();
                    return false;
                }
                self.flush_high();
                self.buf.push(match c {
                    b'b' => 0x08,
                    b'f' => 0x0c,
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    _ => c,
                });
                true
            }
            _ => {
                if !c.is_ascii_hexdigit() {
                    self.esc = 0;
                    self.hex.clear();
                    self.high = 0;
                    return false;
                }
                self.hex.push(c);
                if self.hex.len() < 4 {
                    self.esc += 1;
                    return false;
                }
                let v = std::str::from_utf8(&self.hex).ok().and_then(|h| u32::from_str_radix(h, 16).ok()).unwrap_or(0);
                self.esc = 0;
                self.hex.clear();
                if v & 0xfc00 == 0xd800 {
                    self.flush_high();
                    self.high = v;
                    return false;
                }
                let v = if v & 0xfc00 == 0xdc00 && self.high != 0 {
                    let pair = 0x10000 + ((self.high - 0xd800) << 10) + (v - 0xdc00);
                    self.high = 0;
                    pair
                } else {
                    self.flush_high();
                    v
                };
                push_utf8(&mut self.buf, v);
                true
            }
        }
    }

    fn step(&mut self, old_tok: u8, new_tok: u8, c: u8) {
        self.complete = true;
        let from = if old_tok != TOK_NONE {
            let from = self.buf.len();
            match old_tok {
                TOK_STR | TOK_ESC | TOK_HEX => {
                    if new_tok == TOK_NONE {
                        self.flush_high();
                    } else {
                        self.complete = self.decode(c);
                    }
                }
                TOK_NUM if new_tok == TOK_NUM => self.buf.push(c),
                _ => return,
            }
            from
        } else {
            match new_tok {
                TOK_NUM => {
                    self.buf.clear();
                    self.buf.push(c);
                }
                TOK_STR => {
                    self.buf.clear();
                    self.esc = 0;
                    self.hex.clear();
                    self.high = 0;
                }
                _ => return,
            }
            self.cp = 0;
            0
        };
        self.cp += self.buf[from..].iter().filter(|&&b| b & 0xc0 != 0x80).count();
    }
}

/// One level of the schema walk.
#[derive(Clone, Debug, PartialEq)]
struct Frame {
    depth: usize,
    node: NodeId,
    seen: [u64; 4],
    /// The schema of the value about to start (or being read).
    pending: Option<NodeId>,
    count: i32,
    mode: u8,
    item_start: usize,
    items_seen: Vec<Vec<u8>>,
}

impl Frame {
    fn new(depth: usize, node: NodeId, nodes: &[Node]) -> Frame {
        let n = &nodes[node];
        let pending = if !n.is_object() { n.items.first().copied() } else { None };
        Frame { depth, node, seen: [0; 4], pending, count: 0, mode: MODE_NONE, item_start: 0, items_seen: vec![] }
    }

    fn seen(&self, i: usize) -> bool {
        i < 256 && self.seen[i >> 6] >> (i & 63) & 1 == 1
    }

    fn mark(&mut self, i: usize) {
        if i < 256 {
            self.seen[i >> 6] |= 1 << (i & 63);
        }
    }

    /// Is some declared property still unseen (`FUN_00073264`)?
    fn any_unseen(&self, n: &Node) -> bool {
        (0..n.props.len()).any(|i| i >= 256 || !self.seen(i))
    }
}

/// Run/size tracking for one open array (`G+0x98`).
#[derive(Clone, Debug, Default, PartialEq)]
struct Tracker {
    depth: usize,
    count: i32,
    run: i32,
    prev: Vec<u8>,
    cur: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct State<'g> {
    g: &'g Grammar,
    pub(super) kind: u8,
    lit: &'static [u8],
    lit_pos: usize,
    name: Vec<u8>,
    tool: Option<usize>,
    /// `name|arguments` of each finished call.
    emitted: Vec<Vec<u8>>,
    started: bool,
    lex: Lexer,
    args: Vec<u8>,
    sc: Scalar,
    stacks: Vec<Vec<Frame>>,
    trackers: Vec<Tracker>,
}

fn is_num_byte(c: u8) -> bool {
    c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.' | b'e' | b'E')
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.is_empty() || hay.windows(needle.len()).any(|w| w == needle)
}

fn prefix_of_any(lits: &[Vec<u8>], buf: &[u8]) -> bool {
    lits.iter().any(|l| l.starts_with(buf))
}

/// `atof`: the longest numeric prefix (0 when there is none).
fn atof(b: &[u8]) -> f64 {
    let s = std::str::from_utf8(b).unwrap_or("");
    (1..=s.len()).rev().find_map(|n| s[..n].parse::<f64>().ok().filter(|_| !s[..n].ends_with(['e', 'E', '+']))).unwrap_or(0.0)
}

impl<'g> State<'g> {
    pub(super) fn new(g: &'g Grammar) -> State<'g> {
        State {
            g,
            kind: kind::START,
            lit: b"",
            lit_pos: 0,
            name: vec![],
            tool: None,
            emitted: vec![],
            started: false,
            lex: Lexer::default(),
            args: vec![],
            sc: Scalar::default(),
            stacks: vec![],
            trackers: vec![],
        }
    }

    /// Whether the call list is complete (`</tool_call>` may follow).
    /// `name|arguments` of every call whose arguments closed, in order: the
    /// engine salvages these when the budget cuts a list short.
    pub fn emitted(&self) -> &[Vec<u8>] {
        &self.emitted
    }

    pub fn complete(&self) -> bool {
        self.kind == kind::DONE
    }

    /// Feed bytes; false (and a state to throw away) at the first that
    /// does not fit.
    pub fn feed(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&c| self.step(c))
    }

    /// Would these bytes fit, from here?
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        self.clone().feed(bytes)
    }

    fn nodes(&self) -> &'g [Node] {
        &self.g.schemas.nodes
    }

    fn step(&mut self, c: u8) -> bool {
        match self.kind {
            kind::START => {
                if c != b'[' {
                    return false;
                }
                self.kind = kind::LIST;
                true
            }
            kind::LIST | kind::NEXT_CALL => match c {
                b'{' => {
                    self.kind = kind::NAME_KEY;
                    self.lit = b"\"name\":\"";
                    self.lit_pos = 0;
                    true
                }
                b']' if self.kind == kind::LIST && !self.g.force => {
                    self.kind = kind::DONE;
                    true
                }
                _ => false,
            },
            kind::NAME_KEY | kind::ARGS_KEY => {
                if self.lit.get(self.lit_pos) != Some(&c) {
                    return false;
                }
                self.lit_pos += 1;
                if self.lit_pos < self.lit.len() {
                    return true;
                }
                if self.kind == kind::NAME_KEY {
                    self.kind = kind::NAME;
                    self.name.clear();
                } else {
                    self.begin_args();
                }
                true
            }
            kind::NAME => {
                let allowed = |n: &[u8]| self.g.allowed.is_empty() || self.g.allowed.iter().any(|a| a == n);
                let tools = &self.g.schemas.tools;
                if c != b'"' {
                    self.name.push(c);
                    return tools.iter().any(|t| allowed(&t.name) && t.name.starts_with(&self.name));
                }
                let Some(i) = tools.iter().position(|t| t.name == self.name && allowed(&t.name)) else { return false };
                self.tool = Some(i);
                self.kind = kind::ARGS_KEY;
                self.lit = b",\"arguments\":";
                self.lit_pos = 0;
                true
            }
            kind::ARGS => self.step_args(c),
            kind::CALL_END => {
                if c != b'}' {
                    return false;
                }
                self.kind = kind::AFTER_CALL;
                true
            }
            kind::AFTER_CALL => match c {
                b',' => {
                    self.kind = kind::NEXT_CALL;
                    true
                }
                b']' => {
                    self.kind = kind::DONE;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn begin_args(&mut self) {
        self.kind = kind::ARGS;
        self.lex = Lexer::default();
        self.started = false;
        self.args.clear();
        self.sc = Scalar::default();
        self.stacks.clear();
        if let Some(t) = self.tool {
            let root = self.g.schemas.tools[t].root;
            self.stacks.push(vec![Frame::new(1, root, self.nodes())]);
        }
    }

    fn validated(&self) -> bool {
        self.tool.is_some_and(|t| self.g.schemas.tools[t].validated)
    }

    fn step_args(&mut self, c: u8) -> bool {
        if !self.started {
            if c != b'{' {
                return false;
            }
            self.started = true;
        }
        let (old_depth, old_tok, old_st) = (self.lex.stack.len(), self.lex.tok, self.lex.st);
        let (mut at_depth, mut at_depth_idle) = (false, false);
        if let Some(top) = self.trackers.last() {
            at_depth_idle = top.depth == old_depth && old_tok == TOK_NONE;
            if c == b',' && at_depth_idle && old_st == ST_ARR_NEXT {
                // Seven identical items in a row, or 64 items, end the array.
                if top.run > 5 || top.count > 63 {
                    return false;
                }
                at_depth = true;
            }
        }
        if !self.lex.step(c) {
            return false;
        }
        let depth = self.lex.stack.len();
        if c == b'[' && old_tok == TOK_NONE && depth == old_depth + 1 {
            self.trackers.push(Tracker { depth, ..Tracker::default() });
        } else if let Some(top) = self.trackers.last_mut() {
            let closing = c == b']' && at_depth_idle;
            if !(at_depth || closing) {
                top.cur.push(c);
            }
            if depth == top.depth && self.lex.st == ST_ARR_NEXT && old_st != ST_ARR_NEXT && self.lex.tok == TOK_NONE {
                let before = top.count;
                top.count += 1;
                top.run = if before >= 1 && top.cur == top.prev { top.run + 1 } else { 0 };
                top.prev = std::mem::take(&mut top.cur);
            }
            if closing {
                self.trackers.pop();
            }
        }
        self.sc.step(old_tok, self.lex.tok, c);
        if self.validated() && !self.schema_step(old_tok, old_st, old_depth, c) {
            return false;
        }
        self.args.push(c);
        if self.lex.done {
            let t = self.tool.map(|t| self.g.schemas.tools[t].name.as_slice()).unwrap_or_default();
            let key = [t, b"|", &self.args].concat();
            if self.emitted.contains(&key) {
                return false;
            }
            self.emitted.push(key);
            self.kind = kind::CALL_END;
        }
        true
    }

    fn schema_step(&mut self, old_tok: u8, old_st: u8, old_depth: usize, c: u8) -> bool {
        let stacks = std::mem::take(&mut self.stacks);
        let mut out: Vec<Vec<Frame>> = Vec::with_capacity(stacks.len());
        for s in stacks {
            self.stack_step(s, old_tok, old_st, old_depth, c, &mut out);
        }
        let mut kept: Vec<Vec<Frame>> = Vec::with_capacity(out.len());
        for s in out {
            if !kept.contains(&s) {
                kept.push(s);
            }
        }
        self.stacks = kept;
        !self.stacks.is_empty()
    }

    /// `FUN_00072248`: one stack through one byte; a surviving stack (or
    /// several, when a value opens an `anyOf`) goes to `out`.
    fn stack_step(&self, mut stack: Vec<Frame>, old_tok: u8, old_st: u8, old_depth: usize, c: u8, out: &mut Vec<Vec<Frame>>) {
        let nodes = self.nodes();
        while stack.len() > 1 && stack.last().is_some_and(|f| f.depth > old_depth) {
            stack.pop();
        }
        let Some(top) = stack.last_mut() else {
            out.push(stack);
            return;
        };
        if top.depth != old_depth {
            out.push(stack);
            return;
        }
        let n = &nodes[top.node];
        let is_obj = n.is_object();
        if is_obj && (old_st == ST_KEY || old_st == ST_OBJ_NEXT) {
            top.pending = None;
        } else if !is_obj && old_st == ST_ARR_NEXT {
            top.pending = n.items.first().copied();
        }
        let after_item = old_st == ST_ARR_NEXT;

        if top.mode != MODE_NONE {
            if top.mode == MODE_KEY {
                if self.key_step(top, n, old_tok, c) {
                    out.push(stack);
                }
                return;
            }
            if !self.scalar_step(top, c) {
                return;
            }
            if top.mode != MODE_NONE {
                out.push(stack);
                return;
            }
        }

        if old_st == ST_KEY && c == b'"' && old_tok == TOK_NONE && is_obj {
            if n.props.is_empty() || !top.any_unseen(n) {
                return;
            }
            top.mode = MODE_KEY;
            out.push(stack);
            return;
        }
        if c == b',' && is_obj && (old_tok == TOK_NONE || old_tok == TOK_NUM) && !top.any_unseen(n) {
            return;
        }
        if !is_obj && (c == b']' || c == b',') && n.unique && (old_tok == TOK_NUM || after_item) && top.count > 0 {
            let text = self.args[top.item_start.min(self.args.len())..].to_vec();
            if top.items_seen.contains(&text) {
                return;
            }
            top.items_seen.push(text);
        }
        let value_start = old_st == ST_VALUE && old_tok == TOK_NONE && c != b']';
        if value_start && !is_obj {
            top.count += 1;
            top.item_start = self.args.len();
        }
        if value_start {
            let Some(p) = top.pending else {
                out.push(stack);
                return;
            };
            let pn = &nodes[p];
            if pn.variants.is_empty() {
                if !pn.first_byte_ok(c) || !self.set_mode(top, pn, c) {
                    return;
                }
                if (c == b'[' && pn.is_array()) || (c == b'{' && pn.is_object()) {
                    stack.push(Frame::new(old_depth + 1, p, nodes));
                }
                out.push(stack);
                return;
            }
            if pn.variants.len() + out.len() < 17 {
                for &v in &pn.variants {
                    let vn = &nodes[v];
                    if !vn.first_byte_ok(c) {
                        continue;
                    }
                    let mut fork = stack.clone();
                    let Some(ftop) = fork.last_mut() else { continue };
                    ftop.pending = Some(v);
                    if !self.set_mode(ftop, vn, c) {
                        continue;
                    }
                    if (c == b'[' && vn.is_array()) || (c == b'{' && vn.is_object()) {
                        fork.push(Frame::new(old_depth + 1, v, nodes));
                    }
                    out.push(fork);
                }
                return;
            }
            top.pending = None;
            out.push(stack);
            return;
        }
        if c == b']' && !is_obj {
            if (n.min_items >= 0 && top.count < n.min_items) || (n.max_items >= 0 && top.count > n.max_items) {
                return;
            }
            out.push(stack);
            return;
        }
        if c == b'}' && is_obj {
            for (i, &p) in n.props.iter().enumerate() {
                if nodes[p].required && (i > 255 || !top.seen(i)) {
                    return;
                }
            }
        }
        out.push(stack);
    }

    /// `FUN_00072a1c`: a property name being typed.
    fn key_step(&self, top: &mut Frame, n: &Node, old_tok: u8, c: u8) -> bool {
        let nodes = self.nodes();
        let buf = &self.sc.buf;
        if !(old_tok == TOK_STR && c == b'"') {
            if !self.sc.complete {
                return true;
            }
            return n.props.iter().enumerate().any(|(i, &p)| (i >= 256 || !top.seen(i)) && nodes[p].name.starts_with(buf));
        }
        top.mode = MODE_NONE;
        let Some((i, &p)) = n.props.iter().enumerate().find(|(i, p)| (*i >= 256 || !top.seen(*i)) && &nodes[**p].name == buf) else {
            return false;
        };
        let pn = &nodes[p];
        if pn.grounded
            && let Some(ctx) = &self.g.ctx
            && matches!(pn.ty, ty::NUMBER | ty::INTEGER)
            && !ctx.iter().any(u8::is_ascii_digit)
        {
            return false;
        }
        top.mark(i);
        top.pending = Some(p);
        true
    }

    /// `FUN_00073410`: how a value that starts with `c` is checked.
    fn set_mode(&self, top: &mut Frame, pn: &Node, c: u8) -> bool {
        let number = c == b'-' || c.is_ascii_digit();
        if pn.ty == ty::LITERALS {
            if matches!(c, b'f' | b'n' | b't') {
                return true;
            }
            if c != b'"' {
                top.mode = MODE_NUMBER;
                return prefix_of_any(&pn.nums, &self.sc.buf);
            }
            top.mode = MODE_STRING;
            return true;
        }
        if pn.grounded
            && let Some(ctx) = self.g.ctx.as_deref().filter(|x| !x.is_empty())
        {
            if c == b'"' {
                top.mode = MODE_GROUNDED_STRING;
                return true;
            }
            if number {
                top.mode = MODE_GROUNDED_NUMBER;
                return contains(ctx, &self.sc.buf);
            }
        }
        if (pn.ty == ty::INTEGER || pn.has_min || pn.has_max || pn.multiple_of > 0.0) && number {
            top.mode = MODE_NUMBER;
            return true;
        }
        if c != b'"' || (pn.min_len < 0 && pn.max_len < 0 && pn.regexes.is_empty()) {
            return true;
        }
        top.mode = MODE_STRING;
        true
    }

    /// `FUN_00072d34`: a byte inside a checked scalar.
    fn scalar_step(&self, top: &mut Frame, c: u8) -> bool {
        let Some(p) = top.pending else { return true };
        let pn = &self.nodes()[p];
        let buf = &self.sc.buf;
        let ctx = self.g.ctx.as_deref().unwrap_or_default();
        match top.mode {
            MODE_GROUNDED_STRING => {
                if c == b'"' && self.lex.tok != TOK_STR {
                    top.mode = MODE_NONE;
                    return true;
                }
                !self.sc.complete || contains(ctx, buf)
            }
            MODE_GROUNDED_NUMBER => {
                if !is_num_byte(c) {
                    top.mode = MODE_NONE;
                    return true;
                }
                contains(ctx, buf)
            }
            MODE_STRING => {
                if c != b'"' || self.lex.tok == TOK_STR {
                    if !self.sc.complete {
                        return true;
                    }
                    if !pn.strs.is_empty() && !prefix_of_any(&pn.strs, buf) {
                        return false;
                    }
                    return pn.max_len < 0 || self.sc.cp <= pn.max_len as usize;
                }
                top.mode = MODE_NONE;
                if !pn.strs.is_empty() && !pn.strs.contains(buf) {
                    return false;
                }
                if pn.min_len >= 0 && self.sc.cp < pn.min_len as usize {
                    return false;
                }
                if pn.max_len >= 0 && self.sc.cp > pn.max_len as usize {
                    return false;
                }
                pn.regexes.iter().all(|r| r.is_match(buf))
            }
            MODE_NUMBER => {
                let digit = c.is_ascii_digit();
                if !digit {
                    if !is_num_byte(c) {
                        top.mode = MODE_NONE;
                        if !pn.nums.is_empty() {
                            return pn.nums.contains(buf);
                        }
                        let v = atof(buf);
                        if pn.has_min && (v < pn.min || (pn.excl_min && v == pn.min)) {
                            return false;
                        }
                        if pn.has_max && (v > pn.max || (pn.excl_max && v == pn.max)) {
                            return false;
                        }
                        if pn.multiple_of > 0.0 {
                            let q = v / pn.multiple_of;
                            return (q - q.trunc()).abs() <= 1e-9;
                        }
                        return true;
                    }
                    if pn.ty == ty::INTEGER {
                        return false;
                    }
                }
                if buf.len() > 20 {
                    return false;
                }
                if digit && pn.has_max && !buf.iter().any(|b| matches!(b, b'.' | b'e' | b'E')) && atof(buf) > pn.max {
                    return false;
                }
                pn.nums.is_empty() || prefix_of_any(&pn.nums, buf)
            }
            _ => true,
        }
    }

    /// At the start of an argument value, where the engine may look ahead
    /// (schema-checked arguments only).
    pub fn at_value_start(&self) -> bool {
        self.kind == kind::ARGS
            && self.started
            && self.lex.tok == TOK_NONE
            && self.lex.st == ST_VALUE
            && self.stacks.first().and_then(|s| s.last()).is_some_and(|f| f.mode == MODE_NONE)
    }

    /// The schema of the value about to start, when every live reading of
    /// the arguments agrees on one.
    fn agreed_pending(&self) -> Option<&'g Node> {
        let p = self.stacks.first()?.last()?.pending?;
        self.stacks.iter().all(|s| s.last().is_some_and(|f| f.pending == Some(p))).then(|| &self.g.schemas.nodes[p])
    }

    /// The value about to start is grounded (the engine always looks ahead
    /// for those).
    pub fn pending_grounded(&self) -> bool {
        self.agreed_pending().is_some_and(|n| n.grounded)
    }

    /// The value about to start is a set of string options (the engine
    /// scores those as whole options instead of looking ahead).
    pub fn pending_string_options(&self) -> bool {
        self.agreed_pending().is_some_and(|n| n.ty == ty::LITERALS && !n.strs.is_empty())
    }

    /// A lookahead branch is done: a top-level argument value just ended, or
    /// the arguments did.
    pub fn value_closed(&self) -> bool {
        matches!(self.kind, kind::CALL_END | kind::AFTER_CALL | kind::NEXT_CALL | kind::DONE)
            || (self.kind == kind::ARGS && self.lex.stack.len() == 1 && self.lex.st == ST_OBJ_NEXT)
    }

    /// The bytes the engine expects next, for its sparse logits: a coarse
    /// table by lexer state rather than a probe of the full grammar (a
    /// number admits `,`, `}` and `]` whatever encloses it). `None` inside
    /// a string, where anything goes.
    pub fn next_bytes(&self) -> Option<[bool; 256]> {
        let mut m = [false; 256];
        let mut set = |bytes: &[u8]| bytes.iter().for_each(|&b| m[b as usize] = true);
        match self.kind {
            kind::START => set(b"["),
            kind::LIST => set(b"]{"),
            kind::NAME_KEY | kind::ARGS_KEY => set(&self.lit[self.lit_pos..self.lit_pos + 1]),
            kind::NAME => {
                let allowed = |n: &[u8]| self.g.allowed.is_empty() || self.g.allowed.iter().any(|a| a == n);
                for t in self.g.schemas.tools.iter().filter(|t| allowed(&t.name) && t.name.starts_with(&self.name)) {
                    set(&[t.name.get(self.name.len()).copied().unwrap_or(b'"')]);
                }
            }
            kind::ARGS if !self.started => set(b"{"),
            kind::ARGS => match self.lex.tok {
                TOK_STR => return None,
                TOK_ESC => set(b"\"\\/bfnrtu"),
                TOK_HEX => set(b"0123456789abcdefABCDEF"),
                TOK_NUM => set(b"0123456789eE+,-.}]"),
                TOK_LIT => set(&self.lex.lit[self.lex.lit_pos..self.lex.lit_pos + 1]),
                _ => match self.lex.st {
                    ST_VALUE => set(b"0123456789\"-tfn{[]"),
                    ST_KEY => set(b"\"}"),
                    ST_COLON => set(b":"),
                    ST_OBJ_NEXT => set(b",}"),
                    _ => set(b",]"),
                },
            },
            kind::CALL_END => set(b"}"),
            kind::AFTER_CALL => set(b",]"),
            kind::NEXT_CALL => set(b"{"),
            _ => {}
        }
        Some(m)
    }

    /// The string options of a literal-set value about to open, when every
    /// live reading of the arguments agrees on it (the engine's condition
    /// for scoring whole options).
    pub(super) fn literal_options(&self) -> Option<&'g [Vec<u8>]> {
        if self.kind != kind::ARGS || !self.started || self.lex.tok != TOK_NONE || self.lex.st != ST_VALUE {
            return None;
        }
        let first = self.stacks.first()?.last()?;
        let p = first.pending?;
        if first.mode != MODE_NONE || !self.stacks.iter().all(|s| s.last().is_some_and(|f| f.pending == Some(p))) {
            return None;
        }
        let pn = &self.g.schemas.nodes[p];
        (pn.ty == ty::LITERALS && !pn.strs.is_empty()).then_some(pn.strs.as_slice())
    }
}
