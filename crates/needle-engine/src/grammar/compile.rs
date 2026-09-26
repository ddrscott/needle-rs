//! The schema compiler (`FUN_00006374`, `FUN_0002fd50`, `FUN_0002f35c`,
//! `FUN_000337ec`): each tool's `parameters` becomes a tree of [`Node`]s the
//! byte grammar walks.
//!
//! What it honours: `type` (string or array), `properties`/`required`,
//! `items` (the object form), `enum`/`const`, `minimum`/`maximum` and both
//! drafts of the exclusive bounds, `multipleOf`, `minLength`/`maxLength`,
//! `pattern`, `format`, `minItems`/`maxItems`, `uniqueItems`, `allOf`,
//! `anyOf`/`oneOf`, `$ref` into one `$defs`/`definitions` object, and a
//! `grounded` flag. It also mines allowed values out of descriptions and
//! bounds a few numeric names (`level`, `rating`, `*_minutes`, ...).

use super::regex::{Regex, format_pattern};
use crate::rules::json::Json;

pub type NodeId = usize;

/// The node type (`+0x18`).
pub mod ty {
    pub const ANY: u8 = 0;
    pub const NUMBER: u8 = 1;
    pub const INTEGER: u8 = 2;
    pub const BOOLEAN: u8 = 3;
    pub const STRING: u8 = 4;
    pub const ARRAY: u8 = 5;
    pub const OBJECT: u8 = 6;
    pub const NULL: u8 = 7;
    /// A closed set of literals (enum, const, or mined values).
    pub const LITERALS: u8 = 8;
}

/// One compiled schema (the library's 0xe8-byte node).
#[derive(Clone, Debug)]
pub struct Node {
    pub name: Vec<u8>,
    pub ty: u8,
    pub required: bool,
    pub grounded: bool,
    /// String literals (decoded).
    pub strs: Vec<Vec<u8>>,
    /// Number literals as their JSON source text.
    pub nums: Vec<Vec<u8>>,
    pub lit_true: bool,
    pub lit_false: bool,
    pub lit_null: bool,
    pub has_min: bool,
    pub has_max: bool,
    pub excl_min: bool,
    pub excl_max: bool,
    pub min: f64,
    pub max: f64,
    pub multiple_of: f64,
    pub min_len: i32,
    pub max_len: i32,
    pub regexes: Vec<Regex>,
    pub props: Vec<NodeId>,
    /// Only the first is used.
    pub items: Vec<NodeId>,
    pub variants: Vec<NodeId>,
    pub min_items: i32,
    pub max_items: i32,
    pub unique: bool,
    pub has_props_key: bool,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            name: vec![],
            ty: ty::ANY,
            required: false,
            grounded: false,
            strs: vec![],
            nums: vec![],
            lit_true: false,
            lit_false: false,
            lit_null: false,
            has_min: false,
            has_max: false,
            excl_min: false,
            excl_max: false,
            min: 0.0,
            max: 0.0,
            multiple_of: 0.0,
            min_len: -1,
            max_len: -1,
            regexes: vec![],
            props: vec![],
            items: vec![],
            variants: vec![],
            min_items: -1,
            max_items: -1,
            unique: false,
            has_props_key: false,
        }
    }
}

impl Node {
    pub fn has_literals(&self) -> bool {
        !self.strs.is_empty() || !self.nums.is_empty() || self.lit_true || self.lit_false || self.lit_null
    }

    /// Objects the grammar tracks key by key.
    pub fn is_object(&self) -> bool {
        !self.props.is_empty() || self.has_props_key
    }

    /// Arrays the grammar opens a frame for (`FUN_0007366c`).
    pub fn is_array(&self) -> bool {
        self.ty == ty::ARRAY || !self.items.is_empty() || self.min_items >= 0 || self.max_items >= 0 || self.unique
    }

    /// Does a value starting with byte `c` fit this node's type
    /// (`FUN_000732d8`)?
    pub fn first_byte_ok(&self, c: u8) -> bool {
        let num = c == b'-' || c.is_ascii_digit();
        match self.ty {
            ty::NUMBER | ty::INTEGER => num,
            ty::BOOLEAN => c == b't' || c == b'f',
            ty::STRING => c == b'"',
            ty::ARRAY => c == b'[',
            ty::OBJECT => c == b'{',
            ty::NULL => c == b'n',
            ty::LITERALS => match c {
                b'"' => !self.strs.is_empty(),
                b't' => self.lit_true,
                b'f' => self.lit_false,
                b'n' => self.lit_null,
                _ if num => !self.nums.is_empty(),
                _ => false,
            },
            _ => true,
        }
    }
}

/// A compiled tool.
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: Vec<u8>,
    pub root: NodeId,
    /// Arguments are checked against the schema (the root declared
    /// `properties`); otherwise they are only lexed as a JSON object.
    pub validated: bool,
}

/// Every tool's schema tree, in one arena.
#[derive(Clone, Debug, Default)]
pub struct Schemas {
    pub nodes: Vec<Node>,
    pub tools: Vec<Tool>,
}

const MAX_DEPTH: usize = 32;
const BUDGET: i64 = 0x2000;

struct Compiler<'a> {
    nodes: Vec<Node>,
    budget: i64,
    defs: Option<&'a Json>,
}

fn atof(raw: &[u8]) -> f64 {
    // strtod on a JSON number's source: the whole text parses.
    std::str::from_utf8(raw).ok().and_then(|s| s.parse().ok()).unwrap_or(0.0)
}

fn atoi(raw: &[u8]) -> i32 {
    let (neg, digits) = match raw.first() {
        Some(b'-') => (true, &raw[1..]),
        Some(b'+') => (false, &raw[1..]),
        _ => (false, raw),
    };
    let mut v: i64 = 0;
    for &c in digits.iter().take_while(|c| c.is_ascii_digit()) {
        v = (v * 10 + (c - b'0') as i64).min(i32::MAX as i64 + 1);
    }
    let v = if neg { -v } else { v };
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn type_code(s: &[u8]) -> u8 {
    match s {
        b"number" | b"float" => ty::NUMBER,
        b"integer" => ty::INTEGER,
        b"boolean" => ty::BOOLEAN,
        b"string" => ty::STRING,
        b"array" => ty::ARRAY,
        b"object" | b"dict" => ty::OBJECT,
        b"null" => ty::NULL,
        _ => ty::ANY,
    }
}

/// A literal (`FUN_000332fc`): strings, numbers, true/false/null; objects
/// and arrays are not supported.
enum Lit {
    Str(Vec<u8>),
    Num(Vec<u8>),
    True,
    False,
    Null,
}

fn literal(v: &Json) -> Option<Lit> {
    Some(match v {
        Json::Str(s) => Lit::Str(s.clone()),
        Json::Num(n) => Lit::Num(n.clone()),
        Json::Bool(true) => Lit::True,
        Json::Bool(false) => Lit::False,
        Json::Null => Lit::Null,
        _ => return None,
    })
}

fn add_literal(n: &mut Node, l: Lit) {
    match l {
        Lit::Str(s) => n.strs.push(s),
        Lit::Num(s) => n.nums.push(s),
        Lit::True => n.lit_true = true,
        Lit::False => n.lit_false = true,
        Lit::Null => n.lit_null = true,
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

fn find_from(hay: &[u8], c: u8, from: usize) -> Option<usize> {
    hay.get(from..)?.iter().position(|&x| x == c).map(|p| p + from)
}

/// `FUN_00037a24`: quoted tokens paired left to right. Any token that is
/// empty, 20 bytes or longer, or holds a space, tab or newline abandons
/// the whole list; fewer than two distinct tokens is no list.
fn extract(s: &[u8], q: u8) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = vec![];
    let mut pos = 0;
    while let Some(i) = find_from(s, q, pos) {
        let Some(j) = find_from(s, q, i + 1) else { break };
        let tok = &s[i + 1..j];
        if tok.is_empty() || tok.len() > 19 || tok.iter().any(|c| matches!(c, b' ' | b'\t' | b'\n')) {
            return vec![];
        }
        if !out.iter().any(|o| o == tok) {
            out.push(tok.to_vec());
        }
        pos = j + 1;
    }
    if out.len() >= 2 { out } else { vec![] }
}

/// `FUN_000337ec`: allowed values written into a description.
pub fn mine_description(desc: &[u8]) -> Vec<Vec<u8>> {
    let lower = desc.to_ascii_lowercase();
    if contains(&lower, b"e.g") || contains(&lower, b"such as") || contains(&lower, b"for example") {
        return vec![];
    }
    let v = extract(desc, b'\'');
    if !v.is_empty() {
        return v;
    }
    const CUES: [&[u8]; 7] = [b"options", b"one of", b"possible values", b"allowed values", b"valid values", b"choices", b"can be"];
    if CUES.iter().any(|k| contains(&lower, k)) {
        return extract(desc, b'"');
    }
    vec![]
}

/// The numeric range a property name implies when the schema gives none.
fn name_bounds(name: &[u8]) -> Option<(Option<f64>, Option<f64>)> {
    const PERCENT: [&[u8]; 8] =
        [b"level", b"brightness_level", b"volume_level", b"light_level", b"dim_level", b"fan_level", b"power_level", b"volume"];
    if PERCENT.contains(&name) || contains(name, b"brightness") || contains(name, b"percent") {
        return Some((Some(0.0), Some(100.0)));
    }
    if name == b"rating" {
        return Some((Some(1.0), Some(5.0)));
    }
    if [&b"minutes"[..], b"seconds", b"hours", b"duration", b"count"].iter().any(|k| contains(name, k)) {
        return Some((Some(0.0), None));
    }
    None
}

impl<'a> Compiler<'a> {
    fn alloc(&mut self, n: Node) -> NodeId {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    /// Unconstrained, for `anyOf`: no type, properties or variants.
    fn unconstrained(&self, id: NodeId) -> bool {
        let n = &self.nodes[id];
        n.ty == ty::ANY && n.props.is_empty() && !n.has_props_key && n.variants.is_empty()
    }

    /// `FUN_000320bc`: follow `$ref` (its last path segment, looked up in
    /// the tool's one definitions object), up to 33 hops.
    fn resolve(&self, mut s: &'a Json) -> &'a Json {
        let Some(defs) = self.defs else { return s };
        for _ in 0..33 {
            let Some(Json::Str(r)) = s.get(b"$ref") else { break };
            let key = r.rsplit(|&c| c == b'/').next().unwrap_or(r);
            match defs.get(key) {
                Some(t @ Json::Obj(_)) => s = t,
                _ => break,
            }
        }
        s
    }

    /// `FUN_0002fd50`: compile `schema` into node `id`.
    fn node(&mut self, schema: &'a Json, id: NodeId, depth: usize) {
        if depth > MAX_DEPTH || self.budget == 0 {
            return;
        }
        let schema = self.resolve(schema);
        let Json::Obj(members) = schema else { return };

        // Pass 1: the combinators (keys whose value is an array).
        for (k, v) in members {
            let Json::Arr(elems) = v else { continue };
            match k.as_slice() {
                b"allOf" => {
                    for e in elems.iter().filter(|e| matches!(e, Json::Obj(_))) {
                        if self.budget != 0 {
                            self.budget -= 1;
                            self.node(e, id, depth + 1);
                        }
                    }
                }
                b"anyOf" | b"oneOf" => {
                    let objs: Vec<&'a Json> = elems.iter().filter(|e| matches!(e, Json::Obj(_))).collect();
                    if objs.len() == 1 {
                        self.node(objs[0], id, depth + 1);
                        return;
                    }
                    if objs.len() > 1 {
                        let mut out = vec![];
                        let mut loose = false;
                        let mut ran_out = false;
                        for e in objs {
                            if self.budget == 0 {
                                ran_out = true;
                                break;
                            }
                            self.budget -= 1;
                            let v = self.alloc(Node::default());
                            self.node(e, v, depth + 1);
                            loose |= self.unconstrained(v);
                            out.push(v);
                        }
                        if !ran_out && !loose {
                            self.nodes[id].variants.extend(out);
                        }
                    }
                }
                _ => {}
            }
        }

        // Pass 2: every keyword, in order.
        let mut desc: Option<&'a [u8]> = None;
        let mut mask = 0u8;
        let mut unknown_type = false;
        for (k, v) in members {
            match (k.as_slice(), v) {
                (b"type", Json::Str(s)) => self.nodes[id].ty = type_code(s),
                (b"type", Json::Arr(ts)) => {
                    for t in ts {
                        match t {
                            Json::Str(s) if type_code(s) != ty::ANY => mask |= 1 << type_code(s),
                            _ => unknown_type = true,
                        }
                    }
                }
                (b"properties", Json::Obj(_)) => self.properties(schema, id, depth),
                (b"required", Json::Bool(true)) => self.nodes[id].required = true,
                (b"items", Json::Obj(_)) => {
                    if self.budget != 0 {
                        self.budget -= 1;
                        let it = self.alloc(Node::default());
                        self.node(v, it, depth + 1);
                        self.nodes[id].items.push(it);
                    }
                }
                (b"minItems", Json::Num(n)) => self.nodes[id].min_items = atoi(n),
                (b"maxItems", Json::Num(n)) => self.nodes[id].max_items = atoi(n),
                (b"minLength", Json::Num(n)) => self.nodes[id].min_len = atoi(n),
                (b"maxLength", Json::Num(n)) => self.nodes[id].max_len = atoi(n),
                (b"uniqueItems", Json::Bool(true)) => self.nodes[id].unique = true,
                (b"minimum", Json::Num(n)) => {
                    let node = &mut self.nodes[id];
                    node.has_min = true;
                    node.min = atof(n);
                }
                (b"maximum", Json::Num(n)) => {
                    let node = &mut self.nodes[id];
                    node.has_max = true;
                    node.max = atof(n);
                }
                (b"exclusiveMinimum", Json::Bool(true)) => self.nodes[id].excl_min = true,
                (b"exclusiveMinimum", Json::Num(n)) => {
                    let node = &mut self.nodes[id];
                    node.has_min = true;
                    node.excl_min = true;
                    node.min = atof(n);
                }
                (b"exclusiveMaximum", Json::Bool(true)) => self.nodes[id].excl_max = true,
                (b"exclusiveMaximum", Json::Num(n)) => {
                    let node = &mut self.nodes[id];
                    node.has_max = true;
                    node.excl_max = true;
                    node.max = atof(n);
                }
                (b"multipleOf", Json::Num(n)) => self.nodes[id].multiple_of = atof(n),
                (b"pattern", Json::Str(p)) => {
                    if let Some(r) = Regex::new(p) {
                        self.nodes[id].regexes.push(r);
                    }
                }
                (b"format", Json::Str(f)) => {
                    if let Some(r) = format_pattern(f).and_then(|p| Regex::new(p.as_bytes())) {
                        self.nodes[id].regexes.push(r);
                    }
                }
                (b"const", c) => {
                    if let Some(l) = literal(c) {
                        add_literal(&mut self.nodes[id], l);
                    }
                }
                (b"enum", Json::Arr(opts)) => {
                    let lits: Option<Vec<Lit>> = opts.iter().map(literal).collect();
                    if let Some(lits) = lits.filter(|l| !l.is_empty()) {
                        let node = &mut self.nodes[id];
                        node.strs.clear();
                        node.nums.clear();
                        node.lit_true = false;
                        node.lit_false = false;
                        node.lit_null = false;
                        for l in lits {
                            add_literal(node, l);
                        }
                    }
                }
                (b"description", Json::Str(d)) => desc = Some(d),
                (b"grounded", Json::Bool(true)) => self.nodes[id].grounded = true,
                _ => {}
            }
        }

        let node = &mut self.nodes[id];
        // Mined values: only for an untyped or string node that is neither
        // required (draft 3) nor grounded.
        if !node.has_literals()
            && matches!(node.ty, ty::ANY | ty::STRING)
            && !node.required
            && !node.grounded
            && let Some(d) = desc
        {
            node.strs = mine_description(d);
        }
        if node.has_min || node.has_max || node.multiple_of > 0.0 {
            let (lo, hi, m) = (node.min, node.max, node.multiple_of);
            let (has_min, has_max, excl_min, excl_max) = (node.has_min, node.has_max, node.excl_min, node.excl_max);
            node.nums.retain(|raw| {
                let v = atof(raw);
                if has_min && (v < lo || (excl_min && v == lo)) {
                    return false;
                }
                if has_max && (v > hi || (excl_max && v == hi)) {
                    return false;
                }
                m <= 0.0 || ((v / m) - (v / m).trunc()).abs() <= 1e-9
            });
        }
        if !node.has_literals() {
            if matches!(node.ty, ty::NUMBER | ty::INTEGER) && !node.has_min && !node.has_max {
                node.excl_min = false;
                node.excl_max = false;
                if let Some((lo, hi)) = name_bounds(&node.name) {
                    if let Some(lo) = lo {
                        node.has_min = true;
                        node.min = lo;
                    }
                    if let Some(hi) = hi {
                        node.has_max = true;
                        node.max = hi;
                    }
                }
            }
            if mask != 0 && !unknown_type && node.ty == ty::ANY && node.variants.is_empty() {
                self.expand_type_array(id, mask);
            }
        } else {
            node.ty = ty::LITERALS;
        }

        // Variants of variants are spliced in; variants beside any other
        // constraint are dropped.
        loop {
            let vs = self.nodes[id].variants.clone();
            if !vs.iter().any(|&v| !self.nodes[v].variants.is_empty()) {
                break;
            }
            let flat =
                vs.iter().flat_map(|&v| if self.nodes[v].variants.is_empty() { vec![v] } else { self.nodes[v].variants.clone() }).collect();
            self.nodes[id].variants = flat;
        }
        let n = &self.nodes[id];
        if !n.variants.is_empty() {
            let plain = n.ty == ty::ANY
                && !n.has_literals()
                && !n.has_min
                && !n.has_max
                && n.multiple_of <= 0.0
                && n.min_len < 0
                && n.max_len < 0
                && n.regexes.is_empty()
                && n.props.is_empty()
                && n.items.is_empty();
            if !plain {
                self.nodes[id].variants.clear();
            }
        }
    }

    /// A `type` array: one member sets the type; several become variants,
    /// each carrying the constraints of its kind.
    fn expand_type_array(&mut self, id: NodeId, mask: u8) {
        let bits: Vec<u8> = (1..=7).filter(|b| mask & (1 << b) != 0).collect();
        if bits.len() == 1 {
            self.nodes[id].ty = bits[0];
            return;
        }
        let p = self.nodes[id].clone();
        for t in bits {
            let mut v = Node { ty: t, ..Node::default() };
            match t {
                ty::NUMBER | ty::INTEGER => {
                    v.has_min = p.has_min;
                    v.has_max = p.has_max;
                    v.excl_min = p.excl_min;
                    v.excl_max = p.excl_max;
                    v.min = p.min;
                    v.max = p.max;
                    v.multiple_of = p.multiple_of;
                }
                ty::STRING => {
                    v.min_len = p.min_len;
                    v.max_len = p.max_len;
                    v.regexes = p.regexes.clone();
                }
                ty::ARRAY => {
                    v.items = p.items.clone();
                    v.min_items = p.min_items;
                    v.max_items = p.max_items;
                    v.unique = p.unique;
                }
                ty::OBJECT => {
                    v.props = p.props.clone();
                    v.has_props_key = p.has_props_key;
                }
                _ => {}
            }
            let vid = self.alloc(v);
            self.nodes[id].variants.push(vid);
        }
        let n = &mut self.nodes[id];
        n.has_min = false;
        n.has_max = false;
        n.excl_min = false;
        n.excl_max = false;
        n.min = 0.0;
        n.max = 0.0;
        n.multiple_of = 0.0;
        n.min_len = -1;
        n.max_len = -1;
        n.regexes.clear();
        n.items.clear();
        n.min_items = -1;
        n.max_items = -1;
        n.unique = false;
        n.props.clear();
        n.has_props_key = false;
    }

    /// `FUN_0002f35c`: `properties` and `required` of an object schema.
    fn properties(&mut self, schema: &'a Json, id: NodeId, depth: usize) {
        let Some(Json::Obj(props)) = schema.get(b"properties") else { return };
        self.nodes[id].has_props_key = true;
        for (key, v) in props {
            if self.nodes[id].props.iter().any(|&p| &self.nodes[p].name == key) {
                continue;
            }
            let grounded = key == b"id" || (key.len() >= 4 && key.ends_with(b"_id"));
            if self.budget == 0 {
                let n = &mut self.nodes[id];
                n.props.clear();
                n.has_props_key = false;
                return;
            }
            self.budget -= 1;
            let pid = self.alloc(Node { name: key.clone(), grounded, ..Node::default() });
            if matches!(v, Json::Obj(_)) {
                self.node(v, pid, depth + 1);
            }
            let g = self.nodes[pid].grounded;
            for vid in self.nodes[pid].variants.clone() {
                self.nodes[vid].grounded = g;
            }
            self.nodes[id].props.push(pid);
        }
        if let Some(Json::Arr(req)) = schema.get(b"required") {
            for r in req {
                let Json::Str(r) = r else { continue };
                if let Some(&p) = self.nodes[id].props.iter().find(|&&p| &self.nodes[p].name == r) {
                    self.nodes[p].required = true;
                }
            }
        }
    }
}

impl Schemas {
    /// `FUN_00006374` over the normalized tool list.
    pub fn compile(tools: &[Json]) -> Schemas {
        let mut c = Compiler { nodes: vec![], budget: BUDGET, defs: None };
        let mut out = vec![];
        for t in tools {
            let Some(Json::Str(name)) = t.get(b"name") else { continue };
            if name.is_empty() {
                continue;
            }
            let root = c.alloc(Node { ty: ty::OBJECT, ..Node::default() });
            if c.budget > 0 {
                c.budget -= 1;
            }
            if let Some(params @ Json::Obj(members)) = t.get(b"parameters") {
                c.defs = members.iter().find(|(k, v)| (k == b"$defs" || k == b"definitions") && matches!(v, Json::Obj(_))).map(|(_, v)| v);
                c.properties(params, root, 0);
            }
            let validated = c.nodes[root].is_object();
            out.push(Tool { name: name.clone(), root, validated });
        }
        Schemas { nodes: c.nodes, tools: out }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::json::parse;

    fn compile(params: &str) -> (Schemas, NodeId) {
        let tools = vec![parse(format!(r#"{{"name":"t","parameters":{params}}}"#).as_bytes()).unwrap()];
        let s = Schemas::compile(&tools);
        let root = s.tools[0].root;
        (s, root)
    }

    fn prop<'a>(s: &'a Schemas, root: NodeId, name: &str) -> &'a Node {
        s.nodes[root].props.iter().map(|&p| &s.nodes[p]).find(|n| n.name == name.as_bytes()).unwrap()
    }

    #[test]
    fn mining_and_heuristics() {
        let (s, r) = compile(
            r#"{"properties":{"mode":{"type":"string","description":"One of 'heat', 'cool' or 'auto'."},
            "level":{"type":"integer"},"rating":{"type":"number"},"wait_minutes":{"type":"integer"},
            "eg":{"type":"string","description":"e.g. 'a', 'b'"},
            "req":{"type":"string","required":true,"description":"'a' or 'b'"},
            "room_id":{"type":"string","description":"'a' or 'b'"}}}"#,
        );
        let mode = prop(&s, r, "mode");
        assert_eq!(mode.ty, ty::LITERALS);
        assert_eq!(mode.strs, vec![b"heat".to_vec(), b"cool".to_vec(), b"auto".to_vec()]);
        let level = prop(&s, r, "level");
        assert!(level.has_min && level.has_max && level.min == 0.0 && level.max == 100.0);
        let rating = prop(&s, r, "rating");
        assert!(rating.min == 1.0 && rating.max == 5.0);
        let wait = prop(&s, r, "wait_minutes");
        assert!(wait.has_min && !wait.has_max);
        assert!(prop(&s, r, "eg").strs.is_empty());
        assert!(prop(&s, r, "req").strs.is_empty());
        let rid = prop(&s, r, "room_id");
        assert!(rid.grounded && rid.strs.is_empty());
    }

    #[test]
    fn combinators_and_refs() {
        let (s, r) = compile(
            r##"{"$defs":{"Unit":{"enum":["c","f"]}},"properties":{
            "u":{"$ref":"#/$defs/Unit","description":"ignored"},
            "a":{"anyOf":[{"type":"string"},{"type":"null"}]},
            "b":{"anyOf":[{"type":"string"},{}]},
            "c":{"anyOf":[{"type":"integer"}],"type":"string"},
            "d":{"type":["integer","null"],"minimum":2},
            "e":{"allOf":[{"type":"integer"},{"maximum":9}]},
            "f":{"enum":[1,2.5,3],"maximum":2.5,"multipleOf":0.5},
            "g":{"enum":[1,{"x":1}]}}}"##,
        );
        assert_eq!(prop(&s, r, "u").strs, vec![b"c".to_vec(), b"f".to_vec()]);
        assert_eq!(prop(&s, r, "a").variants.len(), 2);
        assert!(prop(&s, r, "b").variants.is_empty());
        assert_eq!(prop(&s, r, "c").ty, ty::INTEGER);
        let d = prop(&s, r, "d");
        assert_eq!(d.variants.len(), 2);
        assert!(s.nodes[d.variants[0]].has_min && s.nodes[d.variants[0]].ty == ty::INTEGER);
        let e = prop(&s, r, "e");
        assert!(e.ty == ty::INTEGER && e.has_max && e.max == 9.0);
        assert_eq!(prop(&s, r, "f").nums, vec![b"1".to_vec(), b"2.5".to_vec()]);
        let g = prop(&s, r, "g");
        assert!(!g.has_literals() && g.ty == ty::ANY);
    }
}
