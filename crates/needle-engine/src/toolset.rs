//! What `needle_init` does to its inputs before anything reaches the model:
//! the tools JSON is compacted and normalized (OpenAI wrappers unwrapped,
//! tool and property names snake_cased, type aliases mapped, `triggers`
//! dropped), and the system text becomes the facts line the prompt carries.
//!
//! The library's code: `FUN_00004294` (tool list), `FUN_0001a7f0`
//! (snake_case), `FUN_0001abec` (schema walk), `FUN_00020708` (property
//! renames), `FUN_0001fc64` (type names), `FUN_0002b624` (per-tool text for
//! retrieval), and the system handling inside `needle_init` at 0x2c8c-0x3848.

use needle_core::pyjson::float_repr;

use crate::rules::json::{self, Json};

/// Drop spaces, tabs and line breaks outside JSON strings.
pub fn compact_ws(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let (mut in_str, mut esc) = (false, false);
    for &c in s {
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
        } else if !matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
            if c == b'"' {
                in_str = true;
            }
            out.push(c);
        }
    }
    out
}

/// The library's snake_case: word breaks at non-alphanumerics and at case
/// changes (`userId` -> `user_id`, `HTTPServer` -> `http_server`). A name
/// with any non-ASCII byte is kept as it is, and so is one that would come
/// out empty.
pub fn snake_case(s: &[u8]) -> Vec<u8> {
    if s.iter().any(|&c| c >= 0x80) || s.is_empty() {
        return s.to_vec();
    }
    let mut out: Vec<u8> = Vec::with_capacity(s.len() + 4);
    if s[0].is_ascii_alphanumeric() {
        out.push(s[0].to_ascii_lowercase());
    }
    let mut sep = false;
    for i in 1..s.len() {
        let c = s[i];
        if !c.is_ascii_alphanumeric() {
            sep = !out.is_empty();
            continue;
        }
        let upper = c.is_ascii_uppercase();
        let next_lower = s.get(i + 1).is_some_and(u8::is_ascii_lowercase);
        if sep || !upper {
            if sep && out.last() != Some(&b'_') {
                out.push(b'_');
            }
        } else if !out.is_empty() {
            let p = s[i - 1];
            if (p.is_ascii_lowercase() || p.is_ascii_digit() || (p.is_ascii_uppercase() && next_lower)) && out.last() != Some(&b'_') {
                out.push(b'_');
            }
        }
        out.push(c.to_ascii_lowercase());
        sep = false;
    }
    while out.last() == Some(&b'_') {
        out.pop();
    }
    if out.is_empty() { s.to_vec() } else { out }
}

/// A JSON-Schema type name as the library reads it: `List[str]` and
/// `int[]` are arrays, `Dict<K,V>` is an object, `double` is a number, and
/// so on. Anything it does not know is kept.
fn normalize_type(t: &[u8]) -> Vec<u8> {
    if t.len() >= 2 && t.ends_with(b"[]") {
        return b"array".to_vec();
    }
    let mut base = Vec::with_capacity(t.len());
    for &c in t {
        if c == b'<' || c == b'[' {
            break;
        }
        if !matches!(c, b' ' | b'-' | b'_') {
            base.push(c.to_ascii_lowercase());
        }
    }
    let mapped: &[u8] = match base.as_slice() {
        b"string" | b"char" | b"character" | b"any" => b"string",
        b"bool" | b"boolean" => b"boolean",
        b"int" | b"integer" | b"long" | b"short" | b"byte" | b"biginteger" => b"integer",
        b"float" | b"double" | b"decimal" | b"bigdecimal" | b"number" => b"number",
        b"array" | b"list" | b"arraylist" | b"set" | b"tuple" => b"array",
        b"object" | b"dict" | b"dictionary" | b"map" | b"hashmap" => b"object",
        b"null" | b"none" => b"null",
        _ => t,
    };
    mapped.to_vec()
}

/// JSON-Schema keywords a flat `parameters` object (one without
/// `properties`) keeps unrenamed (the table at 0xc0300).
const KEYWORDS: [&[u8]; 42] = [
    b"$defs",
    b"$id",
    b"$ref",
    b"$schema",
    b"additionalProperties",
    b"allOf",
    b"anyOf",
    b"const",
    b"contains",
    b"default",
    b"definitions",
    b"dependentRequired",
    b"dependentSchemas",
    b"description",
    b"else",
    b"enum",
    b"examples",
    b"exclusiveMaximum",
    b"exclusiveMinimum",
    b"format",
    b"if",
    b"items",
    b"maxItems",
    b"maxLength",
    b"maximum",
    b"minItems",
    b"minLength",
    b"minimum",
    b"multipleOf",
    b"not",
    b"oneOf",
    b"pattern",
    b"patternProperties",
    b"prefixItems",
    b"properties",
    b"propertyNames",
    b"required",
    b"then",
    b"title",
    b"type",
    b"unevaluatedProperties",
    b"uniqueItems",
];

/// One property rename: under the property path `path` (the new names of
/// the enclosing properties), `original` became `new`.
#[derive(Clone, Debug)]
struct PropRename {
    path: Vec<Vec<u8>>,
    original: Vec<u8>,
    new: Vec<u8>,
}

/// One tool's renames.
#[derive(Clone, Debug, Default)]
struct ToolRename {
    name: Vec<u8>,
    original: Vec<u8>,
    props: Vec<PropRename>,
}

impl ToolRename {
    /// The new name of `original` under `path`, assigning one on first use:
    /// its snake_case, with `__2`, `__3`, ... when another original under
    /// the same path already took it.
    fn rename(&mut self, path: &[Vec<u8>], original: &[u8]) -> Vec<u8> {
        if let Some(r) = self.props.iter().find(|r| r.path == path && r.original == original) {
            return r.new.clone();
        }
        let base = snake_case(original);
        let mut new = base.clone();
        let mut n = 2;
        while self.props.iter().any(|r| r.path == path && r.new == new) {
            new = [base.as_slice(), b"__", n.to_string().as_bytes()].concat();
            n += 1;
        }
        self.props.push(PropRename { path: path.to_vec(), original: original.to_vec(), new: new.clone() });
        new
    }

    fn lookup(&self, path: &[Vec<u8>], original: &[u8]) -> Option<&[u8]> {
        self.props.iter().find(|r| r.path == path && r.original == original).map(|r| r.new.as_slice())
    }

    fn original_of(&self, path: &[Vec<u8>], new: &[u8]) -> Option<&[u8]> {
        self.props.iter().find(|r| r.path == path && r.new == new).map(|r| r.original.as_slice())
    }

    /// `FUN_0001abec`: rename the properties of a schema (and of every
    /// subschema that shares its path), map type aliases.
    fn schema(&mut self, node: &mut Json, path: &[Vec<u8>], root: bool) {
        let Json::Obj(members) = node else { return };
        let has_props = members.iter().any(|(k, v)| k == b"properties" && matches!(v, Json::Obj(_)));
        if has_props {
            let props = members.iter_mut().find(|(k, v)| k == b"properties" && matches!(v, Json::Obj(_))).map(|(_, v)| v);
            if let Some(Json::Obj(props)) = props {
                for (k, v) in props.iter_mut() {
                    let new = self.rename(path, k);
                    *k = new.clone();
                    let child: Vec<Vec<u8>> = path.iter().cloned().chain([new]).collect();
                    self.schema(v, &child, false);
                }
            }
        } else if root {
            for (k, v) in members.iter_mut() {
                if KEYWORDS.contains(&k.as_slice()) {
                    continue;
                }
                let new = self.rename(path, k);
                *k = new.clone();
                let child: Vec<Vec<u8>> = path.iter().cloned().chain([new]).collect();
                self.schema(v, &child, false);
            }
        }
        for (k, v) in members.iter_mut() {
            match (k.as_slice(), v) {
                (b"required", Json::Arr(items)) => {
                    for item in items {
                        if let Json::Str(s) = item {
                            *s = self.rename(path, s);
                        }
                    }
                }
                (b"dependentRequired", Json::Obj(deps)) => {
                    for (dk, dv) in deps.iter_mut() {
                        *dk = self.rename(path, dk);
                        if let Json::Arr(items) = dv {
                            for item in items {
                                if let Json::Str(s) = item {
                                    *s = self.rename(path, s);
                                }
                            }
                        }
                    }
                }
                (b"dependentSchemas", Json::Obj(deps)) => {
                    for (dk, _) in deps.iter_mut() {
                        *dk = self.rename(path, dk);
                    }
                }
                (b"type", Json::Str(t)) => *t = normalize_type(t),
                (b"type", Json::Arr(items)) => {
                    for item in items {
                        if let Json::Str(t) = item {
                            *t = normalize_type(t);
                        }
                    }
                }
                (
                    b"items"
                    | b"contains"
                    | b"additionalProperties"
                    | b"propertyNames"
                    | b"not"
                    | b"if"
                    | b"then"
                    | b"else"
                    | b"unevaluatedProperties"
                    | b"contentSchema",
                    v @ Json::Obj(_),
                ) => self.schema(v, path, false),
                (b"allOf" | b"anyOf" | b"oneOf" | b"prefixItems", Json::Arr(items)) => {
                    for item in items {
                        self.schema(item, path, false);
                    }
                }
                _ => {}
            }
        }
    }

    /// `FUN_000b00e0`: argument keys back to their declared spelling.
    fn restore_args(&self, v: &mut Json, path: &mut Vec<Vec<u8>>) {
        match v {
            Json::Obj(members) => {
                for (k, x) in members.iter_mut() {
                    let key = k.clone();
                    if let Some(orig) = self.original_of(path, &key) {
                        *k = orig.to_vec();
                    }
                    path.push(key);
                    self.restore_args(x, path);
                    path.pop();
                }
            }
            Json::Arr(items) => {
                for x in items {
                    self.restore_args(x, path);
                }
            }
            _ => {}
        }
    }
}

/// The rename table: how to spell calls back in the caller's names.
#[derive(Clone, Debug, Default)]
pub struct Renames {
    tools: Vec<ToolRename>,
}

impl Renames {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// `FUN_0006b6e4`: each call whose `name` is a normalized tool name gets
    /// the declared name back, and its argument keys the declared property
    /// names. A text that does not parse as a JSON array comes back as it
    /// is; one that does is rewritten compactly.
    pub fn restore(&self, calls_json: &[u8]) -> Vec<u8> {
        if self.tools.is_empty() {
            return calls_json.to_vec();
        }
        let Some(mut v) = json::parse(calls_json) else { return calls_json.to_vec() };
        let Json::Arr(calls) = &mut v else { return calls_json.to_vec() };
        for call in calls {
            let Json::Obj(members) = call else { continue };
            let Some(name) = members.iter().find(|(k, _)| k == b"name").map(|(_, v)| v) else { continue };
            let Json::Str(name) = name else { continue };
            let Some(entry) = self.tools.iter().find(|t| &t.name == name) else { continue };
            if let Some((_, v)) = members.iter_mut().find(|(k, _)| k == b"name") {
                *v = Json::Str(entry.original.clone());
            }
            if let Some((_, args)) = members.iter_mut().find(|(k, _)| k == b"arguments") {
                entry.restore_args(args, &mut vec![]);
            }
        }
        json::write(&v)
    }

    /// The declared name of a normalized tool name.
    pub fn original_name<'a>(&'a self, name: &'a [u8]) -> &'a [u8] {
        self.tools.iter().find(|t| t.name == name).map_or(name, |t| t.original.as_slice())
    }

    /// The normalized name of a declared property of a tool (top level).
    pub fn property(&self, tool: &[u8], original: &[u8]) -> Option<&[u8]> {
        self.tools.iter().find(|t| t.name == tool)?.lookup(&[], original)
    }
}

/// The tools as the engine holds them after `needle_init`.
#[derive(Clone, Debug, Default)]
pub struct Toolset {
    /// The normalized tools JSON (compact), as the prompt carries it.
    pub text: Vec<u8>,
    /// Its elements (empty when the text is not a JSON array).
    pub tools: Vec<Json>,
    pub renames: Renames,
    /// Each tool's `triggers` patterns under its normalized name, in
    /// declaration order (tools without any are left out).
    pub triggers: Vec<(Vec<u8>, Vec<String>)>,
}

impl Toolset {
    /// `FUN_00004294` on the compacted tools text. When the text is not an
    /// array whose every element has a non-empty string `name`, it is kept
    /// as it is and nothing is renamed.
    pub fn normalize(raw: &[u8]) -> Toolset {
        let compact = compact_ws(raw);
        let parsed = json::parse(&compact);
        let fallback = |parsed: Option<Json>| {
            let tools = match parsed {
                Some(Json::Arr(items)) => items,
                _ => vec![],
            };
            Toolset { text: compact.clone(), tools, renames: Renames::default(), triggers: vec![] }
        };
        let Some(Json::Arr(items)) = &parsed else { return fallback(parsed) };
        let mut tools: Vec<Json> = Vec::with_capacity(items.len());
        for item in items {
            let inner = match (item.get(b"function"), item.get(b"type")) {
                (Some(f @ Json::Obj(_)), Some(Json::Str(t))) if t == b"function" => f.clone(),
                _ => item.clone(),
            };
            match inner.get(b"name") {
                Some(Json::Str(n)) if !n.is_empty() => tools.push(inner),
                _ => return fallback(parsed),
            }
        }
        let originals: Vec<Vec<u8>> = tools.iter().map(|t| t.get(b"name").and_then(Json::as_str).unwrap_or_default().to_vec()).collect();
        let canon: Vec<Vec<u8>> = originals.iter().map(|n| snake_case(n)).collect();
        let mut renames = Renames::default();
        let mut triggers = vec![];
        for (i, tool) in tools.iter_mut().enumerate() {
            let mut group: Vec<&[u8]> = (0..canon.len()).filter(|&j| canon[j] == canon[i]).map(|j| originals[j].as_slice()).collect();
            group.sort_unstable();
            group.dedup();
            let name = if group.len() > 1 {
                let rank = group.iter().position(|g| *g == originals[i].as_slice()).unwrap_or(0);
                [canon[i].as_slice(), b"__", (rank + 1).to_string().as_bytes()].concat()
            } else {
                canon[i].clone()
            };
            let mut entry = ToolRename { name: name.clone(), original: originals[i].clone(), props: vec![] };
            let Json::Obj(members) = tool else { continue };
            if let Some((_, v)) = members.iter_mut().find(|(k, _)| k == b"name") {
                *v = Json::Str(name.clone());
            }
            if let Some((_, params)) = members.iter_mut().find(|(k, _)| k == b"parameters") {
                entry.schema(params, &[], true);
            }
            if let Some((_, Json::Arr(pats))) = members.iter().find(|(k, _)| k == b"triggers") {
                let pats: Vec<String> = pats.iter().filter_map(Json::as_str).map(|p| String::from_utf8_lossy(p).into_owned()).collect();
                if !pats.is_empty() {
                    triggers.push((name.clone(), pats));
                }
            }
            members.retain(|(k, _)| k != b"triggers");
            renames.tools.push(entry);
        }
        let text = json::write(&Json::Arr(tools.clone()));
        Toolset { text, tools, renames, triggers }
    }

    /// The prompt text of a subset of the tools (retrieval), each element
    /// in the canonical form `FUN_0002b624` writes.
    pub fn subset_text(&self, keep: &[usize]) -> Vec<u8> {
        let mut out = b"[".to_vec();
        for (n, &i) in keep.iter().enumerate() {
            if n > 0 {
                out.push(b',');
            }
            out.extend(canonical(&self.tools[i]));
        }
        out.push(b']');
        out
    }
}

/// `FUN_0002b624`'s per-element text: compact, with numbers that have a
/// fraction or exponent written as Python's `repr(float)` and a negative
/// zero integer as `0`. This is what retrieval embeds.
pub fn canonical(v: &Json) -> Vec<u8> {
    fn fix(v: &Json) -> Json {
        match v {
            Json::Num(n) => {
                if n.iter().any(|c| matches!(c, b'.' | b'e' | b'E')) {
                    let f: f64 = std::str::from_utf8(n).ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    Json::Num(float_repr(f).into_bytes())
                } else if n.iter().filter(|c| c.is_ascii_digit()).all(|&c| c == b'0') {
                    Json::Num(b"0".to_vec())
                } else {
                    v.clone()
                }
            }
            Json::Arr(items) => Json::Arr(items.iter().map(fix).collect()),
            Json::Obj(members) => Json::Obj(members.iter().map(|(k, x)| (k.clone(), fix(x))).collect()),
            _ => v.clone(),
        }
    }
    json::write(&fix(v))
}

/// The keys a JSON-object system text contributes, in this order.
const FACT_KEYS: [&str; 10] = ["date", "locale", "device", "battery", "network", "location", "user", "assistant", "thermostat", "volume"];
const WEEKDAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];

/// `FUN_0002ddbc`: the first `YYYY-MM-DD[T ]HH:MM` in a text, as (date,
/// time).
fn find_timestamp(s: &[u8]) -> Option<(&[u8], &[u8])> {
    if s.len() < 16 {
        return None;
    }
    let d = |c: u8| c.is_ascii_digit();
    (0..=s.len() - 16).find_map(|p| {
        let w = &s[p..p + 16];
        let hit = w[..4].iter().all(|&c| d(c))
            && w[4] == b'-'
            && d(w[5])
            && d(w[6])
            && w[7] == b'-'
            && d(w[8])
            && d(w[9])
            && matches!(w[10], b'T' | b' ')
            && d(w[11])
            && d(w[12])
            && w[13] == b':'
            && d(w[14])
            && d(w[15]);
        hit.then(|| (&w[..10], &w[11..16]))
    })
}

/// `FUN_0002e088` minus the `date: ` label: `YYYY-MM-DD Www HH:MM`.
fn format_timestamp(date: &[u8], time: &[u8]) -> Vec<u8> {
    let num = |b: &[u8]| -> i64 { std::str::from_utf8(b).ok().and_then(|s| s.parse().ok()).unwrap_or(0) };
    let (y, m, d) = (num(&date[..4]), num(&date[5..7]), num(&date[8..10]));
    let a = (14 - m) / 12;
    let yy = y - a;
    let mm = m + 12 * a;
    let n = 365 * yy + yy.div_euclid(4) - yy / 100 + yy / 400 + (153 * mm - 457) / 5 + d - 719469;
    let w = n.rem_euclid(7) as usize;
    [date, b" ", WEEKDAYS[w].as_bytes(), b" ", time].concat()
}

/// The system text the prompt carries (`S`):
///
/// * a JSON object becomes `key: value` pairs joined by `; `, over a fixed
///   list of keys (a `date` value with a timestamp is reformatted);
/// * plain text is kept verbatim, except that text with a timestamp and no
///   `date: ` fact is replaced by `date: YYYY-MM-DD Www HH:MM` alone.
pub fn normalize_system(system: &[u8]) -> Vec<u8> {
    let first = system.iter().find(|c| !matches!(c, b' ' | b'\t' | b'\n' | b'\r'));
    if first == Some(&b'{')
        && let Some(obj @ Json::Obj(_)) = json::parse(system)
    {
        let Json::Obj(members) = &obj else { unreachable!() };
        let mut facts: Vec<Vec<u8>> = vec![];
        for key in FACT_KEYS {
            let value = members.iter().find_map(|(k, v)| {
                if k != key.as_bytes() || !matches!(v, Json::Bool(_) | Json::Num(_) | Json::Str(_)) || v.text().is_empty() {
                    return None;
                }
                Some(v.text().to_vec())
            });
            let Some(mut value) = value else { continue };
            if key == "date"
                && let Some((d, t)) = find_timestamp(&value)
            {
                value = format_timestamp(d, t);
            }
            facts.push([key.as_bytes(), b": ", &value].concat());
        }
        if !facts.is_empty() {
            return facts.join(&b"; "[..]);
        }
    }
    match find_timestamp(system) {
        Some((d, t)) if !contains(system, b"date: ") => [b"date: ".as_slice(), &format_timestamp(d, t)].concat(),
        _ => system.to_vec(),
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(b: Vec<u8>) -> String {
        String::from_utf8(b).unwrap()
    }

    #[test]
    fn snake_case_matches_the_library() {
        for (a, b) in [
            ("userId", "user_id"),
            ("HTTPServer", "http_server"),
            ("user-name", "user_name"),
            ("a__b", "a_b"),
            ("_id", "id"),
            ("setLights", "set_lights"),
            ("get-weather", "get_weather"),
            ("already_snake", "already_snake"),
            ("ABC", "abc"),
            ("roomID2", "room_id2"),
            ("x2Y", "x2_y"),
            ("__", "__"),
            ("café", "café"),
        ] {
            assert_eq!(s(snake_case(a.as_bytes())), b, "{a}");
        }
    }

    #[test]
    fn normalizes_tools() {
        let raw = br#"[{"type": "function", "function": {"name": "setLights", "triggers": ["x"],
            "parameters": {"type": "object", "properties": {"roomName": {"type": "str"}, "Level": {"type": "int"},
            "tags": {"type": "List[str]", "items": {"type": "String"}}}, "required": ["roomName"]}}},
            {"name": "set_lights", "parameters": {"properties": {}}}]"#;
        let t = Toolset::normalize(raw);
        assert_eq!(
            s(t.text.clone()),
            r#"[{"name":"set_lights__1","parameters":{"type":"object","properties":{"room_name":{"type":"str"},"level":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},"required":["room_name"]}},{"name":"set_lights__2","parameters":{"properties":{}}}]"#
        );
        let back = t.renames.restore(br#"[{"name":"set_lights__1","arguments":{"room_name":"kitchen","level":3}}]"#);
        assert_eq!(s(back), r#"[{"name":"setLights","arguments":{"roomName":"kitchen","Level":3}}]"#);
    }

    #[test]
    fn property_collisions_count_from_two() {
        let t = Toolset::normalize(br#"[{"name":"f","parameters":{"properties":{"userId":{},"user_id":{},"USER_ID":{}}}}]"#);
        assert_eq!(s(t.text), r#"[{"name":"f","parameters":{"properties":{"user_id":{},"user_id__2":{},"user_id__3":{}}}}]"#);
    }

    #[test]
    fn system_text() {
        assert_eq!(s(normalize_system(b"date: 2026-09-25 Fri 10:00; tz: x")), "date: 2026-09-25 Fri 10:00; tz: x");
        assert_eq!(s(normalize_system(b"Now is 2026-09-25T10:00:33Z. Be brief.")), "date: 2026-09-25 Fri 10:00");
        assert_eq!(
            s(normalize_system(br#"{"user":"Ann","date":"2026-09-26 08:15","x":1,"volume":30}"#)),
            "date: 2026-09-26 Sat 08:15; user: Ann; volume: 30"
        );
        assert_eq!(s(normalize_system(br#"{"x":1}"#)), r#"{"x":1}"#);
        assert_eq!(s(normalize_system(b"  be brief ")), "  be brief ");
        assert_eq!(s(format_timestamp(b"1970-01-01", b"00:00")), "1970-01-01 Thu 00:00");
        assert_eq!(s(format_timestamp(b"2000-02-29", b"12:00")), "2000-02-29 Tue 12:00");
    }

    #[test]
    fn canonical_numbers() {
        let v = json::parse(br#"{"a":1.50,"b":-0,"c":1e5,"d":7}"#).unwrap();
        assert_eq!(s(canonical(&v)), r#"{"a":1.5,"b":0,"c":100000.0,"d":7}"#);
    }
}
