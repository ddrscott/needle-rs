//! The tool list as the post-processor sees it (`FUN_00076c68`) and the
//! per-parameter lookup (`FUN_00083ec4`).

use super::json::{self, Json};
use super::text::lower;

#[derive(Clone, Debug, Default)]
pub struct ParamInfo {
    pub found: bool,
    pub required: bool,
    pub has_enum: bool,
    pub has_default: bool,
    pub default: Vec<u8>,
    pub description: Vec<u8>,
    /// The schema `type`, ASCII-lowercased; empty unless it is a string.
    pub ty: Vec<u8>,
    pub items_ty: Vec<u8>,
    /// The string members of `enum`.
    pub options: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub key: Vec<u8>,
    pub info: ParamInfo,
    /// Exempt from `validation.ungrounded`: a non-empty enum, a `const`, or
    /// `"grounded": true` (from the compiled grammar schema, not this spec).
    pub closed: bool,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: Vec<u8>,
    pub description: Vec<u8>,
    pub params: Vec<Param>,
}

/// JSON-Schema keywords that a flat `parameters` object may carry beside
/// its parameters; taken as a parameter only with a `type` or
/// `description`.
const SCHEMA_KEYWORDS: [&str; 42] = [
    "$defs",
    "$id",
    "$ref",
    "$schema",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "contains",
    "default",
    "definitions",
    "dependentRequired",
    "dependentSchemas",
    "description",
    "else",
    "enum",
    "examples",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "if",
    "items",
    "maxItems",
    "maxLength",
    "maximum",
    "minItems",
    "minLength",
    "minimum",
    "multipleOf",
    "not",
    "oneOf",
    "pattern",
    "patternProperties",
    "prefixItems",
    "properties",
    "propertyNames",
    "required",
    "then",
    "title",
    "type",
    "unevaluatedProperties",
    "uniqueItems",
];

/// Parse the tools JSON text into the tool list.
pub fn parse_tools(text: &[u8]) -> Vec<Tool> {
    let Some(Json::Arr(items)) = json::parse(text) else { return vec![] };
    let mut tools = vec![];
    for t in &items {
        let Some(Json::Str(name)) = t.get(b"name") else { continue };
        let description = t.get(b"description").and_then(Json::as_str).unwrap_or_default().to_vec();
        let mut params = vec![];
        if let Some(p @ Json::Obj(pm)) = t.get(b"parameters") {
            let props = p.get(b"properties");
            let req = p.get(b"required");
            let nested = matches!(props, Some(Json::Obj(_)));
            let source: &Vec<(Vec<u8>, Json)> = match props {
                Some(Json::Obj(m)) => m,
                _ => pm,
            };
            for (key, v) in source {
                if !matches!(v, Json::Obj(_)) {
                    continue;
                }
                if !nested
                    && SCHEMA_KEYWORDS.iter().any(|k| k.as_bytes() == key.as_slice())
                    && v.get(b"type").is_none()
                    && v.get(b"description").is_none()
                {
                    continue;
                }
                let required = if nested {
                    matches!(req, Some(Json::Arr(r)) if r.iter().any(|x| x.as_str() == Some(key)))
                } else {
                    matches!(v.get(b"required"), Some(Json::Bool(true)))
                };
                let info = param_info(v, required);
                let closed = info.has_enum || v.get(b"const").is_some() || matches!(v.get(b"grounded"), Some(Json::Bool(true)));
                params.push(Param { key: key.clone(), info, closed });
            }
        }
        tools.push(Tool { name: name.clone(), description, params });
    }
    tools
}

fn param_info(v: &Json, required: bool) -> ParamInfo {
    let (has_enum, options) = match v.get(b"enum") {
        Some(Json::Arr(items)) => (!items.is_empty(), items.iter().filter_map(Json::as_str).map(<[u8]>::to_vec).collect()),
        _ => (false, vec![]),
    };
    let (has_default, default) = match v.get(b"default") {
        Some(d @ (Json::Bool(_) | Json::Num(_) | Json::Str(_))) => (true, d.text().to_vec()),
        _ => (false, vec![]),
    };
    let items_ty = match v.get(b"items") {
        Some(it @ Json::Obj(_)) => it.get(b"type").and_then(Json::as_str).map(lower).unwrap_or_default(),
        _ => vec![],
    };
    ParamInfo {
        found: true,
        required,
        has_enum,
        has_default,
        default,
        description: v.get(b"description").and_then(Json::as_str).unwrap_or_default().to_vec(),
        ty: v.get(b"type").and_then(Json::as_str).map(lower).unwrap_or_default(),
        items_ty,
        options,
    }
}

/// The last tool named exactly `name`.
pub fn find_tool<'a>(tools: &'a [Tool], name: &[u8]) -> Option<&'a Tool> {
    tools.iter().rev().find(|t| t.name == name)
}

/// `FUN_00083ec4`: the parameter spec for (tool, key); all empty when
/// either is unknown. The last matching tool and parameter win.
pub fn info(tools: &[Tool], tool: &[u8], key: &[u8]) -> ParamInfo {
    find_tool(tools, tool).and_then(|t| t.params.iter().rev().find(|p| p.key == key)).map(|p| p.info.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_and_flat_schemas() {
        let src = br#"[{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"String","enum":["p",1]},"y":{"type":["string","null"],"default":3}},"required":["x"]}},
            {"name":"b","parameters":{"q":{"type":"string","required":true},"title":{"minLength":1},"type":{"description":"kind"}}},
            {"description":"no name"}]"#;
        let tools = parse_tools(src);
        assert_eq!(tools.len(), 2);
        let x = info(&tools, b"a", b"x");
        assert!(x.required && x.has_enum && x.ty == b"string" && x.options == vec![b"p".to_vec()]);
        let y = info(&tools, b"a", b"y");
        assert!(!y.required && y.ty.is_empty() && y.has_default && y.default == b"3");
        let keys: Vec<_> = tools[1].params.iter().map(|p| p.key.clone()).collect();
        assert_eq!(keys, vec![b"q".to_vec(), b"type".to_vec()]);
        assert!(info(&tools, b"b", b"q").required);
        assert!(!info(&tools, b"zzz", b"q").found);
    }
}
