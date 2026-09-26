//! `.safetensors` checkpoints and LoRA adapters in the reference layout:
//! `/`-joined Flax parameter paths as tensor names, config and run facts as
//! JSON strings in the metadata.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::{Map, Value};

use crate::config::Config;
use crate::tensor::{Params, Tensor};

pub const CHECKPOINT_FORMAT_VERSION: i64 = 2;

fn require_safetensors(path: &Path, what: &str) -> Result<()> {
    if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        bail!("{what} must be a .safetensors file for security (got {})", path.display());
    }
    Ok(())
}

fn to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    let b = view.data();
    Ok(match view.dtype() {
        Dtype::F32 => b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect(),
        Dtype::F16 => b.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c).to_f32()).collect(),
        Dtype::BF16 => b.as_chunks::<2>().0.iter().map(|c| bf16::from_le_bytes(*c).to_f32()).collect(),
        Dtype::F64 => b.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c) as f32).collect(),
        d => bail!("unsupported tensor dtype {d:?}"),
    })
}

/// Everything in a safetensors file: tensors as f32 plus string metadata.
pub fn read_safetensors(path: &Path) -> Result<(Params, HashMap<String, String>)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: the file is opened read-only and not modified while mapped.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let (_, meta) = SafeTensors::read_metadata(&mmap)?;
    let metadata = meta.metadata().clone().unwrap_or_default();
    let st = SafeTensors::deserialize(&mmap)?;
    let mut params = Params::new();
    for (name, view) in st.tensors() {
        params.insert(name.to_string(), Tensor::new(view.shape().to_vec(), to_f32(&view)?));
    }
    Ok((params, metadata))
}

pub fn write_safetensors(path: &Path, tensors: &BTreeMap<String, Tensor>, metadata: HashMap<String, String>) -> Result<()> {
    let bytes: Vec<(String, Vec<u8>, Vec<usize>)> =
        tensors.iter().map(|(k, t)| (k.clone(), bytemuck::cast_slice::<f32, u8>(&t.data).to_vec(), t.shape.clone())).collect();
    let views: Vec<(String, TensorView<'_>)> =
        bytes.iter().map(|(k, b, s)| Ok((k.clone(), TensorView::new(Dtype::F32, s.clone(), b)?))).collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, Some(metadata), path)?;
    Ok(())
}

fn json_or(text: Option<&String>, default: Value) -> Result<Value> {
    match text.map(String::as_str) {
        None | Some("") => Ok(default),
        Some(t) => Ok(serde_json::from_str(t)?),
    }
}

fn int_or_none(text: Option<&String>) -> Result<Option<i64>> {
    match text.map(String::as_str) {
        None | Some("") => Ok(None),
        Some(t) => Ok(Some(t.parse()?)),
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub format_version: Option<i64>,
    pub params: Params,
    pub config: Map<String, Value>,
    pub step: Option<i64>,
    pub run: Value,
}

pub fn read_checkpoint(path: &Path) -> Result<Checkpoint> {
    require_safetensors(path, "checkpoint")?;
    let (params, meta) = read_safetensors(path)?;
    let config = match json_or(meta.get("config"), Value::Object(Map::new()))? {
        Value::Object(m) => m,
        _ => bail!("checkpoint config metadata is not an object"),
    };
    Ok(Checkpoint {
        format_version: int_or_none(meta.get("format_version"))?,
        params,
        config,
        step: int_or_none(meta.get("step"))?,
        run: json_or(meta.get("run"), Value::Object(Map::new()))?,
    })
}

pub fn write_checkpoint(path: &Path, ckpt: &Checkpoint) -> Result<()> {
    require_safetensors(path, "checkpoint")?;
    let mut meta = HashMap::new();
    meta.insert("format_version".into(), ckpt.format_version.map(|v| v.to_string()).unwrap_or_default());
    meta.insert("config".into(), serde_json::to_string(&ckpt.config)?);
    meta.insert("step".into(), ckpt.step.map(|v| v.to_string()).unwrap_or_default());
    meta.insert("run".into(), serde_json::to_string(&ckpt.run)?);
    write_safetensors(path, &ckpt.params, meta)
}

/// `run.load_checkpoint`: a format-v2 Needle 3 checkpoint, `mtp_*` dropped.
pub fn load_checkpoint(path: &Path) -> Result<(Params, Config, Value)> {
    let ckpt = read_checkpoint(path)?;
    if ckpt.format_version != Some(CHECKPOINT_FORMAT_VERSION) {
        bail!(
            "{} is not a format-v{CHECKPOINT_FORMAT_VERSION} checkpoint (got format_version={:?}). \
             Old encoder-decoder/tool-calling checkpoints are incompatible with this branch.",
            path.display(),
            ckpt.format_version
        );
    }
    if ckpt.config.contains_key("attn_dim") && !ckpt.config.contains_key("qk_head_dim") {
        bail!("{} is a Needle 2 checkpoint; this package fine-tunes and builds Needle 3", path.display());
    }
    let config = Config::from_saved(&ckpt.config)?;
    let params = ckpt.params.into_iter().filter(|(k, _)| !k.starts_with("mtp_")).collect();
    Ok((params, config, ckpt.run))
}

/// One adapted weight: `W + scale * A @ B`, stacked over layers.
#[derive(Clone, Debug)]
pub struct LoraPair {
    pub a: Tensor,
    pub b: Tensor,
}

#[derive(Clone, Debug)]
pub struct Adapter {
    /// Keyed by the adapted parameter path, e.g. `stack/layers/block/self_attn/q_proj/kernel`.
    pub lora: BTreeMap<String, LoraPair>,
    pub scale: f64,
    pub base: Option<String>,
    pub rank: Option<usize>,
    pub seed: Option<i64>,
}

pub fn write_adapter(path: &Path, adapter: &Adapter) -> Result<()> {
    require_safetensors(path, "adapter")?;
    let mut tensors = BTreeMap::new();
    for (name, pair) in &adapter.lora {
        tensors.insert(format!("lora/{name}/A"), pair.a.clone());
        tensors.insert(format!("lora/{name}/B"), pair.b.clone());
    }
    let mut meta = HashMap::new();
    meta.insert("scale".into(), crate::pyjson::float_repr(adapter.scale));
    meta.insert("base".into(), serde_json::to_string(&adapter.base)?);
    meta.insert("rank".into(), serde_json::to_string(&adapter.rank)?);
    meta.insert("seed".into(), serde_json::to_string(&adapter.seed)?);
    write_safetensors(path, &tensors, meta)
}

pub fn read_adapter(path: &Path) -> Result<Adapter> {
    require_safetensors(path, "adapter")?;
    let (tensors, meta) = read_safetensors(path)?;
    let mut lora: BTreeMap<String, (Option<Tensor>, Option<Tensor>)> = BTreeMap::new();
    for (name, t) in tensors {
        let rest = name.strip_prefix("lora/").unwrap_or(&name);
        let (key, matrix) = rest.rsplit_once('/').context("adapter tensor name has no matrix suffix")?;
        let slot = lora.entry(key.to_string()).or_default();
        match matrix {
            "A" => slot.0 = Some(t),
            "B" => slot.1 = Some(t),
            m => bail!("unknown adapter matrix {m}"),
        }
    }
    let lora = lora
        .into_iter()
        .map(|(k, (a, b))| {
            Ok((k.clone(), LoraPair { a: a.with_context(|| format!("{k} has no A"))?, b: b.with_context(|| format!("{k} has no B"))? }))
        })
        .collect::<Result<_>>()?;
    let get = |k: &str| json_or(meta.get(k), Value::Null);
    Ok(Adapter {
        lora,
        scale: get("scale")?.as_f64().context("adapter has no scale")?,
        base: get("base")?.as_str().map(str::to_string),
        rank: get("rank")?.as_u64().map(|r| r as usize),
        seed: get("seed")?.as_i64(),
    })
}

/// `merge_lora`: `W + scale * A @ B` per adapted weight (stacked over layers).
pub fn merge_lora(params: &mut Params, adapter: &Adapter) -> Result<()> {
    for (name, pair) in &adapter.lora {
        let w = params.get_mut(name).with_context(|| format!("adapter targets missing weight {name}"))?;
        let n = w.ndim();
        let (din, dout) = (w.shape[n - 2], w.shape[n - 1]);
        let r = pair.a.shape[pair.a.ndim() - 1];
        let lead: usize = w.shape[..n - 2].iter().product();
        for l in 0..lead {
            let a = &pair.a.data[l * din * r..(l + 1) * din * r];
            let b = &pair.b.data[l * r * dout..(l + 1) * r * dout];
            let wl = &mut w.data[l * din * dout..(l + 1) * din * dout];
            for i in 0..din {
                for j in 0..dout {
                    let mut acc = 0f32;
                    for k in 0..r {
                        acc += a[i * r + k] * b[k * dout + j];
                    }
                    wl[i * dout + j] += (adapter.scale as f32) * acc;
                }
            }
        }
    }
    Ok(())
}
