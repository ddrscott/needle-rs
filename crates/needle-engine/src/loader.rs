//! Load a model from whatever the user points at: a training checkpoint
//! (`.safetensors`) or a deployment archive (`.cact`).

use std::path::Path;

use anyhow::{Context, Result};
use needle_core::Tokenizer;
use needle_core::cact::Archive;
use needle_core::checkpoint::load_checkpoint;

use crate::{Model, Weights};

/// A model and the tokenizer that goes with it.
pub struct Loaded {
    pub model: Model,
    pub tokenizer: Tokenizer,
}

/// The tokenizer for a checkpoint: `tokenizer.model` beside it, in the
/// working directory, in `models/`, or in the Hugging Face download cache.
pub fn find_tokenizer(near: &Path) -> Result<Tokenizer> {
    let mut candidates = vec![];
    if let Some(dir) = near.parent() {
        candidates.push(dir.join("tokenizer.model"));
        candidates.push(dir.join("../tokenizer.model"));
    }
    candidates.push("tokenizer.model".into());
    candidates.push("models/tokenizer.model".into());
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(Path::new(&home).join(".cache/needle-rs/tokenizer.model"));
    }
    for c in &candidates {
        if c.exists() {
            return Tokenizer::from_model_file(c);
        }
    }
    anyhow::bail!("no tokenizer.model found near {} (try `needle download tokenizer`)", near.display())
}

pub fn load(path: &Path) -> Result<Loaded> {
    if path.extension().and_then(|e| e.to_str()) == Some("cact") {
        let archive = Archive::open(path)?;
        let tokenizer = Tokenizer::from_blob(archive.tokenizer_blob()?)?;
        let fast = std::env::var_os("NEEDLE_F32").is_none();
        let (weights, q) = Weights::from_archive_mode(&archive, fast).with_context(|| format!("load {}", path.display()))?;
        let model = match q {
            Some(q) => Model::quantized(weights, q),
            None => Model::new(weights),
        };
        Ok(Loaded { model, tokenizer })
    } else {
        let (params, config, _) = load_checkpoint(path)?;
        let tokenizer = find_tokenizer(path)?;
        if tokenizer.vocab_size() > config.vocab_size {
            anyhow::bail!("tokenizer has {} pieces, the model vocabulary holds {}", tokenizer.vocab_size(), config.vocab_size);
        }
        Ok(Loaded { model: Model::new(Weights::from_checkpoint(&params, &config)?), tokenizer })
    }
}
