//! The Needle 3 specification in Rust.
//!
//! Everything here is framework-free and mirrors the Python reference package
//! (`cactus-needle`) exactly: the model config and its ladder rules, the
//! SentencePiece tokenizer, safetensors checkpoints and LoRA adapters, the CQ
//! quantizer, the `.cact` deployment archive, and the chat prompt rendering.

pub mod cact;
pub mod checkpoint;
pub mod config;
pub mod pyjson;
pub mod quant;
pub mod render;
pub mod rng;
pub mod synth;
pub mod tensor;
pub mod tokenizer;

pub use config::Config;
pub use tensor::{Params, Tensor};
pub use tokenizer::Tokenizer;
