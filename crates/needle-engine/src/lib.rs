//! Needle 3 inference in Rust.

pub mod linalg;
pub mod model;
pub mod weights;

pub use model::{ForwardOut, Model, Numerics, Outputs, Session};
pub use weights::Weights;
pub mod agent;
mod attn;
pub mod cpu;
pub mod generate;
pub mod grammar;
pub mod hada;
pub mod harness;
pub mod heads;
pub mod infer;
pub mod loader;
pub mod prof;
pub mod qlinear;
pub mod rules;
pub mod team;
pub mod toolset;
