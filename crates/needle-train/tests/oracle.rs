//! Training parity with the JAX reference: LoRA init, loss, and gradients
//! (`tests/oracle/fixtures.py grads`).

use std::path::{Path, PathBuf};

use needle_core::Tokenizer;
use needle_core::checkpoint::{load_checkpoint, read_adapter};
use needle_core::render::{encode_example, fit_max_len, read_examples};
use needle_engine::Weights;
use needle_train::finetune::{init_lora, target_path};
use needle_train::graph::{Batch, Graph, LoraParams, TARGETS};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn lora_from(path: &Path) -> LoraParams {
    let ad = read_adapter(path).unwrap();
    let a = std::array::from_fn(|t| ad.lora[&target_path(t)].a.data.clone());
    let b = std::array::from_fn(|t| ad.lora[&target_path(t)].b.data.clone());
    LoraParams { rank: ad.rank.unwrap_or(16), a, b }
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    num / den.max(1e-30)
}

#[test]
fn loss_and_grads_match_jax() {
    let meta = root().join("tests/oracle/grads.json");
    if !meta.exists() {
        eprintln!("skipping: run tests/oracle/fixtures.py grads");
        return;
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta).unwrap()).unwrap();
    let (params, config, _) = load_checkpoint(&root().join("models/needle3.safetensors")).unwrap();
    let tok = Tokenizer::from_model_file(&root().join("models/tokenizer.model")).unwrap();
    let (examples, _) = read_examples(&root().join("tests/oracle/grad_batch.jsonl")).unwrap();
    let max_len = fit_max_len(&examples, &tok, 1024).unwrap();
    let enc: Vec<_> = examples.iter().map(|e| encode_example(&tok, e, max_len).unwrap()).collect();
    let rows: Vec<(&[u32], &[f32])> = enc.iter().map(|(i, m)| (i.as_slice(), m.as_slice())).collect();
    let batch = Batch::from_padded(&rows);

    let mut graph = Graph::new(Weights::from_checkpoint(&params, &config).unwrap(), 2.0);

    // Init matches jax.random bit-for-bit (up to erfinv rounding).
    let init = init_lora(&graph, 16, 0);
    let py_init = lora_from(&root().join("models/oracle_lora_init.safetensors"));
    for t in 0..5 {
        let e = rel_err(&init.a[t], &py_init.a[t]);
        eprintln!("init A {}: rel err {e:.2e}", TARGETS[t]);
        assert!(e < 1e-5, "LoRA init differs for {}", TARGETS[t]);
    }

    graph.merge(&init);
    let l0 = graph.loss(&batch);
    let want0 = meta["loss_init"].as_f64().unwrap() as f32;
    eprintln!("loss at init: rust {l0:.5} jax {want0:.5}");

    let state = lora_from(&root().join("models/oracle_lora_state.safetensors"));
    graph.merge(&state);
    let t0 = std::time::Instant::now();
    let out = graph.step(&batch, &state);
    eprintln!("rust step ({} rows): {:?}", batch.rows(), t0.elapsed());
    eprintln!("{}", needle_engine::prof::report());
    let want = meta["loss"].as_f64().unwrap() as f32;
    eprintln!("loss: rust {:.5} jax {want:.5}", out.loss);
    let g = lora_from(&root().join("models/oracle_lora_grads.safetensors"));
    for t in 0..5 {
        let ea = rel_err(&out.grads.a[t], &g.a[t]);
        let eb = rel_err(&out.grads.b[t], &g.b[t]);
        eprintln!("grad {}: A rel err {ea:.3e}  B rel err {eb:.3e}", TARGETS[t]);
        let _ = (ea, eb);
    }
    assert!((out.loss - want).abs() < 0.02 * want.abs().max(0.1));

    // Float numerics: the reverse pass itself, without quantizer chaos.
    let mut fgraph = Graph::with_numerics(Weights::from_checkpoint(&params, &config).unwrap(), 2.0, false);
    fgraph.merge(&state);
    let fout = fgraph.step(&batch, &state);
    let fwant = meta["float_loss"].as_f64().unwrap() as f32;
    eprintln!("float loss: rust {:.6} jax {fwant:.6}", fout.loss);
    let fg = lora_from(&root().join("models/oracle_lora_fgrads.safetensors"));
    for t in 0..5 {
        let ea = rel_err(&fout.grads.a[t], &fg.a[t]);
        let eb = rel_err(&fout.grads.b[t], &fg.b[t]);
        eprintln!("float grad {}: A rel err {ea:.3e}  B rel err {eb:.3e}", TARGETS[t]);
        assert!(ea < 1e-3 && eb < 1e-3, "float gradient mismatch for {}", TARGETS[t]);
    }
    assert!((fout.loss - fwant).abs() < 1e-4);

    // Prefix sharing is exact: the same loss and gradients as separate rows.
    let unshared = Batch::unshared(&rows);
    eprintln!("rows: shared {} unshared {}", batch.rows(), unshared.rows());
    let t1 = std::time::Instant::now();
    let uout = fgraph.step(&unshared, &state);
    eprintln!("unshared step: {:?}", t1.elapsed());
    assert!((uout.loss - fout.loss).abs() < 1e-5, "{} vs {}", uout.loss, fout.loss);
    for t in 0..5 {
        let e = rel_err(&fout.grads.a[t], &uout.grads.a[t]).max(rel_err(&fout.grads.b[t], &uout.grads.b[t]));
        eprintln!("shared vs unshared {}: {e:.2e}", TARGETS[t]);
        assert!(e < 1e-4);
    }

    assert!((l0 - want0).abs() < 0.02 * want0.abs().max(0.1));
}
