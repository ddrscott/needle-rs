//! Logit parity against the JAX reference (`tests/oracle/fixtures.py logits`).

use std::path::{Path, PathBuf};

use needle_core::Tokenizer;
use needle_core::checkpoint::load_checkpoint;
use needle_engine::{Model, Numerics, Outputs, Weights};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_f32(name: &str) -> Vec<f32> {
    let b = std::fs::read(root().join("models").join(format!("{name}.f32"))).unwrap();
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn compare(ours: &[f32], theirs: &[f32], vocab: usize) -> (f32, f32, usize) {
    let mut max_abs = 0f32;
    let mut max_ref = 0f32;
    let mut argmax_diff = 0;
    for (a, b) in ours.chunks(vocab).zip(theirs.chunks(vocab)) {
        for (x, y) in a.iter().zip(b) {
            max_abs = max_abs.max((x - y).abs());
            max_ref = max_ref.max(y.abs());
        }
        let am = |r: &[f32]| r.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0;
        if am(a) != am(b) {
            argmax_diff += 1;
        }
    }
    (max_abs, max_ref, argmax_diff)
}

fn median_row_max(a: &[f32], b: &[f32], vocab: usize) -> f32 {
    let mut per: Vec<f32> =
        a.chunks(vocab).zip(b.chunks(vocab)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max)).collect();
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per[per.len() / 2]
}

#[test]
fn logits_match_jax() {
    let meta_path = root().join("tests/oracle/logits.json");
    let ckpt = root().join("models/needle3.safetensors");
    if !meta_path.exists() || !ckpt.exists() {
        eprintln!("skipping: oracle fixtures not generated");
        return;
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let tok = Tokenizer::from_model_file(&root().join("models/tokenizer.model")).unwrap();
    let prompt = meta["prompt"].as_str().unwrap();
    let mut want_ids = vec![needle_core::tokenizer::BOS_ID];
    want_ids.extend(tok.encode(prompt));
    assert_eq!(want_ids, ids, "prompt tokenization differs");

    let (params, config, _) = load_checkpoint(&ckpt).unwrap();
    let vocab = config.out_rows();
    let mut model = Model::new(Weights::from_checkpoint(&params, &config).unwrap());

    let mut s = model.session();
    let t0 = std::time::Instant::now();
    let out = model.forward(&mut s, &ids, Outputs::AllLogits);
    eprintln!("rust prefill {} tokens: {:?} (jax {:.1} ms)", ids.len(), t0.elapsed(), meta["forward_s"].as_f64().unwrap() * 1e3);
    let (max_abs, max_ref, am) = compare(&out.data, &load_f32("oracle_logits"), vocab);
    eprintln!("float32: max |diff| {max_abs:.2e} of max |logit| {max_ref:.1}, argmax differs at {am} positions");
    assert!(max_abs < 2e-2 * max_ref.max(1.0), "logits diverge");
    assert_eq!(am, 0);

    // Incremental decode reproduces the full forward.
    let mut s2 = model.session();
    let split = ids.len() - 5;
    model.forward(&mut s2, &ids[..split], Outputs::None);
    let mut inc = Vec::new();
    for &tk in &ids[split..] {
        inc.extend(model.forward(&mut s2, &[tk], Outputs::AllLogits).data);
    }
    let tail = &out.data[split * vocab..];
    let (inc_abs, _, inc_am) = compare(&inc, tail, vocab);
    eprintln!("incremental vs full: max |diff| {inc_abs:.2e}");
    assert!(inc_abs < 1e-3 && inc_am == 0);

    // Finetune numerics: CQ W4 STE weights, A8 activations, int8 KV.
    model.w.cq_fake_quantize(4);
    model.numerics = Numerics { quant: true };
    let mut s3 = model.session();
    let q = model.forward(&mut s3, &ids, Outputs::AllLogits);
    let (qa, qr, qam) = compare(&q.data, &load_f32("oracle_qlogits"), vocab);
    eprintln!("quant: max |diff| {qa:.2e} of {qr:.1}, argmax differs at {qam} positions");
    // The quantized network is chaotic: JAX against itself with weights
    // nudged by 1e-7 differs by median 0.36, max 2.7, 2 argmax flips on
    // this prompt. Parity means staying inside that floor.
    let med = median_row_max(&q.data, &load_f32("oracle_qlogits"), vocab);
    eprintln!("quant: median per-position max |diff| {med:.3}");
    assert!(med < 0.6 && qa < 10.0 && qam <= 4);
}

#[test]
fn quant_parts_match_jax() {
    let ckpt = root().join("models/needle3.safetensors");
    let meta_path = root().join("tests/oracle/logits.json");
    if !meta_path.exists() || !ckpt.exists() || !root().join("models/oracle_wq_logits.f32").exists() {
        return;
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (params, config, _) = load_checkpoint(&ckpt).unwrap();
    let vocab = config.out_rows();

    let mut m = Model::new(Weights::from_checkpoint(&params, &config).unwrap());
    m.numerics = Numerics { quant: true };
    let a = m.forward(&mut m.session(), &ids, Outputs::AllLogits);
    let (d, r, am) = compare(&a.data, &load_f32("oracle_aq_logits"), vocab);
    eprintln!("A8+KV8 only: max |diff| {d:.2e} of {r:.1}, argmax differs at {am}");
    let med = median_row_max(&a.data, &load_f32("oracle_aq_logits"), vocab);
    eprintln!("A8+KV8 only: median {med:.3}");
    assert!(med < 0.6);

    let mut m = Model::new(Weights::from_checkpoint(&params, &config).unwrap());
    m.w.cq_fake_quantize(4);
    let w = m.forward(&mut m.session(), &ids, Outputs::AllLogits);
    let (d, r, am) = compare(&w.data, &load_f32("oracle_wq_logits"), vocab);
    eprintln!("CQ W4 only: max |diff| {d:.2e} of {r:.1}, argmax differs at {am}");
    assert!(d < 1.0 && am == 0);
}

#[test]
fn cells_by_layer() {
    let ckpt = root().join("models/needle3.safetensors");
    if !root().join("models/oracle_qcells.f32").exists() {
        return;
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(root().join("tests/oracle/logits.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (params, config, _) = load_checkpoint(&ckpt).unwrap();
    let (l1, d) = (config.num_layers + 1, config.d_model);
    for (quant, name) in [(false, "oracle_cells"), (true, "oracle_qcells")] {
        let mut m = Model::new(Weights::from_checkpoint(&params, &config).unwrap());
        m.numerics = Numerics { quant };
        let out = m.forward_cells(&mut m.session(), &ids, Outputs::None);
        let ours = out.cells.unwrap();
        let theirs = load_f32(name);
        let t = ids.len();
        let mut line = String::new();
        for l in 0..l1 {
            let (mut mx, mut mref) = (0f32, 0f32);
            let mut first_pos = None;
            for i in 0..t {
                let a = &ours[(i * l1 + l) * d..(i * l1 + l + 1) * d];
                let b = &theirs[(i * l1 + l) * d..(i * l1 + l + 1) * d];
                let m = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                if m > 1e-3 && first_pos.is_none() {
                    first_pos = Some(i);
                }
                mx = mx.max(m);
                mref = mref.max(b.iter().map(|v| v.abs()).fold(0f32, f32::max));
            }
            line += &format!("L{l}:{mx:.1e}/{mref:.0}@{first_pos:?} ");
        }
        eprintln!("quant={quant}: {line}");
    }
}

#[test]
fn confidence_head_matches_jax() {
    let meta_path = root().join("tests/oracle/logits.json");
    let ckpt = root().join("models/needle3.safetensors");
    if !meta_path.exists() || !ckpt.exists() {
        return;
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let (params, config, _) = load_checkpoint(&ckpt).unwrap();
    let m = Model::new(Weights::from_checkpoint(&params, &config).unwrap());
    let cells = needle_engine::heads::cells(&m, &ids);
    let got = needle_engine::heads::confidence_logit(&m, &cells, ids.len()).unwrap();
    let want = meta["confidence_logit"].as_f64().unwrap() as f32;
    eprintln!("confidence logit: rust {got:.5} jax {want:.5}");
    assert!((got - want).abs() < 1e-3);
    let e = needle_engine::heads::embedding(&m, &cells, ids.len()).unwrap();
    assert_eq!(e.len(), 4 * config.d_model);
}

#[test]
#[ignore]
fn bench_archive_qkvg() {
    let path = root().join("models/needle3.cact");
    let a = needle_core::cact::Archive::open(&path).unwrap();
    let (_, q) = Weights::from_archive_mode(&a, true).unwrap();
    let q = q.unwrap();
    let x: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
    let n = 20000;
    for (name, lin) in [("qkvg", &q.qkvg[0]), ("out", &q.out[0])] {
        let t0 = std::time::Instant::now();
        for i in 0..n {
            std::hint::black_box(q.qkvg[i % 20].apply(&x, 1));
        }
        let all_layers = t0.elapsed().as_secs_f64() / n as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(lin.apply(&x, 1));
        }
        eprintln!("{name}: same layer {:.2}us, rotating layers {:.2}us", t0.elapsed().as_secs_f64() / n as f64 * 1e6, all_layers * 1e6);
        let act = needle_engine::qlinear::QAct::new(&x, 1, 768, lin.act_bits());
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(lin.matmul(&act));
        }
        eprintln!("{name}: matmul only {:.2}us", t0.elapsed().as_secs_f64() / n as f64 * 1e6);
    }
}

#[test]
fn rollback_matches_recompute() {
    let ckpt = root().join("models/needle3.cact");
    if !ckpt.exists() {
        return;
    }
    let loaded = needle_engine::loader::load(&ckpt).unwrap();
    let m = &loaded.model;
    let ids: Vec<u32> = loaded.tokenizer.encode("<|im_start|>user\nturn the kitchen lights on please<|im_end|>\n<|im_start|>assistant\n");
    let mut a = m.session();
    m.forward(&mut a, &ids[..10], Outputs::None);
    let want = m.forward(&mut a, &ids[10..], Outputs::LastLogits).data;
    let mut b = m.session();
    m.forward(&mut b, &ids[..10], Outputs::None);
    m.forward(&mut b, &[7, 8, 9, 10, 11], Outputs::None);
    b.rollback(m, 5);
    let got = m.forward(&mut b, &ids[10..], Outputs::LastLogits).data;
    let diff = want.iter().zip(&got).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(diff < 1e-3, "rollback diverged by {diff}");
}

#[test]
fn native_decode_tracks_general() {
    let path = root().join("models/needle3.cact");
    if !path.exists() {
        return;
    }
    let mut loaded = needle_engine::loader::load(&path).unwrap();
    let tok = &loaded.tokenizer;
    let ids: Vec<u32> = std::iter::once(needle_core::tokenizer::BOS_ID)
        .chain(tok.encode("<|im_start|>user\n<tools>[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}]</tools>\nweather in Lagos?<|im_end|>\n<|im_start|>assistant\n"))
        .collect();
    let mut runs = vec![];
    for fast in [false, true] {
        // `quant` routes packed weights through the general forward pass.
        loaded.model.numerics.quant = !fast;
        let m = &loaded.model;
        let mut s = m.session();
        s.cells = Some(vec![]);
        let mut logits = m.forward(&mut s, &ids, Outputs::LastLogits).data;
        let mut seq = vec![];
        let mut all = vec![];
        for _ in 0..24 {
            let next = needle_engine::generate::argmax(&logits) as u32;
            seq.push(next);
            logits = m.forward(&mut s, &[next], Outputs::LastLogits).data;
            all.extend_from_slice(&logits);
        }
        runs.push((seq, all, s.cells.unwrap()));
    }
    assert_eq!(runs[0].0, runs[1].0, "greedy tokens differ");
    let d = runs[0].1.iter().zip(&runs[1].1).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let dc = runs[0].2.iter().zip(&runs[1].2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("native vs general: max logit diff {d:.2e}, max cell diff {dc:.2e}");
    // The porting guide's bar: cosine above 0.999 and the same top choice.
    let v = runs[0].1.len() / 24;
    for step in 0..24 {
        let (a, b) = (&runs[0].1[step * v..(step + 1) * v], &runs[1].1[step * v..(step + 1) * v]);
        let dotp: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dotp / (na * nb);
        assert!(cos > 0.999, "step {step}: cosine {cos}");
    }
}

#[test]
fn native_chunks_track_general() {
    let path = root().join("models/needle3.cact");
    if !path.exists() {
        return;
    }
    let mut loaded = needle_engine::loader::load(&path).unwrap();
    let ids: Vec<u32> = std::iter::once(needle_core::tokenizer::BOS_ID)
        .chain(loaded.tokenizer.encode("<|im_start|>user\n<tools>[{\"name\":\"set_lights\",\"parameters\":{\"type\":\"object\",\"properties\":{\"room\":{\"type\":\"string\"},\"on\":{\"type\":\"boolean\"}}}}]</tools>\nturn the kitchen lights on and dim the den to 20 percent<|im_end|>\n<|im_start|>assistant\n"))
        .collect();
    let mut outs = vec![];
    for fast in [false, true] {
        // `quant` routes packed weights through the general forward pass.
        loaded.model.numerics.quant = !fast;
        let m = &loaded.model;
        let mut s = m.session();
        s.cells = Some(vec![]);
        let mut last = vec![];
        for chunk in ids.chunks(12) {
            last = m.forward(&mut s, chunk, Outputs::LastLogits).data;
        }
        let mut seq = vec![];
        for _ in 0..16 {
            let next = needle_engine::generate::argmax(&last) as u32;
            seq.push(next);
            last = m.forward(&mut s, &[next], Outputs::LastLogits).data;
        }
        outs.push((last, seq, s.cells.unwrap()));
    }
    let (a, b) = (&outs[0].0, &outs[1].0);
    let dotp: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let cos = dotp / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt());
    eprintln!("chunked fused vs general: cosine {cos:.6}");
    assert_eq!(outs[0].1, outs[1].1, "greedy tokens differ");
    assert!(cos > 0.999);
    assert_eq!(outs[0].2.len(), outs[1].2.len());
}

/// Debug: the confidence logit of the token ids in `$NEEDLE_IDS` (a JSON
/// array) through the checkpoint and the archive, fast and f32.
#[test]
#[ignore]
fn confidence_on_ids() {
    let Some(p) = std::env::var_os("NEEDLE_IDS") else { return };
    let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let (params, config, _) = load_checkpoint(&root().join("models/needle3.safetensors")).unwrap();
    let m = Model::new(Weights::from_checkpoint(&params, &config).unwrap());
    let cells = needle_engine::heads::cells(&m, &ids);
    eprintln!("checkpoint f32: {:.4}", needle_engine::heads::confidence_logit(&m, &cells, ids.len()).unwrap());
    let a = needle_core::cact::Archive::open(&root().join("models/needle3.cact")).unwrap();
    for fast in [false, true] {
        let (w, q) = Weights::from_archive_mode(&a, fast).unwrap();
        let m = match q {
            Some(q) => Model::quantized(w, q),
            None => Model::new(w),
        };
        for n in [60, 100, 127, 128, 140, 160, ids.len()] {
            let n = n.min(ids.len());
            let cells = needle_engine::heads::cells(&m, &ids[..n]);
            eprintln!("archive fast={fast} n={n}: {:.4}", needle_engine::heads::confidence_logit(&m, &cells, n).unwrap());
        }
    }
}

/// Debug: the archive's confidence head against the checkpoint's.
#[test]
#[ignore]
fn head_tensors_match() {
    let (params, config, _) = load_checkpoint(&root().join("models/needle3.safetensors")).unwrap();
    let c = Weights::from_checkpoint(&params, &config).unwrap();
    let a = needle_core::cact::Archive::open(&root().join("models/needle3.cact")).unwrap();
    let (w, _) = Weights::from_archive_mode(&a, false).unwrap();
    for (name, x, y) in [("conf", c.confidence.as_ref().unwrap(), w.confidence.as_ref().unwrap())] {
        for (part, u, v) in [
            ("probes", &x.probes, &y.probes),
            ("gain", &x.gain, &y.gain),
            ("query", &x.query, &y.query),
            ("row_bias", &x.row_bias, &y.row_bias),
            ("proj", &x.proj, &y.proj),
            ("bias", &x.bias, &y.bias),
        ] {
            let dot: f64 = u.iter().zip(v.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
            let nu: f64 = u.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nv: f64 = v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let md = u.iter().zip(v.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("{name}.{part}: len {} vs {} cos {:.5} maxdiff {md:.4} norms {nu:.3} {nv:.3}", u.len(), v.len(), dot / (nu * nv));
        }
    }
    eprintln!("embedding head: ckpt {} archive {}", c.embedding_head.is_some(), w.embedding_head.is_some());
}

/// Debug: swap the confidence head between checkpoint and archive models.
#[test]
#[ignore]
fn head_swap() {
    let Some(p) = std::env::var_os("NEEDLE_IDS") else { return };
    let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let (params, config, _) = load_checkpoint(&root().join("models/needle3.safetensors")).unwrap();
    let c = Weights::from_checkpoint(&params, &config).unwrap();
    let a = needle_core::cact::Archive::open(&root().join("models/needle3.cact")).unwrap();
    let (w, _) = Weights::from_archive_mode(&a, false).unwrap();
    let mut c2 = c.clone();
    c2.confidence = w.confidence.clone();
    let mut w2 = w.clone();
    w2.confidence = c.confidence.clone();
    for (name, weights) in
        [("ckpt body + ckpt head", c), ("ckpt body + cact head", c2), ("cact body + ckpt head", w2), ("cact body + cact head", w)]
    {
        let m = Model::new(weights);
        let cells = needle_engine::heads::cells(&m, &ids);
        eprintln!("{name}: {:.4}", needle_engine::heads::confidence_logit(&m, &cells, ids.len()).unwrap());
    }
}

/// Debug: per-depth cosine between checkpoint-body and archive-body cells.
#[test]
#[ignore]
fn cells_by_depth_archive() {
    let Some(p) = std::env::var_os("NEEDLE_IDS") else { return };
    let ids: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let (params, config, _) = load_checkpoint(&root().join("models/needle3.safetensors")).unwrap();
    let c = Model::new(Weights::from_checkpoint(&params, &config).unwrap());
    let a = needle_core::cact::Archive::open(&root().join("models/needle3.cact")).unwrap();
    let (w, _) = Weights::from_archive_mode(&a, false).unwrap();
    let w = Model::new(w);
    let (x, y) = (needle_engine::heads::cells(&c, &ids), needle_engine::heads::cells(&w, &ids));
    if let Some(dir) = std::env::var_os("NEEDLE_DUMP") {
        let dir = PathBuf::from(dir);
        std::fs::write(dir.join("rcells_ckpt.f32"), x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        std::fs::write(dir.join("rcells_cact.f32"), y.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let (l1, d) = (config.num_layers + 1, config.d_model);
    let mut line = String::new();
    for l in 0..l1 {
        let (mut dot, mut nx, mut ny) = (0f64, 0f64, 0f64);
        for t in 0..ids.len() {
            let o = (t * l1 + l) * d;
            for i in 0..d {
                let (u, v) = (x[o + i] as f64, y[o + i] as f64);
                dot += u * v;
                nx += u * u;
                ny += v * v;
            }
        }
        line += &format!("L{l}:{:.4} ", dot / (nx.sqrt() * ny.sqrt()));
    }
    eprintln!("{line}");
}

/// Debug: the record table of an archive (`$NEEDLE_CACT`).
#[test]
#[ignore]
fn archive_records() {
    let Some(p) = std::env::var_os("NEEDLE_CACT") else { return };
    let a = needle_core::cact::Archive::open(std::path::Path::new(&p)).unwrap();
    for (i, r) in a.records.iter().enumerate() {
        eprintln!("{i:4} {:?} {:?} bits {} group {} bytes {}", r.dtype, r.shape, r.bits, r.group, r.nbytes);
    }
}

/// Debug: what a loaded fast-path model keeps in memory.
#[test]
#[ignore]
fn memory_report() {
    let a = needle_core::cact::Archive::open(&root().join("models/needle3.cact")).unwrap();
    let (w, q) = Weights::from_archive_mode(&a, true).unwrap();
    let mb = |n: usize| n as f64 * 4.0 / 1e6;
    let mut per = std::collections::BTreeMap::<&str, usize>::new();
    for l in &w.layers {
        for (k, v) in [
            ("wq", &l.wq),
            ("wk", &l.wk),
            ("wv", &l.wv),
            ("wgate", &l.wgate),
            ("wout", &l.wout),
            ("phi", &l.phi),
            ("taps", &l.q_taps),
            ("cond_u", &l.cond_u),
            ("cond_v", &l.cond_v),
            ("d*", &l.d1),
        ] {
            *per.entry(k).or_default() += v.len();
        }
        *per.entry("kron").or_default() += l.kron.iter().map(Vec::len).sum::<usize>();
    }
    for e in &w.engrams {
        *per.entry("engram tables f32").or_default() += e.tables.len();
        *per.entry("engram wk/wv f32").or_default() += e.wk.len() + e.wv.len();
    }
    *per.entry("embedding f32").or_default() += w.embedding.len();
    for (k, v) in &per {
        eprintln!("{k:20} {:.1} MB", mb(*v));
    }
    let q = q.unwrap();
    eprintln!("packed total {:.1} MB, phi_f32 {:.1} MB", q.bytes() as f64 / 1e6, mb(q.phi_f32.iter().map(Vec::len).sum()));
}
