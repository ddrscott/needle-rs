//! Parity against fixtures written by the Python reference
//! (`tests/oracle/fixtures.py`). Tests skip when a fixture is absent.

use std::path::{Path, PathBuf};

use needle_core::Tokenizer;
use needle_core::cact::{self, Archive};
use needle_core::checkpoint::load_checkpoint;
use needle_core::config::effective_kv_window;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(rel: &str) -> Option<PathBuf> {
    let p = root().join(rel);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skipping: {} not found", p.display());
        None
    }
}

#[test]
fn tokenizer_matches_sentencepiece() {
    let (Some(model), Some(cases)) = (fixture("models/tokenizer.model"), fixture("tests/oracle/tokenizer_cases.json")) else { return };
    let tok = Tokenizer::from_model_file(&model).unwrap();
    let cases: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(cases).unwrap()).unwrap();
    let mut bad = 0;
    for c in &cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text);
        if got != want {
            bad += 1;
            if bad <= 5 {
                eprintln!("encode mismatch for {text:?}\n  want {want:?}\n  got  {got:?}");
            }
        }
        let dec = tok.decode(&want);
        if dec != c["decoded"].as_str().unwrap() {
            bad += 1;
            if bad <= 5 {
                eprintln!("decode mismatch for {text:?}: {dec:?} vs {:?}", c["decoded"]);
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {} tokenizer cases differ", cases.len());
}

#[test]
fn export_matches_python_bytes() {
    let (Some(ckpt), Some(py), Some(model)) =
        (fixture("models/needle3.safetensors"), fixture("models/needle3_py4.cact"), fixture("models/tokenizer.model"))
    else {
        return;
    };
    let (params, config, _) = load_checkpoint(&ckpt).unwrap();
    let tok = Tokenizer::from_model_file(&model).unwrap();
    let blob = tok.to_blob();
    let t = std::time::Instant::now();
    let ours = cact::pack(&params, &config, 4, 128, Some(&blob), effective_kv_window(&config)).unwrap();
    eprintln!("rust export {:?}", t.elapsed());
    let theirs = std::fs::read(py).unwrap();
    assert_eq!(ours.len(), theirs.len());
    let a = Archive::from_bytes(ours.clone()).unwrap();
    let b = Archive::from_bytes(theirs.clone()).unwrap();
    assert_eq!(a.header, b.header);
    assert_eq!(a.codebook, b.codebook);
    assert_eq!(a.records, b.records);
    assert_eq!(a.tokenizer_blob().unwrap(), b.tokenizer_blob().unwrap());
    let diff = ours.iter().zip(&theirs).filter(|(x, y)| x != y).count();
    eprintln!("{diff} of {} bytes differ", ours.len());
    // Per-tensor: every non-CQ tensor must be byte-identical.
    for i in 0..a.records.len() {
        if a.records[i].dtype != cact::Dtype::Cq {
            assert_eq!(a.bytes(i), b.bytes(i), "tensor {i} differs");
        }
    }
    assert!(diff * 100_000 < ours.len(), "too many differing bytes: {diff}");
}

#[test]
fn reads_the_published_archive() {
    let Some(p) = fixture("models/needle3.cact") else { return };
    let a = Archive::open(&p).unwrap();
    assert_eq!(a.header.num_layers, 20);
    assert_eq!(a.header.d_model, 768);
    let tok = Tokenizer::from_blob(a.tokenizer_blob().unwrap()).unwrap();
    assert_eq!(tok.vocab_size(), 8192);
    let bits: std::collections::BTreeMap<u32, usize> =
        a.records.iter().filter(|r| r.dtype == cact::Dtype::Cq).fold(Default::default(), |mut m, r| {
            *m.entry(r.bits).or_default() += 1;
            m
        });
    eprintln!("CQ widths in the published archive: {bits:?}, kv_window {}", a.header.kv_window);
    for i in 0..a.records.len() {
        if a.records[i].dtype != cact::Dtype::Raw {
            a.tensor(i).unwrap();
        }
    }
}
