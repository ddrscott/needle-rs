//! HTTP helpers: Hugging Face Hub downloads (with a local cache) and the
//! error type the networked commands share.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub fn hub_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

fn offline() -> bool {
    std::env::var("HF_HUB_OFFLINE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

fn with_auth(req: ureq::Request) -> ureq::Request {
    match std::env::var("HF_TOKEN") {
        Ok(t) if !t.is_empty() => req.set("Authorization", &format!("Bearer {t}")),
        _ => req,
    }
}

/// The local cache for hub files (`~/.cache/needle-rs/hub/<repo>/<file>`).
pub fn cache_path(repo: &str, file: &str) -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| ".".into());
    home.join(".cache/needle-rs/hub").join(repo).join(file)
}

/// Stream a URL into `dest` (via a temporary file, renamed on success).
pub fn download_url(req: ureq::Request, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let resp = req.call().map_err(|e| anyhow::anyhow!("download failed: {e}"))?;
    let tmp = dest.with_extension("partial");
    let mut out = std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    drop(out);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

/// `hf_hub_download`: a file from a model repo, cached; `force` refetches.
pub fn hub_download(repo: &str, file: &str, force: bool) -> Result<PathBuf> {
    let cached = cache_path(repo, file);
    if cached.exists() && (!force || offline()) {
        return Ok(cached);
    }
    if offline() {
        bail!("{repo}/{file} is not cached and HF_HUB_OFFLINE is set");
    }
    let url = format!("{}/{repo}/resolve/main/{file}", hub_endpoint());
    download_url(with_auth(ureq::get(&url)), &cached).with_context(|| format!("fetch {url}"))?;
    Ok(cached)
}

/// `list_repo_files`.
pub fn hub_list(repo: &str) -> Result<Vec<String>> {
    let url = format!("{}/api/models/{repo}/tree/main?recursive=1", hub_endpoint());
    let items: serde_json::Value = with_auth(ureq::get(&url)).call().map_err(|e| anyhow::anyhow!("list {repo}: {e}"))?.into_json()?;
    Ok(items
        .as_array()
        .into_iter()
        .flatten()
        .filter(|i| i["type"] == "file")
        .filter_map(|i| i["path"].as_str().map(str::to_string))
        .collect())
}

/// Copy `src` to `dest`, creating the directory.
pub fn copy_to(src: &Path, dest: &Path) -> Result<PathBuf> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::copy(src, dest).with_context(|| format!("copy to {}", dest.display()))?;
    Ok(dest.to_path_buf())
}

pub fn megabytes(path: &Path) -> f64 {
    std::fs::metadata(path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0)
}

fn hub_token() -> Result<String> {
    match std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")) {
        Ok(t) if !t.is_empty() => Ok(t),
        _ => {
            let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
            std::fs::read_to_string(home.join(".cache/huggingface/token"))
                .map(|t| t.trim().to_string())
                .map_err(|_| anyhow::anyhow!("set HF_TOKEN (or log in with huggingface-cli) to upload"))
        }
    }
}

/// `HfApi.create_repo(exist_ok=True)` then `upload_file` for one file:
/// preupload check, LFS batch upload for large files, then the commit.
pub fn hub_upload(repo: &str, local: &Path, path_in_repo: &str) -> Result<()> {
    use base64::Engine as _;
    use sha2::Digest;
    let token = hub_token()?;
    let auth = format!("Bearer {token}");
    let ep = hub_endpoint();
    let (org, name) = repo.split_once('/').map_or((None, repo), |(o, n)| (Some(o), n));
    let mut body = serde_json::json!({"type": "model", "name": name});
    if let Some(o) = org {
        body["organization"] = serde_json::json!(o);
    }
    match ureq::post(&format!("{ep}/api/repos/create")).set("Authorization", &auth).send_json(body) {
        Ok(_) | Err(ureq::Error::Status(409, _)) => {}
        Err(e) => bail!("create {repo}: {e}"),
    }
    let bytes = std::fs::read(local)?;
    let sample = base64::engine::general_purpose::STANDARD.encode(&bytes[..bytes.len().min(512)]);
    let pre: serde_json::Value = ureq::post(&format!("{ep}/api/models/{repo}/preupload/main"))
        .set("Authorization", &auth)
        .send_json(serde_json::json!({"files": [{"path": path_in_repo, "size": bytes.len(), "sample": sample}]}))
        .map_err(|e| anyhow::anyhow!("preupload: {e}"))?
        .into_json()?;
    let lfs = pre["files"][0]["uploadMode"] == "lfs";
    let op = if lfs {
        let oid = format!("{:x}", sha2::Sha256::digest(&bytes));
        let batch: serde_json::Value = ureq::post(&format!("{ep}/{repo}.git/info/lfs/objects/batch"))
            .set("Authorization", &auth)
            .set("Accept", "application/vnd.git-lfs+json")
            .set("Content-Type", "application/vnd.git-lfs+json")
            .send_string(&serde_json::json!({"operation": "upload", "transfers": ["basic"], "objects": [{"oid": oid, "size": bytes.len()}], "hash_algo": "sha256"}).to_string())
            .map_err(|e| anyhow::anyhow!("lfs batch: {e}"))?
            .into_json()?;
        let actions = &batch["objects"][0]["actions"];
        if let Some(href) = actions["upload"]["href"].as_str() {
            let mut put = ureq::put(href);
            if let Some(h) = actions["upload"]["header"].as_object() {
                for (k, v) in h {
                    put = put.set(k, v.as_str().unwrap_or(""));
                }
            }
            put.send_bytes(&bytes).map_err(|e| anyhow::anyhow!("lfs upload: {e}"))?;
            if let Some(verify) = actions["verify"]["href"].as_str() {
                ureq::post(verify)
                    .set("Authorization", &auth)
                    .send_json(serde_json::json!({"oid": oid, "size": bytes.len()}))
                    .map_err(|e| anyhow::anyhow!("lfs verify: {e}"))?;
            }
        }
        serde_json::json!({"key": "lfsFile", "value": {"path": path_in_repo, "algo": "sha256", "oid": oid}})
    } else {
        serde_json::json!({"key": "file", "value": {"content": base64::engine::general_purpose::STANDARD.encode(&bytes), "path": path_in_repo, "encoding": "base64"}})
    };
    let header =
        serde_json::json!({"key": "header", "value": {"summary": format!("Upload {path_in_repo} with needle-rs"), "description": ""}});
    ureq::post(&format!("{ep}/api/models/{repo}/commit/main"))
        .set("Authorization", &auth)
        .set("Content-Type", "application/x-ndjson")
        .send_string(&format!("{header}\n{op}\n"))
        .map_err(|e| anyhow::anyhow!("commit: {e}"))?;
    Ok(())
}
