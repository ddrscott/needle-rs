//! `needle.agent.fetch`: the published artifacts (base weights, training
//! checkpoints, per-platform engine folders, the engine library) and where
//! they are cached.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::net::{copy_to, hub_download, hub_list};

pub const PLATFORMS: [&str; 17] = [
    "macos-arm64",
    "linux-x86_64",
    "linux-arm64",
    "linux-armv7",
    "linux-riscv64",
    "linux-mipsel",
    "windows-x86_64",
    "windows-arm64",
    "android-arm64",
    "android-armv7",
    "android-riscv64",
    "ios-arm64",
    "ios-sim-arm64",
    "tvos-arm64",
    "watchos-arm64",
    "wasm",
    "wasm-component",
];
pub const CHECKPOINT_PREFIX: &str = "checkpoints";

pub fn engine_repo(generation: u32) -> Result<&'static str> {
    Ok(match generation {
        2 => "Cactus-Compute/needle2",
        3 => "Cactus-Compute/needle3",
        g => bail!("unsupported Needle generation: {g}"),
    })
}

pub fn engine_version(generation: u32) -> Result<&'static str> {
    Ok(match generation {
        2 => "2.0.4",
        3 => "3.0.2",
        g => bail!("unsupported Needle generation: {g}"),
    })
}

pub fn base_weights(generation: u32) -> Result<&'static str> {
    Ok(match generation {
        2 => "needle2.cact",
        3 => "needle3.cact",
        g => bail!("unsupported Needle generation: {g}"),
    })
}

pub fn cache_dir(generation: u32) -> Result<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| ".".into());
    Ok(home.join(".cache/cactus-needle").join(format!("v{generation}")).join(engine_version(generation)?))
}

/// The download counter the reference pings (best effort).
pub fn register_download(generation: u32) {
    if let Ok(repo) = engine_repo(generation) {
        let _ = crate::net::hub_download(repo, "config.json", true);
    }
}

/// `fetch_weights`: the base `.cact` of a generation, into `dest_dir`
/// (default: next to its engine in the cache).
pub fn fetch_weights(generation: u32, dest_dir: Option<&Path>, force: bool) -> Result<PathBuf> {
    let name = base_weights(generation)?;
    let dir = match dest_dir {
        Some(d) => d.to_path_buf(),
        None => cache_dir(generation)?,
    };
    let out = dir.join(name);
    if out.exists() && !force {
        return Ok(out);
    }
    register_download(generation);
    let cached = hub_download(engine_repo(generation)?, name, force)?;
    copy_to(&cached, &out)
}

/// `fetch_checkpoint`: a training checkpoint (`needle3.safetensors`, ...).
pub fn fetch_checkpoint(name: &str, dest_dir: &Path, generation: u32) -> Result<PathBuf> {
    let repo = engine_repo(generation)?;
    register_download(generation);
    let base = Path::new(name).file_name().and_then(|f| f.to_str()).unwrap_or(name);
    let files = hub_list(repo).unwrap_or_default();
    let remote = [format!("{CHECKPOINT_PREFIX}/{base}"), base.to_string()]
        .into_iter()
        .find(|c| files.is_empty() || files.contains(c))
        .with_context(|| format!("{base} is not published in {repo}"))?;
    let cached = hub_download(repo, &remote, false)?;
    copy_to(&cached, &dest_dir.join(base))
}

/// `download_platform`: every file of a platform folder, runners marked
/// executable.
pub fn download_platform(name: &str, out_dir: &Path, generation: u32, dest: Option<&Path>) -> Result<Vec<PathBuf>> {
    let repo = engine_repo(generation)?;
    register_download(generation);
    let files: Vec<String> = hub_list(repo)?.into_iter().filter(|f| f.starts_with(&format!("{name}/"))).collect();
    if !PLATFORMS.contains(&name) || files.is_empty() {
        bail!("{name} is not a published platform folder in {repo}");
    }
    let dest = dest.map(Path::to_path_buf).unwrap_or_else(|| out_dir.join(name));
    let mut out = vec![];
    for f in files {
        let cached = hub_download(repo, &f, false)?;
        let target = dest.join(Path::new(&f).file_name().unwrap());
        copy_to(&cached, &target)?;
        if matches!(target.file_name().and_then(|n| n.to_str()), Some("needle" | "needle.exe")) {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut p = std::fs::metadata(&target)?.permissions();
                p.set_mode(p.mode() | 0o111);
                std::fs::set_permissions(&target, p)?;
            }
        }
        out.push(target);
    }
    Ok(out)
}

/// The wheel tag of this machine (`_platform_tag`).
pub fn platform_tag() -> String {
    let arm = cfg!(target_arch = "aarch64");
    if cfg!(target_os = "macos") {
        format!("macosx_11_0_{}", if arm { "arm64" } else { "x86_64" })
    } else if cfg!(target_os = "windows") {
        (if arm { "win_arm64" } else { "win_amd64" }).into()
    } else {
        let family = if cfg!(target_env = "musl") { "musllinux_1_2_" } else { "manylinux2014_" };
        format!("{family}{}", if arm { "aarch64" } else { "x86_64" })
    }
}

fn lib_name_for(tag: &str) -> &'static str {
    if tag.starts_with("macosx") {
        "libneedle.dylib"
    } else if tag.starts_with("win") {
        "libneedle.dll"
    } else {
        "libneedle.so"
    }
}

/// `fetch_library`: the native engine from its wheel.
pub fn fetch_library(version: &str, dest_dir: &Path, tag: Option<&str>, generation: u32) -> Result<PathBuf> {
    let tag = tag.map(str::to_string).unwrap_or_else(platform_tag);
    let wheel = format!("cactus_needle-{version}-py3-none-{tag}.whl");
    let repo = engine_repo(generation)?;
    register_download(generation);
    let path = hub_download(repo, &format!("python/{wheel}"), false)?;
    let lib = lib_name_for(&tag);
    let (stem, suffix) = lib.rsplit_once('.').unwrap();
    let member = if generation >= 3 { format!("needle/{stem}{generation}.{suffix}") } else { format!("needle/{lib}") };
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path)?)?;
    let mut entry = archive.by_name(&member).with_context(|| format!("{wheel} has no {member}"))?;
    std::fs::create_dir_all(dest_dir)?;
    let out = dest_dir.join(lib);
    let mut f = std::fs::File::create(&out)?;
    std::io::copy(&mut entry, &mut f)?;
    Ok(out)
}
