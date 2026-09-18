//! Global loaded timsdata.so handle, shared by `io/timsdata.rs` APIs.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};

static TSD_LIB: OnceLock<libloading::Library> = OnceLock::new();

/// Load and memoize the timsdata library.
pub fn load() -> Result<()> {
    if TSD_LIB.get().is_some() {
        return Ok(());
    }
    let lib_path = find_path()?;
    let lib_path = find_path()?;
    let lib = unsafe {
        libloading::Library::new(&lib_path)
            .with_context(|| format!("Cannot load {}", lib_path.display()))?
    };
    let _ = TSD_LIB.set(lib);
    Ok(())
}

/// Get a function pointer from the library.
pub unsafe fn get_fn<T>(name: &[u8]) -> Result<libloading::Symbol<'static, T>> {
    let lib = TSD_LIB.get().context("timsdata.so not loaded, call load() first")?;
    Ok(lib.get(name)?)
}

fn find_path() -> Result<std::path::PathBuf> {
    let candidates = [
        // Original full_dia install
        concat!(env!("CARGO_MANIFEST_DIR"), "/../full_dia/full_dia/alphatims/ext/timsdata.so"),
        // Direct absolute path
        "/home/daichengxin/gitrepo/full_dia/full_dia/alphatims/ext/timsdata.so",
        // System paths
        "/usr/local/lib/timsdata.so",
        "/usr/lib/timsdata.so",
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
    }
    anyhow::bail!("timsdata.so not found (tried {:?})", &candidates[..]);
}
