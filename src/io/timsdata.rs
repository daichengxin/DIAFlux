//! Bruker timsdata C library FFI — official m/z and IM calibration.
//!
//! Uses `libloading` to dynamically load `timsdata.so` provided by Bruker.
//! Provides correct m/z and 1/K0 (ion mobility) calibration via C API.

use std::ffi::CString;
use std::path::Path;

use anyhow::{bail, Result};

/// Open a Bruker .d directory for reading.
pub fn open(path: &Path) -> Result<u64> {
    unsafe { crate::io::timsdalib::load()?; }
    let c_path = CString::new(path.to_string_lossy().as_bytes())?;
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(*const i8, u32) -> u64> =
            crate::io::timsdalib::get_fn(b"tims_open")?;
        let handle = func(c_path.as_ptr(), 0);
        if handle == 0 {
            bail!("tims_open failed for {:?}", path);
        }
        Ok(handle)
    }
}

/// Close a handle.
pub fn close(handle: u64) -> Result<()> {
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(u64)> =
            crate::io::timsdalib::get_fn(b"tims_close")?;
        func(handle);
    }
    Ok(())
}

/// Convert TOF indices to m/z (C API uses *mut for both params).
pub fn index_to_mz(handle: u64, frame_id: i64, tof_indices: &mut [f64], mz_out: &mut [f64]) -> Result<()> {
    if tof_indices.is_empty() { return Ok(()); }
    let count = tof_indices.len().min(mz_out.len()) as u32;
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(u64, i64, *mut f64, *mut f64, u32) -> u32> =
            crate::io::timsdalib::get_fn(b"tims_index_to_mz")?;
        let rc = func(handle, frame_id, tof_indices.as_mut_ptr(), mz_out.as_mut_ptr(), count);
        if rc != 0 { bail!("tims_index_to_mz error {rc}"); }
    }
    Ok(())
}

/// Convert scan numbers to 1/K0 (C API uses *mut for both params).
pub fn scannum_to_oneoverk0(handle: u64, frame_id: i64, scan_nums: &mut [f64], im_out: &mut [f64]) -> Result<()> {
    if scan_nums.is_empty() { return Ok(()); }
    let count = scan_nums.len().min(im_out.len()) as u32;
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(u64, i64, *mut f64, *mut f64, u32) -> u32> =
            crate::io::timsdalib::get_fn(b"tims_scannum_to_oneoverk0")?;
        let rc = func(handle, frame_id, scan_nums.as_mut_ptr(), im_out.as_mut_ptr(), count);
        if rc != 0 { bail!("tims_scannum_to_oneoverk0 error {rc}"); }
    }
    Ok(())
}

/// Recalibrate m/z lookup table.
pub fn recalibrate_mz(handle: u64, frame_id: i64, mz_values: &mut Vec<f32>) -> Result<()> {
    if mz_values.is_empty() { return Ok(()); }
    let n = mz_values.len().min(500000);
    let mut indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut mz_out = vec![0.0f64; n];
    index_to_mz(handle, frame_id, &mut indices, &mut mz_out)?;
    for i in 0..n { mz_values[i] = mz_out[i] as f32; }
    Ok(())
}

/// Recalibrate mobility (1/K0) values.
pub fn recalibrate_mobility(handle: u64, frame_id: i64, mobility: &mut Vec<f32>) -> Result<()> {
    if mobility.is_empty() { return Ok(()); }
    let n = mobility.len();
    let mut indices: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut im_out = vec![0.0f64; n];
    scannum_to_oneoverk0(handle, frame_id, &mut indices, &mut im_out)?;
    for i in 0..n { mobility[i] = im_out[i] as f32; }
    Ok(())
}
