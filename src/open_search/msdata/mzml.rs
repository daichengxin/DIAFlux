//! MsData builder from MzML data — maps mzML spectra to the internal MsData format
//! used throughout the pipeline, with IMS set to constant values.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};

use crate::io::mzml::{self, MzMLData};
use crate::model::*;

// ===========================================================================
// Entry point
// ===========================================================================

pub fn load_ms(path: &Path) -> Result<MsData> {
    let mzml_data = mzml::read_mzml(path)?;

    if mzml_data.ms1_indices.is_empty() {
        bail!("mzML file has no MS1 spectra — cannot determine cycles");
    }
    if mzml_data.ms2_indices.is_empty() {
        bail!("mzML file has no MS2 spectra — DIA data expected");
    }

    let has_ims = false;
    let constant_im: f32 = 1.0; // token IM value when none is available
    let im_gap: f32 = 0.001;

    // ── Determine SWATH boundaries ─────────────────────────────────
    let (swath_boundaries, swath_windows) = if !mzml_data.isolation_windows.is_empty() {
        let windows = mzml_data.isolation_windows.clone();
        let mut boundaries: Vec<f32> = windows.iter().map(|w| w.0).collect();
        if let Some(last) = windows.last() {
            boundaries.push(last.1);
        }
        boundaries.dedup_by(|a, b| (*a - *b).abs() < 0.1);
        (boundaries, windows)
    } else {
        // Fallback: guess SWATH range from MS2 m/z values
        let mz_min = mzml_data.spectra.iter()
            .filter(|s| s.ms_level == 2)
            .flat_map(|s| s.mz_array.iter().copied())
            .filter(|&v| v > 100.0)
            .fold(f32::MAX, f32::min);
        let mz_max = mzml_data.spectra.iter()
            .filter(|s| s.ms_level == 2)
            .flat_map(|s| s.mz_array.iter().copied())
            .filter(|&v| v < 2000.0)
            .fold(f32::MIN, f32::max);
        let mid = (mz_min + mz_max) / 2.0;
        (vec![200.0, 1200.0], vec![(200.0, 1200.0)])
    };

    let n_swath = if swath_boundaries.len() > 1 {
        swath_boundaries.len() - 1
    } else {
        1
    };

    // ── Build cycles from MS1 indices ──────────────────────────────
    let ms1_indices = &mzml_data.ms1_indices;
    let n_cycles = ms1_indices.len();
    let scan_rts: Vec<f32> = ms1_indices.iter()
        .map(|&i| mzml_data.spectra[i].rt)
        .collect();

    tracing::info!(
        "mzML cycles: {} cycles (from {} MS1 spectra), {} SWATH windows, RT [{:.2}s..{:.2}s]",
        n_cycles, ms1_indices.len(), n_swath,
        scan_rts.first().copied().unwrap_or(0.0),
        scan_rts.last().copied().unwrap_or(0.0),
    );

    // ── Assign MS2 spectra to (cycle, swath_id) ────────────────────
    // For each MS2 spectrum, determine which cycle and SWATH it belongs to.
    // Format: ms2_assignments[cycle_idx] = Vec<(swath_id, &Spectrum)>
    let total_est_peaks: usize = mzml_data.spectra.iter()
        .filter(|s| s.ms_level == 2)
        .map(|s| s.mz_array.len())
        .sum();
    let est_per_swath = (total_est_peaks / n_swath).max(1000) / n_cycles.max(1) + 100;

    let mut ms2_maps: HashMap<u8, SwathMsMap> = HashMap::new();
    for swath_id in 1..=n_swath as u8 {
        ms2_maps.insert(swath_id, SwathMsMap {
            scan_rts: scan_rts.clone(),
            cycle_valid_lens: vec![0u32; n_cycles],
            im_values: Vec::with_capacity(est_per_swath * n_cycles),
            mz_values: Vec::with_capacity(est_per_swath * n_cycles),
            height_values: Vec::with_capacity(est_per_swath * n_cycles),
            cycle_valid_lens_cen: Vec::new(),
            im_values_cen: Vec::new(),
            mz_values_cen: Vec::new(),
            height_values_cen: Vec::new(),
            seek_indices: vec![0i64; n_cycles + 1],
            seek_indices_cen: Vec::new(),
        });
    }

    // Iterate through spectra, group MS2 by (cycle, swath)
    for &ms2_idx in &mzml_data.ms2_indices {
        let sp = &mzml_data.spectra[ms2_idx];
        // Find which cycle this MS2 belongs to (RT between MS1 times)
        let cycle_idx = match scan_rts.binary_search_by(|&rt| rt.partial_cmp(&sp.rt).unwrap()) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 { 0 }
                else if i >= n_cycles { n_cycles - 1 }
                else { i - 1 }
            }
        };

        // Determine SWATH ID from isolation window
        let (iso_low, iso_high) = sp.isolation_window;
        let swath_id = if iso_high > iso_low {
            let center = (iso_low + iso_high) / 2.0;
            // Find which SWATH boundary this center falls into
            match swath_boundaries.binary_search_by(|&b| b.partial_cmp(&center).unwrap()) {
                Ok(i) => {
                    if i == 0 { 1 }
                    else if i as usize >= n_swath { n_swath as u8 }
                    else { i as u8 }
                }
                Err(i) => {
                    if i == 0 { 1 }
                    else if i as usize > n_swath { n_swath as u8 }
                    else { i as u8 }
                }
            }
        } else {
            // No isolation window — assign to first SWATH
            1u8
        };

        // Add peaks to the corresponding SWATH map
        if let Some(map) = ms2_maps.get_mut(&swath_id) {
            let n_peaks = sp.mz_array.len().min(sp.intensity_array.len());
            for pi in 0..n_peaks {
                let mz = sp.mz_array[pi];
                let inten = sp.intensity_array[pi];
                // Check for valid values
                if mz > 0.0 && inten >= 0.0 {
                    map.im_values.push(constant_im);
                    map.mz_values.push(mz);
                    map.height_values.push(inten as u32);
                    map.cycle_valid_lens[cycle_idx] += 1;
                }
            }
        }
    }

    // Build seek indices + sort by m/z + populate centroided copies
    for map in ms2_maps.values_mut() {
        sort_map_by_mz_inplace(map, n_cycles);
        let mut cum = 0i64;
        for c in 0..n_cycles {
            map.seek_indices[c] = cum;
            cum += map.cycle_valid_lens[c] as i64;
        }
        map.seek_indices[n_cycles] = cum;
        // Centroided = profile (no centroiding step for mzML)
        map.cycle_valid_lens_cen = map.cycle_valid_lens.clone();
        map.im_values_cen = map.im_values.clone();
        map.mz_values_cen = map.mz_values.clone();
        map.height_values_cen = map.height_values.clone();
        map.seek_indices_cen = map.seek_indices.clone();
    }

    // ── Build MS1 map ─────────────────────────────────────────────
    // Collect all MS1 peaks and split by SWATH
    let mut ms1_full_map = SwathMsMap {
        scan_rts: scan_rts.clone(),
        cycle_valid_lens: vec![0u32; n_cycles],
        im_values: Vec::with_capacity(total_est_peaks / 2),
        mz_values: Vec::with_capacity(total_est_peaks / 2),
        height_values: Vec::with_capacity(total_est_peaks / 2),
        cycle_valid_lens_cen: Vec::new(),
        im_values_cen: Vec::new(),
        mz_values_cen: Vec::new(),
        height_values_cen: Vec::new(),
        seek_indices: vec![0i64; n_cycles + 1],
        seek_indices_cen: Vec::new(),
    };

    for cycle_idx in 0..n_cycles {
        let sp_idx = ms1_indices[cycle_idx];
        let sp = &mzml_data.spectra[sp_idx];
        let n_peaks = sp.mz_array.len().min(sp.intensity_array.len());
        for pi in 0..n_peaks {
            let mz = sp.mz_array[pi];
            let inten = sp.intensity_array[pi];
            if mz > 0.0 && inten >= 0.0 {
                ms1_full_map.im_values.push(constant_im);
                ms1_full_map.mz_values.push(mz);
                ms1_full_map.height_values.push(inten as u32);
                ms1_full_map.cycle_valid_lens[cycle_idx] += 1;
            }
        }
    }

    sort_map_by_mz_inplace(&mut ms1_full_map, n_cycles);
    let mut cum = 0i64;
    for c in 0..n_cycles {
        ms1_full_map.seek_indices[c] = cum;
        cum += ms1_full_map.cycle_valid_lens[c] as i64;
    }
    ms1_full_map.seek_indices[n_cycles] = cum;
    ms1_full_map.cycle_valid_lens_cen = ms1_full_map.cycle_valid_lens.clone();
    ms1_full_map.im_values_cen = ms1_full_map.im_values.clone();
    ms1_full_map.mz_values_cen = ms1_full_map.mz_values.clone();
    ms1_full_map.height_values_cen = ms1_full_map.height_values.clone();
    ms1_full_map.seek_indices_cen = ms1_full_map.seek_indices.clone();

    let ms1_maps = split_ms1_by_swath(&ms1_full_map, &swath_boundaries, &scan_rts);

    // ── Build DiaWindows ─────────────────────────────────────────
    let dia_windows: Vec<DiaWindow> = swath_windows.iter()
        .enumerate()
        .map(|(i, &(mz_low, mz_high))| DiaWindow {
            swath_id: (i + 1) as u8,
            mz_low,
            mz_high,
            im_low: 0.0,
            im_high: 2.0,
        })
        .collect();

    // ── Cycle frame ranges ────────────────────────────────────────
    let mut cycle_frame_ranges = Vec::with_capacity(n_cycles);
    for c in 0..n_cycles - 1 {
        cycle_frame_ranges.push((ms1_indices[c], ms1_indices[c + 1]));
    }
    if !ms1_indices.is_empty() {
        cycle_frame_ranges.push((ms1_indices[n_cycles - 1], mzml_data.spectra.len()));
    }

    let cycle_time = if scan_rts.len() > 1 {
        (scan_rts.last().unwrap() - scan_rts.first().unwrap())
            / (scan_rts.len() - 1) as f32
    } else {
        1.0
    };

    let device_name = format!("mzML-{}", path.file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "unknown".into()));

    Ok(MsData {
        scan_rts: scan_rts.clone(),
        cycle_time,
        n_cycles: n_cycles as u32,
        ms1_maps,
        ms2_maps,
        swath_boundaries,
        dia_windows,
        frames_per_cycle: 1,
        device_name,
        im_gap,
        dir_d: path.to_string_lossy().to_string(),
        push_indptr: Vec::new(),
        tof_indices: Vec::new(),
        intensity_values: Vec::new(),
        frame_info: Vec::new(),
        mz_lookup: Vec::new(),
        mobility_values: vec![constant_im],
        scan_max_index: 1,
        cycle_frame_ranges,
        has_ims,
    })
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Sort each cycle's peaks by m/z in-place.
fn sort_map_by_mz_inplace(map: &mut SwathMsMap, n_cycles: usize) {
    if map.mz_values.is_empty() {
        return;
    }
    let n = map.mz_values.len();
    let mut cum: usize = 0;
    for c in 0..n_cycles.min(map.cycle_valid_lens.len()) {
        let start = cum;
        let len = map.cycle_valid_lens[c] as usize;
        cum += len;
        if len < 2 { continue; }
        let end = start + len;
        if end > n { break; }

        let mut order: Vec<usize> = (start..end).collect();
        order.sort_unstable_by(|&a, &b| {
            map.mz_values[a].partial_cmp(&map.mz_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_mz: Vec<f32> = order.iter().map(|&i| map.mz_values[i]).collect();
        let sorted_im: Vec<f32> = order.iter().map(|&i| map.im_values[i]).collect();
        let sorted_h: Vec<u32> = order.iter().map(|&i| map.height_values[i]).collect();

        for j in 0..len {
            map.mz_values[start + j] = sorted_mz[j];
            map.im_values[start + j] = sorted_im[j];
            map.height_values[start + j] = sorted_h[j];
        }
    }
}

/// Split MS1 full-range map into per-SWATH chunks.
fn split_ms1_by_swath(
    ms1_full: &SwathMsMap,
    swath: &[f32],
    scan_rts: &[f32],
) -> HashMap<u8, SwathMsMap> {
    let mut result = HashMap::new();
    let mass_neutron = 1.0033548378;
    let n_cycles = scan_rts.len();

    for swath_i in 1..swath.len() {
        let swath_id = swath_i as u8;
        result.insert(swath_id, SwathMsMap {
            scan_rts: scan_rts.to_vec(),
            cycle_valid_lens: vec![0u32; n_cycles],
            im_values: Vec::new(),
            mz_values: Vec::new(),
            height_values: Vec::new(),
            cycle_valid_lens_cen: Vec::new(),
            im_values_cen: Vec::new(),
            mz_values_cen: Vec::new(),
            height_values_cen: Vec::new(),
            seek_indices: vec![0i64; n_cycles + 1],
            seek_indices_cen: Vec::new(),
        });
    }

    let n = ms1_full.mz_values.len();
    if n == 0 { return result; }

    let mut cum: usize = 0;
    for c in 0..n_cycles.min(ms1_full.cycle_valid_lens.len()) {
        let start = cum;
        let len = ms1_full.cycle_valid_lens[c] as usize;
        cum += len;
        if len == 0 { continue; }
        let end = start + len;
        if end > n { break; }

        for pi in start..end {
            let mz = ms1_full.mz_values[pi];
            for swath_i in 1..swath.len() {
                let swath_id = swath_i as u8;
                let mz_low = swath[swath_i - 1] - 3.0 * mass_neutron;
                let mz_high = swath[swath_i] + 3.0 * mass_neutron;
                if mz >= mz_low && mz <= mz_high {
                    if let Some(map) = result.get_mut(&swath_id) {
                        map.im_values.push(ms1_full.im_values[pi]);
                        map.mz_values.push(mz);
                        map.height_values.push(ms1_full.height_values[pi]);
                        map.cycle_valid_lens[c] += 1;
                    }
                    break;
                }
            }
        }
    }

    for (_, map) in result.iter_mut() {
        sort_map_by_mz_inplace(map, n_cycles);
        let mut cum_sum = 0i64;
        for c in 0..n_cycles {
            map.seek_indices[c] = cum_sum;
            cum_sum += map.cycle_valid_lens[c] as i64;
        }
        map.seek_indices[n_cycles] = cum_sum;
        map.cycle_valid_lens_cen = map.cycle_valid_lens.clone();
        map.im_values_cen = map.im_values.clone();
        map.mz_values_cen = map.mz_values.clone();
        map.height_values_cen = map.height_values.clone();
        map.seek_indices_cen = map.seek_indices.clone();
    }

    result
}
