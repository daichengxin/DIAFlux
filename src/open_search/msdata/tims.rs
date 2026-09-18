//! MS data construction from Bruker .d directory — lightweight maps.
//!
//! Instead of building fully sorted per-SWATH maps (expensive for large datasets),
//! we build lightweight per-cycle maps without sorting. XIC extraction does
//! a simple linear scan over each cycle's peaks, which is fast enough for
//! typical cycle sizes (~1000 peaks per cycle).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};

use crate::io::bruker::{self, BrukerData, Frame};
use crate::model::*;

pub use crate::model::MsData;

// ===========================================================================
// Entry point
// ===========================================================================

pub fn load_ms(ws: &Path) -> Result<MsData> {
    let bruker = bruker::read_bruker_dir(ws)?;

    let device_name = bruker
        .sql_metadata
        .get("InstrumentName")
        .cloned()
        .unwrap_or_default();

    let gradient = bruker.rt_values.last().copied().unwrap_or(0.0) / 60.0;
    tracing::info!("device_name: {}, gradient: {:.2}min", device_name, gradient);

    let cycle_time = if bruker.rt_values.len() > 1 {
        (bruker.rt_values.last().unwrap() - bruker.rt_values.first().unwrap())
            / (bruker.rt_values.len() - 1) as f32
    } else {
        1.0
    };

    // MS1 frame indices (MsMsType == 0)
    let ms1_indices: Vec<usize> = bruker
        .frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.msms_type == 0)
        .map(|(i, _)| i)
        .collect();

    if ms1_indices.len() < 2 {
        bail!("Need at least 2 MS1 frames, found {}", ms1_indices.len());
    }

    let frames_per_cycle = (ms1_indices[1] - ms1_indices[0]) as u32;

    // Build SWATH boundaries
    let (swath_boundaries, swath_windows) = if !bruker.quad_mz_values.is_empty()
        && bruker.quad_mz_values[0] != (0.0, 0.0)
    {
        let windows: Vec<(f32, f32)> = bruker.quad_mz_values.clone();
        let mut sorted = windows.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut boundaries: Vec<f32> = sorted.iter().map(|w| w.0).collect();
        if let Some(last) = sorted.last() {
            boundaries.push(last.1);
        }
        boundaries.dedup_by(|a, b| (*a - *b).abs() < 0.1);
        (boundaries, sorted)
    } else {
        (vec![200.0, 1200.0], vec![(200.0, 1200.0)])
    };

    tracing::info!("SWATH: {} windows, boundaries: [{:.0}..{:.0}]",
        swath_windows.len(),
        swath_boundaries.first().copied().unwrap_or(0.0),
        swath_boundaries.last().copied().unwrap_or(0.0)
    );

    let im_gap = compute_im_gap(&bruker);

    // ── Try timsdata.so for correct m/z and IM calibration ─────────
    // For now, timsdata integration is optional. The quadratic calibration
    // from MzCalibration metadata provides sufficient accuracy.
    // timsdata integration: search for `timsdata` in codebase

    // ── Build per-SWATH maps (single-pass, no sorting) ─────────────────
    let n_cycles = ms1_indices.len();
    let n_swath = if swath_boundaries.len() > 1 { swath_boundaries.len() - 1 } else { 1 };

    // Scan RTs from MS1 frame times
    let scan_rts: Vec<f32> = ms1_indices
        .iter()
        .map(|&i| if i < bruker.rt_values.len() { bruker.rt_values[i] } else { 0.0 })
        .collect();

    let (ms1_maps, ms2_maps) = build_maps_fast(
        &bruker, &swath_boundaries, &ms1_indices, &scan_rts, n_swath,
    )?;

    tracing::info!(
        "Built {} MS1 + {} MS2 maps for {} cycles ({} peaks total)",
        ms1_maps.len(), ms2_maps.len(), n_cycles,
        bruker.tof_indices.len(),
    );

    // Cycle frame ranges for binary access
    let mut cycle_frame_ranges = Vec::with_capacity(n_cycles);
    for c in 0..n_cycles - 1 {
        cycle_frame_ranges.push((ms1_indices[c], ms1_indices[c + 1]));
    }
    if !ms1_indices.is_empty() {
        cycle_frame_ranges.push((ms1_indices[n_cycles - 1], bruker.frames.len()));
    }

    let dia_windows: Vec<DiaWindow> = swath_windows
        .iter()
        .enumerate()
        .map(|(i, &(mz_low, mz_high))| DiaWindow {
            swath_id: (i + 1) as u8,
            mz_low,
            mz_high,
            im_low: bruker.mobility_min_value,
            im_high: bruker.mobility_max_value,
        })
        .collect();

    Ok(MsData {
        scan_rts: scan_rts.clone(),
        cycle_time,
        n_cycles: n_cycles as u32,
        ms1_maps,
        ms2_maps,
        swath_boundaries,
        dia_windows,
        frames_per_cycle,
        device_name,
        im_gap,
        dir_d: ws.to_string_lossy().to_string(),
        push_indptr: bruker.push_indptr,
        tof_indices: bruker.tof_indices,
        intensity_values: bruker.intensity_values,
        frame_info: Vec::new(),
        mz_lookup: bruker.mz_values,
        mobility_values: bruker.mobility_values,
        scan_max_index: bruker.scan_max_index,
        cycle_frame_ranges,
        has_ims: true,
    })
}

// ===========================================================================
// Single-pass map builder
// ===========================================================================

/// Single-pass map builder that tracks per-cycle peak counts.
fn build_maps_fast(
    bruker: &BrukerData,
    swath: &[f32],
    ms1_indices: &[usize],
    scan_rts: &[f32],
    n_swath: usize,
) -> Result<(HashMap<u8, SwathMsMap>, HashMap<u8, SwathMsMap>)> {
    let n_cycles = ms1_indices.len();
    let _scan_max = bruker.scan_max_index as usize;
    let total_peaks = bruker.tof_indices.len();

    // Estimate: peaks distribute across SWATHs roughly evenly
    let est_per_swath = (total_peaks / n_swath).max(1000);

    let mut ms2_maps: HashMap<u8, SwathMsMap> = HashMap::new();
    for swath_id in 1..=n_swath as u8 {
        ms2_maps.insert(swath_id, SwathMsMap {
            scan_rts: scan_rts.to_vec(),
            cycle_valid_lens: vec![0u32; n_cycles],
            im_values: Vec::with_capacity(est_per_swath),
            mz_values: Vec::with_capacity(est_per_swath),
            height_values: Vec::with_capacity(est_per_swath),
            cycle_valid_lens_cen: Vec::new(),
            im_values_cen: Vec::new(),
            mz_values_cen: Vec::new(),
            height_values_cen: Vec::new(),
            seek_indices: vec![0i64; n_cycles + 1],
            seek_indices_cen: Vec::new(),
        });
    }

    // MS1 full map tracks peaks per MS1 frame
    let mut ms1_full_map = SwathMsMap {
        scan_rts: scan_rts.to_vec(),
        cycle_valid_lens: vec![0u32; n_cycles],
        im_values: Vec::with_capacity(total_peaks / 2),
        mz_values: Vec::with_capacity(total_peaks / 2),
        height_values: Vec::with_capacity(total_peaks / 2),
        cycle_valid_lens_cen: Vec::new(),
        im_values_cen: Vec::new(),
        mz_values_cen: Vec::new(),
        height_values_cen: Vec::new(),
        seek_indices: vec![0i64; n_cycles + 1],
        seek_indices_cen: Vec::new(),
    };

    // Single pass: process each cycle's frames
    // Build frame-level cumulative peak index
    let mut frame_peak_offsets = vec![0usize; bruker.frames.len() + 1];
    for fi in 0..bruker.frames.len() {
        frame_peak_offsets[fi + 1] = frame_peak_offsets[fi] + bruker.frames[fi].num_peaks as usize;
    }

    for cycle_idx in 0..n_cycles - 1 {
        let cycle_start = ms1_indices[cycle_idx];
        let cycle_end = ms1_indices[cycle_idx + 1];

        for fi in cycle_start..cycle_end {
            let frame = &bruker.frames[fi];
            if frame.num_peaks == 0 {
                continue;
            }

            let is_ms1 = frame.msms_type == 0;

            // Use frame-level cumulative peak index for boundaries.
            // Since scan-level index encoding varies (Bruker format versions differ),
            // assign IM values linearly from mobility_values based on position.
            if fi + 1 >= frame_peak_offsets.len() || frame_peak_offsets[fi] >= bruker.tof_indices.len() { continue; }
            let frame_peak_start = frame_peak_offsets[fi];
            let frame_peak_count = frame.num_peaks as usize;
            let frame_peak_end = (frame_peak_start + frame_peak_count).min(bruker.tof_indices.len());
            let n_scans = frame.num_scans.max(1) as usize;

            if is_ms1 {
                // MS1: all peaks go to the full MS1 map
                let mut peak_count = 0usize;
                for pi in frame_peak_start..frame_peak_end.min(bruker.tof_indices.len()).min(bruker.intensity_values.len()) {
                    if pi >= bruker.tof_indices.len() || pi >= bruker.intensity_values.len() { break; }
                    // Assign IM based on position: evenly distribute peaks across scans
                    let scan_i = ((peak_count * n_scans) / frame_peak_count.max(1)).min(n_scans - 1);
                    let im_val = if scan_i < bruker.mobility_values.len() {
                        bruker.mobility_values[scan_i]
                    } else { scan_i as f32 * 0.001 };
                    let tof = bruker.tof_indices[pi] as usize;
                    let mz = if tof < bruker.mz_values.len() { bruker.mz_values[tof] } else { tof as f32 * 0.001 + 100.0 };
                    ms1_full_map.im_values.push(im_val);
                    ms1_full_map.mz_values.push(mz);
                    ms1_full_map.height_values.push(bruker.intensity_values[pi]);
                    peak_count += 1;
                }
                ms1_full_map.cycle_valid_lens[cycle_idx] = peak_count as u32;
            } else {
                // MS2: SWATH ID from frame position in cycle
                let offset = fi - cycle_start;
                if offset == 0 { continue; }
                let swath_id = (offset as u8) % (n_swath as u8);
                let swath_id = if swath_id == 0 { n_swath as u8 } else { swath_id };

                if let Some(map) = ms2_maps.get_mut(&swath_id) {
                    let mut peak_count = 0usize;
                    for pi in frame_peak_start..frame_peak_end.min(bruker.tof_indices.len()).min(bruker.intensity_values.len()) {
                        if pi >= bruker.tof_indices.len() || pi >= bruker.intensity_values.len() { break; }
                        let scan_i = ((peak_count * n_scans) / frame_peak_count.max(1)).min(n_scans - 1);
                        let im_val = if scan_i < bruker.mobility_values.len() {
                            bruker.mobility_values[scan_i]
                        } else { scan_i as f32 * 0.001 };
                        let tof = bruker.tof_indices[pi] as usize;
                        let mz = if tof < bruker.mz_values.len() { bruker.mz_values[tof] } else { tof as f32 * 0.001 + 100.0 };
                        map.im_values.push(im_val);
                        map.mz_values.push(mz);
                        map.height_values.push(bruker.intensity_values[pi]);
                        peak_count += 1;
                    }
                    if peak_count > 0 {
                        if cycle_idx < map.cycle_valid_lens.len() {
                            map.cycle_valid_lens[cycle_idx] += peak_count as u32;
                        }
                    }
                }
            }
        }
    }

    // Build seek indices from cycle_valid_lens + sort by m/z + populate centroided
    for map in ms2_maps.values_mut() {
        sort_map_by_mz_inplace(map, n_cycles);
        let mut cum = 0i64;
        for c in 0..n_cycles {
            map.seek_indices[c] = cum;
            cum += map.cycle_valid_lens[c] as i64;
        }
        map.seek_indices[n_cycles] = cum;
        map.cycle_valid_lens_cen = map.cycle_valid_lens.clone();
        map.im_values_cen = map.im_values.clone();
        map.mz_values_cen = map.mz_values.clone();
        map.height_values_cen = map.height_values.clone();
        map.seek_indices_cen = map.seek_indices.clone();
    }
    {
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
    }

    // Split MS1 into per-SWATH, then clean up temp data
    let ms1_maps = split_ms1_fast(&ms1_full_map, swath, scan_rts);

    Ok((ms1_maps, ms2_maps))
}

/// Sort each cycle's peak data by m/z in-place, enabling binary search in XIC extraction.
fn sort_map_by_mz_inplace(map: &mut SwathMsMap, n_cycles: usize) {
    if map.mz_values.is_empty() {
        return;
    }
    let n = map.mz_values.len();
    let mut cum: usize = 0;
    for c in 0..n_cycles.min(map.cycle_valid_lens.len()) {
        let start = cum;
        let len = map.cycle_valid_lens[c] as usize;
        let end = cum + len;
        cum = end;
        if end > n { break; }
        if len < 2 { continue; }

        // Build sorted index order
        let mut order: Vec<usize> = (start..end).collect();
        order.sort_unstable_by(|&a, &b| {
            map.mz_values[a].partial_cmp(&map.mz_values[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract sorted values
        let sorted_mz: Vec<f32> = order.iter().map(|&i| map.mz_values[i]).collect();
        let sorted_im: Vec<f32> = order.iter().map(|&i| map.im_values[i]).collect();
        let sorted_h: Vec<u32> = order.iter().map(|&i| map.height_values[i]).collect();

        // Write back in-place
        for j in 0..len {
            map.mz_values[start + j] = sorted_mz[j];
            map.im_values[start + j] = sorted_im[j];
            map.height_values[start + j] = sorted_h[j];
        }
    }
}

/// Split MS1 full-range map into per-SWATH chunks, with per-cycle boundary tracking.
fn split_ms1_fast(
    ms1_full: &SwathMsMap,
    swath: &[f32],
    scan_rts: &[f32],
) -> HashMap<u8, SwathMsMap> {
    use std::collections::HashMap;
    let mut result = HashMap::new();
    let mass_neutron = 1.0033548378;
    let n_cycles = scan_rts.len();

    // Initialize per-SWATH maps with proper cycle tracking
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

    // Distribute peaks to per-SWATH maps, tracking cycle origin
    let n = ms1_full.mz_values.len();
    if n == 0 || ms1_full.cycle_valid_lens.is_empty() {
        return result;
    }

    let mut cum: usize = 0;
    for c in 0..n_cycles.min(ms1_full.cycle_valid_lens.len()) {
        let start = cum;
        let len = ms1_full.cycle_valid_lens[c] as usize;
        let end = cum + len;
        cum = end;
        if end > n { break; }

        let mz_low_base = swath[0] - 3.0 * mass_neutron;
        let mz_high_base = swath[swath.len() - 1] + 3.0 * mass_neutron;

        for pi in start..end {
            let mz = ms1_full.mz_values[pi];
            // Determine which SWATH this peak belongs to
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

    // Build seek indices and sort for each map
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

// ===========================================================================
// Utility
// ===========================================================================

pub fn compute_im_gap(bruker: &BrukerData) -> f32 {
    let im_min = bruker.mobility_min_value;
    let im_max = bruker.mobility_max_value;
    let im_count = bruker
        .frames
        .iter()
        .map(|f| f.num_scans)
        .max()
        .unwrap_or(1) as f32
        + 1.0;
    if im_count > 1.0 {
        (im_max - im_min) / im_count
    } else {
        0.001
    }
}
