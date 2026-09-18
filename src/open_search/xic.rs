//! SA scoring and XIC extraction — replaces `full_dia/fxic.py`
//!
//! CPU-optimized implementation.

use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::config::config;
use crate::model::*;

// ===========================================================================
// XIC Extraction (CPU version — GPU version planned)
// ===========================================================================

/// Extract XICs for a batch of precursors from centroid MS data.
///
/// # Arguments
/// * `precursors` — Precursor batch info
/// * `locus` — Cycle index for each precursor (typically 0 for full gradient)
/// * `ms1_map`, `ms2_map` — MS data maps indexed by SWATH
/// * `ppm_tolerance`, `im_tolerance` — Search windows
/// * `rt_tolerance` — RT window (None = full gradient)
/// * `cycle_num` — Number of cycles (None = from rt_tolerance)
/// * `scope` — "center" | "big" | "top6"
/// * `only_xic` — Return only XIC intensities (skip im/mz)
/// * `query_ims` — Optional override for IM centre values.
///   If None, uses `pred_im` (precursors must have it set).
///   If Some(slice), uses the provided values directly (e.g. `pred_iim` from library).
///
/// Returns (cycle_indices, rts, ims, mzs, xics) where:
/// - xics: Array3<f32> [n_pep, n_ion, n_cycle]
/// - ims, mzs: Array3<f32> [n_pep, n_ion, n_cycle] (empty if only_xic)
pub fn extract_xics(
    precursors: &[Precursor],
    locus: &[u32],
    ms1_map: &SwathMsMap,
    ms2_map: &SwathMsMap,
    ppm_tolerance: f32,
    im_tolerance: f32,
    rt_tolerance: Option<f32>,
    cycle_num: Option<usize>,
    scope: &str,
    only_xic: bool,
    query_ims: Option<&[f32]>,
) -> (
    Array2<i64>,
    Array2<f32>,
    Array3<f32>,
    Array3<f32>,
    Array3<f32>,
) {
    // Core: parallel over (peptide × ion) pairs — matches CUDA kernel model
    let n_pep = precursors.len();
    let fg_num = config().fg_num;
    let n_ion = match scope {
        "center" | "left" | "1h" | "2h" => 2 + fg_num,
        "big" => 4 * (2 + fg_num),
        "top6" => 6,
        _ => 2 + fg_num,
    };
    if n_pep > 100 {
        tracing::debug!("extract_xics parallel: {} × {} ions = {} tasks",
            n_pep, n_ion, n_pep * n_ion);
    }
    let scan_rts = &ms1_map.scan_rts;
    let cycle_total = scan_rts.len();
    let biggest_rt = scan_rts.last().copied().unwrap_or(1.0);
    let cycle_time = if cycle_total > 1 {
        (scan_rts[cycle_total - 1] - scan_rts[0]) / (cycle_total - 1) as f32
    } else {
        1.0
    };

    // Determine start indices and cycle counts
    let (idx_start_v, n_cycle) = if let Some(rt_tol) = rt_tolerance {
        let n_cycle_candidate = (rt_tol * 2.0 / cycle_time).ceil() as usize;
        let n_cycle = n_cycle_candidate.min(cycle_total);
        let starts: Vec<i64> = precursors
            .iter()
            .map(|pr| {
                let pred_rt = pr.pred_rt;
                let frac = (pred_rt - rt_tol) / biggest_rt;
                let idx = (frac * (cycle_total - 1) as f32).max(0.0) as i64;
                idx.min((cycle_total - n_cycle) as i64)
            })
            .collect();
        (starts, n_cycle)
    } else if let Some(cn) = cycle_num {
        let cn = cn.min(cycle_total);
        let starts: Vec<i64> = locus
            .iter()
            .map(|&l| {
                let start = l as i64 - (cn as i64 / 2);
                start.max(0).min((cycle_total - cn) as i64)
            })
            .collect();
        (starts, cn)
    } else {
        // Full gradient
        let starts = vec![0i64; n_pep];
        (starts, cycle_total)
    };

    let query_im: Vec<f32> = if let Some(qims) = query_ims {
        qims.to_vec()
    } else {
        // Default to pred_im (must be calibrated before calling)
        precursors.iter().map(|p| if p.pred_im > 0.0 { p.pred_im } else { p.measure_im }).collect()
    };

    // Allocate result arrays (flat for parallel access)
    let stride = n_cycle;
    let total_pep_ion = n_pep * n_ion;
    let mut result_rts = Array2::<f32>::zeros((n_pep, n_cycle));
    let mut result_cycle_idx = Array2::<i64>::zeros((n_pep, n_cycle));
    let mut flat_xics = vec![0.0f32; total_pep_ion * stride];
    let mut flat_ims = vec![0.0f32; total_pep_ion * stride];
    let mut flat_mzs = vec![0.0f32; total_pep_ion * stride];

    // Fill RT/cycle arrays (small, serial is fine)
    for k in 0..n_pep {
        let start = idx_start_v[k] as usize;
        let end = (start + n_cycle).min(cycle_total);
        for c in 0..(end - start) {
            result_cycle_idx[[k, c]] = (start + c) as i64;
            result_rts[[k, c]] = scan_rts[start + c];
        }
    }

    // Pre-build query m/z for all (peptide, ion) pairs
    // qmz_all[k][ion] = m/z value to query
    let qmz_all: Vec<Vec<f32>> = (0..n_pep).map(|k| {
        let pr = &precursors[k];
        let mut q = Vec::with_capacity(n_ion);
        match scope {
            "center" => {
                q.push(pr.pr_mz); q.push(pr.pr_mz);
                for fi in &pr.fragment_ions[..pr.fragment_ions.len().min(pr.fg_num as usize).min(12)] { q.push(fi.mz); }
            }
            "left" => {
                q.push(if pr.pr_mz_left > 0.0 { pr.pr_mz_left } else { pr.pr_mz });
                q.push(if pr.pr_mz_left > 0.0 { pr.pr_mz_left } else { pr.pr_mz });
                for fi in &pr.fragment_ions[..pr.fragment_ions.len().min(pr.fg_num as usize).min(12)] { q.push(if fi.mz_left > 0.0 { fi.mz_left } else { fi.mz }); }
            }
            "1h" => {
                q.push(if pr.pr_mz_1h > 0.0 { pr.pr_mz_1h } else { pr.pr_mz });
                q.push(if pr.pr_mz_1h > 0.0 { pr.pr_mz_1h } else { pr.pr_mz });
                for fi in &pr.fragment_ions[..pr.fragment_ions.len().min(pr.fg_num as usize).min(12)] { q.push(if fi.mz_1h > 0.0 { fi.mz_1h } else { fi.mz }); }
            }
            "2h" => {
                q.push(if pr.pr_mz_2h > 0.0 { pr.pr_mz_2h } else { pr.pr_mz });
                q.push(if pr.pr_mz_2h > 0.0 { pr.pr_mz_2h } else { pr.pr_mz });
                for fi in &pr.fragment_ions[..pr.fragment_ions.len().min(pr.fg_num as usize).min(12)] { q.push(if fi.mz_2h > 0.0 { fi.mz_2h } else { fi.mz }); }
            }
            "big" => {
                let ms1_mz = [pr.pr_mz_left, pr.pr_mz, pr.pr_mz_1h, pr.pr_mz_2h, pr.pr_mz_left, pr.pr_mz, pr.pr_mz_1h, pr.pr_mz_2h];
                q.extend_from_slice(&ms1_mz);
                for iso in 0..4 { for fi in &pr.fragment_ions[..pr.fragment_ions.len().min(pr.fg_num as usize).min(12)] { q.push(match iso { 0 => fi.mz_left, 1 => fi.mz, 2 => fi.mz_1h, _ => fi.mz_2h }); } }
            }
            "top6" => { for fi in pr.fragment_ions.iter().take(6) { q.push(fi.mz); } }
            _ => {}
        }
        while q.len() < n_ion { q.push(0.0); }
        q
    }).collect();

    // ── Parallel (peptide × ion) — matches CUDA kernel model ─────────
    let im_left_all: Vec<f32> = query_im.iter().map(|&im| im - im_tolerance).collect();
    let im_right_all: Vec<f32> = query_im.iter().map(|&im| im + im_tolerance).collect();
    let ppm_tol_f = ppm_tolerance;
    let xics_addr = flat_xics.as_mut_ptr() as usize;
    let ims_addr = flat_ims.as_mut_ptr() as usize;
    let mzs_addr = flat_mzs.as_mut_ptr() as usize;

    // Parallel over (peptide × ion) pairs — matches CUDA kernel design
    let total_tasks = total_pep_ion;
    (0..total_pep_ion).into_par_iter().for_each(|flat_idx| {
        let xics_ptr = xics_addr as *mut f32;
        let ims_ptr = ims_addr as *mut f32;
        let mzs_ptr = mzs_addr as *mut f32;
        let k = flat_idx / n_ion;
        let ion = flat_idx % n_ion;
        let q_mz = qmz_all[k][ion];
        if q_mz <= 0.0 { return; }

        let start = idx_start_v[k] as usize;
        let mz_left = q_mz * (1.0 - ppm_tol_f / 1_000_000.0);
        let mz_right = q_mz * (1.0 + ppm_tol_f / 1_000_000.0);
        let im_l = im_left_all[k];
        let im_r = im_right_all[k];
        let seek_indices = &ms2_map.seek_indices_cen;
        let mz_vals = &ms2_map.mz_values_cen;
        let im_vals = &ms2_map.im_values_cen;
        let h_vals = &ms2_map.height_values_cen;

        for c in 0..n_cycle {
            let scan_idx = start + c;
            let s_start = if scan_idx < seek_indices.len().saturating_sub(1) { seek_indices[scan_idx] as usize } else { 0 };
            let s_end = if scan_idx + 1 < seek_indices.len() { seek_indices[scan_idx + 1] as usize } else { mz_vals.len() };

            let best = find_best_match(
                &mz_vals[s_start..s_end], &im_vals[s_start..s_end], &h_vals[s_start..s_end],
                mz_left, mz_right, im_l, im_r,
            );
            let base = flat_idx * stride + c;
            unsafe {
                if !only_xic { *ims_ptr.add(base) = best.0; *mzs_ptr.add(base) = best.1; }
                *xics_ptr.add(base) = best.2;
            }
        }
    });

    // Convert flat arrays back to ndarray
    let result_xics = Array3::from_shape_vec((n_pep, n_ion, n_cycle), flat_xics)
        .unwrap_or_else(|_| Array3::zeros((n_pep, n_ion, n_cycle)));
    let result_ims = if only_xic { Array3::zeros((1, 1, 1)) }
        else { Array3::from_shape_vec((n_pep, n_ion, n_cycle), flat_ims).unwrap_or_else(|_| Array3::zeros((n_pep, n_ion, n_cycle))) };
    let result_mzs = if only_xic { Array3::zeros((1, 1, 1)) }
        else { Array3::from_shape_vec((n_pep, n_ion, n_cycle), flat_mzs).unwrap_or_else(|_| Array3::zeros((n_pep, n_ion, n_cycle))) };

    (result_cycle_idx, result_rts, result_ims, result_mzs, result_xics)
}

/// Find the best (im, mz, intensity) match within tolerance.
fn find_best_match(
    mz_vals: &[f32],
    im_vals: &[f32],
    height_vals: &[u32],
    mz_left: f32,
    mz_right: f32,
    im_left: f32,
    im_right: f32,
) -> (f32, f32, f32) {
    let mut best_im = -1.0f32;
    let mut best_mz = -1.0f32;
    let mut best_height = 0.0f32;

    let len = mz_vals.len();
    let start = match mz_vals.binary_search_by(|&x| x.partial_cmp(&mz_left).unwrap()) {
        Ok(i) => i,
        Err(i) => i,
    };

    for i in start..len {
        let mz = mz_vals[i];
        if mz > mz_right {
            break;
        }
        if mz < mz_left {
            continue;
        }
        let im = im_vals[i];
        if im > im_left && im < im_right {
            let h = height_vals[i] as f32;
            if h > best_height {
                best_height = h;
                best_mz = mz;
                best_im = im;
            }
        }
    }

    (best_im, best_mz, best_height)
}

// ===========================================================================
// Gaussian smoothing
// ===========================================================================

/// Simple weighted smooth: gaussian-ish 3-point moving average.
/// Equivalent to `fxic.gpu_simple_smooth()`.
pub fn gpu_simple_smooth(xics: &Array3<f32>) -> Array3<f32> {
    let dims = xics.dim();
    let mut result = Array3::zeros(dims);
    for k in 0..dims.0 {
        for ion in 0..dims.1 {
            let row_view = xics.index_axis(Axis(0), k);
            let row = row_view.index_axis(Axis(0), ion);
            let mut out_view = result.index_axis_mut(Axis(0), k);
            let mut out = out_view.index_axis_mut(Axis(0), ion);
            let n = dims.2;
            if n == 0 {
                continue;
            }
            out[0] = 0.667 * row[0] + 0.333 * row[1.min(n - 1)];
            for i in 1..n - 1 {
                out[i] = 0.5 * row[i] + 0.25 * (row[i - 1] + row[i + 1]);
            }
            if n > 1 {
                out[n - 1] = 0.333 * row[n - 2] + 0.667 * row[n - 1];
            }
        }
    }
    result
}
