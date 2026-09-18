//! DIA data model — Rust port of `TimsTOFTranspose` + `TimsTOFTransposeJIT`
//! for Bruker timsTOF `.d` data.
//!
//! The alpharaw/alphatims on-disk layout is a sparse matrix with *pushes*
//! (frame × scan) as rows and *tof indices* (discrete m/z bins) as columns:
//!   - `push_indptr[p]  .. push_indptr[p+1]`  -> tof indices of push p
//!   - `tof_indices[start .. stop]`            -> tof bins
//!   - `intensity_values[start .. stop]`       -> intensities
//!
//! DIAFluX transposes this so that tof indices are rows and pushes are
//! columns (`push_indices`, `tof_indptr`), which is efficient when only a few
//! tof bins are queried across the whole run (i.e. one precursor's fragments).
//! This module reproduces that transposed layout and the frame/scan/tof index
//! methods used by candidate selection (FFT) and scoring.

use std::collections::HashMap;

use anyhow::{bail, Result};
use ndarray::{Array4, Array5};

use crate::io::bruker::BrukerData;
use crate::io::mzml::MzMLData;

// ===========================================================================
// Free helper functions
// ===========================================================================

/// `np.searchsorted(arr, v, "left")` on an ascending array: first index where
/// `arr[i] >= v`, i.e. count of elements `< v`.
#[inline]
fn searchsorted_left(ascending: &[f64], v: f64) -> usize {
    ascending.partition_point(|&x| x < v)
}

/// `np.searchsorted(mobility_values[::-1], v, "right")` where `mobility_values`
/// is *descending*: this equals the count of elements `<= v`, independent of
/// the internal ordering.
#[inline]
fn searchsorted_right_reversed(descending: &[f64], v: f64) -> usize {
    descending.iter().filter(|&&x| x <= v).count()
}

/// `mass_range` from utils.py: query m/z list + ppm tolerance -> [low, high]
/// slice per query.
pub fn mass_range(mz_list: &[f64], ppm_tolerance: f64) -> Vec<[f64; 2]> {
    mz_list
        .iter()
        .map(|&m| {
            let d = ppm_tolerance * m / 1e6;
            [m - d, m + d]
        })
        .collect()
}

/// `make_slice_1d`: `[start, stop]` -> `[start, stop, 1]`.
#[inline]
fn make_slice_1d(start_stop: [i64; 2]) -> [i64; 3] {
    [start_stop[0], start_stop[1], 1]
}

/// `make_slice_2d`: `[[start, stop], ...]` -> `[[start, stop, 1], ...]`.
#[inline]
fn make_slice_2d(start_stop: &[[i64; 2]]) -> Vec<[i64; 3]> {
    start_stop
        .iter()
        .map(|&s| [s[0], s[1], 1])
        .collect()
}

/// Python `math.ceil(scan_len / optimize_size)` (float division) for possibly
/// negative `scan_len`.
#[inline]
fn ceil_div(scan_len: i64, optimize_size: i64) -> i64 {
    (scan_len as f64 / optimize_size as f64).ceil() as i64
}

// ===========================================================================
// Transpose
// ===========================================================================

/// Transpose detector events from (push -> tof) to (tof -> push) CSR layout.
///
/// Input:
///   - `tof_indices`: per-peak tof bin, length n_values
///   - `push_indptr`: CSR over pushes (frame × scan), length n_pushes + 1
///   - `n_tof`: number of tof bins (`len(mz_values)`)
///   - `intensity_values`: per-peak intensity, length n_values
///
/// Output:
///   - `push_indices`: per-peak push index (row of the new sparse matrix)
///   - `tof_indptr`: CSR over tof bins, length n_tof + 1
///   - `new_values`: intensities reordered to the transposed layout
pub fn transpose(
    tof_indices: &[u32],
    push_indptr: &[i64],
    n_tof: usize,
    intensity_values: &[u32],
) -> (Vec<u32>, Vec<i64>, Vec<u32>) {
    let n_values = tof_indices.len();

    // Count peaks per tof bin.
    let mut tof_indcount = vec![0u32; n_tof];
    for &t in tof_indices {
        if (t as usize) < n_tof {
            tof_indcount[t as usize] += 1;
        }
    }

    // Build the tof CSR pointer array.
    let mut tof_indptr = vec![0i64; n_tof + 1];
    for i in 0..n_tof {
        tof_indptr[i + 1] = tof_indptr[i] + tof_indcount[i] as i64;
    }

    // Reset counters (reused as running insertion cursor).
    for v in tof_indcount.iter_mut() {
        *v = 0;
    }

    // Scatter: for each push, for each peak, place (push, intensity) at the
    // cursor of its tof bin.
    let mut push_indices = vec![0u32; n_values];
    let mut new_values = vec![0u32; n_values];
    for push_idx in 0..push_indptr.len().saturating_sub(1) {
        let start = push_indptr[push_idx];
        let stop = push_indptr[push_idx + 1];
        for idx in start..stop {
            let i = idx as usize;
            if i >= n_values {
                break;
            }
            let tof = tof_indices[i] as usize;
            if tof >= n_tof {
                continue;
            }
            let pos = tof_indptr[tof] as usize + tof_indcount[tof] as usize;
            push_indices[pos] = push_idx as u32;
            new_values[pos] = intensity_values[i];
            tof_indcount[tof] += 1;
        }
    }

    (push_indices, tof_indptr, new_values)
}

// ===========================================================================
// DiaData — transposed timsTOF data structure
// ===========================================================================

/// Transposed Bruker timsTOF data, mirroring `TimsTOFTransposeJIT`.
///
/// Field semantics (see alphatims bruker.py):
///   - `rt_values`: retention time per frame (seconds, ascending).
///   - `mobility_values`: 1/K0 per scan index (descending: scan 0 = highest).
///   - `mz_values`: m/z per tof index (ascending).
///   - `cycle_len`: number of frames per DIA cycle (`cycle.shape[1]`).
///   - `cycle_mz`: flattened isolation windows, shape `(cycle_len*scan_max, 2)`
///     (`cycle.reshape(-1, 2)`), `[-1,-1]` for MS1 / precursor gaps.
///   - `dia_precursor_cycle`: precursor index per flattened cycle position.
pub struct DiaData {
    // Frame (RT) axis
    pub rt_values: Vec<f64>,
    pub accumulation_times: Vec<f64>,
    pub intensity_corrections: Vec<f64>,
    pub max_accumulation_time: f64,
    pub frame_max_index: usize,

    // Scan (mobility) axis
    pub mobility_values: Vec<f64>,
    pub mobility_min_value: f64,
    pub mobility_max_value: f64,
    pub scan_max_index: usize,

    // ToF (m/z) axis
    pub mz_values: Vec<f64>,
    pub tof_max_index: usize,

    // Quadrupole / DIA cycle
    pub cycle_len: usize,
    pub cycle_mz: Vec<[f64; 2]>,
    pub dia_precursor_cycle: Vec<i64>,
    pub quad_mz_values: Vec<[f64; 2]>,
    pub precursor_indices: Vec<i64>,
    pub raw_quad_indptr: Vec<i64>,
    pub quad_indptr: Vec<i64>,
    pub quad_max_mz_value: f64,
    pub quad_min_mz_value: f64,
    pub precursor_max_index: usize,
    pub precursor_cycle_max_index: usize,

    // Transposed peak data (tof -> push CSR)
    pub push_indices: Vec<u32>,
    pub tof_indptr: Vec<i64>,
    pub intensity_values: Vec<u32>,

    // Flags
    pub zeroth_frame: bool,
    pub has_ms1: bool,
    pub has_mobility: bool,
    pub use_calibrated_mz_values_as_default: bool,
}

impl DiaData {
    // -----------------------------------------------------------------------
    // Construction from a loaded Bruker `.d`
    // -----------------------------------------------------------------------

    /// Build the transposed DIA data model from a parsed Bruker `.d`.
    pub fn from_bruker(bruker: &BrukerData) -> Result<DiaData> {
        let scan_max_index = bruker.scan_max_index as usize;
        if scan_max_index == 0 {
            bail!("scan_max_index is 0");
        }
        let frame_max_index = bruker.frames.len();
        if frame_max_index == 0 {
            bail!("no frames");
        }

        // --- frame axis ---------------------------------------------------
        let rt_values: Vec<f64> = bruker.frames.iter().map(|f| f.time).collect();
        // TODO(P0-verify): Bruker Frames.AccumulationTime is not read by the
        // current io::bruker reader; use Time as a placeholder. Needed only for
        // quantification (P4), not candidate selection/scoring.
        let accumulation_times = rt_values.clone();
        let max_accumulation_time = accumulation_times
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let intensity_corrections = accumulation_times
            .iter()
            .map(|&t| if t > 0.0 { max_accumulation_time / t } else { 1.0 })
            .collect();

        // --- scan axis -----------------------------------------------------
        let mobility_values: Vec<f64> =
            bruker.mobility_values.iter().map(|&m| m as f64).collect();
        let mobility_min_value = mobility_values
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min);
        let mobility_max_value = mobility_values
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);

        // --- tof axis ------------------------------------------------------
        let mz_values: Vec<f64> = bruker.mz_values.iter().map(|&m| m as f64).collect();
        let tof_max_index = mz_values.len();

        // --- DIA cycle length ---------------------------------------------
        let ms1_indices: Vec<usize> = bruker
            .frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.msms_type == 0)
            .map(|(i, _)| i)
            .collect();
        let cycle_len = if ms1_indices.len() >= 2 {
            (ms1_indices[1] - ms1_indices[0]).max(1)
        } else {
            1
        };

        // --- build cycle_mz + dia_precursor_cycle from per-frame windows ---
        let n_pos = cycle_len * scan_max_index;
        let mut cycle_mz = vec![[-1.0f64; 2]; n_pos];
        let mut dia_precursor_cycle = vec![0i64; n_pos];

        // Map each per-frame isolation window into its (frame-in-cycle, scan)
        // positions. `quad_frames.frame_id` references `Frames.Id` (1-based).
        for qf in &bruker.quad_frames {
            // 0-based frame index within the run.
            let frame_0 = (qf.frame_id - 1).max(0) as usize;
            let frame_in_cycle = frame_0 % cycle_len;
            let low = qf.isolation_mz - qf.isolation_width / 2.0;
            let high = qf.isolation_mz + qf.isolation_width / 2.0;
            let sb = qf.scan_begin.max(0) as usize;
            let se = (qf.scan_end.max(0) as usize).min(scan_max_index);
            for s in sb..se {
                let pos = frame_in_cycle * scan_max_index + s;
                if pos < n_pos {
                    cycle_mz[pos] = [low, high];
                    // alpharaw stores the *frame-in-cycle* index here (0 = MS1,
                    // 1..N = MS2 windows), which is what the scoring quadrupole
                    // uses to index `cycle[0, precursor, scan, :]`.
                    dia_precursor_cycle[pos] = frame_in_cycle as i64;
                }
            }
        }

        let precursor_max_index =
            dia_precursor_cycle.iter().copied().max().unwrap_or(0) as usize + 1;

        // --- quadrupole arrays (best-effort; used by scoring P3+) ---------
        // Flatten the grouped windows into the segment list produced by
        // `_parse_quad_indptr`. For candidate selection only the
        // cycle arrays above are needed.
        let mut quad_mz_values: Vec<[f64; 2]> = Vec::new();
        let mut raw_quad_indptr: Vec<i64> = vec![0];
        for qf in &bruker.quad_frames {
            let low = qf.isolation_mz - qf.isolation_width / 2.0;
            let high = qf.isolation_mz + qf.isolation_width / 2.0;
            quad_mz_values.push([low, high]);
            let frame_0 = (qf.frame_id - 1).max(0) as i64;
            let start = frame_0 * scan_max_index as i64 + qf.scan_begin.max(0);
            let end = frame_0 * scan_max_index as i64 + qf.scan_end.max(0);
            raw_quad_indptr.push(start);
            raw_quad_indptr.push(end);
        }
        raw_quad_indptr.dedup();
        // quad_indptr = push_indptr[raw_quad_indptr]; we no longer keep the
        // pre-transpose push_indptr, so approximate with a monotonic fill.
        let mut quad_indptr = vec![0i64; raw_quad_indptr.len()];
        for i in 0..raw_quad_indptr.len() {
            quad_indptr[i] = raw_quad_indptr[i];
        }
        let precursor_indices: Vec<i64> = bruker
            .quad_frames
            .iter()
            .map(|qf| qf.precursor)
            .collect();
        let quad_max_mz_value = quad_mz_values
            .iter()
            .map(|w| w[1])
            .fold(f64::MIN, f64::max);
        let quad_min_mz_value = quad_mz_values
            .iter()
            .filter(|w| w[0] >= 0.0)
            .map(|w| w[0])
            .fold(f64::MAX, f64::min);

        let zeroth_frame = true;

        // --- transpose ------------------------------------------------------
        // pre-transpose push_indptr is BrukerData.push_indptr (indexed by
        // frame * scan_max_index + scan).
        let (push_indices, tof_indptr, intensity_values) = transpose(
            &bruker.tof_indices,
            &bruker.push_indptr,
            tof_max_index,
            &bruker.intensity_values,
        );

        let precursor_cycle_max_index = frame_max_index / cycle_len;

        Ok(DiaData {
            rt_values,
            accumulation_times,
            intensity_corrections,
            max_accumulation_time,
            frame_max_index,
            mobility_values,
            mobility_min_value,
            mobility_max_value,
            scan_max_index,
            mz_values,
            tof_max_index,
            cycle_len,
            cycle_mz,
            dia_precursor_cycle,
            quad_mz_values,
            precursor_indices,
            raw_quad_indptr,
            quad_indptr,
            quad_max_mz_value,
            quad_min_mz_value,
            precursor_max_index,
            precursor_cycle_max_index,
            push_indices,
            tof_indptr,
            intensity_values,
            zeroth_frame,
            has_ms1: ms1_indices.len() >= 2,
            has_mobility: true,
            use_calibrated_mz_values_as_default: false,
        })
    }

    // -----------------------------------------------------------------------
    // Construction from mzML (LC-MS/MS without ion mobility)
    // -----------------------------------------------------------------------

    /// Build the DIA data model from parsed mzML spectra.
    ///
    /// mzML DIA has no ion-mobility dimension, so the mapping onto the
    /// frame/scan/tof axes mirrors the reference reader:
    ///   - **frame**   = spectrum index (each spectrum is one acquisition event),
    ///   - **scan**    = a synthetic 2-bin mobility axis
    ///                   (`scan_max_index = 2`, `mobility = [1e-6, 0]`),
    ///   - **tof**     = a regular m/z grid over which peaks are binned.
    ///
    /// Every spectrum is pushed into both synthetic scan bins (matching
    /// `AlphaRawJIT.get_dense_intensity`, which writes each spectrum into both
    /// mobility positions), and `get_scan_indices_tolerance` returns the fixed
    /// `[0, 2, 1]` window. This keeps the FFT selection's 2-D kernel well
    /// defined without a real mobility axis.
    ///
    /// Only MS1 survey scans and MS2 isolation windows are kept: mzML from TMT
    /// and other acquisition schemes interleaves MS3 (or higher) scans that the
    /// DIA model does not consume, and including them would both pollute the
    /// frame axis and break the `cycle_len` reconstruction.
    pub fn from_mzml(mzml: &MzMLData) -> Result<DiaData> {
        let spectra = &mzml.spectra;
        if spectra.is_empty() {
            bail!("no spectra");
        }
        if mzml.ms1_indices.is_empty() {
            bail!("no MS1 spectra");
        }

        // --- frame axis (MS1 + MS2 only) --------------------------------
        let frame_idx: Vec<usize> = (0..spectra.len())
            .filter(|&i| {
                let l = spectra[i].ms_level;
                l == 1 || l == 2
            })
            .collect();
        let n_frames = frame_idx.len();
        if n_frames == 0 {
            bail!("no MS1/MS2 spectra");
        }

        let rt_values: Vec<f64> = frame_idx.iter().map(|&i| spectra[i].rt as f64).collect();
        let accumulation_times = rt_values.clone();
        let max_accumulation_time = accumulation_times
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max);
        let intensity_corrections = accumulation_times
            .iter()
            .map(|&t| if t > 0.0 { max_accumulation_time / t } else { 1.0 })
            .collect();

        // --- scan axis (synthetic 2-bin mobility) ------------------------
        // Matches alpharaw's `_scan_max_index`/`mobility_values` defaults:
        // `scan_max_index = 1` flags "no mobility", but the dense extraction
        // always uses a 2-wide scan window, so we model two scan bins directly.
        let scan_max_index = 2usize;
        let mobility_values = vec![1e-6f64, 0.0];
        let mobility_min_value = 0.0;
        let mobility_max_value = 1e-6;

        // --- DIA cycle length --------------------------------------------
        // 1 MS1 + one frame per distinct MS2 isolation window. Derive it from
        // the median gap between consecutive MS1 frames (robust to a stray
        // adjacent MS1), falling back to the distinct-window count when the MS1
        // spacing is irregular (e.g. data-dependent acquisition).
        let ms1_pos: Vec<usize> = frame_idx
            .iter()
            .enumerate()
            .filter(|(_, &i)| spectra[i].ms_level == 1)
            .map(|(pos, _)| pos)
            .collect();
        let mut gaps: Vec<usize> = ms1_pos.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.sort_unstable();
        let median_gap = if gaps.is_empty() { 0 } else { gaps[gaps.len() / 2] };
        let mut windows: Vec<(i64, i64)> = frame_idx
            .iter()
            .filter(|&&i| spectra[i].ms_level == 2)
            .map(|&i| {
                let (lo, hi) = spectra[i].isolation_window;
                ((lo * 1000.0).round() as i64, (hi * 1000.0).round() as i64)
            })
            .collect();
        windows.sort_unstable();
        windows.dedup();
        let n_windows = windows.len().max(1);
        let cycle_len = if median_gap >= 2 { median_gap } else { n_windows + 1 };

        // --- isolation windows per frame-in-cycle ------------------------
        let n_pos = cycle_len * scan_max_index;
        let mut cycle_mz = vec![[-1.0f64; 2]; n_pos];
        let mut dia_precursor_cycle = vec![0i64; n_pos];
        for (fi, &si) in frame_idx.iter().enumerate() {
            let s = &spectra[si];
            let frame_in_cycle = fi % cycle_len;
            if s.ms_level == 2 {
                let (lo, hi) = s.isolation_window;
                // Every synthetic scan bin carries the same window (alpharaw
                // duplicates the spectrum into both mobility positions).
                for scan in 0..scan_max_index {
                    let pos = frame_in_cycle * scan_max_index + scan;
                    cycle_mz[pos] = [lo as f64, hi as f64];
                    dia_precursor_cycle[pos] = frame_in_cycle as i64;
                }
            }
            // MS1 frames keep the [-1,-1] sentinel and precursor cycle 0.
        }
        let precursor_max_index = cycle_len;

        // --- m/z grid ------------------------------------------------------
        let mut mz_min = f64::MAX;
        let mut mz_max = f64::MIN;
        for &si in &frame_idx {
            for &m in &spectra[si].mz_array {
                if m > 0.0 {
                    mz_min = mz_min.min(m as f64);
                    mz_max = mz_max.max(m as f64);
                }
            }
        }
        if mz_min == f64::MAX {
            mz_min = 100.0;
            mz_max = 2000.0;
        }
        mz_min = mz_min.max(50.0);
        mz_max = mz_max.min(5000.0);
        let n_tof = 2_000_000usize;
        let step = (mz_max - mz_min) / (n_tof - 1) as f64;
        let mz_values: Vec<f64> = (0..n_tof).map(|i| mz_min + i as f64 * step).collect();

        // --- bin peaks into (frame × scan, tof, intensity) and build CSR --
        // Each spectrum is pushed into BOTH synthetic scan bins so that the
        // dense extraction sees a uniform 2-wide mobility axis.
        let mut tof_indices: Vec<u32> = Vec::new();
        let mut intensity_values: Vec<u32> = Vec::new();
        let mut push_indptr = vec![0i64; n_frames * scan_max_index + 1];
        for (fi, &si) in frame_idx.iter().enumerate() {
            let s = &spectra[si];
            let n_peaks = s.mz_array.len().min(s.intensity_array.len());
            for scan in 0..scan_max_index {
                for pi in 0..n_peaks {
                    let mz = s.mz_array[pi] as f64;
                    let inten = s.intensity_array[pi];
                    if mz <= 0.0 || inten < 0.0 {
                        continue;
                    }
                    let tof = (((mz - mz_min) / step).round() as i64)
                        .clamp(0, n_tof as i64 - 1) as usize;
                    tof_indices.push(tof as u32);
                    intensity_values.push(inten as u32);
                }
                let push = fi * scan_max_index + scan;
                push_indptr[push + 1] = tof_indices.len() as i64;
            }
        }

        // --- transpose (push -> tof) to (tof -> push) ---------------------
        let (push_indices, tof_indptr, intensity_values) =
            transpose(&tof_indices, &push_indptr, n_tof, &intensity_values);

        // --- quadrupole arrays (best-effort; scoring uses `cycle_mz`) ------
        let mut quad_mz_values: Vec<[f64; 2]> = Vec::new();
        let mut precursor_indices: Vec<i64> = Vec::new();
        let mut raw_quad_indptr: Vec<i64> = vec![0];
        for (fi, &si) in frame_idx.iter().enumerate() {
            let s = &spectra[si];
            if s.ms_level == 2 {
                let (lo, hi) = s.isolation_window;
                quad_mz_values.push([lo as f64, hi as f64]);
                precursor_indices.push((fi % cycle_len) as i64);
                raw_quad_indptr.push(fi as i64);
                raw_quad_indptr.push(fi as i64 + 1);
            }
        }
        raw_quad_indptr.dedup();
        let mut quad_indptr = vec![0i64; raw_quad_indptr.len()];
        for i in 0..raw_quad_indptr.len() {
            quad_indptr[i] = raw_quad_indptr[i];
        }
        let quad_max_mz_value = quad_mz_values.iter().map(|w| w[1]).fold(f64::MIN, f64::max);
        let quad_min_mz_value = quad_mz_values
            .iter()
            .filter(|w| w[0] >= 0.0)
            .map(|w| w[0])
            .fold(f64::MAX, f64::min);

        let precursor_cycle_max_index = n_frames / cycle_len;

        Ok(DiaData {
            rt_values,
            accumulation_times,
            intensity_corrections,
            max_accumulation_time,
            frame_max_index: n_frames,
            mobility_values,
            mobility_min_value,
            mobility_max_value,
            scan_max_index,
            mz_values,
            tof_max_index: n_tof,
            cycle_len,
            cycle_mz,
            dia_precursor_cycle,
            quad_mz_values,
            precursor_indices,
            raw_quad_indptr,
            quad_indptr,
            quad_max_mz_value,
            quad_min_mz_value,
            precursor_max_index,
            precursor_cycle_max_index,
            push_indices,
            tof_indptr,
            intensity_values,
            zeroth_frame: false,
            has_ms1: ms1_pos.len() >= 2,
            has_mobility: false,
            use_calibrated_mz_values_as_default: false,
        })
    }

    // -----------------------------------------------------------------------
    // Frame (RT) index conversion
    // -----------------------------------------------------------------------

    /// Convert an interval of two RT values into a frame-index interval,
    /// rounded up to whole DIA cycles and a multiple of `optimize_size`.
    /// Port of `get_frame_indices` (utils.py) / `_get_frame_indices`.
    pub fn get_frame_indices(
        &self,
        rt_limits: [f64; 2],
        optimize_size: usize,
        min_size: usize,
    ) -> [i64; 3] {
        let zf = self.zeroth_frame as i64;
        let cycle_len = self.cycle_len as i64;
        let precursor_cycle_max = self.precursor_cycle_max_index as i64;

        let mut frame_index = [0i64; 2];
        for (k, &rt) in rt_limits.iter().enumerate() {
            frame_index[k] = searchsorted_left(&self.rt_values, rt) as i64;
        }

        let precursor_cycle_limits = [
            (frame_index[0] + zf) / cycle_len,
            (frame_index[1] + zf) / cycle_len,
        ];
        let precursor_cycle_len = precursor_cycle_limits[1] - precursor_cycle_limits[0];

        let mut optimal_len = precursor_cycle_len.max(min_size as i64);
        optimal_len = (optimize_size as i64)
            * ceil_div(optimal_len, optimize_size as i64);

        let mut oc0 = precursor_cycle_limits[0];
        let mut oc1 = oc0 + optimal_len;
        if oc1 > precursor_cycle_max {
            oc1 = precursor_cycle_max;
            oc0 = precursor_cycle_max - optimal_len;
            if oc0 < 0 {
                oc0 = if precursor_cycle_max % 2 == 0 { 0 } else { 1 };
            }
        }

        let frame_start = oc0 * cycle_len + zf;
        let frame_stop = oc1 * cycle_len + zf;
        make_slice_1d([frame_start, frame_stop])
    }

    /// `get_frame_indices_tolerance`: frame slice for `rt ± tolerance` (seconds).
    pub fn get_frame_indices_tolerance(
        &self,
        rt: f64,
        tolerance: f64,
        optimize_size: usize,
        min_size: usize,
    ) -> [i64; 3] {
        let rt_limits = [rt - tolerance, rt + tolerance];
        self.get_frame_indices(rt_limits, optimize_size, min_size)
    }

    // -----------------------------------------------------------------------
    // Scan (mobility) index conversion
    // -----------------------------------------------------------------------

    /// Convert a mobility interval into a scan-index interval, rounded up to a
    /// multiple of `optimize_size`. Port of `_get_scan_indices`.
    pub fn get_scan_indices(
        &self,
        mobility_limits: [f64; 2],
        optimize_size: usize,
    ) -> [i64; 3] {
        let scan_max = self.scan_max_index as i64;

        let mut scan_index = [0i64; 2];
        for (k, &ml) in mobility_limits.iter().enumerate() {
            scan_index[k] =
                scan_max - searchsorted_right_reversed(&self.mobility_values, ml) as i64;
        }

        let scan_len = scan_index[0] - scan_index[1];
        let optimal_len = (optimize_size as i64) * ceil_div(scan_len, optimize_size as i64);

        let mut start = scan_index[0];
        let mut stop = scan_index[0] - optimal_len;
        if stop < 0 {
            stop = 0;
            start = optimal_len;
            start = start.min(scan_max);
        }
        make_slice_1d([start, stop])
    }

    /// `get_scan_indices_tolerance`: scan slice for `mobility ± tolerance`.
    ///
    /// For data without ion mobility (mzML), the reference reader ignores
    /// mobility entirely and returns a fixed 2-scan window (`[[0, 2, 1]]`).
    pub fn get_scan_indices_tolerance(
        &self,
        mobility: f64,
        tolerance: f64,
        optimize_size: usize,
    ) -> [i64; 3] {
        if !self.has_mobility {
            return [0, 2, 1];
        }
        let mobility_limits = [mobility + tolerance, mobility - tolerance];
        self.get_scan_indices(mobility_limits, optimize_size)
    }

    // -----------------------------------------------------------------------
    // ToF (m/z) index conversion
    // -----------------------------------------------------------------------

    /// Convert an m/z interval into a tof-index interval (left searchsorted).
    pub fn get_tof_indices(&self, mz_limits: [f64; 2]) -> [i64; 2] {
        [
            searchsorted_left(&self.mz_values, mz_limits[0]) as i64,
            searchsorted_left(&self.mz_values, mz_limits[1]) as i64,
        ]
    }

    // -----------------------------------------------------------------------
    // Cycle mask + push indices
    // -----------------------------------------------------------------------

    /// `_cycle_mask`: for each score group (query quadrupole), a boolean mask
    /// over flattened cycle positions whose isolation window overlaps it.
    pub fn cycle_mask(&self, quad_slices: &[[f64; 2]]) -> Vec<Vec<bool>> {
        let n_score_groups = quad_slices.len();
        let mut mask = vec![vec![false; self.cycle_mz.len()]; n_score_groups];
        for (i, &[mz_start, mz_stop]) in self.cycle_mz.iter().enumerate() {
            for (j, &[quad_start, quad_stop]) in quad_slices.iter().enumerate() {
                if quad_start <= mz_stop && quad_stop >= mz_start {
                    mask[j][i] = true;
                }
            }
        }
        mask
    }

    /// `_get_push_indices`: enumerate push indices (and their precursor cycle)
    /// inside the frame/scan window that match the cycle mask.
    pub fn get_push_indices(
        &self,
        frame_limits: [i64; 3],
        scan_limits: [i64; 3],
        cycle_mask: &[Vec<bool>],
    ) -> (Vec<u32>, Vec<i64>) {
        let n_score_groups = cycle_mask.len();
        let len_dia_mz_cycle = self.dia_precursor_cycle.len();
        if len_dia_mz_cycle == 0 {
            return (Vec::new(), Vec::new());
        }

        let scan_max = self.scan_max_index as i64;

        let mut push_indices = Vec::new();
        let mut absolute_precursor_cycle = Vec::new();

        let [frame_start, frame_stop, frame_step] = frame_limits;
        let [scan_start, scan_stop, scan_step] = scan_limits;

        let mut frame_index = frame_start;
        while frame_index < frame_stop {
            let mut scan_index = scan_start;
            while scan_index < scan_stop {
                let push_index = frame_index * scan_max + scan_index;
                let cyclic_push_index = if self.zeroth_frame {
                    push_index - scan_max
                } else {
                    push_index
                };
                let scan_in_dia_mz_cycle = cyclic_push_index.rem_euclid(len_dia_mz_cycle as i64);

                for i in 0..n_score_groups {
                    if cycle_mask[i][scan_in_dia_mz_cycle as usize] {
                        let precursor_cycle = self.dia_precursor_cycle[scan_in_dia_mz_cycle as usize];
                        absolute_precursor_cycle.push(precursor_cycle);
                        push_indices.push(push_index as u32);
                    }
                }
                scan_index += scan_step;
            }
            frame_index += frame_step;
        }

        (push_indices, absolute_precursor_cycle)
    }

    // -----------------------------------------------------------------------
    // Dense assembly
    // -----------------------------------------------------------------------

    /// `_assemble_push_intensity`: sum intensities into a
    /// `(1, n_tof, mobility_len, precursor_cycle_len)` array.
    pub fn assemble_push_intensity(
        &self,
        tof_limits: &[[i64; 3]],
        push_query: &[u32],
        precursor_index: &[i64],
        frame_limits: [i64; 3],
        scan_limits: [i64; 3],
    ) -> (Array4<f32>, Vec<i64>) {
        let n_tof_slices = tof_limits.len();

        let mobility_start = scan_limits[0];
        let mobility_stop = scan_limits[1];
        let mobility_len = (mobility_stop - mobility_start).max(0) as usize;

        let zf = self.zeroth_frame as i64;
        let cycle_len = self.cycle_len as i64;
        let precursor_cycle_start = (frame_limits[0] - zf) / cycle_len;
        let precursor_cycle_stop = (frame_limits[1] - zf) / cycle_len;
        let precursor_cycle_len = (precursor_cycle_stop - precursor_cycle_start).max(0) as usize;

        let mut dense = Array4::<f32>::zeros((1, n_tof_slices, mobility_len, precursor_cycle_len));

        if precursor_index.is_empty() {
            return (dense, Vec::new());
        }

        // unique precursor indices (preserve first-seen order via sorted dedup)
        let mut unique: Vec<i64> = precursor_index.to_vec();
        unique.sort_unstable();
        unique.dedup();

        let scan_max = self.scan_max_index as i64;
        let n_push = push_query.len();

        for (j, &[tof_start, tof_stop, tof_step]) in tof_limits.iter().enumerate() {
            let mut tof_index = tof_start;
            while tof_index < tof_stop {
                let t = tof_index as usize;
                if t + 1 >= self.tof_indptr.len() {
                    tof_index += tof_step;
                    continue;
                }
                let start = self.tof_indptr[t] as usize;
                let stop = self.tof_indptr[t + 1] as usize;

                let mut i = 0usize;
                let mut idx = start;
                while idx < stop && i < n_push {
                    let pq = push_query[i];
                    let pi = self.push_indices[idx];
                    if pq < pi {
                        i += 1;
                    } else {
                        if pq == pi {
                            let frame_index = (pi as i64) / scan_max;
                            let scan_index = (pi as i64) % scan_max;
                            let precursor_cycle_index = (frame_index - zf) / cycle_len;
                            let relative_scan = (scan_index - mobility_start) as usize;
                            let relative_precursor =
                                (precursor_cycle_index - precursor_cycle_start) as usize;

                            if relative_scan < mobility_len
                                && relative_precursor < precursor_cycle_len
                            {
                                dense[[0, j, relative_scan, relative_precursor]] +=
                                    self.intensity_values[idx] as f32;
                            }
                        }
                        idx += 1;
                    }
                }
                tof_index += tof_step;
            }
        }

        (dense, unique)
    }

    /// `_assemble_push`: two-channel dense output (intensity + weighted mass
    /// error / absolute mass), shape
    /// `(2, n_tof, n_precursors, mobility_len, precursor_cycle_len)`.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_push(
        &self,
        tof_limits: &[[i64; 3]],
        mz_values: &[f64],
        push_query: &[u32],
        precursor_index: &[i64],
        frame_limits: [i64; 3],
        scan_limits: [i64; 3],
        ppm_background: f64,
        absolute_masses: bool,
    ) -> (Array5<f32>, Vec<i64>) {
        const HIGH_EPSILON: f64 = 1e-26;
        const LOW_EPSILON: f64 = 1e-36;

        let n_tof_slices = tof_limits.len();

        let mobility_start = scan_limits[0];
        let mobility_stop = scan_limits[1];
        let mobility_len = (mobility_stop - mobility_start).max(0) as usize;

        let zf = self.zeroth_frame as i64;
        let cycle_len = self.cycle_len as i64;
        let precursor_cycle_start = (frame_limits[0] - zf) / cycle_len;
        let precursor_cycle_stop = (frame_limits[1] - zf) / cycle_len;
        let precursor_cycle_len = (precursor_cycle_stop - precursor_cycle_start).max(0) as usize;

        if precursor_index.is_empty() {
            return (
                Array5::<f32>::zeros((0, 0, 0, 0, 0)),
                Vec::new(),
            );
        }

        // unique precursors + reverse index mapping
        let mut unique: Vec<i64> = precursor_index.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let max_unique = unique.iter().copied().max().unwrap_or(0);
        let mut reverse = vec![0u8; (max_unique + 1) as usize];
        for (k, &u) in unique.iter().enumerate() {
            reverse[u as usize] = k as u8;
        }
        let relative_precursor_index: Vec<u8> =
            precursor_index.iter().map(|&p| reverse[p as usize]).collect();

        let n_precursor_indices = unique.len();

        let mut dense = Array5::<f32>::zeros((
            2,
            n_tof_slices,
            n_precursor_indices,
            mobility_len,
            precursor_cycle_len,
        ));

        if !absolute_masses {
            for j in 0..n_tof_slices {
                for p in 0..n_precursor_indices {
                    for s in 0..mobility_len {
                        for c in 0..precursor_cycle_len {
                            dense[[1, j, p, s, c]] = ppm_background as f32;
                        }
                    }
                }
            }
        }

        let scan_max = self.scan_max_index as i64;
        let n_push = push_query.len();

        for (j, &[tof_start, tof_stop, tof_step]) in tof_limits.iter().enumerate() {
            let library_mz = mz_values[j];
            let mut tof_index = tof_start;
            while tof_index < tof_stop {
                let t = tof_index as usize;
                if t + 1 >= self.tof_indptr.len() {
                    tof_index += tof_step;
                    continue;
                }
                let measured_mz = self.mz_values.get(t).copied().unwrap_or(0.0);

                let start = self.tof_indptr[t] as usize;
                let stop = self.tof_indptr[t + 1] as usize;

                let mut i = 0usize;
                let mut idx = start;
                while idx < stop && i < n_push {
                    let pq = push_query[i];
                    let pi = self.push_indices[idx];
                    if pq < pi {
                        i += 1;
                    } else {
                        if pq == pi {
                            let frame_index = (pi as i64) / scan_max;
                            let scan_index = (pi as i64) % scan_max;
                            let precursor_cycle_index = (frame_index - zf) / cycle_len;
                            let relative_scan = (scan_index - mobility_start) as usize;
                            let relative_precursor =
                                (precursor_cycle_index - precursor_cycle_start) as usize;
                            let rel_prec = relative_precursor_index[i] as usize;

                            if relative_scan < mobility_len
                                && relative_precursor < precursor_cycle_len
                            {
                                let accumulated_intensity =
                                    dense[[0, j, rel_prec, relative_scan, relative_precursor]] as f64;
                                let accumulated_dim1 =
                                    dense[[1, j, rel_prec, relative_scan, relative_precursor]] as f64;

                                let mut new_intensity = self.intensity_values[idx] as f64;
                                new_intensity *= (new_intensity > HIGH_EPSILON) as i32 as f64;

                                let new_dim1 = if absolute_masses {
                                    (accumulated_dim1 * accumulated_intensity
                                        + new_intensity * measured_mz
                                        + LOW_EPSILON)
                                        / (accumulated_intensity + new_intensity + LOW_EPSILON)
                                } else {
                                    let new_error =
                                        (measured_mz - library_mz) / library_mz * 1e6;
                                    (accumulated_dim1 * accumulated_intensity
                                        + new_intensity * new_error
                                        + LOW_EPSILON)
                                        / (accumulated_intensity + new_intensity + LOW_EPSILON)
                                };

                                dense[[0, j, rel_prec, relative_scan, relative_precursor]] =
                                    (accumulated_intensity + new_intensity) as f32;
                                dense[[1, j, rel_prec, relative_scan, relative_precursor]] =
                                    new_dim1 as f32;
                            }
                        }
                        idx += 1;
                    }
                }
                tof_index += tof_step;
            }
        }

        (dense, unique)
    }

    // -----------------------------------------------------------------------
    // Public dense extraction entry points
    // -----------------------------------------------------------------------

    /// `get_dense_intensity`: `(1, n_tof, mobility_len, precursor_cycle_len)`.
    pub fn get_dense_intensity(
        &self,
        frame_limits: [i64; 3],
        scan_limits: [i64; 3],
        mz_values: &[f64],
        mass_tolerance: f64,
        quadrupole_mz: &[[f64; 2]],
    ) -> (Array4<f32>, Vec<i64>) {
        let tof_limits = make_slice_2d(
            &self
                .get_tof_indices_from_mass(mz_values, mass_tolerance),
        );
        let mz_mask = self.cycle_mask(quadrupole_mz);
        let (push_query, absolute_precursor_index) =
            self.get_push_indices(frame_limits, scan_limits, &mz_mask);

        let (dense, uniq) = self.assemble_push_intensity(
            &tof_limits,
            &push_query,
            &absolute_precursor_index,
            frame_limits,
            scan_limits,
        );

        (dense, uniq)
    }

    /// `get_dense`: `(2, n_tof, n_precursors, mobility_len, precursor_cycle_len)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_dense(
        &self,
        frame_limits: [i64; 3],
        scan_limits: [i64; 3],
        mz_values: &[f64],
        mass_tolerance: f64,
        quadrupole_mz: &[[f64; 2]],
        absolute_masses: bool,
    ) -> (Array5<f32>, Vec<i64>) {
        let tof_limits = make_slice_2d(
            &self
                .get_tof_indices_from_mass(mz_values, mass_tolerance),
        );
        let mz_mask = self.cycle_mask(quadrupole_mz);
        let (push_query, absolute_precursor_index) =
            self.get_push_indices(frame_limits, scan_limits, &mz_mask);

        self.assemble_push(
            &tof_limits,
            mz_values,
            &push_query,
            &absolute_precursor_index,
            frame_limits,
            scan_limits,
            mass_tolerance,
            absolute_masses,
        )
    }

    /// `mass_range` + `get_tof_indices`: tof [start, stop] per query m/z.
    fn get_tof_indices_from_mass(&self, mz_values: &[f64], mass_tolerance: f64) -> Vec<[i64; 2]> {
        mass_range(mz_values, mass_tolerance)
            .iter()
            .map(|&[lo, hi]| self.get_tof_indices([lo, hi]))
            .collect()
    }

    /// Convenience: number of DIA cycles (frames) present in the run.
    pub fn n_cycles(&self) -> usize {
        self.frame_max_index / self.cycle_len
    }
}

// ===========================================================================
// SpecLibFlat — columnar precursor / fragment library (alphabase SpecLibFlat)
// ===========================================================================

/// Columnar precursor library (mirrors `SpecLibFlat.precursor_df`).
///
/// core addresses columns by string name on a single pandas DataFrame;
/// here we use a Vec-of-structs with the columns needed by candidate
/// selection (P2) and scoring (P3).
#[derive(Debug, Clone, Default)]
pub struct PrecursorDf {
    /// Unique precursor id (e.g. `peptide_charge`).
    pub precursor_idx: Vec<u32>,
    pub sequence: Vec<String>,
    /// Modification annotations (AlphaBase `"ModA;ModB"` format); empty for
    /// unmodified / stripped DIA-NN sequences.
    pub mods: Vec<String>,
    /// Modification site positions (AlphaBase format); empty for stripped
    /// DIA-NN sequences. Used with `sequence`/`mods` to compute `mod_seq_hash`.
    pub mod_sites: Vec<String>,
    pub charge: Vec<u8>,
    pub precursor_mz: Vec<f64>,
    /// Predicted / calibrated retention time (seconds).
    pub rt_pred: Vec<f64>,
    /// Predicted ion mobility (1/K0).
    pub mobility_pred: Vec<f64>,
    /// Decoy flag (0 target, 1 decoy).
    pub decoy: Vec<u8>,
    /// Protein ids / names.
    pub proteins: Vec<String>,
    /// Number of fragments per precursor.
    pub n_fragments: Vec<u32>,
    /// Start index of this precursor's fragments in `FragmentDf` (inclusive).
    pub flat_frag_start_idx: Vec<u32>,
    /// Stop index of this precursor's fragments in `FragmentDf` (exclusive).
    pub flat_frag_stop_idx: Vec<u32>,
    /// Elution-group id linking precursors that compete (default: one per row).
    pub elution_group_idx: Vec<u32>,
    /// Precursor isotope intensities, row-major `n_precursors × n_isotopes`.
    pub isotopes: Vec<f32>,
    /// Number of isotope columns.
    pub n_isotopes: usize,
    /// Monoisotopic position within each isotope row.
    pub mono_isotope_idx: Vec<i32>,
}

/// Columnar fragment library (mirrors `SpecLibFlat.fragment_df`).
#[derive(Debug, Clone, Default)]
pub struct FragmentDf {
    pub precursor_idx: Vec<u32>,
    /// Fragment m/z.
    pub fragment_mz: Vec<f64>,
    /// Fragment library m/z (calibratable; initialized to `fragment_mz`).
    pub mz_library: Vec<f64>,
    /// Relative intensity (0..1).
    pub fragment_intensity: Vec<f32>,
    /// Fragment type (series id: a=97, b=98, c=99, x=120, y=121, z=122).
    pub fragment_type: Vec<u8>,
    /// Neutral-loss type (0 none, 1 lossH, 2 addH, 17 NH3, 18 H2O, 98 modloss).
    pub loss_type: Vec<u8>,
    /// Fragment series number.
    pub fragment_number: Vec<u16>,
    /// Fragment position in sequence (0-based, from N-terminus).
    pub position: Vec<u16>,
    /// Fragment charge.
    pub fragment_charge: Vec<u8>,
    /// Number of precursors sharing this fragment m/z (within an elution group).
    pub cardinality: Vec<u8>,
}

/// Flat spectral library used by the search and scoring stages.
#[derive(Debug, Clone, Default)]
pub struct SpecLibFlat {
    pub precursor_df: PrecursorDf,
    pub fragment_df: FragmentDf,
    /// precursor_idx -> row index in `precursor_df` (for O(1) lookup).
    pub precursor_index_map: HashMap<u32, usize>,
}

impl SpecLibFlat {
    pub fn n_precursors(&self) -> usize {
        self.precursor_df.precursor_idx.len()
    }

    /// Rebuild the internal precursor index map (call after mutating).
    pub fn rebuild_index(&mut self) {
        self.precursor_index_map.clear();
        for (i, &p) in self.precursor_df.precursor_idx.iter().enumerate() {
            self.precursor_index_map.insert(p, i);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_roundtrip() {
        // 2 pushes, 3 tof bins.
        // push 0: peaks at tof {0, 2}
        // push 1: peaks at tof {1}
        let push_indptr = vec![0i64, 2, 3];
        let tof_indices = vec![0u32, 2u32, 1u32];
        let intensity = vec![10u32, 30u32, 20u32];
        let n_tof = 3;

        let (push_indices, tof_indptr, new_values) =
            transpose(&tof_indices, &push_indptr, n_tof, &intensity);

        // tof_indptr CSR: tof 0 -> 1 peak, tof 1 -> 1 peak, tof 2 -> 1 peak
        assert_eq!(tof_indptr, vec![0i64, 1, 2, 3]);

        // Reconstruct per-tof (push, intensity) pairs.
        let mut per_tof: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_tof];
        for t in 0..n_tof {
            for idx in tof_indptr[t]..tof_indptr[t + 1] {
                per_tof[t].push((
                    push_indices[idx as usize],
                    new_values[idx as usize],
                ));
            }
        }
        assert_eq!(per_tof[0], vec![(0, 10)]);
        assert_eq!(per_tof[1], vec![(1, 20)]);
        assert_eq!(per_tof[2], vec![(0, 30)]);
    }

    #[test]
    fn mass_range_ppm() {
        let r = mass_range(&[1000.0], 10.0);
        assert!((r[0][0] - 999.99).abs() < 1e-3);
        assert!((r[0][1] - 1000.01).abs() < 1e-3);
    }

    #[test]
    fn from_mzml_builds_cycles_and_bins_peaks() {
        use crate::io::mzml::{MzMLData, Spectrum};

        // Two DIA cycles, cycle_len = 3: MS1 + two MS2 windows.
        let mk = |ms_level: u8, rt: f32, win: (f32, f32), mz: Vec<f32>, inten: Vec<f32>| {
            Spectrum {
                ms_level,
                rt,
                isolation_window: win,
                mz_array: mz,
                intensity_array: inten,
            }
        };
        let spectra = vec![
            mk(1, 0.0, (0.0, 0.0), vec![500.0], vec![10.0]),   // MS1 cycle 0
            mk(2, 0.5, (400.0, 410.0), vec![405.0], vec![20.0]), // win1
            mk(2, 1.0, (410.0, 420.0), vec![415.0], vec![30.0]), // win2
            mk(1, 1.5, (0.0, 0.0), vec![500.0], vec![40.0]),   // MS1 cycle 1
            mk(2, 2.0, (400.0, 410.0), vec![405.0], vec![50.0]), // win1
            mk(2, 2.5, (410.0, 420.0), vec![415.0], vec![60.0]), // win2
        ];
        let mzml = MzMLData {
            path: "test.mzml".into(),
            spectra,
            isolation_windows: vec![(400.0, 410.0), (410.0, 420.0)],
            ms1_indices: vec![0, 3],
            ms2_indices: vec![1, 2, 4, 5],
        };

        let dia = DiaData::from_mzml(&mzml).unwrap();

        assert_eq!(dia.frame_max_index, 6);
        assert_eq!(dia.cycle_len, 3);
        assert_eq!(dia.scan_max_index, 2);
        assert!(!dia.has_mobility);
        assert!(dia.has_ms1);
        assert_eq!(dia.mobility_values, vec![1e-6f64, 0.0]);

        // cycle_mz is (cycle_len × scan_max, 2): MS1 sentinel + two windows,
        // each duplicated into the two synthetic scan bins.
        assert_eq!(dia.cycle_mz.len(), 6);
        assert_eq!(dia.cycle_mz[0], [-1.0, -1.0]);
        assert_eq!(dia.cycle_mz[1], [-1.0, -1.0]);
        assert_eq!(dia.cycle_mz[2], [400.0, 410.0]);
        assert_eq!(dia.cycle_mz[3], [400.0, 410.0]);
        assert_eq!(dia.cycle_mz[4], [410.0, 420.0]);
        assert_eq!(dia.cycle_mz[5], [410.0, 420.0]);
        assert_eq!(dia.dia_precursor_cycle, vec![0, 0, 1, 1, 2, 2]);

        // Each of the 6 peaks is duplicated into 2 scan bins.
        let total: u64 = dia.intensity_values.iter().map(|&v| v as u64).sum();
        assert_eq!(total, 2 * (10 + 20 + 30 + 40 + 50 + 60));
        assert_eq!(dia.push_indices.len(), dia.intensity_values.len());
        assert_eq!(dia.tof_indptr.len(), dia.tof_max_index + 1);
    }

    #[test]
    fn searchsorted_helpers() {
        let asc = vec![0.0f64, 1.0, 2.0, 3.0];
        assert_eq!(searchsorted_left(&asc, 1.5), 2);
        assert_eq!(searchsorted_left(&asc, 2.0), 2);

        // descending mobility: scan 0 = highest 1/K0
        let desc = vec![1.7f64, 1.5, 1.3, 1.1];
        // count of elements <= v, matching searchsorted_right(desc[::-1], v)
        assert_eq!(searchsorted_right_reversed(&desc, 1.4), 2);
        assert_eq!(searchsorted_right_reversed(&desc, 1.6), 3);
    }
}

