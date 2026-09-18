//! Peak-group candidate selection (the "NG" backend selection).
//!
//! Faithful port of `peak_group_selection` from the reference search engine:
//! for each precursor, extract a per-fragment dense XIC over a wide RT window,
//! smooth each row with a 1-D Gaussian, score each cycle by the
//! library-intensity-weighted log XIC (`axis_log_dot_product`), find strict
//! two-point local maxima, and emit the top `candidate_count` fixed-width
//! windows. This replaces the classic abundance-based argmax selection, which
//! picks co-eluting interference instead of the peptide apex.

use ndarray::Array2;
use rayon::prelude::*;

use crate::engine::data::{DiaData, SpecLibFlat};

use crate::engine::util::{axis_log_dot_product, convolve_1d, find_local_maxima, GaussianKernel};
use crate::engine::ng::{cycle_rt_values, filter_fragments, get_cycle_idx_limits};

/// Peak-group selection parameters (`SelectionParameters`).
#[derive(Debug, Clone)]
pub struct NgSelectionConfig {
    /// FWHM of the RT Gaussian smoothing kernel, seconds (3.0).
    pub fwhm_rt: f64,
    /// Gaussian kernel width in cycles (20).
    pub kernel_size: usize,
    /// Candidate window half-width in cycles (±5).
    pub peak_length: usize,
    /// Fragment m/z extraction tolerance, ppm (7.0).
    pub mass_tolerance: f64,
    /// RT window around the predicted RT, seconds (200.0).
    pub rt_tolerance: f64,
    /// Number of local-maximum candidates per precursor (3).
    pub candidate_count: usize,
    /// Top-k fragments by intensity used for selection (12).
    pub top_k_fragments: usize,
}

impl Default for NgSelectionConfig {
    fn default() -> Self {
        Self {
            fwhm_rt: 3.0,
            kernel_size: 20,
            peak_length: 5,
            mass_tolerance: 7.0,
            rt_tolerance: 200.0,
            candidate_count: 3,
            top_k_fragments: 12,
        }
    }
}

/// One selected peak-group candidate.
#[derive(Debug, Clone)]
pub struct NgCandidate {
    pub precursor_idx: u32,
    pub rank: u8,
    pub score: f32,
    pub cycle_start: usize,
    pub cycle_center: usize,
    pub cycle_stop: usize,
    pub decoy: u8,
    pub elution_group_idx: u32,
}

/// Peak-group candidate selection engine.
pub struct NgSelection {
    kernel: GaussianKernel,
    config: NgSelectionConfig,
}

impl NgSelection {
    pub fn new(config: NgSelectionConfig) -> Self {
        let kernel = GaussianKernel::new(config.fwhm_rt, 1.0, config.kernel_size, 1.0);
        Self { kernel, config }
    }

    /// Select peak-group candidates for every precursor.
    pub fn run(&self, dia_data: &DiaData, spec: &SpecLibFlat) -> Vec<NgCandidate> {
        let cycle_rt = cycle_rt_values(dia_data);
        let mut candidates: Vec<NgCandidate> = (0..spec.n_precursors())
            .into_par_iter()
            .flat_map_iter(|i| self.search_precursor(dia_data, spec, &cycle_rt, i))
            .collect();
        // `flat_map_iter` + `collect` does not preserve candidate order under
        // rayon, and the FDR MLP trains on a seed-fixed shuffle of the first
        // `max_train` feature rows — so an unstable order makes the whole run
        // non-reproducible. Sort by (precursor, rank) to fix a canonical order.
        candidates.sort_unstable_by_key(|c| (c.precursor_idx, c.rank));
        candidates
    }

    fn search_precursor(
        &self,
        dia_data: &DiaData,
        spec: &SpecLibFlat,
        cycle_rt: &[f64],
        i: usize,
    ) -> Vec<NgCandidate> {
        let p = &spec.precursor_df;
        let precursor_idx = p.precursor_idx[i];
        let precursor_mz = p.precursor_mz[i];
        let rt = p.rt_pred[i];
        let decoy = p.decoy[i];
        let elution_group_idx = p.elution_group_idx[i];
        let start = p.flat_frag_start_idx[i] as usize;
        let stop = p.flat_frag_stop_idx[i] as usize;

        let frag = filter_fragments(&spec.fragment_df, start, stop, self.config.top_k_fragments);
        if frag.mz.is_empty() {
            return Vec::new();
        }

        let (cycle_start_idx, cycle_stop_idx) = get_cycle_idx_limits(
            cycle_rt,
            rt,
            self.config.rt_tolerance,
            self.config.kernel_size,
        );
        if cycle_stop_idx <= cycle_start_idx {
            return Vec::new();
        }

        let xic = extract_xic(
            dia_data,
            cycle_start_idx,
            cycle_stop_idx,
            &frag.mz,
            self.config.mass_tolerance,
            precursor_mz,
        );

        let convolved = convolve_1d(&xic, &self.kernel.array);
        let score = axis_log_dot_product(&convolved, &frag.intensity);

        let (indices, values) = find_local_maxima(&score, cycle_start_idx);

        let max_count = self.config.candidate_count.min(indices.len());
        let mut out = Vec::with_capacity(max_count);
        for k in 0..max_count {
            let center = indices[k];
            let cycle_start = center.saturating_sub(self.config.peak_length);
            let cycle_stop = (center + self.config.peak_length + 1).min(cycle_rt.len());
            out.push(NgCandidate {
                precursor_idx,
                rank: k as u8,
                score: values[k],
                cycle_start,
                cycle_center: center,
                cycle_stop,
                decoy,
                elution_group_idx,
            });
        }
        out
    }
}

/// Extract the per-fragment dense XIC over `[cycle_start, cycle_stop)` cycles.
///
/// `get_dense_intensity` with a single scan bin (`scan_limits = [0, 1, 1]`)
/// gives `(1, n_frag, 1, n_cycles)`; for mzML the synthetic 2-bin mobility
/// holds every peak once in bin 0, so summing one bin yields the true XIC.
pub(crate) fn extract_xic(
    dia_data: &DiaData,
    cycle_start: usize,
    cycle_stop: usize,
    fragment_mz: &[f64],
    mass_tolerance: f64,
    precursor_mz: f64,
) -> Array2<f32> {
    let zf = dia_data.zeroth_frame as i64;
    let cycle_len = dia_data.cycle_len as i64;
    let frame_limits = [
        cycle_start as i64 * cycle_len + zf,
        cycle_stop as i64 * cycle_len + zf,
        1,
    ];
    let scan_limits = [0i64, 1, 1];
    let quadrupole_mz = [[precursor_mz, precursor_mz]];

    let (dense, _uniq) = dia_data.get_dense_intensity(
        frame_limits,
        scan_limits,
        fragment_mz,
        mass_tolerance,
        &quadrupole_mz,
    );

    let n_frag = fragment_mz.len();
    let n_cycles = cycle_stop - cycle_start;
    let mut xic = Array2::<f32>::zeros((n_frag, n_cycles));
    for f in 0..n_frag {
        for c in 0..n_cycles {
            xic[[f, c]] = dense[[0, f, 0, c]];
        }
    }
    xic
}
