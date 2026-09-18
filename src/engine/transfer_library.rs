//! Transfer-library accumulation + MS2 quality control (second pass, part 2).
//!
//! Aggregates the per-run requantified PSMs from `transfer_quant` into a single
//! transfer library: top-k samples per modified peptide, delta-max RT
//! normalization, and the MS2 QC that zeroes poorly-correlated fragments. The
//! output is consumed by the PeptDeep fine-tuner (RT + MS2 models) and, via the
//! re-predicted library, the second-pass search.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::engine::transfer_quant::QuantPsm;
use crate::log::info;
use crate::model::chemistry::{F_TYPE_B, F_TYPE_Y};

/// Transfer-library building parameters (reference `default.yaml` L415-441).
#[derive(Debug, Clone)]
pub struct TransferLibraryConfig {
    /// Samples kept per modified peptide (`top_k_samples`).
    pub top_k_samples: usize,
    /// Precursor median-fragment-correlation cutoff for `use_for_ms2`.
    pub precursor_correlation_cutoff: f32,
    /// Fragments below `median * ratio` correlation are zeroed.
    pub fragment_correlation_ratio: f32,
}

impl Default for TransferLibraryConfig {
    fn default() -> Self {
        Self {
            top_k_samples: 3,
            precursor_correlation_cutoff: 0.5,
            fragment_correlation_ratio: 0.75,
        }
    }
}

/// One precursor row of the transfer library (feeds the fine-tuner's
/// `precursor_df`).
#[derive(Debug, Clone)]
pub struct TransferPrecursorRow {
    pub precursor_idx: u32,
    /// Naked *uppercase* sequence (alphabase `sequence`); mods carried in
    /// `mods` / `mod_sites`.
    pub sequence: String,
    /// `;`-separated modification names, `""` if unmodified.
    pub mods: String,
    /// `;`-separated 1-indexed modification sites, `""` if unmodified.
    pub mod_sites: String,
    pub charge: u8,
    pub n_aa: u32,
    /// `rt_observed / max(rt_observed)` (`normalize_rt_max`).
    pub rt_norm: f32,
    pub proba: f32,
    /// Row range into `TransferLibrary::fragments` (wide, `n_aa - 1` rows).
    pub frag_start_idx: u32,
    pub frag_stop_idx: u32,
    /// `median_correlation > precursor_correlation_cutoff`.
    pub use_for_ms2: u8,
    pub median_correlation: f32,
}

/// The transfer library: precursor rows + a wide fragment-intensity matrix.
#[derive(Debug, Clone, Default)]
pub struct TransferLibrary {
    pub precursors: Vec<TransferPrecursorRow>,
    /// Wide matrix, columns `[b_z1, b_z2, y_z1, y_z2]`. Row `r` belongs to the
    /// precursor whose `[frag_start_idx, frag_stop_idx)` contains `r`. Within a
    /// precursor's block, row `k` holds `b_{k+1}` and `y_{n_aa-k-1}` (alphabase
    /// dense-matrix convention: b forward, y reversed).
    pub fragments: Vec<[f32; 4]>,
}

/// Median of a float slice (sorts a copy).
fn median_sorted(mut vals: Vec<f32>) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = vals.len();
    if n % 2 == 0 {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    }
}

/// Build the transfer library from the concatenated requantified PSMs of all
/// runs (one `QuantPsm` per retained PSM).
pub fn build_transfer_library(all_psms: &[QuantPsm], config: &TransferLibraryConfig) -> TransferLibrary {
    // 1. top-k per modified peptide (sequence + mods + mod_sites), proba asc.
    let mut groups: HashMap<(String, String, String), Vec<usize>> = HashMap::new();
    for (i, psm) in all_psms.iter().enumerate() {
        groups
            .entry((psm.sequence.clone(), psm.mods.clone(), psm.mod_sites.clone()))
            .or_default()
            .push(i);
    }
    let mut retained: Vec<usize> = Vec::new();
    for idxs in groups.values_mut() {
        idxs.sort_by(|&a, &b| {
            all_psms[a]
                .proba
                .partial_cmp(&all_psms[b].proba)
                .unwrap_or(Ordering::Equal)
                .then_with(|| all_psms[a].precursor_idx.cmp(&all_psms[b].precursor_idx))
        });
        idxs.truncate(config.top_k_samples);
        retained.extend(idxs.iter().copied());
    }
    retained.sort_unstable();

    // 2. global max observed RT (for `normalize_rt_max`).
    let rt_max = retained
        .iter()
        .map(|&i| all_psms[i].rt_observed)
        .fold(0.0f32, f32::max);

    let mut lib = TransferLibrary::default();
    lib.precursors.reserve(retained.len());

    for &pi in &retained {
        let psm = &all_psms[pi];
        let n_aa = psm.n_aa;
        if n_aa < 2 {
            continue;
        }

        // (ion_type, series_number, charge) -> (intensity, correlation).
        let mut lut: HashMap<(u8, u16, u8), (f32, f32)> = HashMap::with_capacity(psm.fragments.len());
        for f in &psm.fragments {
            lut.insert((f.ion_type, f.series_number, f.charge), (f.intensity, f.correlation));
        }

        // MS2 QC: median correlation over fragments with intensity > 0; zero out
        // fragments whose correlation <= median * ratio.
        let corrs: Vec<f32> = psm
            .fragments
            .iter()
            .filter(|f| f.intensity > 0.0)
            .map(|f| f.correlation)
            .collect();
        let median_corr = median_sorted(corrs);
        let use_for_ms2 = (median_corr > config.precursor_correlation_cutoff) as u8;
        let thr = median_corr * config.fragment_correlation_ratio;

        let frag_start_idx = lib.fragments.len() as u32;
        for k in 0..n_aa - 1 {
            let b_num = (k + 1) as u16;
            let y_num = (n_aa - k - 1) as u16;
            let get = |ion_type: u8, series: u16, charge: u8| -> f32 {
                match lut.get(&(ion_type, series, charge)) {
                    Some(&(intensity, correlation)) if correlation > thr => intensity,
                    _ => 0.0,
                }
            };
            lib.fragments.push([
                get(F_TYPE_B, b_num, 1),
                get(F_TYPE_B, b_num, 2),
                get(F_TYPE_Y, y_num, 1),
                get(F_TYPE_Y, y_num, 2),
            ]);
        }
        let frag_stop_idx = lib.fragments.len() as u32;

        let rt_norm = if rt_max > 0.0 { psm.rt_observed / rt_max } else { 0.0 };

        lib.precursors.push(TransferPrecursorRow {
            precursor_idx: psm.precursor_idx,
            sequence: psm.sequence.clone(),
            mods: psm.mods.clone(),
            mod_sites: psm.mod_sites.clone(),
            charge: psm.charge,
            n_aa: n_aa as u32,
            rt_norm,
            proba: psm.proba,
            frag_start_idx,
            frag_stop_idx,
            use_for_ms2,
            median_correlation: median_corr,
        });
    }

    info!(
        "transfer library: {} precursors, {} fragment rows (top_k={})",
        lib.precursors.len(),
        lib.fragments.len(),
        config.top_k_samples
    );
    lib
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::transfer_quant::{QuantFragment, QuantPsm};

    fn psm(seq: &str, charge: u8, proba: f32, rt: f32) -> QuantPsm {
        QuantPsm {
            precursor_idx: 1,
            proba,
            qval: 0.01,
            rt_observed: rt,
            mz_observed: 500.0,
            sequence: seq.to_string(),
            mods: String::new(),
            mod_sites: String::new(),
            charge,
            n_aa: seq.len(),
            fragments: Vec::new(),
        }
    }

    #[test]
    fn top_k_selection_by_proba() {
        let psms = vec![
            psm("AAAA", 2, 0.9, 100.0),
            psm("AAAA", 3, 0.5, 200.0),
            psm("AAAA", 2, 0.7, 150.0),
            psm("AAAA", 2, 0.1, 300.0),
        ];
        let lib = build_transfer_library(&psms, &TransferLibraryConfig::default());
        // 4 samples, 1 unique peptide, keep top-3.
        assert_eq!(lib.precursors.len(), 3);
        // best proba retained.
        assert!(lib.precursors.iter().any(|p| p.proba == 0.1));
    }

    #[test]
    fn rt_norm_is_scaled_by_max() {
        let psms = vec![
            psm("AAAA", 2, 0.5, 100.0),
            psm("BBBB", 2, 0.5, 400.0),
        ];
        let lib = build_transfer_library(&psms, &TransferLibraryConfig::default());
        let a = lib.precursors.iter().find(|p| p.sequence == "AAAA").unwrap();
        let b = lib.precursors.iter().find(|p| p.sequence == "BBBB").unwrap();
        assert!((a.rt_norm - 0.25).abs() < 1e-5);
        assert!((b.rt_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn wide_fragment_layout_b_forward_y_reversed() {
        // "AAA" (n_aa=3) → 2 rows: k=0 → b1/y2, k=1 → b2/y1.
        let mut p = psm("AAA", 2, 0.5, 100.0);
        p.fragments = vec![
            // b1 z1 intensity 10, y2 z1 intensity 20 (y reversed: row 0 = y2)
            QuantFragment { mz: 100.0, observed_mz: 100.0, ion_type: F_TYPE_B, series_number: 1, charge: 1, intensity: 10.0, correlation: 0.9 },
            QuantFragment { mz: 200.0, observed_mz: 200.0, ion_type: F_TYPE_Y, series_number: 2, charge: 1, intensity: 20.0, correlation: 0.9 },
            QuantFragment { mz: 150.0, observed_mz: 150.0, ion_type: F_TYPE_B, series_number: 2, charge: 1, intensity: 30.0, correlation: 0.9 },
            QuantFragment { mz: 250.0, observed_mz: 250.0, ion_type: F_TYPE_Y, series_number: 1, charge: 1, intensity: 40.0, correlation: 0.9 },
        ];
        let lib = build_transfer_library(std::slice::from_ref(&p), &TransferLibraryConfig::default());
        assert_eq!(lib.precursors.len(), 1);
        assert_eq!(lib.precursors[0].frag_start_idx, 0);
        assert_eq!(lib.precursors[0].frag_stop_idx, 2);
        assert_eq!(lib.fragments.len(), 2);
        // row 0 (k=0): b_z1 = b1 = 10, y_z1 = y2 = 20
        assert!((lib.fragments[0][0] - 10.0).abs() < 1e-5, "b_z1[0]={}", lib.fragments[0][0]);
        assert!((lib.fragments[0][2] - 20.0).abs() < 1e-5, "y_z1[0]={}", lib.fragments[0][2]);
        // row 1 (k=1): b_z1 = b2 = 30, y_z1 = y1 = 40
        assert!((lib.fragments[1][0] - 30.0).abs() < 1e-5, "b_z1[1]={}", lib.fragments[1][0]);
        assert!((lib.fragments[1][2] - 40.0).abs() < 1e-5, "y_z1[1]={}", lib.fragments[1][2]);
    }
}
