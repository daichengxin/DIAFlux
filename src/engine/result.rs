//! Search-result types shared by the DIA engine.
//!
//! `CoreResult` is the end-to-end output of one search run: retained PSMs plus
//! quantification at precursor / peptide / protein levels (currently stubbed —
//! the benchmark goal is the PSM count).

use ndarray::Array2;

/// One quantification level: group ids + linear intensity matrix
/// (`n_groups × n_samples`, missing → 0).
pub struct QuantLevel {
    pub group_ids: Vec<u64>,
    pub intensity: Array2<f32>,
}

/// Quantification results at precursor / peptide / protein levels.
#[derive(Default)]
pub struct QuantOutput {
    pub precursor: Option<QuantLevel>,
    pub peptide: Option<QuantLevel>,
    pub protein: Option<QuantLevel>,
}

/// End-to-end result of one search run: retained PSMs + quantification output.
pub struct CoreResult {
    /// Number of PSMs passing FDR (targets only).
    pub n_psm: usize,
    /// Retained precursor ids (targets, `qval <= fdr_level`).
    ///
    /// NOTE: these are `Precursor::pr_index` values — the *raw* pre-dedup
    /// library row index, not positions into `SpectralLibrary::precursors`
    /// (which is deduplicated by `pr_id`). Map back via `pr_index`.
    pub precursor_idx: Vec<u32>,
    pub qval: Vec<f32>,
    pub proba: Vec<f32>,
    /// Observed retention time (s) of each retained PSM's matched apex
    /// (`FEAT_RT_OBSERVED`); aligned with `precursor_idx`/`qval`/`proba`.
    pub rt_observed: Vec<f32>,
    /// Quantification at precursor / peptide / protein levels.
    pub quant: QuantOutput,
}
