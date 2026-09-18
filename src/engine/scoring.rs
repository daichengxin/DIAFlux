//! Peak-group candidate scoring (the "NG" backend scoring).
//!
//! Faithful port of `peak_group_scoring` from the reference search engine: 40
//! f32 features per candidate, computed from the per-fragment dense XIC and the
//! intensity-weighted absolute m/z over the candidate window. This is the
//! discriminating feature set (per-fragment correlation with the normalized
//! median profile, hyperscores, ion-ladder completeness, IDF-weighted scores)
//! that the runtime-trained FDR MLP consumes.

use ndarray::Array2;
use rayon::prelude::*;

use crate::engine::data::{DiaData, SpecLibFlat};

use crate::engine::util::{correlation_safe, median, std, InverseDocumentFrequency};
use crate::engine::ng::{filter_fragments, FRAG_TYPE_B, FRAG_TYPE_Y};
use crate::engine::selection::NgCandidate;

/// Number of NG scoring features.
pub const NG_FEATURE_COUNT: usize = 40;

/// Feature names in feature-array order (`FEATURE_NAMES` minus the leading
/// `score` metadata column).
pub const NG_FEATURE_NAMES: [&str; NG_FEATURE_COUNT] = [
    "mean_correlation",
    "median_correlation",
    "correlation_std",
    "intensity_correlation",
    "num_fragments",
    "num_scans",
    "num_over_95",
    "num_over_90",
    "num_over_80",
    "num_over_50",
    "num_over_0",
    "num_over_0_rank_0_5",
    "num_over_0_rank_6_11",
    "num_over_0_rank_12_17",
    "num_over_0_rank_18_23",
    "num_over_50_rank_0_5",
    "num_over_50_rank_6_11",
    "num_over_50_rank_12_17",
    "num_over_50_rank_18_23",
    "hyperscore_intensity_observation",
    "hyperscore_intensity_library",
    "hyperscore_inverse_mass_error",
    "rt_observed",
    "delta_rt",
    "longest_b_series",
    "longest_y_series",
    "naa",
    "weighted_mass_error",
    "log10_b_ion_intensity",
    "log10_y_ion_intensity",
    "fwhm_rt",
    "idf_hyperscore",
    "idf_xic_dot_product",
    "idf_intensity_dot_product",
    "median_profile_sum",
    "median_profile_sum_filtered",
    "num_profiles",
    "num_profiles_filtered",
    "num_over_0_top6_idf",
    "num_over_50_top6_idf",
];

/// Peak-group scoring parameters (`ScoringParameters`).
#[derive(Debug, Clone)]
pub struct NgScoringConfig {
    /// Fragment m/z extraction tolerance, ppm (7.0).
    pub mass_tolerance: f64,
    /// Top-k fragments by intensity used for scoring (100).
    pub top_k_fragments: usize,
}

impl Default for NgScoringConfig {
    fn default() -> Self {
        Self {
            mass_tolerance: 7.0,
            top_k_fragments: 100,
        }
    }
}

/// Collected scoring outputs (one row per candidate).
pub struct NgScoringResult {
    /// `n_candidates × NG_FEATURE_COUNT` feature matrix.
    pub features: Array2<f32>,
    pub precursor_idx: Vec<u32>,
    pub decoy: Vec<u8>,
    pub elution_group_idx: Vec<u32>,
}

/// Peak-group scoring engine.
pub struct NgScoring {
    config: NgScoringConfig,
}

impl NgScoring {
    pub fn new(config: NgScoringConfig) -> Self {
        Self { config }
    }

    /// Score every candidate against the shared `DiaData`.
    pub fn score(
        &self,
        dia_data: &DiaData,
        spec: &SpecLibFlat,
        idf: &InverseDocumentFrequency,
        cycle_rt: &[f64],
        candidates: &[NgCandidate],
    ) -> NgScoringResult {
        let n = candidates.len();

        // `DiaData`, `SpecLibFlat`, and `InverseDocumentFrequency` are all
        // immutable shared state here, so scoring is embarrassingly parallel.
        // Write each row directly into a pre-allocated flat buffer via disjoint
        // mutable chunks (no giant intermediate collection) to keep peak memory
        // at the feature matrix itself rather than doubling it.
        let mut flat = vec![0.0f32; n * NG_FEATURE_COUNT];
        flat.par_chunks_mut(NG_FEATURE_COUNT)
            .enumerate()
            .for_each(|(i, row)| {
                let c = &candidates[i];
                let scored = self.score_candidate(dia_data, spec, idf, cycle_rt, c);
                row.copy_from_slice(&scored);
            });
        let features = Array2::from_shape_vec((n, NG_FEATURE_COUNT), flat)
            .expect("feature buffer length must match (n, NG_FEATURE_COUNT)");

        // Precursor / decoy / elution-group bookkeeping is just a projection of
        // `candidates`; build it cheaply after the parallel section.
        let mut precursor_idx = Vec::with_capacity(n);
        let mut decoy = Vec::with_capacity(n);
        let mut elution_group_idx = Vec::with_capacity(n);
        for c in candidates {
            precursor_idx.push(c.precursor_idx);
            decoy.push(c.decoy);
            elution_group_idx.push(c.elution_group_idx);
        }

        NgScoringResult {
            features,
            precursor_idx,
            decoy,
            elution_group_idx,
        }
    }

    fn score_candidate(
        &self,
        dia_data: &DiaData,
        spec: &SpecLibFlat,
        idf: &InverseDocumentFrequency,
        cycle_rt: &[f64],
        candidate: &NgCandidate,
    ) -> [f32; NG_FEATURE_COUNT] {
        let cycle_start_idx = candidate.cycle_start;
        let cycle_stop_idx = candidate.cycle_stop;

        // Precursor metadata.
        let Some(&row) = spec.precursor_index_map.get(&candidate.precursor_idx) else {
            return [0.0; NG_FEATURE_COUNT];
        };
        let p = &spec.precursor_df;
        let precursor_mz = p.precursor_mz[row];
        let precursor_rt = p.rt_pred[row];
        let naa = p.sequence[row].len().min(u8::MAX as usize) as u8;
        let frag_start = p.flat_frag_start_idx[row] as usize;
        let frag_stop = p.flat_frag_stop_idx[row] as usize;

        let frag = filter_fragments(
            &spec.fragment_df,
            frag_start,
            frag_stop,
            self.config.top_k_fragments,
        );
        if frag.mz.is_empty() {
            return [0.0; NG_FEATURE_COUNT];
        }

        // Dense XIC + intensity-weighted absolute m/z over the window.
        let (dense_xic, dense_mz) = extract_xic_mz(
            dia_data,
            cycle_start_idx,
            cycle_stop_idx,
            &frag.mz,
            self.config.mass_tolerance,
            precursor_mz,
        );

        let normalized_xic = normalize_profiles(&dense_xic, 1);
        let filtered_xic = filter_non_zero(&normalized_xic);

        let median_profile = median_axis_0(&normalized_xic);
        let median_profile_filtered = median_axis_0(&filtered_xic);

        let num_profiles = normalized_xic.nrows() as f32;
        let num_profiles_filtered = filtered_xic.nrows() as f32;

        let median_profile_sum: f32 = median_profile.iter().sum();
        let median_profile_sum_filtered: f32 = median_profile_filtered.iter().sum();

        let fwhm_rt = calculate_fwhm_rt(&median_profile_filtered, cycle_start_idx, cycle_rt);

        let correlations = correlation_axis_0(&median_profile_filtered, &normalized_xic);

        let observation_intensities = sum_rows(&dense_xic);

        let intensity_correlation =
            correlation_safe(&observation_intensities, &frag.intensity);

        let mean_correlation = if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f32>() / correlations.len() as f32
        };
        let median_correlation = median(&correlations);
        let correlation_std = std(&correlations);

        let num_over_95 = count_values_above(&correlations, 0.95, None);
        let num_over_90 = count_values_above(&correlations, 0.90, None);
        let num_over_80 = count_values_above(&correlations, 0.80, None);
        let num_over_50 = count_values_above(&correlations, 0.50, None);
        let num_over_0 = count_values_above(&correlations, 0.0, None);

        let mask_0_5 = create_ranked_mask(&frag.intensity, 0, 6);
        let mask_6_11 = create_ranked_mask(&frag.intensity, 6, 12);
        let mask_12_17 = create_ranked_mask(&frag.intensity, 12, 18);
        let mask_18_23 = create_ranked_mask(&frag.intensity, 18, 24);

        let num_over_0_rank_0_5 = count_values_above(&correlations, 0.0, Some(&mask_0_5));
        let num_over_0_rank_6_11 = count_values_above(&correlations, 0.0, Some(&mask_6_11));
        let num_over_0_rank_12_17 = count_values_above(&correlations, 0.0, Some(&mask_12_17));
        let num_over_0_rank_18_23 = count_values_above(&correlations, 0.0, Some(&mask_18_23));

        let num_over_50_rank_0_5 = count_values_above(&correlations, 0.50, Some(&mask_0_5));
        let num_over_50_rank_6_11 = count_values_above(&correlations, 0.50, Some(&mask_6_11));
        let num_over_50_rank_12_17 = count_values_above(&correlations, 0.50, Some(&mask_12_17));
        let num_over_50_rank_18_23 = count_values_above(&correlations, 0.50, Some(&mask_18_23));

        let num_fragments = frag.mz.len() as f32;
        let num_scans = (cycle_stop_idx - cycle_start_idx) as f32;

        let matched_mask: Vec<bool> = observation_intensities.iter().map(|&x| x > 0.0).collect();

        let hyperscore_intensity_observation = calculate_hyperscore(
            &frag.fragment_type,
            &observation_intensities,
            &matched_mask,
        );
        let hyperscore_intensity_library =
            calculate_hyperscore(&frag.fragment_type, &frag.intensity, &matched_mask);

        let (longest_b_series, longest_y_series) = calculate_longest_ion_series(
            &frag.fragment_type,
            &frag.fragment_number,
            &matched_mask,
        );

        // Fragment m/z + mass errors (against the extraction target m/z).
        let fragment_mz_f32: Vec<f32> = frag.mz.iter().map(|&x| x as f32).collect();
        let (_fragment_mz_observed, fragment_mass_errors) =
            calculate_fragment_mz_and_errors(&dense_mz, &dense_xic, &fragment_mz_f32);

        let weighted_mass_error =
            calculate_weighted_mean_absolute_error(&fragment_mass_errors, &frag.intensity);

        let hyperscore_inverse_mass_error = calculate_hyperscore_inverse_mass_error(
            &frag.fragment_type,
            &observation_intensities,
            &matched_mask,
            &fragment_mass_errors,
        );

        let rt_observed = cycle_rt[candidate.cycle_center] as f32;
        let delta_rt = rt_observed - precursor_rt as f32;

        let intensity_b_raw = intensity_ion_series(
            &frag.fragment_type,
            &observation_intensities,
            &matched_mask,
            FRAG_TYPE_B,
        );
        let intensity_y_raw = intensity_ion_series(
            &frag.fragment_type,
            &observation_intensities,
            &matched_mask,
            FRAG_TYPE_Y,
        );

        const EPSILON: f32 = 1e-8;
        let log10_b_ion_intensity = (intensity_b_raw + EPSILON).log10();
        let log10_y_ion_intensity = (intensity_y_raw + EPSILON).log10();

        let idf_values = idf.get_idf(&frag.mz_library);

        let idf_hyperscore = calculate_hyperscore(&frag.fragment_type, &idf_values, &matched_mask);
        let idf_xic_dot_product = dot_product(&idf_values, &correlations);
        let idf_intensity_dot_product = dot_product(&idf_values, &observation_intensities);
        let log_idf_intensity_dot_product = (idf_intensity_dot_product + EPSILON).log10();

        let mask_top6_idf = create_ranked_mask(&idf_values, 0, 6);
        let num_over_0_top6_idf = count_values_above(&correlations, 0.0, Some(&mask_top6_idf));
        let num_over_50_top6_idf = count_values_above(&correlations, 0.50, Some(&mask_top6_idf));

        [
            mean_correlation,
            median_correlation,
            correlation_std,
            intensity_correlation,
            num_fragments,
            num_scans,
            num_over_95 as f32,
            num_over_90 as f32,
            num_over_80 as f32,
            num_over_50 as f32,
            num_over_0 as f32,
            num_over_0_rank_0_5 as f32,
            num_over_0_rank_6_11 as f32,
            num_over_0_rank_12_17 as f32,
            num_over_0_rank_18_23 as f32,
            num_over_50_rank_0_5 as f32,
            num_over_50_rank_6_11 as f32,
            num_over_50_rank_12_17 as f32,
            num_over_50_rank_18_23 as f32,
            hyperscore_intensity_observation,
            hyperscore_intensity_library,
            hyperscore_inverse_mass_error,
            rt_observed,
            delta_rt,
            longest_b_series as f32,
            longest_y_series as f32,
            naa as f32,
            weighted_mass_error,
            log10_b_ion_intensity,
            log10_y_ion_intensity,
            fwhm_rt,
            idf_hyperscore,
            idf_xic_dot_product,
            log_idf_intensity_dot_product,
            median_profile_sum,
            median_profile_sum_filtered,
            num_profiles,
            num_profiles_filtered,
            num_over_0_top6_idf as f32,
            num_over_50_top6_idf as f32,
        ]
    }
}

// ===========================================================================
// Dense extraction (XIC + intensity-weighted absolute m/z)
// ===========================================================================

/// Extract `(dense_xic, dense_mz)` over `[cycle_start, cycle_stop)`.
fn extract_xic_mz(
    dia_data: &DiaData,
    cycle_start: usize,
    cycle_stop: usize,
    fragment_mz: &[f64],
    mass_tolerance: f64,
    precursor_mz: f64,
) -> (Array2<f32>, Array2<f32>) {
    let n_frag = fragment_mz.len();
    let n_cycles = cycle_stop - cycle_start;
    let mut xic = Array2::<f32>::zeros((n_frag, n_cycles));
    let mut mz = Array2::<f32>::zeros((n_frag, n_cycles));

    let zf = dia_data.zeroth_frame as i64;
    let cycle_len = dia_data.cycle_len as i64;
    let frame_limits = [
        cycle_start as i64 * cycle_len + zf,
        cycle_stop as i64 * cycle_len + zf,
        1,
    ];
    let scan_limits = [0i64, 1, 1];
    let quadrupole_mz = [[precursor_mz, precursor_mz]];

    let (dense, _uniq) = dia_data.get_dense(
        frame_limits,
        scan_limits,
        fragment_mz,
        mass_tolerance,
        &quadrupole_mz,
        true,
    );

    // Empty dense result when the quadrupole window matched no cycles.
    if dense.shape()[2] == 0 {
        return (xic, mz);
    }

    for f in 0..n_frag {
        for c in 0..n_cycles {
            xic[[f, c]] = dense[[0, f, 0, 0, c]];
            mz[[f, c]] = dense[[1, f, 0, 0, c]];
        }
    }
    (xic, mz)
}

// ===========================================================================
// Profile normalization / aggregation
// ===========================================================================

/// Filter out all-zero rows of a 2-D array (`filter_non_zero`).
fn filter_non_zero(array: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = array.dim();
    let non_zero_rows: Vec<usize> = (0..rows)
        .filter(|&i| array.row(i).iter().any(|&v| v != 0.0))
        .collect();
    if non_zero_rows.is_empty() {
        return Array2::zeros((0, cols));
    }
    let mut filtered = Array2::zeros((non_zero_rows.len(), cols));
    for (new_idx, &old_idx) in non_zero_rows.iter().enumerate() {
        filtered.row_mut(new_idx).assign(&array.row(old_idx));
    }
    filtered
}

/// Median along axis 0 (across rows) per column (`median_axis_0`).
fn median_axis_0(array: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = array.dim();
    if rows == 0 {
        return vec![0.0; cols];
    }
    let mut result = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut column: Vec<f32> = (0..rows).map(|r| array[[r, col]]).collect();
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let v = if rows % 2 == 0 {
            (column[rows / 2 - 1] + column[rows / 2]) / 2.0
        } else {
            column[rows / 2]
        };
        result.push(v);
    }
    result
}

/// Normalize each fragment row by its mean intensity over the center ±
/// `center_dilations` cycles (`normalize_profiles`); rows with zero center stay
/// zero.
fn normalize_profiles(intensity_slice: &Array2<f32>, center_dilations: usize) -> Array2<f32> {
    let (rows, cols) = intensity_slice.dim();
    let center_idx = cols / 2;

    let mut center_intensity = Vec::with_capacity(rows);
    for i in 0..rows {
        let start = center_idx.saturating_sub(center_dilations);
        let end = (center_idx + center_dilations + 1).min(cols);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for j in start..end {
            sum += intensity_slice[[i, j]];
            count += 1;
        }
        center_intensity.push(if count > 0 { sum / count as f32 } else { 0.0 });
    }

    let mut normalized = Array2::zeros((rows, cols));
    for i in 0..rows {
        if center_intensity[i] > 0.0 {
            for j in 0..cols {
                normalized[[i, j]] = intensity_slice[[i, j]] / center_intensity[i];
            }
        }
    }
    normalized
}

/// Correlation of each row with the (filtered) median profile
/// (`correlation_axis_0`).
fn correlation_axis_0(median_profile: &[f32], dense_xic: &Array2<f32>) -> Vec<f32> {
    let rows = dense_xic.nrows();
    let mut correlations = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_data: Vec<f32> = dense_xic.row(row).to_vec();
        correlations.push(correlation_safe(median_profile, &row_data));
    }
    correlations
}

/// Sum each row of a 2-D array (axis 1) into a Vec.
fn sum_rows(array: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = array.dim();
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut s = 0.0f32;
        for c in 0..cols {
            s += array[[r, c]];
        }
        out.push(s);
    }
    out
}

// ===========================================================================
// Hyperscore / ion series
// ===========================================================================

/// Stirling approximation of `ln(Γ(x))` (`gamma_ln`).
fn gamma_ln(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if (x - 1.0).abs() < 1e-6 || (x - 2.0).abs() < 1e-6 {
        return 0.0;
    }
    let ln_2pi = 1.837_877_f32;
    (x - 0.5) * x.ln() - x + 0.5 * ln_2pi
}

/// `calculate_hyperscore`: `ln(Nb!) + ln(Ny!) + ln(ΣIb) + ln(ΣIy)` over matched
/// b/y ions with positive intensity.
fn calculate_hyperscore(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
) -> f32 {
    calculate_hyperscore_weighted(fragment_types, fragment_intensities, matched_mask, None)
}

/// Weighted hyperscore (used for the inverse-mass-error variant).
fn calculate_hyperscore_weighted(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    weights: Option<&[f32]>,
) -> f32 {
    if fragment_types.len() != fragment_intensities.len()
        || fragment_types.len() != matched_mask.len()
    {
        return 0.0;
    }
    if let Some(w) = weights {
        if w.len() != fragment_types.len() {
            return 0.0;
        }
    }

    let mut n_b = 0u32;
    let mut n_y = 0u32;
    let mut sum_b = 0.0f32;
    let mut sum_y = 0.0f32;

    for i in 0..fragment_types.len() {
        if !matched_mask[i] || fragment_intensities[i] == 0.0 {
            continue;
        }
        let weight = weights.map(|w| w[i]).unwrap_or(1.0);
        let weighted = fragment_intensities[i] * weight;
        match fragment_types[i] {
            FRAG_TYPE_B => {
                n_b += 1;
                sum_b += weighted;
            }
            FRAG_TYPE_Y => {
                n_y += 1;
                sum_y += weighted;
            }
            _ => {}
        }
    }

    if n_b == 0 && n_y == 0 {
        return 0.0;
    }

    let factorial_b = if n_b > 0 { gamma_ln(n_b as f32 + 1.0) } else { 0.0 };
    let factorial_y = if n_y > 0 { gamma_ln(n_y as f32 + 1.0) } else { 0.0 };
    let ln_sum_b = if sum_b > 0.0 { sum_b.ln() } else { 0.0 };
    let ln_sum_y = if sum_y > 0.0 { sum_y.ln() } else { 0.0 };

    let hyperscore = factorial_b + factorial_y + ln_sum_b + ln_sum_y;
    if hyperscore.is_finite() {
        hyperscore
    } else {
        0.0
    }
}

/// Hyperscore weighted by `1/(|mass_error| + 0.1)` per matched ion.
fn calculate_hyperscore_inverse_mass_error(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    mass_errors: &[f32],
) -> f32 {
    if fragment_types.len() != mass_errors.len() {
        return 0.0;
    }
    let weights: Vec<f32> = mass_errors
        .iter()
        .map(|&e| 1.0 / (e.abs() + 0.1))
        .collect();
    calculate_hyperscore_weighted(
        fragment_types,
        fragment_intensities,
        matched_mask,
        Some(&weights),
    )
}

/// Longest consecutive b/y ion series (`calculate_longest_ion_series`).
fn calculate_longest_ion_series(
    fragment_types: &[u8],
    fragment_numbers: &[u16],
    matched_mask: &[bool],
) -> (u8, u8) {
    if fragment_types.len() != matched_mask.len()
        || fragment_types.len() != fragment_numbers.len()
    {
        return (0, 0);
    }

    let mut b_ions: Vec<u16> = Vec::new();
    let mut y_ions: Vec<u16> = Vec::new();
    for i in 0..fragment_types.len() {
        if matched_mask[i] {
            match fragment_types[i] {
                FRAG_TYPE_B => b_ions.push(fragment_numbers[i]),
                FRAG_TYPE_Y => y_ions.push(fragment_numbers[i]),
                _ => {}
            }
        }
    }

    fn longest(mut ions: Vec<u16>) -> u8 {
        if ions.is_empty() {
            return 0;
        }
        ions.sort_unstable();
        let mut max_len = 1u8;
        let mut cur_len = 1u8;
        for i in 1..ions.len() {
            if ions[i] == ions[i - 1] + 1 {
                cur_len += 1;
                max_len = max_len.max(cur_len);
            } else {
                cur_len = 1;
            }
        }
        max_len
    }

    (longest(b_ions), longest(y_ions))
}

/// Total matched observed intensity of one ion series (`intensity_ion_series`).
fn intensity_ion_series(
    fragment_types: &[u8],
    fragment_intensities: &[f32],
    matched_mask: &[bool],
    target_type: u8,
) -> f32 {
    let n = fragment_types.len();
    if n != fragment_intensities.len() || n != matched_mask.len() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for i in 0..n {
        if matched_mask[i] && fragment_intensities[i] > 0.0 && fragment_types[i] == target_type {
            total += fragment_intensities[i];
        }
    }
    total
}

/// Dot product of two equal-length slices (`calculate_dot_product`).
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ===========================================================================
// Fragment m/z + mass error
// ===========================================================================

/// Weighted mean of each row, excluding zero-weight entries.
fn weighted_mean_nonzero_axis_1(values: &Array2<f32>, weights: &Array2<f32>) -> Vec<f32> {
    let n_rows = values.nrows();
    let n_cols = values.ncols();
    let mut result = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let mut wsum = 0.0f32;
        let mut wv = 0.0f32;
        for c in 0..n_cols {
            let w = weights[[r, c]];
            if w > 0.0 {
                wv += values[[r, c]] * w;
                wsum += w;
            }
        }
        result[r] = if wsum > 0.0 { wv / wsum } else { 0.0 };
    }
    result
}

/// Observed fragment m/z + mass errors (ppm) from the dense m/z observation.
fn calculate_fragment_mz_and_errors(
    dense_mz: &Array2<f32>,
    intensity_observed: &Array2<f32>,
    mz_library: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mz_observed = weighted_mean_nonzero_axis_1(dense_mz, intensity_observed);
    let n = mz_library.len();
    let mut mass_errors = vec![0.0f32; n];
    for f in 0..n {
        let obs = mz_observed[f];
        if obs > 0.0 {
            mass_errors[f] = ((obs - mz_library[f]) / mz_library[f]) * 1e6;
        }
    }
    (mz_observed, mass_errors)
}

/// Weighted mean of absolute mass errors, weighted by library intensities.
fn calculate_weighted_mean_absolute_error(
    mass_errors: &[f32],
    intensity_weights: &[f32],
) -> f32 {
    if mass_errors.len() != intensity_weights.len() || mass_errors.is_empty() {
        return 0.0;
    }
    let mut wsum = 0.0f32;
    let mut wv = 0.0f32;
    for i in 0..mass_errors.len() {
        let w = intensity_weights[i];
        if w > 0.0 && mass_errors[i] != 0.0 {
            wv += mass_errors[i].abs() * w;
            wsum += w;
        }
    }
    if wsum > 0.0 {
        wv / wsum
    } else {
        0.0
    }
}

// ===========================================================================
// Ranked masks + threshold counts
// ===========================================================================

/// Boolean mask for ranks `r1..r2` (by value descending, tie-break index asc).
fn create_ranked_mask(intensities: &[f32], r1: usize, r2: usize) -> Vec<bool> {
    if intensities.is_empty() || r1 >= r2 {
        return vec![false; intensities.len()];
    }
    let mut indexed: Vec<(f32, usize)> = intensities
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let mut mask = vec![false; intensities.len()];
    let start = r1.min(indexed.len());
    let end = r2.min(indexed.len());
    for i in start..end {
        mask[indexed[i].1] = true;
    }
    mask
}

/// Count values above a threshold, optionally restricted to a mask.
fn count_values_above(values: &[f32], threshold: f32, mask: Option<&[bool]>) -> usize {
    match mask {
        Some(m) => {
            if m.len() != values.len() {
                return 0;
            }
            values
                .iter()
                .enumerate()
                .filter(|(i, &x)| m[*i] && x > threshold)
                .count()
        }
        None => values.iter().filter(|&&x| x > threshold).count(),
    }
}

// ===========================================================================
// RT FWHM
// ===========================================================================

/// Full width at half maximum of a symmetric profile (`calculate_fwhm_rt`).
fn calculate_fwhm_rt(xic_profile: &[f32], cycle_start_idx: usize, cycle_rt: &[f64]) -> f32 {
    if xic_profile.is_empty() {
        return 0.0;
    }
    let half_size = xic_profile.len() / 2;
    let center_intensity = xic_profile[half_size];
    for i in 0..half_size {
        let mean_intensity = (xic_profile[half_size - i] + xic_profile[half_size + i]) / 2.0;
        if mean_intensity <= center_intensity / 2.0 {
            let left_rt = cycle_rt[cycle_start_idx + half_size - i];
            let right_rt = cycle_rt[cycle_start_idx + half_size + i];
            return (right_rt - left_rt) as f32;
        }
    }
    0.0
}
