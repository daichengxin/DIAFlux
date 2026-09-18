//! Transfer quantification (second-pass library refinement, part 1).
//!
//! After the first-pass search, high-confidence PSMs (q ≤ 0.01) are re-quantified
//! over the *full* b/y z1/z2 fragment ladder (not just the top-12 fragments kept
//! in the predicted search library). This yields per-fragment XIC intensity +
//! correlation + observed RT/mz, which the downstream transfer-library builder
//! aggregates across runs and feeds to the PeptDeep fine-tuner.
//!
//! The reference engine forces every candidate fragment's library
//! intensity to 1 and extracts its XIC with `top_k_fragments=9999`; the observed
//! spectra are used only as *training data*, never as a searched library.

use crate::model::chemistry::{aa_mass, F_TYPE_B, F_TYPE_Y, PROTON};
use crate::model::Precursor;

use crate::engine::calibration::{CalibrationEstimator, PPM_TRANSFORM};
use crate::engine::data::DiaData;
use crate::engine::ng::cycle_rt_values;
use crate::engine::util::correlation_safe;
use crate::log::info;
use ndarray::Array2;

/// Monoisotopic water mass (`H2O`), matching alphabase's `MASS_H2O`.
const MASS_H2O: f64 = 18.010_564_684;

/// Observable fragment m/z window (matches the sidecar's `FRAG_MZ_MIN/MAX`).
const FRAG_MZ_MIN: f64 = 200.0;
const FRAG_MZ_MAX: f64 = 2000.0;

/// One theoretical b/y fragment in the full ladder of a precursor.
#[derive(Debug, Clone, Copy)]
pub struct TransferFragment {
    /// Monoisotopic m/z (modification-aware).
    pub mz: f64,
    /// `F_TYPE_B` (1) or `F_TYPE_Y` (2).
    pub ion_type: u8,
    /// Standard series number: `b_n` / `y_n` (1-indexed, 1..=nAA-1).
    pub series_number: u16,
    /// Fragment charge state (1 or 2).
    pub charge: u8,
}

/// Residue mono-isotopic mass, honouring the lowercase modified residues that
/// `io::library::parse_modified_sequence` emits (`m` = oxidised Met, `c` =
/// carbamidomethyl Cys). All other residues delegate to `chemistry::aa_mass`.
#[inline]
fn aa_mass_mod(aa: char) -> f64 {
    match aa {
        'm' => 147.035_400,  // oxidised methionine (M + O)
        'c' => 160.030_645,  // carbamidomethyl cysteine (C + CH2CONH2)
        _ => aa_mass(aa),
    }
}

/// Generate the full b/y z1/z2 fragment ladder for a precursor from its
/// modification-aware sequence. Series numbers use the standard convention
/// (b ions forward `b1..b_{n-1}`, y ions `y1..y_{n-1}`), matching the library
/// parquet's `Fragment.Series.Number` encoding.
///
/// N-terminal modifications (e.g. acetylation) are not represented in the
/// predicted library (`check_library` allows only C(UniMod:4)/M(UniMod:35)) and
/// are therefore not mass-adjusted here.
pub fn full_fragment_mz(precursor: &Precursor) -> Vec<TransferFragment> {
    // `simple_seq` is the naked sequence with lowercase modified residues.
    let masses: Vec<f64> = precursor.simple_seq.chars().map(aa_mass_mod).collect();
    let n = masses.len();
    if n < 2 {
        return Vec::new();
    }
    let total: f64 = masses.iter().sum();
    let mut out = Vec::with_capacity(4 * (n - 1));

    // Neutral b/y masses follow alphabase `calc_b_y_and_peptide_mass`:
    //   b_n = sum of the first n residues (no water, no proton)
    //   pepmass = total + H2O
    //   y_n = pepmass - b_n = sum of the last L-n residues + H2O
    // and m/z = neutral / charge + MASS_PROTON (alphabase `calc_fragment_mz_values`).
    for charge in [1u8, 2u8] {
        let c = charge as f64;
        let mut prefix = 0.0f64;
        for i in 0..n - 1 {
            prefix += masses[i];
            // b_{i+1}: N-terminal acyl fragment of the first i+1 residues.
            let b_mz = prefix / c + PROTON;
            out.push(TransferFragment {
                mz: b_mz,
                ion_type: F_TYPE_B,
                series_number: (i + 1) as u16,
                charge,
            });
            // y_{n-i-1}: C-terminal fragment of the remaining n-i-1 residues.
            let y_mz = (total - prefix + MASS_H2O) / c + PROTON;
            out.push(TransferFragment {
                mz: y_mz,
                ion_type: F_TYPE_Y,
                series_number: (n - i - 1) as u16,
                charge,
            });
        }
    }
    out
}

// ===========================================================================
// Transfer quantification (requantify retained PSMs over the full fragment set)
// ===========================================================================

/// Transfer-requantification parameters.
#[derive(Debug, Clone)]
pub struct TransferQuantConfig {
    /// Fragment m/z extraction tolerance, ppm (7.0).
    pub mass_tolerance: f64,
    /// Half-width of the RT extraction window in *cycles* around the observed
    /// apex (3). This matches the reference `search.quant_window`, i.e. the
    /// first-pass candidate window `[apex - 3, apex + 4)` = 7 cycles. A wide
    /// RT window here would dilute the fragment-vs-median Pearson correlation
    /// with baseline and destroy the MS2 training signal.
    pub peak_length: usize,
}

impl Default for TransferQuantConfig {
    fn default() -> Self {
        Self {
            mass_tolerance: 7.0,
            peak_length: 3,
        }
    }
}

/// One requantified fragment of a retained PSM.
#[derive(Debug, Clone)]
pub struct QuantFragment {
    pub mz: f64,
    /// Intensity-weighted observed m/z over the extraction window (0 if the
    /// fragment had no signal). Used to derive the second-pass fragment m/z
    /// tolerance.
    pub observed_mz: f64,
    pub ion_type: u8,
    pub series_number: u16,
    pub charge: u8,
    /// Apex XIC intensity over the extraction window.
    pub intensity: f32,
    /// Pearson correlation of the fragment XIC with the precursor median profile.
    pub correlation: f32,
}

/// One requantified PSM (retained precursor + its full fragment ladder).
#[derive(Debug, Clone)]
pub struct QuantPsm {
    pub precursor_idx: u32,
    pub proba: f32,
    pub qval: f32,
    /// Observed apex RT from the first-pass search (s).
    pub rt_observed: f32,
    /// Precursor m/z (library value; Astral has no IMS dimension).
    pub mz_observed: f32,
    /// Naked *uppercase* sequence (alphabase `sequence`); modifications are
    /// carried separately in `mods` / `mod_sites`.
    pub sequence: String,
    /// `;`-separated modification names (`ModName@AA`), `""` if unmodified.
    pub mods: String,
    /// `;`-separated 1-indexed modification sites, `""` if unmodified.
    pub mod_sites: String,
    pub charge: u8,
    /// Number of residues (== sequence length).
    pub n_aa: usize,
    pub fragments: Vec<QuantFragment>,
}

/// Extract per-fragment XICs for a precursor over a narrow `2*peak_length+1`
/// cycle window centred on the observed apex, returning (sum intensity,
/// correlation-with-normalized-median profile) per fragment plus the observed
/// apex RT. Mirrors the NG backend's `PeakGroupQuantification::quantify_precursor`
/// over the first-pass candidate window: apex-normalize each fragment XIC, median
/// over non-zero rows, Pearson against that median profile.
fn extract_fragment_xics(
    dia_data: &DiaData,
    cycle_rt: &[f64],
    precursor_mz: f64,
    rt_center: f64,
    peak_length: usize,
    frag_mz: &[f64],
    mass_tolerance: f64,
) -> (Vec<f32>, Vec<f32>, Vec<f64>, f32) {
    let n_frag = frag_mz.len();
    let empty = (vec![0.0; n_frag], vec![0.0; n_frag], vec![0.0f64; n_frag], 0.0);
    if n_frag == 0 || cycle_rt.is_empty() {
        return empty;
    }

    // Apex cycle = cycle whose RT is closest to the observed apex; window =
    // `[apex - peak_length, apex + peak_length + 1)` (the same ±peak_length
    // candidate window the first-pass selection produced).
    let apex = (0..cycle_rt.len())
        .min_by(|&a, &b| {
            (cycle_rt[a] - rt_center)
                .abs()
                .partial_cmp(&(cycle_rt[b] - rt_center).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let cycle_start = apex.saturating_sub(peak_length);
    let cycle_stop = (apex + peak_length + 1).min(cycle_rt.len());
    if cycle_stop <= cycle_start {
        return empty;
    }
    let n_cycles = cycle_stop - cycle_start;

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
        frag_mz,
        mass_tolerance,
        &quadrupole_mz,
        true,
    );
    if dense.shape()[2] == 0 {
        return empty;
    }

    // xic[f][c] = raw per-cycle intensity (the NG dense XIC). The m/z slice is
    // unused here (a single quadrupole window at the precursor m/z).
    let mut xic = Array2::<f32>::zeros((n_frag, n_cycles));
    for f in 0..n_frag {
        for c in 0..n_cycles {
            xic[[f, c]] = dense[[0, f, 0, 0, c]];
        }
    }

    // Normalize each fragment profile by its mean over the apex ±1 cycle,
    // matching the NG backend's `normalize_profiles(center_dilations=1)`.
    // Rows whose apex-mean is 0 stay 0. The apex is the cycle closest to
    // `rt_center` (the first-pass observed RT), not necessarily n_cycles/2.
    let center_idx = (0..n_cycles)
        .min_by(|&a, &b| {
            (cycle_rt[cycle_start + a] - rt_center)
                .abs()
                .partial_cmp(&(cycle_rt[cycle_start + b] - rt_center).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(n_cycles / 2);
    let cs = center_idx.saturating_sub(1);
    let ce = (center_idx + 1 + 1).min(n_cycles);
    let mut normalized = Array2::<f32>::zeros((n_frag, n_cycles));
    for f in 0..n_frag {
        let mut center_sum = 0.0f32;
        for c in cs..ce {
            center_sum += xic[[f, c]];
        }
        let mean = center_sum / (ce - cs) as f32;
        if mean > 0.0 {
            for c in 0..n_cycles {
                normalized[[f, c]] = xic[[f, c]] / mean;
            }
        }
    }

    // Reference profile = per-cycle median over the non-all-zero normalized rows
    // (`filter_non_zero` + `median_axis_0` in the NG backend).
    let nonzero_rows: Vec<usize> = (0..n_frag)
        .filter(|&f| normalized.row(f).iter().any(|&v| v != 0.0))
        .collect();
    let mut median_profile = vec![0.0f32; n_cycles];
    for c in 0..n_cycles {
        let mut col: Vec<f32> = nonzero_rows.iter().map(|&f| normalized[[f, c]]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        median_profile[c] = if col.is_empty() {
            0.0
        } else if col.len() % 2 == 0 {
            (col[col.len() / 2 - 1] + col[col.len() / 2]) / 2.0
        } else {
            col[col.len() / 2]
        };
    }

    // Per fragment: intensity = raw XIC *sum* over the window (matching the NG
    // `observation_intensities`), correlation = Pearson(normalized profile,
    // median profile), observed m/z = intensity-weighted mean of the m/z slice
    // (`get_dense` channel 1 with `absolute_masses=true`).
    let mut intensity = Vec::with_capacity(n_frag);
    let mut correlation = Vec::with_capacity(n_frag);
    let mut observed_mz = Vec::with_capacity(n_frag);
    for f in 0..n_frag {
        let mut w_num = 0.0f64;
        let mut w_den = 0.0f64;
        for c in 0..n_cycles {
            let v = xic[[f, c]] as f64;
            w_num += v * dense[[1, f, 0, 0, c]] as f64;
            w_den += v;
        }
        observed_mz.push(if w_den > 0.0 { w_num / w_den } else { 0.0 });
        intensity.push(xic.row(f).iter().sum());
        let nrow: Vec<f32> = normalized.row(f).to_vec();
        correlation.push(correlation_safe(&nrow, &median_profile));
    }

    // Observed RT stays the first-pass apex (the candidate cycle_center), as in
    // the reference; it is the RT training target via rt_norm.
    (intensity, correlation, observed_mz, rt_center as f32)
}

/// Requantify the retained (q ≤ 0.01) first-pass PSMs over their full b/y z1/z2
/// fragment ladders. `lib` supplies precursor metadata (sequence / charge /
/// mods / m/z); `retained_idx`, `proba`, `qval`, `rt_observed` are aligned
/// vectors of the retained PSMs (from `CoreResult`).
///
/// `retained_idx` are raw `pr_index` values (pre-dedup), so precursors are
/// looked up via a `pr_index -> &Precursor` map.
pub fn transfer_quantify(
    lib: &crate::model::SpectralLibrary,
    dia_data: &DiaData,
    retained_idx: &[u32],
    proba: &[f32],
    qval: &[f32],
    rt_observed: &[f32],
    config: &TransferQuantConfig,
) -> Vec<QuantPsm> {
    use std::collections::HashMap;

    let mut by_pr_index: HashMap<u32, &Precursor> = HashMap::with_capacity(lib.len());
    for p in &lib.precursors {
        by_pr_index.insert(p.pr_index, p);
    }

    let cycle_rt = cycle_rt_values(dia_data);

    let mut out = Vec::with_capacity(retained_idx.len());
    for i in 0..retained_idx.len() {
        let Some(precursor) = by_pr_index.get(&retained_idx[i]) else {
            continue;
        };
        let fragments = full_fragment_mz(precursor);
        if fragments.is_empty() {
            continue;
        }

        let frag_mz: Vec<f64> = fragments
            .iter()
            .map(|f| f.mz)
            .filter(|&mz| mz >= FRAG_MZ_MIN && mz <= FRAG_MZ_MAX)
            .collect();
        // Keep only fragments within the observable window (same order as
        // `fragments`, so index j maps to a position in `fragments`).
        let kept: Vec<&TransferFragment> = fragments
            .iter()
            .filter(|f| f.mz >= FRAG_MZ_MIN && f.mz <= FRAG_MZ_MAX)
            .collect();
        if kept.is_empty() {
            continue;
        }

        let (intensity, correlation, observed_mz, apex_rt) = extract_fragment_xics(
            dia_data,
            &cycle_rt,
            precursor.pr_mz as f64,
            rt_observed[i] as f64,
            config.peak_length,
            &frag_mz,
            config.mass_tolerance,
        );

        let qfrags: Vec<QuantFragment> = kept
            .iter()
            .enumerate()
            .map(|(j, f)| QuantFragment {
                mz: f.mz,
                observed_mz: observed_mz[j],
                ion_type: f.ion_type,
                series_number: f.series_number,
                charge: f.charge,
                intensity: intensity[j],
                correlation: correlation[j],
            })
            .collect();

        out.push(QuantPsm {
            precursor_idx: retained_idx[i],
            proba: proba[i],
            qval: qval[i],
            rt_observed: apex_rt,
            mz_observed: precursor.pr_mz,
            // peptdeep trains on the *naked* uppercase sequence with mods /
            // mod_sites carried separately (alphabase convention), so use
            // `strip_seq`, not the lowercase-modified `simple_seq`.
            sequence: precursor.strip_seq.clone(),
            mods: precursor.mods.clone(),
            mod_sites: precursor.mod_sites.clone(),
            charge: precursor.pr_charge,
            n_aa: precursor.strip_seq.len(),
            fragments: qfrags,
        });
    }

    let n_nonzero: usize = out
        .iter()
        .map(|p| p.fragments.iter().filter(|f| f.intensity > 0.0).count())
        .sum();
    let max_corr = out
        .iter()
        .flat_map(|p| p.fragments.iter().map(|f| f.correlation))
        .fold(0.0f32, f32::max);
    info!(
        "transfer: requantified {} PSMs over full b/y z1/z2 ladders ({} non-zero fragments, max corr {:.3})",
        out.len(),
        n_nonzero,
        max_corr
    );
    out
}

// ===========================================================================
// Second-pass fragment m/z tolerance re-optimization
// ===========================================================================

/// Reference `optimization.ms2_error` constants: `new_ms2 = factor × ci(percentile)`.
const MS2_UPDATE_FACTOR: f64 = 1.1;
const MS2_UPDATE_PERCENTILE: f64 = 0.99;
const MS2_N_KERNELS: usize = 6;
/// Safety floor on the derived tolerance (ppm), below which a real instrument
/// would start dropping legitimate peaks to calibration drift.
const MS2_TOL_FLOOR_PPM: f64 = 2.0;
/// Cap on the number of fragments fed to the LOESS fit + `ci`. The estimator's
/// prediction is O(n) per query (a full scan over the sorted training set), so
/// `ci` is O(n²); on the ~9M fragments of a 5-file run this took ~2.5 h. We only
/// need the 99th-percentile residual, so a deterministic fixed-size subsample
/// reproduces the tolerance in seconds.
const MS2_MAX_LOESS_POINTS: usize = 300_000;

/// Derive the second-pass fragment m/z tolerance from the first-pass transfer
/// requantification, mirroring the reference's automatic MS2 optimizer: fit a
/// LOESS m/z calibration (ppm) on the observed-vs-predicted fragment m/z of all
/// retained PSMs and return `MS2_UPDATE_FACTOR ×` the central-interval width of
/// the absolute residual. `None` when there are too few observed fragments.
pub fn compute_ms2_tolerance(psms: &[QuantPsm]) -> Option<f64> {
    let mut pred_mz: Vec<f64> = Vec::new();
    let mut obs_mz: Vec<f64> = Vec::new();
    for p in psms {
        for f in &p.fragments {
            if f.intensity > 0.0 && f.observed_mz > 0.0 {
                pred_mz.push(f.mz);
                obs_mz.push(f.observed_mz);
            }
        }
    }
    if pred_mz.len() < 100 {
        return None;
    }

    // Deterministic stride-subsample to a fixed cap (see MS2_MAX_LOESS_POINTS);
    // the original interleaving of PSMs/fragments already samples the whole m/z
    // range, and a 99th-percentile estimate is insensitive to the downsampling.
    let (pred_mz, obs_mz): (Vec<f64>, Vec<f64>) = if pred_mz.len() > MS2_MAX_LOESS_POINTS {
        let step = (pred_mz.len() + MS2_MAX_LOESS_POINTS - 1) / MS2_MAX_LOESS_POINTS;
        (
            pred_mz.iter().step_by(step).copied().collect(),
            obs_mz.iter().step_by(step).copied().collect(),
        )
    } else {
        (pred_mz, obs_mz)
    };

    let mut est = CalibrationEstimator::new("mz", MS2_N_KERNELS, Some(PPM_TRANSFORM));
    est.fit(&pred_mz, &obs_mz);
    if !est.is_fitted() {
        return None;
    }
    let tol = (MS2_UPDATE_FACTOR * est.ci(&pred_mz, &obs_mz, MS2_UPDATE_PERCENTILE))
        .max(MS2_TOL_FLOOR_PPM);
    info!(
        "transfer: pass-2 fragment m/z tolerance {:.2} ppm (1.1·ci(0.99), n={} fragments)",
        tol,
        pred_mz.len()
    );
    Some(tol)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn precursor(seq: &str) -> Precursor {
        let mut p = Precursor::default();
        p.simple_seq = seq.to_string();
        p
    }

    #[test]
    fn unmodified_dipeptide() {
        // "AC" → 4 fragments (b1, y1 at z=1 and z=2).
        let frags = full_fragment_mz(&precursor("AC"));
        assert_eq!(frags.len(), 4);
        let b1 = frags.iter().find(|f| f.ion_type == F_TYPE_B && f.series_number == 1 && f.charge == 1).unwrap();
        let y1 = frags.iter().find(|f| f.ion_type == F_TYPE_Y && f.series_number == 1 && f.charge == 1).unwrap();
        // b1 = A + proton
        assert!((b1.mz - (aa_mass('A') + PROTON)).abs() < 1e-4, "b1 mz {}", b1.mz);
        // y1 = C + H2O + proton
        assert!((y1.mz - (aa_mass('C') + MASS_H2O + PROTON)).abs() < 1e-4, "y1 mz {}", y1.mz);
    }

    #[test]
    fn oxidation_and_carbamidomethyl_mass_shift() {
        // "cm" = carbamidomethyl Cys + oxidised Met: b1 = carbamidomethyl C,
        // y1 = oxidised M.
        let frags = full_fragment_mz(&precursor("cm"));
        let b1 = frags.iter().find(|f| f.ion_type == F_TYPE_B && f.series_number == 1 && f.charge == 1).unwrap();
        assert!((b1.mz - (160.030_645 + PROTON)).abs() < 1e-4, "carbamidomethyl b1 mz {}", b1.mz);
        let y1 = frags.iter().find(|f| f.ion_type == F_TYPE_Y && f.series_number == 1 && f.charge == 1).unwrap();
        assert!((y1.mz - (147.035_400 + MASS_H2O + PROTON)).abs() < 1e-4, "oxidised y1 mz {}", y1.mz);
    }

    #[test]
    fn tripeptide_series_numbers() {
        // "AAA" → b1,b2 and y2,y1 (standard numbering), for each charge.
        let frags = full_fragment_mz(&precursor("AAA"));
        assert_eq!(frags.len(), 8);
        assert!(frags.iter().any(|f| f.ion_type == F_TYPE_B && f.series_number == 1));
        assert!(frags.iter().any(|f| f.ion_type == F_TYPE_B && f.series_number == 2));
        assert!(frags.iter().any(|f| f.ion_type == F_TYPE_Y && f.series_number == 1));
        assert!(frags.iter().any(|f| f.ion_type == F_TYPE_Y && f.series_number == 2));
    }

    fn qfrag(mz: f64, observed_mz: f64, intensity: f32) -> QuantFragment {
        QuantFragment {
            mz,
            observed_mz,
            ion_type: F_TYPE_B,
            series_number: 1,
            charge: 1,
            intensity,
            correlation: 0.9,
        }
    }

    fn qpsm(fragments: Vec<QuantFragment>) -> QuantPsm {
        QuantPsm {
            precursor_idx: 0,
            proba: 0.01,
            qval: 0.01,
            rt_observed: 100.0,
            mz_observed: 500.0,
            sequence: "AAAA".into(),
            mods: String::new(),
            mod_sites: String::new(),
            charge: 2,
            n_aa: 4,
            fragments,
        }
    }

    #[test]
    fn ms2_tolerance_empty_is_none() {
        assert!(compute_ms2_tolerance(&[]).is_none());
    }

    #[test]
    fn ms2_tolerance_from_observed_mz() {
        // 200 fragments at +3 ppm with a deterministic ±2 ppm scatter: the LOESS
        // absorbs the systematic offset, so the residual (and hence the derived
        // tolerance) reflects the scatter, not the bias.
        let frags: Vec<QuantFragment> = (0..200)
            .map(|i| {
                let mz = 200.0 + i as f64;
                let jitter = (i % 5) as f64 - 2.0;
                qfrag(mz, mz * (1.0 + (3.0 + jitter) / 1e6), 100.0)
            })
            .collect();
        let tol = compute_ms2_tolerance(&[qpsm(frags)]).unwrap();
        assert!(tol >= MS2_TOL_FLOOR_PPM && tol < 20.0, "tol={tol}");
    }
}
