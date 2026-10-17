//! Peak-group ("NG") search backend.
//!
//! A self-contained, RT-only search path ported from the reference search
//! engine's compiled peak-group backend: peak-group selection (per-fragment XIC
//! over a wide RT window → Gaussian smoothing → library-intensity-weighted log
//! dot product → strict local maxima) followed by 40-feature peak-group scoring
//! and the shared runtime-trained FDR classifier.

use std::cmp::Ordering;
use std::io::Write;

use crate::engine::calibration::CalibrationEstimator;
use crate::engine::data::{DiaData, FragmentDf, SpecLibFlat};
use crate::engine::fdr::{get_q_values, keep_best, perform_fdr, BinaryClassifier, FdrConfig, FdrOutput};
use crate::engine::result::{CoreResult, QuantOutput};
use crate::engine::scoring;
use crate::log::{info, log_rss};
use crate::model::SpectralLibrary;

pub use crate::engine::scoring::{NgScoring, NgScoringConfig, NG_FEATURE_COUNT, NG_FEATURE_NAMES};
pub use crate::engine::selection::{NgCandidate, NgSelection, NgSelectionConfig};
pub use crate::engine::util::{InverseDocumentFrequency, MzIndex};

/// Fragment-series id for b ions (ASCII 'b').
pub const FRAG_TYPE_B: u8 = 98;
/// Fragment-series id for y ions (ASCII 'y').
pub const FRAG_TYPE_Y: u8 = 121;

/// A precursor's fragments after the reference `filter_sort_fragments` pipeline
/// (intensity > 0, drop y1, top-k by intensity, sort by m/z ascending).
pub struct FilteredFragments {
    /// Fragment extraction-target m/z (calibratable).
    pub mz: Vec<f64>,
    /// Fragment library m/z (IDF input).
    pub mz_library: Vec<f64>,
    /// Library relative intensity (raw, not re-normalized).
    pub intensity: Vec<f32>,
    pub fragment_type: Vec<u8>,
    pub fragment_number: Vec<u16>,
}

/// `filter_sort_fragments`: retain intensity > 0, drop y1, keep top-k by
/// intensity, sort by m/z ascending.
pub fn filter_fragments(
    frag: &FragmentDf,
    start: usize,
    stop: usize,
    top_k: usize,
) -> FilteredFragments {
    let mut rows: Vec<usize> = (start..stop).collect();
    // 1. retain intensity > 0
    rows.retain(|&i| frag.fragment_intensity[i] > 0.0);
    // 2. drop y1 (Y series, number == 1)
    rows.retain(|&i| !(frag.fragment_type[i] == FRAG_TYPE_Y && frag.fragment_number[i] == 1));
    // 3. top-k by intensity descending (stable tie-break by index ascending)
    rows.sort_by(|&a, &b| {
        frag.fragment_intensity[b]
            .partial_cmp(&frag.fragment_intensity[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    if rows.len() > top_k {
        rows.truncate(top_k);
    }
    // 4. sort by m/z ascending
    rows.sort_by(|&a, &b| {
        frag.fragment_mz[a]
            .partial_cmp(&frag.fragment_mz[b])
            .unwrap_or(Ordering::Equal)
    });

    FilteredFragments {
        mz: rows.iter().map(|&i| frag.fragment_mz[i]).collect(),
        mz_library: rows.iter().map(|&i| frag.mz_library[i]).collect(),
        intensity: rows.iter().map(|&i| frag.fragment_intensity[i]).collect(),
        fragment_type: rows.iter().map(|&i| frag.fragment_type[i]).collect(),
        fragment_number: rows.iter().map(|&i| frag.fragment_number[i]).collect(),
    }
}

/// One RT per DIA cycle (the MS1 retention time at `frame_in_cycle == 0`).
pub fn cycle_rt_values(dia_data: &DiaData) -> Vec<f64> {
    let zf = dia_data.zeroth_frame as usize;
    let cycle_len = dia_data.cycle_len.max(1);
    let n_cycles = dia_data.frame_max_index / cycle_len;
    (0..n_cycles)
        .map(|c| dia_data.rt_values[c * cycle_len + zf])
        .collect()
}

/// `get_cycle_idx_limits`: cycle-index window `precursor_rt ± effective_tol`,
/// where `effective_tol = max(rt_tolerance, (10 + kernel_size) * delta_t)`.
pub fn get_cycle_idx_limits(
    rt: &[f64],
    precursor_rt: f64,
    rt_tolerance: f64,
    kernel_size: usize,
) -> (usize, usize) {
    const MIN_CYCLES: f64 = 10.0;
    if rt.is_empty() {
        return (0, 0);
    }
    let delta_t = if rt.len() > 1 {
        (rt[rt.len() - 1] - rt[0]) / (rt.len() - 1) as f64
    } else {
        1.0
    };
    let min_tolerance = (MIN_CYCLES + kernel_size as f64) * delta_t;
    let effective = rt_tolerance.max(min_tolerance);

    let lower_rt = precursor_rt - effective;
    let upper_rt = precursor_rt + effective;

    if upper_rt < rt[0] {
        return (0, 0);
    }
    if lower_rt > rt[rt.len() - 1] {
        return (rt.len(), rt.len());
    }

    let lower_idx = if lower_rt <= rt[0] {
        0
    } else {
        rt.partition_point(|&x| x < lower_rt)
    };
    let upper_idx = if upper_rt >= rt[rt.len() - 1] {
        rt.len()
    } else {
        rt.partition_point(|&x| x < upper_rt)
    };
    (lower_idx, upper_idx)
}

/// End-to-end peak-group search configuration.
#[derive(Debug, Clone)]
pub struct NgConfig {
    pub selection: NgSelectionConfig,
    pub scoring: NgScoringConfig,
    pub fdr: FdrConfig,
    pub fdr_level: f32,
}

impl Default for NgConfig {
    fn default() -> Self {
        Self {
            selection: NgSelectionConfig::default(),
            scoring: NgScoringConfig::default(),
            fdr: FdrConfig::default(),
            fdr_level: 0.01,
        }
    }
}

/// Feature-array index of the observed RT feature (see `NG_FEATURE_NAMES`).
const FEAT_RT_OBSERVED: usize = 22;

/// Minimum target precursors before RT calibration is attempted.
const MIN_CALIBRATION_PRECURSORS: usize = 20;

/// Automatic RT optimizer: `new_param = 1.5 * ci(0.999)` (matches the classic
/// `AutomaticRTOptimizer`).
const RT_UPDATE_FACTOR: f64 = 1.5;
const RT_UPDATE_PERCENTILE: f64 = 0.999;

/// Automatic RT optimizer convergence (reference `calibration` config): iterate
/// search → LOESS calibration → tolerance proposal until the proposed tolerance
/// stabilizes (relative change < `RT_PARAM_CHANGE`) or the feature (targets
/// passing FDR) drops by more than `RT_FEATURE_DECREASE` over two consecutive
/// steps (`try_narrower_values`), after at least `RT_MIN_STEPS` iterations.
const RT_MAX_STEPS: usize = 20;
const RT_MIN_STEPS: usize = 5;
const RT_PARAM_CHANGE: f64 = 0.05;
const RT_FEATURE_DECREASE: f64 = 0.1;

/// One NG search stage: selection → scoring → FDR.
struct NgStage {
    result: scoring::NgScoringResult,
    fdr: FdrOutput,
}

/// Run one selection → scoring → FDR stage at the given absolute RT tolerance.
fn ng_stage(
    dia_data: &DiaData,
    spec: &SpecLibFlat,
    idf: &InverseDocumentFrequency,
    cycle_rt: &[f64],
    config: &NgConfig,
    rt_tolerance: f64,
    classifier: &mut BinaryClassifier,
    reuse_fitted: bool,
) -> NgStage {
    let mut sel_cfg = config.selection.clone();
    sel_cfg.rt_tolerance = rt_tolerance;
    let selection = NgSelection::new(sel_cfg);
    let candidates = selection.run(dia_data, spec);
    info!(
        "ng: selection (rt={:.0}s) -> {} candidates",
        rt_tolerance,
        candidates.len()
    );
    log_rss(&format!("after selection (rt={rt_tolerance:.0}s)"));

    let scoring = NgScoring::new(config.scoring.clone());
    let result = scoring.score(dia_data, spec, idf, cycle_rt, &candidates);
    info!(
        "ng: scored {} candidates ({} × {} feature matrix)",
        result.precursor_idx.len(),
        result.features.nrows(),
        result.features.ncols()
    );
    log_rss("after scoring");

    if std::env::var("DIAFORGE_SIMPLE_FDR").is_ok() {
        report_simple_score_fdr(&result);
    }

    let mut fdr_cfg = config.fdr.clone();
    fdr_cfg.reuse_fitted = reuse_fitted;
    let channel = vec![0u8; result.decoy.len()];
    let fdr = perform_fdr(
        classifier,
        &result.features,
        &result.decoy,
        &result.precursor_idx,
        &result.elution_group_idx,
        &channel,
        &fdr_cfg,
    );
    log_rss("after fdr");
    NgStage { result, fdr }
}

/// Number of target candidates passing `q <= fdr_level` in a stage.
fn count_passing(stage: &NgStage, fdr_level: f32) -> usize {
    (0..stage.result.decoy.len())
        .filter(|&i| stage.result.decoy[i] == 0 && stage.fdr.qval[i] <= fdr_level)
        .count()
}

/// Per-feature target/decoy distribution statistics for one stage.
struct FeatureStats {
    auc: f64,
    target_n: usize,
    decoy_n: usize,
    target_mean: f64,
    target_std: f64,
    decoy_mean: f64,
    decoy_std: f64,
    /// target percentile [p5, p25, p50, p75, p95, p99].
    target_q: [f64; 6],
    /// decoy percentile [p5, p25, p50, p75, p95, p99].
    decoy_q: [f64; 6],
}

/// ROC-AUC of one feature for separating target (decoy == 0) from decoy, via
/// tie-averaged Mann-Whitney rank sum. 0.5 = no separation; >0.5 = target
/// larger; <0.5 = decoy larger.
fn feature_auc(vals: &[f32], is_target: &[bool]) -> f64 {
    let n_pos = is_target.iter().filter(|&&t| t).count();
    let n_neg = is_target.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(Ordering::Equal));
    let n = vals.len();
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && vals[idx[j + 1]] == vals[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            if is_target[idx[k]] {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    let u = rank_sum_pos - (n_pos as f64) * (n_pos as f64 + 1.0) / 2.0;
    u / ((n_pos as f64) * (n_neg as f64))
}

/// Mean, std, and quantile distribution of an already value-sorted slice.
fn dist_stats(sorted: &[f32]) -> (f64, f64, [f64; 6]) {
    let n = sorted.len();
    if n == 0 {
        return (0.0, 0.0, [0.0; 6]);
    }
    let mean = sorted.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var = sorted
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let q = |p: f64| sorted[((n - 1) as f64 * p) as usize] as f64;
    (
        mean,
        var.sqrt(),
        [q(0.05), q(0.25), q(0.50), q(0.75), q(0.95), q(0.99)],
    )
}

/// Compute per-feature target/decoy statistics for one stage.
fn analyze_features(result: &scoring::NgScoringResult) -> Vec<FeatureStats> {
    let n = result.decoy.len();
    let mut out = Vec::with_capacity(NG_FEATURE_COUNT);
    for f in 0..NG_FEATURE_COUNT {
        let col = result.features.column(f);
        let mut vals = Vec::with_capacity(n);
        let mut is_target = Vec::with_capacity(n);
        let mut t_sorted: Vec<f32> = Vec::with_capacity(n / 2);
        let mut d_sorted: Vec<f32> = Vec::with_capacity(n / 2);
        for i in 0..n {
            let v = col[i];
            vals.push(v);
            if result.decoy[i] == 0 {
                is_target.push(true);
                t_sorted.push(v);
            } else {
                is_target.push(false);
                d_sorted.push(v);
            }
        }
        let auc = feature_auc(&vals, &is_target);
        t_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        d_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let (t_mean, t_std, t_q) = dist_stats(&t_sorted);
        let (d_mean, d_std, d_q) = dist_stats(&d_sorted);
        out.push(FeatureStats {
            auc,
            target_n: t_sorted.len(),
            decoy_n: d_sorted.len(),
            target_mean: t_mean,
            target_std: t_std,
            decoy_mean: d_mean,
            decoy_std: d_std,
            target_q: t_q,
            decoy_q: d_q,
        });
    }
    out
}

/// Count target precursors whose best candidate is a "clean" co-eluting peak
/// (high median fragment correlation / many fragments over 0.5 correlation).
/// This measures the selection ceiling: how many precursors COULD be recovered
/// if the FDR MLP were perfect — independent of classifier training. If this
/// count is ~2× the FDR-pass count, the MLP/`keep_best` is the lever; if it is
/// close to the pass count, selection/extraction is the ceiling.
fn report_selection_ceiling(result: &scoring::NgScoringResult, stage: &str) {
    let mut best_corr: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    let mut best_n50: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for i in 0..result.decoy.len() {
        if result.decoy[i] != 0 {
            continue;
        }
        let pi = result.precursor_idx[i];
        let mc = result.features[[i, 1]]; // median_correlation
        let n50 = result.features[[i, 9]]; // num_over_50
        best_corr.entry(pi).and_modify(|v| *v = v.max(mc)).or_insert(mc);
        best_n50.entry(pi).and_modify(|v| *v = v.max(n50)).or_insert(n50);
    }
    let n = best_corr.len();
    let c_gt = |m: &std::collections::HashMap<u32, f32>, t: f32| {
        m.values().filter(|&&v| v > t).count()
    };
    let c_ge = |m: &std::collections::HashMap<u32, f32>, t: f32| {
        m.values().filter(|&&v| v >= t).count()
    };
    info!(
        "ng: {stage} selection ceiling: {n} target precursors | med_corr>0.3={} >0.5={} >0.7={} | n_over_50>=4={} >=6={} >=8={}",
        c_gt(&best_corr, 0.3),
        c_gt(&best_corr, 0.5),
        c_gt(&best_corr, 0.7),
        c_ge(&best_n50, 4.0),
        c_ge(&best_n50, 6.0),
        c_ge(&best_n50, 8.0),
    );
}

/// Env `DIAFORGE_CAND_DUMP=<path>`: write every TARGET candidate of the final
/// stage (library precursor index, FDR q-value / decoy probability, and the key
/// scoring features) to a CSV. Used offline to classify DIA-NN-reference
/// precursors that diaflux missed as *not selected* vs *selected-but-rejected*.
fn dump_candidates(result: &scoring::NgScoringResult, fdr: &FdrOutput, path: &str) {
    use std::io::Write;
    let file = std::fs::File::create(path).expect("create candidate dump");
    let mut w = std::io::BufWriter::new(file);
    writeln!(
        w,
        "precursor_idx,qval,proba,mean_corr,median_corr,num_over_50,num_over_0,idf_xic_dot,hyperscore_obs"
    )
    .unwrap();
    let mut n_targets = 0usize;
    for i in 0..result.decoy.len() {
        if result.decoy[i] != 0 {
            continue; // targets only
        }
        n_targets += 1;
        writeln!(
            w,
            "{},{:.6},{:.6},{:.6},{:.6},{:.2},{:.2},{:.4},{:.4}",
            result.precursor_idx[i],
            fdr.qval[i],
            fdr.proba[i],
            result.features[[i, 0]],
            result.features[[i, 1]],
            result.features[[i, 9]],
            result.features[[i, 10]],
            result.features[[i, 32]],
            result.features[[i, 19]],
        )
        .unwrap();
    }
    w.flush().unwrap();
    info!(
        "ng: dumped {n_targets} target candidates to {path}"
    );
}

/// Diagnose: how many targets pass `q<=0.01` when ranked by a SINGLE feature
/// instead of the MLP? This tests whether the ~94k `num_over_50>=6` selection
/// ceiling is reachable by a simple score (i.e. whether the MLP is the
/// bottleneck) or whether the features themselves cannot rank clean precursors
/// above decoys under target-decoy competition.
fn report_simple_score_fdr(result: &scoring::NgScoringResult) {
    // (feature column, name) — all are "higher = more target-like".
    let probes: &[(usize, &str)] = &[
        (1, "median_correlation"),
        (9, "num_over_50"),
        (19, "hyperscore_intensity_observation"),
        (32, "idf_xic_dot_product"),
    ];
    let n = result.decoy.len();
    let group: Vec<u64> = result.elution_group_idx.iter().map(|&g| g as u64).collect();
    for &(col, name) in probes {
        let c = result.features.column(col);
        // negate: higher feature value = more target-like = lower score
        let score: Vec<f32> = (0..n).map(|i| -c[i]).collect();
        let retained = keep_best(&score, &group);
        let r_score: Vec<f32> = retained.iter().map(|&i| score[i]).collect();
        let r_decoy: Vec<u8> = retained.iter().map(|&i| result.decoy[i]).collect();
        let r_prec: Vec<u32> = retained.iter().map(|&i| result.precursor_idx[i]).collect();
        let q = get_q_values(&r_score, &r_decoy, &r_prec);
        let pass = (0..q.len()).filter(|&k| r_decoy[k] == 0 && q[k] <= 0.01).count();
        info!("ng: simple-fdr [{name}] -> {pass} targets pass q<=0.01 (of {} retained)", retained.len());
    }
}

/// Feature separation between the q-passing targets and the full decoy set of
/// a stage. A high AUC here means the MLP's accepted targets are genuinely
/// distinct from decoys (the pipeline works and sensitivity is the lever); an
/// AUC still near 0.5 means the acceptance is not driven by these features
/// (the upstream selection/scoring is producing noise for the accepted set).
fn report_passing_separation(result: &scoring::NgScoringResult, fdr: &FdrOutput, stage: &str) {
    let pass: Vec<bool> = (0..result.decoy.len())
        .map(|i| result.decoy[i] == 0 && fdr.qval[i] <= 0.01)
        .collect();
    let n_pass = pass.iter().filter(|&&p| p).count();
    info!("ng: {stage} passing-target separation (n_pass={n_pass})");
    if n_pass == 0 {
        return;
    }
    for f in 0..NG_FEATURE_COUNT {
        let col = result.features.column(f);
        let mut vals = Vec::with_capacity(result.decoy.len());
        let mut is_target = Vec::with_capacity(result.decoy.len());
        let mut pass_sum = 0.0f64;
        let mut pass_n = 0usize;
        for i in 0..result.decoy.len() {
            let v = col[i];
            if result.decoy[i] == 1 {
                vals.push(v);
                is_target.push(false);
            } else if pass[i] {
                vals.push(v);
                is_target.push(true);
                pass_sum += v as f64;
                pass_n += 1;
            }
        }
        let auc = feature_auc(&vals, &is_target);
        info!(
            "ng: pass feat {:2} {:30} pass_mean={:12.4e} vs_decoy_auc={:.3}",
            f, NG_FEATURE_NAMES[f], pass_sum / pass_n as f64, auc
        );
    }
}

/// Log a one-line target/decoy separation summary per feature, and (when
/// `DIAFORGE_FEAT_DUMP` is set to an output path) append the full distribution
/// to a CSV for offline analysis.
fn report_feature_diagnostics(stats: &[FeatureStats], stage: &str) {
    for (f, s) in stats.iter().enumerate() {
        info!(
            "ng: feat {:2} {:30} t_mean={:12.4e} d_mean={:12.4e} auc={:.3} (t_n={} d_n={})",
            f, NG_FEATURE_NAMES[f], s.target_mean, s.decoy_mean, s.auc, s.target_n, s.decoy_n
        );
    }

    let Ok(path) = std::env::var("DIAFORGE_FEAT_DUMP") else {
        return;
    };
    let write_header = !std::path::Path::new(&path).exists();
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            info!("ng: feature dump open failed: {e}");
            return;
        }
    };
    if write_header {
        let _ = writeln!(
            file,
            "stage,feature,name,auc,target_n,decoy_n,t_mean,t_std,t_p5,t_p25,t_p50,t_p75,t_p95,t_p99,d_mean,d_std,d_p5,d_p25,d_p50,d_p75,d_p95,d_p99"
        );
    }
    for (f, s) in stats.iter().enumerate() {
        let _ = writeln!(
            file,
            "{},{},{},{:.4},{},{},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e}",
            stage,
            f,
            NG_FEATURE_NAMES[f],
            s.auc,
            s.target_n,
            s.decoy_n,
            s.target_mean,
            s.target_std,
            s.target_q[0],
            s.target_q[1],
            s.target_q[2],
            s.target_q[3],
            s.target_q[4],
            s.target_q[5],
            s.decoy_mean,
            s.decoy_std,
            s.decoy_q[0],
            s.decoy_q[1],
            s.decoy_q[2],
            s.decoy_q[3],
            s.decoy_q[4],
            s.decoy_q[5],
        );
    }
    info!("ng: feature distribution dumped to {path} (stage {stage})");
}

/// Median of a finite float slice.
fn median_abs(vals: &[f64]) -> f64 {
    let mut c: Vec<f64> = vals.iter().copied().filter(|x| x.is_finite()).collect();
    if c.is_empty() {
        return 0.0;
    }
    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = c.len();
    if n % 2 == 0 {
        (c[n / 2 - 1] + c[n / 2]) / 2.0
    } else {
        c[n / 2]
    }
}

/// Pearson correlation of two equal-length slices (0 on degenerate input).
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        0.0
    } else {
        sxy / (sxx.sqrt() * syy.sqrt())
    }
}

/// Fit a LOESS RT estimator on the q≤0.01 targets of an initial stage,
/// recalibrate `spec.precursor_df.rt_pred`, and return the optimized RT
/// tolerance (`RT_UPDATE_FACTOR × ci(RT_UPDATE_PERCENTILE)`, floored at 1.5 s).
///
/// Returns `None` (leaving the library untouched) when there are too few
/// calibration targets or the fit is degenerate.
fn calibrate_rt(spec: &mut SpecLibFlat, stage: &NgStage) -> Option<f64> {
    let mut rt_lib: Vec<f64> = Vec::new();
    let mut rt_obs: Vec<f64> = Vec::new();
    for i in 0..stage.result.decoy.len() {
        if stage.result.decoy[i] != 0 || stage.fdr.qval[i] > 0.01 {
            continue;
        }
        let pi = stage.result.precursor_idx[i];
        let Some(&row) = spec.precursor_index_map.get(&pi) else {
            continue;
        };
        let obs = stage.result.features[[i, FEAT_RT_OBSERVED]] as f64;
        if obs.is_finite() && obs > 0.0 {
            rt_lib.push(spec.precursor_df.rt_pred[row]);
            rt_obs.push(obs);
        }
    }

    if rt_lib.len() < MIN_CALIBRATION_PRECURSORS {
        info!(
            "ng: rt calibration skipped ({} targets < {})",
            rt_lib.len(),
            MIN_CALIBRATION_PRECURSORS
        );
        return None;
    }

    let raw_offset: Vec<f64> = rt_lib
        .iter()
        .zip(&rt_obs)
        .map(|(l, o)| (o - l).abs())
        .collect();

    let mut est = CalibrationEstimator::new("rt", 6, None);
    est.fit(&rt_lib, &rt_obs);
    if !est.is_fitted() {
        return None;
    }

    // Diagnostic: residual distribution after LOESS. A small median residual
    // with a heavy tail means the predicted RT is well-correlated but noisy /
    // only partially covered by stage-1 targets (the wide stage-1 window misses
    // high-RT precursors whose predicted RT is far off).
    let pred_rt = est.predict(&rt_lib);
    let mut resid: Vec<f64> = rt_lib
        .iter()
        .zip(&rt_obs)
        .zip(&pred_rt)
        .map(|((&l, &o), &p)| (o - p).abs())
        .collect();
    resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let rn = resid.len();
    let pct = |q: f64| resid[((rn - 1) as f64 * q) as usize];
    let (rt_min, rt_max) = rt_lib
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    info!(
        "ng: rt loess residual |obs-pred|: median={:.1}s p90={:.1}s p99={:.1}s p99.9={:.1}s (rt_pred range [{:.0},{:.0}]s, n={})",
        pct(0.5), pct(0.9), pct(0.99), pct(0.999), rt_min, rt_max, rn
    );

    // Reference formula: `1.5 × ci(0.999)` (RT_UPDATE_FACTOR × central-interval
    // width of the absolute LOESS residual). This is the correct window when the
    // library RT is pre-calibrated; with a raw library the 99.9th-percentile is
    // heavy-tailed and the window stays wide, which is exactly what the reference
    // does — the wide window is then tightened by the LOESS recalibration, not by
    // an ad-hoc cap. (A 3×median cap was tried and REDUCED recall 68k→33k, so it
    // was reverted.)
    let reference_tol =
        (RT_UPDATE_FACTOR * est.ci(&rt_lib, &rt_obs, RT_UPDATE_PERCENTILE)).max(1.5);
    let rt_tolerance = reference_tol;

    spec.precursor_df.rt_pred = est.predict(&spec.precursor_df.rt_pred);

    info!(
        "ng: rt calibration on {} targets (corr(pred,obs)={:.3}, median |offset| {:.1}s) -> rt_tolerance={:.2}s (1.5·ci)",
        rt_lib.len(),
        pearson(&rt_lib, &rt_obs),
        median_abs(&raw_offset),
        rt_tolerance
    );
    for k in 0..rt_lib.len().min(8) {
        info!(
            "ng:   calib[{k}] rt_pred={:.1}s rt_obs={:.1}s",
            rt_lib[k], rt_obs[k]
        );
    }
    Some(rt_tolerance)
}

/// `AutomaticRTOptimizer._just_converged` (try_narrower_values branch): stop when
/// the proposed tolerance has stopped moving (<5% relative) or the feature
/// (targets passing FDR — proportional to the reference's
/// `precursor_proportion_detected`) has dropped by more than 10% over the last
/// two steps. Called only after `RT_MIN_STEPS` iterations.
fn rt_converged(n_pass_history: &[usize], new_tol: f64, cur_tol: f64) -> bool {
    // parameter-not-substantially-changed
    if cur_tol > 0.0 && (new_tol - cur_tol).abs() / cur_tol < RT_PARAM_CHANGE {
        return true;
    }
    // feature-substantially-decreased over two consecutive steps
    let n = n_pass_history.len();
    if n >= 3 {
        let f_last = n_pass_history[n - 1] as f64;
        let f_2nd = n_pass_history[n - 2] as f64;
        let f_3rd = n_pass_history[n - 3] as f64;
        if f_2nd > 0.0 && f_3rd > 0.0 {
            return (f_last - f_2nd) / f_2nd < -RT_FEATURE_DECREASE
                && (f_last - f_3rd) / f_3rd < -RT_FEATURE_DECREASE;
        }
    }
    false
}

/// Run the full peak-group search: two-stage (initial wide RT → RT calibration
/// → optimized narrow RT), selection → scoring → FDR → target filter.
///
/// Quantification is intentionally left empty for now — the benchmark goal is
/// the PSM count (`n_psm`), which exercises the selection + scoring + FDR path.
/// Pre-transformed library (flat + fragment-filtered + IDF index), independent
/// of the DIA data. Built once per pass and shared across all replicate runs so
/// the ~4.5 min fragment-index construction isn't redone for every file.
pub struct PreparedNgLibrary {
    spec: SpecLibFlat,
    idf: InverseDocumentFrequency,
}

/// Build the search-ready flat library from `lib` (transform + IDF). Hoisted out
/// of `run_ng` because it depends only on the library, which is identical across
/// all replicate runs within a pass.
pub fn prepare_ng_library(lib: &SpectralLibrary) -> PreparedNgLibrary {
    let mut spec = crate::engine::speclib::from_spectral_library(lib);
    crate::engine::libtransform::transform_library(&mut spec, 4);
    info!(
        "ng: library ready ({} precursors, {} fragments)",
        spec.n_precursors(),
        spec.fragment_df.fragment_mz.len()
    );
    let idf = InverseDocumentFrequency::new(&spec.fragment_df.mz_library);
    PreparedNgLibrary { spec, idf }
}

pub fn run_ng(
    prepared: &mut PreparedNgLibrary,
    dia_data: &DiaData,
    config: &NgConfig,
) -> CoreResult {
    // RT calibration overwrites `rt_pred` per run, so snapshot the library's
    // predicted RT up front and restore it after the run — each replicate must
    // start from the original prediction. Working on the shared library in place
    // (instead of cloning the whole ~5 GB spec) keeps peak memory at the old
    // level; a full clone doubled it and OOM'd the run's memory cgroup.
    let spec = &mut prepared.spec;
    let idf = &prepared.idf;
    let original_rt_pred = spec.precursor_df.rt_pred.clone();

    let cycle_rt = cycle_rt_values(dia_data);

    // 2. stage 1: initial search at the reference RT window (rt_tolerance =
    //    200 s, the reference search backend's default). A full-gradient window swamps
    //    the co-elution selection with interference local maxima, so it picks
    //    noise apexes for most precursors instead of the true co-eluting peak.
    //    Env DIAFORGE_INITIAL_RT overrides for quick tuning.
    let initial_rt = std::env::var("DIAFORGE_INITIAL_RT")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(config.selection.rt_tolerance);

    let mut classifier = BinaryClassifier::new(config.fdr.seed);
    let rt_max_steps = std::env::var("DIAFORGE_RT_MAX_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(RT_MAX_STEPS);
    let rt_min_steps = std::env::var("DIAFORGE_RT_MIN_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(RT_MIN_STEPS);
    let manual_rt = std::env::var("DIAFORGE_RT_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok());

    // 3. RT optimization: either a single re-search at a manual fixed tolerance
    //    (experiments), or the automatic convergence loop (search → LOESS
    //    calibration → tolerance proposal until the tolerance stabilizes / the
    //    feature plateaus), returning the history optimum.
    let (stage, stage_name): (NgStage, &str) = if let Some(fixed) = manual_rt {
        // Manual override for experiments (the automatic tolerance can be far
        // too wide when the library RT is poorly calibrated).
        let stage1 = ng_stage(dia_data, &spec, idf, &cycle_rt, config, initial_rt, &mut classifier, false);
        info!(
            "ng: stage 1 (initial, rt={:.0}s) -> {} targets pass q <= {}",
            initial_rt,
            count_passing(&stage1, config.fdr_level),
            config.fdr_level
        );
        spec.precursor_df.rt_pred = original_rt_pred.clone();
        match calibrate_rt(&mut *spec, &stage1) {
            Some(_) => {
                let stage2 = ng_stage(dia_data, &spec, idf, &cycle_rt, config, fixed, &mut classifier, false);
                info!(
                    "ng: stage 2 (fixed, rt={:.2}s) -> {} targets pass q <= {}",
                    fixed,
                    count_passing(&stage2, config.fdr_level),
                    config.fdr_level
                );
                (stage2, "optimized (fixed)")
            }
            None => (stage1, "initial (uncalibrated)"),
        }
    } else {
        // Automatic RT optimizer loop. Every iteration re-fits the same FDR
        // classifier deterministically (fixed seed, no shuffling across runs),
        // so per-iteration target counts are directly comparable for the
        // history-optimum selection below. Iterations after the first warm-start
        // from the previous weights and train only `warm_epochs`.

        // Keep only the (rt_tolerance, n_pass) trail plus the single best stage.
        // Retaining every iteration's full feature matrix (~n_candidates × 40 ×
        // 4 bytes ≈ 8 GB each) is the dominant memory cost and OOMs on large
        // files; the history-optimum only needs `n_pass` to pick the winner.
        let mut history: Vec<(f64, usize)> = Vec::new();
        let mut best_stage: Option<NgStage> = None;
        let mut best_n_pass: usize = 0;
        let mut rt_tolerance = initial_rt;
        loop {
            let stage = ng_stage(dia_data, &spec, idf, &cycle_rt, config, rt_tolerance, &mut classifier, false);
            let n_pass = count_passing(&stage, config.fdr_level);
            info!(
                "ng: opt iter {} (rt={:.2}s) -> {} targets pass q <= {}",
                history.len(),
                rt_tolerance,
                n_pass,
                config.fdr_level
            );

            // Reset to the original prediction so the LOESS always fits
            // original → observed (calibrate_rt overwrites rt_pred in place).
            spec.precursor_df.rt_pred = original_rt_pred.clone();
            let new_tol = calibrate_rt(&mut *spec, &stage);

            // `Iterator::max_by` returns the LAST element on equal `n_pass`, so
            // `>=` reproduces that tie-breaking exactly (bit-identical choice).
            if n_pass >= best_n_pass {
                best_n_pass = n_pass;
                best_stage = Some(stage);
            }
            history.push((rt_tolerance, n_pass));

            let Some(new_tol) = new_tol else {
                break; // too few targets / degenerate fit: keep the last stage
            };
            if history.len() >= rt_min_steps {
                let n_pass: Vec<usize> = history.iter().map(|h| h.1).collect();
                if rt_converged(&n_pass, new_tol, rt_tolerance) {
                    break;
                }
            }
            rt_tolerance = new_tol;
            if history.len() >= rt_max_steps {
                break;
            }
        }

        // History optimum: the iteration with the most targets passing FDR
        // (reference `_find_index_of_optimum`, `favour_narrower_optimum=False`).
        let best_idx = history
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.1.cmp(&b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        info!(
            "ng: rt optimization ran {} iteration(s); best = iter {} (rt={:.2}s, {} targets)",
            history.len(),
            best_idx,
            history[best_idx].0,
            history[best_idx].1
        );
        let best_stage = best_stage.expect("at least one stage is always run");
        (best_stage, "optimized")
    };

    if std::env::var("DIAFORGE_FEAT_DUMP").is_ok() {
        report_feature_diagnostics(&analyze_features(&stage.result), stage_name);
        report_passing_separation(&stage.result, &stage.fdr, stage_name);
    }
    report_selection_ceiling(&stage.result, stage_name);

    // 3.5. optional full-candidate dump for offline loss-point overlap analysis.
    if let Ok(p) = std::env::var("DIAFORGE_CAND_DUMP") {
        dump_candidates(&stage.result, &stage.fdr, &p);
    }

    // 4. retain target PSMs passing the q-value threshold.
    let keep: Vec<usize> = (0..stage.result.decoy.len())
        .filter(|&i| stage.result.decoy[i] == 0 && stage.fdr.qval[i] <= config.fdr_level)
        .collect();
    info!(
        "ng: fdr ({} stage) -> {} targets pass q <= {}",
        stage_name,
        keep.len(),
        config.fdr_level
    );

    let precursor_idx: Vec<u32> = keep.iter().map(|&i| stage.result.precursor_idx[i]).collect();
    let qval: Vec<f32> = keep.iter().map(|&i| stage.fdr.qval[i]).collect();
    let proba: Vec<f32> = keep.iter().map(|&i| stage.fdr.proba[i]).collect();
    let rt_observed: Vec<f32> = keep
        .iter()
        .map(|&i| stage.result.features[[i, FEAT_RT_OBSERVED]])
        .collect();
    let n_psm = precursor_idx.len();

    // Restore the library's predicted RT so the next replicate starts from the
    // original prediction (calibrate_rt overwrote it in place for this run).
    spec.precursor_df.rt_pred = original_rt_pred;

    CoreResult {
        n_psm,
        precursor_idx,
        qval,
        proba,
        rt_observed,
        quant: QuantOutput::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_converges_when_tolerance_stops_moving() {
        // The proposed tolerance is within 5% of the current one → converge
        // regardless of the target-count feature.
        assert!(rt_converged(&[1000, 1005, 1010], 21.0, 20.5));
        // A large relative change does not converge.
        assert!(!rt_converged(&[1000, 1005, 1010], 25.0, 20.0));
    }

    #[test]
    fn rt_converges_on_feature_decrease() {
        // Feature drops >10% over two consecutive steps (narrower window hurts
        // recall) → converge even though the tolerance is still moving.
        let n_pass = [1000usize, 1000, 950, 800]; // -20% then -16% from 950/1000
        assert!(rt_converged(&n_pass, 8.0, 12.0));
    }

    #[test]
    fn rt_does_not_converge_early() {
        // Fewer than 3 steps with a flat/rising feature and a moving tolerance
        // must not converge.
        let n_pass = [1000usize, 1010, 1020, 1030];
        assert!(!rt_converged(&n_pass, 9.0, 12.0));
    }
}
