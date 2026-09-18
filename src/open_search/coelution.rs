//! Co-elution SA scoring — replaces parts of `fxic.py` + `scoring.py`
//!
//! Calculates SA co-elution scores for peak groups using a Gaussian sliding window.

use ndarray::{Array2, Array3};

/// Gaussian kernel for 7-point window.
const GAUSSIAN_7: [f32; 7] = [
    0.0044, 0.054, 0.242, 0.399, 0.242, 0.054, 0.0044,
];
const GAUSSIAN_NORM: f32 = 0.531225;

/// Compute SA between a 7-point vector and the Gaussian kernel.
fn gaussian_sa(v: &[f32]) -> f32 {
    if v.len() < 7 {
        return 0.0;
    }
    let s = v[0] * GAUSSIAN_7[0]
        + v[1] * GAUSSIAN_7[1]
        + v[2] * GAUSSIAN_7[2]
        + v[3] * GAUSSIAN_7[3]
        + v[4] * GAUSSIAN_7[4]
        + v[5] * GAUSSIAN_7[5]
        + v[6] * GAUSSIAN_7[6];
    let norm_x = v.iter().map(|&x| x * x).sum::<f32>().sqrt() + 1e-6;
    let sa = s / (norm_x * GAUSSIAN_NORM);
    sa.min(1.0)
}

/// Compute co-elution SA scores for a batch of XICs.
///
/// # Arguments
/// * `xics` — Shape [n_pep, n_ion, n_cycle]
/// * `window_points` — Sliding window size (fixed = 7)
/// * `valids_num` — Number of valid ions per peptide
///
/// # Returns
/// * `scores` — Mean SA per peptide: [n_pep, n_cycle]
/// * `scores_raw` — Raw SA per ion: [n_pep, n_ion, n_cycle]
pub fn cal_coelution_by_gaussian(
    xics: &Array3<f32>,
    window_points: usize,
    valids_num: &[usize],
) -> (Array2<f32>, Array3<f32>) {
    let (n_pep, n_ion, n_cycle) = xics.dim();
    let half = window_points / 2;

    // Guard: n_cycle must be at least window_points (7) for SA computation
    if n_cycle < window_points || n_pep == 0 || valids_num.len() < n_pep {
        return (
            Array2::<f32>::zeros((n_pep, n_cycle)),
            Array3::<f32>::zeros((n_pep, n_ion, n_cycle)),
        );
    }

    let mut scores_raw = Array3::<f32>::zeros((n_pep, n_ion, n_cycle));

    for k in 0..n_pep {
        for ion in 0..n_ion {
            for c in half..(n_cycle - half) {
                let window: Vec<f32> =
                    (c - half..=c + half).map(|cc| xics[[k, ion, cc]]).collect();
                let sa = gaussian_sa(&window);
                scores_raw[[k, ion, c]] = sa;
            }
        }
    }

    // Convert SA to angular distance
    let mut scores_raw = scores_raw.mapv(|v| 1.0 - 2.0 * v.acos() / std::f32::consts::PI);

    // Mean across ions
    let mut scores = Array2::<f32>::zeros((n_pep, n_cycle));
    for k in 0..n_pep {
        let valid = valids_num[k].max(1);
        for c in 0..n_cycle {
            let sum: f32 = (0..n_ion).map(|ion| scores_raw[[k, ion, c]]).sum();
            scores[[k, c]] = sum / valid as f32;
        }
    }

    // Zero out edges
    for k in 0..n_pep {
        for c in 0..3.min(n_cycle) {
            scores[[k, c]] = 0.0;
            scores[[k, n_cycle - 1 - c]] = 0.0;
            for ion in 0..n_ion.min(14) {
                scores_raw[[k, ion, c]] = 0.0;
                scores_raw[[k, ion, n_cycle - 1 - c]] = 0.0;
            }
        }
    }

    (scores, scores_raw)
}
