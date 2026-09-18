//! LOESS property calibration for the two-stage search.
//!
//! A local LOESS regression with
//!   * `n_kernels` uniformly distributed kernels (bandwidth = range / n_kernels),
//!   * a 2nd-degree polynomial basis (constant + linear + quadratic terms),
//!   * tricubic weights.
//!
//! The deviation between an observed (`target`) and calibrated (`predict(input)`)
//! value is optionally expressed relative to the input (ppm for `1e6`), else
//! absolute (RT / mobility). The `ci` method returns the mean of the endpoints
//! of the central `ci` interval of the absolute residual deviations — the value
//! the optimizers use to propose the next search tolerance.

use std::cmp::Ordering;

use rayon::prelude::*;

/// Transform factor for deviations expressed in parts-per-million.
pub const PPM_TRANSFORM: f64 = 1e6;

/// Tricubic weight kernel: `(1 - |u|^3)^3` for `|u| < 1`, else 0.
#[inline]
fn tricubic(u: f64) -> f64 {
    let a = u.abs();
    if a >= 1.0 {
        0.0
    } else {
        let t = 1.0 - a * a * a;
        t * t * t
    }
}

/// Solve a 3×3 linear system via Gaussian elimination with partial pivoting.
/// The LOESS local polynomial is a 2nd-degree fit, so only `[b0, b1, b2]` is
/// ever needed.
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut m = [[0.0f64; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][j];
        }
        m[i][3] = b[i];
    }
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-15 {
            continue;
        }
        m.swap(col, piv);
        let d = m[col][col];
        for j in col..4 {
            m[col][j] /= d;
        }
        for r in 0..3 {
            if r == col {
                continue;
            }
            let f = m[r][col];
            if f == 0.0 {
                continue;
            }
            for j in col..4 {
                m[r][j] -= f * m[col][j];
            }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}

/// A single LOESS calibration estimator for one property (m/z, RT, mobility).
pub struct CalibrationEstimator {
    /// Property name (logging only).
    pub name: String,
    n_kernels: usize,
    transform_deviation: Option<f64>,
    fitted: bool,
    /// Sorted training input (library) values.
    xs: Vec<f64>,
    /// Training target (observed) values, aligned with `xs`.
    ys: Vec<f64>,
}

impl CalibrationEstimator {
    pub fn new(name: &str, n_kernels: usize, transform_deviation: Option<f64>) -> Self {
        Self {
            name: name.to_string(),
            n_kernels: n_kernels.max(1),
            transform_deviation,
            fitted: false,
            xs: Vec::new(),
            ys: Vec::new(),
        }
    }

    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Fit the estimator on (input, target) pairs. Non-finite values are
    /// dropped; the remaining pairs are sorted by input value.
    pub fn fit(&mut self, input: &[f64], target: &[f64]) {
        let mut pairs: Vec<(f64, f64)> = input
            .iter()
            .zip(target)
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| (x, y))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        self.xs = pairs.iter().map(|p| p.0).collect();
        self.ys = pairs.iter().map(|p| p.1).collect();
        self.fitted = !self.xs.is_empty();
    }

    /// Predict calibrated values for `input`. Returns an empty vector when the
    /// estimator has not been fitted.
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        if !self.fitted {
            return Vec::new();
        }
        input.par_iter().map(|&x| self.predict_one(x)).collect()
    }

    /// Residual deviation at the central `ci` interval (see module docs).
    /// Returns 0 when not fitted.
    pub fn ci(&self, input: &[f64], target: &[f64], ci: f64) -> f64 {
        if !self.fitted || input.is_empty() {
            return 0.0;
        }
        let pred = self.predict(input);
        let mut devs: Vec<f64> = Vec::with_capacity(input.len());
        for i in 0..input.len() {
            let d = target[i] - pred[i];
            let dev = match self.transform_deviation {
                Some(tf) => d / input[i].abs().max(1e-12) * tf,
                None => d,
            };
            devs.push(dev.abs());
        }
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let n = devs.len();
        let nf = (n - 1) as f64;
        // Endpoints of the central `ci` interval, e.g. [2.5%, 97.5%] for 0.95.
        let lower = ((nf * (1.0 - ci) / 2.0).round() as usize).min(n - 1);
        let upper = ((nf * (1.0 - (1.0 - ci) / 2.0)).round() as usize).min(n - 1);
        (devs[lower] + devs[upper]) / 2.0
    }

    /// Local 2nd-degree polynomial LOESS prediction at a single point.
    fn predict_one(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 0 {
            return 0.0;
        }
        let (xmin, xmax) = (self.xs[0], self.xs[n - 1]);
        let span = xmax - xmin;
        let h = if span > 0.0 {
            span / self.n_kernels as f64
        } else {
            0.0
        };
        if h <= 0.0 {
            let s: f64 = self.ys.iter().sum();
            return s / n as f64;
        }

        // Weighted least squares for a quadratic centered at `x`:
        //   y ≈ b0 + b1·(xᵢ−x) + b2·(xᵢ−x)², prediction = b0.
        let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let d = self.xs[i] - x;
            let w = tricubic(d / h);
            if w == 0.0 {
                continue;
            }
            let d2 = d * d;
            let wy = w * self.ys[i];
            s0 += w;
            s1 += w * d;
            s2 += w * d2;
            s3 += w * d2 * d;
            s4 += w * d2 * d2;
            t0 += wy;
            t1 += wy * d;
            t2 += wy * d2;
        }

        if s0 <= 1e-12 {
            // Query outside the kernel support: nearest neighbour fallback.
            let mut best = self.ys[0];
            let mut bd = f64::INFINITY;
            for i in 0..n {
                let d = (self.xs[i] - x).abs();
                if d < bd {
                    bd = d;
                    best = self.ys[i];
                }
            }
            return best;
        }

        solve_3x3(
            [
                [s0, s1, s2],
                [s1, s2, s3],
                [s2, s3, s4],
            ],
            [t0, t1, t2],
        )[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tricubic_kernel_bounds() {
        assert_eq!(tricubic(0.0), 1.0);
        assert!(tricubic(0.5) > 0.0 && tricubic(0.5) < 1.0);
        assert_eq!(tricubic(1.0), 0.0);
        assert_eq!(tricubic(2.0), 0.0);
    }

    #[test]
    fn fit_predict_ppm_reduces_bias() {
        // observed = library * (1 + 10 ppm) + noise
        let n = 200;
        let lib: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 900.0 / n as f64).collect();
        let obs: Vec<f64> = lib
            .iter()
            .enumerate()
            .map(|(i, &m)| m * (1.0 + 1e-5) + (((i * 37) % 11) as f64 - 5.0) * 1e-3)
            .collect();

        let mut est = CalibrationEstimator::new("mz", 2, Some(PPM_TRANSFORM));
        est.fit(&lib, &obs);
        let pred = est.predict(&lib);

        let raw: f64 = lib.iter().zip(&obs).map(|(l, o)| (o - l).abs()).sum::<f64>() / n as f64;
        let cal: f64 = lib.iter().zip(&obs).zip(&pred).map(|((l, o), p)| (o - p).abs()).sum::<f64>() / n as f64;
        assert!(cal < raw, "calibrated residual ({}) should beat raw ({})", cal, raw);
    }

    #[test]
    fn ci_is_zero_before_fit_and_positive_after() {
        let lib = vec![100.0, 200.0, 300.0, 400.0];
        let obs = vec![100.001, 200.002, 300.003, 400.004];
        let mut est = CalibrationEstimator::new("mz", 2, Some(PPM_TRANSFORM));
        assert_eq!(est.ci(&lib, &obs, 0.95), 0.0);
        est.fit(&lib, &obs);
        assert!(est.ci(&lib, &obs, 0.95) > 0.0);
    }

    #[test]
    fn rt_absolute_captures_warp() {
        let n = 500;
        let lib: Vec<f64> = (0..n).map(|i| i as f64 * 100.0 / n as f64).collect();
        let obs: Vec<f64> = lib.iter().map(|&r| r + (r * 0.05).sin()).collect();

        let mut est = CalibrationEstimator::new("rt", 6, None);
        est.fit(&lib, &obs);
        let pred = est.predict(&lib);
        let med = lib.iter().zip(&obs).zip(&pred).map(|((_, o), p)| (o - p).abs()).sum::<f64>() / n as f64;
        assert!(med < 0.5, "median residual {} should be < 0.5", med);
    }
}
