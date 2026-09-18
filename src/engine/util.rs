//! Shared helpers for the peak-group ("NG") backend.
//!
//! Faithful scalar ports of the reference search backend's building blocks:
//! the global 1-ppm geometric m/z index + inverse document frequency, a 1-D
//! Gaussian kernel + convolution (no padding), `axis_log_dot_product`, and
//! strict two-point local-maximum finding.

use ndarray::Array2;

// ===========================================================================
// Global 1-ppm geometric m/z index + inverse document frequency
// ===========================================================================

pub const MZ_RESOLUTION_PPM: f64 = 1.0;
pub const MZ_START: f64 = 150.0;
pub const MZ_END: f64 = 2000.0;

/// Geometric m/z index (`ppm_index`): successive bins grow by 1 ppm, from
/// `mz_start` up to `mz_end`. Used to quantise fragment m/z for the IDF.
pub struct MzIndex {
    pub mz: Vec<f64>,
}

impl MzIndex {
    pub fn new() -> Self {
        Self {
            mz: ppm_index(MZ_RESOLUTION_PPM, MZ_START, MZ_END),
        }
    }

    /// Shared process-wide index (the reference uses a `once_cell::Lazy` static).
    pub fn global() -> &'static MzIndex {
        static GLOBAL: std::sync::OnceLock<MzIndex> = std::sync::OnceLock::new();
        GLOBAL.get_or_init(MzIndex::new)
    }

    /// Index of the m/z bin closest to `mz` (`find_closest_index`).
    pub fn find_closest_index(&self, mz: f32) -> usize {
        let mz = mz as f64;
        let arr = &self.mz;
        // first index with arr[i] >= mz
        let left = arr.partition_point(|&x| x < mz);
        if left == 0 {
            return 0;
        }
        if left == arr.len() {
            return arr.len() - 1;
        }
        let lo = (arr[left] - mz).abs();
        let hi = (arr[left - 1] - mz).abs();
        if lo < hi {
            left
        } else {
            left - 1
        }
    }
}

impl Default for MzIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// `ppm_index`: `[mz_start, mz_start*(1+ppm), mz_start*(1+ppm)^2, ...]`.
fn ppm_index(resolution_ppm: f64, mz_start: f64, mz_end: f64) -> Vec<f64> {
    let mz_start = mz_start.max(50.0);
    let mut index = Vec::new();
    index.push(mz_start);
    let mut current = mz_start;
    while current < mz_end {
        current += current * (resolution_ppm / 1e6);
        index.push(current);
    }
    index
}

/// Inverse document frequency over the global 1-ppm m/z index.
///
/// Built from the full fragment library: `idf(mz) = ln(total_fragments /
/// max(1, count_of_fragments_in_bin(mz)))`.
pub struct InverseDocumentFrequency {
    bin_counts: Vec<u32>,
    total_fragments: f32,
    mz_index: &'static MzIndex,
}

impl InverseDocumentFrequency {
    pub fn new(fragment_mz: &[f64]) -> Self {
        let mz_index = MzIndex::global();
        let total_fragments = fragment_mz.len() as f32;
        let mut bin_counts = vec![0u32; mz_index.mz.len()];
        for &mz in fragment_mz {
            let bin = mz_index.find_closest_index(mz as f32);
            bin_counts[bin] += 1;
        }
        Self {
            bin_counts,
            total_fragments,
            mz_index,
        }
    }

    /// IDF for each of the given fragment m/z values (`get_idf`).
    pub fn get_idf(&self, fragment_mz: &[f64]) -> Vec<f32> {
        if self.total_fragments == 0.0 {
            return vec![1.0; fragment_mz.len()];
        }
        fragment_mz
            .iter()
            .map(|&mz| {
                let bin = self.mz_index.find_closest_index(mz as f32);
                let df = self.bin_counts[bin].max(1) as f32;
                (self.total_fragments / df).ln()
            })
            .collect()
    }
}

// ===========================================================================
// 1-D Gaussian kernel + convolution
// ===========================================================================

/// 1-D Gaussian kernel (`GaussianKernel::new`): even width, normalised to sum 1.
pub struct GaussianKernel {
    pub array: Vec<f32>,
}

impl GaussianKernel {
    pub fn new(fwhm_rt: f64, sigma_scale_rt: f64, kernel_width: usize, rt_resolution: f64) -> Self {
        // Ensure the kernel dimension is even.
        let kernel_width = ((kernel_width as f64 / 2.0).ceil() as usize) * 2;
        let sigma = fwhm_rt / 2.3548;
        let rt_sigma = sigma * sigma_scale_rt / rt_resolution;
        Self {
            array: gaussian_kernel_1d(kernel_width, rt_sigma),
        }
    }
}

fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f32> {
    use std::f64::consts::PI;
    let half = (size / 2) as i32;
    let normalization = 1.0 / (sigma * (2.0 * PI).sqrt());
    let mut weights = vec![0.0f32; size];
    for x in -half..half {
        let exponent = -0.5 * (x as f64).powi(2) / (sigma * sigma);
        let w = normalization * exponent.exp();
        weights[(x + half) as usize] = w as f32;
    }
    let sum: f32 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }
    weights
}

/// 1-D Gaussian convolution per fragment row, no padding (zeros at the edges).
pub fn convolve_1d(xic: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (n_frag, n_points) = xic.dim();
    let mut out = Array2::<f32>::zeros((n_frag, n_points));
    let kernel_size = kernel.len();
    if n_frag == 0 || n_points == 0 || n_points < kernel_size {
        return out;
    }
    let half = kernel_size / 2;
    for f in 0..n_frag {
        for i in half..(n_points - half) {
            let mut s = xic[[f, i]] * kernel[half];
            for d in 1..=half {
                let g = kernel[half - d];
                s += (xic[[f, i - d]] + xic[[f, i + d]]) * g;
            }
            out[[f, i]] = s;
        }
    }
    out
}

/// `axis_log_dot_product`: `result[j] = sum_f ln(array[f][j] + 1) * weights[f]`.
pub fn axis_log_dot_product(array: &Array2<f32>, weights: &[f32]) -> Vec<f32> {
    let (n_rows, n_cols) = array.dim();
    debug_assert_eq!(n_rows, weights.len());
    let mut result = vec![0.0f32; n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            result[j] += (array[[i, j]] + 1.0).ln() * weights[i];
        }
    }
    result
}

/// Strict local maxima over two points each side (`find_local_maxima`).
///
/// A point `i` is a peak iff `a[i-2] < a[i-1] < a[i] > a[i+1] > a[i+2]`.
/// Returns `(indices_+_offset, values)` sorted by value descending.
pub fn find_local_maxima(array: &[f32], offset: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let len = array.len();
    if len < 5 {
        return (indices, values);
    }
    for i in 2..len - 2 {
        if array[i - 2] < array[i - 1]
            && array[i - 1] < array[i]
            && array[i] > array[i + 1]
            && array[i + 1] > array[i + 2]
        {
            indices.push(i + offset);
            values.push(array[i]);
        }
    }
    if !values.is_empty() {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let sorted_indices: Vec<usize> = order.iter().map(|&i| indices[i]).collect();
        let sorted_values: Vec<f32> = order.iter().map(|&i| values[i]).collect();
        (sorted_indices, sorted_values)
    } else {
        (indices, values)
    }
}

// ===========================================================================
// Small numeric helpers
// ===========================================================================

/// Median of a slice (average of the two middle values for even lengths).
pub fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) / 2.0
    } else {
        s[mid]
    }
}

/// Sample standard deviation (n-1 denominator).
pub fn std(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    var.sqrt()
}

/// Pearson correlation, safe (0 on zero variance / zero sums).
pub fn correlation_safe(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let x_sum: f32 = x.iter().sum();
    let y_sum: f32 = y.iter().sum();
    if x_sum == 0.0 || y_sum == 0.0 {
        return 0.0;
    }
    let x_mean = x_sum / x.len() as f32;
    let y_mean = y_sum / y.len() as f32;
    let mut x_var = 0.0f32;
    let mut y_var = 0.0f32;
    let mut cov = 0.0f32;
    for i in 0..x.len() {
        let xd = x[i] - x_mean;
        let yd = y[i] - y_mean;
        x_var += xd * xd;
        y_var += yd * yd;
        cov += xd * yd;
    }
    if x_var == 0.0 || y_var == 0.0 {
        return 0.0;
    }
    (cov / (x_var.sqrt() * y_var.sqrt())).clamp(-1.0, 1.0)
}
