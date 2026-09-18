//! FDR classifier (P4).
//!
//! Mirrors `FeedForwardNN` (BatchNorm1d → (Linear → ReLU → Dropout)×L →
//! Linear → Softmax) and `BinaryClassifierLegacyNewBatching` (Adam + BCELoss).
//!
//! core trains the network with `nn.BCELoss` on the softmax output against
//! one-hot labels `[1 - y, y]`. Because softmax output for two classes sums to
//! one, this loss is algebraically identical to cross-entropy on the
//! pre-softmax logits, so the forward pass exposes logits and the trainer
//! computes cross-entropy directly. This keeps the backward pass numerically
//! stable and equivalent to PyTorch.

use std::cmp::Ordering;
use std::collections::HashSet;

use ndarray::{Array2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::log::info;

const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;
const BN_EPS: f32 = 1e-5;
const BN_MOMENTUM: f32 = 0.1;

#[inline]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

// ===========================================================================
// Linear layer
// ===========================================================================

struct Linear {
    weight: Array2<f32>, // (out, in)
    bias: Vec<f32>,      // (out)
    m_w: Array2<f32>,
    v_w: Array2<f32>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
}

impl Linear {
    /// `nn.Linear` init: U(-1/√in, 1/√in).
    fn new(in_dim: usize, out_dim: usize, rng: &mut StdRng) -> Self {
        let bound = 1.0 / (in_dim as f32).sqrt();
        let weight = Array2::from_shape_fn((out_dim, in_dim), |_| {
            (rng.gen::<f32>() * 2.0 - 1.0) * bound
        });
        let bias: Vec<f32> = (0..out_dim)
            .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * bound)
            .collect();
        Linear {
            weight,
            m_w: Array2::zeros((out_dim, in_dim)),
            v_w: Array2::zeros((out_dim, in_dim)),
            bias,
            m_b: vec![0.0; out_dim],
            v_b: vec![0.0; out_dim],
        }
    }

    /// y = x · Wᵀ + b  (x: (N, in) → (N, out)).
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // `x.dot(&W.t())` uses ndarray's SIMD `matrixmultiply` backend, which is
        // an order of magnitude faster than the scalar triple loop this replaced.
        let mut y = x.dot(&self.weight.t());
        for (o, &b) in self.bias.iter().enumerate() {
            y.column_mut(o).mapv_inplace(|v| v + b);
        }
        y
    }
}

// ===========================================================================
// BatchNorm1d
// ===========================================================================

struct BatchNorm1d {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    m_gamma: Vec<f32>,
    v_gamma: Vec<f32>,
    m_beta: Vec<f32>,
    v_beta: Vec<f32>,
}

impl BatchNorm1d {
    fn new(dim: usize) -> Self {
        BatchNorm1d {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            running_mean: vec![0.0; dim],
            running_var: vec![1.0; dim],
            m_gamma: vec![0.0; dim],
            v_gamma: vec![0.0; dim],
            m_beta: vec![0.0; dim],
            v_beta: vec![0.0; dim],
        }
    }
}

// ===========================================================================
// FeedForwardNN
// ===========================================================================

struct ForwardCache {
    bn_out: Array2<f32>,   // (N, in) after gamma/beta
    bn_x_hat: Array2<f32>, // (N, in) normalized pre-gamma
    pre: Vec<Array2<f32>>,  // hidden pre-activations
    post: Vec<Array2<f32>>, // hidden post-relu (pre-dropout)
    out: Vec<Array2<f32>>,  // hidden post-dropout (next input)
    masks: Vec<Array2<f32>>, // dropout masks
}

/// `FeedForwardNN`: BatchNorm1d → (Linear → ReLU → Dropout)×L → Linear.
pub struct FeedForwardNN {
    dims: Vec<usize>, // [input, *hidden, output]
    bn: BatchNorm1d,
    linears: Vec<Linear>, // len = dims.len() - 1; last has no ReLU/Dropout
    dropout: f32,
    adam_t: usize,
}

impl FeedForwardNN {
    pub fn new(input_dim: usize, output_dim: usize, layers: &[usize], dropout: f32, seed: u64) -> Self {
        let mut dims = Vec::with_capacity(layers.len() + 2);
        dims.push(input_dim);
        dims.extend_from_slice(layers);
        dims.push(output_dim);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut linears = Vec::with_capacity(dims.len() - 1);
        for w in dims.windows(2) {
            linears.push(Linear::new(w[0], w[1], &mut rng));
        }

        FeedForwardNN {
            dims,
            bn: BatchNorm1d::new(input_dim),
            linears,
            dropout,
            adam_t: 0,
        }
    }

    /// Training-mode forward; updates running batch-norm statistics.
    fn forward_train(&mut self, x: &Array2<f32>, rng: &mut StdRng) -> (Array2<f32>, ForwardCache) {
        let (bn_out, bn_x_hat) = self.bn_train(x);
        let mut cache = ForwardCache {
            bn_out: bn_out.clone(),
            bn_x_hat,
            pre: Vec::new(),
            post: Vec::new(),
            out: Vec::new(),
            masks: Vec::new(),
        };
        let mut z = bn_out;
        let last = self.linears.len() - 1;
        for l in 0..last {
            let pre = self.linears[l].forward(&z);
            let post = pre.mapv(relu);
            let (out, mask) = dropout(&post, self.dropout, true, rng);
            cache.pre.push(pre);
            cache.post.push(post);
            cache.masks.push(mask);
            cache.out.push(out.clone());
            z = out;
        }
        let logits = self.linears[last].forward(&z);
        (logits, cache)
    }

    /// Eval-mode forward; uses running batch-norm statistics, no dropout.
    fn forward_eval(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut z = self.bn_eval(x);
        let last = self.linears.len() - 1;
        for l in 0..last {
            let pre = self.linears[l].forward(&z);
            z = pre.mapv(relu);
        }
        self.linears[last].forward(&z)
    }

    fn bn_train(&mut self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let (n, d) = x.dim();
        let mut out = Array2::zeros((n, d));
        let mut x_hat = Array2::zeros((n, d));
        for j in 0..d {
            let mut mu = 0.0f32;
            for i in 0..n {
                mu += x[[i, j]];
            }
            mu /= n as f32;
            let mut var = 0.0f32;
            for i in 0..n {
                let t = x[[i, j]] - mu;
                var += t * t;
            }
            var /= n as f32; // biased, matches PyTorch normalization
            let inv = 1.0 / (var + BN_EPS).sqrt();
            for i in 0..n {
                let xh = (x[[i, j]] - mu) * inv;
                x_hat[[i, j]] = xh;
                out[[i, j]] = self.bn.gamma[j] * xh + self.bn.beta[j];
            }
            // running stats use unbiased variance (PyTorch default)
            let unbiased = if n > 1 { var * n as f32 / (n - 1) as f32 } else { var };
            self.bn.running_mean[j] =
                (1.0 - BN_MOMENTUM) * self.bn.running_mean[j] + BN_MOMENTUM * mu;
            self.bn.running_var[j] =
                (1.0 - BN_MOMENTUM) * self.bn.running_var[j] + BN_MOMENTUM * unbiased;
        }
        (out, x_hat)
    }

    fn bn_eval(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n, d) = x.dim();
        let mut out = Array2::zeros((n, d));
        for j in 0..d {
            let inv = 1.0 / (self.bn.running_var[j] + BN_EPS).sqrt();
            for i in 0..n {
                out[[i, j]] =
                    self.bn.gamma[j] * (x[[i, j]] - self.bn.running_mean[j]) * inv + self.bn.beta[j];
            }
        }
        out
    }

    /// Backward pass through the cached forward + Adam step.
    fn backward_step(
        &mut self,
        cache: &ForwardCache,
        dlogits: &Array2<f32>,
        lr: f32,
        weight_decay: f32,
    ) {
        let n_layers = self.linears.len();
        let last = n_layers - 1;

        let z_in = if n_layers >= 2 {
            &cache.out[last - 1]
        } else {
            &cache.bn_out
        };
        let (mut dz, dw, db) = linear_backward(&self.linears[last].weight, z_in, dlogits);
        let mut d_weights = Vec::with_capacity(n_layers);
        let mut d_biases = Vec::with_capacity(n_layers);
        d_weights.push(dw);
        d_biases.push(db);

        for l in (0..last).rev() {
            let dz_masked = &dz * &cache.masks[l];
            let dz_relu = Array2::from_shape_fn(dz_masked.dim(), |(i, j)| {
                if cache.pre[l][[i, j]] > 0.0 {
                    dz_masked[[i, j]]
                } else {
                    0.0
                }
            });
            let x_in = if l == 0 { &cache.bn_out } else { &cache.out[l - 1] };
            let (dx, dw, db) = linear_backward(&self.linears[l].weight, x_in, &dz_relu);
            d_weights.push(dw);
            d_biases.push(db);
            dz = dx;
        }
        d_weights.reverse();
        d_biases.reverse();

        let (d_gamma, d_beta) = bn_backward(cache, &dz);

        self.adam_t += 1;
        let t = self.adam_t;
        let b1t = 1.0 - ADAM_B1.powi(t as i32);
        let b2t = 1.0 - ADAM_B2.powi(t as i32);

        for j in 0..self.bn.gamma.len() {
            self.bn.gamma[j] -= adam_update(
                self.bn.gamma[j],
                d_gamma[j],
                &mut self.bn.m_gamma[j],
                &mut self.bn.v_gamma[j],
                lr,
                weight_decay,
                b1t,
                b2t,
            );
        }
        for j in 0..self.bn.beta.len() {
            self.bn.beta[j] -= adam_update(
                self.bn.beta[j],
                d_beta[j],
                &mut self.bn.m_beta[j],
                &mut self.bn.v_beta[j],
                lr,
                weight_decay,
                b1t,
                b2t,
            );
        }
        for l in 0..n_layers {
            let (out_d, in_d) = self.linears[l].weight.dim();
            let lin = &mut self.linears[l];
            for o in 0..out_d {
                for k in 0..in_d {
                    let u = adam_update(
                        lin.weight[[o, k]],
                        d_weights[l][[o, k]],
                        &mut lin.m_w[[o, k]],
                        &mut lin.v_w[[o, k]],
                        lr,
                        weight_decay,
                        b1t,
                        b2t,
                    );
                    lin.weight[[o, k]] -= u;
                }
                let u = adam_update(
                    lin.bias[o],
                    d_biases[l][o],
                    &mut lin.m_b[o],
                    &mut lin.v_b[o],
                    lr,
                    weight_decay,
                    b1t,
                    b2t,
                );
                lin.bias[o] -= u;
            }
        }
    }
}

// ===========================================================================
// Primitive ops
// ===========================================================================

fn dropout(x: &Array2<f32>, p: f32, training: bool, rng: &mut StdRng) -> (Array2<f32>, Array2<f32>) {
    let (n, d) = x.dim();
    if !training || p <= 0.0 {
        return (x.clone(), Array2::ones((n, d)));
    }
    let scale = 1.0 / (1.0 - p);
    let mut mask = Array2::zeros((n, d));
    let mut out = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            let keep = if rng.gen::<f32>() < p { 0.0 } else { scale };
            mask[[i, j]] = keep;
            out[[i, j]] = x[[i, j]] * keep;
        }
    }
    (out, mask)
}

fn softmax(logits: &Array2<f32>) -> Array2<f32> {
    let (n, c) = logits.dim();
    let mut out = Array2::zeros((n, c));
    for i in 0..n {
        let mut max = f32::NEG_INFINITY;
        for j in 0..c {
            max = max.max(logits[[i, j]]);
        }
        let mut sum = 0.0f32;
        for j in 0..c {
            let e = (logits[[i, j]] - max).exp();
            out[[i, j]] = e;
            sum += e;
        }
        for j in 0..c {
            out[[i, j]] /= sum;
        }
    }
    out
}

fn linear_backward(
    w: &Array2<f32>,
    x: &Array2<f32>,
    dout: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>, Vec<f32>) {
    // dx = dout · W (n, in), dw = doutᵀ · x (out, in), db = Σᵢ dout (out).
    let dx = dout.dot(w);
    let dw = dout.t().dot(x);
    let db = dout.sum_axis(Axis(0)).to_vec();
    (dx, dw, db)
}

fn bn_backward(cache: &ForwardCache, d_bn_out: &Array2<f32>) -> (Vec<f32>, Vec<f32>) {
    let (n, d) = cache.bn_x_hat.dim();
    let mut d_gamma = vec![0.0f32; d];
    let mut d_beta = vec![0.0f32; d];
    for j in 0..d {
        let mut sg = 0.0f32;
        let mut sb = 0.0f32;
        for i in 0..n {
            sb += d_bn_out[[i, j]];
            sg += d_bn_out[[i, j]] * cache.bn_x_hat[[i, j]];
        }
        d_gamma[j] = sg;
        d_beta[j] = sb;
    }
    (d_gamma, d_beta)
}

/// Adam update with L2 weight decay (PyTorch `optim.Adam` semantics).
#[inline]
#[allow(clippy::too_many_arguments)]
fn adam_update(
    param: f32,
    grad: f32,
    m: &mut f32,
    v: &mut f32,
    lr: f32,
    weight_decay: f32,
    b1t: f32,
    b2t: f32,
) -> f32 {
    let g = grad + weight_decay * param;
    *m = ADAM_B1 * *m + (1.0 - ADAM_B1) * g;
    *v = ADAM_B2 * *v + (1.0 - ADAM_B2) * g * g;
    let m_hat = *m / b1t;
    let v_hat = *v / b2t;
    lr * m_hat / (v_hat.sqrt() + ADAM_EPS)
}

// ===========================================================================
// BinaryClassifier
// ===========================================================================

/// `BinaryClassifierLegacyNewBatching` — wraps `FeedForwardNN`.
pub struct BinaryClassifier {
    pub input_dim: usize,
    pub output_dim: usize,
    pub test_size: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub layers: Vec<usize>,
    pub dropout: f32,
    pub metric_interval: usize,
    pub seed: u64,
    network: Option<FeedForwardNN>,
    fitted: bool,
}

impl BinaryClassifier {
    /// Defaults for `BinaryClassifierLegacyNewBatching` as instantiated
    /// by the PeptideCentricWorkflow `_get_classifier_base` (NOT the class-body
    /// defaults): `test_size=0.001`, `batch_size=5000`, `learning_rate=0.001`.
    pub fn new(seed: u64) -> Self {
        BinaryClassifier {
            input_dim: 10,
            output_dim: 2,
            test_size: 0.001,
            batch_size: 5000,
            epochs: 10,
            learning_rate: 0.001,
            weight_decay: 0.00001,
            layers: vec![100, 50, 20, 5],
            dropout: 0.001,
            metric_interval: 1000,
            seed,
            network: None,
            fitted: false,
        }
    }

    pub fn fitted(&self) -> bool {
        self.fitted
    }

    /// `fit`: initialize network, inner train/test split, Adam + BCELoss training.
    pub fn fit(&mut self, x: &Array2<f32>, y: &[f32]) {
        let n = x.nrows();
        let f = x.ncols();

        if self.network.is_none() || self.input_dim != f {
            self.input_dim = f;
            self.network = Some(FeedForwardNN::new(
                self.input_dim,
                self.output_dim,
                &self.layers,
                self.dropout,
                self.seed,
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(1));

        // The reference trains the MLP on the full candidate set on a GPU. On
        // CPU, that is ~1.5 h per FDR stage for tens of millions of candidates,
        // so subsample the training rows to bound cost. The legacy `test_size`
        // hold-out is ignored here: FDR control comes from target-decoy
        // competition, not a held-out set, and holding data out only starves the
        // classifier of the rare high-signal targets it must rank.
        let max_train = std::env::var("DIAFORGE_FDR_MAX_TRAIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2_000_000);
        // Optional signal-based oversampling: reserve a fraction of the
        // training budget for the highest-signal candidates so the rare
        // real-target signal (~0.5% of candidates) is not drowned by the
        // ~99.5% noise apexes that dominate a uniform subsample. Signal is the
        // feature column with the largest |target − decoy| mean gap.
        let oversample_frac: Option<f64> = std::env::var("DIAFORGE_FDR_OVERSAMPLE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|&f| f > 0.0 && f < 1.0);
        let (x_train, y_train) = if n > max_train {
            let keep: Vec<usize> = match oversample_frac {
                Some(frac) => {
                    let (sig_col, higher_is_target) =
                        signal_column(x, y, 200_000, &mut rng);
                    info!(
                        "fdr: oversampling training rows (frac={frac:.2}, signal col={sig_col}, higher_is_target={higher_is_target})"
                    );
                    oversample_keep(x, sig_col, higher_is_target, n, max_train, frac, &mut rng)
                }
                None => {
                    let mut keep: Vec<usize> = (0..n).collect();
                    keep.shuffle(&mut rng);
                    keep.truncate(max_train);
                    keep
                }
            };
            (
                gather_rows(x, &keep),
                keep.iter().map(|&i| y[i]).collect(),
            )
        } else {
            (x.to_owned(), y.to_vec())
        };

        let n_train = x_train.nrows();
        let num_batches = (n_train / self.batch_size).saturating_sub(1);
        let y_onehot = Array2::from_shape_fn((n_train, 2), |(i, c)| {
            if c == 0 {
                1.0 - y_train[i]
            } else {
                y_train[i]
            }
        });

        let network = self.network.as_mut().unwrap();
        for _epoch in 0..self.epochs {
            let mut order: Vec<usize> = (0..num_batches).collect();
            order.shuffle(&mut rng);
            for &b in &order {
                let start = b * self.batch_size;
                let stop = start + self.batch_size;
                let xb = x_train.slice(ndarray::s![start..stop, ..]).to_owned();
                let yb = y_onehot.slice(ndarray::s![start..stop, ..]).to_owned();
                let (logits, cache) = network.forward_train(&xb, &mut rng);
                let p = softmax(&logits);
                let dlogits = (&p - &yb).mapv(|v| v / self.batch_size as f32);
                network.backward_step(&cache, &dlogits, self.learning_rate, self.weight_decay);
            }
        }
        self.fitted = true;
    }

    /// `predict_proba`: eval-mode softmax over the full input (n, 2).
    pub fn predict_proba(&self, x: &Array2<f32>) -> Array2<f32> {
        let network = self
            .network
            .as_ref()
            .expect("classifier has not been fitted");
        let n = x.nrows();
        // Eval-mode forward is row-independent, so split the (potentially tens
        // of millions of) candidates into chunks and evaluate on all cores. Per
        // row results are bit-identical to the sequential path.
        const CHUNK: usize = 256_000;
        let logits = if n <= CHUNK {
            network.forward_eval(x)
        } else {
            let n_chunks = (n + CHUNK - 1) / CHUNK;
            let parts: Vec<Array2<f32>> = (0..n_chunks)
                .into_par_iter()
                .map(|ci| {
                    let start = ci * CHUNK;
                    let stop = (start + CHUNK).min(n);
                    let xb = x.slice(ndarray::s![start..stop, ..]).to_owned();
                    network.forward_eval(&xb)
                })
                .collect();
            let mut logits = Array2::zeros((n, 2));
            for (ci, part) in parts.into_iter().enumerate() {
                let start = ci * CHUNK;
                let stop = (start + CHUNK).min(n);
                logits
                    .slice_mut(ndarray::s![start..stop, ..])
                    .assign(&part);
            }
            logits
        };
        softmax(&logits)
    }
}

/// Gather rows `idx` of `x` into a new array.
pub(crate) fn gather_rows(x: &Array2<f32>, idx: &[usize]) -> Array2<f32> {
    let ncols = x.ncols();
    let data: Vec<f32> = (0..idx.len() * ncols)
        .into_par_iter()
        .map(|k| x[[idx[k / ncols], k % ncols]])
        .collect();
    Array2::from_shape_vec((idx.len(), ncols), data).expect("gather_rows: shape mismatch")
}

/// Pick the feature column whose extreme tail is most enriched in targets.
///
/// Real targets (~0.5% of candidates) live in the extreme of co-elution
/// features (e.g. `num_over_50`, `median_correlation`), so the mean target−decoy
/// gap is near zero even for informative features. We instead measure the
/// target fraction in the top/bottom 2% of each column's distribution and pick
/// the column with the strongest tail enrichment. Returns `(column,
/// higher_is_target)`, where `higher_is_target` is true when the *high* tail is
/// the target-enriched one.
fn signal_column(x: &Array2<f32>, y: &[f32], sample_cap: usize, rng: &mut StdRng) -> (usize, bool) {
    let n = x.nrows();
    let f = x.ncols();
    if n == 0 || f == 0 {
        return (0, true);
    }
    let mut idx: Vec<usize> = (0..n).collect();
    if n > sample_cap {
        idx.shuffle(rng);
        idx.truncate(sample_cap);
    }
    let m = idx.len();
    let k = (m / 50).max(100).min(m / 2); // top/bottom 2%

    let mut best_col = 0usize;
    let mut best_enrich = 0.5f64; // baseline target fraction is ~0.5
    let mut higher_is_target = true;
    for c in 0..f {
        let mut order: Vec<usize> = idx.clone();
        order.sort_unstable_by(|&a, &b| {
            x[[a, c]]
                .partial_cmp(&x[[b, c]])
                .unwrap_or(Ordering::Equal)
        });
        let top_t = order[m - k..].iter().filter(|&&i| y[i] == 0.0).count() as f64 / k as f64;
        let bot_t = order[..k].iter().filter(|&&i| y[i] == 0.0).count() as f64 / k as f64;
        let (frac, high_is_target) = if top_t >= bot_t {
            (top_t, true)
        } else {
            (bot_t, false)
        };
        if frac > best_enrich {
            best_enrich = frac;
            best_col = c;
            higher_is_target = high_is_target;
        }
    }
    (best_col, higher_is_target)
}

/// Build a `max_train`-row training index: keep the `frac · max_train` rows
/// with the highest signal, then fill the remainder with a uniform sample of
/// the rest. `signal` is `x[.., sig_col]` (negated when targets have the lower
/// mean), so the kept rows are the most target-like candidates.
fn oversample_keep(
    x: &Array2<f32>,
    sig_col: usize,
    higher_is_target: bool,
    n: usize,
    max_train: usize,
    frac: f64,
    rng: &mut StdRng,
) -> Vec<usize> {
    let n_high = ((frac * max_train as f64) as usize).min(n);
    let mut pairs: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = x[[i, sig_col]];
            (if higher_is_target { v } else { -v }, i)
        })
        .collect();
    pairs.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    let mut keep: Vec<usize> = pairs[..n_high].iter().map(|&(_, i)| i).collect();
    let mut rest: Vec<usize> = pairs[n_high..].iter().map(|&(_, i)| i).collect();
    rest.shuffle(rng);
    let n_rest = max_train - n_high;
    if n_rest < rest.len() {
        rest.truncate(n_rest);
    }
    keep.extend_from_slice(&rest);
    keep
}

// =====================================================================
// Target-decoy FDR control (folded from the former `fdr` submodule).
// =====================================================================

/// `FDRManager.fit_predict` / `perform_fdr` options (single-channel default:
/// `decoy_strategy = "precursor"`, `competitive = true`, `group_channels = true`).
#[derive(Debug, Clone)]
pub struct FdrConfig {
    pub competitive: bool,
    pub group_channels: bool,
    pub seed: u64,
    /// Skip training when the classifier is already fitted (reuse stage-1
    /// weights for stage 2 instead of retraining on the stage-2 candidate set).
    pub reuse_fitted: bool,
}

impl Default for FdrConfig {
    fn default() -> Self {
        FdrConfig {
            competitive: true,
            group_channels: true,
            seed: 0,
            reuse_fitted: false,
        }
    }
}

/// Per-candidate FDR output.
pub struct FdrOutput {
    /// Decoy probability (`predict_proba[:, 1]`) per input candidate; `1.0` for
    /// candidates dropped due to missing features.
    pub proba: Vec<f32>,
    /// Final q-value per input candidate; `1.0` for candidates that were not
    /// retained by `keep_best` (or dropped).
    pub qval: Vec<f32>,
    /// Indices (into the input arrays) of the retained best-PSM-per-group set.
    pub retained: Vec<usize>,
}

/// `perform_fdr`: target-decoy classification, q-value estimation, and
/// `keep_best` grouping.
pub fn perform_fdr(
    classifier: &mut BinaryClassifier,
    features: &Array2<f32>,
    decoy: &[u8],
    precursor_idx: &[u32],
    elution_group_idx: &[u32],
    channel: &[u8],
    config: &FdrConfig,
) -> FdrOutput {
    let n = features.nrows();
    let mut proba = vec![1.0f32; n];
    let mut qval = vec![1.0f32; n];

    // dropna(subset=available_columns)
    let valid: Vec<usize> = (0..n)
        .into_par_iter()
        .filter(|&i| features.row(i).iter().all(|v| v.is_finite()))
        .collect();
    let m = valid.len();

    let retained = if m < 2 {
        Vec::new()
    } else {
        let x = gather_rows(features, &valid);
        let y: Vec<f32> = valid.iter().map(|&i| decoy[i] as f32).collect();
        let n_target = y.iter().filter(|&&v| v == 0.0).count();
        let n_decoy = y.iter().filter(|&&v| v == 1.0).count();

        if n_target == 0 || n_decoy == 0 {
            Vec::new()
        } else {
            // Train on the full candidate set (in-sample classification): FDR
            // control comes from target-decoy competition in `get_q_values`,
            // not from a held-out split, so a 20% hold-out only starves the
            // classifier of the rare high-signal targets it must rank.
            if !config.reuse_fitted || !classifier.fitted() {
                classifier.fit(&x, &y);
            }

            let p_full = classifier.predict_proba(&x);
            let proba_valid: Vec<f32> = (0..m).into_par_iter().map(|i| p_full[[i, 1]]).collect();
            for i in 0..m {
                proba[valid[i]] = proba_valid[i];
            }

            let group_ids: Vec<u64> = (0..m)
                .into_par_iter()
                .map(|i| {
                    group_key(
                        config.competitive,
                        config.group_channels,
                        precursor_idx[valid[i]],
                        elution_group_idx[valid[i]],
                        channel[valid[i]],
                    )
                })
                .collect();

            let retained_valid = keep_best(&proba_valid, &group_ids);

            let r_proba: Vec<f32> = retained_valid.iter().map(|&i| proba_valid[i]).collect();
            let r_decoy: Vec<u8> = retained_valid.iter().map(|&i| decoy[valid[i]]).collect();
            let r_prec: Vec<u32> = retained_valid
                .iter()
                .map(|&i| precursor_idx[valid[i]])
                .collect();
            let q_retained = get_q_values(&r_proba, &r_decoy, &r_prec);
            for (k, &ri) in retained_valid.iter().enumerate() {
                qval[valid[ri]] = q_retained[k];
            }

            retained_valid.iter().map(|&ri| valid[ri]).collect()
        }
    };

    FdrOutput {
        proba,
        qval,
        retained,
    }
}

/// `get_q_values`: sort by (score asc, decoy asc, precursor_idx asc), cumulative
/// decoy/target ratio, then `_fdr_to_q_values`. Returns q-values aligned to the
/// input slice order.
pub fn get_q_values(score: &[f32], decoy: &[u8], precursor_idx: &[u32]) -> Vec<f32> {
    let n = score.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        score[a]
            .partial_cmp(&score[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| decoy[a].cmp(&decoy[b]))
            .then_with(|| precursor_idx[a].cmp(&precursor_idx[b]))
    });

    let mut fdr_sorted = vec![0.0f32; n];
    let mut target_cum = 0usize;
    let mut decoy_cum = 0usize;
    for (pos, &i) in order.iter().enumerate() {
        if decoy[i] == 0 {
            target_cum += 1;
        } else {
            decoy_cum += 1;
        }
        fdr_sorted[pos] = if target_cum > 0 {
            decoy_cum as f32 / target_cum as f32
        } else {
            f32::INFINITY
        };
    }

    let q_sorted = fdr_to_q_values(&fdr_sorted);
    let mut result = vec![0.0f32; n];
    for (pos, &i) in order.iter().enumerate() {
        result[i] = q_sorted[pos];
    }
    result
}

/// `_fdr_to_q_values`: for each element, the minimum FDR of itself and all
/// subsequent (worse-scoring) entries.
fn fdr_to_q_values(fdr: &[f32]) -> Vec<f32> {
    let n = fdr.len();
    let mut q = vec![0.0f32; n];
    let mut running_min = f32::INFINITY;
    for j in (0..n).rev() {
        running_min = running_min.min(fdr[j]);
        q[j] = running_min;
    }
    q
}

/// `keep_best`: sort by (score asc, group asc), keep the first (lowest-score)
/// row of each group. Returns retained indices in original (ascending) order.
pub fn keep_best(score: &[f32], group_ids: &[u64]) -> Vec<usize> {
    let n = score.len();
    let mut order: Vec<usize> = (0..n).collect();
    // Parallel stable sort (rayon) — same order as the sequential `sort_by`.
    order.par_sort_by(|&a, &b| {
        score[a]
            .partial_cmp(&score[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| group_ids[a].cmp(&group_ids[b]))
    });
    let mut seen = HashSet::new();
    let mut retained = Vec::new();
    for &i in &order {
        if seen.insert(group_ids[i]) {
            retained.push(i);
        }
    }
    retained.sort_unstable();
    retained
}

/// Group key matching `get_group_columns`:
/// competitive+group_channels → (elution_group_idx, channel), competitive →
/// elution_group_idx, else precursor_idx.
#[inline]
fn group_key(
    competitive: bool,
    group_channels: bool,
    precursor_idx: u32,
    elution_group_idx: u32,
    channel: u8,
) -> u64 {
    if competitive {
        if group_channels {
            ((elution_group_idx as u64) << 8) | (channel as u64)
        } else {
            elution_group_idx as u64
        }
    } else {
        precursor_idx as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 0.0, 2.0]).unwrap();
        let p = softmax(&logits);
        for i in 0..2 {
            let s: f32 = p.row(i).sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn classifier_separates_classes() {
        // Targets: 46-dim vectors centered at +1; decoys centered at -1.
        let f = 46usize;
        let n = 400usize; // n_train/1000 = 0 -> no batches; use direct forward test instead
        let mut x = Array2::zeros((n, f));
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            let is_decoy = i >= n / 2;
            y[i] = is_decoy as u8 as f32;
            for j in 0..f {
                let center = if is_decoy { -1.0 } else { 1.0 };
                x[[i, j]] = center + ((i * 7 + j * 13) % 10) as f32 * 0.05;
            }
        }

        let mut clf = BinaryClassifier::new(42);
        clf.batch_size = 64; // shrink so training actually runs on 400 samples
        clf.epochs = 5;
        clf.fit(&x, &y);
        assert!(clf.fitted());

        let p = clf.predict_proba(&x);
        // decoy probability (col 1) should be higher for decoys on average
        let t_mean: f32 = (0..n / 2).map(|i| p[[i, 1]]).sum::<f32>() / (n / 2) as f32;
        let d_mean: f32 = (n / 2..n).map(|i| p[[i, 1]]).sum::<f32>() / (n / 2) as f32;
        assert!(
            d_mean > t_mean,
            "decoy proba should exceed target proba (t={t_mean:.3} d={d_mean:.3})"
        );
    }
}


#[cfg(test)]
mod tests_fdr {
    use super::*;

    #[test]
    fn fdr_to_qvalues_is_monotonic_min() {
        let fdr = vec![0.5, 0.2, 0.4, 0.1];
        let q = fdr_to_q_values(&fdr);
        // q[j] = min(fdr[j..])
        assert!((q[0] - 0.1).abs() < 1e-6);
        assert!((q[1] - 0.1).abs() < 1e-6);
        assert!((q[2] - 0.1).abs() < 1e-6);
        assert!((q[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn get_qvalues_target_decoy() {
        // best (lowest decoy proba) = targets
        let score = vec![0.05, 0.1, 0.6, 0.7]; // ascending already
        let decoy = vec![0u8, 0, 1, 1];
        let prec = vec![0u32, 1, 2, 3];
        let q = get_q_values(&score, &decoy, &prec);
        // cumulative: T=1 (fdr 0), T=2 (fdr 0), D=1/T=2 (0.5), D=2/T=2 (1.0)
        // fdr = [0,0,0.5,1.0] -> q = min from right = [0,0,0.5,1.0]
        assert!((q[0]).abs() < 1e-6);
        assert!((q[1]).abs() < 1e-6);
        assert!((q[2] - 0.5).abs() < 1e-6);
        assert!((q[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keep_best_keeps_lowest_score_per_group() {
        let score = vec![0.9, 0.1, 0.8, 0.2, 0.7];
        let groups = vec![1u64, 1, 2, 2, 3];
        let retained = keep_best(&score, &groups);
        assert_eq!(retained, vec![1, 3, 4]);
    }
}
