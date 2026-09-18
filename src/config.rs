//! Configuration system — replaces `full_dia/cfg.py` + `full_dia/cfg/default.yaml`
//!
//! Instead of injecting variables into `globals()`, we build a typed `Config` struct
//! via `serde` deserialization from YAML, with sensible defaults.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::model::types;

// ---------------------------------------------------------------------------
// Global config instance (lazily initialised, immutable after bootstrap)
// ---------------------------------------------------------------------------

static GLOBAL_CONFIG: OnceLock<Config> = OnceLock::new();

/// Access the global configuration.
pub fn config() -> &'static Config {
    GLOBAL_CONFIG.get().expect("Config not initialised — call Config::init() first")
}

// ---------------------------------------------------------------------------
// Chemical constants from the YAML
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChemicalConfig {
    pub mass_proton: f32,
    pub mass_aa: HashMap<String, f32>,
}

impl Default for ChemicalConfig {
    fn default() -> Self {
        let mut mass_aa = HashMap::new();
        for (aa, mass) in [
            ("A", 89.0476792233), ("C", 160.030644505), ("D", 133.0375092233),
            ("E", 147.05315928710002), ("F", 165.07897935090006), ("G", 75.0320291595),
            ("H", 155.06947728710003), ("I", 131.0946294147), ("K", 146.10552844660003),
            ("L", 131.0946294147), ("M", 149.05105008089998), ("m", 165.04596508089998),
            ("N", 132.0534932552), ("P", 115.06332928709999), ("Q", 146.06914331900003),
            ("R", 174.11167644660003), ("S", 105.0425942233), ("T", 119.05824428710001),
            ("V", 117.0789793509), ("W", 204.0898783828), ("Y", 181.07389435090005),
            ("c", 178.04121404900002),
        ] {
            mass_aa.insert(aa.to_string(), mass);
        }
        Self { mass_proton: 1.007276466771, mass_aa }
    }
}

impl ChemicalConfig {
    pub fn mass_of(&self, aa: char) -> f32 {
        self.mass_aa.get(&aa.to_string()).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Main configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    // ── Identification ──────────────────────────────────────────────
    /// Maximum number of fragment ions per target peptide.
    pub fg_num: usize,
    /// Estimate tolerance by sampling from library.
    pub sample_ratio: f32,
    /// Q-value threshold from DIA-NN source code.
    pub rubbish_q_cut: f32,
    /// Candidate locus must have SA > ratio of the maximum.
    pub top_sa_cut: f32,
    /// Candidate locus must have Deep > ratio of the maximum.
    pub top_deep_cut: f32,
    /// 1/15 of the gradient as the initial tol_rt.
    pub tol_rt_ratio: f32,
    /// Initial tolerance of 1/K0 for centroid data.
    pub tol_im_xic_before_calib: f32,
    /// Refined tolerance of 1/K0.
    pub tol_im_xic_after_calib: f32,
    /// Current tolerance of 1/K0.
    pub tol_im_xic: f32,
    /// Initial tolerance of ppm for centroid or profile data.
    pub tol_ppm_before_calib: f32,
    pub tol_ppm: f32,
    /// Half-width of IM for DeepMap.
    pub tol_im_map: f32,
    /// Bin width in IM dimension for DeepMap.
    pub map_im_gap: f32,
    /// Dimension of IM in a map.
    pub map_im_dim: i64,
    /// Cycle dimension for map extraction.
    pub map_cycle_dim: i64,
    /// Early-stop patience for model refinement.
    pub patient: usize,
    /// Window size for SA computation (fixed at 7).
    pub window_points: usize,
    /// Which prs will be used for protein-group inference and scoring.
    pub q_cut_infer: f32,

    // ── Quantification ────────────────────────────────────────────
    /// Number of top fragment ions to select for cross-run quant.
    pub top_k_fg: usize,
    /// Number of top peptides to select for protein quant.
    pub top_k_pr: usize,

    // ── Efficiency ────────────────────────────────────────────────
    /// Batch size for analysis.
    pub target_batch_max: f64,

    // ── Chemical constants ─────────────────────────────────────────
    pub chemical: ChemicalConfig,

    // ── Runtime (set programmatically, not from yaml) ──────────────
    #[serde(skip)]
    pub gpu_id: i64,
    #[serde(skip)]
    pub device_name: String,
    #[serde(skip)]
    pub tol_rt: f32,
    #[serde(skip)]
    pub locus_rt_thre: f32,
    #[serde(skip)]
    pub locus_valid_num: f32,
    #[serde(skip)]
    pub is_compare_mode: bool,
    #[serde(skip)]
    pub is_overwrite: bool,
    #[serde(skip)]
    pub low_memory: bool,

    // ── Paths ──────────────────────────────────────────────────────
    #[serde(skip)]
    pub ws_global: PathBuf,
    #[serde(skip)]
    pub dir_out_name: String,
    #[serde(skip)]
    pub dir_out_global: PathBuf,
    #[serde(skip)]
    pub dir_out_single: PathBuf,
    #[serde(skip)]
    pub multi_ws: Vec<PathBuf>,
    #[serde(skip)]
    pub file_num: usize,
    #[serde(skip)]
    pub ws_single: PathBuf,

    // ── Batch sizes (adjusted by GPU VRAM) ─────────────────────────
    #[serde(skip)]
    pub batch_xic_seed: usize,
    #[serde(skip)]
    pub batch_xic_locus: usize,
    #[serde(skip)]
    pub batch_deep_center: usize,
    #[serde(skip)]
    pub batch_deep_big: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fg_num: types::FG_NUM,
            sample_ratio: 0.1,
            rubbish_q_cut: 0.5,
            top_sa_cut: 0.75,
            top_deep_cut: 0.66,
            tol_rt_ratio: 0.0667,
            tol_im_xic_before_calib: 0.08,
            tol_im_xic_after_calib: 0.05,
            tol_im_xic: 0.08,
            tol_ppm_before_calib: 20.0,
            tol_ppm: 20.0,
            tol_im_map: 0.025,
            map_im_gap: types::MAP_IM_GAP,
            map_im_dim: types::MAP_IM_DIM as i64,
            map_cycle_dim: types::MAP_CYCLE_DIM as i64,
            patient: 5,
            window_points: types::WINDOW_POINTS,
            q_cut_infer: 0.05,
            top_k_fg: 5,
            top_k_pr: 1,
            target_batch_max: 450_000.0,
            chemical: ChemicalConfig::default(),
            gpu_id: 0,
            device_name: String::new(),
            tol_rt: 0.0,
            locus_rt_thre: 0.0,
            locus_valid_num: 3.5,
            is_compare_mode: false,
            is_overwrite: false,
            low_memory: false,
            ws_global: PathBuf::new(),
            dir_out_name: "diaflux".to_string(),
            dir_out_global: PathBuf::new(),
            dir_out_single: PathBuf::new(),
            multi_ws: Vec::new(),
            file_num: 0,
            ws_single: PathBuf::new(),
            batch_xic_seed: 5000,
            batch_xic_locus: 10_000,
            batch_deep_center: 5000,
            batch_deep_big: 5000,
        }
    }
}

impl Config {
    /// Initialise config from optional YAML override file, then freeze.
    pub fn init(
        yaml_path: Option<&std::path::Path>,
        gpu_id: i64,
        low_memory: bool,
    ) -> &'static Self {
        let mut cfg = match yaml_path {
            Some(p) if p.is_file() => {
                let contents = std::fs::read_to_string(p)
                    .unwrap_or_else(|e| panic!("Cannot read config file {:?}: {}", p, e));
                serde_yaml::from_str(&contents)
                    .unwrap_or_else(|e| panic!("Invalid YAML in {:?}: {}", p, e))
            }
            _ => Config::default(),
        };

        cfg.gpu_id = gpu_id;
        cfg.low_memory = low_memory;

        if low_memory {
            cfg.target_batch_max /= 2.0;
        }

        GLOBAL_CONFIG.set(cfg).ok().expect("Config already initialised");
        config()
    }

    /// Update runtime tolerance values after calibration.
    pub fn set_tolerances(&mut self, tol_rt: f32, tol_im: f32, tol_ppm: f32) {
        // Since Config is behind OnceLock, we need interior mutability.
        // For simplicity, we store mutable tolerances separately.
        // This method exists as a placeholder — we'll use MutableState instead.
    }
}

// ---------------------------------------------------------------------------
// Mutable runtime state (tolerances that change during pipeline)
// ---------------------------------------------------------------------------

use std::sync::Mutex;

static MUTABLE: OnceLock<Mutex<MutableState>> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct MutableState {
    pub tol_rt: f32,
    pub tol_im_xic: f32,
    pub tol_ppm: f32,
    pub tol_im_xic_before_calib: f32,
    pub tol_ppm_before_calib: f32,
    pub locus_rt_thre: f32,
    pub rubbish_q_cut: f32,
}

impl MutableState {
    pub fn new(cfg: &Config) -> Self {
        Self {
            tol_rt: cfg.tol_rt,
            tol_im_xic: cfg.tol_im_xic,
            tol_ppm: cfg.tol_ppm,
            tol_im_xic_before_calib: cfg.tol_im_xic_before_calib,
            tol_ppm_before_calib: cfg.tol_ppm_before_calib,
            locus_rt_thre: cfg.locus_rt_thre,
            rubbish_q_cut: cfg.rubbish_q_cut,
        }
    }

    pub fn init() -> std::sync::MutexGuard<'static, MutableState> {
        MUTABLE
            .get_or_init(|| Mutex::new(MutableState::new(config())))
            .lock()
            .unwrap()
    }

    pub fn reset_tol_xic(&mut self) {
        self.tol_im_xic = self.tol_im_xic_before_calib;
    }

    pub fn reset_tol_ppm(&mut self) {
        self.tol_ppm = self.tol_ppm_before_calib;
    }
}

/// Convenience accessor.
pub fn mut_state() -> std::sync::MutexGuard<'static, MutableState> {
    MutableState::init()
}

/// Orderly shutdown — no-op in Rust.
pub fn reset() {
    // Nothing to do; OnceLock ensures single initialisation.
}
