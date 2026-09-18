//! Core data types — replaces pandas DataFrame-based data structures.
//!
//! The Python version used `pd.DataFrame` as a universal container with string-column
//! addressing (e.g. `cols = ["fg_mz_" + str(i) for i in range(fg_num)]`).
//! Here every entity is an explicit Rust struct with statically-typed fields.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Fragment-ion annotation
// ---------------------------------------------------------------------------

/// Type of fragment ion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IonType {
    B = 1,
    Y = 2,
    Loss = 3,
}

/// Encoded fragment annotation.
///
/// In Python this was stored as a packed integer: `type * 1000 + index * 10 + charge`.
/// We use an explicit struct instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IonAnnotation {
    pub ion_type: IonType,
    pub series_number: u8, // e.g. b12 → 12
    pub charge: u8,
}

impl IonAnnotation {
    pub fn from_packed(packed: i16) -> Self {
        let charge = (packed % 10) as u8;
        let series_number = ((packed / 10) % 100) as u8;
        // DIA-NN: type & 3 gives base ion type (1=B, 2=Y, 3=Loss)
        // Higher bits are additional flags (e.g., type=5 means Y with some flag)
        let raw_type = packed / 1000;
        let ion_type = match raw_type & 3 {
            1 => IonType::B,
            2 => IonType::Y,
            _ => IonType::Loss, // 0 or 3 → Loss
        };
        Self {
            ion_type,
            series_number,
            charge,
        }
    }

    pub fn to_packed(&self) -> i16 {
        (self.ion_type as i16) * 1000 + (self.series_number as i16) * 10 + (self.charge as i16)
    }
}

// ---------------------------------------------------------------------------
// Fragment ion data
// ---------------------------------------------------------------------------

/// A single fragment ion belonging to a precursor.
#[derive(Debug, Clone)]
pub struct FragmentIon {
    pub mz: f32,
    pub height: f32,
    pub annotation: IonAnnotation,
    // Isotope shifted m/z values (computed on demand)
    pub mz_left: f32,
    pub mz_1h: f32,
    pub mz_2h: f32,
}

// ---------------------------------------------------------------------------
// Precursor (peptide-level) data
// ---------------------------------------------------------------------------

/// A single precursor (peptide + charge) loaded from the spectral library.
#[derive(Debug, Clone)]
pub struct Precursor {
    pub pr_id: String,
    pub pr_charge: u8,
    pub pr_mz: f32,
    pub pr_len: u8,
    pub pred_irt: f32,
    pub pred_iim: f32,
    pub fg_num: u8,
    pub pr_index: u32,
    pub fragment_ions: Vec<FragmentIon>,
    // Extensions during pipeline
    pub simple_seq: String,
    pub strip_seq: String,
    /// `;`-separated modification names in alphabase `ModName@AA` format (e.g.
    /// `Carbamidomethyl@C;Oxidation@M`). Empty for unmodified precursors.
    pub mods: String,
    /// `;`-separated 1-indexed modification sites (`0` = N-term).
    pub mod_sites: String,
    pub swath_id: u8,
    pub decoy: u8,
    /// Elution group: targets are `0..n_target`; a decoy shares the index of the
    /// target it was generated from (competitive-FDR grouping).
    pub elution_group_idx: u32,
    // Calibrated values
    pub pred_rt: f32,
    pub pred_im: f32,
    pub measure_im: f32,
    pub measure_rt: f32,
    // Isotope m/z for the precursor itself
    pub pr_mz_left: f32,
    pub pr_mz_1h: f32,
    pub pr_mz_2h: f32,
}

impl Default for Precursor {
    fn default() -> Self {
        Self {
            pr_id: String::new(),
            pr_charge: 0,
            pr_mz: 0.0,
            pr_len: 0,
            pred_irt: 0.0,
            pred_iim: 0.0,
            fg_num: 0,
            pr_index: 0,
            fragment_ions: Vec::new(),
            simple_seq: String::new(),
            strip_seq: String::new(),
            mods: String::new(),
            mod_sites: String::new(),
            swath_id: 0,
            decoy: 0,
            elution_group_idx: 0,
            pred_rt: 0.0,
            pred_im: 0.0,
            measure_im: 0.0,
            measure_rt: 0.0,
            pr_mz_left: 0.0,
            pr_mz_1h: 0.0,
            pr_mz_2h: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Elution group (a candidate peak group at a specific retention-time locus)
// ---------------------------------------------------------------------------

/// A candidate elution group (PSM-level candidate) for a precursor.
#[derive(Debug, Clone)]
pub struct ElutionGroup {
    pub pr_index: u32,
    pub pr_id: String,
    pub swath_id: u8,
    pub decoy: u8,

    /// The cycle index (locus) at which this elution group is centred.
    pub locus: u32,
    /// Extracted XIC intensities: [n_ion, n_cycle].
    pub xics: Option<ndarray::Array3<f32>>,
    /// Ion mobility values at each cycle.
    pub ims: Option<ndarray::Array3<f32>>,
    /// Measured m/z values at each cycle.
    pub mzs: Option<ndarray::Array3<f32>>,
    /// Retention time at each cycle.
    pub rts: Option<Vec<f32>>,

    // Measured values at apex
    pub measure_rt: f32,
    pub measure_im: f32,
    pub measure_pr_mz: f32,

    // SA scores (co-elution scores)
    pub sa_score: f32,
    pub sa_scores_ion: Vec<f32>,

    // Deep-learning scores
    pub score_center_deep: f32,
    pub score_left_deep: f32,
    pub score_1h_deep: f32,
    pub score_2h_deep: f32,
    pub score_big_deep: f32,

    // Refined deep scores
    pub score_center_deep_refine: f32,
    pub score_left_deep_refine: f32,
    pub score_1h_deep_refine: f32,
    pub score_2h_deep_refine: f32,
    pub score_big_deep_refine: f32,

    // Mall score
    pub score_mall: f32,

    // Comprehensive scores computed by the scoring module
    pub scores: HashMap<String, f32>,

    // C-score (combined discriminant score) and q-value
    pub cscore_pr_run: f32,
    pub q_pr_run: f32,
    pub cscore_pr_global: f32,
    pub q_pr_global: f32,

    // Group rank (1 = best candidate for this precursor)
    pub group_rank: u8,
    // Protein group assignment (set during assembly)
    pub protein_group: String,
}

impl ElutionGroup {
    pub fn new(precursor: &Precursor) -> Self {
        Self {
            pr_index: precursor.pr_index,
            pr_id: precursor.pr_id.clone(),
            swath_id: precursor.swath_id,
            decoy: precursor.decoy,
            locus: 0,
            xics: None,
            ims: None,
            mzs: None,
            rts: None,
            measure_rt: 0.0,
            measure_im: 0.0,
            measure_pr_mz: 0.0,
            sa_score: 0.0,
            sa_scores_ion: Vec::new(),
            score_center_deep: 0.0,
            score_left_deep: 0.0,
            score_1h_deep: 0.0,
            score_2h_deep: 0.0,
            score_big_deep: 0.0,
            score_center_deep_refine: 0.0,
            score_left_deep_refine: 0.0,
            score_1h_deep_refine: 0.0,
            score_2h_deep_refine: 0.0,
            score_big_deep_refine: 0.0,
            score_mall: 0.0,
            scores: HashMap::new(),
            cscore_pr_run: 0.0,
            q_pr_run: 1.0,
            cscore_pr_global: 0.0,
            q_pr_global: 1.0,
            group_rank: 0,
            protein_group: String::new(),
        }
    }

    /// Create a minimal elution group from (pr_index, pr_id, decoy, swath).
    pub fn new_pr(pr_index: u32, pr_id: &str, decoy: u8, swath_id: u8) -> Self {
        let mut scores: HashMap<String, f32> = HashMap::new();
        scores.insert("cscore_pr_global".into(), 0.0f32);
        scores.insert("q_pr_global".into(), 1.0f32);
        Self {
            pr_index,
            pr_id: pr_id.to_string(),
            swath_id,
            decoy,
            locus: 0,
            xics: None,
            ims: None,
            mzs: None,
            rts: None,
            measure_rt: 0.0,
            measure_im: 0.0,
            measure_pr_mz: 0.0,
            sa_score: 0.0,
            sa_scores_ion: Vec::new(),
            score_center_deep: 0.0,
            score_left_deep: 0.0,
            score_1h_deep: 0.0,
            score_2h_deep: 0.0,
            score_big_deep: 0.0,
            score_center_deep_refine: 0.0,
            score_left_deep_refine: 0.0,
            score_1h_deep_refine: 0.0,
            score_2h_deep_refine: 0.0,
            score_big_deep_refine: 0.0,
            score_mall: 0.0,
            scores: HashMap::new(),
            cscore_pr_run: 0.0,
            q_pr_run: 1.0,
            cscore_pr_global: 0.0,
            q_pr_global: 1.0,
            group_rank: 0,
            protein_group: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Protein information
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ProteinInfo {
    pub protein_id: String,
    pub protein_name: String,
    pub gene: String,
    pub is_decoy: bool,
}

// ---------------------------------------------------------------------------
// Peptide-to-protein mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PeptideProteinMap {
    pub pr_index: u32,
    pub protein_id: String,
    pub protein_name: String,
    pub gene: String,
}

// ---------------------------------------------------------------------------
// SWATH / DIA window
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiaWindow {
    pub swath_id: u8,
    pub mz_low: f32,
    pub mz_high: f32,
    pub im_low: f32,
    pub im_high: f32,
}

// ---------------------------------------------------------------------------
// Scan / frame data (a single MS cycle)
// ---------------------------------------------------------------------------

/// A single scan (frame or cycle) of MS data.
#[derive(Debug, Clone)]
pub struct ScanData {
    pub rt: f32,
    pub n_peaks: u32,
    // Profile mode
    pub im_values: Vec<f32>,
    pub mz_values: Vec<f32>,
    pub intensity_values: Vec<u32>,
    // Centroided mode (computed)
    pub im_values_cen: Vec<f32>,
    pub mz_values_cen: Vec<f32>,
    pub intensity_values_cen: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Full MS data container (replaces the `Tims` class and its `d_ms1_maps` / `d_ms2_maps`)
// ---------------------------------------------------------------------------

/// Complete MS dataset for a single Bruker .d file, structured by SWATH window.
#[derive(Debug)]
pub struct MsData {
    /// Per-cycle RT values.
    pub scan_rts: Vec<f32>,
    /// Cycle time (mean delta RT).
    pub cycle_time: f32,
    /// Number of cycles.
    pub n_cycles: u32,

    /// MS1 data indexed by swath_id (1-based).
    pub ms1_maps: HashMap<u8, SwathMsMap>,
    /// MS2 data indexed by swath_id (1-based).
    pub ms2_maps: HashMap<u8, SwathMsMap>,

    /// DIA quadrupole boundaries.
    pub swath_boundaries: Vec<f32>,
    /// DIA window metadata.
    pub dia_windows: Vec<DiaWindow>,

    /// Frames per cycle.
    pub frames_per_cycle: u32,
    /// Device name.
    pub device_name: String,
    /// Ion-mobility bin width.
    pub im_gap: f32,
    /// Data directory.
    pub dir_d: String,

    // ── Direct binary access (avoids building full sorted maps) ─────────
    /// Per-scan peak pointers: push_indptr[scan] = peak_start,
    /// push_indptr[scan+1] = peak_end (from binary reader).
    pub push_indptr: Vec<i64>,
    /// Raw TOF indices for all peaks across all scans.
    pub tof_indices: Vec<u32>,
    /// Raw intensity values for all peaks.
    pub intensity_values: Vec<u32>,
    /// Frame metadata: (frame_id, rt, msms_type, num_scans, scan_start_offset).
    pub frame_info: Vec<FrameInfo>,
    /// TOF→m/z lookup table.
    pub mz_lookup: Vec<f32>,
    /// 1/K0 mobility calibration values per scan index.
    pub mobility_values: Vec<f32>,
    /// Max scans per frame.
    pub scan_max_index: u32,
    /// Per-cycle frame ranges: (frame_start_idx, frame_end_idx) for each cycle.
    pub cycle_frame_ranges: Vec<(usize, usize)>,
    /// Whether the data has ion mobility (TIMS) dimension.
    /// false for mzML input (LC-MS/MS without IMS).
    pub has_ims: bool,
}

/// Minimal frame metadata for direct XIC extraction.
#[derive(Debug, Clone, Copy)]
pub struct FrameInfo {
    pub id: i64,
    pub time: f32,
    pub msms_type: i64,   // 0=MS1, 9=MS/MS
    pub num_scans: i64,
    pub scan_base: usize, // base index for push_indptr lookup
}

/// Per-SWATH-window MS map (profile + centroided data).
#[derive(Debug, Clone)]
pub struct SwathMsMap {
    /// Retention time per cycle.
    pub scan_rts: Vec<f32>,
    /// Number of profile-mode peaks per cycle.
    pub cycle_valid_lens: Vec<u32>,
    /// Concatenated 1/K0 values for all cycles (profile).
    pub im_values: Vec<f32>,
    /// Concatenated m/z values for all cycles (profile).
    pub mz_values: Vec<f32>,
    /// Concatenated intensity values for all cycles (profile).
    pub height_values: Vec<u32>,

    // Centroided counterparts
    pub cycle_valid_lens_cen: Vec<u32>,
    pub im_values_cen: Vec<f32>,
    pub mz_values_cen: Vec<f32>,
    pub height_values_cen: Vec<u32>,

    /// Cumulative seek indices for binary search.
    pub seek_indices: Vec<i64>,
    pub seek_indices_cen: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Library container (replaces `Library` class)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpectralLibrary {
    /// All target precursors.
    pub precursors: Vec<Precursor>,
    /// Peptide-to-protein mapping.
    pub protein_map: Vec<PeptideProteinMap>,
    /// Original file type.
    pub lib_type: String,
}

impl SpectralLibrary {
    pub fn len(&self) -> usize {
        self.precursors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.precursors.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Fragment ion m/z extraction helper
// ---------------------------------------------------------------------------

/// Pre-computed fragment ion m/z matrices for a batch of precursors.
pub struct FragmentMzMatrix {
    pub n_pep: usize,
    pub n_fg_max: usize,
    /// [n_pep, n_fg_max] — m/z values for centre isotopes.
    pub center: ndarray::Array2<f32>,
    /// [n_pep, n_fg_max] — m/z values for -1 Da.
    pub left: ndarray::Array2<f32>,
    /// [n_pep, n_fg_max] — m/z values for +1 Da.
    pub h1: ndarray::Array2<f32>,
    /// [n_pep, n_fg_max] — m/z values for +2 Da.
    pub h2: ndarray::Array2<f32>,
}

// ---------------------------------------------------------------------------
// Score column names (replaces string-generation patterns)
// ---------------------------------------------------------------------------

/// The constant maximum number of fragment ions used throughout the pipeline.
pub const FG_NUM: usize = 12;
/// Number of ions in the "center" map: precursor, unfragmented precursor, + 12 fragments.
pub const CENTER_ION_COUNT: usize = 14;
/// Number of maps: 4 isotopes × 14 ions = 56.
pub const BIG_ION_COUNT: usize = 56;
/// Window size for SA scoring.
pub const WINDOW_POINTS: usize = 7;
/// Default map cycle dimension.
pub const MAP_CYCLE_DIM: usize = 13;
/// Default map IM dimension.
pub const MAP_IM_DIM: usize = 50;
/// Default map IM gap.
pub const MAP_IM_GAP: f32 = 0.001;
