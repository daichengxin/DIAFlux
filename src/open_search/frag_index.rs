//! Fragment-match type for open search.
//!
//! A match between an observed spectrum and a candidate peptide.

/// A match between an observed spectrum and a candidate peptide.
#[derive(Debug, Clone)]
pub struct FragmentMatch {
    pub peptide_id: u32,
    pub matched_b: Vec<(u8, u8)>,   // (position, charge)
    pub matched_y: Vec<(u8, u8)>,
    pub matched_immonium: Vec<u8>,
    pub score_continuous: f32,       // longest continuous b/y match
    pub score_coverage: f32,         // matched / theoretical ratio
    pub score_intensity: f32,
    pub mass_shift: f32,             // precursor mass shift (observed - theoretical)
    pub combined_score: f32,
}
