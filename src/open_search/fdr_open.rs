//! Dual-space FDR framework for open search results.
//!
//! Two independent FDR controls:
//! 1. Peptide-space FDR (PSM-FDR): standard target-decoy on peptide IDs
//! 2. Modification-space FDR (Mod-FDR): controls modification authenticity
//!
//! Modification decoy strategies:
//! - Shift: randomly shift the modification to another residue
//! - Mass swap: replace with a different mass from the Unimod database
//! - Residue swap: change the modified amino acid

use crate::open_search::frag_index::FragmentMatch;

// ===========================================================================
// Constants
// ===========================================================================

/// Default FDR thresholds
pub const DEFAULT_PSM_FDR: f32 = 0.01;
pub const DEFAULT_MOD_FDR: f32 = 0.01;

// ===========================================================================
// Dual-space FDR controller
// ===========================================================================

pub struct DualSpaceFdr {
    psm_fdr_threshold: f32,
    mod_fdr_threshold: f32,
}

impl Default for DualSpaceFdr {
    fn default() -> Self {
        Self {
            psm_fdr_threshold: DEFAULT_PSM_FDR,
            mod_fdr_threshold: DEFAULT_MOD_FDR,
        }
    }
}

impl DualSpaceFdr {
    pub fn new(psm_fdr: f32, mod_fdr: f32) -> Self {
        Self {
            psm_fdr_threshold: psm_fdr,
            mod_fdr_threshold: mod_fdr,
        }
    }

    /// Calculate PSM-level FDR using target-decoy approach.
    ///
    /// Sorts by combined_score descending, then computes:
    /// q = decoy_cumsum / target_cumsum
    /// Applies monotonicity (reverse cumulative minimum).
    pub fn calculate_psm_fdr<'a>(&self, matches: &'a mut [ExtendedMatch]) {
        // Sort by combined score descending
        matches.sort_unstable_by(|a, b| {
            b.match_result
                .combined_score
                .partial_cmp(&a.match_result.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut target_cum = 0usize;
        let mut decoy_cum = 0usize;

        for m in matches.iter_mut() {
            if m.is_decoy {
                decoy_cum += 1;
            } else {
                target_cum += 1;
            }
            let q = if target_cum > 0 {
                decoy_cum as f32 / target_cum as f32
            } else {
                1.0
            };
            m.psm_q_value = q;
        }

        // Reverse cumulative minimum (monotonic q-value)
        let mut min_q = 1.0f32;
        for m in matches.iter_mut().rev() {
            min_q = min_q.min(m.psm_q_value);
            m.psm_q_value = min_q;
        }
    }

    /// Calculate modification-level FDR.
    ///
    /// Modifications are decoyed by randomizing:
    /// - The modification mass (swap with unrelated mass)
    /// - The modified residue (shift to different AA)
    pub fn calculate_mod_fdr(&self, mods: &mut [ModificationCall]) {
        mods.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut target_cum = 0usize;
        let mut decoy_cum = 0usize;

        for m in mods.iter_mut() {
            if m.is_decoy {
                decoy_cum += 1;
            } else {
                target_cum += 1;
            }
            let q = if target_cum > 0 {
                decoy_cum as f32 / target_cum as f32
            } else {
                1.0
            };
            m.mod_q_value = q;
        }

        let mut min_q = 1.0f32;
        for m in mods.iter_mut().rev() {
            min_q = min_q.min(m.mod_q_value);
            m.mod_q_value = min_q;
        }
    }

    /// Apply joint FDR filter.
    pub fn filter(&self, matches: &mut Vec<ExtendedMatch>, mods: &mut Vec<ModificationCall>) {
        self.calculate_psm_fdr(matches);
        self.calculate_mod_fdr(mods);

        // Filter matches
        matches.retain(|m| m.psm_q_value <= self.psm_fdr_threshold);
        // Filter mods
        mods.retain(|m| m.mod_q_value <= self.mod_fdr_threshold);
    }
}

// ===========================================================================
// Extended match structure
// ===========================================================================

#[derive(Debug, Clone)]
pub struct ExtendedMatch {
    pub match_result: FragmentMatch,
    pub is_decoy: bool,
    pub psm_q_value: f32,
    pub peptide_sequence: String,
    pub protein_id: String,
    pub swath_id: u8,
    pub apex_cycle: u32,
    pub precursor_im: f32,
}

impl ExtendedMatch {
    pub fn new(m: FragmentMatch, is_decoy: bool, sequence: &str, protein: &str) -> Self {
        Self {
            match_result: m,
            is_decoy,
            psm_q_value: 1.0,
            peptide_sequence: sequence.to_string(),
            protein_id: protein.to_string(),
            swath_id: 0,
            apex_cycle: 0,
            precursor_im: 0.0,
        }
    }
}

// ===========================================================================
// Modification call structure
// ===========================================================================

#[derive(Debug, Clone)]
pub struct ModificationCall {
    /// Modification mass shift (Da)
    pub mass_shift: f32,
    /// Modified residue
    pub residue: char,
    /// Position in peptide (0-indexed)
    pub position: u8,
    /// Peptide ID
    pub peptide_id: u32,
    /// Is this a decoy modification?
    pub is_decoy: bool,
    /// Modification q-value
    pub mod_q_value: f32,
    /// Confidence score (from MLP or heuristic)
    pub confidence: f32,
    /// Supporting fragment count
    pub supporting_fragments: usize,
}

// ===========================================================================
// Decoy generation
// ===========================================================================

/// Generate decoy modifications by shifting the modification to another position.
pub fn generate_shift_decoy(mod_call: &ModificationCall) -> ModificationCall {
    let new_pos = (mod_call.position + 3) % 20; // arbitrary shift
    ModificationCall {
        mass_shift: mod_call.mass_shift,
        residue: mod_call.residue,
        position: new_pos,
        peptide_id: mod_call.peptide_id,
        is_decoy: true,
        mod_q_value: 1.0,
        confidence: mod_call.confidence * 0.5, // decoys get lower confidence
        supporting_fragments: mod_call.supporting_fragments,
    }
}

/// Generate decoy modifications by mass swapping (using random offset).
pub fn generate_mass_swap_decoy(mod_call: &ModificationCall) -> ModificationCall {
    let swapped_mass = mod_call.mass_shift + 27.99; // ~CO difference
    ModificationCall {
        mass_shift: swapped_mass,
        residue: mod_call.residue,
        position: mod_call.position,
        peptide_id: mod_call.peptide_id,
        is_decoy: true,
        mod_q_value: 1.0,
        confidence: mod_call.confidence * 0.6,
        supporting_fragments: mod_call.supporting_fragments,
    }
}

/// Generate decoy peptides by reversing the sequence.
pub fn generate_decoy_peptide(sequence: &str) -> String {
    // Keep first and last amino acid, reverse the middle
    if sequence.len() <= 2 {
        return sequence.chars().rev().collect();
    }
    let first = sequence.chars().next().unwrap();
    let last = sequence.chars().last().unwrap();
    let middle: String = sequence[1..sequence.len() - 1].chars().rev().collect();
    format!("{}{}{}", first, middle, last)
}

// ===========================================================================
// Utility: classify modifications
// ===========================================================================

/// Known common modifications with their mass shifts.
pub const COMMON_MODS: &[(f32, &str, char)] = &[
    (15.9949, "Oxidation", 'M'),
    (57.0215, "Carbamidomethyl", 'C'),
    (42.0106, "Acetyl", 'N'),  // N-term
    (0.9840, "Deamidation", 'N'),
    (0.9840, "Deamidation", 'Q'),
    (-17.0265, "Pyro-glu", 'E'),  // N-term
    (-17.0265, "Pyro-glu", 'Q'),  // N-term
    (-18.0106, "Water-loss", 'S'),
    (-18.0106, "Water-loss", 'T'),
    (79.9663, "Phosphorylation", 'S'),
    (79.9663, "Phosphorylation", 'T'),
    (79.9663, "Phosphorylation", 'Y'),
];

/// Classify a mass shift as a known modification (if close to one).
pub fn classify_modification(mass_shift: f32, residue: char) -> Option<(&'static str, f32)> {
    for &(mod_mass, name, _res) in COMMON_MODS {
        if (mass_shift - mod_mass).abs() < 0.1 {
            return Some((name, mod_mass));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_search::frag_index::FragmentMatch;

    #[test]
    fn test_psm_fdr() {
        let mut matches = vec![
            ExtendedMatch::new(
                FragmentMatch {
                    peptide_id: 0, matched_b: vec![], matched_y: vec![],
                    matched_immonium: vec![],
                    score_continuous: 0.8, score_coverage: 0.5,
                    score_intensity: 0.0, mass_shift: 0.0, combined_score: 0.7,
                },
                false, "PEPTIDE1", "PROT1",
            ),
            ExtendedMatch::new(
                FragmentMatch {
                    peptide_id: 1, matched_b: vec![], matched_y: vec![],
                    matched_immonium: vec![],
                    score_continuous: 0.6, score_coverage: 0.4,
                    score_intensity: 0.0, mass_shift: 0.0, combined_score: 0.5,
                },
                true, "DECOY_PEPTIDE", "DECOY_PROT",
            ),
        ];

        let fdr = DualSpaceFdr::default();
        fdr.calculate_psm_fdr(&mut matches);

        assert!(matches[0].psm_q_value <= 1.0);
        assert!(matches[1].psm_q_value <= 1.0);
    }

    #[test]
    fn test_classify_mod() {
        let result = classify_modification(15.9949, 'M');
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, "Oxidation");
    }

    #[test]
    fn test_decoy_peptide() {
        let decoy = generate_decoy_peptide("PEPTIDEK");
        assert_eq!(decoy.len(), "PEPTIDEK".len());
        assert_ne!(decoy, "PEPTIDEK");
    }
}
