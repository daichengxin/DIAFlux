//! DIA-NN exact decoy generation.
//!
//! Implements the amino-acid mutation method from diann.cpp `Entry::generate_decoy()`
//! (around line 3590). This is critical: the old DIAFluX decoy method shifted masses
//! differently, producing decoys that didn't match DIA-NN's expected distribution.

use crate::model::chemistry::*;
use crate::model::{IonType, Precursor, SpectralLibrary};

// ============================================================================
// Build mutation lookup from DIA-NN tables
// ============================================================================

/// DIA-NN AA mutation table: AAs[i] → MutateTo[i]
fn build_mutate_map() -> [u8; 256] {
    let mut map = [b'A'; 256];
    for i in 0..AAS_TABLE.len().min(MUTATE_TO_TABLE.len()) {
        let from = AAS_TABLE[i];
        let to = MUTATE_TO_TABLE[i];
        map[from as usize] = to;
        map[from.to_ascii_lowercase() as usize] = to.to_ascii_lowercase();
    }
    map
}

/// Get the mutant AA for a given AA.
pub fn mutate_aa(aa: u8) -> u8 {
    let map = build_mutate_map();
    map[aa as usize]
}

/// Mass shift from mutating the given amino acid.
pub fn mutation_shift(aa: u8) -> f64 {
    let mutant = mutate_aa(aa);
    aa_mass(mutant as char) - aa_mass(aa as char)
}

// ============================================================================
// Fragment recognition (DIA-NN `recognise_fragments`)
// ============================================================================

/// Compute peptide sequence masses from stripped sequence.
pub fn get_sequence_masses(strip_seq: &str) -> Vec<f64> {
    strip_seq.chars().map(aa_mass).collect()
}

/// Sum of sequence masses.
pub fn sum_seq(seq: &[f64]) -> f64 {
    seq.iter().sum()
}

/// Generate fragment ions for a given charge and loss state (DIA-NN `generate_fragments`).
/// Returns (ion_type, index, loss, charge, mz) tuples.
pub fn generate_fragments(
    sequence: &[f64],
    charge: i8,
    loss: u8,
    min_aas: usize,
) -> Vec<Ion> {
    let c = charge as f64;
    let s = sum_seq(sequence);
    let loss_mass = if (loss as usize) < LOSS_MASSES.len() {
        LOSS_MASSES[loss as usize]
    } else {
        0.0
    };
    let mut result = Vec::new();

    let mut curr = 0.0;
    for i in 0..sequence.len().saturating_sub(1) {
        curr += sequence[i];
        let b = curr;
        let y = s - curr + PROTON + OH;

        if i + 1 >= min_aas {
            result.push(Ion::new(
                F_TYPE_B,
                (i + 1) as u8,
                loss,
                charge,
                ((b + c * PROTON - loss_mass) / c) as f32,
            ));
        }
        if sequence.len() - i - 1 >= min_aas {
            result.push(Ion::new(
                F_TYPE_Y,
                (i + 1) as u8,
                loss,
                charge,
                ((y + c * PROTON - loss_mass) / c) as f32,
            ));
        }
    }
    result
}

/// Recognise fragment identities from existing Product list (DIA-NN `recognise_fragments`).
/// For each fragment ion, finds the closest theoretical ion within mass tolerance.
/// Returns recognised Ion identities and the generator accuracy used.
pub fn recognise_fragments(
    sequence: &[f64],
    fragments: &[Product],
    pr_charge: i8,
    full_spectrum: bool,
    max_charge: i8,
    loss_cap: u8,
) -> (Vec<Ion>, f32) {
    let mut gen_acc: f32 = 5.0 / 1_000_000.0; // GeneratorAccuracy
    let mut result = vec![Ion::default(); fragments.len()];

    let s = sum_seq(sequence);

    loop {
        let mut tot = 0;
        let mut charge = 1i8;
        let mut loss = LOSS_NONE;

        // Reset results
        for r in &mut result {
            r.charge = 0;
        }

        'outer: loop {
            let mut v = generate_fragments(sequence, charge, loss, 0);
            if loss == LOSS_NONE {
                // Non-fragmented precursor ion
                v.push(Ion::new(
                    F_TYPE_Y,
                    sequence.len() as u8,
                    loss,
                    pr_charge,
                    ((s + OH + (1.0 + pr_charge as f64) * PROTON - LOSS_MASSES[loss as usize])
                        / pr_charge as f64) as f32,
                ));
            }

            for i in 0..fragments.len() {
                if result[i].charge != 0 {
                    continue;
                }
                let margin = fragments[i].mz as f64 * gen_acc as f64;
                let mut min = margin;
                let mut best_idx = 0usize;
                for j in 0..v.len() {
                    let delta = (v[j].mz as f64 - fragments[i].mz as f64).abs();
                    if delta < min {
                        min = delta;
                        best_idx = j;
                    }
                }
                if min < margin {
                    tot += 1;
                    result[i].init_from_product(fragments.get(i).unwrap());
                    // Override with recognized identity
                    result[i].ion_type = v[best_idx].ion_type & 3;
                    result[i].index = v[best_idx].index;
                    result[i].loss = v[best_idx].loss;
                    result[i].charge = v[best_idx].charge;
                    if tot >= fragments.len() {
                        break 'outer;
                    }
                }
            }

            loss += 1;
            if loss == loss_cap {
                loss = LOSS_NONE;
                charge += 1;
            }
            if charge > max_charge && tot < fragments.len() {
                if full_spectrum {
                    break 'outer;
                }
                gen_acc *= 2.0;
                break;
            }
            if tot < fragments.len() && charge <= max_charge {
                continue;
            }
            if tot >= fragments.len() {
                break 'outer;
            }
        }

        if tot >= fragments.len() || full_spectrum {
            break;
        }
    }

    (result, gen_acc)
}

// ============================================================================
// Get modified state from peptide string (DIA-NN convention)
// ============================================================================

/// Check if position `pos` in the peptide string has a modification (contains '(' or '[').
/// This is a simplified version of DIA-NN's `get_mod_state`.
pub fn has_mod_at(strip_seq: &str, pos: usize) -> bool {
    // In DIA-NN, modification state is stored as an array indicating which positions
    // are modified. For stripped sequences (no mods in the string), all positions
    // are unmodified. We return false for stripped sequences.
    //
    // For modified sequences like "AAC(UniMod:4)DEFR", we'd need to parse the mods.
    false
}

/// Get modification state array for a peptide sequence.
/// Returns a Vec<bool> where true = position is modified.
pub fn get_mod_state(strip_seq: &str) -> Vec<bool> {
    vec![false; strip_seq.len() + 1] // +1 for 1-based indexing in DIA-NN
}

// ============================================================================
// Main decoy generation
// ============================================================================

/// DIA-NN exact decoy generation for one precursor.
///
/// Takes the target fragment ions and produces decoy fragment ions by
/// applying amino acid mutation mass shifts to the N- and C-terminal
/// positions, then shifting fragment m/z values accordingly.
///
/// # Arguments
/// * `strip_seq` — stripped (unmodified) peptide sequence
/// * `target_fragments` — fragment ion Products from the library
/// * `pr_charge` — precursor charge
///
/// # Returns
/// Decoy fragment ions with shifted m/z values.
pub fn generate_decoy_fragments(
    strip_seq: &str,
    target_fragments: &[Product],
    pr_charge: i8,
) -> Vec<Product> {
    if strip_seq.len() < 3 {
        return Vec::new();
    }

    let mut decoy = target_fragments.to_vec();
    let chars: Vec<u8> = strip_seq.as_bytes().to_vec();
    let m = chars.len() as isize - 2; // last residue index (1-based from N-term is 1..m)
    if m < 0 {
        return decoy;
    }

    let mod_state = get_mod_state(strip_seq);

    // ---- N-terminal position ----
    let n_pos = if !mod_state.get(1).copied().unwrap_or(false) {
        1
    } else if !mod_state.get((2.min(m + 1)) as usize).copied().unwrap_or(false) {
        2.min(m + 1) as usize
    } else if !mod_state.get(0).copied().unwrap_or(false) {
        0
    } else {
        1
    };

    let n_shift = mutation_shift(chars[n_pos]);

    // ---- C-terminal position ----
    let c_pos = if !mod_state.get(m as usize).copied().unwrap_or(false) {
        m as usize
    } else if !mod_state.get((0.max(m - 1)) as usize).copied().unwrap_or(false) {
        (0.max(m - 1)) as usize
    } else if !mod_state.get((m + 1) as usize).copied().unwrap_or(false) {
        (m + 1) as usize
    } else {
        m as usize
    };

    let c_shift = mutation_shift(chars[c_pos]);

    // ---- Apply shifts to fragment m/z ----
    for fr in &mut decoy {
        let c = (fr.charge.max(1)) as f64;
        if (fr.ion_type & F_TYPE_Y) != 0 {
            fr.mz += (c_shift / c) as f32;
        } else {
            fr.mz += (n_shift / c) as f32;
        }
    }

    decoy
}

/// DIA-NN exact decoy precursor m/z shift.
///
/// The precursor m/z itself is shifted by the average of N-term and C-term shifts.
pub fn decoy_precursor_mz_shift(strip_seq: &str, pr_charge: i8) -> f32 {
    if strip_seq.len() < 3 {
        return 0.0;
    }

    let chars: Vec<u8> = strip_seq.as_bytes().to_vec();
    let m = chars.len() - 2;
    let mod_state = get_mod_state(strip_seq);

    let n_pos = if !mod_state[1] { 1 }
        else if !mod_state[2.min(m + 1)] { 2.min(m + 1) }
        else if !mod_state[0] { 0 }
        else { 1 };

    let c_pos = if !mod_state[m] { m }
        else if !mod_state[0.max(m as isize - 1) as usize] { 0.max(m as isize - 1) as usize }
        else if !mod_state[m + 1] { m + 1 }
        else { m };

    let n_shift = mutation_shift(chars[n_pos]);
    let c_shift = mutation_shift(chars[c_pos]);
    let pr_charge_f = pr_charge.max(1) as f64;

    ((n_shift + c_shift) / (2.0 * pr_charge_f)) as f32
}

// ============================================================================
// Library-level decoy generation (core `libtransform.decoy`, `diann` type)
// ============================================================================

/// Generate decoys for a target-only `SpectralLibrary` and append them, mirroring
/// the reference `DecoyGenerator` processing step (`decoy_type = "diann"`).
///
/// For each target precursor (with `decoy == 0` and a sequence of length >= 3)
/// a decoy precursor is created by:
///   - mutating sequence positions 1 and `len-2` (`DIANNDecoyGenerator._decoy`),
///   - shifting the precursor m/z by `(n_shift + c_shift) / charge` (the true
///     mutated-mass delta, `calc_precursor_mz`),
///   - shifting fragment m/z by `n_shift / charge` (b-series, index >= 2) or
///     `c_shift / charge` (y-series, index >= 2), matching `calc_fragment_mz_df`.
///
/// Fragments of index 1 and loss-type fragments are left unshifted (they do not
/// contain the mutated positions).
pub fn generate_decoy_library(mut lib: SpectralLibrary) -> SpectralLibrary {
    let n_target = lib.precursors.len();
    let max_idx = lib.precursors.iter().map(|p| p.pr_index).max().unwrap_or(0);

    // Assign elution groups to targets: `0..n_target` (core `harmonize`).
    for (i, p) in lib.precursors.iter_mut().enumerate() {
        p.elution_group_idx = i as u32;
    }

    let mut decoys: Vec<Precursor> = Vec::with_capacity(n_target);

    for (t_idx, p) in lib.precursors.iter().enumerate() {
        if p.decoy != 0 {
            continue;
        }
        let seq = p.strip_seq.as_bytes();
        let len = seq.len();
        if len < 3 || p.fragment_ions.is_empty() {
            continue;
        }

        // N-terminal mutation at position 1 (second residue).
        let n_shift = mutation_shift(seq[1]);
        // C-terminal mutation at position len-2 (second-to-last residue).
        // For len == 3 both positions coincide; mutate once so the fragment and
        // precursor shifts stay consistent.
        let c_shift = if len >= 4 { mutation_shift(seq[len - 2]) } else { 0.0 };

        let charge = p.pr_charge.max(1) as f64;

        let mut dp = p.clone();
        dp.decoy = 1;
        dp.pr_index = max_idx + 1 + decoys.len() as u32;
        // Decoys share the elution group of the target they were generated from
        // (competitive-FDR keeps the best of a target/decoy pair).
        dp.elution_group_idx = t_idx as u32;
        dp.pr_id = format!("DECOY_{}", p.pr_id);

        // Mutated sequence (kept so downstream isotope/hash computation sees it).
        let mut dseq: Vec<u8> = seq.to_vec();
        dseq[1] = mutate_aa(seq[1]);
        if len >= 4 {
            dseq[len - 2] = mutate_aa(seq[len - 2]);
        }
        if let Ok(s) = String::from_utf8(dseq.clone()) {
            dp.strip_seq = s.clone();
            dp.simple_seq = s;
        }

        // Full mutated-mass shift of the precursor m/z.
        dp.pr_mz += ((n_shift + c_shift) / charge) as f32;

        // Shift fragment m/z values.
        for f in dp.fragment_ions.iter_mut() {
            let fc = f.annotation.charge.max(1) as f64;
            let shift = match f.annotation.ion_type {
                IonType::Y if f.annotation.series_number >= 2 => c_shift,
                IonType::B if f.annotation.series_number >= 2 => n_shift,
                _ => 0.0,
            };
            f.mz += (shift / fc) as f32;
        }

        decoys.push(dp);
    }

    let n_decoy = decoys.len();
    lib.precursors.extend(decoys);
    tracing::info!(
        "decoy generation: appended {} decoys to {} target precursors",
        n_decoy,
        n_target
    );
    lib
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutation_table() {
        // From DIA-NN: G→L, A→L, V→L, L→V, F→L, M→L, P→L, S→T, etc.
        assert_eq!(mutate_aa(b'G'), b'L');
        assert_eq!(mutate_aa(b'A'), b'L');
        assert_eq!(mutate_aa(b'V'), b'L');
        assert_eq!(mutate_aa(b'L'), b'V');
        assert_eq!(mutate_aa(b'F'), b'L');
        assert_eq!(mutate_aa(b'M'), b'L');
        assert_eq!(mutate_aa(b'P'), b'L');
        assert_eq!(mutate_aa(b'S'), b'T');
        assert_eq!(mutate_aa(b'R'), b'L');
        assert_eq!(mutate_aa(b'K'), b'L');
    }

    #[test]
    fn test_decoy_generation_basic() {
        // Simple peptide: AAAAR → positions 1=A→L, m=3=A→L
        let fragments = vec![
            Product { mz: 500.0, charge: 1, ion_type: F_TYPE_Y, ..Default::default() },
            Product { mz: 400.0, charge: 1, ion_type: F_TYPE_B, ..Default::default() },
        ];
        let decoy = generate_decoy_fragments("AAAAR", &fragments, 2);
        assert_eq!(decoy.len(), 2);

        // A→L shift: aa_mass('L') - aa_mass('A') = 113.084064 - 71.037114 = 42.04695
        let expected_shift = aa_mass('L') - aa_mass('A');
        // Y-ion gets C_shift, B-ion gets N_shift (both are same here since both positions are A→L)
        assert!((decoy[0].mz - 500.0 - expected_shift as f32).abs() < 0.01);
        assert!((decoy[1].mz - 400.0 - expected_shift as f32).abs() < 0.01);
    }

    #[test]
    fn test_precursor_mz_shift() {
        let shift = decoy_precursor_mz_shift("AAAAR", 2);
        // Average of N and C mass shifts, converted to m/z by dividing by
        // charge: ((L-A) + (L-A))/2 / 2 = (L-A)/2
        let expected = (aa_mass('L') - aa_mass('A')) / 2.0;
        assert!((shift as f64 - expected).abs() < 0.01);
    }

    #[test]
    fn test_generate_decoy_library_dvspslr() {
        use crate::model::{FragmentIon, IonAnnotation};
        // DVSPSLR: seq[1]=V (V->L, +14.01565), seq[5]=L (L->V, -14.01565).
        // Expected: B-ions shift by n_shift=+14.01565, Y-ions by c_shift=-14.01565.
        let mut lib = SpectralLibrary {
            precursors: Vec::new(),
            protein_map: Vec::new(),
            lib_type: "test".into(),
        };
        let mut p = Precursor {
            pr_id: "DVSPSLR2".into(),
            pr_charge: 2,
            pr_mz: 373.1956,
            pr_index: 0,
            strip_seq: "DVSPSLR".into(),
            simple_seq: "DVSPSLR".into(),
            decoy: 0,
            ..Default::default()
        };
        let mk = |ty: IonType, num: u8, ch: u8, mz: f32| FragmentIon {
            mz,
            height: 1.0,
            annotation: IonAnnotation { ion_type: ty, series_number: num, charge: ch },
            mz_left: mz,
            mz_1h: mz,
            mz_2h: mz,
        };
        p.fragment_ions = vec![
            mk(IonType::Y, 2, 1, 517.2729),
            mk(IonType::B, 2, 1, 229.1183),
            mk(IonType::Y, 3, 1, 430.2408),
            mk(IonType::B, 3, 1, 316.1503),
        ];
        lib.precursors = vec![p];

        let out = generate_decoy_library(lib);
        assert_eq!(out.precursors.len(), 2);
        let t = &out.precursors[0];
        let d = &out.precursors[1];
        // B2 (idx 1) should shift by +14.01565; Y2 (idx 0) by -14.01565.
        let n_shift = aa_mass('L') - aa_mass('V'); // +14.01565
        let c_shift = aa_mass('V') - aa_mass('L'); // -14.01565
        assert!((d.fragment_ions[1].mz - t.fragment_ions[1].mz - n_shift as f32).abs() < 0.01);
        assert!((d.fragment_ions[0].mz - t.fragment_ions[0].mz - c_shift as f32).abs() < 0.01);
        assert_eq!(d.strip_seq, "DLSPSVR");
    }
}
