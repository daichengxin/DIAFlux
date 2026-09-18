//! Library transform (P1): flatten, fragment cardinality, and isotope columns.
//!
//! Ports the library-transform processing steps that turn a columnar
//! precursor/fragment library (`SpecLibFlat`) into the flat representation the
//! candidate-selection step (P2) consumes:
//!   - `flat_frag_start_idx` / `flat_frag_stop_idx` fragment ranges
//!   - `elution_group_idx` (default: one group per precursor)
//!   - fragment `cardinality` (shared ions within an elution group)
//!   - precursor isotope columns `i_0 .. i_{n-1}` + `mono_isotope_idx`
//!   - fragment `mz_library`, `loss_type`, `position`

use std::collections::HashMap;

use crate::engine::data::{FragmentDf, SpecLibFlat};

/// Number of shared fragments for each fragment row, mirroring
/// `alphabase.peptide.fragment.calc_fragment_cardinality`.
///
/// Precursors are grouped by `elution_group_idx` (and, when
/// `split_target_decoy`, by decoy class). Within a group, two fragments at the
/// same positional slot are considered shared when their m/z differ by less
/// than `1e-5`.
pub fn calc_fragment_cardinality(
    elution_group_idx: &[u32],
    decoy: &[u8],
    frag_start: &[u32],
    frag_stop: &[u32],
    fragment_mz: &[f64],
    split_target_decoy: bool,
) -> Vec<u8> {
    let n_frag = fragment_mz.len();
    let mut cardinality = vec![1u8; n_frag];

    // Precursor rows sorted (stable) by elution group.
    let mut rows: Vec<usize> = (0..elution_group_idx.len()).collect();
    rows.sort_by_key(|&r| elution_group_idx[r]);

    // Decoy classes to process (a single synthetic class when not splitting).
    let classes: Vec<Option<u8>> = if split_target_decoy {
        let mut v: Vec<u8> = decoy.to_vec();
        v.sort();
        v.dedup();
        v.into_iter().map(Some).collect()
    } else {
        vec![None]
    };

    for class in classes {
        let class_rows: Vec<usize> = rows
            .iter()
            .copied()
            .filter(|&r| class.map_or(true, |c| decoy[r] == c))
            .collect();

        let mut i = 0;
        while i < class_rows.len() {
            // Consecutive rows sharing the same elution group.
            let mut stop = i + 1;
            while stop < class_rows.len()
                && elution_group_idx[class_rows[stop]] == elution_group_idx[class_rows[i]]
            {
                stop += 1;
            }
            let group = &class_rows[i..stop];

            for &ri in group {
                for &rj in group {
                    if ri == rj {
                        continue;
                    }
                    let (si, ei) = (frag_start[ri] as usize, frag_stop[ri] as usize);
                    let (sj, ej) = (frag_start[rj] as usize, frag_stop[rj] as usize);
                    if ei - si != ej - sj {
                        // The reference implementation raises here; skip defensively instead.
                        continue;
                    }
                    for k in 0..(ei - si) {
                        let a = fragment_mz[si + k];
                        let b = fragment_mz[sj + k];
                        if (a - b).abs() < 0.00001 {
                            cardinality[si + k] = cardinality[si + k].saturating_add(1);
                        }
                    }
                }
            }

            i = stop;
        }
    }

    cardinality
}

/// Compute the relative sequence position of each fragment.
///
/// Forward series (`b`/`c`, series id < 120) are numbered from the N-terminus;
/// reverse series (`x`/`y`/`z`, series id >= 120) from the C-terminus, matching
/// the reference `position` semantics.
fn fragment_positions(
    fragments: &FragmentDf,
    seq_len_by_precursor: &HashMap<u32, usize>,
) -> Vec<u16> {
    fragments
        .precursor_idx
        .iter()
        .zip(fragments.fragment_type.iter())
        .zip(fragments.fragment_number.iter())
        .map(|((pid, &ftype), &num)| {
            let len = seq_len_by_precursor.get(pid).copied().unwrap_or(0) as u32;
            let num = num as u32;
            let pos = if ftype >= 120 {
                len.saturating_sub(num)
            } else {
                num.saturating_sub(1)
            };
            pos as u16
        })
        .collect()
}

/// Populate the flat columns of a `SpecLibFlat` in place.
///
/// `n_isotopes` matches the `IsotopeGenerator(n_isotopes=4)` default.
pub fn transform_library(spec: &mut SpecLibFlat, n_isotopes: usize) {
    let n = spec.n_precursors();

    // --- fragment ranges (assumes fragment rows grouped by precursor order) ---
    spec.precursor_df.flat_frag_start_idx = vec![0u32; n];
    spec.precursor_df.flat_frag_stop_idx = vec![0u32; n];
    let mut frag_i = 0usize;
    for p in 0..n {
        let pid = spec.precursor_df.precursor_idx[p];
        let start = frag_i as u32;
        while frag_i < spec.fragment_df.precursor_idx.len()
            && spec.fragment_df.precursor_idx[frag_i] == pid
        {
            frag_i += 1;
        }
        spec.precursor_df.flat_frag_start_idx[p] = start;
        spec.precursor_df.flat_frag_stop_idx[p] = frag_i as u32;
    }

    // --- elution groups (populated by `from_spectral_library`, where a decoy
    //     shares its source target's group; fall back to one group per
    //     precursor for manually-built libraries) ---
    if spec.precursor_df.elution_group_idx.len() != n {
        spec.precursor_df.elution_group_idx = (0..n as u32).collect();
    }

    // --- fragment cardinality ---
    let cardinality = calc_fragment_cardinality(
        &spec.precursor_df.elution_group_idx,
        &spec.precursor_df.decoy,
        &spec.precursor_df.flat_frag_start_idx,
        &spec.precursor_df.flat_frag_stop_idx,
        &spec.fragment_df.fragment_mz,
        true,
    );
    spec.fragment_df.cardinality = cardinality;

    // --- fragment library m/z / loss / position ---
    spec.fragment_df.mz_library = spec.fragment_df.fragment_mz.clone();
    spec.fragment_df.loss_type = vec![0u8; spec.fragment_df.fragment_mz.len()];
    let seq_len_by_precursor: HashMap<u32, usize> = spec
        .precursor_df
        .precursor_idx
        .iter()
        .zip(spec.precursor_df.sequence.iter())
        .map(|(&pid, seq)| (pid, seq.len()))
        .collect();
    spec.fragment_df.position =
        fragment_positions(&spec.fragment_df, &seq_len_by_precursor);

    // --- precursor isotope columns ---
    let (isotopes, mono_idx) = calc_precursor_isotope_intensity(
        &spec.precursor_df.sequence,
        &spec.precursor_df.mods,
        n_isotopes,
        0.001,
    );
    spec.precursor_df.isotopes = isotopes;
    spec.precursor_df.n_isotopes = n_isotopes;
    spec.precursor_df.mono_isotope_idx = mono_idx;
}

// =====================================================================
// Precursor isotope-envelope generation (folded from the former
// `isotopes` module; used only by `transform_library`).
// =====================================================================

/// Number of isotope bins retained after every convolution step.
pub const MAX_ISOTOPE_LEN: usize = 10;

/// Identity isotope distribution (all mass on the monoisotopic peak).
fn empty_dist() -> Vec<f64> {
    let mut d = vec![0.0f64; MAX_ISOTOPE_LEN];
    d[0] = 1.0;
    d
}

/// `(abundance, mass)` pair for one isotope of one element.
#[derive(Debug, Clone, Copy)]
struct IsotopePeak {
    abundance: f64,
    mass: f64,
}

/// Raw NIST isotope data for the elements occurring in peptides.
///
/// Values taken verbatim from `alphabase/constants/const_files/nist_element.yaml`
/// (already mass-sorted ascending).
fn element_isotope_data(elem: &str) -> Option<&'static [IsotopePeak]> {
    use IsotopePeak as P;
    const C: &[IsotopePeak] = &[
        P { abundance: 0.9893, mass: 12.0 },
        P { abundance: 0.0107, mass: 13.00335483507 },
    ];
    const H: &[IsotopePeak] = &[
        P { abundance: 0.999885, mass: 1.00782503223 },
        P { abundance: 0.000115, mass: 2.01410177812 },
    ];
    const N: &[IsotopePeak] = &[
        P { abundance: 0.99636, mass: 14.00307400443 },
        P { abundance: 0.00364, mass: 15.00010889888 },
    ];
    const O: &[IsotopePeak] = &[
        P { abundance: 0.99757, mass: 15.99491461957 },
        P { abundance: 0.00038, mass: 16.9991317565 },
        P { abundance: 0.00205, mass: 17.99915961286 },
    ];
    const S: &[IsotopePeak] = &[
        P { abundance: 0.9499, mass: 31.9720711744 },
        P { abundance: 0.0075, mass: 32.9714589098 },
        P { abundance: 0.0425, mass: 33.967867004 },
        P { abundance: 0.0001, mass: 35.96708071 },
    ];
    const P_: &[IsotopePeak] = &[P { abundance: 1.0, mass: 30.97376199842 }];
    const SE: &[IsotopePeak] = &[
        P { abundance: 0.0089, mass: 73.922475934 },
        P { abundance: 0.0937, mass: 75.919213704 },
        P { abundance: 0.0763, mass: 76.919914154 },
        P { abundance: 0.2377, mass: 77.91730928 },
        P { abundance: 0.4961, mass: 79.9165218 },
        P { abundance: 0.0873, mass: 81.9166995 },
    ];
    match elem {
        "C" => Some(C),
        "H" => Some(H),
        "N" => Some(N),
        "O" => Some(O),
        "S" => Some(S),
        "P" => Some(P_),
        "Se" => Some(SE),
        _ => None,
    }
}

/// Truncate an isotope vector to the `MAX_ISOTOPE_LEN` neighbours of the
/// monoisotopic peak. Returns `(new_mono_idx, start, end)`.
fn truncate_isotope(isotopes: &[f64], mono_idx: i64) -> (i64, usize, usize) {
    let mut trunc_start = mono_idx - 1;
    let mut trunc_end = mono_idx + 1;
    while trunc_start >= 0
        && (trunc_end as usize) < isotopes.len()
        && (trunc_end - trunc_start - 1) < MAX_ISOTOPE_LEN as i64
    {
        if isotopes[trunc_end as usize] >= isotopes[trunc_start as usize] {
            trunc_end += 1;
        } else {
            trunc_start -= 1;
        }
    }
    if (trunc_end - trunc_start - 1) < MAX_ISOTOPE_LEN as i64 {
        if trunc_start == -1 {
            trunc_end = MAX_ISOTOPE_LEN as i64;
        } else if (trunc_end as usize) == isotopes.len() {
            trunc_start = isotopes.len() as i64 - MAX_ISOTOPE_LEN as i64 - 1;
        }
    }
    (
        mono_idx - trunc_start - 1,
        (trunc_start + 1) as usize,
        trunc_end as usize,
    )
}

/// Convolve two isotope distributions and truncate the result.
fn abundance_convolution(d1: &[f64], mono1: i64, d2: &[f64], mono2: i64) -> (Vec<f64>, i64) {
    let mono_idx = mono1 + mono2;
    let mut ret = vec![0.0f64; MAX_ISOTOPE_LEN * 2 - 1];
    for (i, &a) in d1.iter().enumerate() {
        for (j, &b) in d2.iter().enumerate() {
            ret[i + j] += a * b;
        }
    }
    let (new_mono, start, end) = truncate_isotope(&ret, mono_idx);
    (ret[start..end].to_vec(), new_mono)
}

/// Isotope distribution for `n` atoms of one element (exponentiation by
/// squaring, mirroring `alphabase.constants.isotope.one_element_dist`).
fn one_element_dist(elem: &str, n: i64, dist: &HashMap<String, (Vec<f64>, i64)>) -> (Vec<f64>, i64) {
    let (elem_dist, elem_mono) = &dist[elem];
    if n == 0 {
        return (empty_dist(), 0);
    }
    if n == 1 {
        return (elem_dist.clone(), *elem_mono);
    }
    let (mut tmp_dist, mut mono_idx) = one_element_dist(elem, n / 2, dist);
    let (d, m) = abundance_convolution(&tmp_dist, mono_idx, &tmp_dist, mono_idx);
    tmp_dist = d;
    mono_idx = m;
    if n % 2 == 0 {
        (tmp_dist, mono_idx)
    } else {
        abundance_convolution(&tmp_dist, mono_idx, elem_dist, *elem_mono)
    }
}

/// Pre-computed per-element isotope distribution (`CHEM_ISOTOPE_DIST`) and
/// monoisotopic index (`CHEM_MONO_IDX`), built via `alphabase`' `reset_elements`.
#[derive(Debug, Clone)]
pub struct ElementTable {
    dist: HashMap<String, (Vec<f64>, i64)>,
    /// Maximum atom count per element used to clip pathological formulas
    /// (mirrors `IsotopeDistribution.max_elem_num_dict`).
    max_elem: HashMap<String, i64>,
}

impl Default for ElementTable {
    fn default() -> Self {
        let mut dist = HashMap::new();
        let mut max_elem = HashMap::new();
        for (elem, cap) in [("C", 2000), ("H", 5000), ("N", 1000), ("O", 1000), ("S", 200), ("P", 200)] {
            max_elem.insert(elem.to_string(), cap);
        }

        // For each element, sort isotopes by mass, place them on a rounded-mass
        // grid, and record the most abundant isotope's index.
        for elem in ["C", "H", "N", "O", "S", "P", "Se"] {
            let peaks = element_isotope_data(elem).unwrap();
            let mut sorted: Vec<&IsotopePeak> = peaks.iter().collect();
            sorted.sort_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap());

            let masses: Vec<f64> = sorted.iter().map(|p| p.mass).collect();
            let abundances: Vec<f64> = sorted.iter().map(|p| p.abundance).collect();

            let rounded: Vec<i64> = masses.iter().map(|m| m.round() as i64).collect();
            let base = rounded[0];
            let mut isos = vec![0.0f64; MAX_ISOTOPE_LEN];
            for (pos, &ab) in rounded.iter().zip(abundances.iter()) {
                isos[(pos - base) as usize] = ab;
            }
            let mono_idx = isos
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            dist.insert(elem.to_string(), (isos, mono_idx as i64));
        }

        Self { dist, max_elem }
    }
}

/// Chemical formula as an ordered list of `(element, count)`.
pub type Formula = Vec<(String, i64)>;

/// Compute the isotope abundance distribution for a chemical formula.
/// Returns `(distribution, mono_idx)`.
pub fn calc_formula_distribution(table: &ElementTable, formula: &Formula) -> (Vec<f64>, i64) {
    let mut mono = 0i64;
    let mut out = empty_dist();
    for (elem, n) in formula {
        let mut n = *n;
        if let Some(cap) = table.max_elem.get(elem) {
            if n >= *cap {
                n = cap - 1;
            }
        }
        let (elem_dist, elem_mono) = one_element_dist(elem, n, &table.dist);
        let (d, m) = abundance_convolution(&out, mono, &elem_dist, elem_mono);
        out = d;
        mono = m;
    }
    (out, mono)
}

/// Composition (element counts) of the 20 standard amino acids plus the rare
/// selenocysteine (U) and pyrrolysine (O), in the exact element order used by
/// `alphabase` (`amino_acid.tsv`).
///
/// Unknown amino acids (`B`, `X`, `Z`) are represented as `C(1000000)` and are
/// clipped to the carbon cap during distribution calculation.
pub fn amino_acid_composition(aa: char) -> &'static [( &'static str, i64)] {
    match aa {
        'A' => &[("C", 3), ("H", 5), ("N", 1), ("O", 1), ("S", 0)],
        'C' => &[("C", 3), ("H", 5), ("N", 1), ("O", 1), ("S", 1)],
        'D' => &[("C", 4), ("H", 5), ("N", 1), ("O", 3), ("S", 0)],
        'E' => &[("C", 5), ("H", 7), ("N", 1), ("O", 3), ("S", 0)],
        'F' => &[("C", 9), ("H", 9), ("N", 1), ("O", 1), ("S", 0)],
        'G' => &[("C", 2), ("H", 3), ("N", 1), ("O", 1), ("S", 0)],
        'H' => &[("C", 6), ("H", 7), ("N", 3), ("O", 1), ("S", 0)],
        'I' => &[("C", 6), ("H", 11), ("N", 1), ("O", 1), ("S", 0)],
        'K' => &[("C", 6), ("H", 12), ("N", 2), ("O", 1), ("S", 0)],
        'L' => &[("C", 6), ("H", 11), ("N", 1), ("O", 1), ("S", 0)],
        'M' => &[("C", 5), ("H", 9), ("N", 1), ("O", 1), ("S", 1)],
        'N' => &[("C", 4), ("H", 6), ("N", 2), ("O", 2), ("S", 0)],
        'P' => &[("C", 5), ("H", 7), ("N", 1), ("O", 1), ("S", 0)],
        'Q' => &[("C", 5), ("H", 8), ("N", 2), ("O", 2), ("S", 0)],
        'R' => &[("C", 6), ("H", 12), ("N", 4), ("O", 1), ("S", 0)],
        'S' => &[("C", 3), ("H", 5), ("N", 1), ("O", 2), ("S", 0)],
        'T' => &[("C", 4), ("H", 7), ("N", 1), ("O", 2), ("S", 0)],
        'U' => &[("C", 3), ("H", 5), ("N", 1), ("O", 1), ("Se", 1)],
        'V' => &[("C", 5), ("H", 9), ("N", 1), ("O", 1), ("S", 0)],
        'W' => &[("C", 11), ("H", 10), ("N", 2), ("O", 1), ("S", 0)],
        'Y' => &[("C", 9), ("H", 9), ("N", 1), ("O", 2), ("S", 0)],
        'O' => &[("C", 12), ("H", 19), ("N", 3), ("O", 2)],
        'B' | 'X' | 'Z' => &[("C", 1000000)],
        _ => &[],
    }
}

/// Build the chemical formula for a modified peptide sequence
/// (`alphabase.peptide.precursor.get_mod_seq_formula`).
///
/// `mods` follows the AlphaBase format (`"ModA;ModB"`); modifications are
/// currently ignored because diaflux's DIA-NN library carries stripped,
/// unmodified sequences (see TODO).
pub fn get_mod_seq_formula(sequence: &str, mods: &str) -> Formula {
    let mut formula: Vec<(String, i64)> = Vec::new();
    let mut pos: HashMap<String, usize> = HashMap::new();

    for aa in sequence.chars() {
        for &(elem, n) in amino_acid_composition(aa) {
            match pos.get(elem) {
                Some(&i) => formula[i].1 += n,
                None => {
                    pos.insert(elem.to_string(), formula.len());
                    formula.push((elem.to_string(), n));
                }
            }
        }
    }

    if !mods.is_empty() {
        for mod_name in mods.split(';').filter(|m| !m.is_empty()) {
            // TODO(P1): resolve MOD_Composition from a modification table once
            // diaflux libraries carry modification annotations.
            for &(elem, n) in mod_composition(mod_name) {
                match pos.get(elem) {
                    Some(&i) => formula[i].1 += n,
                    None => {
                        pos.insert(elem.to_string(), formula.len());
                        formula.push((elem.to_string(), n));
                    }
                }
            }
        }
    }

    formula
}

/// Modification elemental composition. Empty placeholder (see TODO above).
fn mod_composition(_mod_name: &str) -> &'static [(&'static str, i64)] {
    &[]
}

/// Generate precursor isotope intensities, mirroring
/// `alphabase.peptide.precursor.calc_precursor_isotope_intensity`.
///
/// Returns `(isotopes, mono_isotope_idx)` where `isotopes` is a flat
/// `n_precursors × max_isotope` row-major matrix and `mono_isotope_idx` records
/// the monoisotopic position within each row.
pub fn calc_precursor_isotope_intensity(
    sequences: &[String],
    mods_list: &[String],
    max_isotope: usize,
    min_right_most_intensity: f64,
) -> (Vec<f32>, Vec<i32>) {
    let table = ElementTable::default();
    let n = sequences.len();
    let mut isotopes = vec![0.0f32; n * max_isotope];
    let mut mono_idxes = vec![0i32; n];

    let mono_left_half = (max_isotope / 2) as i64;
    let mono_right_half = if max_isotope % 2 == 0 {
        mono_left_half
    } else {
        mono_left_half + 1
    };

    for i in 0..n {
        let formula = get_mod_seq_formula(&sequences[i], &mods_list[i]);
        let (mut dist, mono) = calc_formula_distribution(&table, &formula);

        for v in dist.iter_mut() {
            if *v <= min_right_most_intensity {
                *v = 0.0;
            }
        }

        let (row, mono_idx) = if mono < mono_left_half {
            let row = &dist[..max_isotope.min(dist.len())];
            (row, mono)
        } else if mono + mono_right_half >= dist.len() as i64 {
            let start = dist.len().saturating_sub(max_isotope);
            let row = &dist[start..];
            (row, max_isotope as i64 + mono - dist.len() as i64 + 1)
        } else {
            let start = (mono - mono_left_half) as usize;
            let end = (mono + mono_right_half) as usize;
            ( &dist[start..end], mono - mono_left_half)
        };

        for (k, &v) in row.iter().enumerate() {
            isotopes[i * max_isotope + k] = v as f32;
        }
        mono_idxes[i] = mono_idx as i32;
    }

    // Normalize to sum == 1 per row (`normalize == "sum"` default).
    for i in 0..n {
        let row = &mut isotopes[i * max_isotope..(i + 1) * max_isotope];
        let sum: f32 = row.iter().sum();
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    (isotopes, mono_idxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_precursors() -> SpecLibFlat {
        let mut spec = SpecLibFlat::default();
        let p = &mut spec.precursor_df;
        p.precursor_idx = vec![0, 1];
        p.sequence = vec!["ACDE".into(), "ACDE".into()];
        p.mods = vec!["".into(), "".into()];
        p.charge = vec![2, 3];
        p.precursor_mz = vec![500.0, 500.0];
        p.rt_pred = vec![10.0, 10.0];
        p.mobility_pred = vec![1.0, 1.0];
        p.decoy = vec![0, 1];
        p.n_fragments = vec![2, 2];

        let f = &mut spec.fragment_df;
        // Precursor 0: two fragments; precursor 1: two fragments.
        for (pid, mz, ty, num) in [
            (0u32, 200.0f64, 98u8, 1u16),
            (0, 400.0, 121, 2),
            (1, 200.0, 98, 1),
            (1, 400.0, 121, 2),
        ] {
            f.precursor_idx.push(pid);
            f.fragment_mz.push(mz);
            f.fragment_intensity.push(0.5);
            f.fragment_type.push(ty);
            f.fragment_number.push(num);
            f.fragment_charge.push(1);
        }
        spec
    }

    #[test]
    fn default_elution_groups_give_cardinality_one() {
        let mut spec = two_precursors();
        transform_library(&mut spec, 4);

        assert_eq!(spec.precursor_df.flat_frag_start_idx, vec![0, 2]);
        assert_eq!(spec.precursor_df.flat_frag_stop_idx, vec![2, 4]);
        assert_eq!(spec.precursor_df.elution_group_idx, vec![0, 1]);
        // Unique groups -> no shared fragments.
        assert_eq!(spec.fragment_df.cardinality, vec![1, 1, 1, 1]);
        assert_eq!(spec.fragment_df.mz_library, spec.fragment_df.fragment_mz);
        assert_eq!(spec.precursor_df.n_isotopes, 4);
        assert_eq!(spec.precursor_df.isotopes.len(), 8);
        assert_eq!(spec.precursor_df.mono_isotope_idx, vec![0, 0]);
    }

    #[test]
    fn shared_fragments_increment_cardinality() {
        // Two precursors in the SAME elution group sharing two identical m/z.
        let elution = vec![0u32, 0];
        let decoy = vec![0u8, 0];
        let start = vec![0u32, 2];
        let stop = vec![2u32, 4];
        let mz = vec![200.0, 400.0, 200.0, 400.0];
        let card = calc_fragment_cardinality(&elution, &decoy, &start, &stop, &mz, true);
        assert_eq!(card, vec![2, 2, 2, 2]);
    }
}


#[cfg(test)]
mod tests_isotopes {
    use super::*;

    #[test]
    fn empty_dist_identity() {
        let d = empty_dist();
        assert_eq!(d[0], 1.0);
        assert!(d[1..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn element_table_c_mono() {
        let t = ElementTable::default();
        let (dist, mono) = &t.dist["C"];
        assert_eq!(*mono, 0);
        assert!((dist[0] - 0.9893).abs() < 1e-6);
        assert!((dist[1] - 0.0107).abs() < 1e-6);
    }

    #[test]
    fn element_table_se_mono_is_not_zero() {
        // Se monoisotope is 80Se (position 6 on the rounded-mass grid).
        let t = ElementTable::default();
        let (dist, mono) = &t.dist["Se"];
        assert_eq!(*mono, 6);
        assert!((dist[6] - 0.4961).abs() < 1e-6);
    }

    #[test]
    fn convolution_is_associative_identity() {
        let t = ElementTable::default();
        // C(1) * empty == C(1)
        let (d, m) = abundance_convolution(&empty_dist(), 0, &t.dist["C"].0, t.dist["C"].1);
        assert_eq!(m, 0);
        assert!((d[0] - 0.9893).abs() < 1e-6);
    }

    #[test]
    fn precursor_isotopes_sum_to_one() {
        // Two simple peptides (unmodified).
        let seqs = vec!["ACDE".to_string(), "GGGG".to_string()];
        let mods = vec!["".to_string(), "".to_string()];
        let (iso, mono) = calc_precursor_isotope_intensity(&seqs, &mods, 4, 0.001);
        assert_eq!(iso.len(), 8);
        assert_eq!(mono.len(), 2);
        for i in 0..2 {
            let row = &iso[i * 4..(i + 1) * 4];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {} sum {}", i, sum);
            // monoisotopic peak is the largest for C/H/N/O/S peptides.
            assert!(row[0] >= row[1], "row {} mono not max", i);
        }
        // Standard peptide mono index is 0.
        assert_eq!(mono, vec![0, 0]);
    }
}
