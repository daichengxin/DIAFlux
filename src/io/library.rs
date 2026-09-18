//! Spectral library loader — replaces `full_dia/library.py`
//!
//! Supports DIA-NN format `.parquet` spectral libraries.
//! Optimised for large libraries (millions of precursors) by filtering
//! Decoy rows at read time and minimising allocations.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use arrow::array::{Array, Float32Array, Float64Array, Int16Array, Int64Array, StringArray, UInt8Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::model::*;

// ===========================================================================
// Main entry point
// ===========================================================================

pub fn load_library(path: &str) -> Result<SpectralLibrary> {
    let path = Path::new(path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    let raw_df = match ext.as_str() {
        "parquet" => read_parquet(path)?,
        "tsv" => read_tsv(path)?,
        other => bail!("Unsupported library format: '.{other}'. Only .parquet and .tsv are supported."),
    };

    check_library(&raw_df)?;

    // Build precursors
    let mut precursors = Vec::with_capacity(raw_df.pr_id.len());
    let mut protein_map = Vec::with_capacity(raw_df.pr_id.len());

    for idx in 0..raw_df.pr_id.len() {
        let is_decoy = raw_df.decoy[idx] != 0;

        let pr_id = &raw_df.pr_id[idx];
        let charge = raw_df.pr_charge[idx] as u8;
        let fg_start = raw_df.fg_start_idx[idx];
        let fg_count = raw_df.fg_count[idx] as usize;
        let fg_end = fg_start + fg_count;

        let mut fragment_ions = Vec::with_capacity(fg_count);
        for fi in fg_start..fg_end {
            if fi >= raw_df.fg_mz.len() { break; }
            fragment_ions.push(FragmentIon {
                mz: raw_df.fg_mz[fi],
                height: raw_df.fg_height[fi],
                annotation: IonAnnotation::from_packed(raw_df.fg_anno[fi]),
                mz_left: 0.0, mz_1h: 0.0, mz_2h: 0.0,
            });
        }
        // DIA-NN Entry::init(): sort by height descending, normalize to [0,1]
        fragment_ions.sort_by(|a, b| b.height.partial_cmp(&a.height).unwrap_or(std::cmp::Ordering::Equal));
        fragment_ions.truncate(12);
        let max_h = fragment_ions.iter().map(|f| f.height).fold(0.0f32, f32::max);
        if max_h > 0.0 {
            for f in fragment_ions.iter_mut() { f.height /= max_h; }
        }

        let mod_seq = &raw_df.modified_sequence[idx];
        let (simple_seq, mods, mod_sites) = parse_modified_sequence(mod_seq);
        let strip_seq = simple_seq.to_uppercase();

        precursors.push(Precursor {
            pr_id: pr_id.clone(),
            pr_charge: charge,
            pr_mz: raw_df.pr_mz[idx],
            pr_len: (raw_df.pr_len[idx] as u8).min(100),
            pred_irt: raw_df.irt[idx],
            pred_iim: raw_df.iim[idx],
            fg_num: (fg_count as u8).min(12),
            pr_index: idx as u32,
            fragment_ions,
            simple_seq,
            strip_seq,
            mods,
            mod_sites,
            swath_id: 0,
            decoy: if is_decoy { 1u8 } else { 0u8 },
            elution_group_idx: 0,
            pred_rt: 0.0, pred_im: 0.0,
            measure_im: 0.0, measure_rt: 0.0,
            pr_mz_left: 0.0, pr_mz_1h: 0.0, pr_mz_2h: 0.0,
        });

        if idx < raw_df.protein_ids.len() {
            protein_map.push(PeptideProteinMap {
                pr_index: idx as u32,
                protein_id: raw_df.protein_ids[idx].clone(),
                protein_name: raw_df.protein_names[idx].clone(),
                gene: raw_df.genes.get(idx).cloned().unwrap_or_default(),
            });
        }
    }

    // Deduplicate by pr_id
    let mut seen_pr: HashMap<&str, usize> = HashMap::new();
    let mut deduped: Vec<Precursor> = Vec::new();
    let mut deduped_map: Vec<PeptideProteinMap> = Vec::new();
    for (i, pr) in precursors.iter().enumerate() {
        use std::collections::hash_map::Entry;
        match seen_pr.entry(&pr.pr_id) {
            Entry::Occupied(_) => {}
            Entry::Vacant(e) => {
                e.insert(i);
                deduped.push(pr.clone());
                if i < protein_map.len() {
                    deduped_map.push(protein_map[i].clone());
                }
            }
        }
    }

    Ok(SpectralLibrary {
        precursors: deduped,
        protein_map: deduped_map,
        lib_type: ext,
    })
}

// ===========================================================================
// Raw Library structure (flat columnar storage)
// ===========================================================================

#[derive(Debug, Default)]
struct RawLibrary {
    pr_id: Vec<String>,
    modified_sequence: Vec<String>,
    stripped_sequence: Vec<String>,
    pr_charge: Vec<i16>,
    decoy: Vec<i16>,
    irt: Vec<f32>,
    iim: Vec<f32>,
    pr_mz: Vec<f32>,
    pr_len: Vec<i16>,
    fg_count: Vec<i16>,
    fg_start_idx: Vec<usize>,
    protein_ids: Vec<String>,
    protein_names: Vec<String>,
    genes: Vec<String>,
    fg_mz: Vec<f32>,
    fg_height: Vec<f32>,
    fg_anno: Vec<i16>,
}

// ===========================================================================
// Parquet reader with streaming rows — minimises intermediate allocations
// ===========================================================================

fn read_parquet(path: &Path) -> Result<RawLibrary> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?;

    // Use larger batch size for better throughput
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(65536)
        .build()?;

    // Per-precursor storage. Fragments are grouped by `Precursor.Id` via a
    // HashMap because DIA-NN report rows are NOT guaranteed to be contiguous
    // per precursor (fragments can be scattered across blocks). The previous
    // contiguous-run assumption silently dropped most fragments.
    let mut pr_id: Vec<String> = Vec::new();
    let mut modified_sequence: Vec<String> = Vec::new();
    let mut stripped_sequence: Vec<String> = Vec::new();
    let mut pr_charge: Vec<i16> = Vec::new();
    let mut decoy: Vec<i16> = Vec::new();
    let mut irt: Vec<f32> = Vec::new();
    let mut iim: Vec<f32> = Vec::new();
    let mut pr_mz: Vec<f32> = Vec::new();
    let mut pr_len: Vec<i16> = Vec::new();
    let mut protein_ids: Vec<String> = Vec::new();
    let mut protein_names: Vec<String> = Vec::new();
    let mut genes: Vec<String> = Vec::new();

    let mut frag_mz: Vec<Vec<f32>> = Vec::new();
    let mut frag_height: Vec<Vec<f32>> = Vec::new();
    let mut frag_anno: Vec<Vec<i16>> = Vec::new();

    let mut pr_index: HashMap<String, usize> = HashMap::new();
    let mut batch_count = 0usize;

    for batch_result in reader {
        let batch = batch_result?;
        batch_count += 1;
        if batch_count % 50 == 0 {
            eprintln!("[LIB] Processing batch {}, {} precursors so far...", batch_count, pr_id.len());
        }

        let pr_id_arry = as_string_col(&batch, "Precursor.Id")?;
        let mod_seq = as_string_col(&batch, "Modified.Sequence")?;
        let strip_seq = as_string_col(&batch, "Stripped.Sequence")?;
        let charge = as_i16_col(&batch, "Precursor.Charge")?;
        let decoy_arr = as_i16_col(&batch, "Decoy")?;
        let rt = as_f32_col(&batch, "RT")?;
        let im = as_f32_col(&batch, "IM")?;
        let mz = as_f32_col(&batch, "Precursor.Mz")?;
        let prod_mz = as_f32_col(&batch, "Product.Mz")?;
        let intensity = as_f32_col(&batch, "Relative.Intensity")?;
        let fg_type = as_string_col(&batch, "Fragment.Type")?;
        let fg_charge = as_i16_col(&batch, "Fragment.Charge")?;
        let fg_series = as_i16_col(&batch, "Fragment.Series.Number")?;
        let prot_ids = as_string_col(&batch, "Protein.Ids")?;
        let prot_names = as_string_col(&batch, "Protein.Names")?;

        let n = batch.num_rows();
        for row in 0..n {
            let current_pr = &pr_id_arry[row];
            let is_decoy = decoy_arr[row] != 0;

            let idx = match pr_index.get(current_pr) {
                Some(&i) => i,
                None => {
                    let i = pr_id.len();
                    pr_index.insert(current_pr.clone(), i);
                    pr_id.push(current_pr.clone());
                    modified_sequence.push(mod_seq[row].clone());
                    let seq_str = strip_seq[row].clone();
                    stripped_sequence.push(seq_str.clone());
                    pr_charge.push(charge[row]);
                    decoy.push(decoy_arr[row]);
                    irt.push(rt[row]);
                    iim.push(im[row]);
                    pr_mz.push(mz[row]);
                    pr_len.push(seq_str.len() as i16);
                    protein_ids.push(prot_ids[row].clone());
                    protein_names.push(prot_names[row].clone());
                    genes.push(String::new());
                    frag_mz.push(Vec::new());
                    frag_height.push(Vec::new());
                    frag_anno.push(Vec::new());
                    i
                }
            };

            // Skip fragment data for decoys (decoy library is regenerated downstream)
            if is_decoy { continue; }

            let anno = match fg_type[row].as_str() {
                "b" => 1i16 * 1000 + fg_series[row] * 10 + fg_charge[row],
                "y" => 2i16 * 1000 + fg_series[row] * 10 + fg_charge[row],
                _ => 3i16 * 1000 + fg_series[row] * 10 + fg_charge[row],
            };

            frag_mz[idx].push(prod_mz[row]);
            frag_height[idx].push(intensity[row]);
            frag_anno[idx].push(anno);
        }
    }

    // Flatten per-precursor fragments into contiguous columnar arrays.
    let mut raw = RawLibrary::default();
    raw.pr_id = pr_id;
    raw.modified_sequence = modified_sequence;
    raw.stripped_sequence = stripped_sequence;
    raw.pr_charge = pr_charge;
    raw.decoy = decoy;
    raw.irt = irt;
    raw.iim = iim;
    raw.pr_mz = pr_mz;
    raw.pr_len = pr_len;
    raw.protein_ids = protein_ids;
    raw.protein_names = protein_names;
    raw.genes = genes;

    let total_frags: usize = frag_mz.iter().map(|v| v.len()).sum();
    raw.fg_mz = Vec::with_capacity(total_frags);
    raw.fg_height = Vec::with_capacity(total_frags);
    raw.fg_anno = Vec::with_capacity(total_frags);
    raw.fg_start_idx = Vec::with_capacity(raw.pr_id.len());
    raw.fg_count = Vec::with_capacity(raw.pr_id.len());

    for i in 0..raw.pr_id.len() {
        raw.fg_start_idx.push(raw.fg_mz.len());
        raw.fg_count.push(frag_mz[i].len() as i16);
        raw.fg_mz.extend_from_slice(&frag_mz[i]);
        raw.fg_height.extend_from_slice(&frag_height[i]);
        raw.fg_anno.extend_from_slice(&frag_anno[i]);
    }

    eprintln!("[LIB] Loaded: {} precursors, {} fragment entries", raw.pr_id.len(), raw.fg_mz.len());
    Ok(raw)
}

// ===========================================================================
// TSV reader
// ===========================================================================

fn read_tsv(path: &Path) -> Result<RawLibrary> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;

    let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();

    let required = [
        "Precursor.Id", "Modified.Sequence", "Precursor.Charge", "Decoy",
        "RT", "IM", "Precursor.Mz", "Product.Mz", "Relative.Intensity",
        "Fragment.Type", "Fragment.Charge", "Fragment.Series.Number",
        "Protein.Ids", "Protein.Names",
    ];
    for name in &required {
        if !headers.contains(&name.to_string()) {
            bail!("Missing required column '{name}' in TSV library");
        }
    }

    let mut raw = RawLibrary::default();
    let mut all_fg_mz = Vec::new();
    let mut all_fg_height = Vec::new();
    let mut all_fg_anno: Vec<i16> = Vec::new();

    for result in reader.records() {
        let record = result?;
        let pr_id = get_tsv(&record, &headers, "Precursor.Id");
        let pr_mz: f32 = get_tsv(&record, &headers, "Precursor.Mz").parse().unwrap_or(0.0);
        let product_mz: f32 = get_tsv(&record, &headers, "Product.Mz").parse().unwrap_or(0.0);
        let intensity: f32 = get_tsv(&record, &headers, "Relative.Intensity").parse().unwrap_or(0.0);
        let charge: i16 = get_tsv(&record, &headers, "Precursor.Charge").parse().unwrap_or(0);
        let fg_charge: i16 = get_tsv(&record, &headers, "Fragment.Charge").parse().unwrap_or(0);
        let fg_series: i16 = get_tsv(&record, &headers, "Fragment.Series.Number").parse().unwrap_or(0);
        let fg_type_str = get_tsv(&record, &headers, "Fragment.Type");
        let decoy: i16 = get_tsv(&record, &headers, "Decoy").parse().unwrap_or(0);
        let rt: f32 = get_tsv(&record, &headers, "RT").parse().unwrap_or(0.0);
        let im: f32 = get_tsv(&record, &headers, "IM").parse().unwrap_or(0.0);
        let mod_seq = get_tsv(&record, &headers, "Modified.Sequence");
        let strip_seq = get_tsv(&record, &headers, "Stripped.Sequence");
        let protein_ids = get_tsv(&record, &headers, "Protein.Ids");
        let protein_names = get_tsv(&record, &headers, "Protein.Names");

        let seq_len = strip_seq.len() as i16;
        let is_new = raw.pr_id.last().map_or(true, |last| last != &pr_id);
        if is_new {
            raw.pr_id.push(pr_id);
            raw.modified_sequence.push(mod_seq);
            raw.stripped_sequence.push(strip_seq);
            raw.pr_charge.push(charge);
            raw.decoy.push(decoy);
            raw.irt.push(rt);
            raw.iim.push(im);
            raw.pr_mz.push(pr_mz);
            raw.pr_len.push(seq_len);
            raw.fg_start_idx.push(all_fg_mz.len());
            raw.protein_ids.push(protein_ids);
            raw.protein_names.push(protein_names);
            raw.genes.push(String::new());
        }

        let fg_type = match fg_type_str.as_str() { "b" => 1i16, "y" => 2i16, _ => 3i16 };
        let anno = fg_type * 1000 + fg_series * 10 + fg_charge;
        all_fg_mz.push(product_mz);
        all_fg_height.push(intensity);
        all_fg_anno.push(anno);
    }

    for i in 0..raw.fg_start_idx.len() {
        let start = raw.fg_start_idx[i];
        let end = if i + 1 < raw.fg_start_idx.len() { raw.fg_start_idx[i+1] } else { all_fg_mz.len() };
        raw.fg_count.push((end - start) as i16);
    }

    raw.fg_mz = all_fg_mz;
    raw.fg_height = all_fg_height;
    raw.fg_anno = all_fg_anno;

    Ok(raw)
}

fn get_tsv<'a>(record: &'a csv::StringRecord, headers: &[String], name: &str) -> String {
    headers.iter().position(|h| h == name)
        .and_then(|i| record.get(i))
        .unwrap_or("")
        .to_string()
}

// ===========================================================================
// Library validation
// ===========================================================================

fn check_library(raw: &RawLibrary) -> Result<()> {
    if raw.pr_id.is_empty() {
        bail!("Spectral library is empty");
    }

    for mod_seq in &raw.modified_sequence {
        let cleaned = mod_seq
            .replace("C(UniMod:4)", "c")
            .replace("M(UniMod:35)", "m");
        if cleaned.contains(|c: char| c == '(' || c == ')' || c.is_ascii_digit()) {
            bail!("Library contains unexpected modifications. Only C(UniMod:4) and M(UniMod:35) allowed.");
        }
    }

    if let Some(&max_charge) = raw.pr_charge.iter().max() {
        if max_charge > 4 {
            bail!("Library contains > 4 charge state. Only charge 1-4 are allowed.");
        }
    }

    for &anno in &raw.fg_anno {
        let fg_type = anno / 1000;
        if fg_type == 0 || fg_type > 3 {
            bail!("Library contains unexpected fragment type (anno={anno}). Only b/y ions allowed.");
        }
    }

    Ok(())
}

// ===========================================================================
// Utility: strip modifications
// ===========================================================================

/// Map a DIA-NN `UniMod:NN` token to its alphabase modification base name
/// (without the `@AA` site suffix). Unknown IDs fall back to the raw token so
/// the modification is still preserved distinctly.
fn unimod_base_name(token: &str) -> String {
    let id: u32 = token.trim_start_matches("UniMod:").parse().unwrap_or(0);
    let base = match id {
        1 => "Acetyl",
        4 => "Carbamidomethyl",
        7 => "Deamidated",
        21 => "Phospho",
        26 => "Pyro-carbamidomethyl",
        27 => "Glu->pyro-Glu",
        28 => "Gln->pyro-Glu",
        34 => "Methyl",
        35 => "Oxidation",
        36 => "Dimethyl",
        40 => "Carbamyl",
        _ => return token.to_string(),
    };
    base.to_string()
}

/// Parse a DIA-NN `Modified.Sequence` (e.g. `AC(UniMod:4)DEM(UniMod:35)K`) into
/// the alphabase `(naked_sequence, mods, mod_sites)` representation.
///
/// * `naked_sequence` — uses lowercase letters for modified residues (alphabase
///   `lower_case_AA` convention); the caller derives `strip_seq` via `.to_uppercase()`.
/// * `mods` — `;`-separated `ModName@AA` list (e.g. `Carbamidomethyl@C;Oxidation@M`).
/// * `mod_sites` — `;`-separated list of 1-indexed sites (`0` = N-term).
fn parse_modified_sequence(seq: &str) -> (String, String, String) {
    let bytes = seq.as_bytes();
    let mut naked = String::with_capacity(seq.len());
    let mut mods: Vec<String> = Vec::new();
    let mut sites: Vec<String> = Vec::new();
    let mut i = 0;
    let mut aa_pos = 0usize; // 1-indexed position of the most recently seen AA

    while i < bytes.len() {
        let c = bytes[i];
        if c == b'(' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b')' {
                j += 1;
            }
            if j >= bytes.len() {
                i += 1; // malformed; drop the '('
                continue;
            }
            let token = &seq[i + 1..j];
            let base = unimod_base_name(token);
            if aa_pos == 0 {
                // N-terminal modification (no preceding AA).
                mods.push(format!("{base}@Any_N-term"));
                sites.push("0".to_string());
            } else {
                // Modification on the last AA; mark it lowercase in `naked`.
                let aa = naked.pop().unwrap().to_ascii_uppercase();
                naked.push(aa.to_ascii_lowercase());
                mods.push(format!("{base}@{aa}"));
                sites.push(aa_pos.to_string());
            }
            i = j + 1;
            continue;
        }
        if c.is_ascii_alphabetic() {
            naked.push(c as char);
            aa_pos += 1;
        }
        i += 1;
    }

    (naked, mods.join(";"), sites.join(";"))
}

#[cfg(test)]
mod tests {
    use super::parse_modified_sequence;

    #[test]
    fn parses_unmodified_sequence() {
        let (naked, mods, sites) = parse_modified_sequence("ACDEFGHIK");
        assert_eq!(naked, "ACDEFGHIK");
        assert_eq!(mods, "");
        assert_eq!(sites, "");
    }

    #[test]
    fn parses_oxidation_at_methionine() {
        let (naked, mods, sites) = parse_modified_sequence("SEVDM(UniMod:35)LK");
        assert_eq!(naked, "SEVDmLK"); // lowercase marks the modified residue
        assert_eq!(mods, "Oxidation@M");
        assert_eq!(sites, "5");
    }

    #[test]
    fn parses_two_modifications_in_order() {
        let (naked, mods, sites) = parse_modified_sequence("AC(UniMod:4)DEM(UniMod:35)K");
        assert_eq!(naked, "AcDEmK");
        assert_eq!(mods, "Carbamidomethyl@C;Oxidation@M");
        assert_eq!(sites, "2;5");
    }

    #[test]
    fn parses_nterm_modification() {
        let (naked, mods, sites) = parse_modified_sequence("(UniMod:1)ACDEFK");
        assert_eq!(naked, "ACDEFK");
        assert_eq!(mods, "Acetyl@Any_N-term");
        assert_eq!(sites, "0");
    }

    #[test]
    fn unknown_unimod_is_preserved() {
        let (naked, mods, sites) = parse_modified_sequence("ACD(UniMod:999)EFK");
        assert_eq!(naked, "ACdEFK");
        assert_eq!(mods, "UniMod:999@D");
        assert_eq!(sites, "3");
    }
}

// ===========================================================================
// Polish library by SWATH
// ===========================================================================

pub fn polish_by_swath(
    lib: SpectralLibrary,
    ms_data: &MsData,
) -> Result<SpectralLibrary> {
    use rayon::prelude::*;
    let swath = &ms_data.swath_boundaries;
    if swath.len() < 2 { bail!("SWATH boundaries unavailable"); }
    let mz_min = swath[0];
    let mz_max = swath[swath.len() - 1];
    let mass_neutron = 1.0033548378;
    let n_precursors = lib.precursors.len();
    let protein_map = lib.protein_map;

    // into_par_iter() moves data (no clone overhead), filter + modify in parallel
    let mut polished: Vec<(Precursor, Option<PeptideProteinMap>)> = lib.precursors
        .into_par_iter()
        .enumerate()
        .filter(|(_, pr)| pr.pr_mz > mz_min && pr.pr_mz < mz_max)
        .filter(|(_, pr)| pr.fg_num >= 4)
        .filter(|(_, pr)| !pr.simple_seq.chars().any(|c| "BJOUXZbjouxz".contains(c)))
        .map(|(i, mut p)| {
            let ch = p.pr_charge as f32;
            p.pr_mz_left = (p.pr_mz * ch - mass_neutron) / ch.max(1.0);
            p.pr_mz_1h = (p.pr_mz * ch + mass_neutron) / ch.max(1.0);
            p.pr_mz_2h = (p.pr_mz * ch + 2.0 * mass_neutron) / ch.max(1.0);
            p.swath_id = swath.windows(2)
                .position(|w| p.pr_mz >= w[0] && p.pr_mz < w[1])
                .map(|si| (si + 1) as u8).unwrap_or(1);
            for fi in p.fragment_ions.iter_mut() {
                let fch = fi.annotation.charge as f32;
                if fch > 0.0 && fi.mz > 0.0 {
                    fi.mz_left = (fi.mz * fch - mass_neutron) / fch;
                    fi.mz_1h = (fi.mz * fch + mass_neutron) / fch;
                    fi.mz_2h = (fi.mz * fch + 2.0 * mass_neutron) / fch;
                }
            }
            let pmap = if i < protein_map.len() { Some(protein_map[i].clone()) } else { None };
            (p, pmap)
        })
        .collect();

    polished.par_sort_by_key(|(p, _)| p.swath_id);
    let mut sorted_precursors = Vec::with_capacity(polished.len());
    let mut sorted_map = Vec::with_capacity(polished.len());
    for (p, m) in polished {
        sorted_precursors.push(p);
        if let Some(pmap) = m { sorted_map.push(pmap); }
    }

    Ok(SpectralLibrary { precursors: sorted_precursors, protein_map: sorted_map, lib_type: lib.lib_type.clone() })
}

// ===========================================================================
// Arrow helpers
// ===========================================================================

fn as_string_col(batch: &arrow::record_batch::RecordBatch, name: &str) -> Result<Vec<String>> {
    let col = batch.column_by_name(name)
        .with_context(|| format!("Column '{name}' not found"))?;
    let arr = col.as_any().downcast_ref::<StringArray>()
        .with_context(|| format!("Column '{name}' is not string type"))?;
    Ok((0..arr.len()).map(|i| arr.value(i).to_string()).collect())
}

fn as_f32_col(batch: &arrow::record_batch::RecordBatch, name: &str) -> Result<Vec<f32>> {
    let col = batch.column_by_name(name)
        .with_context(|| format!("Column '{name}' not found"))?;
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return Ok(arr.values().to_vec());
    }
    // Some libraries (e.g. from convert_speclib.py) store floats as f64.
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
        return Ok(arr.values().iter().map(|&v| v as f32).collect());
    }
    bail!("Column '{name}' is neither float32 nor float64");
}

fn as_i16_col(batch: &arrow::record_batch::RecordBatch, name: &str) -> Result<Vec<i16>> {
    let col = batch.column_by_name(name)
        .with_context(|| format!("Column '{name}' not found"))?;
    if let Some(arr) = col.as_any().downcast_ref::<Int16Array>() {
        return Ok(arr.values().to_vec());
    }
    if let Some(arr) = col.as_any().downcast_ref::<UInt8Array>() {
        return Ok(arr.values().iter().map(|&x| x as i16).collect());
    }
    if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
        return Ok(arr.values().iter().map(|&x| x as i16).collect());
    }
    bail!("Column '{name}' is not Int16/UInt8/Int64");
}

use crate::log::info;