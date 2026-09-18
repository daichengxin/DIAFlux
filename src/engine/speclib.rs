//! Spectral-library adapters: convert the DIA-NN `SpectralLibrary` (diaflux
//! `types.rs`) into the columnar `SpecLibFlat` used by selection and scoring.

use std::collections::HashMap;

use crate::engine::data::{FragmentDf, PrecursorDf, SpecLibFlat};
use crate::model::{IonType, SpectralLibrary};

/// Convert diaflux's `SpectralLibrary` into the flat `SpecLibFlat`.
///
/// The DIA-NN `Precursor` carries fragment ions inline; the engine stores them
/// in a separate fragment table keyed by `precursor_idx`. Predicted RT/IM are
/// read from the calibrated `pred_rt`/`pred_im` fields (falling back to the
/// iRT / iIM predictions when not yet calibrated).
pub fn from_spectral_library(lib: &SpectralLibrary) -> SpecLibFlat {
    let mut precursor_df = PrecursorDf::default();
    let mut fragment_df = FragmentDf::default();

    for p in &lib.precursors {
        precursor_df.precursor_idx.push(p.pr_index);
        precursor_df.sequence.push(p.strip_seq.clone());
        precursor_df.mods.push(p.mods.clone());
        precursor_df.mod_sites.push(p.mod_sites.clone());
        precursor_df.charge.push(p.pr_charge);
        precursor_df.precursor_mz.push(p.pr_mz as f64);
        // Calibrated RT if set, otherwise the predicted iRT.
        let rt = if p.pred_rt != 0.0 {
            p.pred_rt
        } else {
            p.pred_irt
        };
        let im = if p.pred_im != 0.0 {
            p.pred_im
        } else {
            p.pred_iim
        };
        precursor_df.rt_pred.push(rt as f64);
        precursor_df.mobility_pred.push(im as f64);
        precursor_df.decoy.push(p.decoy);
        precursor_df.elution_group_idx.push(p.elution_group_idx);
        precursor_df.n_fragments.push(p.fragment_ions.len() as u32);

        for f in &p.fragment_ions {
            fragment_df.precursor_idx.push(p.pr_index);
            fragment_df.fragment_mz.push(f.mz as f64);
            fragment_df.fragment_intensity.push(f.height);
            fragment_df.fragment_type.push(ion_type_to_u8(f.annotation.ion_type));
            fragment_df.fragment_number.push(f.annotation.series_number as u16);
            fragment_df.fragment_charge.push(f.annotation.charge);
        }
    }

    // Protein mapping (many-to-one on precursor_idx). Index `protein_map` once
    // so this is O(precursors + protein_map) rather than O(precursors ×
    // protein_map), which was a quadratic blow-up on large libraries.
    let mut protein_index: HashMap<u32, Vec<&str>> = HashMap::new();
    for m in &lib.protein_map {
        protein_index
            .entry(m.pr_index)
            .or_default()
            .push(m.protein_id.as_str());
    }
    for p in &lib.precursors {
        let proteins = protein_index
            .get(&p.pr_index)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        precursor_df.proteins.push(proteins.join(";"));
    }

    let mut spec_lib = SpecLibFlat {
        precursor_df,
        fragment_df,
        precursor_index_map: Default::default(),
    };
    spec_lib.rebuild_index();
    spec_lib
}

/// Map a DIA-NN ion type to the reference `series_id` (ASCII code of the series:
/// a=97, b=98, c=99, x=120, y=121, z=122).
#[inline]
fn ion_type_to_u8(t: IonType) -> u8 {
    match t {
        IonType::B => 98,   // b
        IonType::Y => 121,  // y
        // Neutral-loss fragments are reported by DIA-NN without the loss type;
        // represented as a b-series ion (loss_type defaults to 0 in the library
        // transform).
        IonType::Loss => 98,
    }
}
