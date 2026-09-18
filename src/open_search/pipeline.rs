//! Open Search = Standard Search + wide precursor tolerance (±500 Da).
//!
//! For each library precursor in SWATH window:
//!   1. Extract XIC at theoretical fragment m/z (same as seek_locus)
//!   2. Compute SA co-elution score
//!   3. If SA > threshold: confirmed
//!   4. Mass_shift = observed - library → modification detection
//!
//! No clustering, no fragment index — just standard DIA peptide-centric search.

use std::path::Path;
use anyhow::Result;
use crate::config::{config, mut_state};
use crate::log::info;
use crate::open_search::fdr_open::{DualSpaceFdr, ExtendedMatch};
use crate::open_search::frag_index::FragmentMatch;
use crate::open_search::xic::{extract_xics, gpu_simple_smooth};
use crate::open_search::coelution::cal_coelution_by_gaussian;
use crate::model::*;

pub struct OpenSearchRunConfig {
    pub tol_rt_ratio: f32,
    pub sa_threshold: f32,
}

impl Default for OpenSearchRunConfig {
    fn default() -> Self {
        Self { tol_rt_ratio: 0.0667, sa_threshold: 0.4 }
    }
}

pub fn search_core_open(
    cfg: &OpenSearchRunConfig,
    ms_data: &MsData,
    lib: &SpectralLibrary,
    out_dir: &Path,
) -> Result<Vec<ElutionGroup>> {
    info!("===== Open Search (Standard + wide precursor tolerance) =====");
    info!("Library: {} precursors", lib.precursors.len());

    // Set RT tolerance (same as standard search)
    let scan_rts_end = ms_data.scan_rts.last().copied().unwrap_or(0.0);
    let mut st = mut_state();
    st.tol_rt = scan_rts_end * cfg.tol_rt_ratio;
    st.locus_rt_thre = ms_data.cycle_time * 3.5;
    drop(st);
    let tol_rt = mut_state().tol_rt;
    // For non-IMS data (mzML), set tol_im to cover the constant IM=1.0
    let has_ims = ms_data.has_ims;
    let tol_im = if has_ims { mut_state().tol_im_xic } else { 1.0 };
    let tol_ppm = mut_state().tol_ppm;
    info!("  tol_rt={:.1}s, tol_im={:.3}, tol_ppm={:.1} (ims={})", tol_rt, tol_im, tol_ppm, has_ims);

    let swath_ids: Vec<u8> = {
        let mut ids: Vec<u8> = ms_data.ms2_maps.keys().copied().collect();
        ids.sort_unstable();
        ids
    };

    // Pre-filter: assign each precursor to its SWATH
    // ±500 Da range means a precursor could be in multiple SWATHs
    let swath_bounds: Vec<(f32, f32)> = (0..256usize)
        .map(|i| {
            if i+1 < ms_data.swath_boundaries.len() {
                (ms_data.swath_boundaries[i], ms_data.swath_boundaries[i+1])
            } else { (0.0, 2000.0) }
        })
        .collect();

    let mut all_matches: Vec<ExtendedMatch> = Vec::new();
    let batch_size = 500usize;

    for &sid in &swath_ids {
        let m2 = match ms_data.ms2_maps.get(&sid) { Some(m) => m, None => continue };
        let m1 = match ms_data.ms1_maps.get(&sid) { Some(m) => m, None => continue };
        if m2.mz_values_cen.is_empty() { continue; }

        let bounds = swath_bounds.get(sid as usize - 1).copied().unwrap_or((0.0, 2000.0));
        let (win_low, win_high) = bounds;

        // Collect all precursors within ±500 Da of this SWATH window
        let window_prs: Vec<usize> = lib.precursors.iter().enumerate()
            .filter(|(_, pr)| {
                let pm = pr.pr_mz;
                // Open search: library precursor mass can be up to ±500 Da away
                let expand_low = win_low - 500.0;
                let expand_high = win_high + 500.0;
                pm >= expand_low && pm <= expand_high
            })
            .map(|(i, _)| i)
            .collect();

        if window_prs.is_empty() { continue; }
        info!("  SWATH {} ({:.0}-{:.0}Da): {} precursors in ±500Da window",
            sid, win_low, win_high, window_prs.len());

        let mut swath_ids_found = 0usize;

        for chunk in window_prs.chunks(batch_size) {
            let mut batch = Vec::with_capacity(chunk.len());
            let mut loci = Vec::with_capacity(chunk.len());
            let mut ims = Vec::with_capacity(chunk.len());

            for &pi in chunk {
                let pr = &lib.precursors[pi];
                if pr.fragment_ions.is_empty() || pr.fg_num == 0 { continue; }

                // Use iRT to estimate RT locus
                let rt_est = pr.pred_irt * 60.0;
                let frac = (rt_est / scan_rts_end).clamp(0.0, 1.0);
                let locus = (frac * (ms_data.n_cycles - 1) as f32) as u32;

                batch.push(Precursor {
                    pr_id: pr.pr_id.clone(), pr_mz: pr.pr_mz, pr_charge: pr.pr_charge,
                    pred_irt: pr.pred_irt, pred_iim: pr.pred_iim,
                    fg_num: pr.fg_num, pr_index: pr.pr_index,
                    fragment_ions: pr.fragment_ions.clone(),
                    swath_id: sid,
                    pred_im: pr.pred_im, pred_rt: pr.pred_rt,
                    ..Default::default()
                });
                loci.push(locus);
                ims.push(pr.pred_iim.max(0.01));
            }
            if batch.is_empty() { continue; }

            // Batch XIC extraction (same as seek_locus in standard search)
            let (_, _, _, _, xics_raw) = extract_xics(
                &batch, &loci, m1, m2,
                tol_ppm, tol_im, Some(tol_rt), None,
                "center", true, Some(&ims),
            );

            if xics_raw.shape()[0] == 0 { continue; }

            // Batch SA scoring
            for bi in 0..xics_raw.shape()[0].min(batch.len()) {
                if xics_raw.shape()[2] < 7 { continue; }
                let sx = gpu_simple_smooth(&xics_raw.slice(ndarray::s![bi..bi+1, .., ..]).to_owned());
                let n_fg = batch[bi].fg_num as usize;
                let valids = vec![n_fg + 2];
                let (sa_scores, _) = cal_coelution_by_gaussian(&sx, 7, &valids);

                let center = sx.shape()[2] / 2;
                let sa_val = sa_scores[[0, center]];
                if sa_val < cfg.sa_threshold { continue; }

                // Compute mass_shift: observed vs library
                // In centroided mode we don't have MS1 precursor directly,
                // but the SWATH window center gives estimated observed m/z
                let observed_mz = (win_low + win_high) * 0.5; // approximation
                let pi = chunk[bi];
                let pr = &lib.precursors[pi];
                let mass_shift = observed_mz * pr.pr_charge as f32
                    - pr.pr_mz * pr.pr_charge as f32;

                // Count matched fragments via SA scoring internals
                let mut bm = Vec::new();
                let mut ym = Vec::new();
                for fi in &pr.fragment_ions {
                    match fi.annotation.ion_type {
                        IonType::B => bm.push((fi.annotation.series_number, fi.annotation.charge)),
                        _ => ym.push((fi.annotation.series_number, fi.annotation.charge)),
                    }
                }

                let mut em = ExtendedMatch::new(
                    FragmentMatch {
                        peptide_id: pr.pr_index,
                        matched_b: bm, matched_y: ym,
                        matched_immonium: vec![],
                        score_continuous: 0.0,
                        score_coverage: pr.fragment_ions.len() as f32 / pr.fg_num.max(1) as f32,
                        score_intensity: sa_val,
                        mass_shift,
                        combined_score: sa_val,
                    },
                    pr.decoy != 0, &pr.pr_id, "",
                );
                em.swath_id = sid;
                all_matches.push(em);
                swath_ids_found += 1;
            }
        }
        if swath_ids_found > 0 {
            info!("  SWATH {}: {} identifications", sid, swath_ids_found);
        }
    }

    info!("Total: {} identifications", all_matches.len());
    if all_matches.is_empty() { info!("No matches"); return Ok(vec![]); }

    // Deduplicate: keep the best score per unique (pr_id, decoy)
    let n_before = all_matches.len();
    use std::collections::HashMap;
    let mut best: HashMap<(String, bool), ExtendedMatch> = HashMap::new();
    for m in all_matches.drain(..) {
        let key = (m.peptide_sequence.clone(), m.is_decoy);
        let score = m.match_result.combined_score;
        let should_update = best.get(&key)
            .map(|e| score > e.match_result.combined_score)
            .unwrap_or(true);
        if should_update {
            best.insert(key, m);
        }
    }
    let mut all_matches: Vec<ExtendedMatch> = best.into_values().collect();
    info!("Deduplicated: {} → {} unique", n_before, all_matches.len());

    // FDR
    let mut fdr = DualSpaceFdr::new(0.01, 0.01);
    fdr.calculate_psm_fdr(&mut all_matches);
    let n_fdr = all_matches.iter().filter(|m| !m.is_decoy && m.psm_q_value < 0.01).count();
    info!("FDR: {} target PSMs ({} at q<0.01)",
        all_matches.iter().filter(|m| !m.is_decoy).count(), n_fdr);
    all_matches.retain(|m| m.psm_q_value < 0.01);

    // Modification report
    let mut hist: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for m in &all_matches {
        let k = (m.match_result.mass_shift * 100.0).round() as i32;
        *hist.entry(k).or_default() += 1;
    }
    if let Ok(mut rpt) = csv::Writer::from_path(&out_dir.join("open_search_mods.csv")) {
        rpt.write_record(&["mass_shift_da", "count", "type"]).ok();
        let mut s: Vec<_> = hist.into_iter().map(|(k, v)| (k as f32 * 0.01, v)).collect();
        s.sort_by(|a, b| b.1.cmp(&a.1));
        for (m, c) in &s {
            if *c < 2 { continue; }
            let t = match crate::open_search::fdr_open::classify_modification(*m, 'X') {
                Some((n, _)) => format!("known:{}", n),
                None => "novel".to_string(),
            };
            let _ = rpt.write_record([&format!("{:.3}", m), &c.to_string(), &t]);
        }
        rpt.flush()?;
        info!("Mod report: {} distinct shifts", s.len());
    }

    let groups: Vec<ElutionGroup> = all_matches.iter().filter(|m| !m.is_decoy).map(|m| {
        let c = m.match_result.combined_score;
        let mut sc = std::collections::HashMap::new();
        sc.insert("cscore_pr_run".into(), c);
        sc.insert("q_pr_run".into(), m.psm_q_value);
        sc.insert("mass_shift".into(), m.match_result.mass_shift);
        ElutionGroup {
            pr_index: m.match_result.peptide_id, pr_id: m.peptide_sequence.clone(),
            decoy: 0, swath_id: m.swath_id, locus: m.apex_cycle,
            xics: None, ims: None, mzs: None, rts: None,
            measure_rt: 0.0, measure_im: 0.0, measure_pr_mz: 0.0,
            sa_score: c, sa_scores_ion: vec![],
            score_center_deep: 0.0, score_left_deep: 0.0, score_1h_deep: 0.0,
            score_2h_deep: 0.0, score_big_deep: 0.0,
            score_center_deep_refine: 0.0, score_left_deep_refine: 0.0,
            score_1h_deep_refine: 0.0, score_2h_deep_refine: 0.0, score_big_deep_refine: 0.0,
            score_mall: 0.0, scores: sc,
            cscore_pr_run: c, q_pr_run: m.psm_q_value,
            cscore_pr_global: 0.0, q_pr_global: 1.0, group_rank: 1,
            protein_group: String::new(),
        }
    }).collect();
    info!("Open Search: {} ElutionGroups", groups.len());
    Ok(groups)
}
