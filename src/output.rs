//! Cross-run output serialization — parquet / CSV / DIA-NN format helpers.

use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use arrow::array::{Float32Array, Int16Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::fs::File;

use crate::log::info;
use crate::model::*;

// ===========================================================================
// Parquet output
// ===========================================================================

pub fn save_as_parquet(groups: &[ElutionGroup], out_path: &Path) -> Result<()> {
    use arrow::array::{Float32Array, Int16Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;

    let n = groups.len();
    let mut pr_ids = Vec::with_capacity(n);
    let mut pr_indices = Vec::with_capacity(n);
    let mut swath_ids = Vec::with_capacity(n);
    let mut decoys = Vec::with_capacity(n);
    let mut loci = Vec::with_capacity(n);
    let mut sa = Vec::with_capacity(n);
    let mut cscore = Vec::with_capacity(n);
    let mut q = Vec::with_capacity(n);
    let mut cscore_g = Vec::with_capacity(n);
    let mut q_g = Vec::with_capacity(n);
    let mut cscore_pg = Vec::with_capacity(n);
    let mut q_pg = Vec::with_capacity(n);
    let mut gr = Vec::with_capacity(n);
    let mut rt = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    let mut pg = Vec::with_capacity(n);

    for g in groups {
        pr_ids.push(g.pr_id.clone());
        pr_indices.push(g.pr_index as i16);
        swath_ids.push(g.swath_id as i16);
        decoys.push(g.decoy as i16);
        loci.push(g.locus as i16);
        sa.push(g.sa_score);
        cscore.push(g.scores.get("cscore_pr_run").copied().unwrap_or(0.0));
        q.push(g.scores.get("q_pr_run").copied().unwrap_or(1.0));
        cscore_g.push(g.scores.get("cscore_pr_global").copied().unwrap_or(0.0));
        q_g.push(g.scores.get("q_pr_global").copied().unwrap_or(1.0));
        cscore_pg.push(g.scores.get("cscore_pg_global").copied().unwrap_or(0.0));
        q_pg.push(g.scores.get("q_pg_global").copied().unwrap_or(1.0));
        gr.push(g.group_rank as i16);
        rt.push(g.measure_rt);
        im.push(g.measure_im);
        pg.push(g.protein_group.clone());
    }

    let schema = Schema::new(vec![
        Field::new("pr_id", DataType::Utf8, false),
        Field::new("pr_index", DataType::Int16, false),
        Field::new("swath_id", DataType::Int16, false),
        Field::new("decoy", DataType::Int16, false),
        Field::new("locus", DataType::Int16, false),
        Field::new("sa_score", DataType::Float32, false),
        Field::new("cscore_pr_run", DataType::Float32, false),
        Field::new("q_pr_run", DataType::Float32, false),
        Field::new("cscore_pr_global", DataType::Float32, false),
        Field::new("q_pr_global", DataType::Float32, false),
        Field::new("cscore_pg_global", DataType::Float32, false),
        Field::new("q_pg_global", DataType::Float32, false),
        Field::new("group_rank", DataType::Int16, false),
        Field::new("measure_rt", DataType::Float32, false),
        Field::new("measure_im", DataType::Float32, false),
        Field::new("protein_group", DataType::Utf8, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(pr_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
            Arc::new(Int16Array::from(pr_indices)),
            Arc::new(Int16Array::from(swath_ids)),
            Arc::new(Int16Array::from(decoys)),
            Arc::new(Int16Array::from(loci)),
            Arc::new(Float32Array::from(sa)),
            Arc::new(Float32Array::from(cscore)),
            Arc::new(Float32Array::from(q)),
            Arc::new(Float32Array::from(cscore_g)),
            Arc::new(Float32Array::from(q_g)),
            Arc::new(Float32Array::from(cscore_pg)),
            Arc::new(Float32Array::from(q_pg)),
            Arc::new(Int16Array::from(gr)),
            Arc::new(Float32Array::from(rt)),
            Arc::new(Float32Array::from(im)),
            Arc::new(StringArray::from(pg.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        ],
    )?;

    let file = File::create(out_path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    info!("Parquet saved: {} rows", n);
    Ok(())
}

/// Save elution groups as a CSV file with key columns.
pub fn save_as_csv(groups: &[ElutionGroup], out_path: &Path) -> Result<()> {
    let n = groups.len();
    if n == 0 { return Ok(()); }
    let mut wtr = csv::Writer::from_path(out_path)?;

    wtr.write_record(&[
        "pr_id", "decoy", "swath_id", "locus",
        "measure_rt", "measure_im", "measure_pr_mz",
        "sa_score", "cscore_pr_run", "q_pr_run",
        "cscore_pr_global", "q_pr_global",
        "protein_group",
    ])?;

    for g in groups {
        wtr.write_record(&[
            &g.pr_id,
            &g.decoy.to_string(),
            &g.swath_id.to_string(),
            &g.locus.to_string(),
            &format!("{:.4}", g.measure_rt),
            &format!("{:.6}", g.measure_im),
            &format!("{:.4}", g.measure_pr_mz),
            &format!("{:.6}", g.sa_score),
            &format!("{:.6}", g.scores.get("cscore_pr_run").copied().unwrap_or(0.0)),
            &format!("{:.8}", g.scores.get("q_pr_run").copied().unwrap_or(1.0)),
            &format!("{:.6}", g.scores.get("cscore_pr_global").copied().unwrap_or(0.0)),
            &format!("{:.8}", g.scores.get("q_pr_global").copied().unwrap_or(1.0)),
            &g.protein_group,
        ])?;
    }

    wtr.flush()?;
    info!("CSV saved: {} rows to {}", n, out_path.display());
    Ok(())
}

// ===========================================================================
// Transfer-library output (second pass) — consumed by the PeptDeep fine-tuner
// ===========================================================================

/// Write the transfer library as `transfer_precursors.parquet` (alphabase
/// precursor_df: sequence / mods / mod_sites / charge / rt_norm / frag_start_idx
/// / frag_stop_idx) + `transfer_fragments.parquet` (wide `b_z1/b_z2/y_z1/y_z2`
/// intensity matrix). The `frag_start_idx`/`frag_stop_idx` pointers index into
/// the *full* fragment matrix, matching peptdeep's `normalize_fragment_intensities`.
pub fn write_transfer_library(
    dir: &Path,
    lib: &crate::engine::transfer_library::TransferLibrary,
) -> Result<()> {
    use arrow::array::{UInt32Array, UInt8Array};
    std::fs::create_dir_all(dir)?;

    // --- precursors -------------------------------------------------------
    let n = lib.precursors.len();
    let mut seqs = Vec::with_capacity(n);
    let mut mods = Vec::with_capacity(n);
    let mut mod_sites = Vec::with_capacity(n);
    let mut charges = Vec::with_capacity(n);
    let mut n_aa = Vec::with_capacity(n);
    let mut rt_norm = Vec::with_capacity(n);
    let mut proba = Vec::with_capacity(n);
    let mut frag_start = Vec::with_capacity(n);
    let mut frag_stop = Vec::with_capacity(n);
    let mut use_for_ms2 = Vec::with_capacity(n);
    let mut median_corr = Vec::with_capacity(n);
    for p in &lib.precursors {
        seqs.push(p.sequence.clone());
        mods.push(p.mods.clone());
        mod_sites.push(p.mod_sites.clone());
        charges.push(p.charge);
        n_aa.push(p.n_aa);
        rt_norm.push(p.rt_norm);
        proba.push(p.proba);
        frag_start.push(p.frag_start_idx);
        frag_stop.push(p.frag_stop_idx);
        use_for_ms2.push(p.use_for_ms2);
        median_corr.push(p.median_correlation);
    }

    let prec_schema = Schema::new(vec![
        Field::new("sequence", DataType::Utf8, false),
        Field::new("mods", DataType::Utf8, false),
        Field::new("mod_sites", DataType::Utf8, false),
        Field::new("charge", DataType::UInt8, false),
        Field::new("nAA", DataType::UInt32, false),
        Field::new("rt_norm", DataType::Float32, false),
        Field::new("proba", DataType::Float32, false),
        Field::new("frag_start_idx", DataType::UInt32, false),
        Field::new("frag_stop_idx", DataType::UInt32, false),
        Field::new("use_for_ms2", DataType::UInt8, false),
        Field::new("median_correlation", DataType::Float32, false),
    ]);
    let prec_batch = RecordBatch::try_new(
        Arc::new(prec_schema),
        vec![
            Arc::new(StringArray::from(seqs.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
            Arc::new(StringArray::from(mods.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
            Arc::new(StringArray::from(mod_sites.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
            Arc::new(UInt8Array::from(charges)),
            Arc::new(UInt32Array::from(n_aa)),
            Arc::new(Float32Array::from(rt_norm)),
            Arc::new(Float32Array::from(proba)),
            Arc::new(UInt32Array::from(frag_start)),
            Arc::new(UInt32Array::from(frag_stop)),
            Arc::new(UInt8Array::from(use_for_ms2)),
            Arc::new(Float32Array::from(median_corr)),
        ],
    )?;
    let prec_path = dir.join("transfer_precursors.parquet");
    let file = File::create(&prec_path)?;
    let mut writer = ArrowWriter::try_new(file, prec_batch.schema(), None)?;
    writer.write(&prec_batch)?;
    writer.close()?;

    // --- fragments (transpose Vec<[f32;4]> -> 4 columns) ------------------
    let nf = lib.fragments.len();
    let mut b_z1 = Vec::with_capacity(nf);
    let mut b_z2 = Vec::with_capacity(nf);
    let mut y_z1 = Vec::with_capacity(nf);
    let mut y_z2 = Vec::with_capacity(nf);
    for f in &lib.fragments {
        b_z1.push(f[0]);
        b_z2.push(f[1]);
        y_z1.push(f[2]);
        y_z2.push(f[3]);
    }
    let frag_schema = Schema::new(vec![
        Field::new("b_z1", DataType::Float32, false),
        Field::new("b_z2", DataType::Float32, false),
        Field::new("y_z1", DataType::Float32, false),
        Field::new("y_z2", DataType::Float32, false),
    ]);
    let frag_batch = RecordBatch::try_new(
        Arc::new(frag_schema),
        vec![
            Arc::new(Float32Array::from(b_z1)),
            Arc::new(Float32Array::from(b_z2)),
            Arc::new(Float32Array::from(y_z1)),
            Arc::new(Float32Array::from(y_z2)),
        ],
    )?;
    let frag_path = dir.join("transfer_fragments.parquet");
    let file = File::create(&frag_path)?;
    let mut writer = ArrowWriter::try_new(file, frag_batch.schema(), None)?;
    writer.write(&frag_batch)?;
    writer.close()?;

    info!(
        "Transfer library written: {} precursors, {} fragment rows -> {}",
        n,
        nf,
        dir.display()
    );
    Ok(())
}

// ===========================================================================
// DIA-NN format output — matches `utils.py::convert_cols_to_diann()`
// ===========================================================================

pub fn convert_to_diann_format(
    groups: &[ElutionGroup], run_name: &str, out_path: &Path,
) -> Result<()> {
    let n = groups.len();
    if n == 0 { return Ok(()); }
    let fg_num = crate::config::config().fg_num;

    let mut file_names = Vec::with_capacity(n);
    let mut runs = Vec::with_capacity(n);
    let mut pr_ids = Vec::with_capacity(n);
    let mut charges = Vec::with_capacity(n);
    let mut proteotypic = Vec::with_capacity(n);
    let mut q_vals = Vec::with_capacity(n);
    let mut global_q = Vec::with_capacity(n);
    let mut pg_q = Vec::with_capacity(n);
    let mut global_pg_q = Vec::with_capacity(n);
    let mut rts = Vec::with_capacity(n);
    let mut ims = Vec::with_capacity(n);
    let mut pgs = Vec::with_capacity(n);
    let mut pids = Vec::with_capacity(n);
    let mut pnames = Vec::with_capacity(n);
    let mut qr = Vec::with_capacity(n);
    let mut qd = Vec::with_capacity(n);
    let mut qm = Vec::with_capacity(n);
    let mut pqr = Vec::with_capacity(n);
    let mut pqd = Vec::with_capacity(n);
    let mut pqm = Vec::with_capacity(n);
    let mut cscores = Vec::with_capacity(n);
    let mut iq_str = Vec::with_capacity(n);
    let mut is_str = Vec::with_capacity(n);

    for g in groups {
        if g.decoy != 0 { continue; }
        file_names.push(run_name.to_string());
        runs.push(run_name.to_string());
        pr_ids.push(g.pr_id.clone());
        charges.push(g.pr_id.chars().last().and_then(|c| c.to_digit(10)).unwrap_or(2) as i16);
        proteotypic.push(g.scores.get("proteotypic").copied().unwrap_or(1.0) as i16);
        pgs.push(g.protein_group.clone());
        pids.push(g.scores.get("protein_id").map(|_| String::new()).unwrap_or_default());
        pnames.push(g.scores.get("protein_name").map(|_| String::new()).unwrap_or_default());
        q_vals.push(g.scores.get("q_pr_run").copied().unwrap_or(1.0));
        global_q.push(g.scores.get("q_pr_global").copied().unwrap_or(1.0));
        pg_q.push(g.scores.get("q_pg_run").copied().unwrap_or(1.0));
        global_pg_q.push(g.scores.get("q_pg_global").copied().unwrap_or(1.0));
        qr.push(g.scores.get("quant_pr_raw").copied().unwrap_or(0.0));
        qd.push(g.scores.get("quant_pr_deep").copied().unwrap_or(0.0));
        qm.push(g.scores.get("quant_pr_mix").copied().unwrap_or(0.0));
        pqr.push(g.scores.get("quant_pg_raw").copied().unwrap_or(0.0));
        pqd.push(g.scores.get("quant_pg_deep").copied().unwrap_or(0.0));
        pqm.push(g.scores.get("quant_pg_mix").copied().unwrap_or(0.0));
        cscores.push(g.scores.get("cscore_pr_run").copied().unwrap_or(0.0));
        rts.push(g.measure_rt);
        ims.push(g.measure_im);

        let mut iqp = Vec::with_capacity(2 + fg_num);
        let mut isp = Vec::with_capacity(2 + fg_num);
        for i in 0..2 + fg_num {
            iqp.push(format!("{:.2}", g.scores.get(&format!("score_ion_quant_{}", i)).copied().unwrap_or(0.0)));
            isp.push(format!("{:.2}", g.scores.get(&format!("score_ion_sa_{}", i)).copied().unwrap_or(0.0)));
        }
        iq_str.push(iqp.join(";"));
        is_str.push(isp.join(";"));
    }

    let schema = Schema::new(vec![
        Field::new("File.Name", DataType::Utf8, false),
        Field::new("Run", DataType::Utf8, false),
        Field::new("Precursor.Id", DataType::Utf8, false),
        Field::new("Precursor.Charge", DataType::Int16, false),
        Field::new("Proteotypic", DataType::Int16, false),
        Field::new("Protein.Group", DataType::Utf8, false),
        Field::new("Protein.Ids", DataType::Utf8, false),
        Field::new("Protein.Names", DataType::Utf8, false),
        Field::new("Q.Value", DataType::Float32, false),
        Field::new("Global.Q.Value", DataType::Float32, false),
        Field::new("PG.Q.Value", DataType::Float32, false),
        Field::new("Global.PG.Q.Value", DataType::Float32, false),
        Field::new("Precursor.Quantity.Raw", DataType::Float32, false),
        Field::new("Precursor.Quantity.Deep", DataType::Float32, false),
        Field::new("Precursor.Quantity.Mix", DataType::Float32, false),
        Field::new("PG.Quantity.Raw", DataType::Float32, false),
        Field::new("PG.Quantity.Deep", DataType::Float32, false),
        Field::new("PG.Quantity.Mix", DataType::Float32, false),
        Field::new("CScore", DataType::Float32, false),
        Field::new("Fragment.Quant.Raw", DataType::Utf8, false),
        Field::new("Fragment.Correlations", DataType::Utf8, false),
        Field::new("RT", DataType::Float32, false),
        Field::new("IM", DataType::Float32, false),
    ]);

    let batch = RecordBatch::try_new(Arc::new(schema), vec![
        Arc::new(StringArray::from(file_names.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(runs.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(pr_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(Int16Array::from(charges)),
        Arc::new(Int16Array::from(proteotypic)),
        Arc::new(StringArray::from(pgs.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(pids.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(pnames.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(Float32Array::from(q_vals)),
        Arc::new(Float32Array::from(global_q)),
        Arc::new(Float32Array::from(pg_q)),
        Arc::new(Float32Array::from(global_pg_q)),
        Arc::new(Float32Array::from(qr)),
        Arc::new(Float32Array::from(qd)),
        Arc::new(Float32Array::from(qm)),
        Arc::new(Float32Array::from(pqr)),
        Arc::new(Float32Array::from(pqd)),
        Arc::new(Float32Array::from(pqm)),
        Arc::new(Float32Array::from(cscores)),
        Arc::new(StringArray::from(iq_str.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(is_str.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(Float32Array::from(rts)),
        Arc::new(Float32Array::from(ims)),
    ])?;

    let file = File::create(out_path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?; writer.close()?;
    info!("DIA-NN format: {} rows", n);
    Ok(())
}
