//! DIAFluX: open-source DIA proteomics data analysis pipeline.
//!
//! CLI entry point. Two modes:
//!   default       — DIA search (selection → scoring → FDR)
//!   --open-search — Fragment-centric open search (wide mass tolerance)

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context};
use clap::Parser;
use diaflux::cli::{print_run_info, Cli};
use diaflux::config::Config;
use diaflux::log::info;

/// The peptdeep library-prediction sidecar, embedded into the binary so the
/// tool is self-contained (no dependence on the source checkout at runtime).
const SIDECAR_SCRIPT: &str = include_str!("../python/diaflux_peptdeep.py");

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    let ws_raw = std::path::Path::new(&cli.ws);
    let ws_path = if ws_raw.is_file() {
        ws_raw.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| ws_raw.to_path_buf())
    } else {
        ws_raw.to_path_buf()
    };
    let ws_path = ws_path.canonicalize()?;
    let out_dir = ws_path.join(&cli.out_name);
    std::fs::create_dir_all(&out_dir)?;
    diaflux::log::init_logger(&out_dir)?;
    print_run_info(&cli);

    Config::init(None, cli.gpu_id, cli.low_memory);

    let multi_ws = discover_workspaces(ws_raw);
    let file_num = multi_ws.len();
    info!("Found {} workspace(s)", file_num);

    if !cli.open_search {
        // ── DIA search mode (peak-group NG) ─────────────────────
        // Library source: an existing `--lib`, or a peptdeep-predicted library
        // generated from `--fasta` (cached at <out>/<out_name>_predlib.parquet).
        let pass1_lib_out = match &cli.lib_out {
            Some(p) => PathBuf::from(p),
            None => out_dir.join(format!("{}_predlib.parquet", cli.out_name)),
        };
        let lib_path: PathBuf = match (&cli.lib, &cli.fasta) {
            (Some(lib), _) => PathBuf::from(lib),
            (None, Some(fasta)) => predict_library(&cli, &out_dir, fasta, None, &pass1_lib_out)?,
            (None, None) => bail!("one of --lib or --fasta is required"),
        };
        let lib_path = lib_path.to_string_lossy().to_string();
        info!("DIA search mode enabled, loading library: {}", lib_path);
        let lib = diaflux::io::library::load_library(&lib_path)?;
        info!("Library loaded: {} precursors", lib.len());
        let lib = diaflux::decoy::generate_decoy_library(lib);
        info!("Library with decoys: {} precursors", lib.len());
        // Pre-transform the library once per pass; shared across all replicate
        // runs (the fragment index is identical for every file).
        let mut prepared = diaflux::engine::prepare_ng_library(&lib);

        let mut ng_config = diaflux::engine::NgConfig::default();
        let transfer_config = diaflux::engine::transfer_quant::TransferQuantConfig::default();
        let mut all_psms: Vec<diaflux::engine::transfer_quant::QuantPsm> = Vec::new();

        // ── Pass 1: predicted-library search (+ transfer requantification) ──
        for (i, ws_single) in multi_ws.iter().enumerate() {
            info!("================ Run: {}/{} ================", i + 1, file_num);
            info!("File: {}", ws_single.file_name().unwrap().to_string_lossy());
            let run_name = ws_single.file_name().unwrap().to_string_lossy().to_string();
            let out_file = out_dir.join(format!("core_results_{}.parquet", run_name));
            if out_file.exists() && !cli.overwrite { info!("Skipping"); continue; }

            // Build the DIA data model from the raw file.
            let dia_data = load_dia_data(ws_single)?;
            info!("DiaData: {} frames | {} cycles | {} mobility scans",
                dia_data.frame_max_index, dia_data.precursor_cycle_max_index,
                dia_data.scan_max_index);

            let result = diaflux::engine::run_ng(&mut prepared, &dia_data, &ng_config);
            info!("DIA search: {} PSMs (q<=0.01), precursor quant groups: {}",
                result.n_psm,
                result.quant.precursor.as_ref().map(|q| q.group_ids.len()).unwrap_or(0));

            if cli.two_pass {
                let qpsms = diaflux::engine::transfer_quant::transfer_quantify(
                    &lib,
                    &dia_data,
                    &result.precursor_idx,
                    &result.proba,
                    &result.qval,
                    &result.rt_observed,
                    &transfer_config,
                );
                all_psms.extend(qpsms);
            }

            write_core_output(&out_dir, &run_name, &result, "")?;
        }

        // ── Pass 2: transfer-learning library refinement ──────────────────
        if cli.two_pass {
            if all_psms.is_empty() {
                bail!("--two-pass: no q<=0.01 PSMs to build a transfer library");
            }
            let transfer_lib = diaflux::engine::transfer_library::build_transfer_library(
                &all_psms,
                &diaflux::engine::transfer_library::TransferLibraryConfig::default(),
            );
            let transfer_dir = match &cli.transfer_out {
                Some(p) => PathBuf::from(p),
                None => out_dir.join("transfer"),
            };
            diaflux::output::write_transfer_library(&transfer_dir, &transfer_lib)?;

            // Second-pass fragment m/z tolerance: derive from the first-pass
            // observed mass errors (the reference's automatic MS2 optimizer),
            // unless explicitly overridden. Pass 1 searches at 7.0 ppm. Computed
            // here (independent of fine-tuning) so it is logged before the slow
            // fine-tune + re-predict steps.
            let pass2_ms2_tol = match cli.pass2_ms2_tol {
                Some(t) if t > 0.0 => t as f64,
                _ => diaflux::engine::transfer_quant::compute_ms2_tolerance(&all_psms)
                    .unwrap_or(7.0),
            };
            info!("Pass 2 fragment m/z tolerance: {:.2} ppm (pass 1 = 7.0 ppm)", pass2_ms2_tol);
            ng_config.selection.mass_tolerance = pass2_ms2_tol;

            let model_dir = transfer_dir.join("models");
            run_finetune(&cli, &out_dir, &transfer_dir, &model_dir)?;

            // Re-predict the library with the fine-tuned models, then re-search.
            let fasta = cli.fasta.as_deref()
                .expect("--two-pass requires --fasta");
            let pass2_lib_out = out_dir.join(format!("{}_predlib_tl.parquet", cli.out_name));
            let pass2_lib_path = predict_library(&cli, &out_dir, fasta, Some(&model_dir.to_string_lossy()), &pass2_lib_out)?;
            info!("Pass 2 library: {}", pass2_lib_path.display());
            let lib2 = diaflux::io::library::load_library(&pass2_lib_path.to_string_lossy())?;
            let lib2 = diaflux::decoy::generate_decoy_library(lib2);
            info!("Pass 2 library with decoys: {} precursors", lib2.len());
            let mut prepared2 = diaflux::engine::prepare_ng_library(&lib2);

            for (i, ws_single) in multi_ws.iter().enumerate() {
                info!("================ Pass 2 Run: {}/{} ================", i + 1, file_num);
                info!("File: {}", ws_single.file_name().unwrap().to_string_lossy());
                let run_name = ws_single.file_name().unwrap().to_string_lossy().to_string();
                let out_file = out_dir.join(format!("core_results_{}_pass2.parquet", run_name));
                if out_file.exists() && !cli.overwrite { info!("Skipping"); continue; }

                let dia_data = load_dia_data(ws_single)?;
                let result = diaflux::engine::run_ng(&mut prepared2, &dia_data, &ng_config);
                info!("Pass 2 DIA search: {} PSMs (q<=0.01)", result.n_psm);

                write_core_output(&out_dir, &run_name, &result, "_pass2")?;
            }
        }
    } else {
        // ── Open Search mode ────────────────────────────────────────
        let lib_path = cli.lib.as_ref()
            .ok_or_else(|| anyhow::anyhow!("--lib is required with --open-search (for fragment matching)"))?;
        info!("Open Search mode enabled, loading library: {}", lib_path);
        let lib = diaflux::io::library::load_library(lib_path)?;
        info!("Library loaded: {} precursors", lib.len());

        let os_config = diaflux::open_search::pipeline::OpenSearchRunConfig::default();
        for (i, ws_single) in multi_ws.iter().enumerate() {
            info!("================ Run: {}/{} ================", i + 1, file_num);
            info!("File: {}", ws_single.file_name().unwrap().to_string_lossy());
            let out_file = out_dir.join(format!("os_results_{}.parquet",
                ws_single.file_name().unwrap().to_string_lossy()));
            if out_file.exists() && !cli.overwrite { info!("Skipping"); continue; }
            let ms_data = load_ms_data(ws_single)?;
            info!("IMS: {} | cycles: {} | SWATHs: {}",
                if ms_data.has_ims { "TIMS" } else { "none" },
                ms_data.n_cycles, ms_data.ms2_maps.len());
            let groups = diaflux::open_search::pipeline::search_core_open(&os_config, &ms_data, &lib, &out_dir)?;
            if !groups.is_empty() {
                let run_name = ws_single.file_name().unwrap().to_string_lossy().to_string();
                let _ = diaflux::output::save_as_parquet(&groups, &out_file);
                let csv_path = out_dir.join(format!("{}_report.csv", run_name));
                let _ = diaflux::output::save_as_csv(&groups, &csv_path);
                let rpt_path = out_dir.join(format!("{}_report.parquet", run_name));
                let _ = diaflux::output::convert_to_diann_format(&groups, &run_name, &rpt_path);
            }
            info!("Open Search run completed: {} groups", groups.len());
        }
    }

    info!("DIAFluX processing completed.");
    Ok(())
}

/// Generate the predicted spectral library from `fasta` via the peptdeep Python
/// sidecar, caching the result at `lib_out`. When `model_path` is given (second
/// pass), the sidecar loads the fine-tuned `rt.pth`/`ms2.pth` checkpoints
/// instead of the generic models.
fn predict_library(cli: &Cli, out_dir: &Path, fasta: &str, model_path: Option<&str>, lib_out: &Path) -> anyhow::Result<PathBuf> {
    if lib_out.exists() && !cli.overwrite {
        info!("Predicted library already exists, reusing: {}", lib_out.display());
        return Ok(lib_out.to_path_buf());
    }

    let script_path = out_dir.join("diaflux_peptdeep.py");
    std::fs::write(&script_path, SIDECAR_SCRIPT)
        .with_context(|| format!("Cannot write sidecar script to {}", script_path.display()))?;

    let python = std::env::var("DIAFORGE_PYTHON").unwrap_or_else(|_| cli.python.clone());
    info!("Predicting library from FASTA via peptdeep ({}): {}", python, fasta);

    let mut cmd = Command::new(&python);
    cmd.arg(&script_path)
        .arg("predict")
        .arg("--fasta").arg(fasta)
        .arg("--out").arg(lib_out)
        .arg("--nce").arg(cli.nce.to_string())
        .arg("--instrument").arg(&cli.instrument)
        .arg("--gradient-end-sec").arg(cli.gradient_end_sec.to_string());
    if let Some(mp) = model_path {
        cmd.arg("--model-path").arg(mp);
    }
    if cli.predict_cpu {
        cmd.arg("--gpu").arg("cpu");
    } else {
        cmd.arg("--gpu").arg("cuda").arg("--gpu-id").arg(cli.gpu_id.to_string());
    }
    if cli.max_precursors > 0 {
        cmd.arg("--max-precursors").arg(cli.max_precursors.to_string());
    }

    let status = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to run library prediction: {python} {}", script_path.display()))?;
    if !status.success() {
        bail!("Library prediction exited with status: {status}");
    }
    if !lib_out.exists() {
        bail!("Library prediction did not produce {}", lib_out.display());
    }
    info!("Predicted library written: {}", lib_out.display());
    Ok(lib_out.to_path_buf())
}

/// Fine-tune PeptDeep's RT/MS2 models on the accumulated transfer library via
/// the sidecar `finetune` subcommand, saving `rt.pth`/`ms2.pth` into `model_dir`.
fn run_finetune(cli: &Cli, out_dir: &Path, transfer_dir: &Path, model_dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(model_dir)?;
    let script_path = out_dir.join("diaflux_peptdeep.py");
    std::fs::write(&script_path, SIDECAR_SCRIPT)
        .with_context(|| format!("Cannot write sidecar script to {}", script_path.display()))?;

    let python = std::env::var("DIAFORGE_PYTHON").unwrap_or_else(|_| cli.python.clone());
    info!("Fine-tuning PeptDeep RT/MS2 models ({} epochs, {}):",
        cli.finetune_epochs, python);

    let mut cmd = Command::new(&python);
    cmd.arg(&script_path)
        .arg("finetune")
        .arg("--precursors").arg(transfer_dir.join("transfer_precursors.parquet"))
        .arg("--fragments").arg(transfer_dir.join("transfer_fragments.parquet"))
        .arg("--out-dir").arg(model_dir)
        .arg("--nce").arg(cli.nce.to_string())
        .arg("--instrument").arg(&cli.instrument)
        .arg("--epochs").arg(cli.finetune_epochs.to_string())
        .arg("--batch-size").arg(cli.finetune_batch_size.to_string())
        .arg("--lr").arg(cli.finetune_lr.to_string());
    if cli.predict_cpu {
        cmd.arg("--gpu").arg("cpu");
    } else {
        cmd.arg("--gpu").arg("cuda").arg("--gpu-id").arg(cli.gpu_id.to_string());
    }

    let status = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to run fine-tuning: {python} {}", script_path.display()))?;
    if !status.success() {
        bail!("Fine-tuning exited with status: {status}");
    }
    for f in ["rt.pth", "ms2.pth"] {
        if !model_dir.join(f).exists() {
            bail!("Fine-tuning did not produce {}/{}", model_dir.display(), f);
        }
    }
    info!("Fine-tuned models written to {}", model_dir.display());
    Ok(())
}

fn discover_workspaces(ws: &std::path::Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    let is_mzml = |p: &std::path::Path| -> bool {
        p.extension().map_or(false, |e| {
            let e = e.to_string_lossy().to_lowercase();
            e == "mzml" || e == "mzml.gz"
        })
    };
    if ws.extension().map_or(false, |e| e == "d") || is_mzml(ws) {
        results.push(ws.to_path_buf());
    } else if let Ok(entries) = std::fs::read_dir(ws) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.extension().map_or(false, |e| e == "d") {
                results.push(path);
            } else if path.is_file() && is_mzml(&path) {
                results.push(path.clone());
            }
        }
    }
    results.sort();
    results
}

/// Route to the correct MS data loader based on file extension.
fn load_ms_data(path: &std::path::Path) -> anyhow::Result<diaflux::model::MsData> {
    let is_mzml = |p: &std::path::Path| -> bool {
        p.extension().map_or(false, |e| {
            let e = e.to_string_lossy().to_lowercase();
            e == "mzml" || e == "mzml.gz"
        })
    };
    if is_mzml(path) {
        diaflux::open_search::msdata::mzml::load_ms(path)
    } else {
        diaflux::open_search::msdata::tims::load_ms(path)
    }
}

/// Build the DIA data model from a raw file (mzML or Bruker `.d`).
fn load_dia_data(path: &std::path::Path) -> anyhow::Result<diaflux::engine::DiaData> {
    let is_mzml = path.extension().map_or(false, |e| {
        let e = e.to_string_lossy().to_lowercase();
        e == "mzml" || e == "mzml.gz"
    });
    if is_mzml {
        let mzml = diaflux::io::mzml::read_mzml(path)?;
        diaflux::engine::DiaData::from_mzml(&mzml)
    } else {
        let bruker = diaflux::io::bruker::read_bruker_dir(path)?;
        diaflux::engine::DiaData::from_bruker(&bruker)
    }
}

/// Write the DIA PSM table and per-level quantification matrices as parquet.
/// `suffix` is appended to the run name (e.g. `_pass2` for the second pass).
fn write_core_output(
    out_dir: &std::path::Path,
    run_name: &str,
    result: &diaflux::engine::CoreResult,
    suffix: &str,
) -> anyhow::Result<()> {
    use arrow::array::{Float32Array, UInt32Array, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    // PSM table (matches the `core_results_{run}.parquet` skip-if-exists check).
    let psm_path = out_dir.join(format!("core_results_{}{}.parquet", run_name, suffix));
    let schema = Schema::new(vec![
        Field::new("precursor_idx", DataType::UInt32, false),
        Field::new("qval", DataType::Float32, false),
        Field::new("proba", DataType::Float32, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(UInt32Array::from(result.precursor_idx.clone())),
            Arc::new(Float32Array::from(result.qval.clone())),
            Arc::new(Float32Array::from(result.proba.clone())),
        ],
    )?;
    let file = std::fs::File::create(&psm_path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    // Quantification matrices (single-sample: one intensity column).
    for (level, quant) in [
        ("precursor", result.quant.precursor.as_ref()),
        ("peptide", result.quant.peptide.as_ref()),
        ("protein", result.quant.protein.as_ref()),
    ] {
        let Some(q) = quant else { continue };
        let path = out_dir.join(format!("core_{}_{}.parquet", run_name, level));
        let intensities: Vec<f32> = (0..q.group_ids.len())
            .map(|r| if q.intensity.ncols() > 0 { q.intensity[[r, 0]] } else { 0.0 })
            .collect();
        let schema = Schema::new(vec![
            Field::new("group_id", DataType::UInt64, false),
            Field::new("intensity", DataType::Float32, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt64Array::from(q.group_ids.clone())),
                Arc::new(Float32Array::from(intensities)),
            ],
        )?;
        let file = std::fs::File::create(&path)?;
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
        writer.write(&batch)?;
        writer.close()?;
    }

    Ok(())
}
