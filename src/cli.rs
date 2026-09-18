//! Command-line interface (clap).

use clap::Parser;

use crate::log::info;

#[derive(Parser, Debug)]
#[command(name = "diaflux", about = "DIAFluX: open-source DIA proteomics data analysis", version)]
pub struct Cli {
    #[arg(short = 'w', long = "ws", required = true)]
    pub ws: String,
    #[arg(short = 'l', long = "lib", conflicts_with = "fasta")]
    pub lib: Option<String>,
    #[arg(long = "fasta", conflicts_with = "lib")]
    pub fasta: Option<String>,
    #[arg(long = "python", default_value = "python3")]
    pub python: String,
    #[arg(long = "lib_out")]
    pub lib_out: Option<String>,
    #[arg(long = "nce", default_value = "25.0")]
    pub nce: f32,
    #[arg(long = "instrument", default_value = "Lumos")]
    pub instrument: String,
    #[arg(long = "predict-cpu")]
    pub predict_cpu: bool,
    #[arg(long = "max-precursors", default_value = "0")]
    pub max_precursors: usize,
    #[arg(long = "gradient-end-sec", default_value = "1253.0")]
    pub gradient_end_sec: f32,
    #[arg(long = "out_name", default_value = "diaflux")]
    pub out_name: String,
    #[arg(long = "gpu_id", default_value = "0")]
    pub gpu_id: i64,
    #[arg(long = "low_memory")]
    pub low_memory: bool,
    #[arg(long = "overwrite")]
    pub overwrite: bool,
    #[arg(long = "open-search")]
    pub open_search: bool,

    // ── Second-pass (transfer-learning library refinement) ────────────────
    /// Run the two-pass search: pass 1 = predicted-library search + transfer
    /// requantification, then PeptDeep fine-tuning, then pass 2 = re-search.
    #[arg(long = "two-pass")]
    pub two_pass: bool,
    /// Output directory for the transfer (requantified) library. Defaults to
    /// `<out_name>/transfer`.
    #[arg(long = "transfer-out")]
    pub transfer_out: Option<String>,
    /// PeptDeep fine-tuning: number of epochs (RT/CCS and MS2).
    #[arg(long = "finetune-epochs", default_value = "40")]
    pub finetune_epochs: u32,
    /// PeptDeep fine-tuning: batch size.
    #[arg(long = "finetune-batch-size", default_value = "2000")]
    pub finetune_batch_size: u32,
    /// PeptDeep fine-tuning: maximum learning rate.
    #[arg(long = "finetune-lr", default_value = "0.0001")]
    pub finetune_lr: f64,
    /// Override the second-pass fragment m/z tolerance (ppm). Default: derived
    /// from the first-pass fragment mass errors (reference automatic MS2
    /// optimizer). Set a positive value to force a fixed tolerance.
    #[arg(long = "pass2-ms2-tol")]
    pub pass2_ms2_tol: Option<f32>,
}

/// Log the resolved run configuration.
pub fn print_run_info(cli: &Cli) {
    info!("DIAFluX (v{}) on {} OS", env!("CARGO_PKG_VERSION"), std::env::consts::OS);
    info!("Time: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    info!("WS: {}", cli.ws);
    if let Some(lib) = &cli.lib { info!("LIB: {}", lib); }
    if let Some(fasta) = &cli.fasta {
        info!("FASTA: {}", fasta);
        info!("Library prediction: peptdeep (instrument={}, nce={}, {})",
            cli.instrument, cli.nce,
            if cli.predict_cpu { "cpu" } else { "gpu" });
    }
    info!("Output folder: {}", cli.out_name);
    if cli.low_memory { info!("Low memory mode: ON"); }
    if cli.overwrite { info!("Overwrite mode: ON"); }
}
