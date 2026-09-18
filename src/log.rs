//! Simple logging subsystem — replaces `full_dia/log.py`
//!
//! Provides timed logging to both file and stdout.
//! Uses `tracing` with a layered subscriber (file + stdout).

use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tracing_subscriber::layer::SubscriberExt;

/// Global start time for elapsed-time formatting.
static START_TIME: OnceLock<Instant> = OnceLock::new();

/// Initialise the global logger (file + stdout).
///
/// Format: `HH:MM:SS: message`
pub fn init_logger<P: AsRef<Path>>(dir_out: P) -> Result<(), anyhow::Error> {
    START_TIME.get_or_init(Instant::now);

    use std::fs::{self, File};
    let out_dir = dir_out.as_ref();
    fs::create_dir_all(out_dir)?;

    let log_path = out_dir.join("report.log.txt");
    let file = File::create(&log_path)?;
    let file_writer = Arc::new(Mutex::new(file));

    // Custom timer: elapsed time
    let timer = ElapsedTimer;
    let timer_clone = ElapsedTimer;

    // File layer
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(LogWriter {
            inner: file_writer.clone(),
        })
        .with_timer(timer)
        .with_level(false)
        .with_target(false)
        .compact();

    // Stdout layer
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(io::stdout)
        .with_timer(timer_clone)
        .with_level(false)
        .with_target(false)
        .compact();

    // Combine layers with registry
    let subscriber = tracing_subscriber::registry()
        .with(file_layer)
        .with(stdout_layer)
        .with(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::INFO.into())
                .from_env_lossy(),
        );

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow::anyhow!("Failed to set logger: {}", e))?;

    tracing::info!("Log file: {}", log_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Custom timer: elapsed time since start → "HH:MM:SS"
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ElapsedTimer;

impl tracing_subscriber::fmt::time::FormatTime for ElapsedTimer {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let start = START_TIME.get().copied().unwrap_or(Instant::now());
        let elapsed = start.elapsed();
        let secs = elapsed.as_secs();
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        write!(w, "{h:02}:{m:02}:{s:02}")
    }
}

// ---------------------------------------------------------------------------
// LogWriter: Arc<Mutex<File>> that implements MakeWriter
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LogWriter {
    inner: Arc<Mutex<std::fs::File>>,
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogWriter {
    type Writer = LogFileGuard;

    fn make_writer(&'a self) -> Self::Writer {
        LogFileGuard {
            file: self.inner.clone(),
        }
    }
}

struct LogFileGuard {
    file: Arc<Mutex<std::fs::File>>,
}

impl io::Write for LogFileGuard {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.lock().unwrap().flush()
    }
}

// ---------------------------------------------------------------------------
// Module-level convenience re-exports
// ---------------------------------------------------------------------------

pub use tracing::{debug, error, info, warn};
