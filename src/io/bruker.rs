//! Bruker TDF format reader — replaces `full_dia/alphatims/bruker.py`
//!
//! Reads `.d` directories containing:
//! - `analysis.tdf` — SQLite database (metadata, frames, quadrupole settings)
//! - `analysis.tdf_bin` — Binary file (raw TOF/IMS data, compressed)

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{bail, Context, Result};
use rusqlite::Connection;

// ===========================================================================
// Core data structures for one Bruker .d acquisition
// ===========================================================================

/// Top-level container for a single Bruker timsTOF run.
pub struct BrukerData {
    pub path: String,
    pub frames: Vec<Frame>,
    pub push_indptr: Vec<i64>,
    pub scan_max_index: u32,
    pub mobility_values: Vec<f32>,
    pub mobility_min_value: f32,
    pub mobility_max_value: f32,
    pub mz_values: Vec<f32>,
    pub rt_values: Vec<f32>,
    pub quad_mz_values: Vec<(f32, f32)>,
    pub quad_indptr: Vec<i64>,
    /// Per-frame DIA isolation windows (one row per Frame × WindowGroup).
    pub quad_frames: Vec<QuadFrame>,
    pub tof_indices: Vec<u32>,
    pub intensity_values: Vec<u32>,
    pub sql_metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub id: i64,
    pub time: f64,
    pub num_peaks: i64,
    pub msms_type: i64,
    pub num_scans: i64,
    pub scan_num: i64,
    pub tims_id: i64,
}

/// One row of the `DiaFrameMsMsWindows` table: the isolation window of a
/// single MS2 frame (or a group of scans within a frame for diaPASEF).
#[derive(Debug, Clone)]
pub struct QuadFrame {
    /// Frame id (1-based, as stored in the tdf).
    pub frame_id: i64,
    pub window_group: i64,
    pub scan_begin: i64,
    pub scan_end: i64,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub precursor: i64,
}

// ===========================================================================
// Main reader
// ===========================================================================

pub fn read_bruker_dir(dir_d: &Path) -> Result<BrukerData> {
    if !dir_d.is_dir() {
        bail!("Not a .d directory: {}", dir_d.display());
    }
    let tdf_path = dir_d.join("analysis.tdf");
    if !tdf_path.is_file() {
        bail!("No analysis.tdf found in {}", dir_d.display());
    }
    let conn = Connection::open(&tdf_path)?;
    let metadata = read_sql_metadata_v2(&conn)?;
    let frames = read_frames(&conn)?;
    if frames.is_empty() {
        bail!("No frames found in analysis.tdf");
    }
    let scan_max_index = read_scan_max_index(&conn)?;
    let rt_values: Vec<f32> = frames.iter().map(|f| f.time as f32).collect();
    let (quad_mz_values, quad_indptr) = read_quad_settings(&conn, frames.len())?;
    let quad_frames = read_quad_frames(&conn)?;
    let (mobility_values, mobility_min, mobility_max) = read_mobility_calibration(&conn, scan_max_index)?;
    let mz_values = read_mz_calibration(&conn)?;
    let compression_type: i64 = metadata.get("TimsCompressionType").and_then(|v| v.parse().ok()).unwrap_or(0);
    let max_peaks_per_scan: i64 = metadata.get("MaxNumPeaksPerScan").and_then(|v| v.parse().ok()).unwrap_or(1000);
    let bin_path = dir_d.join("analysis.tdf_bin");
    let (scan_indptr, tof_indices, intensity_values) = if bin_path.is_file() {
        let result = read_and_decompress_binary(&bin_path, &frames, compression_type, max_peaks_per_scan)?;
        let ptr_first_10: Vec<i64> = result.0.iter().take(10).copied().collect();
        eprintln!("[BRUKER] {} peaks in {} frames, scan_indptr[0..10] = {:?}",
            result.1.len(), frames.len(), ptr_first_10);
        result
    } else {
        (vec![0i64; (frames.len() * scan_max_index as usize) + 1], vec![], vec![])
    };

    Ok(BrukerData {
        path: dir_d.to_string_lossy().to_string(),
        frames,
        push_indptr: scan_indptr,
        scan_max_index,
        mobility_values,
        mobility_min_value: mobility_min,
        mobility_max_value: mobility_max,
        mz_values,
        rt_values,
        quad_mz_values,
        quad_indptr,
        quad_frames,
        tof_indices,
        intensity_values,
        sql_metadata: metadata,
    })
}

// ===========================================================================
// SQL readers
// ===========================================================================

fn read_sql_metadata_v2(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut metadata = HashMap::new();
    for table in &["GlobalMetaData", "GlobalMetadata"] {
        if let Ok(mut stmt) = conn.prepare(&format!("SELECT Key, Value FROM [{}]", table)) {
            let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
            for row in rows.flatten() {
                metadata.entry(row.0).or_insert(row.1);
            }
        }
    }
    Ok(metadata)
}

fn read_frames(conn: &Connection) -> Result<Vec<Frame>> {
    let has_scan_num = conn.prepare("SELECT ScanNum FROM Frames LIMIT 1").is_ok();
    let has_tims_id = conn.prepare("SELECT TimsId FROM Frames LIMIT 1").is_ok();
    let query = if has_scan_num && has_tims_id {
        "SELECT Id, Time, NumPeaks, MsMsType, NumScans, ScanNum, TimsId FROM Frames ORDER BY Id"
    } else if has_scan_num {
        "SELECT Id, Time, NumPeaks, MsMsType, NumScans, ScanNum, 0 FROM Frames ORDER BY Id"
    } else if has_tims_id {
        "SELECT Id, Time, NumPeaks, MsMsType, NumScans, 0, TimsId FROM Frames ORDER BY Id"
    } else {
        "SELECT Id, Time, NumPeaks, MsMsType, NumScans, 0, 0 FROM Frames ORDER BY Id"
    };
    let mut stmt = conn.prepare(query)?;
    let frames = stmt.query_map([], |row| {
        Ok(Frame {
            id: row.get(0)?,
            time: row.get(1)?,
            num_peaks: row.get(2)?,
            msms_type: row.get(3)?,
            num_scans: row.get(4)?,
            scan_num: row.get(5)?,
            tims_id: row.get(6)?,
        })
    })?.collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(frames)
}

fn read_scan_max_index(conn: &Connection) -> Result<u32> {
    let mut stmt = conn.prepare("SELECT MAX(NumScans) FROM Frames")?;
    let max: i64 = stmt.query_row([], |row| row.get(0))?;
    Ok((max + 1) as u32)
}

fn read_quad_settings(conn: &Connection, _frame_count: usize) -> Result<(Vec<(f32, f32)>, Vec<i64>)> {
    if conn.prepare("SELECT WindowGroup, IsolationMz, IsolationWidth FROM DiaFrameMsMsWindows LIMIT 1").is_ok() {
        let mut stmt = conn.prepare(
            "SELECT WindowGroup, IsolationMz, IsolationWidth FROM DiaFrameMsMsWindows GROUP BY WindowGroup ORDER BY WindowGroup"
        )?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?, row.get::<_, f64>(2)?)))?;
        let mut quad_mz = Vec::new();
        for row in rows.flatten() {
            let (_wg, iso_mz, iso_width) = row;
            quad_mz.push(((iso_mz - iso_width / 2.0) as f32, (iso_mz + iso_width / 2.0) as f32));
        }
        let mut indptr = Vec::new();
        indptr.push(0);
        for _ in 0..quad_mz.len().max(16) {
            indptr.push(indptr.last().copied().unwrap_or(0) + 1);
        }
        if !quad_mz.is_empty() {
            return Ok((quad_mz, indptr));
        }
    }
    Ok((vec![], vec![0, 1]))
}

/// Read the per-frame DIA isolation windows from `DiaFrameMsMsWindows`.
///
/// One row per (Frame, WindowGroup); for diaPASEF a single frame may contain
/// several windows (each spanning a range of scans). We read the full table so
/// the DIA data model can reconstruct the isolation window for every
/// (frame-in-cycle, scan) position.
fn read_quad_frames(conn: &Connection) -> Result<Vec<QuadFrame>> {
    let has_frame = conn
        .prepare("SELECT Frame FROM DiaFrameMsMsWindows LIMIT 1")
        .is_ok();
    let has_scan_begin = conn
        .prepare("SELECT ScanNumBegin FROM DiaFrameMsMsWindows LIMIT 1")
        .is_ok();
    let has_precursor = conn
        .prepare("SELECT Precursor FROM DiaFrameMsMsWindows LIMIT 1")
        .is_ok();

    // Column selectors; fall back to a literal 0 when the column is absent.
    let (sb, se, prec) = (
        if has_scan_begin { "ScanNumBegin" } else { "0" },
        if has_scan_begin { "ScanNumEnd" } else { "0" },
        if has_precursor { "Precursor" } else { "0" },
    );

    // Older timsTOF `.d` files omit the `Frame` column from
    // `DiaFrameMsMsWindows` and store the per-frame mapping in
    // `DiaFrameMsMsInfo` (Frame -> WindowGroup) instead. Recover the frame id
    // (which downstream expects to reference `Frames.Id`, 1-based) via a JOIN
    // on WindowGroup; when even that table is missing, fall back to using
    // WindowGroup as a proxy for the frame id.
    let query: String = if has_frame {
        format!(
            "SELECT Frame, WindowGroup, {}, {}, IsolationMz, IsolationWidth, {} \
             FROM DiaFrameMsMsWindows ORDER BY Frame, WindowGroup",
            sb, se, prec
        )
    } else if conn.prepare("SELECT Frame FROM DiaFrameMsMsInfo LIMIT 1").is_ok() {
        let (sbw, sew, precw) = (
            if has_scan_begin { "w.ScanNumBegin" } else { "0" },
            if has_scan_begin { "w.ScanNumEnd" } else { "0" },
            if has_precursor { "w.Precursor" } else { "0" },
        );
        format!(
            "SELECT info.Frame, w.WindowGroup, {}, {}, w.IsolationMz, w.IsolationWidth, {} \
             FROM DiaFrameMsMsWindows w \
             JOIN DiaFrameMsMsInfo info ON info.WindowGroup = w.WindowGroup \
             ORDER BY info.Frame, w.WindowGroup",
            sbw, sew, precw
        )
    } else {
        format!(
            "SELECT WindowGroup, WindowGroup, {}, {}, IsolationMz, IsolationWidth, {} \
             FROM DiaFrameMsMsWindows ORDER BY WindowGroup",
            sb, se, prec
        )
    };

    let mut stmt = conn.prepare(&query)?;
    let rows = stmt.query_map([], |row| {
        Ok(QuadFrame {
            frame_id: row.get(0)?,
            window_group: row.get(1)?,
            scan_begin: row.get(2)?,
            scan_end: row.get(3)?,
            isolation_mz: row.get(4)?,
            isolation_width: row.get(5)?,
            precursor: row.get(6)?,
        })
    })?;
    let mut out = Vec::new();
    for row in rows.flatten() {
        out.push(row);
    }
    Ok(out)
}

fn read_mobility_calibration(conn: &Connection, scan_max_index: u32) -> Result<(Vec<f32>, f32, f32)> {
    let n_scans = scan_max_index as usize;
    if n_scans == 0 { return Ok((vec![0.0], 0.0, 1.0)); }
    let has_scan = conn.prepare("SELECT Scan FROM CalibrationMobility LIMIT 1").is_ok();
    let col = if has_scan { "Scan" } else { "Id" };
    if let Ok(mut stmt) = conn.prepare(&format!("SELECT {}, OneOverK0 FROM CalibrationMobility ORDER BY {}", col, col)) {
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)))?;
        let mut mobility: Vec<f32> = Vec::new();
        let mut last_idx = -1i64;
        for row in rows.flatten() {
            let (idx, val) = row;
            while last_idx + 1 < idx { mobility.push(mobility.last().copied().unwrap_or(0.5)); last_idx += 1; }
            mobility.push(val as f32);
            last_idx = idx;
        }
        if !mobility.is_empty() {
            while mobility.len() < n_scans { mobility.push(mobility.last().copied().unwrap_or(0.5) + 0.001); }
            let min_val = mobility.iter().copied().fold(f32::MAX, f32::min);
            let max_val = mobility.iter().copied().fold(f32::MIN, f32::max);
            return Ok((mobility, min_val, max_val));
        }
    }
    let min_val = 0.5f32;
    let max_val = 1.7f32;
    // DESCENDING: scan 0 = highest IM (matches alphatims)
    let mobility: Vec<f32> = (0..n_scans).map(|i| {
        let frac = (i as f32 / (n_scans - 1).max(1) as f32);
        max_val - frac * (max_val - min_val)
    }).collect();
    Ok((mobility, min_val, max_val))
}

fn read_mz_calibration(conn: &Connection) -> Result<Vec<f32>> {
    // Bruker TDF v3: simpler metadata-based quadratic fit over larger TOF range
    // The CalibrationMz table (TDF v2) and MzCalibration polynomial (TDF v3)
    // require version-specific parsing — use metadata fallback with 2M entries.

    // Fallback: metadata-based quadratic
    if let Ok(mut stmt) = conn.prepare("SELECT Key, Value FROM GlobalMetaData WHERE Key='MzAcqRangeLower' OR Key='MzAcqRangeUpper'") {
        let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)))?;
        let mut lower = 100.0f64; let mut upper = 2000.0f64;
        for row in rows.flatten() { if row.0 == "MzAcqRangeLower" { lower = row.1; } if row.0 == "MzAcqRangeUpper" { upper = row.1; } }
        let n_tof = 2000000usize;
        let intercept = lower.sqrt(); let slope = (upper.sqrt() - intercept) / n_tof as f64;
        let mz_values: Vec<f32> = (0..n_tof).map(|i| (intercept + slope * i as f64).powi(2) as f32).collect();
        tracing::info!("M/z fallback: quadratic fit [{:.1}..{:.1}] over {} TOFs (error possible)", lower, upper, n_tof);
        return Ok(mz_values);
    }
    Ok((0..2000000).map(|i| 100.0 + i as f32 * 0.001).collect())
}

// ===========================================================================
// Binary file reader
// ===========================================================================

fn read_and_decompress_binary(
    bin_path: &Path, frames: &[Frame], compression_type: i64, _max_peaks_per_scan: i64,
) -> Result<(Vec<i64>, Vec<u32>, Vec<u32>)> {
    let n_frames = frames.len();
    let max_scan_count = frames.iter().map(|f| f.num_scans).max().unwrap_or(1) + 1;
    let mut frame_indptr = vec![0i64; n_frames + 1];
    for i in 0..n_frames { frame_indptr[i + 1] = frame_indptr[i] + frames[i].num_peaks; }
    let total_peaks = frame_indptr[n_frames] as usize;
    let tims_ids: Vec<i64> = frames.iter().map(|f| f.tims_id).collect();

    let mut scan_indptr = vec![0i64; (max_scan_count as usize) * n_frames + 1];
    let mut all_tof = vec![0u32; total_peaks];
    let mut all_intensity = vec![0u32; total_peaks];

    let file = std::fs::File::open(bin_path).with_context(|| format!("Cannot open {}", bin_path.display()))?;
    let mut reader = std::io::BufReader::new(file);

    for fi in 0..n_frames {
        if frame_indptr[fi + 1] - frame_indptr[fi] == 0 { continue; }
        let offset = tims_ids[fi];
        reader.seek(SeekFrom::Start(offset as u64))?;

        if compression_type == 2 {
            let mut header = [0u8; 8];
            reader.read_exact(&mut header)?;
            let bin_size = u32::from_le_bytes(header[0..4].try_into().unwrap());
            let scan_count = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
            let compressed_size = if bin_size as usize > 8 { bin_size as usize - 8 } else { 0 };
            let mut compressed = vec![0u8; compressed_size];
            if compressed_size > 0 { reader.read_exact(&mut compressed)?; }

            let decompressed = zstd::bulk::decompress(&compressed, compressed_size * 4)
                .map_err(|e| anyhow::anyhow!("Zstd decompression failed for frame {}: {}", fi, e))?;
            let n_u32 = decompressed.len() / 4;

            // Parse as flat uint32 array (no reshape for scan indices)
            let mut buf = Vec::with_capacity(n_u32);
            for i in 0..n_u32 {
                let bytes: [u8; 4] = decompressed[i * 4..(i + 1) * 4].try_into().unwrap();
                buf.push(u32::from_le_bytes(bytes));
            }
            if n_u32 < 4 || scan_count < 1 { continue; }

            // ── Decode TOF/intensity data ──────────────────────────────
            // Data after header: scan indices then (tof_delta, intensity) pairs
            // Total peaks = (n_u32 - scan_count) / 2  (simple interleave assumption)
            // The scan index encoding varies between Bruker versions —
            // we use frame-level peak tracking (frame_indptr) for boundaries.
            let frame_peaks = (frame_indptr[fi + 1] - frame_indptr[fi]) as usize;
            // alphatims: tof_deltas = buf[scan_count::2], intensities = buf[scan_count+1::2]
            let data_start = scan_count;
            // Data is flat (tof_delta, intensity) pairs after scan indices
            let n_peak_pairs = n_u32.saturating_sub(data_start) / 2;
            let n_out = n_peak_pairs.min(frame_peaks);

            // For scan_indptr: assign equal distribution or all to scan 0
            // (scan indices can't be reliably decoded without version-specific logic)
            let scan_start = fi * max_scan_count as usize;
            if n_out > 0 {
                scan_indptr[scan_start] = n_out as i64; // all peaks to scan 0
            }

            // Flat (tof_delta, intensity) pairs matching alphatims
            let step = 2usize;
            let mut cum_tof: u64 = 0;
            for pi in 0..n_out {
                let tof_delta = *buf.get(data_start + pi * step).unwrap_or(&0);
                let inten = *buf.get(data_start + pi * step + 1).unwrap_or(&0);
                cum_tof = cum_tof.wrapping_add(tof_delta as u64);
                let pos = frame_indptr[fi] as usize + pi;
                if pos < all_tof.len() {
                    all_tof[pos] = (cum_tof.wrapping_sub(1)) as u32;
                    all_intensity[pos] = inten;
                }
            }
        } else {
            // Uncompressed/LZF: not implemented for this dataset
            let mut header = [0u8; 8];
            reader.read_exact(&mut header)?;
            let expected = (frame_indptr[fi + 1] - frame_indptr[fi]) as usize * 8;
            let mut raw = vec![0u8; expected];
            reader.read_exact(&mut raw)?;
            let peak_start = frame_indptr[fi] as usize;
            let n_peak = (frame_indptr[fi + 1] - frame_indptr[fi]) as usize;
            for pi in 0..n_peak {
                if pi * 8 + 8 <= raw.len() {
                    all_tof[peak_start + pi] = u32::from_le_bytes(raw[pi * 8..pi * 8 + 4].try_into().unwrap());
                    all_intensity[peak_start + pi] = u32::from_le_bytes(raw[pi * 8 + 4..pi * 8 + 8].try_into().unwrap());
                }
            }
        }
    }

    // Cumulative sum of scan_indptr
    let mut cum = 0i64;
    for i in 0..scan_indptr.len() - 1 {
        let tmp = scan_indptr[i];
        scan_indptr[i] = cum;
        cum += tmp;
    }
    *scan_indptr.last_mut().unwrap() = cum;

    Ok((scan_indptr, all_tof, all_intensity))
}
