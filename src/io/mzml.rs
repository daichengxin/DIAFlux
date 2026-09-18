//! mzML format reader — parses any mzML file (Thermo, SCIEX, Agilent, Bruker).
//!
//! Stream-parses the XML with quick-xml, handles:
//! - zlib-compressed binary data arrays
//! - 32-bit and 64-bit floats
//! - DIA isolation window extraction
//! - MS1/MS2 spectrum classification

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::{bail, Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;

// ===========================================================================
// Data structures
// ===========================================================================

/// A single decoded spectrum from an mzML file.
#[derive(Debug, Clone)]
pub struct Spectrum {
    pub ms_level: u8,
    /// Retention time in seconds.
    pub rt: f32,
    /// (low_mz, high_mz) for MS2; (0.0, 0.0) for MS1.
    pub isolation_window: (f32, f32),
    pub mz_array: Vec<f32>,
    pub intensity_array: Vec<f32>,
}

/// Decoded metadata + spectra from an mzML file.
#[derive(Debug, Clone)]
pub struct MzMLData {
    pub path: String,
    pub spectra: Vec<Spectrum>,
    /// Sorted, deduplicated DIA isolation windows from MS2 scans.
    pub isolation_windows: Vec<(f32, f32)>,
    /// Indices into `spectra` for MS1 scans.
    pub ms1_indices: Vec<usize>,
    /// Indices into `spectra` for MS2 scans.
    pub ms2_indices: Vec<usize>,
}

// ===========================================================================
// Helpers: base64 + zlib decode
// ===========================================================================

fn decode_binary_data(text: &str, is_64bit: bool, is_zlib: bool) -> Result<Vec<f32>> {
    let raw = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        text.as_bytes(),
    )
    .map_err(|e| anyhow::anyhow!("Base64 decode error: {}", e))?;

    let bytes = if is_zlib {
        use flate2::read::ZlibDecoder;
        use std::io::Read;
        let mut decoder = ZlibDecoder::new(&raw[..]);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out)
            .map_err(|e| anyhow::anyhow!("Zlib decompress error: {}", e))?;
        out
    } else {
        raw
    };

    if is_64bit {
        let n = bytes.len() / 8;
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * 8;
            let arr: [u8; 8] = bytes[start..start + 8].try_into().unwrap_or([0; 8]);
            let v = f64::from_le_bytes(arr) as f32;
            vals.push(v);
        }
        Ok(vals)
    } else {
        let n = bytes.len() / 4;
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * 4;
            let arr: [u8; 4] = bytes[start..start + 4].try_into().unwrap_or([0; 4]);
            vals.push(f32::from_le_bytes(arr));
        }
        Ok(vals)
    }
}

// ===========================================================================
// Main reader — single-pass streaming XML parser
// ===========================================================================

pub fn read_mzml(path: &Path) -> Result<MzMLData> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open mzML: {}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut xml = Reader::from_reader(reader);
    xml.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut spectra: Vec<Spectrum> = Vec::new();

    // ── Stack-based state machine ──────────────────────────────────
    // depth counters — how many levels deep we are in each context
    let mut depth_spectrum: usize = 0;
    let mut depth_binary_array: usize = 0;

    // Spectrum accumulator
    let mut ms_level: u8 = 0;
    let mut rt_sec: f32 = 0.0;
    let mut isolation_low: f32 = 0.0;
    let mut isolation_high: f32 = 0.0;
    let mut precursor_target: Option<f32> = None;
    let mut precursor_lo: f32 = 0.0;
    let mut precursor_hi: f32 = 0.0;

    // Binary accumulator
    let mut binary_mz: bool = false;
    let mut binary_int: bool = false;
    let mut binary_64bit: bool = false;
    let mut binary_zlib: bool = false;
    let mut binary_text: String = String::new();
    let mut current_mz: Vec<f32> = Vec::new();
    let mut current_int: Vec<f32> = Vec::new();

    loop {
        buf.clear();
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "spectrum" => {
                        depth_spectrum += 1;
                        ms_level = 0; rt_sec = 0.0;
                        isolation_low = 0.0; isolation_high = 0.0;
                        precursor_target = None;
                        precursor_lo = 0.0; precursor_hi = 0.0;
                        current_mz.clear(); current_int.clear();
                    }
                    "binaryDataArray" => {
                        depth_binary_array += 1;
                        binary_mz = false; binary_int = false;
                        binary_64bit = false; binary_zlib = false;
                        binary_text.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                // Self-closing tag: <cvParam .../>
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if tag != "cvParam" { continue; }

                // Collect attributes
                let mut name = String::new();
                let mut value = String::new();
                let mut accession = String::new();
                for a in e.attributes().flatten() {
                    let k = String::from_utf8_lossy(a.key.as_ref()).to_string();
                    let v = String::from_utf8_lossy(&a.value).to_string();
                    if k == "name" { name = v; }
                    else if k == "value" { value = v; }
                    else if k == "accession" { accession = v; }
                }

                if depth_spectrum > 0 && depth_binary_array == 0 {
                    if accession == "MS:1000511" {
                        ms_level = value.parse::<u8>().unwrap_or(1);
                    }
                    // Only parse scan start time if we're in a <scan> context
                    if accession == "MS:1000016" && rt_sec == 0.0 {
                        rt_sec = value.parse::<f32>().unwrap_or(0.0);
                        if rt_sec < 1000.0 { rt_sec *= 60.0; } // minutes → seconds
                    }
                    if accession == "MS:1000827" {
                        precursor_target = value.parse::<f32>().ok();
                    }
                    if accession == "MS:1000828" {
                        precursor_lo = value.parse::<f32>().unwrap_or(0.0);
                    }
                    if accession == "MS:1000829" {
                        precursor_hi = value.parse::<f32>().unwrap_or(0.0);
                    }
                }

                if depth_binary_array > 0 {
                    if accession == "MS:1000514" { binary_mz = true; }
                    if accession == "MS:1000515" { binary_int = true; }
                    if accession == "MS:1000523" { binary_64bit = true; }
                    if accession == "MS:1000521" { binary_64bit = false; }
                    if accession == "MS:1000574" { binary_zlib = true; }
                }
            }
            Ok(Event::Text(ref e)) => {
                if depth_binary_array > 0 {
                    let txt = e.unescape().unwrap_or_default();
                    if !txt.trim().is_empty() {
                        binary_text.push_str(txt.trim());
                    }
                }
            }
            Ok(Event::End(ref e)) => {
                let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag.as_str() {
                    "spectrum" => {
                        depth_spectrum = depth_spectrum.saturating_sub(1);
                        if ms_level > 0 {
                            // Compute isolation window for MS2
                            if ms_level == 2 {
                                if let Some(t) = precursor_target {
                                    if precursor_lo > 0.0 || precursor_hi > 0.0 {
                                        isolation_low = t - precursor_lo;
                                        isolation_high = t + precursor_hi;
                                    }
                                }
                            }
                            // Keep empty spectra too: DIA cycle structure depends
                            // on every frame (MS1 + N MS2) being present in
                            // acquisition order, including scans with zero peaks.
                            // Dropping them shifts `fi % cycle_len` and misaligns
                            // the MS1 frame at cycle position 0.
                            spectra.push(Spectrum {
                                ms_level,
                                rt: rt_sec,
                                isolation_window: (isolation_low, isolation_high),
                                mz_array: std::mem::take(&mut current_mz),
                                intensity_array: std::mem::take(&mut current_int),
                            });
                        }
                    }
                    "binaryDataArray" => {
                        depth_binary_array = depth_binary_array.saturating_sub(1);
                        // Decode the collected binary text
                        if !binary_text.is_empty() {
                            match decode_binary_data(&binary_text, binary_64bit, binary_zlib) {
                                Ok(vals) => {
                                    if binary_mz {
                                        current_mz = vals;
                                    } else if binary_int {
                                        current_int = vals;
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("Binary decode error: {}", e);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => bail!("XML parse error: {}", e),
            _ => {}
        }
    }

    if spectra.is_empty() {
        bail!("No spectra with binary data found in mzML: {}", path.display());
    }

    // Sort by RT
    spectra.sort_by(|a, b| a.rt.partial_cmp(&b.rt).unwrap_or(std::cmp::Ordering::Equal));

    // Build MS1/MS2 indices
    let ms1_indices: Vec<usize> = spectra.iter().enumerate()
        .filter(|(_, s)| s.ms_level == 1)
        .map(|(i, _)| i)
        .collect();
    let ms2_indices: Vec<usize> = spectra.iter().enumerate()
        .filter(|(_, s)| s.ms_level == 2)
        .map(|(i, _)| i)
        .collect();

    // Extract DIA isolation windows
    let mut windows: Vec<(f32, f32)> = spectra.iter()
        .filter(|s| s.ms_level == 2 && s.isolation_window.1 > s.isolation_window.0)
        .map(|s| s.isolation_window)
        .collect();
    windows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    windows.dedup();

    // Build swath boundaries from windows
    let boundaries = if windows.is_empty() && !ms2_indices.is_empty() {
        // No explicit windows — try to infer from the MS2 scans
        // This happens for some non-DIA formats
        tracing::warn!("No DIA isolation windows found in mzML — MS2 may not be DIA");
        windows
    } else {
        windows
    };

    let path_str = path.to_string_lossy().to_string();

    tracing::info!(
        "mzML: {} spectra (MS1:{}, MS2:{}), {} DIA windows, RT [{:.1}s..{:.1}s]",
        spectra.len(), ms1_indices.len(), ms2_indices.len(),
        boundaries.len(),
        spectra.first().map(|s| s.rt).unwrap_or(0.0),
        spectra.last().map(|s| s.rt).unwrap_or(0.0),
    );

    Ok(MzMLData {
        path: path_str,
        spectra,
        isolation_windows: boundaries,
        ms1_indices,
        ms2_indices,
    })
}
