//! Test mzML parsing and MsData conversion.

use std::path::Path;

fn mzml_path() -> std::path::PathBuf {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/test_data/RD139_Narrow_UPS1_0_1fmol_inj1.mzML"
    ))
    .to_path_buf()
}

#[test]
fn test_parse_mzml() {
    let path = mzml_path();
    if !path.exists() {
        eprintln!("SKIP: mzML test file not found");
        return;
    }

    let result = diaflux::io::mzml::read_mzml(&path);
    let data = result.expect("mzML parsing failed");

    println!("=== mzML Parse Results ===");
    println!("  Spectra: {}", data.spectra.len());
    println!("  MS1 indices: {}", data.ms1_indices.len());
    println!("  MS2 indices: {}", data.ms2_indices.len());
    println!("  Isolation windows: {} windows", data.isolation_windows.len());
    println!("  First MS1: RT={:.2}s, peaks={}", data.spectra[0].rt, data.spectra[0].mz_array.len());
    println!("  Last MS1: RT={:.2}s", data.spectra[*data.ms1_indices.last().unwrap()].rt);

    // Verify basic properties
    assert!(!data.spectra.is_empty(), "Should have spectra");
    assert!(!data.ms1_indices.is_empty(), "Should have MS1");
    assert!(!data.ms2_indices.is_empty(), "Should have MS2");
    assert!(!data.isolation_windows.is_empty(), "Should have DIA windows");
    assert!(data.isolation_windows.len() >= 10, "Many DIA windows expected");

    // Check isolation windows
    for (i, &(lo, hi)) in data.isolation_windows.iter().enumerate().take(5) {
        assert!(hi > lo, "Window {i}: high > low");
        assert!(lo >= 100.0, "Window {i}: low too small");
    }
    let (first_lo, _) = data.isolation_windows[0];
    assert!(first_lo >= 200.0, "First window should start >= 200 Da");

    // Check RT conversion: minutes → seconds
    let last_sp = &data.spectra[*data.ms1_indices.last().unwrap()];
    assert!(last_sp.rt > 100.0, "RT should be in seconds (>100s), got {:.1}s", last_sp.rt);
    assert!(last_sp.rt < 36000.0, "RT < 10h limit");

    // Check peak arrays consistency
    for (i, sp) in data.spectra.iter().enumerate().take(10) {
        assert_eq!(sp.mz_array.len(), sp.intensity_array.len(),
            "Spectrum {i}: mz/intensity length mismatch");
    }

    println!("  All mzML parse assertions passed!");
}

#[test]
fn test_mzml_to_msdata() {
    let path = mzml_path();
    if !path.exists() {
        eprintln!("SKIP: mzML test file not found");
        return;
    }

    let result = diaflux::open_search::msdata::mzml::load_ms(&path);
    let ms_data = result.expect("MsData conversion failed");

    println!("=== MsData Conversion Results ===");
    println!("  Scan RTs: {} cycles", ms_data.scan_rts.len());
    println!("  Cycle time: {:.3}s", ms_data.cycle_time);
    println!("  N cycles: {}", ms_data.n_cycles);
    println!("  SWATH boundaries: {:?} (n={})",
        &ms_data.swath_boundaries[..5.min(ms_data.swath_boundaries.len())],
        ms_data.swath_boundaries.len());
    println!("  MS1 maps: {}", ms_data.ms1_maps.len());
    println!("  MS2 maps: {}", ms_data.ms2_maps.len());
    println!("  has_ims: {}", ms_data.has_ims);
    println!("  im_gap: {}", ms_data.im_gap);
    println!("  Device: {}", ms_data.device_name);

    // Verify properties
    assert!(!ms_data.scan_rts.is_empty(), "Should have RTs");
    assert!(ms_data.n_cycles > 0, "Should have cycles");
    assert!(!ms_data.has_ims, "mzML should report has_ims=false");
    assert!(!ms_data.ms1_maps.is_empty(), "Should have MS1 maps");
    assert!(!ms_data.ms2_maps.is_empty(), "Should have MS2 maps");
    assert_eq!(ms_data.ms1_maps.len(), ms_data.ms2_maps.len(),
        "MS1 and MS2 maps should have same number of SWATH entries");

    // Check the map data is populated
    for (swath_id, map) in &ms_data.ms2_maps {
        let total_peaks: usize = map.cycle_valid_lens.iter().map(|&v| v as usize).sum();
        assert!(total_peaks > 0, "SWATH {swath_id} should have peaks");
        assert_eq!(map.mz_values.len(), total_peaks,
            "SWATH {swath_id}: mz_values length mismatch");
        assert_eq!(map.im_values.len(), total_peaks,
            "SWATH {swath_id}: im_values length mismatch");
        assert_eq!(map.height_values.len(), total_peaks,
            "SWATH {swath_id}: height_values length mismatch");

        // Check seek indices are valid
        let n_cycles = map.seek_indices.len().saturating_sub(1);
        assert!(n_cycles > 0, "SWATH {swath_id}: seek_indices should have entries");
        assert_eq!(map.seek_indices[n_cycles] as usize, total_peaks,
            "SWATH {swath_id}: seek_indices last entry should equal total peaks");
    }

    // Check centroided data is populated (for mzML, centroided = profile)
    for (swath_id, map) in &ms_data.ms2_maps {
        assert!(!map.im_values_cen.is_empty(),
            "SWATH {swath_id}: centroided data should be populated");
        assert_eq!(map.mz_values_cen.len(), map.mz_values.len(),
            "SWATH {swath_id}: centroided length should match profile");
    }

    // Check cycle counts consistency
    let first_map = ms_data.ms2_maps.values().next().unwrap();
    assert_eq!(first_map.scan_rts.len(), ms_data.scan_rts.len(),
        "scan_rts length should match across maps");

    println!("  Peak distribution per SWATH (first 5):");
    for (swath_id, map) in ms_data.ms2_maps.iter().take(5) {
        let total: u32 = map.cycle_valid_lens.iter().sum();
        let nonzero = map.cycle_valid_lens.iter().filter(|&&v| v > 0).count();
        println!("    SWATH {swath_id}: {total} peaks, {nonzero} non-empty cycles");
    }

    println!("  All MsData conversion assertions passed!");
}
