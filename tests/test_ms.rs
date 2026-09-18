//! Test MS data loading from .d file — with performance debugging
use std::path::Path;
use std::time::Instant;

#[test]
fn test_load_ms_data() {
    let d_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/test_data/100424_TA-C-004_Slot2-40_1_2472_copy.d"
    ));
    if !d_path.join("analysis.tdf").exists() {
        eprintln!("Test .d file not found, skipping");
        return;
    }

    let t0 = Instant::now();
    let ms = diaflux::open_search::msdata::tims::load_ms(d_path)
        .expect("Should load MS data");
    let t_load = t0.elapsed();
    eprintln!("Loading completed in {:.1}s", t_load.as_secs_f32());

    println!("=== MS Data Debug ===");
    println!("scan_rts: {} elements", ms.scan_rts.len());
    if !ms.scan_rts.is_empty() {
        println!("  first={:.1}, last={:.1}",
            ms.scan_rts[0], ms.scan_rts.last().unwrap());
        println!("  last * 0.0667 = {:.2}",
            ms.scan_rts.last().unwrap() * 0.0667);
    }
    println!("n_cycles: {}", ms.n_cycles);
    println!("cycle_time: {:.3}s", ms.cycle_time);
    println!("swath_boundaries: {:?}", ms.swath_boundaries);
    println!("ms1_maps keys: {:?}", ms.ms1_maps.keys().collect::<Vec<_>>());
    println!("ms2_maps keys: {:?}", ms.ms2_maps.keys().collect::<Vec<_>>());

    let mut total_ms1_peaks = 0usize;
    for (k, v) in &ms.ms1_maps {
        total_ms1_peaks += v.mz_values.len();
        println!("  MS1 swath {}: {} cycles, {} peaks total",
            k, v.scan_rts.len(), v.mz_values.len());
        if !v.scan_rts.is_empty() && v.mz_values.len() > 0 {
            println!("    first cycle peaks: {:?}",
                &v.cycle_valid_lens[..3.min(v.cycle_valid_lens.len())]);
        }
    }
    println!("Total MS1 peaks: {}", total_ms1_peaks);

    let mut total_ms2_peaks = 0usize;
    for (k, v) in &ms.ms2_maps {
        total_ms2_peaks += v.mz_values.len();
        println!("  MS2 swath {}: {} cycles, {} peaks total",
            k, v.scan_rts.len(), v.mz_values.len());
        if !v.scan_rts.is_empty() && v.mz_values.len() > 0 {
            println!("    first cycle peaks: {:?}",
                &v.cycle_valid_lens[..3.min(v.cycle_valid_lens.len())]);
        }
    }
    println!("Total MS2 peaks: {}", total_ms2_peaks);

    assert!(!ms.scan_rts.is_empty(), "scan_rts should not be empty");
    assert!(ms.n_cycles > 0, "should have cycles");
    assert!(total_ms1_peaks > 0, "should have MS1 peaks");
    assert!(total_ms2_peaks > 0, "should have MS2 peaks");
}
