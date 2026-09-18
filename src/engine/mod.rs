//! DIA search engine — the peak-group ("NG") pipeline.
//!
//!   - `data`         — transposed `DiaData` model + `SpecLibFlat`
//!   - `speclib`      — spectral-library adapters
//!   - `libtransform` — library isotope/cardinality transform
//!   - `calibration`  — LOESS property calibration (two-stage search)
//!   - `fdr`          — feed-forward-NN target-decoy q-values
//!   - `selection`    — peak-group candidate selection
//!   - `scoring`      — 40-feature peak-group scoring
//!   - `util`         — IDF / m/z index / correlation helpers
//!   - `ng`           — two-stage search pipeline (`run_ng`)
//!   - `transfer_quant` — second-pass full-fragment requantification
//!   - `transfer_library` — aggregate transfer spectra + MS2 QC

pub mod calibration;
pub mod data;
pub mod fdr;
pub mod libtransform;
pub mod ng;
pub mod result;
pub mod selection;
pub mod scoring;
pub mod speclib;
pub mod transfer_library;
pub mod transfer_quant;
pub mod util;

pub use data::{
    mass_range, transpose, DiaData, FragmentDf, PrecursorDf, SpecLibFlat,
};
pub use ng::{prepare_ng_library, run_ng, NgConfig, PreparedNgLibrary};
pub use result::{CoreResult, QuantLevel, QuantOutput};
