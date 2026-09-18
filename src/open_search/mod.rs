//! Open Search — fragment-centric search with wide mass tolerance.
//!
//! Searches spectral libraries against DIA data using XIC co-elution scoring,
//! with ±500 Da precursor mass tolerance for modification discovery.

pub mod coelution;
pub mod fdr_open;
pub mod frag_index;
pub mod msdata;
pub mod pipeline;
pub mod xic;

pub use fdr_open::*;
pub use frag_index::*;
pub use pipeline::*;
