//! DIA-NN amino-acid chemistry and fragment-ion types used by decoy generation.
//!
//! Trimmed from the full DIA-NN data-type port; only the symbols `decoy.rs`
//! depends on are kept (AA mutation tables, masses, and the `Ion`/`Product`
//! fragment-ion structs).

// ============================================================================
// Fragment type constants
// ============================================================================

pub const F_TYPE_B: u8 = 1;
pub const F_TYPE_Y: u8 = 2;

// ============================================================================
// Neutral loss constants
// ============================================================================

pub const LOSS_NONE: u8 = 0;

/// Loss masses: [loss_none, loss_H2O, loss_NH3, loss_CO, loss_N, loss_other]
pub const LOSS_MASSES: [f64; 6] = [0.0, 18.011113035, 17.026548, 27.994915, 0.0, 0.0];

// ============================================================================
// Product — fragment ion defined in the spectral library
// ============================================================================

/// A fragment ion entry in the spectral library.
#[derive(Debug, Clone, Copy, Default)]
pub struct Product {
    pub mz: f32,
    pub charge: i8,
    pub index: i32,       // product index within the peptide
    pub ion_type: u8,     // 1=B, 2=Y, 3=loss (fTypeB/fTypeY)
    pub loss: u8,         // neutral loss type
    pub fr_type: u8,      // original type from library (includes higher bits)
    pub corr: f32,        // correlation with library spectrum
}

// ============================================================================
// Ion — matched fragment ion
// ============================================================================

/// A matched fragment ion identity.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ion {
    pub ion_type: u8,   // 1=b, 2=y
    pub index: u8,      // number of AAs from cleavage site
    pub loss: u8,       // neutral loss type
    pub charge: i8,
    pub mz: f32,
}

impl PartialEq for Ion {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.ion_type == other.ion_type
            && self.charge == other.charge && self.loss == other.loss
    }
}

impl Ion {
    pub fn new(ion_type: u8, index: u8, loss: u8, charge: i8, mz: f32) -> Self {
        Self { ion_type, index, loss, charge, mz }
    }

    /// Copy identity fields from another Ion-like object (Product or Ion).
    pub fn init_from_product(&mut self, other: &Product) {
        self.ion_type = other.ion_type & 3;
        self.index = other.index as u8;
        self.loss = other.loss;
        self.charge = other.charge;
        self.mz = other.mz;
    }
}

// ============================================================================
// DIA-NN AA masses
// ============================================================================

/// Amino acid mono-isotopic masses used in DIA-NN.
pub fn aa_mass(aa: char) -> f64 {
    match aa.to_ascii_uppercase() {
        'G' => 57.021464,
        'A' => 71.037114,
        'S' => 87.032028,
        'P' => 97.052764,
        'V' => 99.068414,
        'T' => 101.047679,
        'C' => 103.009185,
        'I' => 113.084064,
        'L' => 113.084064,
        'N' => 114.042927,
        'D' => 115.026943,
        'Q' => 128.058578,
        'K' => 128.094963,
        'E' => 129.042593,
        'M' => 131.040485,
        'H' => 137.058912,
        'F' => 147.068414,
        'R' => 156.101111,
        'Y' => 163.063329,
        'W' => 186.079313,
        'm' => 147.035400,  // oxidized M
        'c' => 160.030645,  // carbamidomethyl C
        _ => 0.0,
    }
}

pub const PROTON: f64 = 1.007276466771;
pub const OH: f64 = 17.00733495902; // proton + H2O

/// DIA-NN decoy amino acid mutation table.
pub const AAS_TABLE: &[u8] = b"GAVLIFMPWSCTYHKRQEND";
pub const MUTATE_TO_TABLE: &[u8] = b"LLLVVLLLLTSSSSLLNDQE";
