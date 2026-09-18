# DIAFluX

DIAFluX is an open-source DIA (data-independent acquisition) proteomics data
analysis pipeline, written in pure Rust.

## Search modes

- **DIA search mode (default)** — selection → scoring → FDR → directLFQ
  quantification (peptide-centric, library-based DIA search).
- **Open search mode (`--open-search`)** — fragment-centric open search with a
  wide mass tolerance.

## Supported data

- Bruker `.d` (TIMS)
- mzML

## Build

```bash
cargo build --release
```

## Usage

```bash
# DIA search mode (default)
./target/release/diaflux -w <run.d|run.mzML> -l <library.parquet>

# Open search mode
./target/release/diaflux -w <run.d|run.mzML> -l <library.parquet> --open-search
```

## License

Apache-2.0
