#!/usr/bin/env python3
"""DIAFluX PeptDeep sidecar.

Two subcommands:

  predict   Digest a FASTA and predict a spectral library (RT / ion mobility /
            MS2 fragment intensities) using peptdeep, then write it in the
            15-column DIA-NN parquet format that DIAFluX's Rust library loader
            reads (`src/io/library.rs`). Optionally load fine-tuned model
            checkpoints via `--model-path`.

  finetune  Fine-tune peptdeep's RT + MS2 models on the empirical transfer
            library that DIAFluX's second pass produces (the requantified,
            QC'd b/y z1/z2 fragments of the q<=0.01 first-pass PSMs), and save
            `rt.pth` / `ms2.pth` for `predict --model-path` to reload.

This script is embedded into the `diaflux` binary and shelled out to, so it
must stay self-contained (peptdeep + alphabase + pyarrow + torch).

Predict output columns (one row per fragment):
  Precursor.Id, Modified.Sequence, Stripped.Sequence, Precursor.Charge, Decoy,
  RT, IM, Precursor.Mz, Product.Mz, Relative.Intensity, Fragment.Type,
  Fragment.Charge, Fragment.Series.Number, Protein.Ids, Protein.Names.

Usage:
  python diaflux_peptdeep.py predict --fasta <in.fasta> --out <out.parquet> \
      [--nce 25.0] [--instrument Lumos] [--gpu cuda] [--gpu-id 0] [--cpu] \
      [--model-path <dir with rt.pth/ms2.pth>] [--max-precursors N]
  python diaflux_peptdeep.py finetune --precursors <transfer_precursors.parquet> \
      --fragments <transfer_fragments.parquet> --out-dir <dir> \
      [--epochs 40] [--batch-size 2000] [--lr 0.0001]
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.protein.fasta import SpecLibFasta, protease_dict
from peptdeep.pretrained_models import ModelManager
from peptdeep.settings import global_settings

# UniMod ids used in DIA-NN Modified.Sequence notation.
MOD_NAME_TO_UNIMOD = {
    "Carbamidomethyl@C": 4,
    "Oxidation@M": 35,
    "Acetyl@Protein_N-term": 1,
    "Acetyl@N-term": 1,
}

FRAG_MZ_MIN = 200.0
FRAG_MZ_MAX = 2000.0
TOP_K_FRAGMENTS = 12
MIN_FRAGMENT_INTENSITY = 0.01


def build_modified_sequence(sequence: str, mods: str, mod_sites: str) -> str:
    """Convert alphabase `sequence` + `mods` + `mod_sites` to DIA-NN notation.

    DIA-NN writes the residue followed by "(UniMod:N)", e.g. a carbamidomethyl
    cysteine becomes "C(UniMod:4)". Multiple mods are "; "-separated in alphabase
    and position-indexed by `mod_sites` (1-based).
    """
    if not mods or mods.strip() == "":
        return sequence
    mod_names = [m.strip() for m in mods.split(";") if m.strip()]
    sites = [int(s) for s in mod_sites.split(";") if s.strip() != ""]
    pos_mods: dict[int, list[int]] = {}
    for name, site in zip(mod_names, sites):
        uni = MOD_NAME_TO_UNIMOD.get(name)
        if uni is None:
            continue
        pos_mods.setdefault(site - 1, []).append(uni)
    if not pos_mods:
        return sequence
    out = []
    for i, aa in enumerate(sequence):
        out.append(aa)
        for uni in sorted(pos_mods.get(i, [])):
            out.append(f"(UniMod:{uni})")
    return "".join(out)


def build_modified_sequences_vec(seqs, mods, mod_sites):
    """Vector-ish wrapper: build DIA-NN modified sequences for many precursors."""
    return np.array(
        [build_modified_sequence(s, m, st) for s, m, st in zip(seqs, mods, mod_sites)],
        dtype=object,
    )


def top_k_mask(group_ids: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting the top-`k` (by `scores`) rows within each group.

    `group_ids` need not be sorted; the result is a mask in the original order.
    """
    n = len(group_ids)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-scores, group_ids))
    gid = group_ids[order]
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = gid[1:] != gid[:-1]
    starts = np.flatnonzero(is_start)
    ranks = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    keep = np.zeros(n, dtype=bool)
    keep[order] = ranks < k
    return keep


def _set_default_nce_instrument(args) -> None:
    global_settings["model_mgr"]["default_nce"] = args.nce
    global_settings["model_mgr"]["default_instrument"] = args.instrument


def _move_to_device(mgr: ModelManager, args) -> None:
    if args.gpu in ("cuda", "gpu"):
        mgr.ms2_model.set_device("cuda", [args.gpu_id])
        mgr.rt_model.set_device("cuda", [args.gpu_id])
        mgr.ccs_model.set_device("cuda", [args.gpu_id])
        print(f"Pinned models to cuda:{args.gpu_id}", flush=True)
    else:
        print("Running on CPU", flush=True)


# ── predict ────────────────────────────────────────────────────────────────

def cmd_predict(args) -> None:
    ftypes = get_charged_frag_types(["b", "y"], 2)

    # Set NCE / instrument on the model-manager settings BEFORE construction so
    # ModelManager.__init__ picks them up (set_default_nce_instrument reads
    # self.nce / self.instrument, which are assigned from global_settings).
    _set_default_nce_instrument(args)

    # --- digest -----------------------------------------------------------
    print("Digesting FASTA ...", flush=True)
    fasta_lib = SpecLibFasta(
        ftypes,
        protease=protease_dict["trypsin"],
        max_missed_cleavages=args.missed_cleavages,
        peptide_length_min=args.pep_len_min,
        peptide_length_max=args.pep_len_max,
        precursor_charge_min=args.charge_min,
        precursor_charge_max=args.charge_max,
        precursor_mz_min=args.pr_mz_min,
        precursor_mz_max=args.pr_mz_max,
        var_mods=["Oxidation@M"],
        fix_mods=["Carbamidomethyl@C"],
        decoy=None,
    )
    fasta_lib.get_peptides_from_fasta_list([args.fasta])
    fasta_lib.add_modifications()
    fasta_lib.add_charge()
    fasta_lib.hash_precursor_df()
    fasta_lib.calc_precursor_mz()
    precursor_df = fasta_lib.precursor_df
    precursor_df = precursor_df[
        (precursor_df["precursor_mz"] > args.pr_mz_min)
        & (precursor_df["precursor_mz"] < args.pr_mz_max)
    ]
    # Drop non-canonical amino acids.
    forbidden = ["B", "J", "X", "Z"]
    for aa in forbidden:
        precursor_df = precursor_df[~precursor_df["sequence"].str.contains(aa)]
    precursor_df = precursor_df.reset_index(drop=True)
    print(f"Digested {len(precursor_df):,} precursors", flush=True)

    if args.max_precursors and len(precursor_df) > args.max_precursors:
        precursor_df = precursor_df.sample(n=args.max_precursors, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(precursor_df):,} precursors for testing", flush=True)

    # Map protein id onto each precursor for the Protein.Ids column.
    protein_ids = fasta_lib.protein_df["protein_id"].tolist()

    def proteins_of(idx_str: str) -> str:
        try:
            idxs = [int(i) for i in idx_str.split(";")]
        except ValueError:
            return ""
        return ";".join(protein_ids[i] for i in idxs)

    precursor_df["proteins"] = precursor_df["protein_idxes"].map(proteins_of)

    # --- predict ----------------------------------------------------------
    print("Loading peptdeep models ...", flush=True)
    # Construct on CPU to avoid the DataParallel wrapper (see the comment in the
    # reference script), then move the raw modules onto cuda:<gpu_id> directly.
    mgr = ModelManager(device="cpu")
    mgr.load_installed_models("generic")
    # Fine-tuned model checkpoints (second pass). load_external_models is a
    # no-op per-file when the .pth is missing, so a partial dir still works.
    if getattr(args, "model_path", None):
        mgr.load_external_models(
            ms2_model_file=os.path.join(args.model_path, "ms2.pth"),
            rt_model_file=os.path.join(args.model_path, "rt.pth"),
            ccs_model_file=os.path.join(args.model_path, "ccs.pth"),
        )
        print(f"Loaded external models from {args.model_path}", flush=True)
    _move_to_device(mgr, args)

    # predict_all (multiprocessing path, >=3000 precursors) returns
    # res["precursor_df"] in an ARBITRARY order, so every needed column must be
    # carried into the prediction input and res_prec iterated, never precursor_df.
    pred_input = precursor_df[
        ["sequence", "mods", "mod_sites", "charge", "precursor_mz", "proteins"]
    ].copy()
    pred_input["nAA"] = pred_input["sequence"].str.len().astype(np.int32)
    mgr.set_default_nce_instrument(pred_input)
    print("Predicting RT / MS2 / mobility ...", flush=True)
    res = mgr.predict_all(
        pred_input,
        predict_items=["rt", "ms2", "mobility"],
        frag_types=ftypes,
        process_num=8,
    )

    res_prec = res["precursor_df"]
    frag_mz = res["fragment_mz_df"]          # dense: (total_rows, 4)
    frag_int = res["fragment_intensity_df"]  # dense: (total_rows, 4)

    rt_pred = res_prec["rt_pred"].to_numpy(dtype=np.float64)
    print(f"rt_pred range: [{rt_pred.min():.4f}, {rt_pred.max():.4f}] "
          f"mean={rt_pred.mean():.4f}", flush=True)

    # --- RT calibration: robust percentile min-max map rt_pred -> seconds ----
    # Pure peptdeep: maps the 1st..99th percentile of predicted RT onto the full
    # gradient. No DIA-NN-derived calibration is applied.
    lo = float(np.percentile(rt_pred, 1.0))
    hi = float(np.percentile(rt_pred, 99.0))
    span = hi - lo
    if span < 1e-9:
        rt_sec = np.full_like(rt_pred, (args.gradient_start_sec + args.gradient_end_sec) / 2.0)
    else:
        rt_sec = np.clip((rt_pred - lo) / span, 0.0, 1.0) * (
            args.gradient_end_sec - args.gradient_start_sec
        ) + args.gradient_start_sec
    mobility = res_prec["mobility_pred"].to_numpy(dtype=np.float64)

    # --- build long-format rows (vectorized) -----------------------------
    frag_start = res_prec["frag_start_idx"].to_numpy()
    frag_stop = res_prec["frag_stop_idx"].to_numpy()
    seqs = res_prec["sequence"].to_numpy()
    mods = res_prec["mods"].to_numpy()
    mod_sites = res_prec["mod_sites"].to_numpy()
    charges = res_prec["charge"].to_numpy()
    pr_mz = res_prec["precursor_mz"].to_numpy(dtype=np.float64)
    proteins = res_prec["proteins"].to_numpy()

    n_prec = len(res_prec)
    counts = (frag_stop - frag_start).astype(np.int64)  # positions per precursor
    total_frag = int(counts.sum())

    # frag type -> (fragment_type_char, fragment_charge); fixed column order.
    TYPE_MAP = {
        "b_z1": ("b", 1),
        "b_z2": ("b", 2),
        "y_z1": ("y", 1),
        "y_z2": ("y", 2),
    }
    type_cols = list(TYPE_MAP.keys())
    n_types = len(type_cols)
    type_chars = np.array([TYPE_MAP[c][0] for c in type_cols])     # b,b,y,y
    type_charges = np.array([TYPE_MAP[c][1] for c in type_cols], dtype=np.int16)

    # Dense fragment matrices (total_frag, n_types).
    fmz = frag_mz[type_cols].to_numpy(dtype=np.float64)
    fint = frag_int[type_cols].to_numpy(dtype=np.float64)

    # Per-fragment precursor index and 1-based series number.
    prec_idx = np.repeat(np.arange(n_prec, dtype=np.int64), counts)
    pos = (np.arange(total_frag, dtype=np.int64)
           - np.repeat(frag_start.astype(np.int64), counts) + 1)

    # Flatten over fragment types (column-major so type index == col index).
    all_prec = np.tile(prec_idx, n_types)
    all_pos = np.tile(pos, n_types)
    all_type = np.repeat(np.arange(n_types, dtype=np.int64), total_frag)
    all_mz = fmz.ravel(order="F")
    all_int = fint.ravel(order="F")
    del fmz, fint

    # Fragment series number. alphabase stores b-ion masses forward (index 0 =
    # b1) but y-ion masses REVERSE (index 0 = y_{n-1}), so forward `pos` is
    # correct for b ions but must be reversed for y ions:
    # y series = counts[prec] - pos + 1. type index 0,1 = b_z1,b_z2; 2,3 = y.
    max_series = counts[all_prec]
    all_pos = np.where(all_type >= 2, max_series - all_pos + 1, all_pos)

    mask = (all_mz >= FRAG_MZ_MIN) & (all_mz <= FRAG_MZ_MAX) & (all_int >= MIN_FRAGMENT_INTENSITY)
    # Optional precursor m/z polish (0 = off).
    if args.mz_min > 0.0:
        mask &= pr_mz[all_prec] > args.mz_min
    if args.mz_max > 0.0:
        mask &= pr_mz[all_prec] < args.mz_max
    all_prec = all_prec[mask]; all_pos = all_pos[mask]; all_type = all_type[mask]
    all_mz = all_mz[mask]; all_int = all_int[mask]

    keep = top_k_mask(all_prec, all_int, TOP_K_FRAGMENTS)
    f_prec = all_prec[keep]
    f_pos = all_pos[keep].astype(np.int16)
    f_type = all_type[keep]
    f_mz = all_mz[keep].astype(np.float32)
    f_int = all_int[keep].astype(np.float32)

    # Optional minimum-fragment polish (0 = off).
    if args.min_frags > 0:
        cnt = np.bincount(f_prec, minlength=n_prec)
        keep_prec = cnt >= args.min_frags
        keep_frag = keep_prec[f_prec]
        f_prec = f_prec[keep_frag]; f_pos = f_pos[keep_frag]; f_type = f_type[keep_frag]
        f_mz = f_mz[keep_frag]; f_int = f_int[keep_frag]

    # Per-precursor attributes (computed once, indexed by fragment precursor).
    mod_seqs = build_modified_sequences_vec(seqs, mods, mod_sites)
    pr_ids = np.array([f"{m}{c}" for m, c in zip(mod_seqs, charges)], dtype=object)
    f_prid = pr_ids[f_prec]
    f_modseq = mod_seqs[f_prec]
    f_seq = seqs[f_prec]
    f_charge = charges[f_prec].astype(np.int16)
    f_rt = rt_sec[f_prec].astype(np.float32)
    f_im = mobility[f_prec].astype(np.float32)
    f_pr_mz = pr_mz[f_prec].astype(np.float32)
    f_prot = proteins[f_prec]
    f_ftype = type_chars[f_type]
    f_fcharge = type_charges[f_type]

    n = len(f_prec)
    print(f"Total fragment records: {n:,}", flush=True)

    # --- write parquet ----------------------------------------------------
    table = pa.table({
        "Precursor.Id": pa.array(f_prid, type=pa.string()),
        "Modified.Sequence": pa.array(f_modseq, type=pa.string()),
        "Stripped.Sequence": pa.array(f_seq, type=pa.string()),
        "Precursor.Charge": pa.array(f_charge, type=pa.int16()),
        "Decoy": pa.array(np.zeros(n, dtype=np.int16), type=pa.int16()),
        "RT": pa.array(f_rt, type=pa.float32()),
        "IM": pa.array(f_im, type=pa.float32()),
        "Precursor.Mz": pa.array(f_pr_mz, type=pa.float32()),
        "Product.Mz": pa.array(f_mz, type=pa.float32()),
        "Relative.Intensity": pa.array(f_int, type=pa.float32()),
        "Fragment.Type": pa.array(f_ftype, type=pa.string()),
        "Fragment.Charge": pa.array(f_fcharge, type=pa.int16()),
        "Fragment.Series.Number": pa.array(f_pos, type=pa.int16()),
        "Protein.Ids": pa.array(f_prot, type=pa.string()),
        "Protein.Names": pa.array(f_prot, type=pa.string()),
    })
    pq.write_table(table, args.out, compression="zstd")
    print(f"Saved {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB, {n:,} rows)",
          flush=True)


# ── finetune ───────────────────────────────────────────────────────────────

def _refine_transfer_df(precursor_df: pd.DataFrame, fragment_df: pd.DataFrame):
    """Sort the transfer library by nAA and reindex the fragment rows.

    PeptDeep's MS2 model writes predictions back with a fast in-order slice only
    when the precursor dataframe is "refined" (index from 0, nAA monotonic
    increasing, contiguous indices); otherwise `update_sliced_fragment_dataframe`
    drops the result and `ms2_model.test()` spuriously reports PCC=0. Sorting
    here keeps every `train_*`/`test` call on the in-order path.
    """
    order = precursor_df.sort_values("nAA").index.values
    precursor_df = precursor_df.loc[order].reset_index(drop=True)
    starts = []
    stops = []
    blocks = []
    start = 0
    for i in range(len(precursor_df)):
        s = int(precursor_df.loc[i, "frag_start_idx"])
        e = int(precursor_df.loc[i, "frag_stop_idx"])
        blocks.append(fragment_df.iloc[s:e])
        starts.append(start)
        start += e - s
        stops.append(start)
    precursor_df["frag_start_idx"] = starts
    precursor_df["frag_stop_idx"] = stops
    fragment_df = pd.concat(blocks, ignore_index=True)
    return precursor_df, fragment_df


def _ms2_similarity_report(mgr: ModelManager, precursor_df: pd.DataFrame,
                           fragment_df: pd.DataFrame) -> None:
    """Compute PCC/COS/SPC between the fine-tuned model and the transfer target.

    PeptDeep 1.3.2's `ms2_model.test()` passes `reference_frag_df`, which forces
    the out-of-order predict path whose `update_sliced_fragment_dataframe` write
    is dropped (a stock alphabase bug), so `test()` spuriously reports PCC=0.
    Here we predict in-order (no reference) and slice the target ourselves.
    """
    import torch
    from peptdeep.model.ms2 import pearson_correlation, spearman_correlation
    from alphabase.peptide.fragment import get_sliced_fragment_dataframe

    sub = precursor_df.copy()
    mgr.set_default_nce_instrument(sub)
    pred = mgr.ms2_model.predict(sub, reference_frag_df=None)
    cols = ["b_z1", "b_z2", "y_z1", "y_z2"]
    pred_np = pred[cols].to_numpy(dtype=np.float32)

    # Normalize the target per precursor (max -> 1).
    tgt = fragment_df[cols].to_numpy(dtype=np.float32).copy()
    for start, stop in zip(precursor_df["frag_start_idx"],
                           precursor_df["frag_stop_idx"], strict=True):
        blk = tgt[start:stop]
        mx = blk.max()
        if mx > 0:
            blk[:] = blk / mx

    pccs, coss, spcs = [], [], []
    device = "cpu"
    for start, stop in zip(precursor_df["frag_start_idx"],
                           precursor_df["frag_stop_idx"], strict=True):
        tt = torch.tensor(tgt[start:stop], dtype=torch.float32).reshape(1, -1)
        if int((tt > 0).sum()) < 1:
            continue
        pp = torch.tensor(pred_np[start:stop], dtype=torch.float32).reshape(1, -1)
        pccs.append(float(pearson_correlation(pp, tt)[0]))
        coss.append(float(torch.cosine_similarity(pp, tt, dim=1)[0]))
        spcs.append(float(spearman_correlation(pp, tt, torch.device(device))[0]))
    pccs = np.array(pccs)
    coss = np.array(coss)
    spcs = np.array(spcs)
    print(f"MS2 similarity (n={len(pccs):,}): "
          f"PCC median={np.median(pccs):.4f} mean={pccs.mean():.4f} | "
          f"COS median={np.median(coss):.4f} | SPC median={np.median(spcs):.4f}",
          flush=True)


def cmd_finetune(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    precursor_df = pd.read_parquet(args.precursors)
    fragment_df = pd.read_parquet(args.fragments)
    print(f"Loaded {len(precursor_df):,} transfer precursors, "
          f"{len(fragment_df):,} fragment rows", flush=True)
    precursor_df, fragment_df = _refine_transfer_df(precursor_df, fragment_df)

    # Training hyperparameters are read from global_settings in
    # ModelManager.__init__, so set them before construction.
    _set_default_nce_instrument(args)
    t = global_settings["model_mgr"]["transfer"]
    t["epoch_rt_ccs"] = args.epochs
    t["epoch_ms2"] = args.epochs
    t["batch_size_rt_ccs"] = args.batch_size
    t["batch_size_ms2"] = args.batch_size
    t["lr_rt_ccs"] = args.lr
    t["lr_ms2"] = args.lr
    t["verbose"] = True

    print("Loading peptdeep models ...", flush=True)
    mgr = ModelManager(device="cpu")
    mgr.load_installed_models("generic")
    _move_to_device(mgr, args)

    print("Fine-tuning RT model ...", flush=True)
    mgr.train_rt_model(precursor_df)

    print("Fine-tuning MS2 model ...", flush=True)
    mgr.train_ms2_model(precursor_df, fragment_df)
    _ms2_similarity_report(mgr, precursor_df, fragment_df)

    # Save RT + MS2 only. CCS/mobility are absent on Astral data; `predict
    # --model-path` loads rt.pth + ms2.pth and keeps the pretrained CCS model.
    mgr.rt_model.save(os.path.join(args.out_dir, "rt.pth"))
    mgr.ms2_model.save(os.path.join(args.out_dir, "ms2.pth"))
    print(f"Saved fine-tuned models to {args.out_dir}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="diaflux_peptdeep")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("predict", help="predict a spectral library from a FASTA")
    pp.add_argument("--fasta", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--nce", type=float, default=25.0,
                    help="HCD normalised collision energy (DIAFluX Astral = 25)")
    pp.add_argument("--instrument", default="Lumos",
                    help="instrument group for the peptdeep models")
    pp.add_argument("--gpu", default="cuda",
                    help="device type: 'cuda'/'gpu' or 'cpu'")
    pp.add_argument("--gpu-id", type=int, default=0,
                    help="which CUDA device to pin to")
    pp.add_argument("--model-path", default=None,
                    help="directory with fine-tuned rt.pth/ms2.pth/ccs.pth to load")
    pp.add_argument("--max-precursors", type=int, default=0,
                    help="limit precursors for a quick test (0 = all)")
    pp.add_argument("--gradient-end-sec", type=float, default=1253.0,
                    help="observed gradient length in seconds (Astral 21min run)")
    pp.add_argument("--gradient-start-sec", type=float, default=0.0)
    pp.add_argument("--missed-cleavages", type=int, default=2)
    pp.add_argument("--pep-len-min", type=int, default=7)
    pp.add_argument("--pep-len-max", type=int, default=40)
    pp.add_argument("--charge-min", type=int, default=2)
    pp.add_argument("--charge-max", type=int, default=4)
    pp.add_argument("--pr-mz-min", type=float, default=300.0)
    pp.add_argument("--pr-mz-max", type=float, default=1200.0)
    # Optional polish (reproduces the previous polish_library.py step; 0 = off).
    pp.add_argument("--mz-min", type=float, default=0.0)
    pp.add_argument("--mz-max", type=float, default=0.0)
    pp.add_argument("--min-frags", type=int, default=0)
    pp.set_defaults(func=cmd_predict)

    pf = sub.add_parser("finetune",
                        help="fine-tune RT/MS2 models on a transfer library")
    pf.add_argument("--precursors", required=True,
                    help="transfer_precursors.parquet (sequence/mods/mod_sites/charge/rt_norm/frag_start_idx/frag_stop_idx)")
    pf.add_argument("--fragments", required=True,
                    help="transfer_fragments.parquet (wide b_z1/b_z2/y_z1/y_z2)")
    pf.add_argument("--out-dir", required=True,
                    help="directory to write rt.pth/ms2.pth")
    pf.add_argument("--nce", type=float, default=25.0)
    pf.add_argument("--instrument", default="Lumos")
    pf.add_argument("--gpu", default="cuda")
    pf.add_argument("--gpu-id", type=int, default=0)
    pf.add_argument("--epochs", type=int, default=40)
    pf.add_argument("--batch-size", type=int, default=2000)
    pf.add_argument("--lr", type=float, default=0.0001)
    pf.set_defaults(func=cmd_finetune)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
