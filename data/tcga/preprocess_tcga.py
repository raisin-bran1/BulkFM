#!/usr/bin/env python3
"""Preprocess TCGA expression data with metadata attachment.

Uses the comprehensive tcga_gene_ensg_mapping.csv so every TCGA gene
resolves to an ENSG ID, maximizing protein-coding gene recovery.

Reads:
  - tcga_matrix.h5
  - tcga_gene_vocabulary.csv   (from scan_tcga.py)
  - tcga_gene_ensg_mapping.csv (from build_tcga_ensg_mapping.py)
  - human_exon_lengths_df.csv  (reference)

Writes:
  - tcga_processed.parquet     (samples × genes float32, with metadata)
"""

import argparse
import os
import sys
import traceback

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Preprocess TCGA data")
parser.add_argument("--datapath", type=str,
                    default="/media/volume/bulkrnadata/tcga_matrix.h5",
                    help="TCGA H5 file path")
parser.add_argument("--refdir", type=str,
                    default="/media/volume/bulkrnadata/",
                    help="Directory with reference files")
parser.add_argument("--tcga_outdir", type=str,
                    default="/media/volume/bulkrnadata/tcgadata/",
                    help="Directory with scan_tcga.py output")
parser.add_argument("--outputdir", type=str,
                    default="/media/volume/bulkrnadata/tcgadata/",
                    help="Output directory")
parser.add_argument("--lengths", type=str, default="human_exon_lengths_df.csv",
                    help="Exon lengths filename (in refdir)")
parser.add_argument("--nonzero_min", type=int, default=10000,
                    help="Minimum nonzero protein-coding genes per sample")
args = parser.parse_args()

try:
    out_file = os.path.join(args.outputdir, 'tcga_processed.parquet')
    if os.path.exists(out_file):
        print("Output already exists, skipping.")
        sys.exit(0)

    os.makedirs(args.outputdir, exist_ok=True)

    # ── Load precomputed gene metadata ────────────────────────────────
    vocab = pd.read_csv(os.path.join(args.tcga_outdir, 'tcga_gene_vocabulary.csv'))
    target_genes = vocab['genes'].tolist()
    target_set = set(target_genes)
    print(f"Target gene vocabulary: {len(target_genes)} PC genes")

    # Load the comprehensive TCGA→ENSG mapping
    tcga_map = pd.read_csv(os.path.join(args.tcga_outdir, 'tcga_gene_ensg_mapping.csv'))
    sym_to_ensg = dict(zip(tcga_map['symbol'], tcga_map['ensembl_gene_id']))

    # ── Load exon lengths ─────────────────────────────────────────────
    exon_lengths = pd.read_csv(os.path.join(args.refdir, args.lengths))
    exon_lengths['gene name'] = exon_lengths['gene name'].str.upper()
    exon_lengths = exon_lengths.set_index('gene name')

    def convert_to_TPM(counts_df, length_df):
        L = length_df.reindex(counts_df.index)["length"]
        L = L.fillna(1000).replace(0, 1000)
        rpk = counts_df.astype('float32').div(L / 1000, axis=0)
        tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
        return tpm

    # ── Read TCGA H5 ─────────────────────────────────────────────────
    print("=== Loading TCGA data from H5 ===")
    with h5py.File(args.datapath, 'r') as f:
        expr = f['data/expression'][:]
        tcga_symbols = [s.decode().upper() for s in f['meta/genes'][:]]
        meta_data = {}
        for k in f['meta'].keys():
            if k == 'genes':
                continue
            arr = f['meta'][k][:]
            meta_data[k] = [x.decode('utf-8', errors='replace') if isinstance(x, bytes) else str(x) for x in arr]

    n_samples, n_genes = expr.shape
    print(f"  Expression shape: {expr.shape}")
    print(f"  Metadata fields: {len(meta_data)}")

    # ── Map symbols → ENSG ───────────────────────────────────────────
    print("=== Mapping genes ===")
    tcga_ensg = [sym_to_ensg.get(sym) for sym in tcga_symbols]
    valid_mask = np.array([e is not None for e in tcga_ensg])
    valid_ensg = np.array(tcga_ensg)
    print(f"  Mapped genes: {valid_mask.sum()}/{n_genes}")

    # ── Build expression DataFrame indexed by ENSG ───────────────────
    expr_mapped = expr[:, valid_mask].T
    mapped_ensg = valid_ensg[valid_mask]

    expr_df = pd.DataFrame(expr_mapped, index=mapped_ensg)
    expr_df.index.name = 'ensg'

    # Handle duplicate ENSG IDs (multiple symbols → same ENSG)
    if expr_df.index.duplicated().any():
        n_dup = expr_df.index.duplicated().sum()
        print(f"  Aggregating {n_dup} duplicate ENSG entries (summing counts)")
        expr_df = expr_df.groupby(level=0).sum()
    print(f"  Expression shape (unique ENSGs): {expr_df.shape}")

    # ── TPM conversion ───────────────────────────────────────────────
    print("=== TPM conversion ===")
    # Need symbols for exon length lookup — use reverse mapping
    ensg_to_sym = {}
    for s, e in sym_to_ensg.items():
        if pd.notna(e):
            ensg_to_sym.setdefault(e, s)

    sym_for_lengths = [ensg_to_sym.get(e, '') for e in expr_df.index]
    expr_sym = expr_df.copy()
    expr_sym.index = sym_for_lengths
    expr_sym.index.name = 'symbol'
    tpm_df = convert_to_TPM(expr_sym, exon_lengths)
    logtpm_df = np.log1p(tpm_df).round(4).astype('float32')
    logtpm_df.index = expr_df.index

    # ── Filter to PC gene vocabulary ─────────────────────────────────
    print("=== Filtering to PC gene vocabulary ===")
    in_vocab = np.array([e in target_set for e in logtpm_df.index])
    filtered_df = logtpm_df.loc[in_vocab]
    print(f"  Genes: {filtered_df.shape[0]} / {logtpm_df.shape[0]} in vocabulary")

    # Reindex to ensure exact vocabulary order
    # First ensure no duplicates in target_genes or filtered_df index
    target_unique = list(dict.fromkeys(target_genes))  # deduplicate preserving order
    if filtered_df.index.duplicated().any():
        filtered_df = filtered_df.groupby(level=0).sum()
    filtered_df = filtered_df.reindex(target_unique)

    # Drop any NaN rows (genes in vocab but not in TCGA expression)
    orig_rows = filtered_df.shape[0]
    filtered_df = filtered_df.dropna(how='all')
    if filtered_df.shape[0] < orig_rows:
        print(f"  Dropped {orig_rows - filtered_df.shape[0]} genes not in TCGA expression")
    print(f"  After reindex: {filtered_df.shape}")

    # ── QC: filter samples by nonzero PC genes ────────────────────────
    print("=== QC ===")
    nz_per_sample = (filtered_df > 0).sum(axis=0)
    qc_mask = nz_per_sample >= args.nonzero_min
    print(f"  Samples before QC: {filtered_df.shape[1]}, after: {qc_mask.sum()}")
    filtered_df = filtered_df.loc[:, qc_mask]

    # Transpose to (samples, genes)
    final_df = filtered_df.transpose()
    final_df.index.name = 'sample_id'

    # ── Attach metadata ──────────────────────────────────────────────
    print("=== Attaching metadata ===")
    meta_trimmed = {}
    for k, v in meta_data.items():
        arr = np.array(v)
        meta_trimmed[k] = arr[qc_mask]

    meta_df = pd.DataFrame(meta_trimmed, index=final_df.index)
    final_df = pd.concat([meta_df, final_df], axis=1)
    print(f"  Final shape: {final_df.shape}")
    print(f"  Meta columns ({len(meta_trimmed)}): {list(meta_trimmed.keys())[:10]}...")

    # ── Save ─────────────────────────────────────────────────────────
    final_df.to_parquet(out_file, engine='pyarrow', compression='snappy')
    print(f"Saved → {out_file}")

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
