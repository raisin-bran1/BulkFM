#!/usr/bin/env python3
"""Preprocess a downloaded expression chunk.

Reads:
  - input_chunk_{id}.parquet  (raw counts, rows = genes, cols = samples)
  - gene_ensg_mapping.csv     (precomputed by scan_h5.py)
  - pc_ensg_ids.npy           (precomputed by scan_h5.py)
  - human_exon_lengths_df.csv (reference)

Writes:
  - processed_human_{id}.parquet  (log1p(TPM), rows = samples, cols = genes)
"""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Preprocess human expression chunk")
parser.add_argument("--datadir", type=str,
                    default="/media/volume/bulkrnadata/humandata/",
                    help="Directory with input parquet")
parser.add_argument("--outputdir", type=str,
                    default="/media/volume/bulkrnadata/humandata/",
                    help="Output directory")
parser.add_argument("--refdir", type=str,
                    default="/media/volume/bulkrnadata/",
                    help="Directory with reference files (exon lengths)")
parser.add_argument("--input", type=str, default="input_chunk_0.parquet",
                    help="Input parquet filename")
parser.add_argument("--lengths", type=str, default="human_exon_lengths_df.csv",
                    help="Human exon lengths filename (in refdir)")
parser.add_argument("--nonzero_min", type=int, default=10000,
                    help="Minimum nonzero protein-coding genes per sample")
parser.add_argument("--chunk_id", type=int, required=True,
                    help="Index of the chunk to process")
args = parser.parse_args()

try:
    out_file = os.path.join(args.outputdir, f'processed_human_{args.chunk_id:03d}.parquet')
    if os.path.exists(out_file):
        print(f"[{args.chunk_id:03d}] Output already exists, skipping.")
        sys.exit(0)

    # ── Load precomputed gene metadata (no H5 access) ──────────────────
    mapping_df = pd.read_csv(os.path.join(args.outputdir, 'gene_ensg_mapping.csv'))
    sym_to_ensg = dict(zip(mapping_df['symbol'], mapping_df['ensembl_gene_id']))
    shared_gene_pool = set(np.load(
        os.path.join(args.outputdir, 'pc_ensg_ids.npy'),
        allow_pickle=True
    ))
    print(f"[{args.chunk_id:03d}] Loaded {len(sym_to_ensg)} mappings, {len(shared_gene_pool)} PC genes")

    # ── Read input chunk ───────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(args.datadir, args.input))
    df = df.rename(columns={df.columns[0]: 'genes'})
    df['genes'] = df['genes'].str.upper()

    df = df.groupby(['genes'], as_index=False).sum()
    print(f"[{args.chunk_id:03d}] After aggregation: {df.shape}")

    # ── TPM with exon lengths ──────────────────────────────────────────
    exon_lengths = pd.read_csv(os.path.join(args.refdir, args.lengths))
    exon_lengths['gene name'] = exon_lengths['gene name'].str.upper()
    exon_lengths = exon_lengths.set_index('gene name')

    def convert_to_TPM(counts_df, length_df):
        L = length_df.reindex(counts_df.index)["length"]
        L = L.fillna(1000).replace(0, 1000)
        rpk = counts_df.astype('float32').div(L / 1000, axis=0)
        tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
        return tpm

    df = df.set_index('genes')
    tpm_df = convert_to_TPM(df, exon_lengths)
    print(f"[{args.chunk_id:03d}] TPM shape: {tpm_df.shape}")

    # ── Map symbols → ENSG, filter to PC genes ─────────────────────────
    tpm_df['genes'] = tpm_df.index.to_series().map(sym_to_ensg)
    mask = tpm_df['genes'].isin(shared_gene_pool)
    print(f"[{args.chunk_id:03d}] PC genes kept: {sum(mask)}/{len(mask)}")
    processed_df = tpm_df[mask].set_index('genes').sort_index()
    print(f"[{args.chunk_id:03d}] Processed shape: {processed_df.shape}")

    # ── QC ─────────────────────────────────────────────────────────────
    nz_per_sample = (processed_df > 0).sum(axis=0)
    qc_mask = nz_per_sample >= args.nonzero_min
    print(f"[{args.chunk_id:03d}] Samples before QC: {processed_df.shape[1]}, after: {qc_mask.sum()}")
    processed_df = processed_df.loc[:, qc_mask]

    # ── log1p transform and save ───────────────────────────────────────
    logtpm_df = np.log1p(processed_df).round(4).astype('float32')
    final_df = logtpm_df.transpose()
    final_df.index.name = 'sample_id'
    print(f"[{args.chunk_id:03d}] Final shape: {final_df.shape}")

    final_df.to_parquet(out_file, engine='pyarrow', compression='snappy')
    print(f"[{args.chunk_id:03d}] Saved {out_file}")

except Exception as e:
    print(f"[{args.chunk_id:03d}] ERROR: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
