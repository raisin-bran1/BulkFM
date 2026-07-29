#!/usr/bin/env python3
"""Scan TCGA H5 and build gene vocabulary using the comprehensive ENSG mapping.

Reads:
  - tcga_matrix.h5
  - tcga_gene_ensg_mapping.csv  (from build_tcga_ensg_mapping.py)
  - pc_ensg_ids.npy             (from ARCHS4 scan)

Writes (all to outputdir):
  tcga_gene_vocabulary.csv  — protein-coding ENSG IDs present in TCGA
  tcga_metadata_fields.txt  — list of available metadata field names
"""

import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Scan TCGA H5 and build gene vocabulary")
parser.add_argument("--datapath", type=str,
                    default="/media/volume/bulkrnadata/tcga_matrix.h5",
                    help="TCGA H5 file path")
parser.add_argument("--tcga_outdir", type=str,
                    default="/media/volume/bulkrnadata/tcgadata/",
                    help="Directory with tcga_gene_ensg_mapping.csv")
parser.add_argument("--archs4_outdir", type=str,
                    default="/media/volume/bulkrnadata/humandata/",
                    help="Directory with ARCHS4 pc_ensg_ids.npy")
parser.add_argument("--outputdir", type=str,
                    default="/media/volume/bulkrnadata/tcgadata/",
                    help="Output directory")
args = parser.parse_args()

os.makedirs(args.outputdir, exist_ok=True)
t0 = time.time()

# ── 1. Load TCGA comprehensive ENSG mapping ─────────────────────────
print("=== Loading TCGA ENSG mapping ===")
tcga_map = pd.read_csv(os.path.join(args.tcga_outdir, 'tcga_gene_ensg_mapping.csv'))
print(f"  TCGA genes in mapping: {len(tcga_map)}")
print(f"  Resolved: {tcga_map['ensembl_gene_id'].notna().sum()} / {len(tcga_map)}")

# ── 2. Load PC gene pool from ARCHS4 ─────────────────────────────────
print("=== Loading ARCHS4 PC gene pool ===")
pc_ensg_ids = np.load(os.path.join(args.archs4_outdir, 'pc_ensg_ids.npy'),
                      allow_pickle=True)
pc_set = set(pc_ensg_ids)
print(f"  Protein-coding ENSG pool: {len(pc_set)}")

# ── 3. Determine PC genes present in TCGA ────────────────────────────
print("=== Determining PC genes in TCGA ===")
tcga_ensg_resolved = tcga_map.dropna(subset=['ensembl_gene_id'])
tcga_pc_ensgs = sorted(set(
    e for e in tcga_ensg_resolved['ensembl_gene_id'] if e in pc_set
))
print(f"  PC genes found in TCGA: {len(tcga_pc_ensgs)} / {len(pc_set)}")
print(f"  PC genes lost: {len(pc_set) - len(tcga_pc_ensgs)}")

# ── 4. Save TCGA gene vocabulary ──────────────────────────────────
vocab_file = os.path.join(args.outputdir, 'tcga_gene_vocabulary.csv')
pd.Series(tcga_pc_ensgs, name="genes").to_csv(
    vocab_file, index=False, header=['genes'])
print(f"  Saved gene vocabulary → {vocab_file}")

# ── 5. Save metadata field list ────────────────────────────────────
with h5py.File(args.datapath, 'r') as f:
    meta_fields = sorted(f['meta'].keys())

meta_file = os.path.join(args.outputdir, 'tcga_metadata_fields.txt')
with open(meta_file, 'w') as f:
    for field in meta_fields:
        f.write(f"{field}\n")
print(f"  Saved {len(meta_fields)} metadata fields → {meta_file}")

elapsed = (time.time() - t0) / 60
print(f"Scan complete in {elapsed:.2f} minutes")
