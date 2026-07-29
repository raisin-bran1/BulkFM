#!/usr/bin/env python3
"""One-time scan of the mouse ARCHS4 H5 file.

Reads the H5 once to:
  1. Pre-filter samples by single-cell probability (< threshold).
  2. Extract the ARCHS4 gene symbol -> ENSMUSG mapping.
  3. Identify the protein-coding gene set (from GENCODE M39).
  4. Save all metadata so downstream chunk preprocessing is fast and H5-free.

Outputs (all written to *outputdir*):
  passing_indices.npy     -- shuffled sample indices that pass SC filter
  gene_symbol_mapping.csv -- symbol -> ENSMUSG for all ARCHS4 genes
  pc_ensmusg_ids.npy      -- set of protein-coding ENSMUSG IDs (from GENCODE)
  gene_vocabulary.csv     -- sorted list of protein-coding ENSMUSG IDs for model vocab
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Scan mouse ARCHS4 H5 and save metadata")
parser.add_argument("--datapath", type=str,
                    default="/media/volume/bulkrnamouse/mouse_gene_v2.latest.h5",
                    help="H5 file path")
parser.add_argument("--refdir", type=str,
                    default="/home/exouser/BulkFM/data/archs4mouse/",
                    help="Directory with GENCODE reference file")
parser.add_argument("--outputdir", type=str,
                    default="/media/volume/bulkrnamouse/mousedata/",
                    help="Output directory")
parser.add_argument("--sc_threshold", type=float, default=0.5,
                    help="Single-cell probability threshold (samples < threshold pass)")
parser.add_argument("--pc_genes", type=str, default="mouse_protein_coding_genes.csv",
                    help="Mouse protein-coding genes CSV (in refdir)")
args = parser.parse_args()

os.makedirs(args.outputdir, exist_ok=True)
t0 = time.time()

# -- 1. Sample pre-filter by SC probability ------------------------------
print("=== Scanning H5: sample pre-filter ===")
with h5py.File(args.datapath, 'r') as f:
    n_total = f["data/expression"].shape[1]
    sc_prob = f["meta/samples/singlecellprobability"][:]

sc_mask = sc_prob < args.sc_threshold
passing = np.where(sc_mask)[0]
print(f"Total samples: {n_total}  | SC prob < {args.sc_threshold}: {len(passing)}")

np.random.seed(42)
np.random.shuffle(passing)

indices_file = os.path.join(args.outputdir, 'passing_indices.npy')
np.save(indices_file, passing)
print(f"Saved {len(passing)} indices to {indices_file}")

# -- 2. Gene symbol -> ENSMUSG mapping from H5 ---------------------------
print("=== Extracting gene metadata from H5 ===")
with h5py.File(args.datapath, 'r') as f:
    archs4_symbols = [s.decode() for s in f['meta/genes/gene_symbol'][:]]
    archs4_ensmusg = [e.decode() for e in f['meta/genes/ensembl_gene_id'][:]]

# Build mapping (handle symbols with multiple ENSMUSG IDs)
sym_to_ensmusgs = defaultdict(set)
for sym, e in zip(archs4_symbols, archs4_ensmusg):
    sym_to_ensmusgs[sym].add(e)

print(f"ARCHS4 gene symbols: {len(sym_to_ensmusgs)}")

# -- 3. Load GENCODE M39 protein-coding gene set -------------------------
PC_PATH = os.path.join(args.refdir, args.pc_genes)
pc_df = pd.read_csv(PC_PATH)
shared_gene_pool = set(pc_df['gene_id'])
print(f"Protein-coding ENSMUSG IDs from GENCODE: {len(shared_gene_pool)}")

# -- 4. Build preferred mapping (GENCODE protein-coding first) -----------
archs4_sym_to_ensmusg = {}
for sym, ensmusgs in sym_to_ensmusgs.items():
    for e in ensmusgs:
        if e in shared_gene_pool:
            archs4_sym_to_ensmusg[sym] = e
            break
    else:
        archs4_sym_to_ensmusg[sym] = next(iter(ensmusgs))

# Save symbol -> ENSMUSG mapping for downstream use
mapping_df = pd.DataFrame([
    {"symbol": sym, "ensembl_gene_id": ensmusg}
    for sym, ensmusg in archs4_sym_to_ensmusg.items()
])
mapping_file = os.path.join(args.outputdir, 'gene_symbol_mapping.csv')
mapping_df.to_csv(mapping_file, index=False)
print(f"Saved {len(mapping_df)} gene mappings to {mapping_file}")

# Save protein-coding ENSMUSG ID set
pc_ids_sorted = sorted(shared_gene_pool)
pc_file = os.path.join(args.outputdir, 'pc_ensmusg_ids.npy')
np.save(pc_file, np.array(pc_ids_sorted, dtype=object))

# Determine which mapped ENSMUSGs are in the protein-coding set
final_ensmusgs = [e for e in archs4_sym_to_ensmusg.values() if e in shared_gene_pool]
final_ensmusgs = sorted(set(final_ensmusgs))

# Save gene vocabulary (sorted PC ENSMUSG IDs from ARCHS4)
vocab_df = pd.Series(final_ensmusgs, name="genes")
vocab_file = os.path.join(args.outputdir, 'gene_vocabulary.csv')
vocab_df.to_csv(vocab_file, index=False, header=['genes'])
print(f"Gene vocabulary: {len(final_ensmusgs)} protein-coding genes -> {vocab_file}")

elapsed = (time.time() - t0) / 60
print(f"Scan complete in {elapsed:.2f} minutes.")
