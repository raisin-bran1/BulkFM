#!/usr/bin/env python3
"""One-time scan of the ARCHS4 H5 file.

Reads the H5 once to:
  1. Pre-filter samples by single-cell probability (< threshold).
  2. Extract the ARCHS4 symbol→ENSG mapping.
  3. Identify the protein-coding gene set (intersection with HGNC).
  4. Save all metadata so downstream chunk preprocessing is fast and H5-free.

Outputs (all written to *outputdir*):
  passing_indices.npy  — shuffled sample indices that pass SC filter
  gene_ensg_mapping.csv — symbol → ensembl_gene_id for all ARCHS4 genes
  pc_ensg_ids.npy      — set of protein-coding ENSG IDs (intersection)
  gene_vocabulary.csv  — sorted list of protein-coding ENSG IDs for model vocab
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Scan ARCHS4 H5 and save metadata")
parser.add_argument("--datapath", type=str,
                    default="/media/volume/bulkrnadata/human_gene_v2.latest.h5",
                    help="H5 file path")
parser.add_argument("--refdir", type=str,
                    default="/media/volume/bulkrnadata/",
                    help="Directory with HGNC reference file")
parser.add_argument("--outputdir", type=str,
                    default="/media/volume/bulkrnadata/humandata/",
                    help="Output directory")
parser.add_argument("--sc_threshold", type=float, default=0.5,
                    help="Single-cell probability threshold (samples < threshold pass)")
parser.add_argument("--hgnc", type=str, default="hgnc_complete_set.txt",
                    help="HGNC complete set filename (in refdir)")
args = parser.parse_args()

os.makedirs(args.outputdir, exist_ok=True)
t0 = time.time()

# ── 1. Sample pre-filter by SC probability ─────────────────────────────
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

# ── 2. Gene symbol → ENSG mapping from H5 ──────────────────────────────
print("=== Extracting gene metadata from H5 ===")
with h5py.File(args.datapath, 'r') as f:
    archs4_symbols = [s.decode().upper() for s in f['meta/genes/symbol'][:]]
    archs4_ensg = [e.decode() for e in f['meta/genes/ensembl_gene'][:]]

# Build mapping (handle symbols with multiple ENSG IDs)
sym_to_ensgs = defaultdict(set)
for sym, e in zip(archs4_symbols, archs4_ensg):
    sym_to_ensgs[sym].add(e)

# ── 3. Load HGNC protein-coding gene set ───────────────────────────────
HGNC_PATH = os.path.join(args.refdir, args.hgnc)
hgnc = pd.read_csv(HGNC_PATH, sep='\t', low_memory=False)
pc_genes = hgnc[hgnc['locus_group'] == 'protein-coding gene'].copy()
pc_genes = pc_genes.dropna(subset=['ensembl_gene_id'])
shared_gene_pool = set(pc_genes['ensembl_gene_id'])
print(f"Protein-coding ENSG IDs from HGNC: {len(shared_gene_pool)}")

# ── 4. Build preferred mapping (HGNC protein-coding first) ─────────────
archs4_sym_to_ensg = {}
for sym, ensgs in sym_to_ensgs.items():
    for e in ensgs:
        if e in shared_gene_pool:
            archs4_sym_to_ensg[sym] = e
            break
    else:
        archs4_sym_to_ensg[sym] = next(iter(ensgs))

# Save symbol→ENSG mapping for downstream use
mapping_df = pd.DataFrame([
    {"symbol": sym, "ensembl_gene_id": ensg}
    for sym, ensg in archs4_sym_to_ensg.items()
])
mapping_file = os.path.join(args.outputdir, 'gene_ensg_mapping.csv')
mapping_df.to_csv(mapping_file, index=False)
print(f"Saved {len(mapping_df)} gene mappings to {mapping_file}")

# Save protein-coding ENSG ID set
pc_ids_sorted = sorted(shared_gene_pool)
pc_file = os.path.join(args.outputdir, 'pc_ensg_ids.npy')
np.save(pc_file, np.array(pc_ids_sorted, dtype=object))

# Determine which mapped ENSGs are in the protein-coding set
final_ensgs = [e for e in archs4_sym_to_ensg.values() if e in shared_gene_pool]
final_ensgs = sorted(set(final_ensgs))

# Save gene vocabulary (sorted PC ENSG IDs from ARCHS4)
vocab_df = pd.Series(final_ensgs, name="genes")
vocab_file = os.path.join(args.outputdir, 'gene_vocabulary.csv')
vocab_df.to_csv(vocab_file, index=False, header=['genes'])
print(f"Gene vocabulary: {len(final_ensgs)} protein-coding genes → {vocab_file}")

elapsed = (time.time() - t0) / 60
print(f"Scan complete in {elapsed:.2f} minutes.")
