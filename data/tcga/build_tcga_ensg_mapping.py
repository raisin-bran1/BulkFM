#!/usr/bin/env python3
"""Build comprehensive TCGA symbol→ENSG mapping.

Uses (in priority order):
  1. ARCHS4 gene_ensg_mapping.csv (primary — authoritative for pretraining vocab)
  2. HGNC: current approved symbol match
  3. HGNC: previous symbol match (preferred over alias)
  4. HGNC: alias symbol match (weakest)
  5. mygene.info API (final fallback for truly unresolved)

Outputs tcga_gene_ensg_mapping.csv with every TCGA gene mapped to an ENSG ID
whenever possible.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import requests

parser = argparse.ArgumentParser(description="Build TCGA symbol→ENSG mapping")
parser.add_argument("--datapath", type=str,
                    default="/media/volume/bulkrnadata/tcga_matrix.h5",
                    help="TCGA H5 file path")
parser.add_argument("--refdir", type=str,
                    default="/media/volume/bulkrnadata/",
                    help="Directory with HGNC reference file")
parser.add_argument("--archs4_outdir", type=str,
                    default="/media/volume/bulkrnadata/humandata/",
                    help="Directory with ARCHS4 gene_ensg_mapping.csv")
parser.add_argument("--outputdir", type=str,
                    default="/media/volume/bulkrnadata/tcgadata/",
                    help="Output directory")
parser.add_argument("--hgnc", type=str, default="hgnc_complete_set.txt",
                    help="HGNC complete set filename (in refdir)")
parser.add_argument("--no-api", action="store_true",
                    help="Skip mygene.info API lookup (local resolution only)")
args = parser.parse_args()

os.makedirs(args.outputdir, exist_ok=True)
t0 = time.time()

# ── 1. Load TCGA gene symbols ──────────────────────────────────────
print("=== Reading TCGA gene symbols ===")
with h5py.File(args.datapath, 'r') as f:
    tcga_symbols = [s.decode().upper() for s in f['meta/genes'][:]]
n_tcga = len(tcga_symbols)
print(f"  Total TCGA genes: {n_tcga}")

# ── 2. Load ARCHS4 primary mapping + PC vocabulary ─────────────────
print("=== ARCHS4 mapping ===")
mapping_df = pd.read_csv(os.path.join(args.archs4_outdir, 'gene_ensg_mapping.csv'))
archs4_sym_to_ensg = dict(zip(mapping_df['symbol'], mapping_df['ensembl_gene_id']))

# Load PC vocabulary (ENSG IDs we care about keeping)
pc_vocab = pd.read_csv(os.path.join(args.archs4_outdir, 'gene_vocabulary.csv'))
pc_ensg_set = set(pc_vocab['genes'])

resolved = {}  # symbol -> (ensg, source)
unresolved = []

for sym in tcga_symbols:
    if sym in archs4_sym_to_ensg:
        resolved[sym] = (archs4_sym_to_ensg[sym], 'archs4')
    else:
        unresolved.append(sym)

print(f"  Resolved via ARCHS4: {len(resolved)} / {n_tcga}")
print(f"  Remaining: {len(unresolved)}")

# ── 3. HGNC-based resolution (priority: current > prev > alias) ────
print("=== HGNC resolution ===")
hgnc = pd.read_csv(os.path.join(args.refdir, args.hgnc), sep='\t', low_memory=False)
hgnc_with_ensg = hgnc.dropna(subset=['ensembl_gene_id'])

# 3a: Current symbol match
for _, r in hgnc_with_ensg.iterrows():
    sym = str(r['symbol']).upper().strip()
    if sym in unresolved:
        resolved[sym] = (r['ensembl_gene_id'], 'hgnc_current')

# 3b: Previous symbol match (preferred over alias)
prev_alias_map = {}
for _, r in hgnc_with_ensg.iterrows():
    if pd.notna(r['prev_symbol']):
        for prev in str(r['prev_symbol']).split('|'):
            p = prev.strip().upper()
            if p and p in unresolved and p not in resolved:
                prev_alias_map.setdefault(p, []).append(
                    (r['ensembl_gene_id'], 'hgnc_prev', r['symbol'])
                )

# 3c: Alias symbol match (only if neither current nor prev resolved)
alias_only = {}
for _, r in hgnc_with_ensg.iterrows():
    if pd.notna(r['alias_symbol']):
        for als in str(r['alias_symbol']).split('|'):
            a = als.strip().upper()
            if a and a in unresolved and a not in resolved and a not in prev_alias_map:
                alias_only.setdefault(a, []).append(
                    (r['ensembl_gene_id'], 'hgnc_alias', r['symbol'])
                )

# Resolve prev_symbol matches — prefer PC vocabulary genes
for sym, candidates in prev_alias_map.items():
    if len(candidates) == 1:
        resolved[sym] = (candidates[0][0], candidates[0][1])
    else:
        # Multiple candidates — prefer one in the PC vocabulary
        pc_cands = [c for c in candidates if c[0] in pc_ensg_set]
        chosen = pc_cands[0] if pc_cands else candidates[0]
        resolved[sym] = (chosen[0], chosen[1])

# Resolve alias-only matches
for sym, candidates in alias_only.items():
    if len(candidates) == 1:
        resolved[sym] = (candidates[0][0], candidates[0][1])
    else:
        pc_cands = [c for c in candidates if c[0] in pc_ensg_set]
        chosen = pc_cands[0] if pc_cands else candidates[0]
        resolved[sym] = (chosen[0], chosen[1])

hgnc_resolved = sum(1 for s in tcga_symbols if s in resolved)
print(f"  Resolved via HGNC: {hgnc_resolved - len([s for s in tcga_symbols if s in archs4_sym_to_ensg])}")
remaining = [s for s in unresolved if s not in resolved]
print(f"  Remaining after HGNC: {len(remaining)}")

# ── 4. mygene.info API resolution ─────────────────────────────────
if not args.no_api and remaining:
    print("=== mygene.info API resolution ===")
    api_resolved = 0
    api_failed = []

    batch_size = 500
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_str = ' '.join(batch)

        try:
            payload = {
                'q': batch_str,
                'scopes': 'symbol,alias,prev_symbol',
                'fields': 'symbol,ensembl.gene',
                'species': 'human'
            }
            r = requests.post(
                'https://mygene.info/v3/query',
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            r.raise_for_status()
            results = r.json()

            # Group hits by query symbol
            from collections import defaultdict
            hits_by_query = defaultdict(list)
            for hit in results:
                q = hit.get('query', '').upper()
                if q in remaining:
                    hits_by_query[q].append(hit)

            for sym, hits in hits_by_query.items():
                if sym in resolved:
                    continue
                # Among hits with ENSG, prefer:
                # 1. Hit whose current symbol is in ARCHS4 mapping
                # 2. Hit in PC vocabulary
                # 3. Any hit with ENSG
                candidates = []
                for h in hits:
                    if 'ensembl' not in h or 'gene' not in h['ensembl']:
                        continue
                    ensg = h['ensembl']['gene']
                    hit_sym = h.get('symbol', '')
                    score = h.get('_score', 0)
                    in_archs4 = hit_sym in archs4_sym_to_ensg
                    in_pc = ensg in pc_ensg_set
                    candidates.append((ensg, hit_sym, in_archs4, in_pc, score))

                if candidates:
                    candidates.sort(key=lambda x: (not x[2], not x[3], -x[4]))
                    best = candidates[0]
                    resolved[sym] = (best[0], 'mygene')
                    api_resolved += 1

            time.sleep(0.5)

        except Exception as e:
            api_failed.extend(batch)

        done = min(batch_start + batch_size, len(remaining))
        print(f"  Progress: {done}/{len(remaining)} queried, {api_resolved} resolved")

    print(f"  Resolved via mygene.info: {api_resolved}")
    print(f"  API failures: {len(api_failed)}")
    if api_failed:
        print(f"  Failed examples: {api_failed[:10]}")

# ── 5. Build final ENSG list ──────────────────────────────────────
tcga_ensg = [resolved[s][0] if s in resolved else None for s in tcga_symbols]
source_col = [resolved[s][1] if s in resolved else 'unresolved' for s in tcga_symbols]

final_unresolved = [s for i, s in enumerate(tcga_symbols) if tcga_ensg[i] is None]
print(f"\n=== Final ===")
print(f"  Total TCGA genes: {n_tcga}")
print(f"  Resolved: {len(resolved)}")
print(f"  Unresolved: {len(final_unresolved)}")

if final_unresolved:
    print(f"  Unresolved examples: {final_unresolved[:20]}")

source_counts = pd.Series(source_col).value_counts()
print(f"  By source: {source_counts.to_dict()}")

# ── 6. PC gene recovery stats ─────────────────────────────────────
tcga_pc_ensgs = set(e for e in tcga_ensg if e and e in pc_ensg_set)
archs4_only_pc = pc_ensg_set - tcga_pc_ensgs
print(f"\n  PC genes in TCGA: {len(tcga_pc_ensgs)} / {len(pc_ensg_set)}")
print(f"  PC genes lost: {len(archs4_only_pc)}")

# ── 7. Save mapping ───────────────────────────────────────────────
out_df = pd.DataFrame({
    'symbol': tcga_symbols,
    'ensembl_gene_id': tcga_ensg,
    'mapping_source': source_col,
})
out_file = os.path.join(args.outputdir, 'tcga_gene_ensg_mapping.csv')
out_df.to_csv(out_file, index=False)
print(f"\nSaved → {out_file}")

elapsed = (time.time() - t0) / 60
print(f"Done in {elapsed:.2f} minutes")
