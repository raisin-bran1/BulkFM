import archs4py as a4
import h5py
import pandas as pd
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser(description="Preprocess data in parallel.")
parser.add_argument("--datapath", type=str, default="/global/scratch/users/brianzhou/mouse_gene_v2.latest.h5", help="Data file path")
parser.add_argument("--outputdir", type=str, default="/global/scratch/users/brianzhou/subset_100k_mouse/", help="Output folder")
parser.add_argument("--chunk_id", type=int, required=True, help="Index of the chunk to process")
parser.add_argument("--chunk_size", type=int, default=10000, help="Size of each chunk")
parser.add_argument("--total_subset", type=int, default=100000, help="Total size of the subset pool")
# TOTAL SIZE: 1145967
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.outputdir, exist_ok=True)

H5_PATH = args.datapath
CHUNK_SIZE = args.chunk_size
TOTAL_SUBSET = args.total_subset

# 1. Reproducible Random Selection
with h5py.File(H5_PATH, 'r') as f:
    n_total = f["data/expression"].shape[1]
    all_indices = np.arange(n_total)
    
    # CRITICAL: Using the same seed across ALL parallel jobs ensures 
    # they all "see" the same 100k subset before slicing their piece.
    np.random.seed(42)
    subset_indices = np.random.choice(all_indices, TOTAL_SUBSET, replace=False)

# 2. Slice the specific chunk for THIS task
i = args.chunk_id
start_idx = i * CHUNK_SIZE
end_idx = (i + 1) * CHUNK_SIZE

if start_idx >= TOTAL_SUBSET:
    print(f"Error: Chunk ID {i} exceeds total subset size.")
    exit(1)

shard_indices = subset_indices[start_idx : end_idx]

print(f"--- Task {i}: Processing samples {start_idx} to {end_idx} ---")
start_time = time.time()

# 3. Use the parallel 'index' function (uses 16 cores within this task)
df = a4.data.index(H5_PATH, shard_indices.tolist())

# 4. Save to Parquet
out_file = os.path.join(args.outputdir, f'input_chunk_{i:03d}.parquet')
df.to_parquet(out_file, engine='pyarrow', compression='snappy')

elapsed = (time.time() - start_time) / 60
print(f"Finished {out_file} in {elapsed:.2f} minutes.")