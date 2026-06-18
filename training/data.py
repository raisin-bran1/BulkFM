# coding=utf-8
# Copyright 2026 The Google Research Authors.

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

def _parquet_stored_value_type(t):
    """Unwrap dictionary-encoded columns to the stored value type."""
    while pa.types.is_dictionary(t):
        t = t.value_type
    return t

def _parquet_numeric_gene_columns(schema: pa.Schema) -> list:
    """Column names to use as gene expression: numeric types only."""
    excluded = {'geo_accession', '__index_level_0__'}
    out = []
    for i in range(len(schema)):
        field = schema.field(i)
        if field.name in excluded:
            continue
        t = _parquet_stored_value_type(field.type)
        if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_decimal(t):
            out.append(field.name)
    return out

class ExpressionMLMDataset(Dataset):
    """Expression dataset with MLM-style masking."""

    def __init__(self, expr_array, mask_ratio=0.15, mask_token=-10, 
                 mask_token_prob=0.8, random_token_prob=0.1, num_bins=50):
        self.X = expr_array.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob

        # Precompute bins (quantiles per sample)
        B, G = self.X.shape
        print(f"[DATA] Pre-calculating quantile bins for {B} samples...")
        self.target_bins = np.zeros((B, G), dtype=np.int64)
        
        # Vectorized ranking across samples
        is_nonzero = (self.X > 0)
        num_nonzero = is_nonzero.sum(axis=1, keepdims=True)
        
        # Assign rank to each nonzero value per sample
        # We handle zero values by setting them to a large negative number
        ranked_x = self.X.copy()
        ranked_x[~is_nonzero] = -1e9
        
        # Sort twice to get the rank in [0, G-1]
        ranks = np.argsort(np.argsort(ranked_x, axis=1), axis=1)
        
        # Number of zeros/masks that come before nonzero values in the rank
        num_zeros = (self.X <= 0).sum(axis=1, keepdims=True)
        
        # Map nonzero ranks to [1, num_bins]
        shifted_ranks = (ranks - num_zeros).clip(min=0)
        q_bins = (shifted_ranks * num_bins // num_nonzero.clip(min=1)) + 1
        
        # Final bins: 0 for zero, 1..num_bins for expression quantiles
        self.target_bins = np.where(is_nonzero, q_bins, 0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_orig = self.X[idx].copy()
        y_bins = self.target_bins[idx].copy()
        num_genes = x_orig.shape[0]

        # Select genes for masking
        num_mask = max(1, int(num_genes * self.mask_ratio))
        mask_indices = np.random.choice(num_genes, num_mask, replace=False)

        x_masked = x_orig.copy()
        nonzero_vals = x_orig[x_orig > 0]
        probs = np.random.random(num_mask)
        
        # 1. Mask token replacements
        mask_token_mask = probs < self.mask_token_prob
        x_masked[mask_indices[mask_token_mask]] = self.mask_token
        
        # 2. Random value replacements
        random_token_mask = (probs >= self.mask_token_prob) & (probs < (self.mask_token_prob + self.random_token_prob))
        num_random = np.sum(random_token_mask)
        if num_random > 0:
            if len(nonzero_vals) > 0:
                x_masked[mask_indices[random_token_mask]] = np.random.choice(nonzero_vals, size=num_random)
            else:
                x_masked[mask_indices[random_token_mask]] = np.random.uniform(0, 10, size=num_random)

        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(y_bins, dtype=torch.long),
            torch.tensor(mask_indices, dtype=torch.long),
        )

        return (
            torch.tensor(x_masked, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(mask_indices, dtype=torch.long),
        )

def get_sample_indices(batch_dir, train_chunks=None, val_chunks=None, 
                       train_subset=None, val_subset=None, 
                       balanced_sampling=True, seed=42, verbose=True):
    """
    Build sample index lists for train/val from specified parquet chunks.
    
    Args:
        batch_dir: Directory containing *.parquet chunks.
        train_chunks: int (count) or list of indices for training.
        val_chunks: int (count) or list of indices for validation.
        train_subset: optional max samples for train.
        val_subset: optional max samples for val.
        balanced_sampling: if True, balance by species within the selected chunks.
        seed: random seed.
        verbose: print progress.
    """
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    if not batch_files:
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    
    rng = np.random.default_rng(seed)

    # Determine chunk indices
    num_total_chunks = len(batch_files)
    if train_chunks is None:
        # Default: use 80% for train
        train_idxs = list(range(int(0.8 * num_total_chunks)))
    elif isinstance(train_chunks, int):
        train_idxs = list(range(min(train_chunks, num_total_chunks)))
    else:
        train_idxs = train_chunks

    if val_chunks is None:
        # Default: use remaining
        val_idxs = [i for i in range(num_total_chunks) if i not in train_idxs]
    elif isinstance(val_chunks, int):
        # Pick 'val_chunks' count after train_idxs
        remaining = [i for i in range(num_total_chunks) if i not in train_idxs]
        val_idxs = remaining[:val_chunks]
    else:
        val_idxs = val_chunks

    if verbose:
        print(f"[DATA] Chunks: {len(train_idxs)} for training, {len(val_idxs)} for validation")

    # Load metadata (optional)
    metadata_file = batch_dir.parent / "samples.json"
    sample_to_species = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            samples_meta = json.load(f)
        sample_to_species = {s["id"]: s["species"] for s in samples_meta if "species" in s}

    manifest_path = batch_dir.parent / "batch_manifest.json"
    batch_manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            batch_manifest = json.load(f)

    def _collect_samples_from_chunks(chunk_idxs):
        samples = []
        for b_idx in chunk_idxs:
            batch_file = batch_files[b_idx]
            sample_ids = []
            if batch_manifest:
                sample_ids = batch_manifest.get(batch_file.name)
                if sample_ids is None:
                    # try sorted keys fallback
                    keys = sorted(batch_manifest.keys())
                    if b_idx < len(keys):
                        sample_ids = batch_manifest[keys[b_idx]]
            
            if not sample_ids:
                # read parquet index
                pf = pq.ParquetFile(str(batch_file))
                cols = pf.schema_arrow.names
                idx_col = 'geo_accession' if 'geo_accession' in cols else (
                    '__index_level_0__' if '__index_level_0__' in cols else None
                )
                if idx_col:
                    table = pf.read(columns=[idx_col], use_threads=True)
                    sample_ids = table.column(0).to_pylist()
                else:
                    sample_ids = [str(i) for i in range(pf.metadata.num_rows)]
            
            for s_idx, s_id in enumerate(sample_ids):
                species = sample_to_species.get(s_id, "unknown")
                samples.append((b_idx, s_idx, species))
        return samples

    train_all = _collect_samples_from_chunks(train_idxs)
    val_all = _collect_samples_from_chunks(val_idxs)

    def _subset_and_balance(samples, max_count, balance):
        if not samples: return []
        
        # Partition by species
        by_sp = {}
        for b, s, sp in samples:
            by_sp.setdefault(sp, []).append((b, s))
        
        if balance and len(by_sp) > 1:
            per_sp = max_count // len(by_sp) if max_count else min(len(v) for v in by_sp.values())
            balanced = []
            for sp, items in by_sp.items():
                if len(items) > per_sp:
                    selected = rng.choice(len(items), per_sp, replace=False)
                    balanced.extend([items[i] for i in selected])
                else:
                    balanced.extend(items)
            final = balanced
        else:
            # Flatten to (batch, row)
            final = [(b, s) for b, s, sp in samples]
            if max_count and len(final) > max_count:
                selected = rng.choice(len(final), max_count, replace=False)
                final = [final[i] for i in selected]
        
        rng.shuffle(final)
        return final

    train_indices = _subset_and_balance(train_all, train_subset, balanced_sampling)
    val_indices = _subset_and_balance(val_all, val_subset, balanced_sampling)

    if verbose:
        print(f"       Train: {len(train_indices):,} samples from {len(train_idxs)} chunks")
        print(f"       Val:   {len(val_indices):,} samples from {len(val_idxs)} chunks")

    return train_indices, val_indices

def load_batch_data(batch_dir, sample_indices, verbose=True):
    """Load selected samples from parquet chunks into a single numpy array."""
    batch_dir = Path(batch_dir)
    batch_files = sorted(batch_dir.glob("*.parquet"))
    
    from collections import defaultdict
    batch_to_samples = defaultdict(list)
    for idx, (batch_idx, sample_in_batch) in enumerate(sample_indices):
        batch_to_samples[batch_idx].append((idx, sample_in_batch))
    
    if not batch_files:
        raise FileNotFoundError(f"No parquet files in {batch_dir}")

    first_pf = pq.ParquetFile(str(batch_files[0]))
    gene_cols = _parquet_numeric_gene_columns(first_pf.schema_arrow)
    num_genes = len(gene_cols)
    result = np.empty((len(sample_indices), num_genes), dtype=np.float32)
    
    total_batches = len(batch_to_samples)
    for i, (batch_idx, idx_pairs) in enumerate(batch_to_samples.items(), start=1):
        table = pq.read_table(batch_files[batch_idx], columns=gene_cols, use_threads=True)
        cols = [table.column(j).combine_chunks().to_numpy(zero_copy_only=False)
                for j in range(table.num_columns)]
        data = np.stack(cols, axis=1).astype(np.float32, copy=False)
        for out_idx, sample_in_batch in idx_pairs:
            result[out_idx] = data[sample_in_batch]

        if verbose and (i % 25 == 0 or i == total_batches):
            print(f"  ...loaded {i}/{total_batches} selected chunks", flush=True)
    
    if verbose:
        print(f"  ✓ Loaded {result.shape[0]:,} samples × {result.shape[1]:,} genes")
    
    return result

def get_num_genes_from_batches(batch_dir):
    """Infer number of genes from parquet schema."""
    batch_files = sorted(Path(batch_dir).glob("*.parquet"))
    if not batch_files:
        raise FileNotFoundError(f"No parquet files found in {batch_dir}")
    pf = pq.ParquetFile(str(batch_files[0]))
    return len(_parquet_numeric_gene_columns(pf.schema_arrow))
