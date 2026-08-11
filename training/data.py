# coding=utf-8
# Copyright 2026 The Google Research Authors.

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq


def _list_expression_files(batch_dir):
    """List parquet files that look like processed expression matrices (samples × genes).

    Excludes intermediate preprocessing files (e.g., ``input_chunk_*`` which have
    genes as rows and samples as columns).
    """
    files = sorted(Path(batch_dir).glob("*.parquet"))
    files = [f for f in files if not f.name.startswith('input_chunk_')]
    return files

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
    def __init__(self, expr_array, mask_ratio=0.15, mask_token=-10,
                 mask_token_prob=0.8, random_token_prob=0.1, num_bins=50,
                 expression_embedding='binned', masking_strategy='mask_token',
                 dynamic_mask_range=None):
        self.X = expr_array.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.dynamic_mask_range = dynamic_mask_range
        self.mask_token = mask_token
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.expression_embedding = expression_embedding
        self.masking_strategy = masking_strategy

        B, G = self.X.shape
        if expression_embedding == 'binned':
            print(f"[DATA] Pre-calculating quantile bins for {B} samples...")
            is_nonzero = self.X > 0
            num_nonzero = is_nonzero.sum(axis=1, keepdims=True)
            ranked_x = self.X.copy()
            ranked_x[~is_nonzero] = -1e9
            ranks = np.argsort(np.argsort(ranked_x, axis=1), axis=1)
            num_zeros = (self.X <= 0).sum(axis=1, keepdims=True)
            shifted_ranks = (ranks - num_zeros).clip(min=0)
            q_bins = (shifted_ranks * num_bins // num_nonzero.clip(min=1)) + 1
            self.target_bins = np.where(is_nonzero, q_bins, 0).astype(np.int64)
        else:
            self.target_bins = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_orig = self.X[idx].copy()
        num_genes = x_orig.shape[0]

        if self.dynamic_mask_range is not None:
            # Training loop handles masking dynamically; return raw input
            x_masked = x_orig.copy()
            mask_indices = np.array([], dtype=np.int64)
        else:
            num_mask = max(1, int(num_genes * self.mask_ratio))
            mask_indices = np.random.choice(num_genes, num_mask, replace=False)
            if self.masking_strategy == 'mask_token':
                x_masked = x_orig.copy()
                nonzero_vals = x_orig[x_orig > 0]
                probs = np.random.random(num_mask)
                mask_token_mask = probs < self.mask_token_prob
                x_masked[mask_indices[mask_token_mask]] = self.mask_token
                random_token_mask = (probs >= self.mask_token_prob) & \
                    (probs < (self.mask_token_prob + self.random_token_prob))
                num_random = np.sum(random_token_mask)
                if num_random > 0:
                    if len(nonzero_vals) > 0:
                        x_masked[mask_indices[random_token_mask]] = \
                            np.random.choice(nonzero_vals, size=num_random)
                    else:
                        x_masked[mask_indices[random_token_mask]] = \
                            np.random.uniform(0, 10, size=num_random)
            else:
                x_masked = x_orig.copy()

        if self.expression_embedding == 'binned':
            target = self.target_bins[idx].copy()
            target = torch.tensor(target, dtype=torch.long)
        else:
            target = torch.tensor(x_orig, dtype=torch.float32)

        return (
            torch.tensor(x_masked, dtype=torch.float32),
            target,
            torch.tensor(mask_indices, dtype=torch.long),
        )

def apply_dynamic_mask(x_input, ratio, masking_strategy='mask_token',
                       mask_token=-10, mask_token_prob=0.8,
                       random_token_prob=0.1):
    """Mask ``ratio`` of genes per sample with a per-sample permutation.

    Returns (x_input_masked, mask_idx) where mask_idx[i] holds the gene
    columns that were masked for sample i. The input is modified **exactly**
    at the positions in mask_idx: genes drawn to be masked are replaced with
    ``mask_token`` (with prob ``mask_token_prob``) or a random non-zero value
    from the batch (with prob ``random_token_prob``); the remaining masked
    genes are left untouched (i.e., "keep").
    """
    B, G = x_input.shape
    num_mask = max(1, int(G * ratio))
    # Per-sample random permutation (argsort of iid uniforms).
    idxs = torch.argsort(torch.rand(B, G, device=x_input.device), dim=1)
    mask_idx = idxs[:, :num_mask]

    if masking_strategy == 'mask_token':
        x_input = x_input.clone().contiguous()
        flat = x_input.view(-1)
        probs = torch.rand(B, num_mask, device=x_input.device)
        is_mask = probs < mask_token_prob
        is_rand = (probs >= mask_token_prob) & (probs < mask_token_prob + random_token_prob)
        mask_pos = is_mask.nonzero()
        if len(mask_pos) > 0:
            flat[mask_pos[:, 0] * G + mask_idx[mask_pos[:, 0], mask_pos[:, 1]]] = mask_token
        rand_pos = is_rand.nonzero()
        if len(rand_pos) > 0:
            nonzero = x_input[x_input > 0]
            if len(nonzero) > 0:
                rand_vals = nonzero[torch.randint(len(nonzero), (len(rand_pos),), device=x_input.device)]
            else:
                rand_vals = torch.empty(len(rand_pos), device=x_input.device).uniform_(0, 10)
            flat[rand_pos[:, 0] * G + mask_idx[rand_pos[:, 0], rand_pos[:, 1]]] = rand_vals
    else:
        x_input = x_input.clone()

    return x_input, mask_idx


def get_sample_indices(batch_dir, train_chunks=None, val_chunks=None,
                       train_subset=None, val_subset=None,
                       seed=42, verbose=True):
    batch_dir = Path(batch_dir)
    batch_files = _list_expression_files(batch_dir)
    if not batch_files:
        raise FileNotFoundError(f"No expression parquet files found in {batch_dir}")

    rng = np.random.default_rng(seed)

    num_total_chunks = len(batch_files)
    if train_chunks is None:
        train_idxs = list(range(int(0.8 * num_total_chunks)))
    elif isinstance(train_chunks, int):
        train_idxs = list(range(min(train_chunks, num_total_chunks)))
    else:
        train_idxs = train_chunks

    if val_chunks is None:
        val_idxs = [i for i in range(num_total_chunks) if i not in train_idxs]
    elif isinstance(val_chunks, int):
        remaining = [i for i in range(num_total_chunks) if i not in train_idxs]
        val_idxs = remaining[:val_chunks]
    else:
        val_idxs = val_chunks

    if verbose:
        print(f"[DATA] Chunks: {len(train_idxs)} for training, {len(val_idxs)} for validation")

    def _collect_samples(chunk_idxs):
        samples = []
        for b_idx in chunk_idxs:
            batch_file = batch_files[b_idx]
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

            for s_idx in range(len(sample_ids)):
                samples.append((b_idx, s_idx))
        return samples

    train_all = _collect_samples(train_idxs)
    val_all = _collect_samples(val_idxs)

    def _subset_and_shuffle(samples, max_count):
        if not samples:
            return []
        if max_count and len(samples) > max_count:
            selected = rng.choice(len(samples), max_count, replace=False)
            samples = [samples[i] for i in selected]
        rng.shuffle(samples)
        return samples

    train_indices = _subset_and_shuffle(train_all, train_subset)
    val_indices = _subset_and_shuffle(val_all, val_subset)

    if verbose:
        print(f"       Train: {len(train_indices):,} samples from {len(train_idxs)} chunks")
        print(f"       Val:   {len(val_indices):,} samples from {len(val_idxs)} chunks")

    return train_indices, val_indices

def load_batch_data(batch_dir, sample_indices, verbose=True):
    """Load selected samples from parquet chunks into a single numpy array."""
    batch_dir = Path(batch_dir)
    batch_files = _list_expression_files(batch_dir)
    
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
    batch_files = _list_expression_files(batch_dir)
    if not batch_files:
        raise FileNotFoundError(f"No expression parquet files found in {batch_dir}")
    pf = pq.ParquetFile(str(batch_files[0]))
    return len(_parquet_numeric_gene_columns(pf.schema_arrow))


def group_indices_by_chunk(sample_indices, chunks_in_memory=None):
    """Split sample indices into groups, each spanning up to ``chunks_in_memory`` chunks.

    Returns a list of lists of ``(batch_idx, sample_in_batch)`` pairs (the same
    structure produced by :func:`get_sample_indices`). Groups are ordered by
    chunk index so that data can be loaded and trained on incrementally.
    If ``chunks_in_memory`` is None, all chunks go into a single group.
    """
    from collections import OrderedDict

    by_chunk = OrderedDict()
    for entry in sample_indices:
        batch_idx = entry[0]
        by_chunk.setdefault(batch_idx, []).append(entry)

    chunk_order = sorted(by_chunk.keys())
    if chunks_in_memory is None or chunks_in_memory <= 0:
        chunks_in_memory = len(chunk_order)

    groups = []
    for i in range(0, len(chunk_order), chunks_in_memory):
        group = []
        for c in chunk_order[i:i + chunks_in_memory]:
            group.extend(by_chunk[c])
        groups.append(group)
    return groups


def get_gene_vocabulary(batch_dir):
    """Return list of gene column names from parquet schema."""
    batch_files = _list_expression_files(batch_dir)
    if not batch_files:
        raise FileNotFoundError(f"No expression parquet files found in {batch_dir}")
    pf = pq.ParquetFile(str(batch_files[0]))
    return _parquet_numeric_gene_columns(pf.schema_arrow)
