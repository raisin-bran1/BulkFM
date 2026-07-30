"""
Prepare TCGA embeddings for downstream classification benchmarking.

Reads:
  /media/volume/bulkrnadata/tcgadata/tcga_processed.parquet

Creates:
  results/embeddings/tcga_raw.pt         - Full gene expression [N, 18819]
  results/embeddings/tcga_pca256.pt      - 256-dim PCA reduction [N, 256]
  results/embeddings/tcga_labels.parquet - Metadata (cancertype, submitter_id, etc.)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
OUT_DIR = "results/embeddings"

def main():
    parser = argparse.ArgumentParser(description="Prepare TCGA embeddings")
    parser.add_argument("--tcga_parquet", type=str, default=TCGA_PARQUET)
    parser.add_argument("--outdir", type=str, default=OUT_DIR)
    parser.add_argument("--n_pca", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=== Loading TCGA processed parquet ===")
    df = pd.read_parquet(args.tcga_parquet)
    print(f"  Shape: {df.shape}")

    meta_cols = [c for c in df.columns if not isinstance(df[c].dtype, pd.Float32Dtype)
                 and df[c].dtype != np.float32]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Gene features: {len(gene_cols)}, Metadata columns: {len(meta_cols)}")

    X = df[gene_cols].values.astype(np.float32)
    print(f"  Expression tensor: {X.shape}")

    print("=== Saving raw embeddings ===")
    torch.save(torch.from_numpy(X), os.path.join(args.outdir, "tcga_raw.pt"))
    print(f"  Saved tcga_raw.pt")

    print(f"=== Fitting PCA ({args.n_pca} components) ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(args.n_pca, X.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    print(f"  PCA shape: {X_pca.shape}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    torch.save(torch.from_numpy(X_pca), os.path.join(args.outdir, f"tcga_pca{args.n_pca}.pt"))
    print(f"  Saved tcga_pca{args.n_pca}.pt")

    print("=== Saving labels ===")
    label_df = df[meta_cols].copy()
    label_df.to_parquet(os.path.join(args.outdir, "tcga_labels.parquet"))
    print(f"  Saved tcga_labels.parquet ({len(label_df)} rows)")

    print("Done.")

if __name__ == "__main__":
    main()
