"""
Imputation benchmark: predict expression from embeddings via Ridge regression.

Runs on every .pt embedding file in EMBEDDINGS_DIR.

Usage:
  python downstream/tcga/imputation.py
  python downstream/tcga/imputation.py --embedding_dir path/to/embeddings
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
LABELS_PATH = "embeddings/tcga_labels.parquet"
EMBEDDINGS_DIR = "embeddings"


def load_raw_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    meta_cols = [c for c in df.columns if not isinstance(df[c].dtype, pd.Float32Dtype)
                 and df[c].dtype != np.float32]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    X = df[gene_cols].values.astype(np.float32)
    return X, df, gene_cols


def load_embeddings(path):
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def global_pearson(true, pred):
    t = true.ravel()
    p = pred.ravel()
    if np.std(t) == 0 or np.std(p) == 0:
        return 0.0
    r, _ = pearsonr(t, p)
    return r if not np.isnan(r) else 0.0


def per_gene_pearson(true, pred):
    t = torch.from_numpy(true)
    p = torch.from_numpy(pred)
    t_c = t - t.mean(dim=0, keepdim=True)
    p_c = p - p.mean(dim=0, keepdim=True)
    r_num = (t_c * p_c).sum(dim=0)
    r_den = torch.sqrt((t_c ** 2).sum(dim=0) * (p_c ** 2).sum(dim=0))
    r = r_num / r_den.clamp(min=1e-8)
    return r.nanmean().item()


def per_sample_pearson(true, pred):
    t = torch.from_numpy(true)
    p = torch.from_numpy(pred)
    t_c = t - t.mean(dim=1, keepdim=True)
    p_c = p - p.mean(dim=1, keepdim=True)
    r_num = (t_c * p_c).sum(dim=1)
    r_den = torch.sqrt((t_c ** 2).sum(dim=1) * (p_c ** 2).sum(dim=1))
    r = r_num / r_den.clamp(min=1e-8)
    return r.nanmean().item()


class EmbeddingImputer:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.ridge = None
        self.scaler = StandardScaler()

    def fit(self, X_train, embeddings_train):
        self.ridge = Ridge(alpha=self.alpha, random_state=42, solver="svd")
        emb_scaled = self.scaler.fit_transform(embeddings_train)
        self.ridge.fit(emb_scaled, X_train)

    def predict(self, embeddings_test):
        emb_scaled = self.scaler.transform(embeddings_test)
        return self.ridge.predict(emb_scaled)


def main():
    warnings.filterwarnings("ignore", message="An input array is constant")
    warnings.filterwarnings("ignore", message="Ill-conditioned matrix")
    warnings.filterwarnings("ignore", message="LinAlgWarning")
    parser = argparse.ArgumentParser(description="TCGA imputation benchmark")
    parser.add_argument("--embedding_dir", type=str, default=EMBEDDINGS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Loading TCGA data ===")
    X, df, gene_cols = load_raw_data(TCGA_PARQUET)
    print(f"  Samples: {X.shape[0]}, Genes: {X.shape[1]}")

    label_df = pd.read_parquet(LABELS_PATH)
    groups = label_df["gdc_cases.submitter_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(gss.split(X, np.zeros(len(X)), groups))
    X_train, X_test = X[train_idx], X[test_idx]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    results = []

    # ── Embedding-based imputation ──
    emb_paths = sorted(glob.glob(os.path.join(args.embedding_dir, "*.pt")))
    emb_paths = [p for p in emb_paths if not os.path.basename(p).startswith("tcga_labels")]

    for emb_path in emb_paths:
        name = os.path.splitext(os.path.basename(emb_path))[0]
        print(f"\n--- {name} (Ridge regression) ---")

        embeddings = load_embeddings(emb_path)
        assert len(embeddings) == len(X), f"Shape mismatch: {len(embeddings)} vs {len(X)}"

        emb_train = embeddings[train_idx]
        emb_test = embeddings[test_idx]

        imputer = EmbeddingImputer(alpha=1.0)
        imputer.fit(X_train, emb_train)
        X_pred = imputer.predict(emb_test)

        results.append({
            "Method": f"{name}",
            "Global PCC": global_pearson(X_test, X_pred),
            "Per-gene PCC": per_gene_pearson(X_test, X_pred),
            "Per-sample PCC": per_sample_pearson(X_test, X_pred),
            "MSE": float(np.mean((X_test - X_pred) ** 2)),
        })
        r = results[-1]
        print(f"  Global PCC:     {r['Global PCC']:.4f}")
        print(f"  Per-gene PCC:   {r['Per-gene PCC']:.4f}")
        print(f"  Per-sample PCC: {r['Per-sample PCC']:.4f}")
        print(f"  MSE:            {r['MSE']:.6f}")

    # ── Summary ──
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/imputation_results.csv", index=False)
    print(f"\n{'='*72}")
    print("IMPUTATION BENCHMARK SUMMARY")
    print(f"{'='*72}")
    print(f"{'Method':<15} {'Global PCC':<12} {'Gene PCC':<12} {'Sample PCC':<12} {'MSE':<10}")
    print("-" * 61)
    for r in results:
        print(f"{r['Method']:<15} {r['Global PCC']:<12.4f} {r['Per-gene PCC']:<12.4f} {r['Per-sample PCC']:<12.4f} {r['MSE']:<10.6f}")
    print(f"{'='*72}")
    print(f"Results saved to results/imputation_results.csv")


if __name__ == "__main__":
    main()
