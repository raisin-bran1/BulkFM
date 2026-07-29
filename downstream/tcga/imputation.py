"""
Imputation benchmark: mask genes, predict their expression.

Usage:
  python downstream/tcga/imputation.py --seed 42
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import pearsonr

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
LABELS_PATH = "embeddings/tcga_labels.parquet"


def load_raw_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    meta_cols = [c for c in df.columns if not isinstance(df[c].dtype, pd.Float32Dtype)
                 and df[c].dtype != np.float32]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    X = df[gene_cols].values.astype(np.float32)
    return X, df, gene_cols


def mask_expression(X, mask_ratio=0.3, seed=42):
    rng = np.random.RandomState(seed)
    B, G = X.shape
    mask = rng.rand(B, G) < mask_ratio
    X_masked = X.copy()
    X_masked[mask] = 0.0
    return X_masked, mask


def global_pearson(true, pred, mask):
    """Single Pearson correlation over all masked entries (flattened)."""
    t = true[mask]
    p = pred[mask]
    if np.std(t) == 0 or np.std(p) == 0:
        return 0.0
    r, _ = pearsonr(t, p)
    return r if not np.isnan(r) else 0.0


def per_gene_pearson(true, pred, mask):
    """Average per-gene Pearson (each gene weighted equally)."""
    G = true.shape[1]
    r_values = []
    for g in range(G):
        masked = mask[:, g]
        if masked.sum() < 3:
            continue
        t = true[masked, g]
        p = pred[masked, g]
        if np.std(t) == 0 or np.std(p) == 0:
            continue
        r, _ = pearsonr(t, p)
        if not np.isnan(r):
            r_values.append(r)
    return np.mean(r_values) if r_values else 0.0


def per_sample_pearson(true, pred, mask):
    """Average per-sample Pearson (each sample weighted equally)."""
    B = true.shape[0]
    r_values = []
    for b in range(B):
        masked = mask[b, :]
        if masked.sum() < 3:
            continue
        t = true[b, masked]
        p = pred[b, masked]
        if np.std(t) == 0 or np.std(p) == 0:
            continue
        r, _ = pearsonr(t, p)
        if not np.isnan(r):
            r_values.append(r)
    return np.mean(r_values) if r_values else 0.0


def mse_masked(true, pred, mask):
    return float(np.mean((true[mask] - pred[mask]) ** 2))


class MeanBaseline:
    def __init__(self):
        self.gene_means = None

    def fit(self, X_train):
        self.gene_means = X_train.mean(axis=0)

    def predict(self, X_masked):
        return np.broadcast_to(self.gene_means, X_masked.shape)


class PCABaseline:
    def __init__(self, n_components=256):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit(self, X_train):
        self.pca.fit(X_train)

    def predict(self, X_masked):
        return self.pca.inverse_transform(self.pca.transform(X_masked))


def main():
    warnings.filterwarnings("ignore", message="An input array is constant")
    parser = argparse.ArgumentParser(description="TCGA imputation benchmark")
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_baselines", action="store_true")
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

    print(f"\n=== Masking {args.mask_ratio*100:.0f}% of genes per sample ===")
    X_masked, mask = mask_expression(X_test, mask_ratio=args.mask_ratio, seed=args.seed)
    print(f"  Masked: {mask.mean()*100:.1f}% of entries ({mask.sum():,} positions)")
    X_true = X_test

    results = []

    if not args.no_baselines:
        for name, bline in [("Mean", MeanBaseline()), ("PCA-256", PCABaseline(256))]:
            print(f"\n--- {name} ---")
            bline.fit(X_train)
            X_pred = bline.predict(X_masked)
            results.append({
                "Method": name,
                "Global PCC": global_pearson(X_true, X_pred, mask),
                "Per-gene PCC": per_gene_pearson(X_true, X_pred, mask),
                "Per-sample PCC": per_sample_pearson(X_true, X_pred, mask),
                "MSE": mse_masked(X_true, X_pred, mask),
            })
            r = results[-1]
            print(f"  Global PCC:     {r['Global PCC']:.4f}  (flattened, single r)")
            print(f"  Per-gene PCC:   {r['Per-gene PCC']:.4f}  (avg over 18,819 genes)")
            print(f"  Per-sample PCC: {r['Per-sample PCC']:.4f}  (avg over {len(X_test)} samples)")
            print(f"  MSE:            {r['MSE']:.6f}")

    print(f"\n{'='*72}")
    print("IMPUTATION BENCHMARK SUMMARY")
    print(f"{'='*72}")
    cols = ["Method", "Global PCC", "Per-gene PCC", "Per-sample PCC", "MSE"]
    print(f"{'Method':<12} {'Global PCC':<12} {'Gene PCC':<12} {'Sample PCC':<12} {'MSE':<10}")
    print("-" * 58)
    for r in results:
        print(f"{r['Method']:<12} {r['Global PCC']:<12.4f} {r['Per-gene PCC']:<12.4f} {r['Per-sample PCC']:<12.4f} {r['MSE']:<10.6f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
