"""
Benchmark cancer type classification using logistic regression.

Evaluates one or more embedding representations on TCGA cancer type
classification. For each embedding set, trains a multi-class logistic
regression classifier and reports accuracy.

Usage:
  python downstream/tcga/classify_cancer_type.py \\
      --embeddings results/embeddings/tcga_pca256.pt \\
      --raw \\
      --embedding_names pca256 raw
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
LABELS_PATH = "embeddings/tcga_labels.parquet"
EMBEDDINGS_DIR = "results/embeddings"


def load_raw_data(parquet_path):
    """Load raw gene expression from TCGA parquet."""
    df = pd.read_parquet(parquet_path)
    meta_cols = [c for c in df.columns if not isinstance(df[c].dtype, pd.Float32Dtype)
                 and df[c].dtype != np.float32]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    X = df[gene_cols].values.astype(np.float32)
    return X, df


def load_embeddings(path):
    """Load embeddings from a .pt file."""
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def evaluate_embeddings(X, y, groups, num_classes, label_encoder, seed=42):
    """Train logistic regression and evaluate with group-stratified split."""
    rng = np.random.RandomState(seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    present = np.unique(y_test)
    target_names = [str(label_encoder.classes_[i]) for i in present]
    report = classification_report(
        y_test, preds, labels=present, target_names=target_names, zero_division=0
    )
    return acc, f1, report, len(X_train), len(X_test), model


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TCGA cancer type classification"
    )
    parser.add_argument(
        "--embeddings", nargs="*", default=[],
        help="Paths to embedding .pt files"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Include raw gene expression from TCGA parquet"
    )
    parser.add_argument(
        "--embedding_names", nargs="*", default=[],
        help="Display names for embeddings (must match --embeddings order)"
    )
    parser.add_argument(
        "--labels", type=str, default=LABELS_PATH,
        help="Path to labels parquet"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    if not args.embeddings and not args.raw:
        print("ERROR: provide at least one --embeddings path or --raw", file=sys.stderr)
        sys.exit(1)

    if args.embedding_names and len(args.embedding_names) != len(args.embeddings):
        print("ERROR: --embedding_names must match --embeddings count", file=sys.stderr)
        sys.exit(1)

    names = list(args.embedding_names)
    if args.raw:
        names.append("raw")

    print("=== Loading labels ===")
    label_df = pd.read_parquet(args.labels)
    print(f"  Labels: {len(label_df)} samples")

    le = LabelEncoder()
    y = le.fit_transform(label_df["cancertype"].values)
    num_classes = len(le.classes_)
    print(f"  Cancer types: {num_classes}")
    print(f"  Classes: {list(le.classes_)}")

    groups = label_df["gdc_cases.submitter_id"].values
    print(f"  Unique patients (groups): {len(np.unique(groups))}")

    results = []

    emb_idx = 0
    for i, emb_path in enumerate(args.embeddings):
        display_name = names[i] if names else os.path.splitext(os.path.basename(emb_path))[0]
        print(f"\n{'='*60}")
        print(f"[{display_name}] Loading embeddings from {emb_path}")
        X = load_embeddings(emb_path)
        assert len(X) == len(y), f"Shape mismatch: {len(X)} vs {len(y)}"
        acc, f1, report, n_train, n_test, _ = evaluate_embeddings(
            X, y, groups, num_classes, le, args.seed
        )
        print(f"  Train: {n_train}, Test: {n_test}")
        print(f"  Accuracy: {acc*100:.2f}% | Weighted F1: {f1*100:.2f}%")
        results.append({"Embedding": display_name, "Accuracy": acc, "Weighted F1": f1, "Train": n_train, "Test": n_test})

    if args.raw:
        display_name = names[-1] if args.embedding_names else "raw"
        print(f"\n{'='*60}")
        print(f"[{display_name}] Loading raw gene expression")
        X_raw, _ = load_raw_data(TCGA_PARQUET)
        assert len(X_raw) == len(y), f"Shape mismatch: {len(X_raw)} vs {len(y)}"
        acc, f1, report, n_train, n_test, _ = evaluate_embeddings(
            X_raw, y, groups, num_classes, le, args.seed
        )
        print(f"  Train: {n_train}, Test: {n_test}")
        print(f"  Accuracy: {acc*100:.2f}% | Weighted F1: {f1*100:.2f}%")
        results.append({"Embedding": display_name, "Accuracy": acc, "Weighted F1": f1, "Train": n_train, "Test": n_test})

    print(f"\n{'='*72}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*72}")
    summary = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    summary["Accuracy"] = summary["Accuracy"].map("{:.2%}".format)
    summary["Weighted F1"] = summary["Weighted F1"].map("{:.2%}".format)
    print(summary.to_string(index=False))
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
