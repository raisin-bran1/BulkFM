"""
Generate 2D UMAP plots for each embedding file (excluding raw), colored by cancer type.

Usage:
  python downstream/tcga/visualize_embeddings.py
"""

import argparse
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import umap
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_DIR = "embeddings"
LABELS_PATH = "embeddings/tcga_labels.parquet"
OUTPUT_DIR = "results"


def load_embeddings(path):
    return torch.load(path, map_location="cpu", weights_only=True).float().numpy()


def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of TCGA embeddings")
    parser.add_argument("--embedding_dir", type=str, default=EMBEDDINGS_DIR)
    parser.add_argument("--labels", type=str, default=LABELS_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legend", action="store_true",
                        help="Show legend (hidden by default)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_df = pd.read_parquet(args.labels)
    cancer_types = label_df["cancertype"].values
    unique_cancers = sorted(pd.unique(cancer_types))
    color_map = plt.cm.tab20
    cancer_to_color = {c: color_map(i / len(unique_cancers))
                       for i, c in enumerate(unique_cancers)}

    emb_paths = sorted(glob.glob(os.path.join(args.embedding_dir, "*.pt")))
    emb_paths = [p for p in emb_paths
                 if not os.path.basename(p).startswith("tcga_labels")]
    emb_paths = [p for p in emb_paths if "raw" not in os.path.basename(p).lower()]
    if not emb_paths:
        print(f"No embedding .pt files found in {args.embedding_dir}", file=sys.stderr)
        sys.exit(1)

    for emb_path in emb_paths:
        name = os.path.splitext(os.path.basename(emb_path))[0]
        out_path = os.path.join(args.output_dir, f"umap_{name}.png")
        if os.path.exists(out_path):
            print(f"  Skipping {name} (already exists)")
            continue
        print(f"Plotting {name}...")

        X = load_embeddings(emb_path)
        if X.shape[0] != len(cancer_types):
            print(f"  Skip: shape mismatch {X.shape[0]} vs {len(cancer_types)} labels")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist, random_state=args.seed)
        X_2d = reducer.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(11, 9))
        for ct in unique_cancers:
            mask = cancer_types == ct
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[cancer_to_color[ct]], label=ct, s=5, alpha=0.6)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{name} — UMAP (dim={X.shape[1]})")
        if args.legend:
            ax.legend(markerscale=3, fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        print("Done.")


if __name__ == "__main__":
    main()
