import torch
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# Example usage:
# python classify_embeddings/plotter.py --column organ --type pca --output results/plots/osdr_pca_organ.png
# python classify_embeddings/plotter.py --column spaceflight --type tsne --output results/plots/spaceflight_tsne.png

def parse_args():
    parser = argparse.ArgumentParser(description="Plot OSDR Embeddings")
    parser.add_argument("--embeddings", type=str, default="data/osdr/osdr_embeddings.pt", help="Path to embeddings .pt file")
    parser.add_argument("--labels", type=str, default="data/osdr/organ_spaceflight_batch_labels.parquet", help="Path to labels parquet")
    parser.add_argument("--column", type=str, default="spaceflight", help="Column name to color by")
    parser.add_argument("--type", type=str, default="pca", choices=["pca", "tsne", "umap"], help="Type of plot: pca, tsne, umap")
    parser.add_argument("--output", type=str, default=None, help="Path to save the plot (e.g., results/plots/plot.png)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_scale", action="store_true", help="Disable scaling of embeddings before reduction")
    parser.add_argument("--filter_organ", type=str, default=None, help="Only include samples from this organ (e.g., 'Soleus')")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist parameter")
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    print(f"[INFO] Loading embeddings from {args.embeddings}")
    if not os.path.exists(args.embeddings):
        print(f"[ERROR] Embeddings file not found: {args.embeddings}")
        return
    embeddings = torch.load(args.embeddings, map_location='cpu', weights_only=True).float().numpy()
    
    print(f"[INFO] Loading labels from {args.labels}")
    if not os.path.exists(args.labels):
        print(f"[ERROR] Labels file not found: {args.labels}")
        return
    labels_df = pd.read_parquet(args.labels)
    
    # Handle mismatches
    if len(embeddings) != len(labels_df):
        print(f"[WARNING] Dimension mismatch: Embeddings ({len(embeddings)}) vs Labels ({len(labels_df)})")
        min_len = min(len(embeddings), len(labels_df))
        embeddings = embeddings[:min_len]
        labels_df = labels_df.iloc[:min_len]
    
    if args.column not in labels_df.columns:
        print(f"[ERROR] Column '{args.column}' not found in labels. Available: {labels_df.columns.tolist()}")
        return

    # Filter by organ if requested
    if args.filter_organ:
        if 'organ' not in labels_df.columns:
            print(f"[ERROR] 'organ' column not found. Available: {labels_df.columns.tolist()}")
            return
        mask = labels_df['organ'] == args.filter_organ
        if not mask.any():
            print(f"[ERROR] No samples found for organ: '{args.filter_organ}'")
            print(f"Available organs: {labels_df['organ'].unique().tolist()}")
            return
        embeddings = embeddings[mask.values]
        labels_df = labels_df[mask].reset_index(drop=True)
        print(f"[INFO] Filtered to organ: '{args.filter_organ}' ({len(labels_df)} samples remaining)")

    y = labels_df[args.column].values
    
    # Scale embeddings
    if not args.no_scale:
        print("[INFO] Scaling embeddings...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    print(f"[INFO] Reducing dimensions using {args.type.upper()}...")
    if args.type == "pca":
        reducer = PCA(n_components=2, random_state=args.seed)
        reduced = reducer.fit_transform(embeddings)
    elif args.type == "tsne":
        reducer = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto')
        reduced = reducer.fit_transform(embeddings)
    elif args.type == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2, 
                random_state=args.seed,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist
            )
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            print("[ERROR] 'umap-learn' is not installed. Please install it with: pip install umap-learn")
            return
            
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(y)
    
    # Use a colormap for better visuals if there are many labels
    cmap = plt.get_cmap('tab20')
    
    for i, label in enumerate(unique_labels):
        mask = (y == label)
        plt.scatter(
            reduced[mask, 0], 
            reduced[mask, 1], 
            label=label, 
            alpha=0.7, 
            s=15, 
            color=cmap(i % 20)
        )
        
    plt.title(f"{args.type.upper()} Visualization of OSDR Embeddings\nColored by: {args.column} (Organ: {args.filter_organ if args.filter_organ else 'All'})")
    plt.xlabel(f"{args.type.upper()} 1")
    plt.ylabel(f"{args.type.upper()} 2")
    
    # Move legend outside
    plt.legend(title=args.column, bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output, dpi=300)
        print(f"[INFO] Plot saved to {args.output}")
    else:
        # Since we might be in a headless environment, saving a default file if show() might fail
        try:
            plt.show()
        except Exception as e:
            # Create plots directory if it doesn't exist
            if not os.path.exists('results/plots'):
                os.makedirs('results/plots')
            default_out = f"results/plots/osdr_{args.type}_{args.column}_{args.filter_organ if args.filter_organ else 'all'}.png"
            plt.savefig(default_out, dpi=300)
            print(f"[WARNING] Could not show plot ({e}). Saved to {default_out} instead.")

if __name__ == "__main__":
    main()
