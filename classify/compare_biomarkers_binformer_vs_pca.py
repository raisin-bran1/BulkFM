import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr
import os

def get_stable_genes(organ, embeddings, labels_df, expression_df, gene_names):
    mask = labels_df['organ'] == organ
    X = embeddings[mask.values]
    y = labels_df[mask]['spaceflight'].values
    groups = labels_df[mask]['osd_batch'].values
    expr = expression_df.loc[mask].values
    
    logo = LeaveOneGroupOut()
    all_coeffs = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            all_coeffs.append(model.coef_[0])
        except:
            continue
            
    if not all_coeffs:
        return None, None
        
    mean_coeffs = np.mean(all_coeffs, axis=0)
    std_coeffs = np.std(all_coeffs, axis=0)
    stability = np.abs(mean_coeffs) / (std_coeffs + 1e-6)
    
    top_dim = np.argsort(stability)[-1]
    
    X_dim = X[:, top_dim]
    
    # Pre-calculate variances to avoid ConstantInputWarning
    gene_variances = np.var(expr, axis=0)
    dim_variance = np.var(X_dim)
    
    corrs = []
    if dim_variance > 0:
        for i in range(len(gene_names)):
            if gene_variances[i] > 0:
                c, _ = pearsonr(X_dim, expr[:, i])
                if not np.isnan(c):
                    corrs.append((gene_names[i], c))
            
    corrs.sort(key=lambda x: x[1], reverse=True)
    return top_dim, corrs

def main():
    bin_path = "osdr/osdr_embeddings.pt"
    pca_path = "osdr/encodings_pca.pt"
    labels_path = "osdr/organ_spaceflight_batch_labels.parquet"
    expr_path = "osdr/osdr_processed_spaceflight.parquet"
    organ = "Thymus"
    
    labels_df = pd.read_parquet(labels_path)
    expr_df = pd.read_parquet(expr_path)
    bin_emb = torch.load(bin_path, map_location='cpu', weights_only=True).float().numpy()
    pca_emb = torch.load(pca_path, map_location='cpu', weights_only=True).float().numpy()
    
    min_len = min(len(labels_df), len(expr_df), len(bin_emb), len(pca_emb))
    labels_df = labels_df.iloc[:min_len]
    expr_df = expr_df.iloc[:min_len]
    bin_emb = bin_emb[:min_len]
    pca_emb = pca_emb[:min_len]
    
    gene_cols = [c for c in expr_df.columns if c.startswith('ENSMUSG')]
    gene_names = gene_cols
    
    print(f"[INFO] Analyzing {organ} biomarkers...")
    bin_dim, bin_corrs = get_stable_genes(organ, bin_emb, labels_df, expr_df[gene_cols], gene_names)
    pca_dim, pca_corrs = get_stable_genes(organ, pca_emb, labels_df, expr_df[gene_cols], gene_names)
    
    print(f"\nTop 10 Positively Correlated Genes (Spaceflight Direction):")
    print(f"{'Rank':<5} | {'Binformer (Dim ' + str(bin_dim) + ')':<25} | {'PCA (Dim ' + str(pca_dim) + ')':<25}")
    print("-" * 65)
    for i in range(10):
        b_gene, b_c = bin_corrs[i]
        p_gene, p_c = pca_corrs[i]
        print(f"{i+1:<5} | {b_gene:<18} ({b_c:>6.3f}) | {p_gene:<18} ({p_c:>6.3f})")

    # Check intersection
    bin_top_100 = set([g for g, c in bin_corrs[:100]])
    pca_top_100 = set([g for g, c in pca_corrs[:100]])
    intersect = bin_top_100.intersection(pca_top_100)
    print(f"\nOverlap in Top 100 Genes: {len(intersect)}%")

if __name__ == "__main__":
    main()
