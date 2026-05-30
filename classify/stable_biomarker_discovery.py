import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import pearsonr
import os

def find_stable_biomarkers(organ, embeddings, labels_df, expression_df):
    print(f"\n--- Analyzing Stable Biomarkers for: {organ} ---")
    
    mask = labels_df['organ'] == organ
    X_org = embeddings[mask.values]
    y_org = labels_df[mask]['spaceflight'].values
    groups_org = labels_df[mask]['osd_batch'].values
    
    # Sync with expression
    expression_data = expression_df.loc[mask].filter(like='ENSMUSG').values
    gene_names = expression_df.filter(like='ENSMUSG').columns.tolist()
    
    logo = LeaveOneGroupOut()
    all_coeffs = []
    
    for train_idx, test_idx in logo.split(X_org, y_org, groups_org):
        X_train, y_train = X_org[train_idx], y_org[train_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train
        model = LogisticRegression(max_iter=1000, random_state=42)
        try:
            model.fit(X_train_scaled, y_train)
            all_coeffs.append(model.coef_[0])
        except:
            continue
            
    if not all_coeffs:
        print(f"[ERROR] Could not train models for {organ}")
        return
        
    # Find stable dimensions (mean weight / std of weight across folds)
    mean_coeffs = np.mean(all_coeffs, axis=0)
    std_coeffs = np.std(all_coeffs, axis=0)
    
    # Stability score: how consistently high is the weight?
    stability_score = np.abs(mean_coeffs) / (std_coeffs + 1e-6)
    
    top_stable_dim = np.argsort(stability_score)[-1]
    mean_w = mean_coeffs[top_stable_dim]
    print(f"Top Stable Dimension: {top_stable_dim} (Stability Score: {stability_score[top_stable_dim]:.2f}, Mean Weight: {mean_w:.4f})")
    
    # Correlate this stable dimension with genes across ALL samples of this organ
    X_dim = X_org[:, top_stable_dim]
    
    # Pre-calculate raw means for verification
    is_flight = y_org == 1
    is_control = y_org == 0
    
    # Pre-calculate variances to avoid ConstantInputWarning
    gene_variances = np.var(expression_data, axis=0)
    dim_variance = np.var(X_dim)
    
    results = []
    if dim_variance > 0:
        for i, gene in enumerate(gene_names):
            if gene_variances[i] > 0:
                corr, _ = pearsonr(X_dim, expression_data[:, i])
                if not np.isnan(corr):
                    # Model Direction: Sign of weight * Sign of correlation
                    # w > 0 means high dim = flight
                    # r > 0 means high gene = high dim
                    # Thus w*r > 0 means high gene = flight (UP)
                    model_dir = "UP" if (mean_w * corr) > 0 else "DOWN"
                    
                    # Raw Direction: Actual mean difference
                    f_mean = expression_data[is_flight, i].mean()
                    c_mean = expression_data[is_control, i].mean()
                    raw_dir = "UP" if f_mean > c_mean else "DOWN"
                    
                    results.append({
                        'gene': gene,
                        'corr': corr,
                        'model_dir': model_dir,
                        'raw_dir': raw_dir,
                        'f_mean': f_mean,
                        'c_mean': c_mean
                    })
            
    results.sort(key=lambda x: abs(x['corr']), reverse=True)
    
    print(f"\nTop 'Stable' Genes for {organ} (linked to Dim {top_stable_dim}):")
    print(f"{'Gene ID':<20} | {'Corr':<8} | {'Model':<6} | {'Raw':<6} | {'Match?'}")
    print("-" * 55)
    for res in results[:10]:
        match = "YES" if res['model_dir'] == res['raw_dir'] else "NO"
        print(f"{res['gene']:<20} | {res['corr']:<8.4f} | {res['model_dir']:<6} | {res['raw_dir']:<6} | {match}")

def main():
    embeddings_path = "osdr/osdr_embeddings_harmony.pt"
    labels_path = "osdr/organ_spaceflight_batch_labels.parquet"
    expression_path = "osdr/osdr_processed_spaceflight.parquet"

    
    embeddings = torch.load(embeddings_path, map_location='cpu', weights_only=True).float().numpy()
    labels_df = pd.read_parquet(labels_path)
    expression_df = pd.read_parquet(expression_path)
    
    min_len = min(len(embeddings), len(labels_df), len(expression_df))
    embeddings = embeddings[:min_len]
    labels_df = labels_df.iloc[:min_len]
    expression_df = expression_df.iloc[:min_len]
    
    find_stable_biomarkers("Thymus", embeddings, labels_df, expression_df)

if __name__ == "__main__":
    main()
