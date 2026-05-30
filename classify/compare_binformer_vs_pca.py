# DOES NOT ENFORCE DOWNSAMPLING 

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import os

def run_logo_cv(embeddings, labels_df, organ):
    mask = labels_df['organ'] == organ
    X = embeddings[mask.values]
    y = labels_df[mask]['spaceflight'].values
    groups = labels_df[mask]['osd_batch'].values
    
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return None
        
    logo = LeaveOneGroupOut()
    fold_accs = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_accs.append(accuracy_score(y_test, preds))
        except:
            continue
            
    return np.mean(fold_accs) if fold_accs else None

def main():
    binformer_path = "osdr/osdr_embeddings.pt"
    pca_path = "osdr/encodings_pca.pt"
    labels_path = "osdr/organ_spaceflight_batch_labels.parquet"
    
    labels_df = pd.read_parquet(labels_path)
    bin_emb = torch.load(binformer_path, map_location='cpu', weights_only=True).float().numpy()
    pca_emb = torch.load(pca_path, map_location='cpu', weights_only=True).float().numpy()
    
    # Sync
    min_len = min(len(labels_df), len(bin_emb), len(pca_emb))
    labels_df = labels_df.iloc[:min_len]
    bin_emb = bin_emb[:min_len]
    pca_emb = pca_emb[:min_len]
    
    organs = ["Thymus", "Heart", "Skin", "Brain", "Soleus"]
    
    print(f"{'Tissue':<20} | {'Binformer Acc':<15} | {'PCA Acc':<15} | {'Delta'}")
    print("-" * 60)
    
    for organ in organs:
        bin_acc = run_logo_cv(bin_emb, labels_df, organ)
        pca_acc = run_logo_cv(pca_emb, labels_df, organ)
        
        if bin_acc is not None and pca_acc is not None:
            delta = bin_acc - pca_acc
            print(f"{organ:<20} | {bin_acc*100:<14.2f}% | {pca_acc*100:<14.2f}% | {delta*100:+.2f}%")

if __name__ == "__main__":
    main()
