import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from collections import Counter
import os
import sys

# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if root_path not in sys.path:
    sys.path.append(root_path)

def main():
    embeddings_path = "data/osdr/osdr_embeddings_harmony.pt"
    labels_path = "data/osdr/organ_spaceflight_batch_labels.parquet"
    
    print(f"[INFO] Loading data for LOGO Cross-Validation...")
    embeddings = torch.load(embeddings_path, map_location='cpu', weights_only=True).float().numpy()
    labels_df = pd.read_parquet(labels_path)
    
    # Sync
    min_len = min(len(embeddings), len(labels_df))
    embeddings = embeddings[:min_len]
    labels_df = labels_df.iloc[:min_len]
    
    organs = labels_df['organ'].unique()
    
    results = []
    
    print(f"\n{'Tissue':<25} | {'Batches':<8} | {'Samples':<8} | {'Mean Acc':<10} | {'Std Dev'}")
    print("-" * 75)
    
    for organ in organs:
        mask = labels_df['organ'] == organ
        X_org = embeddings[mask.values]
        y_org = labels_df[mask]['spaceflight'].values
        groups_org = labels_df[mask]['osd_batch'].values
        
        # Check if we have enough classes and groups
        unique_y = np.unique(y_org)
        unique_groups = np.unique(groups_org)
        
        if len(unique_y) < 2 or len(unique_groups) < 2:
            continue
            
        # Use full organ data (no balancing)
        X_final = X_org
        y_final = y_org
        groups_final = groups_org

        logo = LeaveOneGroupOut()
        fold_accs = []
        
        for train_idx, test_idx in logo.split(X_final, y_final, groups_final):
            X_train, X_test = X_final[train_idx], X_final[test_idx]
            y_train, y_test = y_final[train_idx], y_final[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train Logistic Regression
            model = LogisticRegression(max_iter=1000, random_state=42)
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                fold_accs.append(accuracy_score(y_test, preds))
            except ValueError:
                # Occurs if a fold has only one class in training
                continue
        
        if fold_accs:
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            print(f"{organ:<25} | {len(unique_groups):<8} | {len(y_final):<8} | {mean_acc*100:<9.2f}% | {std_acc*100:.2f}%")
            
            results.append({
                'Tissue': organ,
                'Batches': len(unique_groups),
                'Samples': len(y_final),
                'Mean_Accuracy': mean_acc,
                'Std_Deviation': std_acc
            })

    results_df = pd.DataFrame(results).sort_values(by='Mean_Accuracy', ascending=False)
    results_df.to_csv('results/plots/logo_cv_results.csv', index=False)
    print("\n[INFO] LOGO CV results saved to results/plots/logo_cv_results.csv")

if __name__ == "__main__":
    main()
