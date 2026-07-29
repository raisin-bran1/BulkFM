import pandas as pd
import torch
import numpy as np

def check_alignment():
    print("--- Checking Alignment ---")
    
    # 1. Parquet Files
    df_data = pd.read_parquet('data/osdr/osdr_processed_spaceflight.parquet')
    df_labels = pd.read_parquet('data/osdr/organ_spaceflight_batch_labels.parquet')
    
    print(f"Data Parquet samples: {len(df_data)}")
    print(f"Labels Parquet samples: {len(df_labels)}")
    
    # Check sample ID alignment
    data_ids = df_data['sample_id'].tolist()
    label_ids = df_labels['id.sample name'].tolist()
    
    matches = sum([1 for d, l in zip(data_ids, label_ids) if d == l])
    print(f"Sample ID matches (positional): {matches} / {len(data_ids)}")
    
    if matches != len(data_ids):
        print("!!! WARNING: Parquet files are NOT positionally aligned by sample ID !!!")
        # Check if they have the same set of IDs
        if set(data_ids) == set(label_ids):
            print("The ID sets match, but the order is different.")
        else:
            diff = set(data_ids) ^ set(label_ids)
            print(f"ID sets differ by {len(diff)} elements.")
    else:
        print("✓ Parquet files are perfectly aligned.")

    # 2. Embeddings
    print("\n--- Checking Embeddings ---")
    binformer_emb = torch.load('data/osdr/osdr_embeddings.pt', map_location='cpu', weights_only=True)
    pca_emb = torch.load('data/osdr/encodings_pca.pt', map_location='cpu', weights_only=True)
    
    print(f"Binformer embedding shape: {binformer_emb.shape}")
    print(f"PCA embedding shape: {pca_emb.shape}")
    
    # Check if they are just zeros or NaNs
    print(f"Binformer NaNs: {torch.isnan(binformer_emb).sum().item()}")
    print(f"PCA NaNs: {torch.isnan(pca_emb).sum().item()}")
    
    # Heuristic check: do they look "reasonable"?
    print(f"Binformer std: {binformer_emb.std():.4f}")
    print(f"PCA std: {pca_emb.std():.4f}")

if __name__ == "__main__":
    check_alignment()
