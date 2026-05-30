import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def main():
    input_path = 'osdr/osdr_processed_spaceflight.parquet'
    output_path = 'osdr/osdr_embeddings_pca.pt'
    
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Drop non-numeric columns
    if 'sample_id' in df.columns:
        df = df.drop(columns=['sample_id'])
    
    # Fill NaNs
    print("Handling missing values...")
    data = df.fillna(0).values
    
    # Optional: Scaling
    # PCA is sensitive to the scale of the features. 
    # Usually, we standardize to mean=0, std=1.
    print("Standardizing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    print("Computing PCA (256 components)...")
    pca = PCA(n_components=256, random_state=42)
    pca_features = pca.fit_transform(data_scaled)
    
    print(f"Explained variance ratio (sum): {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Convert to torch tensor
    pca_tensor = torch.tensor(pca_features, dtype=torch.float32)
    
    print(f"Saving to {output_path}...")
    torch.save(pca_tensor, output_path)
    print("✓ Done.")

if __name__ == "__main__":
    main()
