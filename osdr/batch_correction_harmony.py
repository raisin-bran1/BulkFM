import torch
import pandas as pd
import numpy as np
import harmonypy as hm
from sklearn.preprocessing import StandardScaler

def run_unsupervised_harmony(input_pt_path, batch_ids, output_prefix="corrected_data"):
    """
    Runs unsupervised Harmony on PCA embeddings and saves the results.
    """
    # 1. Load the PCA embeddings
    print(f"Loading embeddings from {input_pt_path}...")
    embeddings = torch.load(input_pt_path)
    
    # Ensure it's a numpy array for Harmony
    if torch.is_tensor(embeddings):
        data_mat = embeddings.detach().cpu().numpy()
    else:
        data_mat = embeddings

    # 2. Prepare Metadata (Only Batch Knowledge)
    # Harmony expects a DataFrame for metadata
    metadata = pd.DataFrame({'batch': batch_ids})
    
    print(f"Running Harmony on {data_mat.shape[0]} samples across {len(np.unique(batch_ids))} batches...")
    
    # 3. Run Harmony
    # We only use 'batch' as the covariate to avoid "cheating" with labels
    ho = hm.run_harmony(data_mat, metadata, 'batch', max_iter_harmony=20)
    
    # 4. Extract Corrected Embeddings
    # Harmony returns (n_components, n_samples), so we transpose back to (n_samples, n_components)
    corrected_embeddings = ho.Z_corr
    
    # 5. Save Results
    print("Saving corrected embeddings...")
    
    # Save as PyTorch tensor
    torch.save(torch.tensor(corrected_embeddings), f"{output_prefix}.pt")
    
    # Save as CSV for easy inspection/plotting
    df_corrected = pd.DataFrame(corrected_embeddings)
    df_corrected.to_csv(f"{output_prefix}.csv", index=False)
    
    print(f"Done! Saved to {output_prefix}.pt and {output_prefix}.csv")
    return corrected_embeddings

# --- Usage Example ---
batch_list = pd.read_parquet('osdr/organ_spaceflight_batch_labels.parquet')['osd_batch'].tolist()
corrected = run_unsupervised_harmony('osdr/osdr_embeddings_pca.pt', batch_list, output_prefix='osdr/osdr_embeddings_harmony')