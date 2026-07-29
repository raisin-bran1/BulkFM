# RUN FROM DIRECTORY binformer

import torch
import os
import sys

# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from downstream.osdr.extract_binformer import extract_osdr_embeddings

# Set up paths
input_csv = 'data/osdr/osdr_processed_spaceflight.csv'
output_csv = 'data/osdr/osdr_embeddings.csv'

# Run extraction
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings_df = extract_osdr_embeddings(
    input_csv=input_csv,
    output_csv=output_csv,
    model_dir='best_model',
    device=device
)

if embeddings_df is not None:
    print(f"Final shape: {embeddings_df.shape}")
    
    # Save as PyTorch tensor for high-precision downstream tasks
    output_pt = output_csv.replace('.csv', '.pt')
    embeddings_tensor = torch.tensor(embeddings_df.values)
    torch.save(embeddings_tensor, output_pt)
    print(f"✓ PyTorch tensor saved to {output_pt}")
