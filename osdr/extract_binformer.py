import torch
import pandas as pd
import numpy as np
import json
import os
import sys

# Ensure we can find binformer when running from the root
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from binformer import Binformer
from binformer.osdr.utils import main_gene_selection, extract_feature

def load_binformer(model_dir_path='models', device='cpu'):
    """
    Loads the Binformer model from the specified directory.
    Expects config.json and base_model.pt to exist in model_dir.
    """

    config_path = os.path.join(model_dir_path, 'config.json')
    ckpt_path = os.path.join(model_dir_path, 'base_model.pt')
    vocab_path = os.path.join(model_dir_path, 'gene_vocabulary.csv')

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load vocabulary to get number of genes
    vocab = pd.read_csv(vocab_path)
    gene_list = vocab['genes'].tolist()
    num_genes = len(gene_list)

    # Initialize model
    model = Binformer(
        num_genes=num_genes,
        hidden_dim=config['hidden_dim'],
        n_heads=config['num_heads'],
        n_layers=config['num_layers'],
        ffn_dim=config['ffn_dim'],
        num_bins=config['num_bins'],
        mask_token_id=config['mask_token'],
        feature_type=config['feature_type'],
        compute_type=config['compute_type']
    ).to(device)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check if we need to remove 'module.' prefix from DDP keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"✓ Model loaded with {num_genes} genes.")
    return model, gene_list

def extract_osdr_embeddings(input_csv, output_csv='osdr_embeddings.csv', model_dir='best_model', device='cpu'):
    """
    Main pipeline for extracting OSDR embeddings.
    """
    # 1. Load model and vocabulary
    model, gene_list = load_binformer(model_dir, device)

    if not os.path.exists(input_csv):
        print(f"Error: OSDR file '{input_csv}' not found.")
        return

    print(f"Loading data from {input_csv}...")
    osdr_df = pd.read_csv(input_csv, index_col=0 if 'Unnamed' in input_csv else None) 
    
    # We expect osdr_df to have samples as rows and genes as columns
    if osdr_df.shape[1] < 100: 
        print("Transposing data (assuming genes were rows)...")
        osdr_df = osdr_df.T

    # 3. Align genes to the vocabulary
    aligned_df, missing, mask = main_gene_selection(osdr_df, gene_list)
    print(f"✓ Data aligned. Missing {len(missing)} genes out of {len(gene_list)}.")

    # 4. Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_feature(
        model=model,
        expr_array=aligned_df.values.astype(np.float32),
        output_feature_type='sample_level',
        device=device,
        batch_size=8 if device == 'cuda' else 1
    )

    # 5. Save results
    result_df = pd.DataFrame(embeddings.numpy(), index=osdr_df.index)
    
    result_df.to_csv(output_csv)
    print(f"✓ Done. Embeddings saved to {output_csv}")
    
    return result_df
