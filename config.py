# coding=utf-8
# Copyright 2026 The Google Research Authors.

# ============================================================
# MASTER SWITCH
# ============================================================
# Set to True for a 30-second sanity check.
# Set to False for real training runs.
USE_SMOKE = False 

# ============================================================
# PRODUCTION CONFIG (Full Training)
# ============================================================
TRAIN_CONFIG = {
    # Model Architecture
    'hidden_dim': 256,            
    'ffn_dim': 1024,               # 4x hidden
    'num_heads': 8,
    'num_layers': 4,
    'num_bins': 50,                # Match the fixed value in sweep.yaml
    'feature_type': 'sqr',
    'compute_type': 'iter',        # Iterative with large chunks is best for 16.5k
    
    # Optimizer
    'learning_rate': 4e-4,
    'weight_decay': 1e-4,
    'batch_size': 4,              
    'epochs': 20,                  
    'early_stopping': True,
    'patience': 5,                # Lower patience for shorter runs
    'seed': 42,
    
    # Data Selection
    'train_chunks': 4,            # ~20,000 samples
    'val_chunks': 1,
    'train_subset': None,        
    'val_subset': 2000,            # Small fast validation
    'balanced_sampling': True,
    'mask_ratio': 0.15,
    'mask_token': -10,
    'mask_token_prob': 0.8,
    'random_token_prob': 'auto',
    
    # Performance
    'num_workers': 2,             # 2 CPUs per GPU (matching Slurm allocation)
    'persistent_workers': True,
    
    # Paths
    'data_dir': '/global/scratch/users/brianzhou/archs4_mouse',
    'checkpoint_dir': 'checkpoints',
}

# ============================================================
# SMOKE CONFIG (Quick Sanity Check)
# ============================================================
SMOKE_CONFIG = {
    **TRAIN_CONFIG,              # Inherit everything from Train
    'train_chunks': 1,           # Then override with tiny values
    'val_chunks': 1,
    'train_subset': 100,
    'val_subset': 20,
    'epochs': 1,
    'batch_size': 2,
    'hidden_dim': 128,           
    'ffn_dim': 512,
    'early_stopping': False,
}

# Final config used by the code
CONFIG = SMOKE_CONFIG if USE_SMOKE else TRAIN_CONFIG
