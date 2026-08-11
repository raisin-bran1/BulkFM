# Agenda
# Test if adding sinusoidal & sample-level embeddings help. Turns out they don't help
# Run to dethrone bulkformer: add together full data, dynamic masking & do multiple epochs
# Simple masked autoencoder imputation baseline
# Test continued pretrain on tcga dataset

import os

USE_SMOKE = os.environ.get("USE_SMOKE", "0") == "1"

TRAIN_CONFIG = {
    # Model Architecture
    'hidden_dim': 256,
    'ffn_dim': 1024,
    'num_heads': 8,
    'num_layers': 8,
    'feature_type': 'elu+1',
    'compute_type': 'iter',

    # Expression embedding
    'expression_embedding': 'continuous', # binned or continous
    'num_bins': 50,
    'continuous_loss': 'mse', # mse or poisson
    'expression_projection': 'linear', # linear or nonlinear (sinusoidal)

    # Sample-level MLP embedding added to every token (like BulkFormer's global_expr_proj).
    # A small MLP (G -> r -> d, with ReLU) compresses the FULL gene-expression profile of a
    # sample (all G genes) into a single vector that is broadcast to every token. The value is
    # the bottleneck rank r of the G-wide layer: params(G-wide) = G * r.
    #   0 / False        -> off (default)
    #   64               -> +~1.2M params (recommended; low-rank, keeps 7.8M-model scale)
    #   256              -> +~5.0M params (d = hidden_dim, matches original width)
    #   4*hidden_dim     -> +~20M params (exact BulkFormer 4d width; ~3.5x total model)
    'sample_level_emb': 0,

    # Masking strategy
    'masking_strategy': 'mask_token', # mask_token or cls_bottleneck
    'mask_ratio': 0.30,
    'dynamic_mask_range': [0.15, 0.75], # None or smth like [0.1, 0.5]
    'val_mask_ratio': 0.15, # fixed masking ratio for validation loss when dynamic masking is used
    'mask_token': -10,
    'mask_token_prob': 0.8,
    'random_token_prob': 'auto',

    # Optimizer
    'learning_rate': 4e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 2,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,

    # Data Selection
    'train_chunks': 147,
    'val_chunks': 1,
    'train_subset': None,
    'val_subset': 2000,
    'chunks_in_memory': 4,  # number of data chunks held in RAM at a time

    # Performance & Stability
    'num_workers': 2,
    'persistent_workers': True,
    'torch_compile': True,
    'validations_per_epoch': 20,
    'grad_clip_norm': 1.0,

    # Paths
    'data_dir': '/media/volume/bulkrnadata/humandata',
    'checkpoint_dir': 'checkpoints',
}

SMOKE_CONFIG = {
    **TRAIN_CONFIG,
    'train_chunks': 1,
    'val_chunks': 1,
    'train_subset': 100,
    'val_subset': 20,
    'chunks_in_memory': 1,
    'epochs': 1,
    'batch_size': 2,
    'hidden_dim': 128,
    'ffn_dim': 512,
    'early_stopping': False,
}

CONFIG = SMOKE_CONFIG if USE_SMOKE else TRAIN_CONFIG
