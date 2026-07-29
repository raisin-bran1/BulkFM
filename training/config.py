import os

USE_SMOKE = os.environ.get("USE_SMOKE", "0") == "1"

TRAIN_CONFIG = {
    # Model Architecture
    'hidden_dim': 256,
    'ffn_dim': 1024,
    'num_heads': 8,
    'num_layers': 4,
    'feature_type': 'sqr',
    'compute_type': 'iter',

    # Expression embedding
    'expression_embedding': 'binned',
    'num_bins': 50,
    'continuous_loss': 'mse',

    # Masking strategy
    'masking_strategy': 'mask_token',
    'mask_ratio': 0.15,
    'dynamic_mask_range': None,
    'mask_token': -10,
    'mask_token_prob': 0.8,
    'random_token_prob': 'auto',

    # Optimizer
    'learning_rate': 4e-4,
    'weight_decay': 1e-4,
    'batch_size': 4,
    'epochs': 20,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,

    # Data Selection
    'train_chunks': 4,
    'val_chunks': 1,
    'train_subset': None,
    'val_subset': 2000,
    'balanced_sampling': True,

    # Performance
    'num_workers': 2,
    'persistent_workers': True,

    # Paths
    'data_dir': '/global/scratch/users/brianzhou/archs4_human',
    'checkpoint_dir': 'checkpoints',
}

SMOKE_CONFIG = {
    **TRAIN_CONFIG,
    'train_chunks': 1,
    'val_chunks': 1,
    'train_subset': 100,
    'val_subset': 20,
    'epochs': 1,
    'batch_size': 2,
    'hidden_dim': 128,
    'ffn_dim': 512,
    'early_stopping': False,
}

CONFIG = SMOKE_CONFIG if USE_SMOKE else TRAIN_CONFIG
