import os

USE_SMOKE = os.environ.get("USE_SMOKE", "0") == "1"

TRAIN_CONFIG = {
    # Model Architecture
    'hidden_dim': 256,
    'ffn_dim': 1024,
    'num_heads': 8,
    'num_layers': 4,
    'feature_type': 'elu+1',
    'compute_type': 'iter',

    # Expression embedding
    'expression_embedding': 'continuous', # binned or continous
    'num_bins': 50,
    'continuous_loss': 'mse', # mse or poisson
    'expression_projection': 'linear', # linear or nonlinear (MLP)

    # Masking strategy
    'masking_strategy': 'mask_token', # mask_token or cls_bottleneck
    'mask_ratio': 0.3,
    'dynamic_mask_range': None, # None or smth like [0.1, 0.5]
    'mask_token': -10,
    'mask_token_prob': 0.8,
    'random_token_prob': 'auto',

    # Optimizer
    'learning_rate': 4e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'epochs': 20,
    'early_stopping': True,
    'patience': 5,
    'seed': 42,

    # Data Selection
    'train_chunks': 4,
    'val_chunks': 1,
    'train_subset': None,
    'val_subset': 2000,

    # Performance & Stability
    'num_workers': 2,
    'persistent_workers': True,
    'torch_compile': True, # Turn off when dynamic masking is on
    'validations_per_epoch': 0,
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
    'epochs': 1,
    'batch_size': 2,
    'hidden_dim': 128,
    'ffn_dim': 512,
    'early_stopping': False,
}

CONFIG = SMOKE_CONFIG if USE_SMOKE else TRAIN_CONFIG
