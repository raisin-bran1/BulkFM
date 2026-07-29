# coding=utf-8
# Fine-tuning Configuration for NASA OSDR Spaceflight Classification

FINETUNE_CONFIG = {
    # General
    'seed': 42,
    'num_classes': 2,
    'smoke_test': False,          # If True, runs only 1 batch per epoch
    'device': 'auto',             # 'auto', 'cuda', or 'cpu'
    
    # Hardware & Performance
    'num_workers': 2,             # Number of CPU workers for data loading
    'pin_memory': True,           # Faster data transfer to GPU
    
    # Model Strategy
    'use_lora': True,
    'lora_rank': 16,
    'lora_alpha': 32,
    'pooling_type': 'attention',   # Options: 'attention', 'mean'
    
    # Data & Hardware
    'batch_size': 4,
    'accum_iter': 8,               # Effective batch size = batch_size * accum_iter
    
    # STAGE 1: Training pooling layer and classification head (Base Model Frozen)
    'stage1': {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 1,        
        'epochs': 3,               # Short phase, usually constant LR
    },
    
    # STAGE 2: Training LoRA weights + pooling/head (or just pooling/head if no LoRA)
    'stage2': {
        'lora_lr': 2e-5,
        'head_lr': 5e-5,
        'no_lora_lr': 5e-4,        # Used if use_lora is False
        'weight_decay': 1e-4,
        'warmup_epochs': 2,        # Linear warmup before cosine decay
        'epochs': 20,
    },
    
    # File Paths
    'paths': {
        'train_features': "data/osdr/osdr_train_features.parquet",
        'train_labels': "data/osdr/osdr_train_labels.parquet",
        'val_features': "data/osdr/osdr_val_features.parquet",
        'val_labels': "data/osdr/osdr_val_labels.parquet",
        'base_model': "models/base_model.pt",
        'base_config': "models/config.json",
        'vocab': "models/gene_vocabulary.csv",
        'results_dir': "results",  # Base directory for all runs
    }
}
