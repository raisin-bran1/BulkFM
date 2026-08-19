# coding=utf-8
"""Autoencoder-specific configuration, kept separate from the training script.

Only the keys defined here override the shared training config
(training/config.py), which still supplies data paths, batch size, learning
rate, masking ranges, etc. Edit this file to change the architecture or
training settings (e.g. epochs) without touching training/train_autoencoder.py.
"""

import os

from training.config import USE_SMOKE

AUTOENCODER_CONFIG = {
    # ── Architecture: genes -> hidden_dims -> genes ──────────────
    'hidden_dims': (1024, 2048, 1024),
    'mask_value': 0.0,

    # ── Training overrides (rest inherited from training/config.py) ──
    'epochs': 5,
}

SMOKE_AUTOENCODER_CONFIG = {
    **AUTOENCODER_CONFIG,
    'hidden_dims': (64, 128, 64),
    'epochs': 1,
}

CONFIG = SMOKE_AUTOENCODER_CONFIG if USE_SMOKE else AUTOENCODER_CONFIG
