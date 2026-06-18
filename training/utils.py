def format_float_for_tag(v: float) -> str:
    """Formats float for inclusion in a directory name/tag."""
    s = f"{v:.2e}" if (abs(v) < 1e-3 or abs(v) >= 1e3) else f"{v:.6f}"
    return s.replace('.', 'p').replace('+', '').replace('-', 'm')

def build_run_tag(cfg: dict) -> str:
    """Builds a human-readable tag string for the training run."""
    return (
        f"lr-{format_float_for_tag(cfg['learning_rate'])}"
        f"_wd-{format_float_for_tag(cfg['weight_decay'])}"
        f"_mask-{format_float_for_tag(cfg['mask_ratio'])}"
        f"_bins-{format_float_for_tag(cfg['num_bins'])}"
    )

def _coerce_config_types(cfg: dict) -> None:
    """Coerces string config values to their correct numeric types."""
    # List of keys that should be int
    int_keys = ['hidden_dim', 'ffn_dim', 'num_heads', 'num_layers', 'num_bins', 
                'batch_size', 'epochs', 'seed', 'train_chunks', 'val_chunks', 
                'patience', 'num_workers']
    # List of keys that should be float
    float_keys = ['learning_rate', 'weight_decay', 'mask_ratio', 
                  'mask_token_prob', 'random_token_prob']
    
    for k in int_keys:
        if k in cfg and cfg[k] is not None:
            try:
                cfg[k] = int(cfg[k])
            except (ValueError, TypeError):
                pass
                
    for k in float_keys:
        if k in cfg and cfg[k] is not None and cfg[k] != 'auto':
            try:
                cfg[k] = float(cfg[k])
            except (ValueError, TypeError):
                pass
