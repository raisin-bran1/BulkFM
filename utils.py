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
