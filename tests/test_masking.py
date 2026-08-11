"""Rigorous tests for the fixed masking logic.

Run with:  python tests/test_masking.py
Exits nonzero on the first failure. Every function is named test_* so this
file is also pytest-compatible.
"""

import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import torch

from training.data import apply_dynamic_mask
from models.bulkfm import BulkFM, BulkFMConfig

G = 19239
B = 8
RATIO = 0.15
MASK_TOKEN = -10
MASK_P = 0.8
RAND_P = 0.1


def _rand_input(b=B, g=G, lo=0.0, hi=10.0, p_zero=0.4):
    rng = np.random.default_rng(0)
    x = rng.uniform(lo, hi, size=(b, g)).astype(np.float32)
    x[rng.random((b, g)) < p_zero] = 0.0
    return torch.tensor(x)


def _assert_close(a, b, msg=""):
    if not torch.equal(a, b):
        raise AssertionError(f"{msg}\n  {a[:8].tolist() if a.dim() else a} != {b[:8].tolist() if b.dim() else b}")


# ── core correctness ───────────────────────────────────────────────

def test_num_mask_exact():
    x = _rand_input()
    for ratio in (0.15, 0.3, 0.75, 0.001):
        xm, mask_idx = apply_dynamic_mask(x, ratio, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
        expected = max(1, int(G * ratio))
        assert mask_idx.shape == (B, expected), f"ratio {ratio}: got {mask_idx.shape}"
        # per-sample row of mask_idx is a valid subset of gene columns
        for row in mask_idx:
            assert row.min() >= 0 and row.max() < G
            assert len(torch.unique(row)) == expected


def test_masked_positions_equal_mask_idx():
    """Regression test for the original bug: masked input must differ from the
    original ONLY at the random mask_idx gene columns (not at prefix columns).
    With the 80/10/10 recipe ~90% of mask_idx columns change (10% are "keep")."""
    x = _rand_input()
    xm, mask_idx = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
    num_mask = mask_idx.shape[1]
    diff_cols = (xm != x)
    for b in range(B):
        cols = set(diff_cols[b].nonzero().squeeze(-1).tolist())
        expected = set(mask_idx[b].tolist())
        unexpected = cols - expected
        assert not unexpected, f"sample {b}: changed outside mask_idx: {sorted(unexpected)[:10]}"
        frac = len(cols) / num_mask
        assert abs(frac - (MASK_P + RAND_P)) < 0.03, \
            f"sample {b}: changed fraction {frac:.3f} != {MASK_P + RAND_P}"


def test_mask_token_and_random_fractions():
    """With mask_p=0.8/rand_p=0.1 the ~80% of masked genes hold exactly the
    mask token and ~10% hold a positive replacement value; the rest are kept."""
    x = _rand_input()
    xm, mask_idx = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
    num_mask = int(G * RATIO)
    for b in range(B):
        masked = xm[b, mask_idx[b]]
        orig = x[b, mask_idx[b]]
        n_token = (masked == MASK_TOKEN).sum().item()
        n_rand = ((masked != orig) & (masked != MASK_TOKEN)).sum().item()
        n_keep = (masked == orig).sum().item()
        assert n_token + n_rand + n_keep == num_mask, f"sample {b}"
        # ~80% mask tokens
        assert abs(n_token / num_mask - MASK_P) < 0.05, f"sample {b} token frac {n_token / num_mask:.3f}"
        # ~10% random replacements, and they are positive non-mask values
        assert abs(n_rand / num_mask - RAND_P) < 0.05, f"sample {b} rand frac {n_rand / num_mask:.3f}"
        if n_rand > 0:
            rand_vals = masked[(masked != orig) & (masked != MASK_TOKEN)]
            assert (rand_vals > 0).all(), f"sample {b} random replacements not positive"


def test_keep_only_leaves_input_unchanged():
    x = _rand_input()
    xm, mask_idx = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, 0.0, 0.0)
    _assert_close(xm, x, "keep-only masking must not modify the input")


def test_non_mask_token_strategy_returns_clone():
    x = _rand_input()
    xm, mask_idx = apply_dynamic_mask(x, RATIO, "cls_bottleneck", MASK_TOKEN, MASK_P, RAND_P)
    _assert_close(xm, x, "non-mask_token strategy must not modify input")
    assert mask_idx.shape == (B, int(G * RATIO))


# ── side-issue regression: per-sample permutations ────────────────

def test_mask_idx_differs_across_samples():
    """Regression test for the shared-permutation bug: every sample must get an
    independent mask_idx (old code expanded one randperm across the batch)."""
    x = _rand_input()
    for _ in range(5):
        xm, mask_idx = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
        pairs = [mask_idx[0], mask_idx[1], mask_idx[2], mask_idx[3]]
        assert len({tuple(p.tolist()) for p in pairs}) == len(pairs), "mask_idx rows must be distinct"


def test_bulkfm_get_mask_is_per_sample():
    cfg = BulkFMConfig(hidden_dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                       expression_embedding="continuous", mask_ratio=0.3)
    model = BulkFM(512, cfg)
    x = torch.zeros(8, 512)
    mask_idx = model._get_mask(x)
    rows = [tuple(r.tolist()) for r in mask_idx]
    assert len(set(rows)) == len(rows), "BulkFM._get_mask must give per-sample masks"


# ── robustness ─────────────────────────────────────────────────────

def test_non_contiguous_input():
    big = _rand_input(b=16, g=G)
    x = big[::2]  # strided, non-contiguous slice
    assert not x.is_contiguous()
    xm, mask_idx = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
    for b in range(B):
        changed = (xm[b] != x[b]).nonzero().squeeze(-1)
        assert set(changed.tolist()) <= set(mask_idx[b].tolist())
        assert (xm[b, mask_idx[b]][xm[b, mask_idx[b]] == MASK_TOKEN] == MASK_TOKEN).all()


def test_input_cloned_not_aliased():
    x = _rand_input()
    x_ref = x.clone()
    xm, _ = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
    _assert_close(x, x_ref, "apply_dynamic_mask must not mutate its input")


def test_cpu_and_cuda_agree():
    x = _rand_input()
    x_cpu, idx_cpu = apply_dynamic_mask(x, RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
    if torch.cuda.is_available():
        x_cuda, idx_cuda = apply_dynamic_mask(x.cuda(), RATIO, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
        x_cuda, idx_cuda = x_cuda.cpu(), idx_cuda.cpu()
        assert idx_cpu.shape == idx_cuda.shape
        # changed positions must be a subset of mask_idx on CUDA too
        for b in range(B):
            changed = (x_cuda[b] != x[b]).nonzero().squeeze(-1).tolist()
            assert set(changed) <= set(idx_cuda[b].tolist())
            # and every token position really holds the mask token
            masked = x_cuda[b, idx_cuda[b]]
            assert (masked[masked == MASK_TOKEN] == MASK_TOKEN).all()


# ── end-to-end model + backward ────────────────────────────────────

def test_model_forward_backward_with_fixed_mask():
    torch.manual_seed(0)
    cfg = BulkFMConfig(hidden_dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                       expression_embedding="continuous", mask_ratio=0.15)
    model = BulkFM(G, cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _rand_input(b=4)
    xm, mask_idx = apply_dynamic_mask(x, 0.3, "mask_token", MASK_TOKEN, MASK_P, RAND_P)
    out = model(xm, mask_idx=mask_idx)
    biv = torch.arange(4).unsqueeze(1)
    pred = out[biv, mask_idx]
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
    loss = torch.nn.functional.mse_loss(pred, x[biv, mask_idx])
    loss.backward()
    opt.step()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "no parameters received gradient"
    assert loss.item() > 0


# ── learning smoke test: masked reconstruction actually learns ─────

def test_learning_smoke():
    """On structured data the model must learn to reconstruct masked genes
    (loss on masked positions must fall well below its init value). With the
    old buggy masking the loss was computed at unmasked positions and
    collapsed to ~0 instantly without learning imputation."""
    torch.manual_seed(1234)
    K, N, Gg = 3, 256, 128
    z = torch.randn(N, K)
    W = torch.randn(K, Gg)
    X = (z @ W + 0.2 * torch.randn(N, Gg)).clamp(min=0).float()

    cfg = BulkFMConfig(hidden_dim=64, ffn_dim=128, num_heads=4, num_layers=2,
                       expression_embedding="continuous", mask_ratio=0.3,
                       dynamic_mask_range=[0.3, 0.5])
    model = BulkFM(Gg, cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    def masked_loss():
        total, n = 0.0, 0
        model.eval()
        with torch.no_grad():
            for s in range(0, N, 64):
                xb = X[s:s + 64]
                xm, mask_idx = apply_dynamic_mask(xb, 0.3, "mask_token", MASK_TOKEN, 0.8, 0.1)
                out = model(xm, mask_idx=mask_idx)
                biv = torch.arange(xb.shape[0]).unsqueeze(1)
                pred = out[biv, mask_idx]
                if pred.dim() == 3:
                    pred = pred.squeeze(-1)
                total += torch.nn.functional.mse_loss(pred, xb[biv, mask_idx]).item() * xb.shape[0]
                n += xb.shape[0]
        return total / n

    init_loss = masked_loss()

    for step in range(300):
        idx = torch.randperm(N)[:32]
        xb = X[idx]
        ratio = float(np.random.uniform(0.3, 0.5))
        xm, mask_idx = apply_dynamic_mask(xb, ratio, "mask_token", MASK_TOKEN, 0.8, 0.1)
        out = model(xm, mask_idx=mask_idx)
        biv = torch.arange(32).unsqueeze(1)
        pred = out[biv, mask_idx]
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, xb[biv, mask_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_loss = masked_loss()
    assert init_loss > 0.5, f"init masked loss unexpectedly low: {init_loss:.4f}"
    assert final_loss < init_loss * 0.5, \
        f"masked reconstruction did not learn: init {init_loss:.4f} -> final {final_loss:.4f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        print(f"  {fn} ...", flush=True)
        fn()
        print(f"    PASS", flush=True)
    print(f"\nAll {len(fns)} tests passed.")
