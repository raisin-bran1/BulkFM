"""
MLP autoencoder baseline for benchmarking against the transformer models.

Architecture: genes -> 256 -> 512 -> 256 -> genes with GELU activations and
LayerNorm. Masked genes are fed into the network as 0s and the reconstruction
loss (MSE) is computed only on the masked positions.

Masking is compatible with BulkFM's dynamic masking: the ratio is sampled
uniformly from a range (e.g. [0.15, 0.75]) per batch, and each sample gets an
independent random permutation of gene indices. Like BulkFM, ``forward`` takes
the unmasked input plus ``mask_idx`` and masks the genes internally; unlike
BulkFM there is no mask token, so masked genes are simply zeroed out.

Training/eval compatibility: ``forward`` returns a full ``(B, G)``
reconstruction, so existing loss patterns such as ``out[biv, mask_idx]`` keep
working. ``compute_loss`` wraps masking + MSE-on-masked-positions. The encoder
output (last hidden state, e.g. 256-dim) is exposed via ``output_hidden=True``
for sample-level embedding extraction (the MLP has no CLS token).
"""

import dataclasses
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class MLPAutoencoderConfig:
    hidden_dims: tuple = (256, 512, 256)
    mask_ratio: float = 0.15
    dynamic_mask_range: Optional[tuple] = (0.15, 0.75)
    mask_value: float = 0.0


class MLPAutoencoder(nn.Module):
    def __init__(self, num_genes, cfg: MLPAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.num_genes = num_genes

        dims = [num_genes] + list(cfg.hidden_dims)
        blocks = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(nn.Linear(in_dim, out_dim))
            blocks.append(nn.LayerNorm(out_dim))
            blocks.append(nn.GELU())
        self.mlp = nn.Sequential(*blocks)
        self.output = nn.Linear(dims[-1], num_genes)

    def _get_mask(self, x):
        B, G = x.shape
        if self.cfg.dynamic_mask_range is not None:
            lo, hi = self.cfg.dynamic_mask_range
            mask_ratio = random.uniform(lo, hi)
            num_mask = max(1, int(G * mask_ratio))
        else:
            num_mask = max(1, int(G * self.cfg.mask_ratio))
        idxs = torch.argsort(torch.rand(B, G, device=x.device), dim=1)
        mask_idx = idxs[:, :num_mask]
        return mask_idx

    def forward(self, x, mask_idx=None, output_hidden=False):
        """Reconstruct all genes from ``x`` with masked genes zeroed.

        Args:
            x: [batch, num_genes] expression values (unmasked).
            mask_idx: [batch, num_mask] gene columns to mask. If None, a mask
                is sampled per sample (dynamic ratio in ``dynamic_mask_range``).
            output_hidden: if True, return the encoder output
                ([batch, hidden_dims[-1]]) instead of the reconstruction.

        Returns:
            [batch, num_genes] reconstructed expression values, or the encoder
            output hidden state if ``output_hidden=True``.
        """
        B, G = x.shape
        if mask_idx is None:
            mask_idx = self._get_mask(x)

        x_in = x.clone()
        x_in.scatter_(1, mask_idx, self.cfg.mask_value)
        h = self.mlp(x_in)
        if output_hidden:
            return h
        return self.output(h)

    def compute_loss(self, x, targets, mask_idx=None):
        """MSE reconstruction loss computed only on the masked genes.

        Args:
            x: [batch, num_genes] expression values (unmasked).
            targets: [batch, num_genes] original expression values.
            mask_idx: [batch, num_mask] gene columns to mask. If None, a mask
                is sampled per sample (same distribution as ``forward``).

        Returns:
            Scalar MSE loss over masked positions.
        """
        if mask_idx is None:
            mask_idx = self._get_mask(x)

        pred = self.forward(x, mask_idx=mask_idx)
        B = x.shape[0]
        biv = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, mask_idx.shape[1])
        masked_pred = pred[biv, mask_idx]
        masked_targets = targets[biv, mask_idx]
        return F.mse_loss(masked_pred, masked_targets)
