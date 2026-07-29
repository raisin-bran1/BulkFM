import dataclasses
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.slim_performer_model import SLiMPerformerLayer


@dataclasses.dataclass
class GeneralizedBinformerConfig:
    hidden_dim: int = 256
    ffn_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    feature_type: str = 'sqr'
    compute_type: str = 'iter'

    expression_embedding: Literal['binned', 'continuous'] = 'binned'
    num_bins: int = 50

    continuous_loss: Literal['mse', 'poisson'] = 'mse'

    mask_ratio: float = 0.15
    dynamic_mask_range: Optional[tuple[float, float]] = None
    mask_token_id: float = -10.0

    masking_strategy: Literal['mask_token', 'cls_bottleneck'] = 'mask_token'


class PoissonNLLLogSpace(nn.Module):
    def forward(self, log1p_input, log1p_target):
        return log1p_input.exp() - log1p_target.exp() * log1p_input


class RectifiedTanh(nn.Module):
    def __init__(self, upper_bound: int = 100000):
        super().__init__()
        self.upper_bound = np.log1p(upper_bound)

    def forward(self, x):
        return self.upper_bound * F.relu(2 * torch.sigmoid(x / (2 * np.e)) - 1)


class BinExpressionEmbedding(nn.Module):
    def __init__(self, dim, bins, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.mask_token_id = mask_token_id
        self.embedding = nn.Embedding(bins + 2, dim)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

    def get_bin_indices(self, x):
        B, G = x.shape
        device = x.device
        is_mask = (x == self.mask_token_id)
        is_zero = (x == 0)
        is_nonzero = ~(is_mask | is_zero)
        bin_indices = torch.zeros_like(x, dtype=torch.long)
        if not is_nonzero.any():
            if is_mask.any():
                bin_indices[is_mask] = self.bins + 1
            return bin_indices
        temp_x = x.clone()
        temp_x[~is_nonzero] = -1e9
        ranks = torch.argsort(torch.argsort(temp_x, dim=1), dim=1)
        num_nonzero = is_nonzero.sum(dim=1, keepdim=True)
        num_zeros = is_zero.sum(dim=1, keepdim=True)
        num_masks = is_mask.sum(dim=1, keepdim=True)
        shifted_ranks = (ranks - (num_zeros + num_masks)).clamp(min=0)
        sample_bins = (shifted_ranks * self.bins // num_nonzero.clamp(min=1)) + 1
        bin_indices = torch.where(is_nonzero, sample_bins, bin_indices)
        if is_mask.any():
            bin_indices[is_mask] = self.bins + 1
        return bin_indices

    def forward(self, x):
        bin_indices = self.get_bin_indices(x)
        return self.embedding(bin_indices)


class ContinuousExpressionEmbedding(nn.Module):
    def __init__(self, dim, mask_token_id=-10):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.expr_proj = nn.Linear(1, dim, bias=False)
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, gene_emb):
        is_mask = (x == self.mask_token_id).unsqueeze(-1)
        x_safe = x.clamp(min=0)
        x_log1p = torch.log1p(x_safe)
        expr_emb = self.expr_proj(x_log1p.unsqueeze(-1))
        expr_emb = torch.where(is_mask, self.mask_embedding, expr_emb)
        return gene_emb + expr_emb


class GeneralizedBinformer(nn.Module):
    def __init__(self, num_genes, cfg: GeneralizedBinformerConfig):
        super().__init__()
        self.cfg = cfg
        self.num_genes = num_genes

        self.gene_embedding = nn.Embedding(num_genes, cfg.hidden_dim)

        if cfg.expression_embedding == 'binned':
            self.expr_embedding = BinExpressionEmbedding(
                cfg.hidden_dim, bins=cfg.num_bins, mask_token_id=cfg.mask_token_id
            )
            self.output_dim = cfg.num_bins + 1
        else:
            self.expr_embedding = ContinuousExpressionEmbedding(
                cfg.hidden_dim, mask_token_id=cfg.mask_token_id
            )
            self.output_dim = 1

        self.layers = nn.ModuleList([
            SLiMPerformerLayer(cfg.hidden_dim, cfg.ffn_dim, cfg.num_heads,
                               cfg.feature_type, cfg.compute_type, on_gptln=True)
            for _ in range(cfg.num_layers)
        ])

        if cfg.masking_strategy == 'cls_bottleneck':
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
            dec_hidden = max(cfg.hidden_dim, 256)
            self.decoder = nn.Sequential(
                nn.Linear(cfg.hidden_dim, dec_hidden),
                nn.GELU(),
                nn.Linear(dec_hidden, cfg.hidden_dim),
            )
            if cfg.expression_embedding == 'continuous':
                self.output_head = nn.Linear(cfg.hidden_dim, num_genes)
                self.output_activation = RectifiedTanh() if cfg.continuous_loss == 'poisson' else nn.Identity()
            else:
                self.output_head = nn.Linear(cfg.hidden_dim, num_genes * self.output_dim)
                self.output_activation = nn.Identity()
            self.output_map = None
        else:
            self.output_map = nn.Linear(cfg.hidden_dim, self.output_dim)
            self.cls_token = None
            self.decoder = None
            self.output_head = None
            self.output_activation = None

    def _get_mask(self, x):
        B, G = x.shape
        if self.cfg.dynamic_mask_range is not None:
            lo, hi = self.cfg.dynamic_mask_range
            mask_ratio = torch.empty(1, device=x.device).uniform_(lo, hi).item()
            num_mask = max(1, int(G * mask_ratio))
        else:
            num_mask = max(1, int(G * self.cfg.mask_ratio))
        rand = torch.rand(B, G, device=x.device)
        mask_idx = rand.argsort(dim=1)[:, :num_mask]
        return mask_idx

    def forward(self, x, mask_idx=None, output_hidden=False, output_cls=False):
        B, G = x.shape
        device = x.device
        cfg = self.cfg

        if mask_idx is None:
            mask_idx = self._get_mask(x)

        gene_ids = torch.arange(G, device=device)
        gene_emb = self.gene_embedding(gene_ids).unsqueeze(0).expand(B, -1, -1)

        if cfg.masking_strategy == 'mask_token':
            if cfg.expression_embedding == 'binned':
                h = self.expr_embedding(x)
                h = gene_emb + h
            else:
                h = self.expr_embedding(x, gene_emb)

            for layer in self.layers:
                rfs = layer.attention.sample_rfs(device)
                h = layer.full_forward(h, rfs)

            if output_hidden:
                return h
            return self.output_map(h)

        unmask_idx = ~torch.zeros(B, G, dtype=torch.bool, device=device)\
            .scatter(1, mask_idx, True)

        cls_out_list = []
        for b in range(B):
            um = unmask_idx[b].nonzero(as_tuple=True)[0]

            if cfg.expression_embedding == 'binned':
                gb = gene_emb[b:b+1, um]
                h = gb + self.expr_embedding(x[b:b+1, um])
            else:
                gb = gene_emb[b:b+1, um]
                h = self.expr_embedding(x[b:b+1, um], gb)

            seq = torch.cat([h, self.cls_token], dim=1)

            for layer in self.layers:
                rfs = layer.attention.sample_rfs(device)
                seq = layer.full_forward(seq, rfs)

            cls_out_list.append(seq[0, -1])

        cls_out = torch.stack(cls_out_list, dim=0)

        if output_cls:
            return cls_out

        dec = self.decoder(cls_out)
        pred = self.output_head(dec)
        pred = self.output_activation(pred)

        if cfg.expression_embedding == 'binned':
            pred = pred.view(B, G, cfg.num_bins + 1)

        if output_hidden:
            return dec

        return pred

    @torch.no_grad()
    def extract_gene_embeddings(self, gene_names=None):
        device = next(self.parameters()).device
        G = self.num_genes
        gene_ids = torch.arange(G, device=device)
        gene_emb = self.gene_embedding(gene_ids).cpu().numpy()
        import pandas as pd
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(G)]
        return pd.DataFrame(gene_emb, index=gene_names)
