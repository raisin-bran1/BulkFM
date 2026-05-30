"""
Binformer: Bulk RNA foundation model using binned expression embeddings

Adapted from Google's SLiMPerformer for continuous gene expression data.
Architecture:
  - Gene identity embedding (learned, like BERT token IDs)
  - Bin tokens for value-based positional encoding
  - N Transformer layers (multi-head attention + FFN + LayerNorm)
  - Output projection for per-gene expression reconstruction

Training objective: MLM-style masking (mask 15% of genes, predict their bin)
"""

import torch
import torch.nn as nn
from slim_performer_model import SLiMPerformerLayer

# ============================================================
# BIN EXPRESSION EMBEDDING (BEE)
# ============================================================
class BinExpressionEmbedding(nn.Module):
    """
    Bin Expression Embedding (BEE): Converts continuous gene expression
    values into discrete bin embeddings.

    Divides the sample's gene expressions into n+2 bins:
    - Bin 0: All zero values (Fixed zero vector).
    - Bins 1..n: Non-zero values uniformly distributed by quantile (Learned).
    - Bin n+1: Mask tokens (Learned).
    """

    def __init__(self, dim, bins, mask_token_id=-10):
        super().__init__()
        self.dim = dim
        self.bins = bins
        self.mask_token_id = mask_token_id

        # bins (quantiles) + 1 (zeros) + 1 (mask)
        self.embedding = nn.Embedding(bins + 2, dim)

        # Initialize index 0 (extra bin) to zeros
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

    def get_bin_indices(self, x):
        """
        Computes the bin index for each expression value.
        Vectorized version to avoid Python loops.
        """
        B, G = x.shape
        device = x.device

        # Identify mask tokens and zero values
        is_mask = (x == self.mask_token_id)
        is_zero = (x == 0)
        is_nonzero = ~(is_mask | is_zero)

        # Initialize bin indices
        bin_indices = torch.zeros_like(x, dtype=torch.long)

        if not is_nonzero.any():
            if is_mask.any():
                bin_indices[is_mask] = self.bins + 1
            return bin_indices

        # Vectorized rank calculation for non-zero values
        # We replace zero/mask with very small values so they don't affect sorting of non-zeros
        # (Assuming expression values are positive)
        temp_x = x.clone()
        temp_x[~is_nonzero] = -1e9
        
        # argsort twice gives the rank [0, G-1]
        ranks = torch.argsort(torch.argsort(temp_x, dim=1), dim=1)
        
        # Now we need the number of non-zero elements per sample to scale ranks correctly
        num_nonzero = is_nonzero.sum(dim=1, keepdim=True) # [B, 1]
        
        # Number of zeros per sample (to shift ranks so non-zeros start at rank 0)
        num_zeros = is_zero.sum(dim=1, keepdim=True)
        num_masks = is_mask.sum(dim=1, keepdim=True)
        
        # For non-zero elements, their rank in the sorted list starts after zeros and masks.
        # However, because we set zero/mask to -1e9, their ranks will be [0, num_zeros+num_masks-1].
        # Non-zeros will have ranks [num_zeros+num_masks, G-1].
        # Shifted rank for non-zeros: rank - (num_zeros + num_masks)
        shifted_ranks = (ranks - (num_zeros + num_masks)).clamp(min=0)
        
        # Map shifted ranks to bins [1, self.bins]
        sample_bins = (shifted_ranks * self.bins // num_nonzero.clamp(min=1)) + 1
        
        # Mask back non-zero positions
        bin_indices = torch.where(is_nonzero, sample_bins, bin_indices)
        
        # Set mask tokens to bin n+1
        if is_mask.any():
            bin_indices[is_mask] = self.bins + 1
            
        return bin_indices

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_genes] expression values

        Returns:
            [batch_size, num_genes, dim] bin embeddings
        """
        bin_indices = self.get_bin_indices(x)
        return self.embedding(bin_indices)


# ============================================================
# BINFORMER MODEL
# ============================================================
class Binformer(nn.Module):
    """
    Binformer: Transformer for gene expression data.
    Uses SLiMPerformer's linear attention (O(n) memory) from Google Research.

    Input:  [batch, num_genes] expression values (with masked positions = -10)
    Output: [batch, num_genes, num_bins + 1] bin logits

    Embeddings (summed, like BERT):
      1. Gene identity embedding — learned per-gene vector (like BERT token IDs)
      2. Bins
    """

    def __init__(self, num_genes, hidden_dim=256, n_heads=8, n_layers=4,
                 ffn_dim=1024, num_bins=10, mask_token_id=-10,
                 feature_type='sqr', compute_type='iter'):
        super().__init__()
        self.num_genes = num_genes
        self._hidden_dim = hidden_dim
        self.num_bins = num_bins

        # Gene identity embedding (like BERT's token embedding)
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)

        # Bin Expression Embedding
        self.bee = BinExpressionEmbedding(hidden_dim, bins=num_bins,
                                            mask_token_id=mask_token_id)

        # SLiMPerformer layers (linear O(n) attention via prefix sums)
        self.layers = nn.ModuleList([
            SLiMPerformerLayer(hidden_dim, ffn_dim, n_heads,
                               feature_type, compute_type, on_gptln=True)
            for _ in range(n_layers)
        ])

        # Output: predict bin probability distribution per gene
        self.output_map = nn.Linear(hidden_dim, num_bins + 1)

    def forward(self, x, output_hidden=False):
        """
        Args:
            x: [batch, num_genes] expression values
            output_hidden: If True, returns the hidden state before the output_map
        Returns:
            [batch, num_genes, num_bins + 1] predicted bin logits, OR
            [batch, num_genes, hidden_dim] hidden state
        """
        B, G = x.shape
        device = x.device

        # Gene identity embeddings: [G, hidden_dim] → broadcast to [B, G, hidden_dim]
        gene_ids = torch.arange(G, device=device)
        gene_emb = self.gene_embedding(gene_ids)

        # BEE from expression values: [B, G, hidden_dim]
        bee_emb = self.bee(x)

        # Sum embeddings (like BERT: token + position)
        h = gene_emb.unsqueeze(0) + bee_emb

        # Pass through SLiMPerformer layers (linear attention)
        for layer in self.layers:
            rfs = layer.attention.sample_rfs(device)
            h = layer.full_forward(h, rfs)

        if output_hidden:
            return h

        # Project to logits per bin
        out = self.output_map(h)  # [B, G, num_bins + 1]

        return out