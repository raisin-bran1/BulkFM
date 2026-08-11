import torch
import torch.nn as nn
from models.bulkformer.BulkFormer_block import BulkFormer_block
from models.bulkformer.Rope import PositionalExprEmbedding

class BulkFormer(nn.Module):
    def __init__(self, 
                 dim, graph, gene_length, gene_emb = None,
                 bin_head=4, full_head=4, bins=10,
                 gb_repeat=3, p_repeat=1):
        super().__init__()
        self.dim = dim
        self.gene_length = gene_length
        self.graph = graph

        # # === gene embedding ===
        # self.gene_emb = nn.Parameter(gene_emb) // train their own embeddings(?)

        # one-hot embedding initialization
        self.gene_emb_onehot_layer = nn.Embedding(gene_length, dim)
        nn.init.xavier_uniform_(self.gene_emb_onehot_layer.weight)

        self.gene_emb_proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        self.expr_emb = PositionalExprEmbedding(dim)
        self.x_proj = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        # === main module ===
        is_graph = True
        self.gb_formers = nn.ModuleList([
            BulkFormer_block(dim, gene_length, is_graph, bin_head, full_head, bins, p_repeat)
            for _ in range(gb_repeat)
        ])

        self.layernorm = nn.LayerNorm(dim)

        # === sample-level overall expression embedding ===
        self.global_expr_proj = nn.Sequential(
            nn.Linear(gene_length, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

        # === output head: per-gene prediction ===
        self.head = nn.Sequential(
            nn.LayerNorm(dim + 3),
            nn.Linear(dim + 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.ReLU()
        )

    def forward(self, x, mask_prob=None, output_expr = False):
        b, g = x.shape
        x_input = x.clone()

        gene_emb_onehot = self.gene_emb_onehot_layer.weight
        # === gene expression embedding ===
        gene_emb_proj = self.gene_emb_proj(gene_emb_onehot)

        x = self.expr_emb(x) + gene_emb_proj + self.global_expr_proj(x).unsqueeze(1).expand(-1, g, -1)

        x = self.x_proj(x)

        # === backbone ===
        for layer in self.gb_formers:
            x = layer(x, self.graph)
       
        # === gene token output ===
        gene_emb = self.layernorm(x)

        # === sample statistical features ===
        # mask rate
        mask_scalar = torch.full((b, g, 1), mask_prob or 0.0, device=x.device)
        # === mask information ===
        mask_token_val = -10.0
        mask = (x_input == mask_token_val).float()       # 1 indicates masked genes
        valid_mask = 1 - mask                            # 1 indicates non-masked genes

        # mean expression value (excluding masked positions)
        expr_mean = (x_input * valid_mask).sum(dim=1, keepdim=True) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        expr_mean = expr_mean.unsqueeze(-1).expand(-1, g, -1)

        # non-zero ratio
        nonzero_ratio = (x_input != 0).float().sum(dim=1, keepdim=True) / g
        nonzero_ratio = nonzero_ratio.unsqueeze(-1).expand(-1, g, -1)

        # === concatenate features (no CLS) === [b, g, dim+3]
        gene_emb_output = torch.cat([gene_emb, mask_scalar, expr_mean, nonzero_ratio],dim=-1)  

        # === output prediction ===
        pred = self.head(gene_emb_output).squeeze(-1)

        # === mean correction ===
        # 2. mean prediction over non-masked positions
        pred_valid_mean = (pred * valid_mask).sum(dim=1, keepdim=True) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        # 3. observed mean (mean of the non-masked part of the true expression)
        observed_mean = (x_input * valid_mask).sum(dim=1, keepdim=True) / (valid_mask.sum(dim=1, keepdim=True) + 1e-8)
        # 4. correction applied only to masked positions
        pred_corrected = pred.clone()
        pred_corrected = pred_corrected - mask * (pred_valid_mean - observed_mean)

        if output_expr:
            return pred_corrected
        else:
            return gene_emb_output