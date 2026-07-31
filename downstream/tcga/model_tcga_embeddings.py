"""
Extract sample-level embeddings from a trained BulkFM model on TCGA data.

Missing TCGA genes are handled by masking (cls_bottleneck) or
by setting to mask_token (mask_token), never zero-filled.

Usage:
  python downstream/tcga/model_tcga_embeddings.py \
      --checkpoint checkpoints/train_20260729_230537_local/best_model.pt \
      --output tcga_bulkfm.pt
"""

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import time
import torch
import numpy as np
import pandas as pd
from models.bulkfm import BulkFM, BulkFMConfig, ContinuousExpressionEmbedding

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
TCGA_VOCAB = "/media/volume/bulkrnadata/tcgadata/tcga_gene_vocabulary.csv"
MODEL_VOCAB = "checkpoints/gene_vocabulary.csv"
OUTPUT_DIR = "embeddings"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Extract BulkFM sample embeddings on TCGA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="tcga_bulkfm.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tcga_parquet", type=str, default=TCGA_PARQUET)
    parser.add_argument("--tcga_vocab", type=str, default=TCGA_VOCAB)
    parser.add_argument("--model_vocab", type=str, default=MODEL_VOCAB)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ──
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    ckpt_config = checkpoint.get("config", {})
    model_cfg = BulkFMConfig(
        hidden_dim=ckpt_config.get("hidden_dim", 256),
        ffn_dim=ckpt_config.get("ffn_dim", 1024),
        num_heads=ckpt_config.get("num_heads", 8),
        num_layers=ckpt_config.get("num_layers", 4),
        expression_embedding=ckpt_config.get("expression_embedding", "continuous"),
        masking_strategy=ckpt_config.get("masking_strategy", "cls_bottleneck"),
        num_bins=ckpt_config.get("num_bins", 50),
        mask_ratio=ckpt_config.get("mask_ratio", 0.75),
        mask_token_id=ckpt_config.get("mask_token", -10),
        continuous_loss=ckpt_config.get("continuous_loss", "mse"),
        simple_projection=ckpt_config.get("expression_projection", "nonlinear") == "linear",
    )
    print(f"  Config: hidden={model_cfg.hidden_dim}, expr={model_cfg.expression_embedding}, "
          f"mask={model_cfg.masking_strategy}")

    num_genes = state_dict["gene_embedding.weight"].shape[0]
    model = BulkFM(num_genes, model_cfg)

    # TEMPORARY: handle old checkpoint with single Linear expr_proj (no GELU / second layer)
    if ("expr_embedding.expr_proj.weight" in state_dict
            and "expr_embedding.expr_proj.0.weight" not in state_dict):
        print("  Detected old expr_proj format — switching to single Linear")
        model.expr_embedding = ContinuousExpressionEmbedding(
            model_cfg.hidden_dim,
            mask_token_id=model_cfg.mask_token_id,
            simple_projection=True,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  Model: {num_genes} genes, {sum(p.numel() for p in model.parameters()):,} params")

    # ── Load TCGA data ──
    print(f"Loading TCGA data: {args.tcga_parquet}")
    t0 = time.time()
    df = pd.read_parquet(args.tcga_parquet)
    print(f"  Shape: {df.shape} ({time.time()-t0:.1f}s)")

    meta_cols = [c for c in df.columns if not isinstance(df[c].dtype, pd.Float32Dtype)
                 and df[c].dtype != np.float32]
    tcga_gene_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  TCGA genes: {len(tcga_gene_cols)}, metadata: {len(meta_cols)}")

    metadata = df[meta_cols].copy()

    # ── Align genes ──
    model_genes = pd.read_csv(args.model_vocab)["genes"].tolist()
    tcga_genes = pd.read_csv(args.tcga_vocab)["genes"].tolist()

    tcga_gene_to_col = {g: i for i, g in enumerate(tcga_gene_cols)}
    tcga_vocab_to_idx = {g: i for i, g in enumerate(tcga_genes)}

    col_indices = []
    valid_mask = []
    for mg in model_genes:
        idx = tcga_gene_to_col.get(mg, tcga_vocab_to_idx.get(mg, -1))
        col_indices.append(idx)
        valid_mask.append(idx >= 0)

    n_missing = num_genes - sum(valid_mask)
    print(f"  Genes matched: {sum(valid_mask)}/{num_genes} ({n_missing} missing)")

    X_tcga = df[tcga_gene_cols].values.astype(np.float32)

    # ── Build input aligned to model gene order ──
    N = len(df)
    X_aligned = np.zeros((N, num_genes), dtype=np.float32)
    valid_mask_np = np.array(valid_mask)
    X_aligned[:, valid_mask_np] = X_tcga[:, [c for c in col_indices if c >= 0]]

    mask_token_id = model_cfg.mask_token_id
    valid_mask_t = torch.tensor(valid_mask, device=device)
    missing_indices = torch.where(~valid_mask_t)[0]  # shape [n_missing]
    missing_indices_batched = missing_indices.unsqueeze(0)  # [1, n_missing]

    empty_mask = torch.zeros(args.batch_size, 0, dtype=torch.long, device=device)

    print(f"  Aligned matrix: {X_aligned.shape}")

    # ── Extract embeddings ──
    print(f"Extracting embeddings (batch_size={args.batch_size})...")
    t0 = time.time()
    all_embeddings = []

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        B = end - start

        if model_cfg.masking_strategy == "cls_bottleneck":
            batch = torch.tensor(X_aligned[start:end], device=device)
            mask_idx = missing_indices_batched.expand(B, -1)
            emb = model(batch, mask_idx=mask_idx, output_cls=True)

        else:
            batch = torch.tensor(X_aligned[start:end], device=device)
            batch[:, ~valid_mask_t] = mask_token_id
            h = model(batch, mask_idx=empty_mask[:B],
                      output_hidden=True)
            emb = h[:, valid_mask_t].mean(dim=1)

        all_embeddings.append(emb.cpu())

        if (start // args.batch_size) % 20 == 0:
            print(f"  [{start}/{N}] ({time.time()-t0:.1f}s)")

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"  Embeddings: {embeddings.shape} ({time.time()-t0:.1f}s)")

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, args.output)
    torch.save(embeddings, out_path)
    print(f"Saved: {out_path}")

    labels_path = os.path.join(OUTPUT_DIR, "tcga_labels.parquet")
    if not os.path.exists(labels_path):
        metadata.to_parquet(labels_path)
        print(f"Saved labels: {labels_path}")
    else:
        print(f"Labels already exist: {labels_path}")

    print("Done.")


if __name__ == "__main__":
    main()
