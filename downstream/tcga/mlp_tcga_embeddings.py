"""
Extract sample-level embeddings from a trained MLPAutoencoder on TCGA.

The MLP has no CLS token and no per-gene token states, so the sample embedding
is the encoder output (last hidden state, hidden_dims[-1] e.g. 256-dim),
exposed by the model via output_hidden=True. Missing TCGA genes are
zero-filled (0 = undetected, consistent with log1p(TPM)=0).

The output .pt file drops straight into the model-agnostic benchmark stage
(classification / reconstruction / visualization).

Usage:
  python downstream/tcga/mlp_tcga_embeddings.py \
      --checkpoint checkpoints/train_mlp_20260811_214938_local/best_model.pt \
      --output tcga_mlp.pt
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
from models.autoencoder import MLPAutoencoder, MLPAutoencoderConfig

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
TCGA_VOCAB = "/media/volume/bulkrnadata/tcgadata/tcga_gene_vocabulary.csv"
MODEL_VOCAB = "checkpoints/gene_vocabulary.csv"
OUTPUT_DIR = "embeddings"


def _strip_state_dict(state_dict):
    return {k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()}


def load_mlp_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _strip_state_dict(checkpoint.get("model_state_dict", checkpoint))
    cfg = checkpoint.get("config", {})

    model_cfg = MLPAutoencoderConfig(
        hidden_dims=tuple(cfg.get("hidden_dims", (256, 512, 256))),
        mask_ratio=cfg.get("mask_ratio", 0.15),
        dynamic_mask_range=cfg.get("dynamic_mask_range"),
        mask_value=0.0,
    )

    num_genes = state_dict["mlp.0.weight"].shape[1]
    model = MLPAutoencoder(num_genes, model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_cfg, cfg


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Extract MLP sample embeddings on TCGA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="tcga_mlp.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tcga_parquet", type=str, default=TCGA_PARQUET)
    parser.add_argument("--tcga_vocab", type=str, default=TCGA_VOCAB)
    parser.add_argument("--model_vocab", type=str, default=MODEL_VOCAB)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_cfg, _ = load_mlp_checkpoint(args.checkpoint, device)
    num_genes = model.num_genes
    emb_dim = model_cfg.hidden_dims[-1]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Model: {num_genes} genes, hidden {list(model_cfg.hidden_dims)}, "
          f"emb_dim={emb_dim}, {sum(p.numel() for p in model.parameters()):,} params")

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
    N = len(df)
    X_aligned = np.zeros((N, num_genes), dtype=np.float32)
    valid_mask_np = np.array(valid_mask)
    X_aligned[:, valid_mask_np] = X_tcga[:, [c for c in col_indices if c >= 0]]
    print(f"  Aligned matrix: {X_aligned.shape} (missing genes zero-filled)")

    # ── Extract embeddings ──
    print(f"Extracting embeddings (batch_size={args.batch_size})...")
    t0 = time.time()
    all_embeddings = []
    empty_mask = torch.zeros(args.batch_size, 0, dtype=torch.long, device=device)

    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        B = end - start
        batch = torch.tensor(X_aligned[start:end], device=device)
        emb = model(batch, mask_idx=empty_mask[:B], output_hidden=True)
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
