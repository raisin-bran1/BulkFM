"""
Extract sample-level embeddings from the official BulkFormer model on TCGA.

Faithfully follows the official BulkFormer inference notebook:
  - Gene graph built as SparseTensor(row=ei[1], col=ei[0], value=weights).t()
  - Input aligned to the BulkFormer gene vocabulary (ensg_id); genes absent
    from the TCGA matrix are set to the mask token (-10).
  - mask_prob = fraction of genes missing from the input (used as an aux
    feature by the model).
  - Sample-level embedding = mean over genes of [N, genes, dim], dropping the
    3 auxiliary features (mask_scalar, expr_mean, nonzero_ratio).

Usage:
  python downstream/tcga/bulkformer_tcga_embeddings.py \
      --weights weights/bulkformer/BulkFormer_50M.pt \
      --output BulkFormer-50M.pt
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.typing import SparseTensor

from models.bulkformer.BulkFormer import BulkFormer
from models.bulkformer.Bulkformer_params import get_params

TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
OUTPUT_DIR = "embeddings"
GRAPH_PATH = "weights/bulkformer/G_tcga.pt"
GRAPH_WEIGHTS_PATH = "weights/bulkformer/G_tcga_weight.pt"
GENE_INFO_PATH = "weights/bulkformer/bulkformer_gene_info.csv"
MODEL_VARIANTS = {
    "BulkFormer_37M.pt": 1,
    "BulkFormer_50M.pt": 2,
    "BulkFormer_93M.pt": 3,
    "BulkFormer_127M.pt": 4,
    "BulkFormer_147M.pt": 0,
}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Extract BulkFormer sample embeddings on TCGA")
    parser.add_argument("--weights", type=str, default="weights/bulkformer/BulkFormer_50M.pt")
    parser.add_argument("--output", type=str, default="BulkFormer-50M.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tcga_parquet", type=str, default=TCGA_PARQUET)
    parser.add_argument("--graph", type=str, default=GRAPH_PATH)
    parser.add_argument("--graph_weights", type=str, default=GRAPH_WEIGHTS_PATH)
    parser.add_argument("--gene_info", type=str, default=GENE_INFO_PATH)
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no mixed precision)")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # ── Pick model config from checkpoint name ──
    base = os.path.basename(args.weights)
    idx = MODEL_VARIANTS.get(base)
    if idx is None:
        for name, i in MODEL_VARIANTS.items():
            if name in args.weights:
                idx = i
                break
    if idx is None:
        print(f"Warning: unknown model variant for {base}, defaulting to BulkFormer-50M")
        idx = 2
    params = get_params(idx)
    print(f"Model variant: {base} | dim={params['dim']}, p_repeat={params['p_repeat']}")

    # ── Load graph (official construction) ──
    print("Loading gene graph...")
    edge_index = torch.load(args.graph, map_location="cpu", weights_only=True)
    edge_weight = torch.load(args.graph_weights, map_location="cpu", weights_only=True)
    graph = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight).t().to(device)

    # ── Build model ──
    model = BulkFormer(
        dim=params["dim"],
        graph=graph,
        gene_emb=None,
        gene_length=params["gene_length"],
        bin_head=params["bin_head"],
        full_head=params["full_head"],
        bins=params["bins"],
        gb_repeat=params["gb_repeat"],
        p_repeat=params["p_repeat"],
    ).to(device)

    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing state dict keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected state dict keys skipped: {len(unexpected)}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} params")

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

    # ── Align TCGA genes to BulkFormer vocabulary ──
    gene_info = pd.read_csv(args.gene_info)
    bf_genes = gene_info["ensg_id"].tolist()
    tcga_col_to_idx = {g: i for i, g in enumerate(tcga_gene_cols)}

    X_tcga = df[tcga_gene_cols].values.astype(np.float32)

    N = len(df)
    n_genes = len(bf_genes)
    X_aligned = np.full((N, n_genes), -10.0, dtype=np.float32)
    present = np.zeros(n_genes, dtype=bool)
    for i, gene in enumerate(bf_genes):
        j = tcga_col_to_idx.get(gene)
        if j is not None:
            X_aligned[:, i] = X_tcga[:, j]
            present[i] = True

    n_missing = n_genes - present.sum()
    mask_prob = n_missing / n_genes
    print(f"  Genes matched: {present.sum()}/{n_genes} ({n_missing} missing, mask_prob={mask_prob:.4f})")

    # ── Extract sample-level embeddings ──
    print(f"Extracting embeddings (batch_size={args.batch_size})...")
    t0 = time.time()
    all_embeddings = []
    amp_ctx = (torch.amp.autocast("cuda", enabled=True) if device.type == "cuda"
               else torch.amp.autocast("cuda", enabled=False))
    with torch.no_grad(), amp_ctx:
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            # Retry with a smaller batch on CUDA OOM (GPU is shared with other jobs).
            bsz = end - start
            while True:
                try:
                    batch = torch.tensor(X_aligned[start:start + bsz], device=device)
                    gene_emb = model(batch, mask_prob=mask_prob, output_expr=False)
                    final_emb = torch.mean(gene_emb, dim=1)[:, :-3]
                    all_embeddings.append(final_emb.float().cpu())
                    del gene_emb, final_emb, batch
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bsz == 1:
                        raise
                    bsz = max(1, bsz // 2)
                    print(f"  [OOM] retrying slice [{start},{start + bsz}) with batch_size={bsz}")

            if (start // args.batch_size) % 250 == 0:
                elapsed = time.time() - t0
                rate = (start + (end - start)) / max(elapsed, 1e-6)
                print(f"  [{end}/{N}] ({elapsed:.1f}s, {rate:.0f} samples/s)")

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
