"""
Evaluate the BulkFormer checkpoint with masked reconstruction loss on either:

  * the last ARCHS4 chunk (data_dir/processed_human_<last>.parquet), or
  * TCGA (tcga_processed.parquet), genes aligned to the BulkFormer vocab.

Masking follows the BulkFormer recipe: masked genes are set to -10 (the
model's mask token); targets are log1p(TPM) values. Genes in the BulkFormer
vocab but absent from the data are filled with 0.0 (undetected -> log1p(0)),
not mask tokens. Appends the result row to the same results CSV as
eval_masked_loss.py (checkpoint column records the BulkFormer checkpoint name).

Usage:
  python downstream/impute/eval_masked_loss_bulkformer.py \
      --checkpoint weights/bulkformer/BulkFormer_147M.pt \
      --data archs4_last --mask_ratio 0.15 --seed 42

  python downstream/impute/eval_masked_loss_bulkformer.py \
      --checkpoint weights/bulkformer/BulkFormer_147M.pt \
      --data tcga --mask_ratio 0.15 --seed 42
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor

from models.bulkformer.BulkFormer import BulkFormer
from models.bulkformer.Bulkformer_params import get_params

ARCHS4_DIR = "/media/volume/bulkrnadata/humandata"
TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
GENE_INFO_PATH = "weights/bulkformer/bulkformer_gene_info.csv"
GRAPH_PATH = "weights/bulkformer/G_tcga.pt"
GRAPH_WEIGHTS_PATH = "weights/bulkformer/G_tcga_weight.pt"
DEFAULT_OUTPUT = "results/eval_masked_loss.csv"
MASK_TOKEN = -10.0


def load_model(checkpoint, device):
    edge_index = torch.load(GRAPH_PATH, map_location="cpu", weights_only=True)
    edge_weight = torch.load(GRAPH_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    graph = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight).t().to(device)

    params = get_params(0)  # BulkFormer-147M
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

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, params


def load_gene_list(gene_info_path):
    df = pd.read_csv(gene_info_path)
    return df["ensg_id"].dropna().astype(str).tolist()


def load_data(data_name, model_genes, num_samples):
    gene_set = set(model_genes)
    if data_name == "archs4_last":
        files = sorted(Path(ARCHS4_DIR).glob("processed_human_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No processed_human_*.parquet in {ARCHS4_DIR}")
        path = files[-1]
        df = pd.read_parquet(str(path))
        cols = [c for c in df.columns if c != "sample_id" and c in gene_set]
        src = str(path)
    else:
        df = pd.read_parquet(TCGA_PARQUET)
        cols = [c for c in df.columns if c in gene_set]
        src = TCGA_PARQUET

    X = df[cols].values.astype(np.float32)
    col_to_idx = {g: i for i, g in enumerate(cols)}
    aligned = np.zeros((len(X), len(model_genes)), dtype=np.float32)
    valid = np.zeros(len(model_genes), dtype=bool)
    for i, g in enumerate(model_genes):
        if g in col_to_idx:
            aligned[:, i] = X[:, col_to_idx[g]]
            valid[i] = True

    if num_samples:
        aligned = aligned[:num_samples]
    return aligned, valid, src, len(cols)


def make_mask(x, valid_pos, ratio, device):
    B = x.shape[0]
    Gm = len(valid_pos)
    num_mask = max(1, int(Gm * ratio))
    mask_idx = torch.stack([
        valid_pos[torch.randperm(Gm, device=device)[:num_mask]]
        for _ in range(B)
    ])
    x_masked = x.clone()
    flat = x_masked.view(-1)
    rows = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask).reshape(-1)
    flat[rows * x.shape[1] + mask_idx.reshape(-1)] = MASK_TOKEN
    return x_masked, mask_idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate BulkFormer masked reconstruction loss")
    parser.add_argument("--checkpoint", type=str,
                        default="weights/bulkformer/BulkFormer_147M.pt")
    parser.add_argument("--data", type=str, choices=["archs4_last", "tcga"], default="archs4_last")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, params = load_model(args.checkpoint, device)
    num_genes = params["gene_length"]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Model: {num_genes} genes, dim={params['dim']}, p_repeat={params['p_repeat']}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    model_genes = load_gene_list(GENE_INFO_PATH)
    if len(model_genes) != num_genes:
        print(f"  [WARN] gene_info ({len(model_genes)}) != model genes ({num_genes})")

    print(f"\nLoading data: {args.data}")
    t0 = time.time()
    X, valid, src, n_cols = load_data(args.data, model_genes, args.num_samples)
    print(f"  Source: {src}")
    if args.data == "tcga":
        print(f"  TCGA gene columns matched: {n_cols}")
    n_missing = int((~valid).sum())
    print(f"  Samples: {X.shape[0]} × {X.shape[1]}, genes matched: {int(valid.sum())}/{num_genes} "
          f"({n_missing} missing)  [{time.time()-t0:.1f}s]")

    valid_pos = torch.nonzero(torch.tensor(valid)).reshape(-1).to(device)
    targets = torch.tensor(X)
    Xt = torch.tensor(X)

    loss_sum = torch.tensor(0.0, device=device)
    r2_sum = torch.tensor(0.0, device=device)
    pcc_sum = torch.tensor(0.0, device=device)
    pcc_count = 0
    n_batches = 0
    total_masked = 0

    print(f"\nEvaluating (mask_ratio={args.mask_ratio}, seed={args.seed})...")
    t0 = time.time()
    for start in range(0, len(Xt), args.batch_size):
        end = min(start + args.batch_size, len(Xt))
        x_batch = Xt[start:end].to(device)
        B = end - start

        x_masked, mask_idx = make_mask(x_batch, valid_pos, args.mask_ratio, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=False):
            out = model(x_masked, mask_prob=args.mask_ratio, output_expr=True)

            t = targets[start:end].to(device)
            biv = torch.arange(B, device=device).unsqueeze(1).expand(-1, mask_idx.shape[1])
            masked_out = out[biv, mask_idx]
            masked_targets = t[biv, mask_idx]
            total_masked += masked_targets.numel()

            loss = F.mse_loss(masked_out, masked_targets)
            loss_sum += loss.detach().float()
            ss_res = ((masked_targets - masked_out) ** 2).sum()
            ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
            r2_sum += (1 - ss_res / ss_tot.clamp(min=1e-8)).detach().float()

            p = masked_out - masked_out.mean(dim=1, keepdim=True)
            t = masked_targets - masked_targets.mean(dim=1, keepdim=True)
            denom = torch.sqrt((p ** 2).sum(1) * (t ** 2).sum(1))
            valid_den = denom > 1e-8
            pcc_sum += ((p * t).sum(1) / denom.clamp(min=1e-8))[valid_den].sum().detach().float()
            pcc_count += valid_den.sum().item()

        n_batches += 1
        if n_batches % 10 == 0:
            print(f"  [{end}/{len(Xt)}] ({time.time()-t0:.1f}s)")

    loss = (loss_sum / n_batches).item()
    print(f"\n  Done in {time.time()-t0:.1f}s. Masked positions evaluated: {total_masked:,}")

    row = {
        "checkpoint": Path(args.checkpoint).name,
        "data": args.data,
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "samples": len(Xt),
        "genes_matched": int(valid.sum()),
        "n_masked": total_masked,
        "loss": loss,
        "r2": (r2_sum / n_batches).item(),
        "pcc": (pcc_sum / max(pcc_count, 1)).item(),
    }
    print(f"  Loss: {loss:.6f} | R2: {row['r2']:.4f} | PCC: {row['pcc']:.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results = pd.DataFrame([row])
        if os.path.exists(args.output):
            prev = pd.read_csv(args.output)
            results = pd.concat([prev, results], ignore_index=True)
        results.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
