"""
Evaluate a trained MLPAutoencoder checkpoint with masked reconstruction loss
(the same objective as train_autoencoder.py) on either:

  * the last ARCHS4 chunk (data_dir/processed_human_<last>.parquet), or
  * TCGA (tcga_processed.parquet, genes aligned to the model vocab).

Mirror of eval_masked_loss.py for models/autoencoder.py: only continuous
expression, masked genes are zeroed by the model and MSE is computed only on
the masked positions. Missing genes (absent from the data) are zero-filled.
Masking is restricted to genes present in the data and follows the fixed
--mask_ratio / --seed, so results are comparable with BulkFM's eval.

Usage:
  python downstream/impute/eval_masked_loss_mlp.py \
      --checkpoint checkpoints/train_mlp_20260811_225111_local/MLP-BIG.pt \
      --data archs4_last --mask_ratio 0.15 --seed 42

  python downstream/impute/eval_masked_loss_mlp.py \
      --checkpoint checkpoints/train_mlp_<run>/best_model.pt \
      --data tcga --mask_ratio 0.15 --seed 42 --num_samples 2000
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.autoencoder import MLPAutoencoder, MLPAutoencoderConfig

ARCHS4_DIR = "/media/volume/bulkrnadata/humandata"
TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
TCGA_VOCAB = "/media/volume/bulkrnadata/tcgadata/tcga_gene_vocabulary.csv"
MODEL_VOCAB = "checkpoints/gene_vocabulary.csv"
DEFAULT_OUTPUT = "results/eval_masked_loss_mlp.csv"


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


def load_archs4_last(archs4_dir, model_genes):
    files = sorted(Path(archs4_dir).glob("processed_human_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed_human_*.parquet in {archs4_dir}")
    path = files[-1]
    df = pd.read_parquet(str(path))
    gene_cols = [c for c in df.columns if c != "sample_id"]
    X = df[gene_cols].values.astype(np.float32)

    col_to_idx = {g: i for i, g in enumerate(gene_cols)}
    aligned = np.zeros((len(X), len(model_genes)), dtype=np.float32)
    valid = np.zeros(len(model_genes), dtype=bool)
    for i, g in enumerate(model_genes):
        if g in col_to_idx:
            aligned[:, i] = X[:, col_to_idx[g]]
            valid[i] = True
    return aligned, valid, str(path)


def load_tcga(tcga_parquet, tcga_vocab, model_genes):
    df = pd.read_parquet(tcga_parquet)
    vocab = set(pd.read_csv(tcga_vocab)["genes"].tolist())
    gene_cols = [c for c in df.columns if c in vocab]
    X = df[gene_cols].values.astype(np.float32)

    col_to_idx = {g: i for i, g in enumerate(gene_cols)}
    aligned = np.zeros((len(X), len(model_genes)), dtype=np.float32)
    valid = np.zeros(len(model_genes), dtype=bool)
    for i, g in enumerate(model_genes):
        if g in col_to_idx:
            aligned[:, i] = X[:, col_to_idx[g]]
            valid[i] = True
    return aligned, valid, gene_cols


def make_mask(x, valid_positions, ratio, device):
    """Mask ``ratio`` of valid genes per sample; returns mask_idx only.

    The model zeroes masked genes internally, so the input tensor is left
    untouched (unlike eval_masked_loss.py's make_mask for BulkFM).
    """
    B = x.shape[0]
    Gm = len(valid_positions)
    num_mask = max(1, int(Gm * ratio))
    mask_idx = torch.stack([
        valid_positions[torch.randperm(Gm, device=device)[:num_mask]]
        for _ in range(B)
    ])
    return mask_idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP masked reconstruction loss")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, choices=["archs4_last", "tcga"], default="archs4_last")
    parser.add_argument("--archs4_dir", type=str, default=ARCHS4_DIR)
    parser.add_argument("--tcga_parquet", type=str, default=TCGA_PARQUET)
    parser.add_argument("--tcga_vocab", type=str, default=TCGA_VOCAB)
    parser.add_argument("--model_vocab", type=str, default=MODEL_VOCAB)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, model_cfg, _ = load_mlp_checkpoint(args.checkpoint, device)
    num_genes = model.num_genes
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Model: {num_genes} genes, hidden {list(model_cfg.hidden_dims)}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    model_genes = pd.read_csv(args.model_vocab)["genes"].tolist()
    if len(model_genes) != num_genes:
        print(f"  [WARN] vocab ({len(model_genes)}) != model genes ({num_genes}); using model size")

    print(f"\nLoading data: {args.data}")
    t0 = time.time()
    if args.data == "archs4_last":
        X, valid, src = load_archs4_last(args.archs4_dir, model_genes)
        print(f"  Source: {src}")
    else:
        X, valid, src = load_tcga(args.tcga_parquet, args.tcga_vocab, model_genes)
        print(f"  Source: {args.tcga_parquet}")
        print(f"  TCGA gene columns matched: {len(src)}")

    if args.num_samples:
        X = X[:args.num_samples]
    n_missing = int((~valid).sum())
    print(f"  Samples: {X.shape[0]} × {X.shape[1]}, genes matched: {int(valid.sum())}/{num_genes} "
          f"({n_missing} missing, zero-filled)  [{time.time()-t0:.1f}s]")

    valid_pos = torch.nonzero(torch.tensor(valid)).reshape(-1).to(device)
    targets = torch.tensor(X)
    Xt = torch.tensor(X)

    G = Xt.shape[1]
    gene_mean = Xt.mean(dim=0).to(device)

    loss_sum = torch.tensor(0.0, device=device)
    r2_sum = torch.tensor(0.0, device=device)
    pcc_sum = torch.tensor(0.0, device=device)
    pcc_count = 0
    n_batches = 0
    total_masked = 0

    baseline_loss_sum = torch.tensor(0.0, device=device)
    baseline_r2_sum = torch.tensor(0.0, device=device)
    baseline_pcc_sum = torch.tensor(0.0, device=device)
    baseline_pcc_count = 0

    pooled_p = torch.tensor([], device="cpu")
    pooled_t = torch.tensor([], device="cpu")
    base_pooled_p = torch.tensor([], device="cpu")

    zero = torch.zeros(G, device="cpu")
    gs_pred = zero.clone(); gs_tgt = zero.clone()
    gs_pred2 = zero.clone(); gs_tgt2 = zero.clone()
    gs_prod = zero.clone(); gs_n = zero.clone()
    b_gs_pred = zero.clone(); b_gs_tgt = zero.clone()
    b_gs_pred2 = zero.clone(); b_gs_tgt2 = zero.clone()
    b_gs_prod = zero.clone(); b_gs_n = zero.clone()

    print(f"\nEvaluating (mask_ratio={args.mask_ratio}, seed={args.seed})...")
    t0 = time.time()
    for start in range(0, len(Xt), args.batch_size):
        end = min(start + args.batch_size, len(Xt))
        x_batch = Xt[start:end].to(device)
        B = end - start

        mask_idx = make_mask(x_batch, valid_pos, args.mask_ratio, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x_batch, mask_idx=mask_idx)

            t = targets[start:end].to(device)
            biv = torch.arange(B, device=device).unsqueeze(1).expand(-1, mask_idx.shape[1])
            masked_out = out[biv, mask_idx]
            masked_targets = t[biv, mask_idx]
            baseline_out = gene_mean[mask_idx]
            total_masked += masked_targets.numel()

            loss = F.mse_loss(masked_out, masked_targets)
            loss_sum += loss.detach().float()
            ss_res = ((masked_targets - masked_out) ** 2).sum()
            ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
            r2_sum += (1 - ss_res / ss_tot.clamp(min=1e-8)).detach().float()

            p = masked_out - masked_out.mean(dim=1, keepdim=True)
            t_c = masked_targets - masked_targets.mean(dim=1, keepdim=True)
            denom = torch.sqrt((p ** 2).sum(1) * (t_c ** 2).sum(1))
            valid_den = denom > 1e-8
            pcc_sum += ((p * t_c).sum(1) / denom.clamp(min=1e-8))[valid_den].sum().detach().float()
            pcc_count += valid_den.sum().item()

            bloss = F.mse_loss(baseline_out, masked_targets)
            baseline_loss_sum += bloss.detach().float()
            b_ss_res = ((masked_targets - baseline_out) ** 2).sum()
            baseline_r2_sum += (1 - b_ss_res / ss_tot.clamp(min=1e-8)).detach().float()
            bp = baseline_out - baseline_out.mean(dim=1, keepdim=True)
            b_denom = torch.sqrt((bp ** 2).sum(1) * (t_c ** 2).sum(1))
            b_valid_den = b_denom > 1e-8
            baseline_pcc_sum += ((bp * t_c).sum(1) / b_denom.clamp(min=1e-8))[b_valid_den].sum().detach().float()
            baseline_pcc_count += b_valid_den.sum().item()

            pooled_p = torch.cat([pooled_p, masked_out.detach().float().reshape(-1).cpu()])
            pooled_t = torch.cat([pooled_t, masked_targets.detach().float().reshape(-1).cpu()])
            base_pooled_p = torch.cat([base_pooled_p, baseline_out.detach().float().reshape(-1).cpu()])

            f_gi = mask_idx.reshape(-1).cpu()
            f_p = masked_out.detach().float().reshape(-1).cpu()
            f_t = masked_targets.detach().float().reshape(-1).cpu()
            f_bp = baseline_out.detach().float().reshape(-1).cpu()
            gs_pred.index_add_(0, f_gi, f_p); gs_tgt.index_add_(0, f_gi, f_t)
            gs_pred2.index_add_(0, f_gi, f_p ** 2); gs_tgt2.index_add_(0, f_gi, f_t ** 2)
            gs_prod.index_add_(0, f_gi, f_p * f_t)
            gs_n.index_add_(0, f_gi, torch.ones_like(f_p))
            b_gs_pred.index_add_(0, f_gi, f_bp); b_gs_tgt.index_add_(0, f_gi, f_t)
            b_gs_pred2.index_add_(0, f_gi, f_bp ** 2); b_gs_tgt2.index_add_(0, f_gi, f_t ** 2)
            b_gs_prod.index_add_(0, f_gi, f_bp * f_t)
            b_gs_n.index_add_(0, f_gi, torch.ones_like(f_p))

        n_batches += 1
        if n_batches % 10 == 0:
            print(f"  [{end}/{len(Xt)}] ({time.time()-t0:.1f}s)")

    loss = (loss_sum / n_batches).item()
    print(f"\n  Done in {time.time()-t0:.1f}s. Masked positions evaluated: {total_masked:,}")

    n = gs_n.clamp(min=1.0)
    per_gene_den = torch.sqrt((n * gs_pred2 - gs_pred ** 2) * (n * gs_tgt2 - gs_tgt ** 2))
    per_gene_pcc = ((n * gs_prod - gs_pred * gs_tgt) / per_gene_den.clamp(min=1e-8))
    valid_g = (gs_n > 30) & (per_gene_den > 1e-8)
    per_gene_r2 = 1 - (gs_tgt2 - 2 * gs_prod + gs_pred2) / (gs_tgt2 - gs_tgt ** 2 / n).clamp(min=1e-8)

    b_n = b_gs_n.clamp(min=1.0)
    b_den = torch.sqrt((b_n * b_gs_pred2 - b_gs_pred ** 2) * (b_n * b_gs_tgt2 - b_gs_tgt ** 2))
    b_pcc = ((b_n * b_gs_prod - b_gs_pred * b_gs_tgt) / b_den.clamp(min=1e-8))
    b_valid_g = (b_gs_n > 30) & (b_den > 1e-8)
    b_r2 = 1 - (b_gs_tgt2 - 2 * b_gs_prod + b_gs_pred2) / (b_gs_tgt2 - b_gs_tgt ** 2 / b_n).clamp(min=1e-8)

    def pooled_pcc(p, t):
        pc = p - p.mean(); tc = t - t.mean()
        d = torch.sqrt((pc ** 2).sum() * (tc ** 2).sum())
        return ((pc * tc).sum() / d.clamp(min=1e-8)).item()

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
        "global_pcc": pooled_pcc(pooled_p, pooled_t),
        "per_gene_pcc_mean": per_gene_pcc[valid_g].mean().item(),
        "per_gene_pcc_median": per_gene_pcc[valid_g].median().item(),
        "pct_genes_pcc_gt_0.9": (per_gene_pcc[valid_g] > 0.9).float().mean().item(),
        "per_gene_r2_mean": per_gene_r2[valid_g].mean().item(),
        "per_gene_r2_median": per_gene_r2[valid_g].median().item(),
        "baseline_loss": (baseline_loss_sum / n_batches).item(),
        "baseline_r2": (baseline_r2_sum / n_batches).item(),
        "baseline_pcc": (baseline_pcc_sum / max(baseline_pcc_count, 1)).item(),
        "baseline_global_pcc": pooled_pcc(base_pooled_p, pooled_t),
        "baseline_per_gene_pcc_mean": b_pcc[b_valid_g].mean().item(),
        "baseline_per_gene_r2_mean": b_r2[b_valid_g].mean().item(),
    }
    print(f"  Loss: {loss:.6f} | R2: {row['r2']:.4f} | PCC: {row['pcc']:.4f} | global PCC: {row['global_pcc']:.4f}")
    print(f"  Per-gene PCC: mean {row['per_gene_pcc_mean']:.4f} median {row['per_gene_pcc_median']:.4f} "
          f"(>{0.9}: {row['pct_genes_pcc_gt_0.9']*100:.1f}%) | per-gene R2: mean {row['per_gene_r2_mean']:.4f} "
          f"median {row['per_gene_r2_median']:.4f}")
    print(f"  Mean-imputation baseline: PCC {row['baseline_pcc']:.4f} global {row['baseline_global_pcc']:.4f} "
          f"R2 {row['baseline_r2']:.4f} | per-gene PCC {row['baseline_per_gene_pcc_mean']:.4f} "
          f"R2 {row['baseline_per_gene_r2_mean']:.4f}")

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
