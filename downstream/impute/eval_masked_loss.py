"""
Evaluate a trained BulkFM checkpoint with masked reconstruction loss
(the same loss as train.py's validation) on either:

  * the last ARCHS4 chunk (data_dir/processed_human_<last>.parquet), or
  * TCGA (tcga_processed.parquet, genes aligned to the model vocab).

Only supports continuous-expression models (mask_token / cls_bottleneck
strategies). The masking ratio and random seed are configurable; the masking
recipe (mask_token vs random token vs keep) follows the checkpoint's config.

Usage:
  python downstream/impute/eval_masked_loss.py \
      --checkpoint checkpoints/train_20260810_072705_local/BulkFM-FULL-VAR.pt \
      --data archs4_last --mask_ratio 0.15 --seed 42

  python downstream/impute/eval_masked_loss.py \
      --checkpoint checkpoints/train_20260809_075922_local/best_model.pt \
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

from models.bulkfm import BulkFM, BulkFMConfig, PoissonNLLLogSpace

ARCHS4_DIR = "/media/volume/bulkrnadata/humandata"
TCGA_PARQUET = "/media/volume/bulkrnadata/tcgadata/tcga_processed.parquet"
TCGA_VOCAB = "/media/volume/bulkrnadata/tcgadata/tcga_gene_vocabulary.csv"
MODEL_VOCAB = "checkpoints/gene_vocabulary.csv"
DEFAULT_OUTPUT = "results/eval_masked_loss.csv"


def _strip_state_dict(state_dict):
    return {k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _strip_state_dict(checkpoint.get("model_state_dict", checkpoint))
    ckpt_cfg = checkpoint.get("config", {})

    model_cfg = BulkFMConfig(
        hidden_dim=ckpt_cfg.get("hidden_dim", 256),
        ffn_dim=ckpt_cfg.get("ffn_dim", 1024),
        num_heads=ckpt_cfg.get("num_heads", 8),
        num_layers=ckpt_cfg.get("num_layers", 4),
        feature_type=ckpt_cfg.get("feature_type", "elu+1"),
        compute_type=ckpt_cfg.get("compute_type", "iter"),
        expression_embedding=ckpt_cfg.get("expression_embedding", "continuous"),
        num_bins=ckpt_cfg.get("num_bins", 50),
        continuous_loss=ckpt_cfg.get("continuous_loss", "mse"),
        mask_ratio=ckpt_cfg.get("mask_ratio", 0.15),
        dynamic_mask_range=ckpt_cfg.get("dynamic_mask_range"),
        mask_token_id=ckpt_cfg.get("mask_token", -10),
        masking_strategy=ckpt_cfg.get("masking_strategy", "mask_token"),
        simple_projection=ckpt_cfg.get("expression_projection", "nonlinear") == "linear",
        sample_level_emb=ckpt_cfg.get("sample_level_emb", 0),
    )

    num_genes = state_dict["gene_embedding.weight"].shape[0]
    model = BulkFM(num_genes, model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_cfg, ckpt_cfg


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


def make_mask(x, valid_positions, ratio, mask_token, mask_p, rand_p, strategy, device):
    """Mask ``ratio`` of valid genes per sample.

    Returns (x_masked, mask_idx). mask_idx indexes columns of x.
    """
    B = x.shape[0]
    Gm = len(valid_positions)
    num_mask = max(1, int(Gm * ratio))
    mask_idx = torch.stack([
        valid_positions[torch.randperm(Gm, device=device)[:num_mask]]
        for _ in range(B)
    ])

    if strategy == "mask_token":
        x_masked = x.clone().contiguous()
        flat = x_masked.view(-1)
        probs = torch.rand(B, num_mask, device=device)
        is_mask = probs < mask_p
        is_rand = (probs >= mask_p) & (probs < mask_p + rand_p)
        mask_pos = is_mask.nonzero()
        flat[mask_pos[:, 0] * x.shape[1] + mask_idx[mask_pos[:, 0], mask_pos[:, 1]]] = mask_token
        rand_pos = is_rand.nonzero()
        if len(rand_pos) > 0:
            nonzero = x_masked[x_masked > 0]
            if len(nonzero) > 0:
                rand_vals = nonzero[torch.randint(len(nonzero), (len(rand_pos),), device=device)]
            else:
                rand_vals = torch.empty(len(rand_pos), device=device).uniform_(0, 10)
            flat[rand_pos[:, 0] * x.shape[1] + mask_idx[rand_pos[:, 0], rand_pos[:, 1]]] = rand_vals
    else:
        x_masked = x.clone()

    return x_masked, mask_idx


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate masked reconstruction loss")
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

    model, model_cfg, ckpt_cfg = load_checkpoint(args.checkpoint, device)
    num_genes = model.num_genes
    if model_cfg.expression_embedding == "binned":
        raise SystemExit("This script only supports continuous-expression models; "
                         f"checkpoint uses expression_embedding={model_cfg.expression_embedding}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Model: {num_genes} genes, expr={model_cfg.expression_embedding}, "
          f"strategy={model_cfg.masking_strategy}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    mask_token = model_cfg.mask_token_id
    mask_p = ckpt_cfg.get("mask_token_prob", 0.8)
    rand_p = ckpt_cfg.get("random_token_prob", 0.1)
    if isinstance(rand_p, str):
        rand_p = (1.0 - mask_p) / 2.0

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
          f"({n_missing} missing)  [{time.time()-t0:.1f}s]")

    missing_pos = torch.nonzero(~torch.tensor(valid)).reshape(-1).to(device)
    valid_pos = torch.nonzero(torch.tensor(valid)).reshape(-1).to(device)

    targets = torch.tensor(X)
    if model_cfg.continuous_loss == "poisson":
        cont_loss_fn = PoissonNLLLogSpace()
    else:
        cont_loss_fn = F.mse_loss

    Xt = torch.tensor(X)
    if n_missing > 0:
        Xt[:, ~torch.tensor(valid)] = mask_token

    if model_cfg.masking_strategy == "cls_bottleneck" and n_missing > 0:
        extra_missing = missing_pos.expand(len(Xt), -1)
    else:
        extra_missing = None

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

        x_masked, mask_idx = make_mask(
            x_batch, valid_pos, args.mask_ratio, mask_token, mask_p, rand_p,
            model_cfg.masking_strategy, device)

        model_mask = mask_idx
        if extra_missing is not None:
            model_mask = torch.cat([extra_missing[:B], mask_idx], dim=1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x_masked, mask_idx=model_mask)

            t = targets[start:end].to(device)
            biv = torch.arange(B, device=device).unsqueeze(1).expand(-1, mask_idx.shape[1])
            masked_out = out[biv, mask_idx]
            masked_targets = t[biv, mask_idx]
            total_masked += masked_targets.numel()

            if model_cfg.continuous_loss == "poisson":
                loss = cont_loss_fn(masked_out.squeeze(-1), masked_targets.log1p()).mean()
            else:
                if masked_out.dim() == 3:
                    masked_out = masked_out.squeeze(-1)
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
