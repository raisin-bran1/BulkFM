# coding=utf-8
"""Train the MLP autoencoder baseline (models/autoencoder.py).

Mirrors training/train.py so runs are directly comparable with BulkFM: DDP
via torchrun, chunked data loading, dynamic masking (ratio sampled uniformly
in dynamic_mask_range, default [0.15, 0.75]), validation at a fixed mask
ratio, early stopping, and BulkFM-format checkpoints (model_state_dict +
config). The autoencoder supports continuous expression only: masked genes are
zeroed and MSE is computed only on the masked positions.

Configuration: shared data/optimization settings come from training/config.py;
autoencoder-specific values (architecture, epochs) come from
training/autoencoder_config.py, which overrides them.

Usage:
  torchrun --nproc_per_node=4 training/train_autoencoder.py
"""

import os
import sys
import time
import json
import math
import logging
import argparse
import random
from pathlib import Path

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed as dist
import pandas as pd

from training.config import CONFIG as BASE_CONFIG
from training.config_mlp import CONFIG as AUTO_CONFIG
from training.data import (
    get_sample_indices, load_batch_data, ExpressionMLMDataset,
    get_num_genes_from_batches, group_indices_by_chunk, apply_dynamic_mask,
)
from training.utils import _coerce_config_types
from models.autoencoder import MLPAutoencoder, MLPAutoencoderConfig

# Effective config: shared training config, overridden by autoencoder-specific
# values (architecture, epochs, ...) from training/autoencoder_config.py.
TRAIN_CONFIG = {**BASE_CONFIG, **AUTO_CONFIG}

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_logging(ckpt_dir, rank):
    """Set up logging to file and console (same layout as train.py)."""
    log_file = ckpt_dir / f"train_rank{rank}.log"
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if rank == 0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="MLPAutoencoder Training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args, _ = parser.parse_known_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    # Unique run ID (same scheme as train.py).
    ckpt_base = Path(TRAIN_CONFIG['checkpoint_dir'])
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_prefix = "smoke_mlp" if os.environ.get("USE_SMOKE", "0") == "1" else "train_mlp"
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_id_list = [f"{run_prefix}_{run_timestamp}_{slurm_job_id}" if is_main else None]
    dist.broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]
    ckpt_dir = ckpt_base / run_id
    if is_main:
        ckpt_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logging(ckpt_dir, rank)

    # Broadcast config from rank 0 so all ranks use the same hyperparams.
    config_list = [TRAIN_CONFIG if is_main else None]
    dist.broadcast_object_list(config_list, src=0)
    TRAIN_CONFIG.update(config_list[0])

    if TRAIN_CONFIG.get('random_token_prob') == 'auto':
        mask_p = float(TRAIN_CONFIG.get('mask_token_prob', 0.8))
        TRAIN_CONFIG['random_token_prob'] = (1.0 - mask_p) / 2.0
    _coerce_config_types(TRAIN_CONFIG)

    # ── DATA ──────────────────────────────────────────────────────
    data_dir = Path(TRAIN_CONFIG['data_dir'])
    batch_dir = data_dir / "batch_files"
    if not batch_dir.exists():
        batch_dir = data_dir

    if is_main:
        logger.info("[DATA] Building sample indices from chunks...")
    t0 = time.time()
    train_indices = val_indices = None
    if is_main:
        train_indices, val_indices = get_sample_indices(
            batch_dir,
            train_chunks=TRAIN_CONFIG.get('train_chunks'),
            val_chunks=TRAIN_CONFIG.get('val_chunks'),
            train_subset=TRAIN_CONFIG.get('train_subset'),
            val_subset=TRAIN_CONFIG.get('val_subset'),
            seed=TRAIN_CONFIG['seed'],
            verbose=True,
        )
    broadcast_list = [train_indices if is_main else None,
                      val_indices if is_main else None]
    dist.broadcast_object_list(broadcast_list, src=0)
    train_indices, val_indices = broadcast_list
    if is_main:
        logger.info(f"  ✓ Index time: {time.time()-t0:.1f}s")

    X_val = load_batch_data(batch_dir, val_indices, verbose=is_main)
    num_genes = get_num_genes_from_batches(batch_dir)

    dataset_kwargs = {
        'mask_ratio': TRAIN_CONFIG['mask_ratio'],
        'dynamic_mask_range': TRAIN_CONFIG.get('dynamic_mask_range'),
        'mask_token': TRAIN_CONFIG['mask_token'],
        'mask_token_prob': TRAIN_CONFIG.get('mask_token_prob', 0.8),
        'random_token_prob': TRAIN_CONFIG.get('random_token_prob', 0.1),
        'num_bins': TRAIN_CONFIG['num_bins'],
        'expression_embedding': 'continuous',
        'masking_strategy': 'mask_token',
    }
    val_ds = ExpressionMLMDataset(X_val, **dataset_kwargs)

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                     rank=rank, shuffle=False, seed=42)
    loader_kwargs = {'num_workers': int(TRAIN_CONFIG.get('num_workers', 0)),
                     'pin_memory': True}
    if loader_kwargs['num_workers'] > 0:
        loader_kwargs['prefetch_factor'] = int(TRAIN_CONFIG.get('prefetch_factor', 2))
        loader_kwargs['persistent_workers'] = bool(TRAIN_CONFIG.get('persistent_workers', False))
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG['batch_size'],
                            sampler=val_sampler, **loader_kwargs)

    chunks_in_memory = TRAIN_CONFIG.get('chunks_in_memory', 4)
    train_groups = group_indices_by_chunk(train_indices, chunks_in_memory)

    if is_main:
        logger.info(f"[DATA] Train: {len(train_indices):,} samples across "
                    f"{len(train_groups)} groups ({chunks_in_memory} chunks in memory)")
        logger.info(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    dist.barrier()

    # ── MODEL ─────────────────────────────────────────────────────
    model_cfg = MLPAutoencoderConfig(
        hidden_dims=tuple(TRAIN_CONFIG['hidden_dims']),
        mask_ratio=TRAIN_CONFIG['mask_ratio'],
        dynamic_mask_range=TRAIN_CONFIG.get('dynamic_mask_range'),
        mask_value=TRAIN_CONFIG['mask_value'],
    )
    model = MLPAutoencoder(num_genes=num_genes, cfg=model_cfg).to(device)

    if world_size > 1:
        if TRAIN_CONFIG.get('torch_compile', False):
            model = torch.compile(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    else:
        if TRAIN_CONFIG.get('torch_compile', False):
            model = torch.compile(model)

    model_module = model.module if world_size > 1 else model
    total_params = sum(p.numel() for p in model_module.parameters())
    if is_main:
        logger.info(f"[MODEL] MLPAutoencoder, genes {num_genes}, "
                    f"hidden {list(model_cfg.hidden_dims)}")
        logger.info(f"  ✓ Parameters: {total_params:,}")

    # ── OPTIMIZER & SCHEDULER ─────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'],
                      weight_decay=TRAIN_CONFIG['weight_decay'])
    warmup_epochs = TRAIN_CONFIG.get('warmup_epochs', 0)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, TRAIN_CONFIG['epochs'] - warmup_epochs))
    scaler = torch.amp.GradScaler('cuda')

    # ── RESUME ────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model_module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'warmup_scheduler_state_dict' in checkpoint:
            warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        if 'cosine_scheduler_state_dict' in checkpoint:
            cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        if is_main:
            logger.info(f"  ✓ Resumed from epoch {start_epoch}, "
                        f"best val loss: {best_val_loss:.6f}")

    # ── DYNAMIC MASKING HELPER (same as train.py) ─────────────────
    def _apply_dynamic_mask(x_input, ratio=None):
        if ratio is None:
            lo, hi = TRAIN_CONFIG['dynamic_mask_range']
            ratio = random.uniform(lo, hi)
        return apply_dynamic_mask(
            x_input, ratio,
            masking_strategy='mask_token',
            mask_token=TRAIN_CONFIG['mask_token'],
            mask_token_prob=TRAIN_CONFIG.get('mask_token_prob', 0.8),
            random_token_prob=TRAIN_CONFIG.get('random_token_prob', 0.1),
        )

    def _masked_out_and_targets(out, targets, mask_idx):
        B = out.shape[0]
        biv = torch.arange(B, device=out.device).unsqueeze(1).expand(-1, mask_idx.shape[1])
        return out[biv, mask_idx], targets[biv, mask_idx]

    # ── VALIDATION ────────────────────────────────────────────────
    def _validate():
        model.eval()
        val_loss_t = torch.tensor(0.0, device=device)
        val_r2_sum_t = torch.tensor(0.0, device=device)
        val_batches_t = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for x_input, targets, _mask_idx in val_loader:
                x_input = x_input.to(device)
                targets = targets.to(device)
                if TRAIN_CONFIG.get('dynamic_mask_range') is not None:
                    x_input, mask_idx = _apply_dynamic_mask(
                        x_input, ratio=TRAIN_CONFIG.get('val_mask_ratio', 0.15))
                else:
                    mask_idx = _mask_idx.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x_input, mask_idx=mask_idx)
                    masked_out, masked_targets = _masked_out_and_targets(out, targets, mask_idx)
                    v_loss = F.mse_loss(masked_out, masked_targets)
                    val_loss_t += v_loss.detach()
                    ss_res = ((masked_targets - masked_out) ** 2).sum()
                    ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
                    val_r2_sum_t += (1 - ss_res / ss_tot.clamp(min=1e-8))

                val_batches_t += 1.0

        dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_r2_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_t, op=dist.ReduceOp.SUM)
        epoch_val_loss = (val_loss_t / val_batches_t.clamp(min=1)).item()
        epoch_val_r2 = (val_r2_sum_t / val_batches_t.clamp(min=1)).item()

        model.train()
        return epoch_val_loss, epoch_val_r2

    # ── TRAINING LOOP ─────────────────────────────────────────────
    if is_main:
        logger.info("\n" + "=" * 70)
        logger.info(f"[TRAIN] Starting training from epoch {start_epoch}...")
        logger.info("=" * 70 + "\n")

    n_validations = TRAIN_CONFIG.get('validations_per_epoch', 0) or 0
    total_train_batches = max(1, math.ceil(len(train_indices) / (world_size * TRAIN_CONFIG['batch_size'])))
    if n_validations > 0:
        interval = max(1, total_train_batches // (n_validations + 1))
        val_positions = {(i + 1) * interval for i in range(n_validations)}
    else:
        val_positions = set()
    report_interval = max(1, total_train_batches // 4)

    for epoch in range(start_epoch, TRAIN_CONFIG['epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0
        global_batch = 0
        stop_training = False

        for group_idx, group_indices in enumerate(train_groups, start=1):
            t_load = time.time()
            X_train = load_batch_data(batch_dir, group_indices, verbose=is_main)
            if is_main:
                logger.info(f"[DATA] Group {group_idx}/{len(train_groups)} loaded "
                            f"in {time.time()-t_load:.1f}s ({X_train.shape[0]:,} samples)")

            train_ds = ExpressionMLMDataset(X_train, **dataset_kwargs)
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                               rank=rank, shuffle=True, seed=42)
            train_sampler.set_epoch(epoch)
            train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG['batch_size'],
                                      sampler=train_sampler, **loader_kwargs)

            for batch_idx, (x_input, targets, _mask_idx) in enumerate(train_loader):
                x_input = x_input.to(device)
                targets = targets.to(device)

                if TRAIN_CONFIG.get('dynamic_mask_range') is not None:
                    x_input, mask_idx = _apply_dynamic_mask(x_input)
                else:
                    mask_idx = _mask_idx.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x_input, mask_idx=mask_idx)
                    masked_out, masked_targets = _masked_out_and_targets(out, targets, mask_idx)
                    loss = F.mse_loss(masked_out, masked_targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_clip = TRAIN_CONFIG.get('grad_clip_norm')
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1
                global_batch += 1

                if global_batch in val_positions:
                    avg_train = running_loss / num_batches
                    if is_main:
                        logger.info(f"  Epoch {epoch+1} | Batch {global_batch}/{total_train_batches} "
                                    f"| Loss: {loss.item():.6f} | Avg: {avg_train:.6f}")
                    val_loss, val_r2 = _validate()
                    if is_main:
                        logger.info(f"  [Val] Loss: {val_loss:.6f} | R2: {val_r2:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if is_main:
                            torch.save({
                                'model_state_dict': model_module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch + 1,
                                'val_loss': val_loss,
                                'val_acc': val_r2,
                                'config': TRAIN_CONFIG,
                            }, ckpt_dir / "best_model.pt")
                            logger.info(f"  ✓ New best! Saved best_model.pt")
                    else:
                        if TRAIN_CONFIG['early_stopping']:
                            patience_counter += 1
                            if patience_counter >= TRAIN_CONFIG['patience']:
                                if is_main:
                                    logger.info(f"  ⚠ Early stopping at batch {global_batch}")
                                stop_training = True
                                break
                elif global_batch % report_interval == 0 and is_main:
                    avg_train = running_loss / num_batches
                    pct = global_batch / total_train_batches * 100
                    logger.info(f"  Epoch {epoch+1} | {pct:.0f}% | Batch {global_batch}/"
                                f"{total_train_batches} | Loss: {loss.item():.6f} | Avg: {avg_train:.6f}")

            del X_train, train_ds, train_loader, train_sampler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if stop_training:
                break

        epoch_train_loss = running_loss / max(1, num_batches)
        epoch_val_loss, epoch_val_r2 = _validate()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        epoch_time = time.time() - epoch_start
        if is_main:
            logger.info(f"Epoch {epoch+1} | Train {epoch_train_loss:.6f} | "
                        f"Val {epoch_val_loss:.6f} | R2 {epoch_val_r2:.4f} | {epoch_time:.1f}s")

            last_checkpoint = {
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_r2,
                'config': TRAIN_CONFIG,
            }

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(last_checkpoint, ckpt_dir / "best_model.pt")
                logger.info(f"  ✓ New best! Saved best_model.pt")
            else:
                if TRAIN_CONFIG['early_stopping']:
                    patience_counter += 1
                    if patience_counter >= TRAIN_CONFIG['patience']:
                        if is_main:
                            logger.info(f"  ⚠ Early stopping at epoch {epoch+1}")
                        break

    if is_main:
        torch.save(last_checkpoint, ckpt_dir / "epoch_final.pt")
        with open(ckpt_dir / "config.json", 'w') as f:
            json.dump({**TRAIN_CONFIG, 'best_val_loss': best_val_loss,
                       'run_id': run_id}, f, indent=2)
        pd.DataFrame({'epoch': range(len(train_losses)),
                      'train_loss': train_losses,
                      'val_loss': val_losses}).to_csv(ckpt_dir / "loss_history.csv", index=False)

        if HAS_MATPLOTLIB:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(ckpt_dir / "loss_plot.png")
            plt.close()

        logger.info("=" * 70)
        logger.info(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
        logger.info(f"Checkpoints saved to {ckpt_dir}")
        logger.info("=" * 70)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
