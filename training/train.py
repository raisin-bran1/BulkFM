# coding=utf-8
# Copyright 2026 The Google Research Authors.

"""
Usage:
  torchrun --nproc_per_node=4 train.py
"""

import os
import sys
import time
import json
import logging
import argparse
import random
from pathlib import Path

# Add project root to sys.path for absolute imports
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

# Import from our new modules
from training.config import CONFIG, USE_SMOKE
from training.data import get_sample_indices, load_batch_data, ExpressionMLMDataset, get_gene_vocabulary
from training.utils import _coerce_config_types, build_run_tag
from models.bulkfm import BulkFM, BulkFMConfig, PoissonNLLLogSpace

# Force unbuffered output for DDP visibility
try:
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
except Exception as e:
    print(f"[WARN] Could not set unbuffered output: {e}", file=sys.stderr)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_logging(ckpt_dir, rank):
    """Set up logging to file and console."""
    log_file = ckpt_dir / f"train_rank{rank}.log"
    
    # Use different loggers per rank to avoid mixing
    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.DEBUG) # <--- Set to capture EVERYTHING
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler (console)
    if rank == 0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="BulkFM Training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args, _ = parser.parse_known_args()

    # Initialize DDP early to get rank
    dist.init_process_group(backend="nccl")
    
    # Standard DDP rank extraction
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Use LOCAL_RANK (0-3 on every node) for device selection
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Use LOCAL_RANK (0-3 on every node) for device selection
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Diagnostics: Ensure we can see the GPUs we expect
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"Rank {rank} (Local {local_rank}) is trying to use GPU {local_rank}, "
            f"but only {num_gpus} GPU(s) are visible to this process. "
            f"Check CUDA_VISIBLE_DEVICES."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0

    # Determine unique run ID
    ckpt_base = Path(CONFIG['checkpoint_dir'])
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Prefix with smoke or train
    run_prefix = "smoke" if USE_SMOKE else "train"
    
    # Use WANDB Run ID if available (standard for sweeps), otherwise fallback to Job ID
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    if wandb_run_id:
        run_id = f"{run_prefix}_{wandb_run_id}"
    else:
        run_id = f"{run_prefix}_{run_timestamp}_{slurm_job_id}"
    
    # We need to broadcast the run_id from main rank if using wandb
    run_id_list = [run_id if is_main else None]
    dist.broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]
    ckpt_dir = ckpt_base / run_id
    
    if is_main:
        ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Set up logger
    logger = setup_logging(ckpt_dir, rank)

    if is_main:
        logger.info("=" * 70)
        logger.info(f"BulkFM Training — DDP ({world_size} processes)")
        logger.info("=" * 70)
        logger.info(f"[SETUP] Run ID: {run_id}")
        logger.info(f"[SETUP] Rank: {rank}, Local Rank: {local_rank}, Device: {device}")

    # WANDB (init early so sweep can override CONFIG)
    # ─────────────────────────────────────────────────────────
    if is_main:
        if HAS_WANDB:
            # If we're in a sweep, wandb.init() handles project/entity automatically
            # via the sweep_id passed to the agent.
            _sweep_id = os.environ.get("WANDB_SWEEP_ID")
            project_name = "binformer-smoke" if USE_SMOKE else "binformer-full"
            
            logger.info(f"[WANDB] Initializing...")
            wandb.init(
                project=None if _sweep_id else project_name, # Auto-detect in sweep
                name=os.environ.get("WANDB_RUN_NAME") or run_id,
                id=run_id,
                resume="allow",
                dir=ckpt_dir,
                config=CONFIG,
            )
            # Pull everything from wandb.config into CONFIG
            # (Allows any sweep parameter to override config.py)
            for key, val in wandb.config.items():
                CONFIG[key] = val

            logger.info(f"  ✓ WANDB Run URL: {wandb.run.get_url()}")
        else:
            logger.warning("[WANDB] wandb module not found. Logging to W&B is disabled.")

    # Broadcast CONFIG from rank 0 so all ranks use the same hyperparams
    config_list = [CONFIG if is_main else None]
    dist.broadcast_object_list(config_list, src=0)
    CONFIG.update(config_list[0])

    # 1. Handle auto-calculated parameters for ALL ranks
    if CONFIG.get('random_token_prob') == 'auto':
        mask_p = float(CONFIG.get('mask_token_prob', 0.8))
        CONFIG['random_token_prob'] = (1.0 - mask_p) / 2.0
        if is_main:
            logger.info(f"[CONFIG] Auto-calculated random_token_prob: {CONFIG['random_token_prob']:.4f}")
    
    # 2. Coerce types for ALL ranks
    _coerce_config_types(CONFIG)

    # 3. Update W&B UI on main rank only
    if is_main and HAS_WANDB and wandb.run:
        wandb.config.update(CONFIG, allow_val_change=True)
        
    # LOAD DATA
    # ─────────────────────────────────────────────────────────
    data_dir = Path(CONFIG['data_dir'])
    batch_dir = data_dir / "batch_files"
    if not batch_dir.exists():
        batch_dir = data_dir

    if is_main:
        logger.info("[DATA] Building sample indices from chunks...")

    t0 = time.time()
    train_indices = None
    val_indices = None
    if is_main:
        train_indices, val_indices = get_sample_indices(
            batch_dir,
            train_chunks=CONFIG.get('train_chunks'),
            val_chunks=CONFIG.get('val_chunks'),
            train_subset=CONFIG.get('train_subset'),
            val_subset=CONFIG.get('val_subset'),
            seed=CONFIG['seed'],
            verbose=True,
        )

    train_indices_list = [train_indices if is_main else None]
    val_indices_list = [val_indices if is_main else None]
    dist.broadcast_object_list(train_indices_list, src=0)
    dist.broadcast_object_list(val_indices_list, src=0)
    train_indices = train_indices_list[0]
    val_indices = val_indices_list[0]
    
    if is_main:
        logger.info(f"  ✓ Index time: {time.time()-t0:.1f}s")
        logger.info("[DATA] Loading data into memory...")
    
    X_train = load_batch_data(batch_dir, train_indices, verbose=is_main)
    X_val = load_batch_data(batch_dir, val_indices, verbose=is_main)

    num_genes = X_train.shape[1]
    
    dataset_kwargs = {
        'mask_ratio': CONFIG['mask_ratio'],
        'dynamic_mask_range': CONFIG.get('dynamic_mask_range'),
        'mask_token': CONFIG['mask_token'],
        'mask_token_prob': CONFIG.get('mask_token_prob', 0.8),
        'random_token_prob': CONFIG.get('random_token_prob', 0.1),
        'num_bins': CONFIG['num_bins'],
        'expression_embedding': CONFIG['expression_embedding'],
        'masking_strategy': CONFIG['masking_strategy'],
    }

    train_ds = ExpressionMLMDataset(X_train, **dataset_kwargs)
    val_ds = ExpressionMLMDataset(X_val, **dataset_kwargs)

    if is_main:
        logger.info(f"[CHECK] num_genes={num_genes}")

    # ─────────────────────────────────────────────────────────
    # DATASETS & DATALOADERS
    # ─────────────────────────────────────────────────────────
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=rank, shuffle=True, seed=42)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                      rank=rank, shuffle=False, seed=42)

    num_workers = int(CONFIG.get('num_workers', 0))
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = int(CONFIG.get('prefetch_factor', 2))
        loader_kwargs['persistent_workers'] = bool(CONFIG.get('persistent_workers', False))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            sampler=val_sampler, **loader_kwargs)

    if is_main:
        logger.info(f"[DATA] Train: {len(train_ds):,} samples, {len(train_loader)} batches")
        logger.info(f"[DATA] Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # Synchronize after data loading
    dist.barrier()

    # ─────────────────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────────────────
    if is_main:
        logger.info("[MODEL] Building BulkFM...")

    model_cfg = BulkFMConfig(
        hidden_dim=CONFIG['hidden_dim'],
        ffn_dim=CONFIG['ffn_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        feature_type=CONFIG['feature_type'],
        compute_type=CONFIG['compute_type'],
        expression_embedding=CONFIG['expression_embedding'],
        num_bins=CONFIG['num_bins'],
        continuous_loss=CONFIG.get('continuous_loss', 'mse'),
        mask_ratio=CONFIG['mask_ratio'],
        dynamic_mask_range=CONFIG.get('dynamic_mask_range'),
        mask_token_id=CONFIG['mask_token'],
        masking_strategy=CONFIG['masking_strategy'],
        simple_projection=CONFIG.get('expression_projection', 'nonlinear') == 'linear',
    )

    model = BulkFM(num_genes=num_genes, cfg=model_cfg).to(device)

    if world_size > 1:
        if CONFIG.get('torch_compile', False):
            if is_main:
                logger.info("[COMPILE] Applying torch.compile...")
            model = torch.compile(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    else:
        if CONFIG.get('torch_compile', False):
            if is_main:
                logger.info("[COMPILE] Applying torch.compile...")
            model = torch.compile(model)

    model_module = model.module if world_size > 1 else model

    total_params = sum(p.numel() for p in model_module.parameters())
    if is_main:
        logger.info(f"  ✓ Parameters: {total_params:,}")

    # ─────────────────────────────────────────────────────────
    # OPTIMIZER & SCHEDULER
    # ─────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                      weight_decay=CONFIG['weight_decay'])
    
    # Cosine annealing with linear warmup
    warmup_epochs = CONFIG.get('warmup_epochs', 0)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, CONFIG['epochs'] - warmup_epochs))
    
    scaler = torch.amp.GradScaler('cuda')

    if is_main:
        logger.info(f"  ✓ AdamW (lr={CONFIG['learning_rate']}, warmup={warmup_epochs})")

    # ─────────────────────────────────────────────────────────
    # RESUME
    # ─────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    if args.resume:
        if is_main:
            logger.info(f"[RESUME] Loading checkpoint from {args.resume}")

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
            logger.info(f"  ✓ Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")



    is_binned = CONFIG['expression_embedding'] == 'binned'
    num_bins = CONFIG['num_bins']

    if not is_binned and CONFIG.get('continuous_loss') == 'poisson':
        cont_loss_fn = PoissonNLLLogSpace()
    else:
        cont_loss_fn = F.mse_loss

    # ─────────────────────────────────────────────────────────
    # VALIDATION HELPER
    # ─────────────────────────────────────────────────────────
    def _validate():
        model.eval()
        val_loss_t = torch.tensor(0.0, device=device)
        val_acc_sum_t = torch.tensor(0.0, device=device)
        val_top3_sum_t = torch.tensor(0.0, device=device)
        val_r2_sum_t = torch.tensor(0.0, device=device)
        val_batches_t = torch.tensor(0.0, device=device)
        bin_counts = torch.zeros(num_bins + 2, device=device) if is_binned else None
        true_bin_counts = torch.zeros(num_bins + 2, device=device) if is_binned else None

        with torch.no_grad():
            for x_input, targets, mask_idx in val_loader:
                x_input = x_input.to(device)
                targets = targets.to(device)
                mask_idx = mask_idx.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x_input, mask_idx=mask_idx)
                    B = x_input.shape[0]
                    num_mask = mask_idx.shape[1]
                    if not hasattr(_validate, '_biv') or _validate._biv.shape[0] != B or _validate._biv.shape[1] != num_mask:
                        _validate._biv = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask)
                    batch_idx_vec = _validate._biv

                    if is_binned:
                        C = out.shape[-1]
                        masked_out = out[batch_idx_vec, mask_idx]
                        masked_targets = targets[batch_idx_vec, mask_idx]
                        v_loss = F.cross_entropy(masked_out.reshape(-1, C), masked_targets.reshape(-1))
                        val_loss_t += v_loss.detach()
                        preds = masked_out.argmax(dim=-1)
                        val_acc_sum_t += (preds == masked_targets).float().mean()
                        _, top3 = masked_out.topk(min(3, C), dim=-1)
                        correct_top3 = top3.eq(masked_targets.unsqueeze(-1).expand_as(top3))
                        val_top3_sum_t += correct_top3.any(dim=-1).float().mean()
                        ones = torch.ones_like(preds, dtype=torch.float32)
                        bin_counts.scatter_add_(0, preds.reshape(-1), ones.reshape(-1))
                        true_ones = torch.ones_like(masked_targets, dtype=torch.float32)
                        true_bin_counts.scatter_add_(0, masked_targets.reshape(-1), true_ones.reshape(-1))
                    else:
                        if out.dim() == 3:
                            masked_out = out[batch_idx_vec, mask_idx].squeeze(-1)
                        else:
                            masked_out = out[batch_idx_vec, mask_idx]
                        masked_targets = targets[batch_idx_vec, mask_idx]
                        if CONFIG.get('continuous_loss') == 'poisson':
                            v_loss = cont_loss_fn(masked_out, masked_targets.log1p()).mean()
                        else:
                            v_loss = F.mse_loss(masked_out, masked_targets)
                        val_loss_t += v_loss.detach()
                        ss_res = ((masked_targets - masked_out) ** 2).sum()
                        ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
                        val_r2_sum_t += (1 - ss_res / ss_tot.clamp(min=1e-8))

                val_batches_t += 1.0

        dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_t, op=dist.ReduceOp.SUM)
        epoch_val_loss = (val_loss_t / val_batches_t.clamp(min=1)).item()

        if is_binned:
            dist.all_reduce(val_acc_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_top3_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(true_bin_counts, op=dist.ReduceOp.SUM)
            epoch_val_acc = (val_acc_sum_t / val_batches_t.clamp(min=1)).item()
            epoch_val_top3 = (val_top3_sum_t / val_batches_t.clamp(min=1)).item()
            total_preds = bin_counts.sum()
            bin_dist = (bin_counts / total_preds.clamp(min=1)) * 100
            total_true = true_bin_counts.sum()
            true_dist = (true_bin_counts / total_true.clamp(min=1)) * 100
        else:
            dist.all_reduce(val_r2_sum_t, op=dist.ReduceOp.SUM)
            epoch_val_acc = (val_r2_sum_t / val_batches_t.clamp(min=1)).item()
            epoch_val_top3 = 0.0
            bin_dist = true_dist = None

        model.train()
        return epoch_val_loss, epoch_val_acc, epoch_val_top3, bin_dist, true_dist

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    if is_main:
        logger.info("\n" + "=" * 70)
        logger.info(f"[TRAIN] Starting training from epoch {start_epoch}...")
        logger.info("=" * 70 + "\n")

    n_validations = CONFIG.get('validations_per_epoch', 0) or 0
    if n_validations > 0:
        interval = max(1, len(train_loader) // (n_validations + 1))
        val_positions = { (i + 1) * interval for i in range(n_validations) }
    else:
        val_positions = set()
    report_interval = max(1, len(train_loader) // 4)

    for epoch in range(start_epoch, CONFIG['epochs']):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_input, targets, _mask_idx) in enumerate(train_loader):
            x_input = x_input.to(device)

            if CONFIG.get('dynamic_mask_range') is not None:
                B, G = x_input.shape
                lo, hi = CONFIG['dynamic_mask_range']
                ratio = random.uniform(lo, hi)
                num_mask = max(1, int(G * ratio))
                idxs = torch.randperm(G, device=device).unsqueeze(0).expand(B, -1)
                mask_idx = idxs[:, :num_mask]
                if CONFIG['masking_strategy'] == 'mask_token':
                    x_input = x_input.clone()
                    mask_token_v = CONFIG['mask_token']
                    mask_p = CONFIG.get('mask_token_prob', 0.8)
                    rand_p = CONFIG.get('random_token_prob', 0.1)
                    probs = torch.rand(B, num_mask, device=device)
                    is_mask = probs < mask_p
                    is_rand = (probs >= mask_p) & (probs < mask_p + rand_p)
                    mask_pos = mask_idx[is_mask]
                    x_input.view(-1)[mask_pos[:, 0] * G + mask_pos[:, 1]] = mask_token_v
                    rand_pos = mask_idx[is_rand]
                    nonzero = x_input[x_input > 0]
                    if len(nonzero) > 0:
                        rand_vals = nonzero[torch.randint(len(nonzero), (rand_pos.shape[0],), device=device)]
                    else:
                        rand_vals = torch.empty(rand_pos.shape[0], device=device).uniform_(0, 10)
                    x_input.view(-1)[rand_pos[:, 0] * G + rand_pos[:, 1]] = rand_vals
            else:
                mask_idx = _mask_idx.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x_input, mask_idx=mask_idx)

                B = x_input.shape[0]
                num_mask = mask_idx.shape[1]
                if not hasattr(_validate, '_biv') or _validate._biv.shape[0] != B or _validate._biv.shape[1] != num_mask:
                    _validate._biv = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask)
                batch_idx_vec = _validate._biv

                if is_binned:
                    targets = targets.to(device)
                    C = out.shape[-1]
                    masked_out = out[batch_idx_vec, mask_idx]
                    masked_targets = targets[batch_idx_vec, mask_idx]
                    loss = F.cross_entropy(masked_out.reshape(-1, C), masked_targets.reshape(-1))
                else:
                    targets = targets.to(device)
                    if out.dim() == 3:
                        masked_out = out[batch_idx_vec, mask_idx].squeeze(-1)
                    else:
                        masked_out = out[batch_idx_vec, mask_idx]
                    masked_targets = targets[batch_idx_vec, mask_idx]
                    if CONFIG.get('continuous_loss') == 'poisson':
                        loss = cont_loss_fn(masked_out, masked_targets.log1p()).mean()
                    else:
                        loss = F.mse_loss(masked_out, masked_targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = CONFIG.get('grad_clip_norm')
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) in val_positions:
                avg_train = running_loss / num_batches
                if is_main:
                    logger.info(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f} | Avg: {avg_train:.6f}")

                val_loss, val_acc, val_top3, bin_dist, true_dist = _validate()

                if is_main:
                    if is_binned:
                        logger.info(f"  [Val] Loss: {val_loss:.6f} | Acc: {val_acc:.4f} | Top3: {val_top3:.4f}")
                    else:
                        logger.info(f"  [Val] Loss: {val_loss:.6f} | R2: {val_acc:.4f}")

                    if HAS_WANDB:
                        log_dict = {
                            'batch': batch_idx + 1,
                            'train_loss': avg_train,
                            'val_loss': val_loss,
                            'lr': optimizer.param_groups[0]['lr'],
                        }
                        if is_binned:
                            log_dict['val_acc'] = val_acc
                            log_dict['val_top3'] = val_top3
                        else:
                            log_dict['val_r2'] = val_acc
                        wandb.log(log_dict)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if is_main:
                        checkpoint_payload = {
                            'model_state_dict': model_module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                            'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                            'epoch': epoch + 1,
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'config': CONFIG,
                        }
                        torch.save(checkpoint_payload, ckpt_dir / "best_model.pt")
                        if is_main:
                            logger.info(f"  ✓ New best! Saved best_model.pt")
                else:
                    if CONFIG['early_stopping']:
                        patience_counter += 1
                        if patience_counter >= CONFIG['patience']:
                            if is_main:
                                logger.info(f"  ⚠ Early stopping at batch {batch_idx+1}")
                            break
            elif (batch_idx + 1) % report_interval == 0 and is_main:
                avg_train = running_loss / num_batches
                pct = (batch_idx + 1) / len(train_loader) * 100
                logger.info(f"  Epoch {epoch+1} | {pct:.0f}% | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f} | Avg: {avg_train:.6f}")

        epoch_train_loss = running_loss / max(1, num_batches)
        epoch_val_loss, epoch_val_acc, epoch_val_top3, bin_dist, true_dist = _validate()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        epoch_time = time.time() - epoch_start

        if is_main:
            if is_binned:
                logger.info(f"Epoch {epoch+1} | Train {epoch_train_loss:.6f} | Val {epoch_val_loss:.6f} | Acc {epoch_val_acc:.4f} | Top3 {epoch_val_top3:.4f} | {epoch_time:.1f}s")
                pred_str = f"Pred: B0:{bin_dist[0]:.1f}% B1:{bin_dist[1]:.1f}% B25:{bin_dist[25]:.1f}% B50:{bin_dist[50]:.1f}%"
                true_str = f"True: B0:{true_dist[0]:.1f}% B1:{true_dist[1]:.1f}% B25:{true_dist[25]:.1f}% B50:{true_dist[50]:.1f}%"
                logger.info(f"  {pred_str}")
                logger.info(f"  {true_str}")
            else:
                logger.info(f"Epoch {epoch+1} | Train {epoch_train_loss:.6f} | Val {epoch_val_loss:.6f} | R2 {epoch_val_acc:.4f} | {epoch_time:.1f}s")

            checkpoint_payload = {
                'model_state_dict': model_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'config': CONFIG,
            }
            last_checkpoint = checkpoint_payload

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(checkpoint_payload, ckpt_dir / "best_model.pt")
                logger.info(f"  ✓ New best! Saved best_model.pt")
            else:
                if CONFIG['early_stopping']:
                    patience_counter += 1
                    if patience_counter >= CONFIG['patience']:
                        if is_main:
                            logger.info(f"  ⚠ Early stopping at epoch {epoch+1}")
                        break

    if is_main:
        vocab = get_gene_vocabulary(CONFIG['data_dir'])
        vocab_path = ckpt_base / "gene_vocabulary.csv"
        pd.DataFrame({"genes": vocab}).to_csv(vocab_path, index=False)
        logger.info(f"  ✓ Gene vocabulary saved ({len(vocab)} genes) to {vocab_path}")

        torch.save(last_checkpoint, ckpt_dir / "epoch_final.pt")

        with open(ckpt_dir / "config.json", 'w') as f:
            json.dump({**CONFIG, 'best_val_loss': best_val_loss, 'run_id': run_id}, f, indent=2)

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

    if is_main and HAS_WANDB:
        wandb.finish()

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
