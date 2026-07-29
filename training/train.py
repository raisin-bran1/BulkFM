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
from training.data import get_sample_indices, load_batch_data, ExpressionMLMDataset
from training.utils import _coerce_config_types, build_run_tag
from models.generalized_binformer import GeneralizedBinformer, GeneralizedBinformerConfig

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
    # Initialize DDP early to get rank
    dist.init_process_group(backend="nccl")
    
    # Standard DDP rank extraction
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
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
        logger.info(f"Binformer Training — DDP ({world_size} processes)")
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
            balanced_sampling=CONFIG.get('balanced_sampling', True),
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
        logger.info("[MODEL] Building GeneralizedBinformer...")

    model_cfg = GeneralizedBinformerConfig(
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
    )

    model = GeneralizedBinformer(num_genes=num_genes, cfg=model_cfg).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)

    total_params = sum(p.numel() for p in model.parameters())
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
    
    if is_main:
        logger.info(f"  ✓ AdamW (lr={CONFIG['learning_rate']}, warmup={warmup_epochs})")

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────
    if is_main:
        logger.info("\n" + "=" * 70)
        logger.info("[TRAIN] Starting training...")
        logger.info("=" * 70 + "\n")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    # Load global best val loss
    global_best_path = ckpt_base / 'global_best_val_loss.json'
    global_best_val_loss = float('inf')
    if is_main and global_best_path.exists():
        with open(global_best_path) as f:
            global_best_val_loss = json.load(f).get('val_loss', float('inf'))

    is_binned = CONFIG['expression_embedding'] == 'binned'
    num_bins = CONFIG['num_bins']

    if not is_binned and CONFIG.get('continuous_loss') == 'poisson':
        from models.generalized_binformer import PoissonNLLLogSpace
        cont_loss_fn = PoissonNLLLogSpace()
    else:
        cont_loss_fn = F.mse_loss

    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x_input, targets, mask_idx) in enumerate(train_loader):
            x_input = x_input.to(device)
            mask_idx = mask_idx.to(device)

            out = model(x_input, mask_idx=mask_idx)

            B = x_input.shape[0]
            num_mask = mask_idx.shape[1]
            batch_idx_vec = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask)

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
                    loss = cont_loss_fn(masked_out, masked_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if is_main and (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                avg = running_loss / num_batches
                logger.info(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.6f} | Avg: {avg:.6f}")

        epoch_train_loss = running_loss / max(1, num_batches)

        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        val_top3_sum = 0.0
        val_r2_sum = 0.0
        val_batches = 0

        bin_counts = torch.zeros(num_bins + 2, device=device) if is_binned else None
        true_bin_counts = torch.zeros(num_bins + 2, device=device) if is_binned else None

        with torch.no_grad():
            for x_input, targets, mask_idx in val_loader:
                x_input = x_input.to(device)
                targets = targets.to(device)
                mask_idx = mask_idx.to(device)

                out = model(x_input, mask_idx=mask_idx)

                B = x_input.shape[0]
                num_mask = mask_idx.shape[1]
                batch_idx_vec = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_mask)

                if is_binned:
                    C = out.shape[-1]
                    masked_out = out[batch_idx_vec, mask_idx]
                    masked_targets = targets[batch_idx_vec, mask_idx]

                    v_loss = F.cross_entropy(masked_out.reshape(-1, C), masked_targets.reshape(-1))
                    val_loss += v_loss.item()

                    preds = masked_out.argmax(dim=-1)
                    val_acc_sum += (preds == masked_targets).float().mean().item()

                    _, top3 = masked_out.topk(min(3, C), dim=-1)
                    correct_top3 = top3.eq(masked_targets.unsqueeze(-1).expand_as(top3))
                    val_top3_sum += correct_top3.any(dim=-1).float().mean().item()

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
                    val_loss += v_loss.item()

                    ss_res = ((masked_targets - masked_out) ** 2).sum()
                    ss_tot = ((masked_targets - masked_targets.mean()) ** 2).sum()
                    val_r2_sum += (1 - ss_res / ss_tot.clamp(min=1e-8)).item()

                val_batches += 1

        vl = torch.tensor(val_loss, device=device); dist.all_reduce(vl, op=dist.ReduceOp.SUM)
        vb = torch.tensor(float(val_batches), device=device); dist.all_reduce(vb, op=dist.ReduceOp.SUM)
        epoch_val_loss = (vl / vb.clamp(min=1)).item()

        if is_binned:
            va = torch.tensor(val_acc_sum, device=device); dist.all_reduce(va, op=dist.ReduceOp.SUM)
            vt3 = torch.tensor(val_top3_sum, device=device); dist.all_reduce(vt3, op=dist.ReduceOp.SUM)
            dist.all_reduce(bin_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(true_bin_counts, op=dist.ReduceOp.SUM)
            epoch_val_acc = (va / vb.clamp(min=1)).item()
            epoch_val_top3 = (vt3 / vb.clamp(min=1)).item()
            total_preds = bin_counts.sum()
            bin_dist = (bin_counts / total_preds.clamp(min=1)) * 100
            total_true = true_bin_counts.sum()
            true_dist = (true_bin_counts / total_true.clamp(min=1)) * 100
        else:
            vr2 = torch.tensor(val_r2_sum, device=device); dist.all_reduce(vr2, op=dist.ReduceOp.SUM)
            epoch_val_acc = (vr2 / vb.clamp(min=1)).item()
            epoch_val_top3 = 0.0

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

            if HAS_WANDB:
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                if is_binned:
                    log_dict['val_acc'] = epoch_val_acc
                    log_dict['val_top3'] = epoch_val_top3
                    for b_idx in range(num_bins + 1):
                        log_dict[f'bin_dist/pred_B{b_idx}'] = bin_dist[b_idx].item()
                        log_dict[f'bin_dist/true_B{b_idx}'] = true_dist[b_idx].item()
                else:
                    log_dict['val_r2'] = epoch_val_acc
                wandb.log(log_dict)

            checkpoint_payload = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'config': CONFIG,
            }

            torch.save(checkpoint_payload, ckpt_dir / f"epoch_{epoch:02d}.pt")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(checkpoint_payload, ckpt_dir / "best_model.pt")
                logger.info(f"  ✓ New best! Saved best_model.pt")

                if epoch_val_loss < global_best_val_loss:
                    global_best_val_loss = epoch_val_loss
                    torch.save(checkpoint_payload, ckpt_base / "best_model.pt")
                    with open(global_best_path, 'w') as f:
                        json.dump({'val_loss': global_best_val_loss, 'run_id': run_id, 'timestamp': run_timestamp}, f, indent=2)
                    logger.info(f"  ★ New global best! {epoch_val_loss:.6f}")
            else:
                if CONFIG['early_stopping']:
                    patience_counter += 1
                    if patience_counter >= CONFIG['patience']:
                        if is_main:
                            logger.info(f"  ⚠ Early stopping at epoch {epoch+1}")
                        break

    if is_main:
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
