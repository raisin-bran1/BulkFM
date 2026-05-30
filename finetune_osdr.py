import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from binformer import Binformer
from lora import apply_lora, get_lora_params
from osdr_data import get_osdr_loaders
import json
from tqdm import tqdm
from collections import Counter
from finetune_config import FINETUNE_CONFIG
from sklearn.metrics import classification_report, confusion_matrix

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use near-zero initialization to ensure uniform attention at the start of training.
        # This effectively anchors the model to a "Mean Pooling" state.
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.001)
        self.key_map = nn.Linear(dim, dim)
        
        # Initialize key_map weights to be very small
        nn.init.normal_(self.key_map.weight, std=0.001)
        if self.key_map.bias is not None:
            nn.init.zeros_(self.key_map.bias)
            
        self.scale = dim ** -0.5

    def forward(self, x):
        # x: [B, G, D]
        # query: [1, 1, D] -> broadcast to [B, 1, D]
        q = self.query.expand(x.size(0), -1, -1)
        k = self.key_map(x)   # [B, G, D]
        
        # Identity Values: We use the original embeddings as values.
        # This preserves the foundation model's features and simplifies the task to "weighting" only.
        v = x

        # Attention calculation: [B, 1, G]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Output: [B, 1, D] -> [B, D]
        pooled = torch.matmul(attn, v).squeeze(1)
        return pooled

class BinformerClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2, pooling_type='attention'):
        super().__init__()
        self.base_model = base_model
        d = base_model._hidden_dim
        self.pooling_type = pooling_type
        if pooling_type == 'attention':
            self.pooler = AttentionPooling(d)
        self.head = nn.Linear(d, num_classes)
        
    def forward(self, x):
        h = self.base_model(x, output_hidden=True)
        if self.pooling_type == 'attention':
            pooled = self.pooler(h)
        else:
            pooled = h.mean(dim=1)
        logits = self.head(pooled)
        return logits

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, accum_iter=8, smoke_test=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(tqdm(loader, desc="Training", leave=False)):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y) / accum_iter
        loss.backward()
        if (i + 1) % accum_iter == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
        total_loss += loss.item() * accum_iter * x.size(0)
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        if smoke_test: break
    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, criterion, device, smoke_test=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_true = [], []
    for x, y in tqdm(loader, desc="Validating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        _, predicted = logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_true.extend(y.cpu().numpy().tolist())
        if smoke_test: break
    return total_loss / total, correct / total, all_true, all_preds

def main():
    cfg = FINETUNE_CONFIG
    torch.manual_seed(cfg['seed'])
    
    # Setup Run Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(cfg['paths']['results_dir'], run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config used
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
        
    # Device selection
    if cfg['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['device'])
    
    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Using device: {device}")
    
    # Load architecture config
    with open(cfg['paths']['base_config'], "r") as f:
        arch_config = json.load(f)
    
    vocab = pd.read_csv(cfg['paths']['vocab'])
    num_genes = len(vocab)
    
    base_model = Binformer(
        num_genes=num_genes,
        hidden_dim=arch_config.get('hidden_dim', 256),
        n_heads=arch_config.get('num_heads', 8),
        n_layers=arch_config.get('num_layers', 4),
        ffn_dim=arch_config.get('ffn_dim', 1024),
        num_bins=arch_config.get('num_bins', 50),
        mask_token_id=arch_config.get('mask_token', -10),
        feature_type=arch_config.get('feature_type', 'sqr'),
        compute_type=arch_config.get('compute_type', 'iter')
    )
    
    checkpoint = torch.load(cfg['paths']['base_model'], map_location=device, weights_only=True)
    base_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    
    if cfg['use_lora']:
        print(f"[INFO] Applying LoRA (rank={cfg['lora_rank']})...")
        base_model = apply_lora(base_model, rank=cfg['lora_rank'], alpha=cfg['lora_alpha'])
    
    model = BinformerClassifier(base_model, num_classes=cfg['num_classes'], pooling_type=cfg['pooling_type']).to(device)
    
    train_loader, val_loader = get_osdr_loaders(
        cfg['paths']['train_features'], cfg['paths']['train_labels'],
        cfg['paths']['val_features'], cfg['paths']['val_labels'],
        batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], pin_memory=cfg['pin_memory']
    )
    
    criterion = nn.CrossEntropyLoss()
    accum_iter = cfg['accum_iter']
    history = []
    best_val_acc = 0

    # STAGE 1: Head Only (Simplified Scheduler)
    print(f"\n[STAGE 1] Training pooling/head (Base Frozen)...")
    for n, p in model.named_parameters():
        p.requires_grad = ('base_model' not in n)
            
    optimizer_s1 = optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                                lr=cfg['stage1']['lr'], weight_decay=cfg['stage1']['weight_decay'])
    
    # Simple linear warmup then constant
    w_e1 = cfg['stage1']['warmup_epochs']
    sched_s1 = optim.lr_scheduler.LambdaLR(optimizer_s1, lr_lambda=lambda e: min(1.0, (e + 1) / max(1, w_e1)))
    
    for epoch in range(cfg['stage1']['epochs']):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer_s1, criterion, device, accum_iter=accum_iter, smoke_test=cfg['smoke_test'])
        sched_s1.step()
        v_loss, v_acc, y_true, y_pred = validate(model, val_loader, criterion, device, smoke_test=cfg['smoke_test'])
        print(f"Epoch {epoch+1}/{cfg['stage1']['epochs']} (S1): Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f}")
        print(f"  Val Pred Dist: {dict(Counter(y_pred))}")
        history.append({'stage': 1, 'epoch': epoch+1, 'train_loss': t_loss, 'train_acc': t_acc, 'val_loss': v_loss, 'val_acc': v_acc})

    # STAGE 2: LoRA + Head
    print(f"\n[STAGE 2] Training LoRA + head...")
    if cfg['use_lora']:
        lora_params = get_lora_params(model.base_model)
        for p in lora_params: p.requires_grad = True
        optimizer_s2 = optim.AdamW([
            {'params': lora_params, 'lr': cfg['stage2']['lora_lr']},
            {'params': [p for n, p in model.named_parameters() if 'base_model' not in n], 'lr': cfg['stage2']['head_lr']}
        ], weight_decay=cfg['stage2']['weight_decay'])
    else:
        optimizer_s2 = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['stage2']['no_lora_lr'], weight_decay=cfg['stage2']['weight_decay'])
    
    w_e2, e2 = cfg['stage2']['warmup_epochs'], cfg['stage2']['epochs']
    s2_warmup = optim.lr_scheduler.LambdaLR(optimizer_s2, lr_lambda=lambda e: (e + 1) / max(1, w_e2))
    s2_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=max(1, e2 - w_e2))
    
    for epoch in range(e2):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer_s2, criterion, device, accum_iter=accum_iter, smoke_test=cfg['smoke_test'])
        if epoch < w_e2: s2_warmup.step()
        else: s2_cosine.step()
        v_loss, v_acc, y_true, y_pred = validate(model, val_loader, criterion, device, smoke_test=cfg['smoke_test'])
        print(f"Epoch {epoch+1}/{e2} (S2): Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f}")
        print(f"  Val Pred Dist: {dict(Counter(y_pred))}")
        history.append({'stage': 2, 'epoch': epoch+1, 'train_loss': t_loss, 'train_acc': t_acc, 'val_loss': v_loss, 'val_acc': v_acc})
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            print(f"  --> Saved new best model (Acc: {v_acc:.4f})")

    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    
    # Final Report
    print(f"\n{'='*40}\nFINETUNING COMPLETE\n{'='*40}\nBest Val Acc: {best_val_acc:.4f}\nResults: {run_dir}")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device, weights_only=True))
    _, _, y_true, y_pred = validate(model, val_loader, criterion, device)
    
    report = classification_report(y_true, y_pred, target_names=['Ground', 'Spaceflight'])
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n[FINAL REPORT]\n", report)
    print("\n[CONFUSION MATRIX]\n", cm)
    
    with open(os.path.join(run_dir, "final_report.txt"), "w") as f:
        f.write(f"Run ID: {run_id}\nBest Val Acc: {best_val_acc:.4f}\n\n")
        f.write("[CLASSIFICATION REPORT]\n")
        f.write(report)
        f.write("\n\n[CONFUSION MATRIX]\n")
        f.write(str(cm))

if __name__ == "__main__":
    main()
