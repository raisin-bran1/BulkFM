import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import Counter

# default embeddings are from binformer, can change via --embeddings
# organs: python classify_embeddings/classifier.py --column organ --epochs 50
# spaceflight: python classify_embeddings/classifier.py --column spaceflight --balance --epochs 20

def parse_args():
    parser = argparse.ArgumentParser(description="Classify OSDR Embeddings")
    parser.add_argument("--embeddings", type=str, default="osdr/osdr_embeddings.pt", help="Path to embeddings .pt file")
    parser.add_argument("--labels", type=str, default="osdr/organ_spaceflight_batch_labels.parquet", help="Path to labels parquet")
    parser.add_argument("--column", type=str, default="spaceflight", help="Column name to classify")
    parser.add_argument("--balance", action="store_true", help="Whether to perfectly balance classes by downsampling")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_group_split", action="store_true", help="Disable group-based splitting (allow batch leakage)")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "logistic"], help="Type of model to use: 'mlp' (PyTorch+Adam) or 'logistic' (Sklearn+LBFGS)")
    parser.add_argument("--filter_organ", type=str, default=None, help="Only include samples from this organ (e.g., 'Liver')")
    return parser.parse_args()

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        # Using a simple linear layer as requested/original
        self.network = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.network(x)

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"[INFO] Loading embeddings from {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location='cpu', weights_only=True).float()
    
    print(f"[INFO] Loading labels from {args.labels}")
    labels_df = pd.read_parquet(args.labels)
    
    # Handle mismatches
    if len(embeddings) != len(labels_df):
        print(f"[WARNING] Dimension mismatch: Embeddings ({len(embeddings)}) vs Labels ({len(labels_df)})")
        min_len = min(len(embeddings), len(labels_df))
        embeddings = embeddings[:min_len]
        labels_df = labels_df.iloc[:min_len]

    # Filter by organ if requested
    if args.filter_organ:
        if 'organ' not in labels_df.columns:
            print(f"[ERROR] 'organ' column not found. Available: {labels_df.columns.tolist()}")
            return
        
        mask = labels_df['organ'] == args.filter_organ
        if not mask.any():
            print(f"[ERROR] No samples found for organ: '{args.filter_organ}'")
            print(f"Available organs: {labels_df['organ'].unique().tolist()}")
            return
        
        embeddings = embeddings[mask.values]
        labels_df = labels_df[mask].reset_index(drop=True)
        print(f"[INFO] Filtered to organ: '{args.filter_organ}' ({len(labels_df)} samples remaining)")
    
    if args.column not in labels_df.columns:
        # Fallback for organ_labels which might have different naming
        print(f"[WARN] Column '{args.column}' not found. Available: {labels_df.columns.tolist()}")
        if len(labels_df.columns) >= 2:
            args.column = labels_df.columns[1]
            print(f"[INFO] Using column: '{args.column}'")
    
    y_raw = labels_df[args.column].values
    groups = labels_df['osd_batch'].values
        
    # Encode labels if they are strings/categorical
    le = LabelEncoder()
    y_all = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"[INFO] Classification task: {args.column} ({num_classes} classes)")
    print(f"[INFO] Class mapping: {dict(zip(range(num_classes), le.classes_))}")

    indices = np.arange(len(y_all))
    
    if args.balance:
        print(f"[INFO] Balancing classes by downsampling to minimum...")
        counts = Counter(y_all)
        min_samples = min(counts.values())
        print(f"[INFO] Samples per class: {min_samples}")
        
        balanced_indices = []
        rng = np.random.default_rng(args.seed)
        for cls in range(num_classes):
            cls_indices = indices[y_all == cls]
            chosen = rng.choice(cls_indices, min_samples, replace=False)
            balanced_indices.extend(chosen)
        
        indices = np.array(balanced_indices)
        y_balanced = y_all[indices]
    else:
        y_balanced = y_all

    # Split by group (osd_batch) to prevent leakage
    if not args.no_group_split:
        print(f"[INFO] Splitting by 'osd_batch' groups to prevent leakage...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx_inner, test_idx_inner = next(gss.split(indices, y_all[indices], groups=groups[indices]))
        train_idx = indices[train_idx_inner]
        test_idx = indices[test_idx_inner]
    else:
        print(f"[INFO] Using standard stratified split (batch leakage ALLOWED)...")
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=y_all[indices])
    
    print(f"[INFO] Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    if not args.no_group_split:
        print(f"[INFO] Train groups: {len(np.unique(groups[train_idx]))}, Test groups: {len(np.unique(groups[test_idx]))}")
    
    X_train = embeddings[train_idx].numpy()
    y_train = y_all[train_idx]
    X_test = embeddings[test_idx].numpy()
    y_test = y_all[test_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if args.model_type == "logistic":
        print(f"[INFO] Training Scikit-learn Logistic Regression (LBFGS + L2 Regularization)...")
        # solver='lbfgs' is the default and handles multiclass well
        model = LogisticRegression(max_iter=1000, random_state=args.seed)
        model.fit(X_train, y_train)
        
        all_preds = model.predict(X_test)
        all_true = y_test
    else:
        # Existing PyTorch MLP Path
        train_ds = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
        test_ds = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        
        model = SimpleMLP(input_dim=embeddings.shape[1], num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Training
        train_losses = []
        test_losses = []
        
        for epoch in range(args.epochs):
            model.train()
            running_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            
            model.eval()
            running_test_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    running_test_loss += loss.item()
                    
            train_losses.append(running_train_loss / len(train_loader))
            test_losses.append(running_test_loss / len(test_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy().tolist())
                all_true.extend(batch_y.numpy().tolist())
            
    print("\n--- Results ---")
    print(f"Accuracy: {accuracy_score(all_true, all_preds) * 100:.2f}%")
    print("\n[CLASSIFICATION REPORT]")
    
    # Get labels present in the test set to avoid mismatch
    present_classes = np.unique(all_true)
    target_names = [str(le.classes_[i]) for i in present_classes]
    
    print(classification_report(all_true, all_preds, labels=present_classes, target_names=target_names))
    
    # Convert keys/values to standard Python ints for cleaner printing
    dist = dict(Counter(all_preds))
    clean_dist = {int(k): int(v) for k, v in dist.items()}
    print(f"Prediction Distribution: {clean_dist}")

if __name__ == "__main__":
    main()
