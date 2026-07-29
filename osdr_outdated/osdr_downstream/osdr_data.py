import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_path not in sys.path:
    sys.path.append(root_path)

class OSDRDataset(Dataset):
    def __init__(self, features, labels):
        """
        features: numpy array [N, G]
        labels: numpy array [N]
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_osdr_loaders(train_feat_path, train_lab_path, val_feat_path, val_lab_path, batch_size=4, num_workers=0, pin_memory=False):
    print(f"[DATA] Loading pre-split OSDR data...")
    
    train_feat = pd.read_parquet(train_feat_path).values
    train_lab = pd.read_parquet(train_lab_path).values.flatten()
    
    val_feat = pd.read_parquet(val_feat_path).values
    val_lab = pd.read_parquet(val_lab_path).values.flatten()
    
    train_ds = OSDRDataset(train_feat, train_lab)
    val_ds = OSDRDataset(val_feat, val_lab)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader
