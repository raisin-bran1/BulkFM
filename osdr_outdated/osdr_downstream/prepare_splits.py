import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
import argparse
import json

# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_path not in sys.path:
    sys.path.append(root_path)

def prepare_splits(features_path, labels_path, vocab_path, column='spaceflight', output_dir="data/osdr", test_size=0.2, seed=42, mask_token=-10, balance=True):
    print(f"[INFO] Loading data...")
    vocab = pd.read_csv(vocab_path)
    gene_list = vocab['genes'].tolist()
    
    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)
    
    # Drop duplicates to ensure 1-to-1 mapping
    features_df = features_df.drop_duplicates(subset='sample_id')
    
    # Identify sample name column (usually first column or contains 'id')
    id_col = None
    for col in labels_df.columns:
        if 'id' in col.lower() or 'sample' in col.lower():
            id_col = col
            break
    if not id_col: id_col = labels_df.columns[0]
    
    labels_df = labels_df.drop_duplicates(subset=id_col)
    
    # Merge
    merged_df = features_df.merge(labels_df, left_on='sample_id', right_on=id_col)
    
    if column not in merged_df.columns:
        raise ValueError(f"Column '{column}' not found in labels. Available: {labels_df.columns.tolist()}")

    # Encode labels
    le = LabelEncoder()
    merged_df['label'] = le.fit_transform(merged_df[column])
    num_classes = len(le.classes_)
    
    label_map = dict(zip(range(num_classes), le.classes_.tolist()))
    print(f"[INFO] Task: {column} ({num_classes} classes)")
    
    # BALANCING LOGIC
    if balance:
        print(f"[INFO] Creating balanced splits...")
        counts = merged_df['label'].value_counts()
        n_samples_per_class = counts.min()
        
        balanced_dfs = []
        for cls in range(num_classes):
            cls_df = merged_df[merged_df['label'] == cls]
            balanced_dfs.append(cls_df.sample(n=n_samples_per_class, random_state=seed))
        
        merged_df = pd.concat(balanced_dfs).sample(frac=1, random_state=seed)
        print(f"[INFO] Balanced to {n_samples_per_class} samples per class.")

    # Split
    train_df, val_df = train_test_split(merged_df, test_size=test_size, random_state=seed, stratify=merged_df['label'])
    
    print(f"[INFO] Train set: {len(train_df)} samples")
    print(f"[INFO] Val set:   {len(val_df)} samples")
    
    # Align and fill missing with mask_token
    def align_genes(df):
        feature_cols = []
        for gene in gene_list:
            if gene in df.columns:
                feature_cols.append(df[gene])
            else:
                feature_cols.append(pd.Series([mask_token] * len(df), name=gene, index=df.index))
        
        final_feat = pd.concat(feature_cols, axis=1)
        final_feat.columns = gene_list
        return final_feat

    print(f"[INFO] Aligning genes for splits...")
    train_feat = align_genes(train_df)
    val_feat = align_genes(val_df)
    
    train_lab = train_df[['label']]
    val_lab = val_df[['label']]
    
    # Save to parquet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    prefix = f"{column}_" if column != 'spaceflight' else "osdr_"
    
    train_feat.to_parquet(os.path.join(output_dir, f"{prefix}train_features.parquet"))
    val_feat.to_parquet(os.path.join(output_dir, f"{prefix}val_features.parquet"))
    train_lab.to_parquet(os.path.join(output_dir, f"{prefix}train_labels.parquet"))
    val_lab.to_parquet(os.path.join(output_dir, f"{prefix}val_labels.parquet"))
    
    # Save label mapping
    with open(os.path.join(output_dir, f"{prefix}label_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    
    print(f"[SUCCESS] Splits and label map created in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/osdr/osdr_processed_spaceflight.parquet")
    parser.add_argument("--labels", type=str, default="data/osdr/spaceflight_labels_clean.parquet")
    parser.add_argument("--vocab", type=str, default="models/gene_vocabulary.csv")
    parser.add_argument("--column", type=str, default="spaceflight")
    parser.add_argument("--output_dir", type=str, default="data/osdr")
    parser.add_argument("--no_balance", action="store_false", dest="balance")
    args = parser.parse_args()
    
    prepare_splits(
        features_path=args.features,
        labels_path=args.labels,
        vocab_path=args.vocab,
        column=args.column,
        output_dir=args.output_dir,
        balance=args.balance
    )
