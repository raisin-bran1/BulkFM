import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import torch
import pandas as pd
from models.bulkfm import BulkFM, BulkFMConfig


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k == "_gene_emb_base":
            continue
        cleaned[k] = v
    return cleaned


def infer_num_genes(state_dict: dict) -> int:
    for key, val in state_dict.items():
        if "gene_embedding.weight" in key:
            return val.shape[0]
    raise ValueError("Could not infer num_genes from checkpoint")


def config_from_checkpoint(checkpoint_path: str) -> tuple[BulkFMConfig, bool]:
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            data = json.load(f)
        return BulkFMConfig(
            hidden_dim=data.get("hidden_dim", 256),
            ffn_dim=data.get("ffn_dim", 1024),
            num_heads=data.get("num_heads", 8),
            num_layers=data.get("num_layers", 4),
            num_bins=data.get("num_bins", 50),
            expression_embedding=data.get("expression_embedding", "binned"),
            masking_strategy=data.get("masking_strategy", "mask_token"),
            mask_ratio=data.get("mask_ratio", 0.15),
            mask_token_id=data.get("mask_token", -10.0),
            simple_projection=data.get("expression_projection", "nonlinear") == "linear",
            sample_level_emb=data.get("sample_level_emb", 0),
        ), True
    return None, False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Extract gene embeddings for gene-embedding-benchmarks"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gene-list", type=str, default=None,
                        help="TXT with gene IDs (one per line, model order)")
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument("--output-name", type=str, default="bulkfm")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=50)
    parser.add_argument("--expression-embedding", type=str, default="binned",
                        choices=["binned", "continuous"])
    parser.add_argument("--masking-strategy", type=str, default="mask_token",
                        choices=["mask_token", "cls_bottleneck"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = clean_state_dict(state_dict)

    num_genes = infer_num_genes(state_dict)

    # Try loading config from checkpoint directory
    cfg, found = config_from_checkpoint(args.checkpoint)
    if found:
        print("Loaded config from checkpoint directory")
    else:
        cfg = BulkFMConfig(
            hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_bins=args.num_bins,
            expression_embedding=args.expression_embedding,
            masking_strategy=args.masking_strategy,
        )

    model = BulkFM(num_genes, cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    gene_ids = torch.arange(num_genes, device=device)
    gene_emb = model.gene_embedding(gene_ids).cpu().numpy()

    if args.gene_list:
        with open(args.gene_list) as f:
            gene_names = [l.strip() for l in f if l.strip()]
        if len(gene_names) != num_genes:
            print(f"Warning: gene list has {len(gene_names)} genes but model expects {num_genes}")
        if len(gene_names) < num_genes:
            gene_names = gene_names + [f"gene_{i}" for i in range(len(gene_names), num_genes)]
    else:
        gene_names = [f"gene_{i}" for i in range(num_genes)]

    df = pd.DataFrame(gene_emb, index=gene_names)
    subfolder = os.path.join(args.output_dir, args.output_name)
    os.makedirs(subfolder, exist_ok=True)

    csv_path = os.path.join(subfolder, f"{args.output_name}emb.csv")
    txt_path = os.path.join(subfolder, f"{args.output_name}genelist.txt")

    df.to_csv(csv_path, header=False, index=False)
    with open(txt_path, "w") as f:
        for g in gene_names:
            f.write(f"{g}\n")

    print(f"Saved embeddings ({df.shape[0]} genes x {df.shape[1]} dims) to:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"Place in: data/embeddings/all_genes/{args.output_name}/ or data/embeddings/intersect/{args.output_name}/")


if __name__ == "__main__":
    main()
