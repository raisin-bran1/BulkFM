import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import torch
import pandas as pd
from models.generalized_binformer import GeneralizedBinformer, GeneralizedBinformerConfig


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Extract gene embeddings for gene-embedding-benchmarks"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gene-list", type=str, default=None,
                        help="TXT with Entrez gene IDs (one per line, model order)")
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument("--output-name", type=str, default="generalized_binformer")
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

    cfg = GeneralizedBinformerConfig(
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_bins=args.num_bins,
        expression_embedding=args.expression_embedding,
        masking_strategy=args.masking_strategy,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    num_genes = None
    for key, val in state_dict.items():
        if "gene_embedding.weight" in key:
            num_genes = val.shape[0]
            break
    if num_genes is None:
        raise ValueError("Could not infer num_genes from checkpoint")

    model = GeneralizedBinformer(num_genes, cfg)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
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
