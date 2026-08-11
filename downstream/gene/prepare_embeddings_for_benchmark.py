import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import time
import torch
import pandas as pd
import numpy as np
from models.bulkfm import BulkFM, BulkFMConfig


def load_gene_vocabulary(path: str) -> list[str]:
    df = pd.read_csv(path)
    return df["genes"].dropna().astype(str).tolist()


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


def map_ensg_to_entrez(ensg_ids: list[str], cache_path: str) -> dict[str, str]:
    if os.path.exists(cache_path):
        print(f"  Loading cached ENSG→Entrez mapping from {cache_path}")
        cache_df = pd.read_csv(cache_path)
        return dict(zip(cache_df["ensg"], cache_df["entrez"].astype(str)))

    try:
        import mygene
    except ImportError:
        print("ERROR: mygene not available. Install it: pip install mygene", file=sys.stderr)
        sys.exit(1)

    mg = mygene.MyGeneInfo()

    batch_size = 1000
    mapping = {}
    all_missing = []
    for i in range(0, len(ensg_ids), batch_size):
        batch = ensg_ids[i:i + batch_size]
        results = mg.querymany(
            batch, scopes="ensembl.gene", fields="entrezgene",
            species="human", returnall=True,
        )
        for r in results["out"]:
            q = r["query"]
            entrez = r.get("entrezgene")
            if entrez is not None:
                mapping[q] = str(int(entrez))
        all_missing.extend(results["missing"])
        print(f"    batch {i // batch_size + 1}/{(len(ensg_ids) + batch_size - 1) // batch_size}: "
              f"{len(mapping)} mapped so far", end="\r")
        time.sleep(0.5)

    print()
    if all_missing:
        print(f"  Warning: {len(all_missing)} ENSG IDs had no Entrez mapping")
    print(f"  Mapped {len(mapping)} / {len(ensg_ids)} ENSG IDs to Entrez")

    pd.DataFrame({"ensg": list(mapping.keys()), "entrez": list(mapping.values())}).to_csv(
        cache_path, index=False
    )
    print(f"  Saved mapping cache to {cache_path}")
    return mapping


def extract_embeddings(checkpoint_path: str, cfg: BulkFMConfig, device: torch.device,
                       extraction_mode: str = "table"):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = clean_state_dict(state_dict)

    num_genes = None
    for key, val in state_dict.items():
        if "gene_embedding.weight" in key:
            num_genes = val.shape[0]
            break
    if num_genes is None:
        raise ValueError("Could not infer num_genes from checkpoint")

    model = BulkFM(num_genes, cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if extraction_mode == "table":
        gene_ids = torch.arange(num_genes, device=device)
        gene_emb = model.gene_embedding(gene_ids).cpu().numpy()
    elif extraction_mode == "forward_mask":
        if cfg.masking_strategy != "mask_token":
            raise ValueError(
                "forward_mask extraction requires masking_strategy='mask_token'")
        x = torch.full((1, num_genes), cfg.mask_token_id,
                       dtype=torch.float32, device=device)
        with torch.no_grad():
            h = model(x, output_hidden=True)
        gene_emb = h[0].cpu().numpy()
    else:
        raise ValueError(f"Unknown extraction mode: {extraction_mode}")
    return gene_emb, num_genes


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Extract BulkFM gene embeddings and prepare for gene-embedding-benchmarks"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="BulkFM")
    parser.add_argument("--benchmark-dir", type=str,
                        default=os.path.expanduser("~/gene-embedding-benchmarks"),
                        help="Path to gene-embedding-benchmarks repo")
    parser.add_argument("--mapping-cache", type=str, default=None,
                        help="Path to cache ENSG→Entrez mapping CSV")
    parser.add_argument("--extraction-mode", type=str, default="table",
                        choices=["table", "forward_mask"],
                        help="table: raw gene_embedding table; forward_mask: run model "
                             "on all-mask-token input and take hidden states (requires "
                             "mask_token masking strategy)")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=50)
    parser.add_argument("--expression-embedding", type=str, default=None,
                        choices=["binned", "continuous"],
                        help="Overrides auto-detection from config.json")
    parser.add_argument("--masking-strategy", type=str, default=None,
                        choices=["mask_token", "cls_bottleneck"],
                        help="Overrides auto-detection from config.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    embeddings_base = os.path.join(benchmark_dir, "data", "embeddings")
    os.makedirs(embeddings_base, exist_ok=True)

    # Auto-detect config from checkpoint dir if available
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            data = json.load(f)
        print(f"Loaded config from {config_path}")
        cfg = BulkFMConfig(
            hidden_dim=data.get("hidden_dim", args.hidden_dim),
            ffn_dim=data.get("ffn_dim", args.ffn_dim),
            num_heads=data.get("num_heads", args.num_heads),
            num_layers=data.get("num_layers", args.num_layers),
            num_bins=data.get("num_bins", args.num_bins),
            expression_embedding=args.expression_embedding or data.get("expression_embedding", "continuous"),
            masking_strategy=args.masking_strategy or data.get("masking_strategy", "mask_token"),
            simple_projection=data.get("expression_projection", "nonlinear") == "linear",
            sample_level_emb=data.get("sample_level_emb", 0),
        )
    else:
        print("No config.json found, using CLI args")
        cfg = BulkFMConfig(
            hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_bins=args.num_bins,
            expression_embedding=args.expression_embedding or "continuous",
            masking_strategy=args.masking_strategy or "mask_token",
        )

    gene_vocab_path = os.path.join(root_path, "checkpoints", "gene_vocabulary.csv")
    if not os.path.exists(gene_vocab_path):
        print(f"ERROR: gene_vocabulary.csv not found at {gene_vocab_path}", file=sys.stderr)
        sys.exit(1)

    print("=== Step 1: Load gene vocabulary ===")
    ensg_ids = load_gene_vocabulary(gene_vocab_path)
    print(f"  Loaded {len(ensg_ids)} ENSG IDs")

    print("=== Step 2: Extract gene embeddings ===")
    t0 = time.time()
    gene_emb, num_genes = extract_embeddings(args.checkpoint, cfg, device,
                                             args.extraction_mode)
    dt = time.time() - t0
    print(f"  Extracted {num_genes} genes x {gene_emb.shape[1]} dims in {dt:.1f}s "
          f"(mode={args.extraction_mode})")

    print("=== Step 3: Map ENSG→Entrez ===")
    mapping_cache = args.mapping_cache or os.path.join(
        embeddings_base, "ensg_to_entrez.csv"
    )
    ensg_to_entrez = map_ensg_to_entrez(ensg_ids, mapping_cache)

    entrez_ids = [ensg_to_entrez.get(e, "") for e in ensg_ids]

    if gene_emb.shape[0] != len(ensg_ids):
        n = min(gene_emb.shape[0], len(ensg_ids))
        print(f"  Warning: truncating to {n} rows")
        gene_emb = gene_emb[:n]
        entrez_ids = entrez_ids[:n]

    print("=== Step 4: Save to benchmark repo ===")

    has_entrez = [i for i, e in enumerate(entrez_ids) if e]
    n_mapped = len(has_entrez)
    print(f"  {n_mapped} / {len(entrez_ids)} genes have Entrez IDs")

    # Deduplicate: when multiple ENSG map to same Entrez, keep the first occurrence
    def deduplicate(emb: np.ndarray, gene_list: list[str]):
        seen = set()
        keep = []
        for i, g in enumerate(gene_list):
            if g not in seen:
                seen.add(g)
                keep.append(i)
        return emb[keep], [gene_list[i] for i in keep]

    def save(subdir: str, emb: np.ndarray, gene_list: list[str]):
        emb, gene_list = deduplicate(emb, gene_list)
        subfolder = os.path.join(embeddings_base, subdir, args.output_name)
        os.makedirs(subfolder, exist_ok=True)
        csv_path = os.path.join(subfolder, f"{args.output_name}emb.csv")
        txt_path = os.path.join(subfolder, f"{args.output_name}genelist.txt")
        pd.DataFrame(emb).to_csv(csv_path, header=False, index=False)
        with open(txt_path, "w") as f:
            for g in gene_list:
                f.write(f"{g}\n")
        print(f"  -> {subdir}/{args.output_name}/ ({len(gene_list)} genes x {emb.shape[1]} dims)")

    all_emb, all_genes = deduplicate(gene_emb[has_entrez],
                                     [entrez_ids[i] for i in has_entrez])
    save("all_genes", all_emb, all_genes)

    intersect_path = os.path.join(
        benchmark_dir, "data", "data_splits", "gene_level_benchmark",
        "go_generalization_folds_splits", "intersect_ref_genelist.txt"
    )
    if os.path.exists(intersect_path):
        with open(intersect_path) as f:
            intersect_set = set(l.strip() for l in f if l.strip())
        inter_idx = [i for i, e in enumerate(entrez_ids) if e and e in intersect_set]
        inter_emb = gene_emb[inter_idx]
        inter_genes = [entrez_ids[i] for i in inter_idx]
        save("intersect", inter_emb, inter_genes)
    else:
        print("  intersect_ref_genelist.txt not found, skipping intersect variant")

    print("=== Done ===")


if __name__ == "__main__":
    main()
