import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.typing import SparseTensor

from models.bulkformer.BulkFormer import BulkFormer
from models.bulkformer.Bulkformer_params import get_params

GRAPH_PATH = "weights/bulkformer/G_tcga.pt"
GRAPH_WEIGHTS_PATH = "weights/bulkformer/G_tcga_weight.pt"
GENE_INFO_PATH = "weights/bulkformer/bulkformer_gene_info.csv"
MODEL_VARIANTS = {
    "BulkFormer_37M.pt": 1,
    "BulkFormer_50M.pt": 2,
    "BulkFormer_93M.pt": 3,
    "BulkFormer_127M.pt": 4,
    "BulkFormer_147M.pt": 0,
}


def load_gene_vocabulary(gene_info_path: str) -> list[str]:
    df = pd.read_csv(gene_info_path)
    return df["ensg_id"].dropna().astype(str).tolist()


def map_ensg_to_entrez(ensg_ids: list[str], cache_path: str) -> dict[str, str]:
    if os.path.exists(cache_path):
        print(f"  Loading cached ENSG→Entrez mapping from {cache_path}")
        cache_df = pd.read_csv(cache_path, dtype={"entrez": str})
        return dict(zip(cache_df["ensg"], cache_df["entrez"]))

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


@torch.no_grad()
def extract_gene_embeddings(model: BulkFormer, num_genes: int, source: str,
                            device: torch.device) -> np.ndarray:
    if source == "table":
        emb = model.gene_emb_onehot_layer.weight.detach().cpu().numpy()
        return emb

    if source == "forward_zero":
        x_ref = torch.zeros(1, num_genes, device=device)
        mask_prob = 0.0
    elif source == "forward_mask":
        x_ref = torch.full((1, num_genes), model.expr_emb.mask_token_id,
                           dtype=torch.float32, device=device)
        mask_prob = 1.0
    else:
        raise ValueError(f"Unknown embedding source: {source}")

    gene_emb = model(x_ref, mask_prob=mask_prob, output_expr=False)
    return gene_emb[0, :, :model.dim].detach().cpu().numpy()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Extract BulkFormer gene embeddings and prepare for gene-embedding-benchmarks"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="weights/bulkformer/BulkFormer_147M.pt")
    parser.add_argument("--output-name", type=str, default="BulkFormer-147M")
    parser.add_argument("--embedding-source", type=str, default="forward_mask",
                        choices=["forward_mask", "forward_zero", "table"],
                        help="forward_mask: run model with all-mask-token input (in-distribution "
                             "contextualized via graph+attention); forward_zero: run model on zero "
                             "input; table: raw gene_emb_onehot_layer.weight")
    parser.add_argument("--gene-info", type=str, default=GENE_INFO_PATH)
    parser.add_argument("--graph", type=str, default=GRAPH_PATH)
    parser.add_argument("--graph-weights", type=str, default=GRAPH_WEIGHTS_PATH)
    parser.add_argument("--benchmark-dir", type=str,
                        default=os.path.expanduser("~/gene-embedding-benchmarks"),
                        help="Path to gene-embedding-benchmarks repo")
    parser.add_argument("--mapping-cache", type=str, default=None,
                        help="Path to cache ENSG→Entrez mapping CSV")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # ── Pick model config from checkpoint name ──
    base = os.path.basename(args.checkpoint)
    idx = MODEL_VARIANTS.get(base)
    if idx is None:
        for name, i in MODEL_VARIANTS.items():
            if name in args.checkpoint:
                idx = i
                break
    if idx is None:
        print(f"Warning: unknown model variant for {base}, defaulting to BulkFormer-147M")
        idx = 0
    params = get_params(idx)
    print(f"Model variant: {base} | dim={params['dim']}, p_repeat={params['p_repeat']}")

    # ── Load gene graph (official construction) ──
    print("Loading gene graph...")
    edge_index = torch.load(args.graph, map_location="cpu", weights_only=True)
    edge_weight = torch.load(args.graph_weights, map_location="cpu", weights_only=True)
    graph = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight).t().to(device)

    # ── Build model ──
    model = BulkFormer(
        dim=params["dim"],
        graph=graph,
        gene_emb=None,
        gene_length=params["gene_length"],
        bin_head=params["bin_head"],
        full_head=params["full_head"],
        bins=params["bins"],
        gb_repeat=params["gb_repeat"],
        p_repeat=params["p_repeat"],
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing state dict keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected state dict keys skipped: {len(unexpected)}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} params")

    num_genes = params["gene_length"]

    print("=== Step 1: Extract gene embeddings ===")
    t0 = time.time()
    gene_emb = extract_gene_embeddings(model, num_genes, args.embedding_source, device)
    print(f"  Extracted {gene_emb.shape[0]} genes x {gene_emb.shape[1]} dims "
          f"in {time.time()-t0:.1f}s (source={args.embedding_source})")

    print("=== Step 2: Load gene vocabulary & map ENSG→Entrez ===")
    ensg_ids = load_gene_vocabulary(args.gene_info)
    if len(ensg_ids) != num_genes:
        print(f"  Warning: gene_info has {len(ensg_ids)} genes but model expects {num_genes}")
    print(f"  Loaded {len(ensg_ids)} ENSG IDs")

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    embeddings_base = os.path.join(benchmark_dir, "data", "embeddings")
    os.makedirs(embeddings_base, exist_ok=True)
    mapping_cache = args.mapping_cache or os.path.join(
        embeddings_base, "ensg_to_entrez.csv"
    )
    ensg_to_entrez = map_ensg_to_entrez(ensg_ids, mapping_cache)

    entrez_ids = [ensg_to_entrez.get(e, "") for e in ensg_ids]
    if gene_emb.shape[0] != len(entrez_ids):
        n = min(gene_emb.shape[0], len(entrez_ids))
        print(f"  Warning: truncating to {n} rows")
        gene_emb = gene_emb[:n]
        entrez_ids = entrez_ids[:n]

    print("=== Step 3: Save to benchmark repo ===")

    has_entrez = [i for i, e in enumerate(entrez_ids) if e]
    n_mapped = len(has_entrez)
    print(f"  {n_mapped} / {len(entrez_ids)} genes have Entrez IDs")

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
