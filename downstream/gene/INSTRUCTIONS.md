# Gene Embedding Benchmarking

This directory contains scripts and instructions for extracting BulkFM gene embeddings and evaluating them on the [gene-embedding-benchmarks](https://github.com/ylaboratory/gene-embedding-benchmarks) suite.

## Prerequisites

- A trained BulkFM checkpoint + `config.json` (in the checkpoint directory).
- `gene_vocabulary.csv` at `checkpoints/gene_vocabulary.csv` — the ENSG IDs in model order.
- Two conda environments:
  - `nasa` — BulkFM repo environment (used for extraction scripts).
  - `gene_embed_benchmark` — gene-embedding-benchmarks repo environment.
- The benchmark repo cloned at `~/gene-embedding-benchmarks`.

## Scripts

### `extract_gene_embeddings.py`

Low-level extraction: loads a checkpoint and saves raw embeddings as CSV + gene list.

```bash
conda activate nasa
python downstream/gene/extract_gene_embeddings.py \
    --checkpoint checkpoints/train_XXXXXX/best_model.pt \
    --gene-list checkpoints/gene_vocabulary.csv \
    --output-dir /tmp/embeddings \
    --output-name BulkFM-MODELNAME \
    --expression-embedding continuous \
    --masking-strategy cls_bottleneck
```

**Key details:**
- Loads `config.json` from the checkpoint directory if present (auto-detects `hidden_dim`, `masking_strategy`, etc.).
- Detects old checkpoint formats: strips `_orig_mod.` prefix, handles `_gene_emb_base` buffer, and sets `simple_projection=True` for checkpoints with single-layer projection.
- Output: `<output-name>emb.csv` (rows=genes, cols=embedding dims, no header) and `<output-name>genelist.txt` (gene IDs in same order).

### `prepare_embeddings_for_benchmark.py`

Full pipeline: loads checkpoint, extracts embeddings, maps ENSG→Entrez via mygene (with cache), deduplicates, and saves to the benchmark repo in both `all_genes/` and `intersect/` variants.

```bash
conda activate nasa
python downstream/gene/prepare_embeddings_for_benchmark.py \
    --checkpoint checkpoints/train_XXXXXX/best_model.pt \
    --output-name BulkFM-MODELNAME \
    --expression-embedding continuous \
    --masking-strategy cls_bottleneck
```

**Key details:**
- Reads `checkpoints/gene_vocabulary.csv` for ENSG IDs.
- Batch-queries mygene API (1000 at a time, 0.5s delay) to map ENSG→Entrez; caches to `data/embeddings/ensg_to_entrez.csv`.
- Deduplicates: when multiple ENSG IDs map to the same Entrez ID, keeps the first occurrence.
- Saves to `data/embeddings/all_genes/<output-name>/` (all mapped Entrez genes, 19213 unique).
- Saves to `data/embeddings/intersect/<output-name>/` (filtered to benchmark's 11355-gene reference, 11325 unique).
- Registers the model in `data/embed_meta.csv` (adds row with output name, dimensionality, method).

## Running the Benchmarks

Activate the benchmark environment:

```bash
conda activate gene_embed_benchmark
cd ~/gene-embedding-benchmarks
```

### Gene-Level Benchmarks (GO & OMIM)

The script does NOT support a `--suffix` flag; output filenames are derived from the subfolder basename. To avoid overwriting, back up existing results.

**GO (56 terms):**

```bash
python src/gene_level_benchmark/gene_level_benchmarks.py \
    --subfolder data/embeddings/intersect/BulkFM-MODELNAME \
    --cv-fold1-pkl data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold1_dict_all.pkl \
    --cv-fold2-pkl data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold2_dict_all.pkl \
    --cv-fold3-pkl data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold3_dict_all.pkl \
    --holdout-pkl data/data_splits/gene_level_benchmark/go_folds_splits/go_holdout_dict_all.pkl \
    -d results/tsvs
```

Output: `results/tsvs/<modelname>_fold_results.pkl` and `results/tsvs/<modelname>_holdout_results.pkl`.

**OMIM (103 diseases):**

```bash
python src/gene_level_benchmark/gene_level_benchmarks.py \
    --subfolder data/embeddings/intersect/BulkFM-MODELNAME \
    --cv-fold1-pkl data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold1_dict_all.pkl \
    --cv-fold2-pkl data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold2_dict_all.pkl \
    --cv-fold3-pkl data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold3_dict_all.pkl \
    --holdout-pkl data/data_splits/gene_level_benchmark/omim_folds_splits/omim_holdout_dict_all.pkl \
    -d results/tsvs
```

To extract average holdout AUROC from the pkl:

```python
import pickle
with open('results/tsvs/<modelname>_holdout_results.pkl', 'rb') as f:
    data = pickle.load(f)
aucs = [df['AUC'].values[0] for df in data.values()]
print(f"Mean AUROC: {sum(aucs)/len(aucs):.4f}")
```

### Gene-Pair Benchmarks

Four pair types: `ng` (negative genetic interaction, ~10k pairs), `sl` (synthetic lethal, ~2k pairs), `tf` (transcription factor, ~10k pairs), `pombe` (yeast, ~6k pairs). Uses `sum` operation to combine gene pair embeddings.

Use a unique `--suffix` per run to avoid overwriting:

```bash
for pair in ng sl tf pombe; do
    python src/gene_pair_benchmark/gene_pair_benchmarks.py \
        --subfolder data/embeddings/intersect/BulkFM-MODELNAME \
        --operation sum \
        --cv-pkl data/data_splits/gene_pair_benchmark/${pair}_nested_cv_splits.pkl \
        --out-root results/tsvs/ \
        --suffix sum_intersected_${pair}
done
```

Output: `results/tsvs/<modelname>_sum_intersected_<pair>.csv` with fold and average metrics.

### Registering a Model

Add an entry to `data/embed_meta.csv` with columns: `embedding` (folder name), `dim` (embedding dimensionality), `method` (human-readable name).

## What Was Run (This Session)

Two models were benchmarked:

| Model | Checkpoint | Expression | Masking | Mask Ratio |
|-------|-----------|-----------|---------|-----------|
| BulkFM-CLS | `train_20260729_230537_local` | continuous | cls_bottleneck | 75% |
| BulkFM-MASK | `train_20260730_080718_local` | continuous | mask_token | 15% |

Both use `intersect` gene set (11,325 Entrez IDs × 256 dims). Gene-pair benchmarks used `sum` operation.

## Results

Full results are in `results/gene_benchmark_results.csv`. Quick summary:

| Task | BulkFM-CLS | BulkFM-MASK | Frontier (GENEPT-MODEL3) |
|------|-----------|-------------|--------------------------|
| GO (AUROC) | 0.550 | 0.507 | **0.926** |
| OMIM (AUROC) | 0.512 | 0.484 | **0.877** |
| NG pair (AUC) | 0.545 | 0.598 | **0.709** |
| SL pair (AUC) | 0.398 | 0.666 | **0.898** |
| TF pair (AUC) | 0.473 | 0.463 | **0.797** |
| pombe (AUC) | 0.441 | 0.501 | **0.613** |

Both BulkFM variants underperform all protein foundation models (ESM2, T5, GENEPT-MODEL3) and even simpler baselines (MASHUP, BIOCONCEPTVEC) by 20-40 AUROC points.

## Notes

- The gene-level benchmark script writes to the same filenames regardless of task; always back up or rename pkl files between GO and OMIM runs.
- The mygene API has rate limits; use the cached `ensg_to_entrez.csv` for repeat runs.
- The benchmark output CSVs for different pair types share the same output filename pattern; always use `--suffix` to distinguish them.
- Old checkpoint formats with `_orig_mod.` prefix, `_gene_emb_base` buffer, or single-layer projection are handled automatically by the extraction scripts.
