## 1. Configuration

Edit `training/config.py` before running:

| Key | Description | Default |
|---|---|---|
| `data_dir` | Path to ARCHS4 parquet chunks | `/global/scratch/users/brianzhou/archs4_human` |
| `checkpoint_dir` | Where checkpoints are saved | `checkpoints` |
| `batch_size` | Per-GPU batch size | 4 |
| `epochs` | Number of epochs | 20 |
| `expression_embedding` | `"binned"` or `"continuous"` | `"binned"` |
| `continuous_loss` | `"mse"` or `"poisson"` (continuous only) | `"mse"` |
| `masking_strategy` | `"mask_token"` or `"cls_bottleneck"` | `"mask_token"` |
| `mask_ratio` | Fraction of genes to mask | 0.15 |
| `dynamic_mask_range` | Tuple `(lo, hi)` for per-batch uniform sampling, or `null` for fixed | `null` |

Override via environment variables in W&B sweep (see sweep.yaml).

## 2. Training

### Smoke test (10 seconds, 1 GPU)

```bash
cd /path/to/BulkFMtemp

# Verify your data path first
USE_SMOKE=1 torchrun --nproc_per_node=1 training/train.py
```

This runs 1 epoch on 100 samples. Check that loss decreases and no errors occur.

### Full training (single GPU)

```bash
torchrun --nproc_per_node=1 training/train.py
```

### Full training (multi-GPU, e.g., 4 GPUs)

```bash
torchrun --nproc_per_node=4 training/train.py
```

The default config trains `binned + mask_token` (original Binformer). To change config:

```bash
# Edit training/config.py directly, or override via env:
# This doesn't work directly -- edit config.py instead
```

### Checkpoints

During training, checkpoints are saved to `checkpoints/<run_id>/`:
- `epoch_XX.pt` — end of each epoch
- `best_model.pt` — best validation loss so far
- `config.json` — run config + best loss
- `loss_history.csv` — per-epoch losses
- `loss_plot.png` — loss curve

A global best model is also saved at `checkpoints/best_model.pt`.

## 3. Gene Embedding Extraction

After training, extract gene embeddings for the benchmark:

```bash
python scripts/extract_embeddings.py \
    --checkpoint checkpoints/<run_id>/best_model.pt \
    --gene-list /path/to/entrez_gene_list.txt \
    --output-dir embeddings \
    --output-name generalized_binformer_binned
```

**Important**: `--gene-list` must contain Entrez gene IDs (one per line) in the **same order** as the model's gene vocabulary (i.e., matching the parquet column order). If your model was trained on ARCHS4 human, the column order matches the Ensembl IDs in the parquet schema. Create a mapping:

```python
# One-time: generate gene list from parquet schema
python -c "
import pyarrow.parquet as pq
import pandas as pd

# Read gene column names from a batch file
pf = pq.ParquetFile('/path/to/archs4_human/batch_00000.parquet')

# These are the column names (likely Ensembl IDs)
genes = [c for c in pf.schema_arrow.names if c not in {'geo_accession', '__index_level_0__'}]

# Map to Entrez IDs using mygene
import mygene
mg = mygene.MyGeneInfo()
results = mg.querymany(genes, scopes='ensembl.gene', fields='entrezgene', species='human')

# Save mapping
mapping = []
for r in results:
    mapping.append({'model_id': r['query'],
                    'entrez_id': r.get('entrezgene', r['query'])})
pd.DataFrame(mapping).to_csv('gene_mapping.csv', index=False)

# Save just the Entrez IDs in model order
entrez_ids = [str(r.get('entrezgene', r['query'])) for r in results]
with open('entrez_gene_list.txt', 'w') as f:
    for eid in entrez_ids:
        f.write(f'{eid}\n')
"
```

Output is placed in `embeddings/generalized_binformer_binned/`:
- `generalized_binformer_binnedemb.csv` — embedding matrix (no header, rows=genes)
- `generalized_binformer_binnedgenelist.txt` — Entrez gene IDs, one per line

## 4. Benchmarking with gene-embedding-benchmarks

### Clone the benchmark repo

```bash
git clone https://github.com/ylaboratory/gene-embedding-benchmarks.git
cd gene-embedding-benchmarks
conda env create -f env.yml
conda activate gene_embed_benchmark
```

### Place your embeddings

```bash
# Copy your extracted embeddings into the benchmark repo
mkdir -p gene-embedding-benchmarks/data/embeddings/all_genes/generalized_binformer_binned
cp /path/to/BulkFMtemp/embeddings/generalized_binformer_binned/* \
   gene-embedding-benchmarks/data/embeddings/all_genes/generalized_binformer_binned/
```

### (Optional) Intersect with common genes

If you want to compare against other methods on the same gene set:

```bash
cd gene-embedding-benchmarks
python src/other/preprocess_embedding/preprocess.ipynb  # Run the intersect cell
```

This creates filtered embeddings in `data/embeddings/intersect/`.

### Run benchmarks

**Gene-level benchmarks** (disease gene prediction, GO function prediction):

```bash
cd gene-embedding-benchmarks

# Download data splits if not present
# (provided in repo under data/data_splits/)

# Run all gene-level tasks
python src/gene_level_benchmark/gene_level_benchmarks.py \
    --subfolder data/embeddings/all_genes/generalized_binformer_binned \
    --cv-fold1-pkl data/data_splits/gene_level/task_cv_fold1_dict_all.pkl \
    --cv-fold2-pkl data/data_splits/gene_level/task_cv_fold2_dict_all.pkl \
    --cv-fold3-pkl data/data_splits/gene_level/task_cv_fold3_dict_all.pkl \
    --holdout-pkl data/data_splits/gene_level/task_holdout_dict.pkl \
    --out-root results/tsvs
```

Or use the provided shell script:

```bash
bash src/gene_level_benchmark/run_gene_level.sh
```

**Paired-gene benchmarks** (genetic interaction, TF target prediction):

```bash
python src/gene_pair_benchmark/gene_pair_benchmarks.py \
    --subfolder data/embeddings/all_genes/generalized_binformer_binned \
    --task_idx 0 \
    --data-dir data/paired_gene_interaction_data \
    --out-dir results/tsvs
```

Or use the shell script:

```bash
bash src/gene_pair_benchmark/run_gene_pair.sh
```

**Gene-set benchmarks** (pathway enrichment, requires ANDES):

```bash
bash src/gene_set_benchmark/run_andes_batch.sh
```

Results are written to `results/tsvs/`. See the benchmark repo README for interpretation.

## Full Workflow Example (Single VM)

```bash
# === PHASE 1: Setup ===
git clone https://github.com/your-org/BulkFMtemp.git
cd BulkFMtemp
conda env create -f environment.yml -n bulkfm
conda activate bulkfm

# === PHASE 2: Download data ===
# (Use archs4py or copy from existing location)
# Set data_dir in training/config.py

# === PHASE 3: Smoke test ===
USE_SMOKE=1 torchrun --nproc_per_node=1 training/train.py

# === PHASE 4: Train ===
# Edit config.py for your experiment, then:
torchrun --nproc_per_node=1 training/train.py

# === PHASE 5: Extract embeddings ===
# Generate Entrez gene list (one-time)
python -c "
import pyarrow.parquet as pq, pandas as pd, mygene
pf = pq.ParquetFile('/path/to/archs4_human/batch_00000.parquet')
genes = [c for c in pf.schema_arrow.names if c not in {'geo_accession', '__index_level_0__'}]
mg = mygene.MyGeneInfo(); results = mg.querymany(genes, scopes='ensembl.gene', fields='entrezgene', species='human')
with open('entrez_gene_list.txt', 'w') as f:
    for r in results: f.write(f\"{str(r.get('entrezgene', r['query']))}\n\")
"

python scripts/extract_embeddings.py \
    --checkpoint checkpoints/best_model.pt \
    --gene-list entrez_gene_list.txt \
    --output-dir embeddings \
    --output-name generalized_binformer

# === PHASE 6: Benchmark ===
cd ..
git clone https://github.com/ylaboratory/gene-embedding-benchmarks.git
cd gene-embedding-benchmarks
conda env create -f env.yml -n bench
conda activate bench
mkdir -p data/embeddings/all_genes/generalized_binformer
cp ../BulkFMtemp/embeddings/generalized_binformer/* data/embeddings/all_genes/generalized_binformer/
bash src/gene_level_benchmark/run_gene_level.sh
```

### Expected wall times (single A100, 1 GPU)

| Config | Samples | Epochs | Time |
|---|---|---|---|
| Smoke (`USE_SMOKE=1`) | 100 | 1 | ~10s |
| Default (4 chunks) | ~20K | 20 | ~2h |
| Full (20 chunks) | ~100K | 20 | ~10h |
