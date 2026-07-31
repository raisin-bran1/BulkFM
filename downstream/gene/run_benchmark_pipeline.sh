#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash downstream/gene/run_benchmark_pipeline.sh \
#       <checkpoint_path> <output_name> <masking_strategy>
#
# Example:
#   bash downstream/gene/run_benchmark_pipeline.sh \
#       checkpoints/train_20260730_223417_local/best_model.pt \
#       BulkFM-CLS-CONT-15 cls_bottleneck
#
# Prerequisites:
#   - conda environments: nasa (BulkFM), gene_embed_benchmark (benchmark repo)
#   - gene-embedding-benchmarks cloned at ~/gene-embedding-benchmarks
#   - checkpoints/gene_vocabulary.csv exists
#   - Benchmark data splits exist

CKPT="$1"
OUTPUT_NAME="$2"
MASKING_STRATEGY="$3"

BENCH_DIR="$HOME/gene-embedding-benchmarks"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$ROOT_DIR/results/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  BulkFM Benchmark Pipeline"
echo "  Checkpoint: $CKPT"
echo "  Output:     $OUTPUT_NAME"
echo "  Masking:    $MASKING_STRATEGY"
echo "=============================================="

# ---- Step 1: Extract embeddings & prepare ----
echo ""
echo "=== Step 1: Extract and prepare embeddings ==="
conda run -n nasa python "$ROOT_DIR/downstream/gene/prepare_embeddings_for_benchmark.py" \
    --checkpoint "$CKPT" \
    --output-name "$OUTPUT_NAME" \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_${OUTPUT_NAME}_extract.log"

# ---- Step 2: Run gene-level benchmarks ----
echo ""
echo "=== Step 2a: GO benchmark ==="
# Save existing OMIM results if present (script overwrites output)
for f in "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_fold_results.pkl" \
         "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results.pkl"; do
    if [ -f "$f" ]; then
        mv "$f" "${f%.pkl}_omim.pkl" 2>/dev/null || true
    fi
done

conda run -n gene_embed_benchmark python \
    "$BENCH_DIR/src/gene_level_benchmark/gene_level_benchmarks.py" \
    --subfolder "$BENCH_DIR/data/embeddings/intersect/$OUTPUT_NAME" \
    --cv-fold1-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold1_dict_all.pkl" \
    --cv-fold2-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold2_dict_all.pkl" \
    --cv-fold3-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/go_folds_splits/go_cv_fold3_dict_all.pkl" \
    --holdout-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/go_folds_splits/go_holdout_dict_all.pkl" \
    -d "$BENCH_DIR/results/tsvs" \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_${OUTPUT_NAME}_go.log"

# Move GO results aside
for f in "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_fold_results.pkl" \
         "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results.pkl"; do
    if [ -f "$f" ]; then
        mv "$f" "${f%.pkl}_go.pkl"
    fi
done

# Restore OMIM results if they were backed up
for f in "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_fold_results_omim.pkl" \
         "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results_omim.pkl"; do
    if [ -f "$f" ]; then
        mv "$f" "${f%_omim.pkl}.pkl"
    fi
done

echo ""
echo "=== Step 2b: OMIM benchmark ==="
conda run -n gene_embed_benchmark python \
    "$BENCH_DIR/src/gene_level_benchmark/gene_level_benchmarks.py" \
    --subfolder "$BENCH_DIR/data/embeddings/intersect/$OUTPUT_NAME" \
    --cv-fold1-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold1_dict_all.pkl" \
    --cv-fold2-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold2_dict_all.pkl" \
    --cv-fold3-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/omim_folds_splits/omim_cv_fold3_dict_all.pkl" \
    --holdout-pkl "$BENCH_DIR/data/data_splits/gene_level_benchmark/omim_folds_splits/omim_holdout_dict_all.pkl" \
    -d "$BENCH_DIR/results/tsvs" \
    2>&1 | tee "$LOG_DIR/${TIMESTAMP}_${OUTPUT_NAME}_omim.log"

# ---- Step 3: Run gene-pair benchmarks (in parallel) ----
echo ""
echo "=== Step 3: Gene-pair benchmarks (parallel) ==="
PAIR_PIDS=()
for pair in ng sl tf pombe; do
    echo "  Pair: $pair"
    conda run -n gene_embed_benchmark python \
        "$BENCH_DIR/src/gene_pair_benchmark/gene_pair_benchmarks.py" \
        --subfolder "$BENCH_DIR/data/embeddings/intersect/$OUTPUT_NAME" \
        --operation sum \
        --cv-pkl "$BENCH_DIR/data/data_splits/gene_pair_benchmark/${pair}_nested_cv_splits.pkl" \
        --out-root "$BENCH_DIR/results/tsvs" \
        --suffix "sum_intersected_${pair}" \
        2>&1 | tee "$LOG_DIR/${TIMESTAMP}_${OUTPUT_NAME}_pair_${pair}.log" &
    PAIR_PIDS+=("$!")
done
for pid in "${PAIR_PIDS[@]}"; do
    wait "$pid" || { echo "ERROR: one of the gene-pair benchmarks failed (pid $pid)"; exit 1; }
done

# ---- Step 4: Append results to CSV ----
echo ""
echo "=== Step 4: Append results to results/gene_benchmark_results.csv ==="
RESULTS_CSV="$ROOT_DIR/results/gene_benchmark_results.csv"

# GO: extract mean holdout AUROC from go pkl
if [ -f "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results_go.pkl" ]; then
    GO_AUROC=$(python3 -c "
import pickle, sys
with open('$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results_go.pkl', 'rb') as f:
    data = pickle.load(f)
aucs = [df['AUC'].values[0] for df in data.values()]
print(f'{sum(aucs)/len(aucs):.4f}')
")
    echo "${OUTPUT_NAME},GO,holdout_AUROC,${GO_AUROC}" >> "$RESULTS_CSV"
    echo "  GO: $GO_AUROC"
fi

# OMIM: extract mean holdout AUROC from omim pkl
if [ -f "$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results.pkl" ]; then
    OMIM_AUROC=$(python3 -c "
import pickle
with open('$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results.pkl', 'rb') as f:
    data = pickle.load(f)
aucs = [df['AUC'].values[0] for df in data.values()]
print(f'{sum(aucs)/len(aucs):.4f}')
")
    echo "${OUTPUT_NAME},OMIM,holdout_AUROC,${OMIM_AUROC}" >> "$RESULTS_CSV"
    echo "  OMIM: $OMIM_AUROC"
fi

# Gene-pair: extract outer_AUC from each pair CSV
for pair in ng sl tf pombe; do
    csv="$BENCH_DIR/results/tsvs/${OUTPUT_NAME}_sum_intersected_${pair}.csv"
    if [ -f "$csv" ]; then
        PAIR_AUC=$(tail -1 "$csv" | cut -d, -f7)
        echo "${OUTPUT_NAME},${pair},outer_AUC,${PAIR_AUC}" >> "$RESULTS_CSV"
        echo "  ${pair}: $PAIR_AUC"
    fi
done

echo "  Appended to $RESULTS_CSV"

# ---- Summary ----
echo ""
echo "=============================================="
echo "  Pipeline complete for $OUTPUT_NAME"
echo ""
echo "  Results:"
echo "    GO:    $BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results_go.pkl"
echo "    OMIM:  $BENCH_DIR/results/tsvs/${OUTPUT_NAME}_holdout_results.pkl"
echo "    NG:    $BENCH_DIR/results/tsvs/${OUTPUT_NAME}_sum_intersected_ng.csv"
echo "    SL:    $BENCH_DIR/results/tsvs/${OUTPUT_NAME}_sum_intersected_sl.csv"
echo "    TF:    $BENCH_DIR/results/tsvs/${OUTPUT_NAME}_sum_intersected_tf.csv"
echo "    pombe: $BENCH_DIR/results/tsvs/${OUTPUT_NAME}_sum_intersected_pombe.csv"
echo "  Logs:   $LOG_DIR/${TIMESTAMP}_${OUTPUT_NAME}_*.log"
echo "=============================================="
