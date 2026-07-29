#!/bin/bash
# Preprocess all downloaded mouse chunks in parallel.
# No H5 access -- reads precomputed gene metadata + input parquet files.
#
# Usage:
#   conda activate archs4
#   bash preprocess_mouse.sh [preprocess_jobs] [ref_dir] [output_dir] [--retry-failed]
#
# Defaults: preprocess_jobs=8, --retry-failed will re-run only failed chunks from last run.
# Must have run scan_download_mouse.sh first.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${2:-$SCRIPT_DIR}"
OUTPUT_DIR="${3:-/media/volume/bulkrnamouse/mousedata/}"
PREPROCESS_JOBS="${1:-8}"
RETRY=false
[ "$4" = "--retry-failed" ] && RETRY=true

JOBLOG="$OUTPUT_DIR/.preprocess_joblog"

echo "Ref dir:          $DATA_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "Preprocess jobs:  $PREPROCESS_JOBS"

N_CHUNKS=$(ls "$OUTPUT_DIR"/input_chunk_*.parquet 2>/dev/null | wc -l)
if [ "$N_CHUNKS" -eq 0 ]; then
    echo "Error: no input_chunk_*.parquet found in $OUTPUT_DIR"
    echo "Run scan_download_mouse.sh first."
    exit 1
fi
echo "Found $N_CHUNKS input chunks to preprocess."

echo ""
echo "=== Preprocessing chunks in parallel ($PREPROCESS_JOBS jobs) ==="

if [ "$RETRY" = true ]; then
    for i in $(seq 0 $((N_CHUNKS - 1))); do
        out="processed_mouse_$(printf '%03d' "$i").parquet"
        if [ ! -f "$OUTPUT_DIR/$out" ]; then
            printf "%d\tinput_chunk_%03d.parquet\n" "$i" "$i"
        fi
    done | parallel -j "$PREPROCESS_JOBS" --line-buffer --colsep '\t' \
        --joblog "$JOBLOG" \
        python3 -u "$SCRIPT_DIR/preprocessing_mouse.py" \
            --chunk_id '{1}' \
            --datadir "$OUTPUT_DIR" \
            --outputdir "$OUTPUT_DIR" \
            --refdir "$DATA_DIR" \
            --input '{2}'
else
    for i in $(seq 0 $((N_CHUNKS - 1))); do
        printf "%d\tinput_chunk_%03d.parquet\n" "$i" "$i"
    done | parallel -j "$PREPROCESS_JOBS" --line-buffer --colsep '\t' \
        --joblog "$JOBLOG" \
        python3 -u "$SCRIPT_DIR/preprocessing_mouse.py" \
            --chunk_id '{1}' \
            --datadir "$OUTPUT_DIR" \
            --outputdir "$OUTPUT_DIR" \
            --refdir "$DATA_DIR" \
            --input '{2}'
fi

echo ""
DONE=$(ls "$OUTPUT_DIR"/processed_mouse_*.parquet 2>/dev/null | wc -l)
echo "==== Summary: $DONE / $N_CHUNKS chunks processed ===="
if [ "$DONE" -lt "$N_CHUNKS" ]; then
    echo "Missing chunks:"
    for i in $(seq 0 $((N_CHUNKS - 1))); do
        out="processed_mouse_$(printf '%03d' "$i").parquet"
        if [ ! -f "$OUTPUT_DIR/$out" ]; then
            echo "  chunk $i"
        fi
    done
    echo ""
    echo "To retry only missing chunks, run:"
    echo "  bash $0 $PREPROCESS_JOBS $DATA_DIR $OUTPUT_DIR --retry-failed"
    echo ""
    echo "If chunks keep failing, try reducing parallelism: PREPROCESS_JOBS=4 or 2"
fi
echo "Output: ${OUTPUT_DIR}/processed_mouse_*.parquet"
