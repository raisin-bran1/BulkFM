#!/bin/bash
# Phase 1: scan_mouse_h5.py -- one-time H5 scan (reads H5 once, saves metadata)
# Phase 2: split_mouse_data.py -- parallel download from H5 (expression only)
#
# Usage:
#   conda activate archs4
#   bash scan_download_mouse.sh [h5_jobs] [h5_path] [ref_dir] [output_dir]
#
# Defaults: h5_jobs=2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
H5="${2:-/media/volume/bulkrnamouse/mouse_gene_v2.latest.h5}"
DATA_DIR="${3:-$SCRIPT_DIR}"
OUTPUT_DIR="${4:-/media/volume/bulkrnamouse/mousedata/}"
CHUNK_SIZE=5000
H5_JOBS="${1:-2}"

echo "H5:               $H5"
echo "Ref dir:          $DATA_DIR"
echo "Output dir:       $OUTPUT_DIR"
echo "Chunk size:       $CHUNK_SIZE"
echo "H5 download jobs: $H5_JOBS  (keep low to avoid swapping)"

mkdir -p "$OUTPUT_DIR"

# Phase 1: Pre-scan (single H5 pass)
echo ""
echo "=== Phase 1: Pre-scan (SC prob filter + gene metadata) ==="
python3 -u "$SCRIPT_DIR/scan_mouse_h5.py" \
    --datapath "$H5" \
    --refdir "$DATA_DIR" \
    --outputdir "$OUTPUT_DIR" \
    --sc_threshold 0.5

# Count chunks
N_PASSING=$(python3 -c "
import numpy as np
a = np.load('$OUTPUT_DIR/passing_indices.npy')
print(len(a))
")

N_CHUNKS=$(( (N_PASSING + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo ""
echo "Pre-filtered samples: $N_PASSING -> $N_CHUNKS chunks of $CHUNK_SIZE"

# Phase 2: Parallel H5 download (limited workers to avoid swapping)
echo ""
echo "=== Phase 2: Downloading chunks from H5 ($H5_JOBS workers) ==="

seq 0 $((N_CHUNKS - 1)) | parallel -j "$H5_JOBS" --line-buffer \
    python3 -u "$SCRIPT_DIR/split_mouse_data.py" \
        --datapath "$H5" \
        --outputdir "$OUTPUT_DIR" \
        --chunk_id {} \
        --chunk_size "$CHUNK_SIZE"

echo ""
echo "All $N_CHUNKS chunks downloaded at $(date)."
echo "Output: ${OUTPUT_DIR}/input_chunk_*.parquet"
