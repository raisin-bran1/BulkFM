#!/bin/bash
# Preprocess TCGA data end-to-end.
#
# Steps:
#   1. build_tcga_ensg_mapping.py  — comprehensive symbol→ENSG mapping
#   2. scan_tcga.py                — build gene vocabulary
#   3. preprocess_tcga.py          — TPM → log1p, filter, attach metadata
#
# Usage:
#   conda activate archs4
#   bash preprocess_tcga.sh

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-/media/volume/bulkrnadata/}"
TCGA_OUTDIR="${2:-/media/volume/bulkrnadata/tcgadata/}"
ARCHS4_OUTDIR="${3:-/media/volume/bulkrnadata/humandata/}"

echo "Data dir:      $DATA_DIR"
echo "TCGA out dir:  $TCGA_OUTDIR"
echo "ARCHS4 out dir: $ARCHS4_OUTDIR"

mkdir -p "$TCGA_OUTDIR"

# ── Step 1: Build comprehensive ENSG mapping ───────────────────────
echo ""
echo "=== Step 1: build_tcga_ensg_mapping.py ==="
python3 -u "$SCRIPT_DIR/build_tcga_ensg_mapping.py" \
    --datapath "$DATA_DIR/tcga_matrix.h5" \
    --refdir "$DATA_DIR" \
    --archs4_outdir "$ARCHS4_OUTDIR" \
    --outputdir "$TCGA_OUTDIR"

if [ $? -ne 0 ]; then
    echo "ERROR: build_tcga_ensg_mapping.py failed"
    exit 1
fi

# ── Step 2: Scan TCGA, build vocabulary ────────────────────────────
echo ""
echo "=== Step 2: scan_tcga.py ==="
python3 -u "$SCRIPT_DIR/scan_tcga.py" \
    --datapath "$DATA_DIR/tcga_matrix.h5" \
    --tcga_outdir "$TCGA_OUTDIR" \
    --archs4_outdir "$ARCHS4_OUTDIR" \
    --outputdir "$TCGA_OUTDIR"

if [ $? -ne 0 ]; then
    echo "ERROR: scan_tcga.py failed"
    exit 1
fi

# ── Step 3: Preprocess expression + attach metadata ─────────────────
echo ""
echo "=== Step 3: preprocess_tcga.py ==="
python3 -u "$SCRIPT_DIR/preprocess_tcga.py" \
    --datapath "$DATA_DIR/tcga_matrix.h5" \
    --refdir "$DATA_DIR" \
    --tcga_outdir "$TCGA_OUTDIR" \
    --outputdir "$TCGA_OUTDIR"

if [ $? -ne 0 ]; then
    echo "ERROR: preprocess_tcga.py failed"
    exit 1
fi

echo ""
echo "==== Done ===="
echo "Output: ${TCGA_OUTDIR}/tcga_processed.parquet"
echo ""
python3 -c "
import pandas as pd
df = pd.read_parquet('${TCGA_OUTDIR}/tcga_processed.parquet')
expr_cols = [c for c in df.columns if c.startswith('ENSG')]
meta_cols = [c for c in df.columns if not c.startswith('ENSG')]
print(f'  Samples: {len(df)}')
print(f'  Genes:   {len(expr_cols)}')
print(f'  Meta:    {len(meta_cols)} fields')
if 'cancertype' in df.columns:
    print(f'  Cancer types: {df[\"cancertype\"].nunique()}')
"
