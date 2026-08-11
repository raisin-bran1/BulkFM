#!/usr/bin/env bash
set -euo pipefail

python downstream/tcga/visualize_embeddings.py --legend
python downstream/tcga/classify_cancer_type.py
python downstream/tcga/reconstruction.py