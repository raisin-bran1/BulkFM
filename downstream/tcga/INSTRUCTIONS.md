# Extract embeddings from your trained model
python downstream/tcga/model_tcga_embeddings.py \
    --checkpoint checkpoints/train_20260729_230537_local/best_model.pt \
    --output tcga_bulkfm.pt

# Run visualization and benchmarks — automatically includes all .pt files
python downstream/tcga/visualize_embeddings.py --legend
python downstream/tcga/classify_cancer_type.py
python downstream/tcga/imputation.py