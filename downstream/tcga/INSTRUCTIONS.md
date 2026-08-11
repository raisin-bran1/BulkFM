# Extract embeddings from your trained model
python downstream/tcga/bulkfm_tcga_embeddings.py \
    --checkpoint checkpoints/train_20260810_072705_local/best_model.pt \
    --output BulkFM-FULL-VAR.pt

# Run visualization and benchmarks — automatically includes all .pt files
bash downstream/tcga/run_benchmark_pipeline.sh

"If you want UMAP-style separation: visualize the residual after dropping the top 4 PCs (it will cluster, matching the kNN jump), or change the model so the latent preserves more structure (full attention, or a contrastive sample-level training objective). If you want to match BulkFormer on the benchmark: you basically already do."