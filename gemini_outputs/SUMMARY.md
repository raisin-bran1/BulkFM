# Research Summary: OSDR Spaceflight Embedding Analysis

## 1. Methodology: Robust Biomarker Discovery
To move beyond random statistical flukes, we implemented a three-stage "Stable Discovery" pipeline:
- **Leave-One-Group-Out (LOGO) Cross-Validation**: Evaluated model performance by training on $N-1$ experimental batches and testing on an entirely unseen batch. This proved that simple random splits were overestimating accuracy due to batch effects.
- **Stability Selection**: Identified embedding dimensions that were consistently important across every single cross-validation fold. We used the ratio of Mean Weight to Standard Deviation ($|\mu|/\sigma$) as a stability metric.
- **Gene Correlation Mapping**: Correlated these stable dimensions back to the original ~16,000 genes to identify the biological "meaning" of the model's latent space.

## 2. Binformer vs. PCA Comparison
Head-to-head testing in the Thymus showed:
- **Accuracy**: Binformer out-performed PCA by **+9.6%** in cross-validation.
- **Signal Quality**: Binformer discovered a unique biological signature with **0% overlap** in the top 100 genes compared to PCA.
- **Interpretation**: PCA was "distracted" by technical noise (small nuclear RNAs and gene length bias), while Binformer focused on universal biological stressors (ribosomes/mitochondria).

## 3. Final Toolset (gemini_scripts/)
- `logo_cross_validation.py`: Robust accuracy benchmarking.
- `stable_biomarker_discovery.py`: Identifying stable gene-level drivers.
- `compare_binformer_vs_pca.py`: Statistical head-to-head testing.
- `plotter.py` (in `classify_embeddings/`): High-quality visualization with organ-filtering and UMAP tuning.
