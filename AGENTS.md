The core goal of this project is to benchmark various architecture choices for bulk RNA foundation models. Currently, there is a limited number of such models in the literature and no clear consensus for what the best approaches are. Bulkformer uses continuous expression embeddings, linear attention, and intermixes GCN layers. On the other hand, BulkRNABert uses binned expression embeddings and full attention blocks. By taking different architectures with the same parameter counts trained on several matching dataset sizes, and evaluating them on the same downstream tasks, we will reveal what advantages these different approaches have. This will enable more informed architectural decisions for future development of bulk RNA foundation models.

ARCS4 human will be used for pretraining, and https://github.com/ylaboratory/gene-embedding-benchmarks/tree/master will be used for benchmarking. This repo has been cloned as ~/gene-embedding-benchmarks and has a conda environment gene_embed_benchmark (separate fomr ~/BulkFM with its own conda environment nasa).

Adjustable versions of the model:
1. Continuous vs binned expression embeddings
2. Different masking ratios, including dynamic masking
3. Encoder bottleneck: Mask tokens vs cls reconstruction

Other architecture choices:
Gene embeddings: initialized randomly vs frozen esm (learned linear projection)
How to combine gene & expression embeddings?
- I should make expression projection nonlinear oops
Attention: how much worse is linear attention?

Training & validation data (ARCHS4) is in /media/volume/bulkrnadata/humandata
Eval data (TCGA) is in /media/volume/bulkrnadata/tcgadata

No need to look at osdr_outdated, it won't have any useful contents