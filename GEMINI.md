The core goal of this project is to benchmark various architecture choices for bulk RNA foundation models. Currently, there is a limited number of such models in the literature and no clear consensus for what the best approaches are. Bulkformer uses continuous expression embeddings, linear attention, and intermixes GCN layers. On the other hand, BulkRNABert uses binned expression embeddings and full attention blocks. By taking different architectures with the same parameter counts trained on several matching dataset sizes, and evaluating them on the same downstream tasks (e.g. TCGA classification & regression), we will reveal what advantages these different approaches have. This will enable more informed architectural decisions for future development of bulk RNA foundation models.

ARCS4 will be used for pretraining, but a big decision that is yet to be made is whether the models will train on human data, mouse data, or both. The well-known benchmarks in the literature like TCGA are for human data. However, NASA’s OSDR mouse dataset is a good benchmark for tissue classification, since foundation models demonstrably outperform traditional methods on this benchmark due to the low sample size and high batch noise. 

Possible architecture choices:
Expression embeddings: binned vs continuous sinusoidal vs continuous rope vs scalar multiplication
Attention: full flash attention vs linear attention vs attention only on highly variable genes
Gene embeddings: initialized randomly vs frozen esm (learned linear projection) vs initialized esm but still learning (also learned linear projection)
Add GCN or not
Different MLM approaches? Like variable masking schedules, block pathway masking

Possible evaluation metrics:
Imputation
TCGA: cancer type & subtype classification (human only)
NASA OSDR tissue classification (mouse only)
Latent space analysis (clustering, silhouette score, etc.)
Other potential options: cancer survival regression, perturbation prediction, cell type deconvolution

Other notes:
For evaluation, always separate train & eval sets by batch or use leave one out cross validation
Pretraining & benchmarking datasets should ideally have been processed using the same sequencing methods (one of the NASA scientists mentioned that OSDR and ARCHS4 use different methods?)
After benchmarking, possibly scale up the best performing model
