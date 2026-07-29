The core goal of this project is to benchmark various architecture choices for bulk RNA foundation models. Currently, there is a limited number of such models in the literature and no clear consensus for what the best approaches are. Bulkformer uses continuous expression embeddings, linear attention, and intermixes GCN layers. On the other hand, BulkRNABert uses binned expression embeddings and full attention blocks. By taking different architectures with the same parameter counts trained on several matching dataset sizes, and evaluating them on the same downstream tasks (e.g. TCGA classification & regression tasks), we will reveal what advantages these different approaches have. This will enable more informed architectural decisions for future development of bulk RNA foundation models.

ARCS4 human will be used for pretraining, and TCGA will be used for evaluation. The datasets are independent of each other.

First experiment: Continuous (txfm style) vs binned (binformer style) + Masking ratio 15% vs 45% vs 75% vs U(15%, 75%)

Second experiment: Encoder bottleneck is a vector for each gene (default) vs a single cls vector (maybe need to tweak masking ratio). Also for continuous embeddings check loss function MSE (default) vs Poisson (including txfm activation)

Other architecture choices:
Gene embeddings: initialized randomly vs frozen esm (learned linear projection)
How to combine gene & expression embeddings?
Attention: how much worse is linear attention?

Possible evaluation metrics:
Gene embedding space analysis
TCGA: cancer type & subtype classification
Also normal vs tumor for specific cancer genes?
Imputation
Other potential options: cancer survival regression, perturbation prediction, cell type deconvolution

Other notes:
For evaluation, always separate train & eval sets by batch or use leave one out cross validation
Pretraining & benchmarking datasets should ideally have been processed using the same sequencing methods (kallisto, star, etc.)
Archs4 contains tcga data?
After benchmarking, possibly scale up the best performing model

Training & validation data (ARCHS4) is in /media/volume/bulkrnadata/humandata
Eval data (TCGA) is in /media/volume/bulkrnadata/tcgadata
OSDR data is in /media/volume/bulkrnamouse/osdrdata
ARCHS4 mouse is yet to be added and processed in /media/volume/bulkrnamouse
