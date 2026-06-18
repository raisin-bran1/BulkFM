Overview:
A foundation model, binformer.py, was pretrained on ~20k bulk RNA mouse samples. It is a transformer with performer attention blocks, and it learns a masked language modeling based objective of predicting the quantile bins of gene expression values. The weights, config, and gene vocabulary are in the models folder.

Currently looking at comparing embeddings from the pretrained model and PCA for organ classification.

Behavior:
You may use but not edit existing python files, it is preferred you write your code in new python files so the repository is nicely organized.

For any bug reports, before proposing changes, first think deeply and simply share your thoughts on why the bug may have occured.

Make your scripts in the gemini_scripts folder.