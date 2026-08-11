# Taken from bulkformer_extract_feature.ipynb on the Bulkformer repo

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

def main_gene_selection(X_df, gene_list):
    """
    Aligns a gene expression matrix to a predefined gene list by adding placeholder values
    for missing genes and generating a binary mask indicating imputed entries.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix with rows representing samples and columns representing genes.
        The entries are typically log-transformed or normalized expression values.

    gene_list : list of str
        A predefined list of gene identifiers (Ensembl Gene IDs) to be retained
        in the final matrix. This list defines the desired gene space for downstream analyses.

    Returns
    -------
    X_df : pandas.DataFrame
        A gene expression matrix aligned to `gene_list`, with missing genes filled with a constant
        placeholder value (-10) and columns ordered accordingly.

    to_fill_columns : list of str
        A list of genes from `gene_list` that were not present in the original `X_df`
        and were therefore added with placeholder values.

    var : pandas.DataFrame
        A DataFrame with one row per gene, containing a binary column `'mask'` indicating
        whether a gene was imputed (1) or originally present (0). This can be used for masking
        in training or evaluation of models that distinguish observed and imputed entries.

    Notes
    -----
    This function ensures that all samples share a consistent gene space, which is essential
    for tasks such as model training, cross-dataset integration, or visualization. Placeholder
    values (-10) are used to maintain matrix shape while avoiding unintended bias in downstream
    statistical analyses or machine learning models.
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    padding_df = pd.DataFrame(np.full((X_df.shape[0], len(to_fill_columns)), -10), 
                            columns=to_fill_columns, 
                            index=X_df.index)

    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var

def extract_feature(model,
                    expr_array, 
                    output_feature_type,
                    device,
                    batch_size,
                    mask_prob = 0.1,
                    output_expr = False,
                    esm2_emb = None):
    
    """
    Extracts sample-level or gene-level feature representations from input expression profiles
    using a pre-trained deep learning model.

    Parameters
    ----------
    expr_array : np.ndarray
        A NumPy array of shape [N_samples, N_genes] representing gene expression profiles
        (e.g., log-transformed TPM values).

    mask_prob : float, optional
        Masking probability used during feature extraction to simulate missing genes.
        This parameter controls the fraction of randomly masked genes during inference,
        enabling robustness to incomplete gene sets relative to the pretrained vocabulary.

    interested_gene_idx : list or np.ndarray
        Indices of your interested genes used for sample-level embedding aggregation.

    feature_type : str
        Specifies the type of feature to extract. Options:
            - 'sample_level': aggregate gene embeddings to a single sample-level vector.
            - 'gene_level': retain per-gene embeddings for downstream fusion with external embeddings (ESM2).

    aggregate_type : str
        Aggregation method used when `feature_type='sample_level'`. Options include:
            - 'max': use maximum value across selected genes.
            - 'mean': use average value.
            - 'median': use median value.
            - 'all': combine all three strategies by summation.

    device : torch.device
        Computation device (e.g., 'cuda' or 'cpu') for model inference.

    batch_size : int
        Number of samples per batch during feature extraction.

    output_expr : bool, optional
        If True, return predicted gene expression values instead of extracted embeddings.

    esm2_emb : torch.Tensor, optional
        Precomputed ESM2 embeddings for all genes, used in gene-level feature concatenation.
        Required if `feature_type='gene_level'`.

    Returns
    -------
    result_emb : torch.Tensor
        The extracted feature representations:
            - [N_samples, D] for sample-level features.
            - [N_samples, N_genes, D_concat] for gene-level features.

    or (if `output_expr=True`)
    expr_predictions : np.ndarray
        Model-predicted expression profiles.
    """

    # Convert NumPy array to PyTorch tensor and move to target device
    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)

    # Create dataset and dataloader for batched inference
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False)

    # Switch model to evaluation mode
    model.eval()

    all_emb_list = []
    all_pred_expr_list = []

    # Use inference mode with automatic mixed precision on GPU
    with torch.inference_mode():

        # ----------------------------------------------------------
        # 1. SAMPLE-LEVEL FEATURE EXTRACTION
        # ----------------------------------------------------------
        if output_feature_type == 'sample_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)

                # Obtain per-gene embeddings from the model
                gene_emb = model(X, mask_prob=mask_prob, output_expr=output_expr)
                final_emb = torch.mean(gene_emb, dim=1)

                all_emb_list.append(final_emb.detach().cpu())

            # Stack all embeddings across batches
            result_emb = torch.cat(all_emb_list, dim=0)

        # ----------------------------------------------------------
        # 2. GENE-LEVEL FEATURE EXTRACTION (OPTIONALLY CONCAT WITH ESM2)
        # ----------------------------------------------------------
        elif output_feature_type == 'gene_level':
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)

                gene_emb = model(X, mask_prob=mask_prob, output_expr=output_expr)
                gene_emb = gene_emb.detach().cpu().numpy()
                all_emb_list.append(gene_emb)

                del gene_emb
                torch.cuda.empty_cache()

            # Stack to [N_samples, N_genes, D]
            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb, device='cpu', dtype=torch.float32)

            # Expand ESM2 embeddings to align with batch dimension: [N_samples, N_genes, D_esm2]
            esm2_emb_expanded = esm2_emb.unsqueeze(0).expand(all_emb_tensor.shape[0], -1, -1)

            # Concatenate model embeddings with ESM2 embeddings
            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)

        # ----------------------------------------------------------
        # 3. EXPRESSION-LEVEL OUTPUT (MODEL PREDICTS EXPRESSION)
        # ----------------------------------------------------------
        elif output_feature_type == 'expr_level':
            output_expr = True
            for (X,) in tqdm(myloader, total=len(myloader)):
                X = X.to(device)

                pred_expr = model(X, mask_prob=mask_prob, output_expr=output_expr)
                all_pred_expr_list.append(pred_expr.detach().cpu().numpy())

    # ----------------------------------------------------------
    # Final return logic
    # ----------------------------------------------------------
    if output_expr:
        return np.vstack(all_pred_expr_list)
    else:
        return result_emb