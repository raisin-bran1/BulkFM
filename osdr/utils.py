import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

def main_gene_selection(X_df, gene_list):
    """
    Aligns a gene expression matrix to a predefined gene list.
    Missing genes are filled with -10 (the mask token value).
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    padding_df = pd.DataFrame(np.full((X_df.shape[0], len(to_fill_columns)), -10), 
                            columns=to_fill_columns, 
                            index=X_df.index)

    X_df = pd.concat([X_df, padding_df], axis=1)
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var

def extract_feature(model,
                    expr_array, 
                    output_feature_type,
                    device,
                    batch_size = 1,
                    output_expr = False):
    
    """
    Extracts embeddings from input expression profiles using a pre-trained Binformer model.

    Parameters
    ----------
    model : Binformer
        The pre-trained Binformer model.
    expr_array : np.ndarray
        [N_samples, N_genes] expression matrix.
    output_feature_type : str
        'sample_level' (mean across genes) or 'gene_level' (all gene embeddings).
    device : torch.device
        'cuda' or 'cpu'.
    batch_size : int
    output_expr : bool
        If True, returns bin logits instead of embeddings.

    Returns
    -------
    result_emb : torch.Tensor or np.ndarray
    """

    expr_tensor = torch.tensor(expr_array, dtype=torch.float32)
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(mydataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_emb_list = []
    all_pred_expr_list = []

    with torch.no_grad():
        for (X,) in tqdm(myloader, total=len(myloader), desc="Extracting features"):
            X = X.to(device)

            # Binformer forward call
            # We use output_hidden=True to get embeddings unless output_expr is True
            out = model(X, output_hidden=not output_expr)
            
            if output_expr:
                all_pred_expr_list.append(out.detach().cpu().numpy())
            else:
                if output_feature_type == 'sample_level':
                    # Average across all genes
                    sample_emb = torch.mean(out, dim=1)
                    all_emb_list.append(sample_emb.detach().cpu())
                elif output_feature_type == 'gene_level':
                    all_emb_list.append(out.detach().cpu())

    if output_expr:
        return np.vstack(all_pred_expr_list)
    
    result_emb = torch.cat(all_emb_list, dim=0)
    
    return result_emb
