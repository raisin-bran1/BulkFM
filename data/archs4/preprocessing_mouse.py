import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Preprocess mouse data.")
parser.add_argument("--datadir", type=str, default="/global/scratch/users/kmeng/", help="Data directory")
parser.add_argument("--outputdir", type=str, default="/global/scratch/users/kmeng/", help="Output directory")
parser.add_argument("--input", type=str, default="input_chunk_0.parquet", help="Expression matrix")
parser.add_argument("--sc", type=str, default="singlecellprob_mouse.csv", help="Single cell probabilities")
parser.add_argument("--genes", type=str, default="human_mouse_orthologs.csv", help="Mouse gene vocabulary")
parser.add_argument("--lengths", type=str, default="mouse_exon_lengths_df.csv", help="Mouse gene exon lengths")
parser.add_argument("--chunk_id", type=int, required=True, help="Index of the chunk to process")
args = parser.parse_args()

# Read input parquet file
df = pd.read_parquet(args.datadir + args.input)
df = df.reset_index()
df = df.rename(columns={df.columns[0]: 'genes'})
df['genes'] = df['genes'].str.upper()

# Gene aggregation (sums rows with duplicate genes)
def aggregate_duplicate_genes(df):
    return df.groupby(['genes'], as_index = False).sum()
agg_dup_df = aggregate_duplicate_genes(df)

# Quality control function: remove samples with low nonzero entries
def filter_nonzero(df, nonzero_entries=14000):
    data_matrix = df.iloc[:, 1:].values
    sample_counts = np.count_nonzero(data_matrix, axis=0)
    passing_samples = df.columns[1:][sample_counts >= nonzero_entries]
    keep_cols = [df.columns[0]] + passing_samples.tolist()
    return df[keep_cols]
filt_nonzero_df = filter_nonzero(agg_dup_df)

# Single cell probability filter, threshold of 0.5 for now
scprob_df = pd.read_csv(args.datadir + args.sc)
scprob_dict = scprob_df.set_index('geo_accession')['singlecellprobability'].to_dict()
scprob_dict['genes'] = 0
scprob_threshold = 0.5
probabilities = np.array([scprob_dict.get(col, 1.0) for col in filt_nonzero_df.columns])
mask = probabilities < scprob_threshold
sc_df = filt_nonzero_df.loc[:, mask].copy()
print(sc_df.shape)
print(sc_df.head(5))

# -------------------------------------
# TPM Normalization with exon lengths

# load in precomputed exon length csv files
mouse_exon_lengths = pd.read_csv(args.datadir + args.lengths)
mouse_exon_lengths['gene name'] = mouse_exon_lengths['gene name'].str.upper()

# TPM_i = 10^6 * (C_i / L_i) / sum_j (C_j, L_j)
# C_i = raw count of gene i (from human gene data)
# L_i = length of gene i (from GENCODE)
# N = num genes (num rows human gene data)

def convert_to_TPM(counts_df, length_df):
    L = length_df.reindex(counts_df.index)["length"]
    L = L.fillna(1000).replace(0, 1000)
    
    # Calculate RPK
    rpk = counts_df.astype('float32').div(L / 1000, axis=0)
    
    # Calculate TPM
    tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
    return tpm

# convert_to_TPM assumes that the the df indices are gene names

sc_df = sc_df.set_index('genes')
mouse_exon_lengths2 = mouse_exon_lengths.set_index('gene name')

tpm_df = convert_to_TPM(sc_df, mouse_exon_lengths2)
print(tpm_df.shape)
print(tpm_df.head(5))

# -------------------------------------

# Read ortholog genes list 
ORTHOLOG_PATH = args.datadir + args.genes
orthologs = pd.read_csv(ORTHOLOG_PATH) 
orthologs = orthologs[orthologs['Mouse homology type'] == 'ortholog_one2one'] # only one to one orthologs
orthologs['Mouse gene name'] = orthologs['Mouse gene name'].str.upper()
shared_gene_pool = orthologs['Mouse gene stable ID'].squeeze()
print(shared_gene_pool.shape)
print(shared_gene_pool.head(5))

# Rename genes to enmusg ids
name_to_id = dict(zip(orthologs['Mouse gene name'], orthologs['Mouse gene stable ID']))
tpm_df['genes'] = tpm_df.index.map(name_to_id, na_action='ignore')

# Filter ARCHS4 to only have ortholog genes and sort
mask = tpm_df['genes'].isin(shared_gene_pool)
print(f"Number of kept genes: {sum(mask)}")
processed_df = tpm_df[mask].set_index('genes')
processed_df = processed_df.sort_index()
print(processed_df.shape)
print(processed_df.head(5))

# Extract gene vocabulary list for the model
def extract_genes(df, filename = args.outputdir + 'gene_vocabulary.csv'):
    pd.Series(df.index).to_csv(filename, index=False, header=['genes'])
extract_genes(processed_df)

# Final touches
logtpm_df = np.log1p(processed_df)
compressed_df = logtpm_df.round(4).astype('float32')
final_df = compressed_df.transpose()
final_df.index.name = 'sample_id'
print(final_df.shape)
print(final_df.head(5))

# Compile processed data as parquet file for training
final_df.to_parquet(args.outputdir + f'processed_mouse_{args.chunk_id}.parquet', engine='pyarrow', compression='snappy')
