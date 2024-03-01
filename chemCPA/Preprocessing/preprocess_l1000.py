import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Preprocess Sciplex3 drug perturbation data')
# Define the command-line arguments
parser.add_argument('--load_path', type=str, help='Path to downloaded sciplex data')
parser.add_argument('--save_path', type=str, help='Save path')
# Parse the arguments from the command line
args = parser.parse_args()
# Access the parameter values
load_path = args.load_path
save_path = args.save_path
if save_path=='':
    save_path=load_path



import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import post_preprocess as ppp
import re
    
"""
This loads and preprocesses the Lincs L1000 data

https://lincsproject.org/LINCS/tools/workflows/find-the-best-place-to-obtain-the-lincs-l1000-data
https://www.cell.com/cell/pdf/S0092-8674(17)31309-0.pdf

I downloaded the anndata from chemCPA, from https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar, 
"""

def remove_non_alphanumeric(input_string):
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)

load_path='/home/manu/chemCPA/chemCPA/anndatas'
adata=sc.read(f'{load_path}/lincs_full.h5ad')

# It's not single cell expression data, so I will not normalize+log1p it

#sc.pp.normalize_total(adata, target_sum=1e4)
#sc.pp.log1p(adata)

adata.obs['cell_line']=adata.obs['cell_id']
adata.obs['treatment']=adata.obs['pert_id']
adata.obs['treatment']=[a if a!='DMSO' else 'control' for a in adata.obs['treatment']]
adata.obs["treatment"] = adata.obs["treatment"].apply(remove_non_alphanumeric)

adata.obs['treatment_dose_uM']=adata.obs['pert_dose'].astype('str')
adata.obs['treatment_dose_uM']=[a if a!='-666.0' else '0' for a in adata.obs['treatment_dose_uM']]
adata.obs['treatment_time_h']=adata.obs['pert_time']
adata.obs['treatment_type']='drug_perturbation'
adata.obs['dataset']='L1000'

adata.obs=adata.obs[['cell_line', 'treatment', 'treatment_dose_uM', 'treatment_time_h', 'treatment_type', 'dataset']]
adata.obs


# Map gene symbols to Ensemble IDs
import os
current_directory = os.getcwd()
data_path='/'+os.path.join(*current_directory.split('/')[:-1])+'/non_anndata_data'

Gene_Dict=np.load(f'{data_path}/Gene_Dict.npy', allow_pickle=True).item()
adata.var['Gene_symbol_reduced']=[remove_non_alphanumeric(a) for a in adata.var.index]
adata.var['Ensemble_ID']=[Gene_Dict[a] for a in adata.var['Gene_symbol_reduced']]
adata.var.index=adata.var['Ensemble_ID']

for c in adata.var.columns:
    del adata.var[c]

# We don't have an embedding for all genes, so we need to filter out these knockouts where we dont have the embedding:
adata=ppp.map_and_filter_based_on_embedding(adata)

# This calls DEGs and filters out treatments with to little cells
adata=ppp.postprocess_anndata(adata, n_top_genes=2000)


# It writes/reads faster when not using compression
adata.write(f'{save_path}/L1000_pp.h5ad', compression='gzip')

print('Done')