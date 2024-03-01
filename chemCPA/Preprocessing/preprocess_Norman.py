import argparse
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Preprocess Norman gene knockout data')
# Define the command-line arguments
parser.add_argument('--load_path', type=str, help='Path to downloaded Norman data')
parser.add_argument('--save_path', type=str, help='Save path')
# Parse the arguments from the command line
args = parser.parse_args()
# Access the parameter values
load_path = args.load_path
save_path = args.save_path


import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import post_preprocess as ppp
    
"""
Loads and preporcesses the Norman et al, 2019 data. I downloaded the data from: https://figshare.com/articles/dataset/Normal_et_al_2019/19134071
They (Yuge Ji) have the raw data and the preprocessed one. I'll take the preprocessed one

Alternatively you can also downlaod the data using pertpy
"""
#import pertpy as pt
#adata = pt.data.norman_2019()

adata=sc.read_h5ad(f"{load_path}/Norman_2019.h5ad")

adata.obs['cell_line']='K562'
adata.obs['treatment']=adata.obs['perturbation_name'].astype('str')
adata.obs['treatment_dose_uM']=np.NaN
# Suppl. methods say 7 posttransduction the cells where measured
adata.obs['treatment_time_h']=7*24.0
adata.obs['treatment_type']='gene_knockout'
adata.obs['dataset']='Norman'

# We only need these columns
adata.obs=adata.obs[['cell_line', 'treatment', 'treatment_dose_uM', 'treatment_time_h', 'treatment_type', 'dataset']]

# We only need the Ensemble-names
adata.var.index=adata.var['index']
for c in adata.var.columns:
    del adata.var[c]
del adata.uns
del adata.obsm
del adata.varm
del adata.layers
del adata.obsp


# We don't have an embedding for all genes, so we need to filter out these knockouts where we dont have the embedding:
adata=ppp.map_and_filter_based_on_embedding(adata)


# This calls DEGs and filters out treatments with to little cells
adata=ppp.postprocess_anndata(adata, n_top_genes=2000)
# It writes/reads faster when not using compression
adata.write(f'{save_path}/Norman_pp.h5ad', compression='gzip')
print('Done')