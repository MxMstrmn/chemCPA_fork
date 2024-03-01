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
# I download from https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387

Or download it using pertpy
"""

#import pertpy as pt
#adata = pt.data.replogle_2022_rpe1()


adata=sc.read_h5ad(f"{load_path}/rpe1_raw_singlecell_01.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.obs['cell_line']='RPE1'
adata.obs['treatment']=adata.obs['gene']
adata.obs['treatment']=[a if a!='non-targeting' else 'control' for a in adata.obs['treatment']]
adata.obs['treatment_dose_uM']=np.NaN
# Suppl. methods say 7 posttransduction the cells where measured
adata.obs['treatment_time_h']=7*24.0
adata.obs['treatment_type']='gene_knockout'
adata.obs['dataset']='Replogle_RPE1'

# We only need these columns
adata.obs=adata.obs[['cell_line', 'treatment', 'treatment_dose_uM', 'treatment_time_h', 'treatment_type', 'dataset']]

# We only need the Ensemble-names
#adata.var.index=adata.var['index']
for c in adata.var.columns:
    del adata.var[c]
    
# We don't have an embedding for all genes, so we need to filter out these knockouts where we dont have the embedding:
adata=ppp.map_and_filter_based_on_embedding(adata)
    
# This calls DEGs and filters out treatments with to little cells
adata=ppp.postprocess_anndata(adata, n_top_genes=2000)
# It writes/reads faster when not using compression
adata.write(f'{save_path}/Replogle_K562_pp.h5ad', compression='gzip')
print('Done')