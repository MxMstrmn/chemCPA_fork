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
import re
import post_preprocess as ppp
    
"""
Loads and preporcesses the Sciplex3 et al, 2019 data, https://www.science.org/doi/10.1126/science.aax6234

I downloaded these 5 junks from chemCPA, from https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar, and concatenated with this

# Get raw data
# These raw sciplex junks I got from chemCPA https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar

#adatas = []
#for i in range(5):
#    adatas.append(sc.read(f'{Path}/sciplex_raw_chunk_{i}.h5ad'))
#adata = adatas[0].concatenate(adatas[1:])
#adata.write(f'{Path}/sciplex.h5ad', compression='gzip')


Alternatively you can also downlaod the data using pertpy, but syntax might change
#import pertpy as pt
#adata = pt.data.srivatsan_2020_sciplex3()
"""

def remove_non_alphanumeric(input_string):
    return re.sub(r"[^a-zA-Z0-9]", "", input_string)

adata=sc.read(f'{load_path}/sciplex.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.subsample(adata, fraction=0.1)


adata.obs['cell_line']=adata.obs['cell_type']
adata.obs['treatment']=[a.split(' (')[0] for a in adata.obs['product_name']]
adata.obs['treatment']=[a if a!='Vehicle' else 'control' for a in adata.obs['treatment']]
adata.obs["treatment"] = adata.obs["treatment"].apply(remove_non_alphanumeric)
adata.obs['treatment_dose_uM']=adata.obs['dose'].astype('str')
# I looked this up in the publication
adata.obs['treatment_time_h']=24.0
adata.obs['treatment_type']='drug_perturbation'
adata.obs['dataset']='Sciplex'

# We only need these columns
adata.obs=adata.obs[['cell_line', 'treatment', 'treatment_dose_uM', 'treatment_time_h', 'treatment_type', 'dataset']]
adata.obs



# They have some Ensemble Id's with .?_PAR_Y in the end, where ? is the release number of the assembly
# These are X/Y-chromosomal genes that behave like any other non-sex genes.
# "From release 44 onwards, the chromosome Y PAR annotation has their own identifiers."
# I'll just sum them up, from what I can tell they are no longer treated in a different way
# (as they are de facto autosomal)

# For all the other genes I'll remove the version number

adata.var.index=adata.var['id']

adata.var['cut']=[a.split('.')[0] for a in adata.var.id]

av=adata.var['cut'].value_counts()
duplicated=list(av[av>1].index)
adata.var[adata.var['cut'].isin(duplicated)]

# This takes 2min
inds_to_remove=[]
for d in duplicated:
    inds=np.where(adata.var['cut']==d)[0]
    v=adata.X[:,inds]
    v=v[:,0]+v[:,1]
    adata.X[:,inds[0]]=v
    inds_to_remove.append(inds[1])
    
exclude=[list(adata.var.index)[a] for a in inds_to_remove]
adata = adata[:, ~adata.var_names.isin(exclude)].copy()
# I'll ignore the ensemble version number
adata.var.index=[a.split('.')[0] for a in adata.var.index]
for c in adata.var.columns:
    del adata.var[c]
    

# We don't have an embedding for all genes, so we need to filter out these knockouts where we dont have the embedding:
adata=ppp.map_and_filter_based_on_embedding(adata)

# This calls DEGs and filters out treatments with to little cells
adata=ppp.postprocess_anndata(adata, n_top_genes=2000)

# It writes/reads faster when not using compression
adata.write(f'{save_path}/sciplex_pp_sub.h5ad', compression='gzip')

print('Done')

