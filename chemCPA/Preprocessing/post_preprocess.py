import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import warnings
# Filter out PerformanceWarning related to a highly fragmented DataFrame
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="anndata.base")


import re
def remove_non_alphanumeric(input_string):
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)


import os
current_directory = os.getcwd()
data_path='/'+os.path.join(*current_directory.split('/')[:-1])+'/non_anndata_data'



def map_and_filter_based_on_embedding(adata):    
    Gene_names_dict= np.load(f'{data_path}/Gene_names_dict.npy', allow_pickle=True).item()
    Gene_names_dict_keys=list(Gene_names_dict.keys())
    
    
    treatments=sorted(set(adata.obs['treatment']))
    Specific_dict={}

    for i in range(len(treatments)):
        new_entry=''
        treatment=treatments[i]
        for t in treatment.split('+'):
            if t in Gene_names_dict_keys:
                t=Gene_names_dict[t]
            new_entry=new_entry+'+'+remove_non_alphanumeric(t)
        Specific_dict[treatment]=new_entry[1:]
    adata.obs['treatment']=adata.obs['treatment'].map(Specific_dict)
    treatments=sorted(set(adata.obs['treatment']))
    

    embeddings=torch.load(f'{data_path}/rdkit2D_embedding.pt')
    embedding_drugs=list(embeddings.keys())
    
    embeddings=torch.load(f'{data_path}/gene_name_mapped_embedding.pt')
    embedding_genes=list(embeddings.keys())
    
    
    embedding_features=embedding_drugs+embedding_genes
    
    
    treatments_kept=[]
    treatments_removed=[]
    for ts in treatments:
        got_all=True
        for t in ts.split('+'):
            if not t in embedding_features:
                got_all=False
        if got_all:
            treatments_kept.append(ts)
        else:
            treatments_removed.append(ts)
    print(f'Treatments kept: {len(treatments_kept)}')
    print(f'Treatments removed: {len(treatments_removed)}')
    adata=adata[adata.obs['treatment'].isin(treatments_kept)].copy()
    return(adata)




def postprocess_anndata(adata, n_top_genes=2000):
    
    sc.pp.filter_genes(adata, min_cells=3)
    adata.uns['log1p']={'base': None}
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=True, subset=True)

    adata.obs['control']=(adata.obs['treatment']=='control').astype(int)
    adata.obs['pert_category']=adata.obs['cell_line'].astype('str')+'_'+adata.obs['treatment'].astype('str')
    
    D_SMILES=np.load(f'{data_path}/D_smiles.npy', allow_pickle=True).item()
    D_SMILES['control']='CS(=O)C'
    SMILES_keys=list(D_SMILES.keys())
    treatments=sorted(set(adata.obs['treatment']))
    # ToDo: for gene knockouts control is sometimes/often not DMSO but an empty vector = vehicle
    for t in treatments:
        if not t in SMILES_keys:
            D_SMILES[t]=''
            
    adata.obs['SMILES']=adata.obs['treatment'].map(D_SMILES)
    
    
    # Filter out conditions where there are less than e.g. 10 cells
    min_cells=10
    avc=adata.obs['pert_category'].value_counts()
    adata=adata[adata.obs['pert_category'].isin(avc[avc>min_cells].index)].copy()

    
    # Remove cell lines with too litle control cells, and once with no perturbation data
    cell_line_kept=[]
    for cell_line in sorted(set(adata.obs['cell_line'])):
        n_control=len(adata[(adata.obs['cell_line']==cell_line)&(adata.obs['treatment']=='control')])
        n_treated=len(adata[(adata.obs['cell_line']==cell_line)&(adata.obs['treatment']!='control')])
        if n_control>min_cells and n_treated>min_cells:
            cell_line_kept.append(cell_line)
    adata=adata[adata.obs['cell_line'].isin(cell_line_kept)].copy()
    rank_genes_groups_by_cov(adata, groupby='pert_category', control_group='control', covariate='cell_line')

    
    # Split into train/test/ood sets
    adata.obs['split'] = 'train'
    # take ood from top occurring perturbations to avoid losing data on low occ ones
    number_of_treatments=len(sorted(set(adata.obs['pert_category'])))
    masked_treatments=list(adata.obs.pert_category.value_counts().index[:int(0.1*len(set(adata.obs['pert_category'])))])
    masked_treatments=[a for a in masked_treatments if a.split('_')[1]!='control']

    ood_idx = adata[adata.obs['pert_category'].isin(masked_treatments)].obs.index
    adata.obs.loc[ood_idx, 'split'] = 'ood'

    # take test from a random subsampling of the rest
    fr=0.15
    test_idx = sc.pp.subsample(adata[adata.obs.split != 'ood'], fr, copy=True).obs.index
    adata.obs.loc[test_idx, 'split'] = 'test'
    return(adata)




def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added="rank_genes_groups_cov",
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        print(cov_cat)
        # name of the control group in the groupby obs column
        control_group_cov = "_".join([cov_cat, control_group])

        # subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate] == cov_cat]

        # compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
        )

        # add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns["rank_genes_groups"]["names"])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict
    