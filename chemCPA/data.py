import logging
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from rdkit import Chem

warnings.simplefilter(action="ignore", category=FutureWarning)

def ranks_to_df(data, key="rank_genes_groups"):
    """Converts an `sc.tl.rank_genes_groups` result into a MultiIndex dataframe.

    You can access various levels of the MultiIndex with `df.loc[[category]]`.

    Params
    ------
    data : `AnnData`
    key : str (default: 'rank_genes_groups')
        Field in `.uns` of data where `sc.tl.rank_genes_groups` result is
        stored.
    """
    d = data.uns[key]
    dfs = []
    for k in d.keys():
        if k == "params":
            continue
        series = pd.DataFrame.from_records(d[k]).unstack()
        series.name = k
        dfs.append(series)

    return pd.concat(dfs, axis=1)
def canonicalize_smiles(smiles: Optional[str]):
    if smiles:
        return Chem.CanonSmiles(smiles)
    else:
        return None

def drug_names_to_once_canon_smiles(
        drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
):
    """
    Converts a list of drug names to a list of SMILES. The ordering is of the list is preserved.

    TODO: This function will need to be rewritten to handle datasets with combinations.
    This is not difficult to do, mainly we need to standardize how combinations of SMILES are stored in anndata.
    """
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in dataset.obs.groupby(
            [perturbation_key, smiles_key]
        ).groups.keys()
    }
    return [name_to_smiles_map.get(name) for name in drug_names]


def indx(a,i):
    if isinstance(a, torch.nn.Embedding) and isinstance(i, torch.Tensor):
        return a(i)
    elif a is not None:
        return a[i]
    else:
        return None


class Dataset:
    covariate_keys: Optional[List[str]]
    drugs_idx: list  # stores the integer indices of the drugs applied to each cell.
    dosages: list  # stores the dosages of drugs applied to each cell
    drugs_names_unique_sorted: np.ndarray  # sorted list of all drug names in the dataset

    def __init__(
        self,
        data: str,
        drug_key=None,
        dose_key=None,
        drugs_embeddings=None,
        covariate_keys=None,
        smiles_key=None,
        degs_key=None,
        pert_category="cov_geneid",
        split_key="split",

    ):
        """
        :param covariate_keys: Names of obs columns which stores covariate names (eg cell type).
        :param drug_key: Name of obs column which stores drug-perturbation name (eg drug name).
            Combinations of perturbations are separated with `+`.
            Combinations of perturbations are separated with `+`.
        :param dose_key: Name of obs column which stores perturbation dose.
            Combinations of perturbations are separated with `+`.
        :param pert_category: Name of obs column with stores covariate + perturbation + dose as one string.
            Example: cell type + drug name + drug dose. This is used during evaluation.
        """
        logging.info(f"Starting to read in data: {data}\n...")
        if isinstance(data, AnnData):
            data = data
        else:
            data = sc.read(data)
        logging.info(f"Finished data loading.")
        self.genes = torch.Tensor(data.X.A)
        self.num_genes = self.genes.shape[1]
        self.var_names = data.var_names

        self.drug_key = drug_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys
        self.smiles_key = smiles_key
        if degs_key is not None:
            self.de_genes = data.uns[degs_key]
        else: self.de_genes = None

        if drug_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'drug_key'({drug_key})."
                )

            self.drugs_names = np.array(data.obs[drug_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            # get unique drugs
            drugs_names_unique = set()
            for d in self.drugs_names:
                [drugs_names_unique.add(i) for i in d.split("+")]

            self.drugs_names_unique_sorted = list(sorted(drugs_names_unique))
            self.num_drugs = len(self.drugs_names_unique_sorted)

            #only allow for one unqiue name for control
            self.drug_ctrl_name = np.unique(data[data.obs["control"] == 1].obs[self.drug_key])[0]


            self._drugs_name_to_idx = {
                smiles: idx for idx, smiles in enumerate(self.drugs_names_unique_sorted)
            }
            self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
                list(self.drugs_names_unique_sorted), data, drug_key, smiles_key
            )


            drugs_idx = []
            for comb in self.drugs_names:
                drugs_combos = comb.split("+")
                drugs_combos_idx = [self._drugs_name_to_idx[name] for name in drugs_combos]
                drugs_combos_idx = torch.tensor(drugs_combos_idx, dtype=torch.int32)
                drugs_idx.append(drugs_combos_idx)
            self.drugs_idx = np.array(drugs_idx, dtype=object)

            dosages = []
            for comb in self.dose_names:
                dosages_combos = comb.split("+")
                dosages_combos = [float(i) for i in dosages_combos]
                dosages_combos = torch.tensor(dosages_combos, dtype=torch.float32)
                dosages.append(dosages_combos)
            self.dosages = np.array(dosages, dtype=object)

            if isinstance(drugs_embeddings, torch.nn.Embedding):
                self.drugs_embeddings = drugs_embeddings
            elif isinstance(drugs_embeddings, str):
                drugs_embeddings_df = pd.read_parquet(drugs_embeddings)
                drugs_embeddings = torch.tensor(drugs_embeddings_df.loc[self.canon_smiles_unique_sorted].values,
                                                dtype=torch.float32, device=self.device)
                self.drugs_embeddings = torch.nn.Embedding.from_pretrained(drugs_embeddings, freeze=True)
            else:
                # maybe provided with None, create random embeddings
                self.drugs_embeddings = torch.nn.Embedding(self.num_drugs, 256, _freeze=True)

        else:
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.num_drugs = 0
            self.drugs_idx = None
            self.dosages = None
            self.drug_ctrl_name = None
            self.drugs_embeddings = None


        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self._covariates_names_to_idx = {}
            self.covariates_idx = []
            for cov in covariate_keys:
                # assume each cell only falls into one covariate category
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])
                self._covariates_names_to_idx[cov] = {
                    cov_name: idx for idx, cov_name in enumerate(self.covariate_names_unique[cov])
                }
                covariate_idx = [self._covariates_names_to_idx[cov][cov_name] for cov_name in self.covariate_names[cov]]
                self.covariates_idx.append(torch.tensor(covariate_idx, dtype=torch.int32))
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.covariates_idx = None
            self.num_covariates = [0]

        if pert_category is not None:
            self.pert_categories = np.array(data.obs[pert_category].values)
        else: self.pert_categories = None


        self.ctrl = data.obs["control"].values
        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs["control"] == 1)[0].tolist(),
            "treated": np.where(data.obs["control"] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }


    def subset(self, split, condition="all"):
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubDataset(self, idx)

    def drug_name_to_idx(self, drug_name: str):
        """
        For the given drug, return it's index. The index will be persistent for each dataset (since the list is sorted).
        Raises ValueError if the drug doesn't exist in the dataset.
        """
        return self._drugs_name_to_idx[drug_name]


    def __getitem__(self, i):
        return (
            self.genes[i],
            indx(self.drugs_idx, i),
            indx(self.dosages, i),
            indx(self.drugs_embeddings, indx(self.drugs_idx, i)),
            *[indx(cov, i) for cov in self.covariates_idx],
        )


    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset: Dataset, indices):
        self.drug_key = dataset.drug_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.smiles_key = dataset.smiles_key
        self.drugs_embeddings = dataset.drugs_embeddings

        self.genes = dataset.genes[indices]
        self.drugs_idx = indx(dataset.drugs_idx, indices)
        self.dosages = indx(dataset.dosages, indices)
        self.covariates_idx = [indx(cov, indices) for cov in dataset.covariates_idx] if dataset.covariates_idx is not None else None

        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}
        assert (
            "cell_type" in self.covariate_keys
        ), "`cell_type` must be provided as a covariate"
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.drug_ctrl_name = dataset.drug_ctrl_name

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs


    def __getitem__(self, i):
        return (
            self.genes[i],
            indx(self.drugs_idx, i),
            indx(self.dosages, i),
            indx(self.drugs_embeddings, indx(self.drugs_idx, i)),
            *[indx(cov, i) for cov in self.covariates_idx],
        )

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
    dataset_path: str,
    drug_key: Union[str, None],
    dose_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    smiles_key: Union[str, None],
    degs_key=None,
    pert_category: str = "cov_geneid",
    split_key: str = "split",
    return_dataset: bool = False,
    drugs_embeddings = None,
):
    dataset = Dataset(
        dataset_path,
        drug_key,
        dose_key,
        drugs_embeddings,
        covariate_keys,
        smiles_key,
        degs_key,
        pert_category,
        split_key,
    )

    splits = {
        "training": dataset.subset("train", "all"),
        "training_control": dataset.subset("train", "control"),
        "training_treated": dataset.subset("train", "treated"),
        "test": dataset.subset("test", "all"),
        "test_control": dataset.subset("test", "control"),
        "test_treated": dataset.subset("test", "treated"),
        "ood": dataset.subset("ood", "all"),
    }

    if return_dataset:
        return splits, dataset
    else:
        return splits
