# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import warnings
import torch

import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)
import scanpy as sc
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from .helper import graph_from_smiles
from typing import Union


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


indx = lambda a, i: a[i] if a is not None else None


class Dataset:
    def __init__(
        self,
        fname,
        perturbation_key=None,
        dose_key=None,
        covariate_keys=None,
        smiles_key=None,
        split_key="split",
        mol_featurizer="canonical",
    ):

        data = sc.read(fname)
        self.genes = torch.Tensor(data.X.A)
        self.var_names = data.var_names

        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys
        self.smiles_key = smiles_key
        mol_featurizers = ["canonical", "AttentiveFP", "Pretrain"]
        self.mol_featurizer = mol_featurizer

        if perturbation_key is not None:
            if dose_key is None:
                raise ValueError(
                    f"A 'dose_key' is required when provided a 'perturbation_key'({perturbation_key})."
                )
            self.pert_categories = np.array(data.obs["cov_drug_dose_name"].values)
            self.de_genes = data.uns["rank_genes_groups_cov"]

            self.drugs_names = np.array(data.obs[perturbation_key].values)
            self.dose_names = np.array(data.obs[dose_key].values)

            # get unique drugs
            drugs_names_unique = set()
            for d in self.drugs_names:
                [drugs_names_unique.add(i) for i in d.split("+")]
            self.drugs_names_unique = np.array(list(drugs_names_unique))

            # save encoder for a comparison with Mo's model
            # later we need to remove this part
            encoder_drug = OneHotEncoder(sparse=False)
            encoder_drug.fit(self.drugs_names_unique.reshape(-1, 1))

            # Store as attribute for molecular featurisation
            self.encoder_drug = encoder_drug

            self.atomic_drugs_dict = dict(
                zip(
                    self.drugs_names_unique,
                    encoder_drug.transform(self.drugs_names_unique.reshape(-1, 1)),
                )
            )

            # get drug combinations
            drugs = []
            for i, comb in enumerate(self.drugs_names):
                drugs_combos = encoder_drug.transform(
                    np.array(comb.split("+")).reshape(-1, 1)
                )
                dose_combos = str(data.obs[dose_key].values[i]).split("+")
                for j, d in enumerate(dose_combos):
                    if j == 0:
                        drug_ohe = float(d) * drugs_combos[j]
                    else:
                        drug_ohe += float(d) * drugs_combos[j]
                drugs.append(drug_ohe)
            self.drugs = torch.Tensor(drugs)

            atomic_ohe = encoder_drug.transform(self.drugs_names_unique.reshape(-1, 1))

            self.drug_dict = {}
            for idrug, drug in enumerate(self.drugs_names_unique):
                i = np.where(atomic_ohe[idrug] == 1)[0][0]
                self.drug_dict[i] = drug
        else:
            self.pert_categories = None
            self.de_genes = None
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique = None
            self.atomic_drugs_dict = None
            self.drug_dict = None
            self.drugs = None

        if (
            smiles_key is not None
        ):  # TODO: Generalise to more than just SMILES, mol_rep_key
            if mol_featurizer not in mol_featurizers:
                raise ValueError(f"mol_featurizer must be one of {mol_featurizers}")
            graph_tuple = graph_from_smiles(
                data.obs[[self.perturbation_key, self.smiles_key]],
                self.perturbation_key,
                self.smiles_key,
                self.encoder_drug,
                mol_featuriser=self.mol_featurizer,
            )
            self.batched_graph_collection = graph_tuple[0]
            self.idx_wo_smiles = graph_tuple[1]
            self.graph_feats_shape = graph_tuple[2]
        else:
            self.batched_graph_collection = None
            self.idx_wo_smiles = None
            self.graph_feats_shape = None

        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self.atomic_сovars_dict = {}
            self.covariates = []
            for cov in covariate_keys:
                self.covariate_names[cov] = np.array(data.obs[cov].values)
                self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])

                names = self.covariate_names_unique[cov]
                encoder_cov = OneHotEncoder(sparse=False)
                encoder_cov.fit(names.reshape(-1, 1))

                self.atomic_сovars_dict[cov] = dict(
                    zip(list(names), encoder_cov.transform(names.reshape(-1, 1)))
                )

                names = self.covariate_names[cov]
                self.covariates.append(
                    torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float()
                )
        else:
            self.covariate_names = None
            self.covariate_names_unique = None
            self.atomic_сovars_dict = None
            self.covariates = None

        self.ctrl = data.obs["control"].values

        if perturbation_key is not None:
            self.ctrl_name = list(
                np.unique(data[data.obs["control"] == 1].obs[self.perturbation_key])
            )
        else:
            self.ctrl_name = None

        if self.covariates is not None:
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.num_covariates = [0]
        self.num_genes = self.genes.shape[1]
        self.num_drugs = len(self.drugs_names_unique) if self.drugs is not None else 0

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

    def __getitem__(self, i):
        return (
            self.genes[i],
            indx(self.drugs, i),
            *[indx(cov, i) for cov in self.covariates],
        )

    def __len__(self):
        return len(self.genes)


class SubDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset, indices):
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.smiles_key = dataset.smiles_key

        self.batched_graph_collection = dataset.batched_graph_collection
        self.idx_wo_smiles = dataset.idx_wo_smiles
        self.graph_feats_shape = dataset.graph_feats_shape

        self.perts_dict = dataset.atomic_drugs_dict
        self.covars_dict = dataset.atomic_сovars_dict

        self.genes = dataset.genes[indices]
        self.drugs = indx(dataset.drugs, indices)
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]

        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = indx(dataset.ctrl_name, 0)

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

    def __getitem__(self, i):
        return (
            self.genes[i],
            indx(self.drugs, i),
            *[indx(cov, i) for cov in self.covariates],
        )

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
    dataset_path: str,
    perturbation_key: Union[str, None],
    dose_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    smiles_key: Union[str, None],
    split_key: str,
    mol_featurizer: str = "canonical",
    return_dataset: bool = False,
):

    dataset = Dataset(
        dataset_path,
        perturbation_key,
        dose_key,
        covariate_keys,
        smiles_key,
        split_key,
        mol_featurizer,
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