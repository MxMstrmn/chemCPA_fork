import logging
import warnings
from typing import List, Optional, Union
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from rdkit import Chem
import lightning as L
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

warnings.simplefilter(action="ignore", category=FutureWarning)

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
    knockouts_idx: list  #stores the integer indices of the gene knockouts applied to each cell
    dosages: list  # stores the dosages of drugs applied to each cell
    drugs_names_unique_sorted: np.ndarray  # sorted list of all drug names in the dataset
    knockouts_names_unique_sorted: np.ndarray #sorted list of all gene knockout names in the dataset

    def __init__(
        self,
        data: str,
        drug_key=None,
        dose_key=None,
        drugs_embeddings=None,
        knockout_key=None,
        knockouts_embeddings=None,
        covariate_keys=None,
        smiles_key=None,
        pert_category="cov_geneid",
        split_key="split",
        degs_key = 'rank_genes_groups_cov'
        
    ):
        """
        :param covariate_keys: Names of obs columns which stores covariate names (eg cell type).
        :param drug_key: Name of obs column which stores drug-perturbation name (eg drug name).
            Combinations of perturbations are separated with `+`.
        :param knockout_key: Name of obs column which stores knockout-perturbation name (eg gene ID).
            Combinations of perturbations are separated with `+`.
        :param dose_key: Name of obs column which stores perturbation dose.
            Combinations of perturbations are separated with `+`.
        :param pert_category: Name of obs column with stores covariate + (drug_perturbation + dose) + (knockout_perturbation) as one string.
        """
        logging.info(f"Starting to read in data: {data}\n...")
        if isinstance(data, AnnData):
            data = data
        else:
            data = sc.read(data)
        logging.info(f"Finished data loading.")
        #be flexible about the dense or sparse matrices
        try:
            self.genes = torch.Tensor(data.X.A)
        except:
            self.genes = torch.Tensor(data.X) 
        self.num_genes = self.genes.shape[1]
        self.var_names = data.var_names

        self.drug_key = drug_key
        self.knockout_key = knockout_key
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
                             dtype=torch.float32)
                self.drugs_embeddings = torch.nn.Embedding.from_pretrained(drugs_embeddings, freeze=True)
            else:
                # maybe provided with None, create random embeddings
                self.drugs_embeddings = torch.nn.Embedding(self.num_drugs, 256, _freeze=True)
            self.drug_embedding_dimension = self.drugs_embeddings.embedding_dim
        else:
            self.drugs_names = None
            self.dose_names = None
            self.drugs_names_unique_sorted = None
            self.num_drugs = 0
            self.drugs_idx = None
            self.dosages = None
            self.drug_ctrl_name = None
            self.drugs_embeddings = None
            self.drug_embedding_dimension = None

        if knockout_key is not None:
            self.knockouts_names = np.array(data.obs[knockout_key].values)
        
            # get unique gene knockouts
            knockouts_names_unique = set()
            for d in self.knockouts_names:
                [knockouts_names_unique.add(i) for i in d.split("+")]
            self.knockouts_names_unique_sorted = list(sorted(knockouts_names_unique))
            self.num_knockouts = len(self.knockouts_names_unique_sorted)
    
            #only allow for one unqiue name for control
            self.knockout_ctrl_name = np.unique(data[data.obs["control"] == 1].obs[self.knockout_key])[0]

            self._knockouts_name_to_idx = {
                gene_id: idx for idx, gene_id in enumerate(self.knockouts_names_unique_sorted)
            }

            #use -1 as place holder
            knockouts_idx = []
            for comb in self.knockouts_names:
                knockouts_combos = comb.split("+")
                knockouts_combos_idx = [self._knockouts_name_to_idx[name] for name in knockouts_combos]
                knockouts_combos_idx = torch.tensor(knockouts_combos_idx, dtype=torch.int32)
                knockouts_idx.append(knockouts_combos_idx)
            self.knockouts_idx = np.array(knockouts_idx, dtype=object)

            if isinstance(knockouts_embeddings, dict):
                self.knockouts_embeddings = {self._knockouts_name_to_idx[name]: knockouts_embeddings[name] for name in self.knockouts_names_unique_sorted}
            elif isinstance(knockouts_embeddings, str):
                self.knockouts_embeddings = torch.load(knockouts_embeddings, map_location=torch.device('cpu'))
                self.knockouts_embeddings = {self._knockouts_name_to_idx[name]: self.knockouts_embeddings[name] for name in self.knockouts_names_unique_sorted}
            else:
                # maybe provided with None, create random embeddings
                self.knockouts_embeddings = {self._knockouts_name_to_idx[name]: torch.randn(256) for name in self.knockouts_names_unique_sorted}
            self.knockouts_embeddings = torch.nn.Embedding.from_pretrained(torch.stack([self.knockouts_embeddings[i] for i in range(self.num_knockouts)]),
                                                                           freeze=True)
            self.knockout_embedding_dimension = self.knockouts_embeddings.embedding_dim
        else:
            self.knockouts_names = None
            self.knockouts_names_unique_sorted = None
            self.num_knockouts = 0
            self.knockouts_idx = None
            self.knockout_ctrl_name = None
            self.knockouts_embeddings = None
            self.knockout_embedding_dimension = None

        if isinstance(covariate_keys, list) and covariate_keys:
            if not len(covariate_keys) == len(set(covariate_keys)):
                raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
            self.covariate_names = {}
            self.covariate_names_unique = {}
            self._covariates_names_to_idx = {}
            self.covariates_idx = []
            for cov in covariate_keys:
                #assume each cell only falls into one covariate category
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
    
    def knockout_name_to_idx(self, knockout_name: str):
        """
        For the given gene knockout, return it's index. The index will be persistent for each dataset (since the list is sorted).
        Raises ValueError if the drug doesn't exist in the dataset.
        """
        return self._knockouts_name_to_idx[knockout_name]

    def __getitem__(self, i):
        return (
            self.genes[i],
            indx(self.drugs_idx, i),
            indx(self.dosages, i),
            indx(self.drugs_embeddings, indx(self.drugs_idx, i)),
            indx(self.knockouts_idx, i),
            indx(self.knockouts_embeddings, indx(self.knockouts_idx, i)),
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
        self.knockout_key = dataset.knockout_key
        self.covariate_keys = dataset.covariate_keys
        self.smiles_key = dataset.smiles_key
        self.drugs_embeddings = dataset.drugs_embeddings
        self.drug_embedding_dimension = dataset.drug_embedding_dimension
        self.knockouts_embeddings = dataset.knockouts_embeddings
        self.knockout_embedding_dimension = dataset.knockout_embedding_dimension
        
        self.genes = dataset.genes[indices]
        self.drugs_idx = indx(dataset.drugs_idx, indices)
        self.dosages = indx(dataset.dosages, indices)
        self.knockouts_idx = indx(dataset.knockouts_idx, indices)
        self.covariates_idx = [indx(cov, indices) for cov in dataset.covariates_idx] if dataset.covariates_idx is not None else None


        self.drugs_names = indx(dataset.drugs_names, indices)
        self.knockouts_names = indx(dataset.knockouts_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        self.covariate_names = {}
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.drug_ctrl_name = dataset.drug_ctrl_name
        self.knockout_ctrl_name = dataset.knockout_ctrl_name

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs
        self.num_knockouts = dataset.num_knockouts


    def __getitem__(self, i):
        return (
            self.genes[i],
            indx(self.drugs_idx, i),
            indx(self.dosages, i),
            indx(self.drugs_embeddings, indx(self.drugs_idx, i)),
            indx(self.knockouts_idx, i),
            indx(self.knockouts_embeddings, indx(self.knockouts_idx, i)),
            *[indx(cov, i) for cov in self.covariates_idx],
        )

    def __len__(self):
        return len(self.genes)


def load_dataset_splits(
    dataset_path: str,
    drug_key: Union[str, None],
    dose_key: Union[str, None],
    knockout_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    smiles_key: Union[str, None],
    pert_category: str = "cov_geneid",
    split_key: str = "split",
    degs_key='rank_genes_groups_cov',
    return_dataset: bool = False,
    drugs_embeddings = None,
    knockouts_embeddings = None
):
    dataset = Dataset(
        dataset_path,
        drug_key,
        dose_key,
        drugs_embeddings,
        knockout_key,
        knockouts_embeddings,
        covariate_keys,
        smiles_key,
        pert_category,
        split_key,
        degs_key
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


def custom_collate_train(batch):
    genes, drugs_idx, dosages, drugs_emb, knockouts_idx, knockouts_emb, cov = zip(*batch)
    genes = torch.stack(genes, 0)
    drugs_idx = None if drugs_idx[0] is None else [d for d in drugs_idx]
    dosages = None if dosages[0] is None else [d for d in dosages]
    drugs_emb = None if drugs_emb[0] is None else [d for d in drugs_emb]
    knockouts_idx = None if knockouts_idx[0] is None else [d for d in knockouts_idx]
    knockouts_emb = None if knockouts_emb[0] is None else [d for d in knockouts_emb]
    cov = None if cov[0] is None else  torch.stack(cov, 0)
    return [genes, drugs_idx, dosages, drugs_emb, knockouts_idx, knockouts_emb, cov]


def custom_collate_validate_r2(batch):
    dataset_test_treated = batch[0][0]
    dataset_test_control_genes = batch[0][1].genes
    return dataset_test_treated, dataset_test_control_genes


def custom_collate_full_evaluation(batch):
    datasets = batch[0]
    return datasets


class DataModule(L.LightningDataModule):
    def __init__(self,
                batch_size: int,
                full_eval_during_train: bool,
                dataset_path: str,
                drug_key: Union[str, None],
                dose_key: Union[str, None],
                knockout_key: Union[str, None],
                covariate_keys: Union[list, str, None],
                smiles_key: Union[str, None],
                pert_category: str = "cov_geneid",
                split_key: str = "split",
                degs_key='rank_genes_groups_cov',
                return_dataset: bool = False,
                drugs_embeddings = None,
                knockouts_embeddings = None
                ):
        super().__init__()
        self.batch_size = batch_size
        self.full_eval_during_train = full_eval_during_train
        if return_dataset:
            self.datasets, self.dataset = load_dataset_splits(dataset_path,
                                                drug_key,
                                                dose_key,
                                                knockout_key,
                                                covariate_keys,
                                                smiles_key,
                                                pert_category = pert_category,
                                                split_key = split_key,
                                                degs_key=degs_key,
                                                return_dataset = return_dataset,
                                                drugs_embeddings = drugs_embeddings,
                                                knockouts_embeddings = knockouts_embeddings)
        else:
            self.datasets = load_dataset_splits(dataset_path,
                                                drug_key,
                                                dose_key,
                                                knockout_key,
                                                covariate_keys,
                                                smiles_key,
                                                pert_category = pert_category,
                                                split_key = split_key,
                                                degs_key=degs_key,
                                                return_dataset = return_dataset,
                                                drugs_embeddings = drugs_embeddings,
                                                knockouts_embeddings = knockouts_embeddings)
        
    def train_dataloader(self):
        return DataLoader(
                        self.datasets["training"],
                        batch_size=self.batch_size,
                        collate_fn=custom_collate_train,
                        shuffle=True
                    )
    def val_dataloader(self):
        if self.full_eval_during_train:
            return DataLoader(
                            [self.datasets],
                            batch_size= 1,
                            collate_fn = custom_collate_full_evaluation,
                            shuffle=False
                            )
        else:
            return DataLoader(
                            [[self.datasets["test_treated"], self.datasets['test_control']]],
                            batch_size= 1,
                            collate_fn = custom_collate_validate_r2,
                            shuffle=False
                            )
    def test_dataloader(self):
        return DataLoader(
                        [self.datasets],
                        batch_size= 1,
                        collate_fn = custom_collate_full_evaluation,
                        shuffle=False
                        )