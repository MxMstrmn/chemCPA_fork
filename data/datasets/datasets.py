from pathlib import Path
import scanpy as sc
import argparse
import pandas as pd

from .download import download


def norman(output_file_path="../data/Norman_2019.h5ad", return_data=False):  # pragma: no cover
    output_file_path = Path(output_file_path)
    if not output_file_path.exists():
        download(
            url="https://figshare.com/ndownloader/files/34002548",
            output_file_name=output_file_path.name,
            output_path=str(output_file_path.parent),
            is_zip=False,
        )
    if return_data:
        adata = sc.read_h5ad(output_file_path)
        return adata


def replogle(output_file_path="../data/Reploge_2022.h5ad", return_data=False):  # pragma: no cover
    output_file_path = Path(output_file_path)
    if not output_file_path.exists():
        download(
            url="https://plus.figshare.com/ndownloader/files/35773075",
            output_file_name=output_file_path.name,
            output_path=str(output_file_path.parent),
            is_zip=False,
        )
    if return_data:
        adata = sc.read_h5ad(output_file_path)
        return adata

def esm_gene_embeddings_3b(output_file_path="../data/ESM_gene_embeddings_3b.pt", return_data=False):
    output_file_path = Path(output_file_path)
    if not output_file_path.exists():
        download(
            url="https://figshare.com/ndownloader/files/44644951",
            output_file_name=output_file_path.name,
            output_path=str(output_file_path.parent),
            is_zip=False,
        )
    if return_data:
        adata = sc.read_h5ad(output_file_path)
        return adata


def rdkit2D_embedding_lincs_trapnell(output_file_path="../data/rdkit2D_embedding_lincs_trapnell.parquet", return_data=False):
    output_file_path = Path(output_file_path)
    if not output_file_path.exists():
        download(
            url="https://figshare.com/ndownloader/files/46373320",
            output_file_name=output_file_path.name,
            output_path=str(output_file_path.parent),
            is_zip=False,
        )
    if return_data:
        data = pd.read_parquet(output_file_path)
        return data

def sciplex_complete_middle_subset_lincs_genes(output_file_path="../data/sciplex_complete_middle_subset_lincs_genes.h5ad", return_data=False):
    output_file_path = Path(output_file_path)
    if not output_file_path.exists():
        download(
            url="https://figshare.com/ndownloader/files/46373176",
            output_file_name=output_file_path.name,
            output_path=str(output_file_path.parent),
            is_zip=False,
        )
    if return_data:
        adata = sc.read_h5ad(output_file_path)
        return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for chemCPA.")
    parser.add_argument("dataset", type=str, help="The name of the dataset to download.")
    parser.add_argument("--output_path", type=str, default="../data", help="The path to save the dataset.")
    parser.add_argument("--return_data", action="store_true", help="Option to return the dataset.")
    args = parser.parse_args()

    if args.dataset == "norman":
        norman(args.output_path, args.return_data)
    elif args.dataset == "replogle":
        replogle(args.output_path, args.return_data)
    elif args.dataset == "esm_gene_embeddings_3b":
        esm_gene_embeddings_3b(args.output_path, args.return_data)
    elif args.dataset == "rdkit2D_embedding_lincs_trapnell":
        download_rdkit2D_embedding_lincs_trapnell(args.output_path, args.return_data)
    elif args.dataset == "sciplex_complete_middle_subset_lincs_genes":
        download_sciplex_complete_middle_subset_lincs_genes(args.output_path, args.return_data)
    else:
        print(f"Dataset {args.dataset} is not recognized.")
