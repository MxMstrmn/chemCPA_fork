from pathlib import Path
import scanpy as sc
import argparse

from data.datasets.download import download

output_file_dir = "../data"


def norman():  # pragma: no cover
    """112^2 gene-knockouts of K562 cell-line

    Cell lines: only K562 cells (are a poorly differentiated, erythroid-like cancer cell line).
    They deactivated a combination of 2 genes (out of the set of 112 genes → 112*112 combinations) at a time using CRISPR.
    They did this in 6 different screens,
    3 of which were unstimulated BMDC, BMDC stimulated at 3hr,
    TFs in K562 at 7 and 13 days post transduction,
    and 13 days at a higher MOI of perturbations.

    References:
        Exploring genetic interactionmanifolds constructed from richsingle-cell phenotypes
        Thomas M. Norman, Jonathan S. Weissman et. al.
        Science 2019
        DOI: https://doi.org/10.1016/j.cell.2016.11.038
    Returns:
        :class:`~anndata.AnnData` object
    """
    output_file_name = "Norman_2019.h5ad"
    output_file_path = Path(output_file_dir) / output_file_name
    if not Path(output_file_path).exists():
        download(
            url="https://figshare.com/ndownloader/files/34002548",
            output_file_name=output_file_name,
            output_path=output_file_dir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata



def replogle():  # pragma: no cover
    """~2285 genes on K562 cell-line;

    3 Cell lines:
    K562: 9,867 genes
    K562: 2,285 (essential) genes
    RPE1 (human Retinal Pigment Epithelial): 2,285 (essential) genes

    In median >100 cells per condition (>2.5 million cells in total).


    Returns:
        :class:`~anndata.AnnData` objec
    """
    output_file_name = "Reploge_2022.h5ad"
    output_file_path = Path(output_file_dir) / output_file_name
    if not Path(output_file_path).exists():
        download(
            url="https://plus.figshare.com/ndownloader/files/35773075",
            output_file_name=output_file_name,
            output_path=output_file_dir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata

def esm_gene_embeddings_3b():
    """
    Embeddings of 3b model of ESM
    """
    output_file_name = "ESM_gene_embeddings_3b.pt"
    output_file_path = Path(output_file_dir) / output_file_name
    if not Path(output_file_dir).exists():
        download(
            url="https://figshare.com/ndownloader/files/44644951",
            output_file_name=output_file_name,
            output_path=output_file_dir,
            is_zip=False,
        )
    adata = sc.read_h5ad(output_file_path)

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for chemCPA.")
    parser.add_argument("dataset", type=str, help="The name of the dataset to download.")
    args = parser.parse_args()

    if args.dataset == "norman":
        norman()
    elif args.dataset == "replogle":
        replogle()
    elif args.dataset == "esm_gene_embeddings_3b":
        esm_gene_embeddings_3b()
    else:
        print(f"Dataset {args.dataset} is not recognized.")



