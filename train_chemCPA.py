from pathlib import Path

import lightning as L
import pandas as pd
import scanpy as sc
import wandb
from hydra import compose, initialize
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import OmegaConf
from plotnine import aes, geom_boxplot, ggplot, scale_y_continuous
from pytorch_lightning.loggers import WandbLogger

from chemCPA.data import DataModule
from chemCPA.model import ComPert
from chemCPA.train import evaluate_logfold_r2, evaluate_r2, evaluate_r2_sc

if __name__ == "__main__":

    wandb.login()

    ### Load config

    with initialize(version_base=None, config_path="experiments/hydra_config"):
        config = compose(config_name="defaults", overrides=[])

    print(OmegaConf.to_yaml(config.dataset.data_params))
    assert (Path(config.dataset.data_params.dataset_path)).exists(), "Config `dataset_path` is not correct!"
    assert (Path(config.dataset.data_params.drugs_embeddings)).exists(), "Config `drugs_embeddings` is not correct"

    dm = DataModule(
        batch_size=config.model["hparams"]["batch_size"],
        full_eval_during_train=config.train["full_eval_during_train"],
        num_workers=config.train["num_workers"],
        # num_workers=19,
        **config.dataset["data_params"],
    )

    data_train = dm.datasets["training"]

    model = ComPert(
        data_train.num_genes,
        data_train.num_drugs,
        data_train.num_knockouts,
        data_train.num_covariates,
        config.model.hparams,
        config.train,
        config.test,
        **config.model.additional_params,
        drug_embedding_dimension=data_train.drug_embedding_dimension,
        knockout_embedding_dimension=data_train.knockout_embedding_dimension,
    )

    project_str = f"{config.model['model_type']}_{config.dataset['dataset_type']}"
    wandb_logger = WandbLogger(project=project_str, save_dir=config.model["save_dir"])

    inference_mode = (not config.train["run_eval_disentangle"]) and (not config.test["run_eval_disentangle"])
    early_stop_callback = EarlyStopping(
        "average_r2_score", patience=model.hparams.training_params["patience"], mode="max"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.train["num_epochs"],
        max_time=config.train["max_time"],
        check_val_every_n_epoch=config.train["checkpoint_freq"],
        default_root_dir=config.model["save_dir"],
        # profiler="advanced",
        callbacks=[early_stop_callback, lr_monitor],
        inference_mode=inference_mode,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)
