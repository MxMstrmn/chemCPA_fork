import hydra
from omegaconf import DictConfig

import sys
sys.path.append('.')

import wandb
wandb.login()

from chemCPA.data import DataModule
from chemCPA.model import ComPert
import torch
import numpy as np

import lightning as L
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

@hydra.main(config_path='../chemCPA/config', config_name='config')
def main(cfg: DictConfig):
    dm = DataModule(cfg.model.hparams.batch_size,
                    cfg.model.training_params.full_eval_during_train,
                    **cfg.dataset.data_params
                    )

    model = ComPert(
        dm.datasets['training'].num_genes,
        dm.datasets['training'].num_drugs,
        dm.datasets['training'].num_knockouts,
        dm.datasets['training'].num_covariates,
        cfg.model.hparams,
        cfg.model.training_params,
        cfg.model.test_params,
        **cfg.model.additional_params,
        drug_embedding_dimension=dm.datasets['training'].drug_embedding_dimension,
        knockout_embedding_dimension=dm.datasets['training'].knockout_embedding_dimension
        )
    
    early_stop_callback = EarlyStopping('average_r2_score', 
                                    patience=cfg.model.training_params.patience, 
                                    mode='max')

    if (not cfg.model.training_params.run_eval_disentangle) and (not cfg.model.test_params.run_eval_disentangle):
        trainer = L.Trainer(
            logger=WandbLogger(log_model="all"),
            max_epochs=cfg.model.training_params.num_epochs,
            max_time=cfg.model.training_params.max_time,
            check_val_every_n_epoch= cfg.model.training_params.checkpoint_freq,
            default_root_dir=cfg.model.save_dir,
            profiler="advanced",
            callbacks=[early_stop_callback],
            #inference_mode=False
        )
    else: 
        trainer = L.Trainer(
            logger=WandbLogger(project='CPA', log_model="all", save_dir=cfg.model.save_dir),
            max_epochs=cfg.model.training_params.num_epochs,
            max_time=cfg.model.training_params.max_time,
            check_val_every_n_epoch= cfg.model.training_params.checkpoint_freq,
            default_root_dir=cfg.model.save_dir,
            profiler="advanced",
            callbacks=[early_stop_callback],
            inference_mode=False
        )


    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()

