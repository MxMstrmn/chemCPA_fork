import hydra
from omegaconf import DictConfig

import sys
sys.path.append('.')

import wandb
wandb.login()

from chemCPA.data import DataModule
from chemCPA.model import ComPert


import lightning as L
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base=None, config_path='../experiments/hydra_optuna_config', config_name='config')
def main(cfg: DictConfig):
    hparams = {
        'dim': cfg.dim,
        'dosers_width': cfg.dosers_width,
        'dosers_depth': cfg.dosers_depth,
        'dosers_lr': cfg.dosers_lr,
        'dosers_wd': cfg.dosers_wd,
        'knockout_effects_width': cfg.knockout_effects_width,
        'knockout_effects_depth': cfg.knockout_effects_depth,
        'knockout_effects_lr': cfg.knockout_effects_lr,
        'knockout_effects_wd': cfg.knockout_effects_wd,
        'autoencoder_width': cfg.autoencoder_width,
        'autoencoder_depth': cfg.autoencoder_depth,
        'adversary_width': cfg.adversary_width,
        'adversary_depth': cfg.adversary_depth,
        'reg_adversary_drug': cfg.reg_adversary_drug,
        'reg_adversary_knockout': cfg.reg_adversary_knockout,
        'reg_adversary_cov': cfg.reg_adversary_cov,
        'penalty_adversary': cfg.penalty_adversary,
        'autoencoder_lr': cfg.autoencoder_lr,
        'adversary_lr': cfg.adversary_lr,
        'autoencoder_wd': cfg.autoencoder_wd,
        'adversary_wd': cfg.adversary_wd,
        'adversary_steps': cfg.adversary_steps,
        'batch_size': cfg.batch_size,
        'step_size_lr': cfg.step_size_lr,
        'embedding_encoder_width': cfg.embedding_encoder_width,
        'embedding_encoder_depth': cfg.embedding_encoder_depth
    }

    dm = DataModule(hparams['batch_size'],
                    cfg.train.full_eval_during_train,
                    **cfg.dataset.data_params
                    )

    model = ComPert(
        dm.datasets['training'].num_genes,
        dm.datasets['training'].num_drugs,
        dm.datasets['training'].num_knockouts,
        dm.datasets['training'].num_covariates,
        hparams,
        cfg.train,
        cfg.test,
        **cfg.model.additional_params,
        drug_embedding_dimension=dm.datasets['training'].drug_embedding_dimension,
        knockout_embedding_dimension=dm.datasets['training'].knockout_embedding_dimension
        )

    wandb_logger = WandbLogger(
        project = cfg.model.model_type + "_" + cfg.dataset.dataset_type + "_optuna",
        save_dir = cfg.model.save_dir      
    )

    inference_mode = ((not cfg.train.run_eval_disentangle) and (not cfg.test.run_eval_disentangle))
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=cfg.train.num_epochs,
        max_time=cfg.train.max_time,
        check_val_every_n_epoch= cfg.train.checkpoint_freq,
        default_root_dir=cfg.model.save_dir,
        profiler="advanced",
        inference_mode=inference_mode,
        enable_checkpointing=False
    )
    
    trainer.fit(model, datamodule=dm)
    score = trainer.test(model, datamodule=dm)[0]['average_r2_score']
    return score
    
if __name__ == "__main__":
    main()

