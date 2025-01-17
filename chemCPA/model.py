import json
import logging
from collections import OrderedDict
from typing import Any, Union
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from chemCPA.train import evaluate_r2, evaluate_r2_sc, evaluate_logfold_r2, evaluate


if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

class NBLoss(torch.nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-8):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        dim = yhat.size(1) // 2
        # means of the negative binomial (has to be positive support)
        mu = yhat[:, :dim]
        # inverse dispersion parameter (has to be positive support)
        theta = yhat[:, dim:]

        if theta.ndimension() == 1:
            # In this case, we reshape theta for broadcasting
            theta = theta.view(1, theta.size(0))
        t1 = (
                torch.lgamma(theta + eps)
                + torch.lgamma(y + 1.0)
                - torch.lgamma(y + theta + eps)
        )
        t2 = (theta + y) * torch.log(1.0 + (mu / (theta + eps))) + (
                y * (torch.log(theta + eps) - torch.log(mu + eps))
        )
        final = t1 + t2
        final = _nan2inf(final)

        return torch.mean(final)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.3, gamma=3, reduction="mean") -> None:
        """Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .

        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        """Compute the FocalLoss

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        from torchvision.ops import focal_loss

        loss = focal_loss.sigmoid_focal_loss(
            inputs,
            target,
            reduction=self.reduction,
            gamma=self.gamma,
            alpha=self.alpha,
        )
        return loss


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
    """

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, yhat, y):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]

        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))

        return (term1 + term2).mean()


class CELoss(torch.nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, preds, targets):
        """
        Used for calculating the cross entropy loss when multiclass targets exist
        @param preds: dimension [batch_size, total_num_perts]
        @param target: an array or list of tensors, each tensor is of dim [num_target_in_cell]
        """
        
        softmax = torch.nn.Softmax(dim=1)
        preds = softmax(preds)

        batch_size = len(targets)
        preds = [preds[j,:][targets[j].long()] for j in range(batch_size)]
        loss = torch.tensor([-torch.log(pred).sum() for pred in preds])
        return (loss.sum() / batch_size)


class MLP(L.LightningModule):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.

    Careful: if activation is set to ReLU, ReLU is only applied to the first half of NN outputs!
    """

    def __init__(
            self,
            sizes,
            batch_norm=True,
            last_layer_act="linear",
            append_layer_width=None,
            append_layer_position=None,
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        # We add another layer either at the front / back of the sequential model. It gets a different name
        # `append_XXX`. The naming of the other layers stays consistent.
        # This allows us to load the state dict of the "non_appended" MLP without errors.
        if append_layer_width:
            assert append_layer_position in ("first", "last")
            if append_layer_position == "first":
                layers_dict = OrderedDict()
                layers_dict["append_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0]
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[0])
                layers_dict["append_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module
            else:
                layers_dict = OrderedDict(
                    {str(i): module for i, module in enumerate(layers)}
                )
                layers_dict["append_bn1d"] = torch.nn.BatchNorm1d(sizes[-1])
                layers_dict["append_relu"] = torch.nn.ReLU()
                layers_dict["append_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width
                )
        else:
            layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = torch.nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((self.relu(x[:, :dim]), x[:, dim:]), dim=1)
        return self.network(x)


class GeneralizedSigmoid(L.LightningModule):
    """
    Sigmoid, log-sigmoid or linear functions for encoding dose-response for
    drug perurbations.
    """

    def __init__(self, dim, nonlin="sigm"):
        """Sigmoid modeling of continuous variable.
        Params
        ------
        nonlin : str (default: logsigm)
            One of logsigm, sigm or None. If None, just returns the input unchanged.
        """
        super(GeneralizedSigmoid, self).__init__()
        assert nonlin in ("sigm", "logsigm", "original")
        self.nonlin = nonlin
        if self.nonlin in ["sigm", "logsigm"]:
            self.beta = torch.nn.Parameter(
                torch.ones(1, dim), requires_grad=True
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(1, dim), requires_grad=True
            )

    def forward(self, x, idx: list):
        if self.nonlin == "logsigm":
            bias = self.bias[0][idx]
            beta = self.beta[0][idx]
            c0 = bias.sigmoid()
            return (torch.log1p(x) * beta + bias).sigmoid() - c0
        elif self.nonlin == "sigm":
            bias = self.bias[0][idx]
            beta = self.beta[0][idx]
            c0 = bias.sigmoid()
            return (x * beta + bias).sigmoid() - c0
        else:
            return x


class ComPert(L.LightningModule):
    """
    Our main module, the ComPert autoencoder
    """

    num_drugs: int  # number of unique drugs in the dataset, including control
    num_knockouts: int # number of unqiue gene knockouts in the dataset, including control

    def __init__(
        self,
        num_genes: int,
        num_drugs: int,
        num_knockouts: int,
        num_covariates: list,
        hparams,
        training_params,
        test_params,
        seed=0,
        doser_type="logsigm",
        knockout_effect_type="original",
        decoder_activation="linear",
        drug_embedding_dimension=None,
        knockout_embedding_dimension=None,
        append_layer_width=None,
    ):
        super(ComPert, self).__init__()
        # set generic attributes
        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.num_knockouts = num_knockouts
        self.num_covariates = num_covariates
        self.seed = seed
        self.drug_embedding_dimension = drug_embedding_dimension
        self.knockout_embedding_dimension = knockout_embedding_dimension


        #manual optimization
        self.automatic_optimization = False

        if not isinstance(hparams, dict):
            hparams = self.set_hparams_(seed, hparams)
        #save hyperparameters
        self.save_hyperparameters()
        
        self.encoder = MLP(
            [num_genes]
            + [self.hparams.hparams['autoencoder_width']] * self.hparams.hparams['autoencoder_depth']
            + [self.hparams.hparams['dim']],
            append_layer_width=append_layer_width,
            append_layer_position="first",
            )

        self.decoder = MLP(
            [self.hparams.hparams['dim']]
            + [self.hparams.hparams['autoencoder_width']] * self.hparams.hparams['autoencoder_depth']
            + [num_genes * 2],
            last_layer_act=decoder_activation,
            append_layer_width=2 * append_layer_width if append_layer_width else None,
            append_layer_position="last",
            )

        if append_layer_width:
            self.num_genes = append_layer_width


        if self.num_drugs > 0:
            self.adversary_drugs = MLP(
                [self.hparams.hparams['dim']]
                + [self.hparams.hparams['adversary_width']] * self.hparams.hparams['adversary_depth']
                + [self.num_drugs]
            )

            self.drug_embedding_encoder = MLP(
                [self.drug_embedding_dimension]
                + [self.hparams.hparams['embedding_encoder_width']]
                * self.hparams.hparams['embedding_encoder_depth']
                + [self.hparams.hparams['dim']],
                last_layer_act="linear",
            )

            self.loss_adversary_drugs = CELoss()
            
            # set dosers
            assert doser_type in ("mlp", "sigm", "logsigm", "amortized", "original")
            if doser_type == "mlp":
                self.dosers = torch.nn.ModuleList()
                for _ in range(self.num_drugs):
                    self.dosers.append(
                        MLP(
                            [1]
                            + [self.hparams.hparams['dosers_width']]
                            * self.hparams.hparams['dosers_depth']
                            + [1],
                            batch_norm=False,
                            )
                    )
            elif doser_type == "amortized":
                # should this also have `batch_norm=False`?
                self.dosers = MLP(
                    [self.drug_embedding_dimension + 1]
                    + [self.hparams.hparams['dosers_width']] * self.hparams.hparams['dosers_depth']
                    + [1],
                    )
            else:
                assert doser_type in ("sigm", "logsigm", "original")
                self.dosers = GeneralizedSigmoid(
                    self.num_drugs, nonlin=doser_type
                )
            self.doser_type = doser_type


        if self.num_knockouts > 0:
            self.adversary_knockouts = MLP(
                [self.hparams.hparams['dim']]
                + [self.hparams.hparams['adversary_width']] * self.hparams.hparams['adversary_depth']
                + [self.num_knockouts]
            )
            self.knockout_embedding_encoder = MLP(
                [self.knockout_embedding_dimension]
                + [self.hparams.hparams['embedding_encoder_width']]
                * self.hparams.hparams['embedding_encoder_depth']
                + [self.hparams.hparams['dim']],
                last_layer_act="linear",
            )

            self.loss_adversary_knockout = CELoss()
            
            # set knockouts' effect
            assert knockout_effect_type in ("mlp", "sigm", "logsigm", "amortized", "original")
            if knockout_effect_type == "mlp":
                self.knockout_effects = torch.nn.ModuleList()
                for _ in range(self.num_knockouts):
                    self.knockout_effects.append(
                        MLP(
                            [1]
                            + [self.hparams.hparams['knockout_effects_width']]
                            * self.hparams.hparams['knockout_effects_depth']
                            + [1],
                            batch_norm=False,
                        )
                    )
            elif knockout_effect_type == "amortized":
                # should this also have `batch_norm=False`?
                self.knockout_effects = MLP(
                    [self.knockout_embedding_dimension + 1]
                    + [self.hparams.hparams['knockout_effects_width']] 
                    * self.hparams.hparams['knockout_effects_depth']
                    + [1],
                )
            else:
                assert knockout_effect_type in ("sigm", "logsigm", "original")
                self.knockout_effects = GeneralizedSigmoid(
                    self.num_knockouts, nonlin=knockout_effect_type
                )
            self.knockout_effect_type = knockout_effect_type
        

        if self.num_covariates == [0]:
            pass    
        else:
            assert 0 not in self.num_covariates
            self.adversary_covariates = []
            self.loss_adversary_covariates = []
            self.covariates_embeddings = (
                []
            )  # TODO: Continue with checking that dict assignment is possible via covaraites names and if dict are possible to use in optimisation
            for num_covariate in self.num_covariates:
                self.adversary_covariates.append(
                    MLP(
                        [self.hparams.hparams['dim']]
                        + [self.hparams.hparams['adversary_width']]
                        * self.hparams.hparams['adversary_depth']
                        + [num_covariate]
                    ).to(_device)
                )
                self.loss_adversary_covariates.append(CELoss())
                self.covariates_embeddings.append(
                    torch.nn.Embedding(num_covariate, self.hparams.hparams['dim'], device=_device)
                )

        self.loss_autoencoder = torch.nn.GaussianNLLLoss()

        self.iteration = 0
        

    def configure_optimizers(self):
        has_drugs = self.num_drugs > 0
        has_knockouts = self.num_knockouts > 0
        has_covariates = self.num_covariates[0] > 0

        get_params = lambda model: list(model.parameters()) 
        _parameters = (
            get_params(self.encoder)
            + get_params(self.decoder)
        )
        if has_drugs:
            _parameters.extend(get_params(self.drug_embedding_encoder))
        if has_knockouts:
            _parameters.extend(get_params(self.knockout_embedding_encoder))
        if has_covariates:
            for emb in self.covariates_embeddings:
                _parameters.extend(get_params(emb))
        optimizer_autoencoder = torch.optim.Adam(
            _parameters,
            lr=self.hparams.hparams['autoencoder_lr'],
            weight_decay=self.hparams.hparams['autoencoder_wd'],
        )
        scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(
            optimizer_autoencoder,
            step_size=self.hparams.hparams['step_size_lr'],
            gamma=0.5,
        )

        _parameters = []
        if has_drugs:
            _parameters.extend(get_params(self.adversary_drugs))
        if has_knockouts:
            _parameters.extend(get_params(self.adversary_knockouts))
        if has_covariates:
            for adv in self.adversary_covariates:
                _parameters.extend(get_params(adv))
        optimizer_adversaries = torch.optim.Adam(
            _parameters,
            lr=self.hparams.hparams['adversary_lr'],
            weight_decay=self.hparams.hparams['adversary_wd'],
        )
        scheduler_adversary = torch.optim.lr_scheduler.StepLR(
            optimizer_adversaries,
            step_size=self.hparams.hparams['step_size_lr'],
            gamma=0.5,
        )

        if has_drugs and (self.doser_type != "original"):
            optimizer_dosers = torch.optim.Adam(
                self.dosers.parameters(),
                lr=self.hparams.hparams['dosers_lr'],
                weight_decay=self.hparams.hparams['dosers_wd'],
            )
            scheduler_dosers = torch.optim.lr_scheduler.StepLR(
                optimizer_dosers,
                step_size=self.hparams.hparams['step_size_lr'],
                gamma=0.5,
            )
        else: 
            optimizer_dosers = None
            scheduler_dosers = None
        if has_knockouts and (self.knockout_effect_type != "original"):
            optimizer_knockout_effects = torch.optim.Adam(
                self.knockout_effects.parameters(),
                lr=self.hparams.hparams['knockout_effects_lr'],
                weight_decay=self.hparams.hparams['knockout_effects_wd'],
            )
            scheduler_knockout_effects = torch.optim.lr_scheduler.StepLR(
                optimizer_knockout_effects,
                step_size=self.hparams.hparams['step_size_lr'],
                gamma=0.5,
            )
        else: 
            optimizer_knockout_effects = None
            scheduler_knockout_effects = None

        if optimizer_dosers is not None:
            if optimizer_knockout_effects is not None:
                return [{'optimizer': optimizer_autoencoder, "lr_scheduler": {"scheduler": scheduler_autoencoder}},
                        {'optimizer': optimizer_adversaries, "lr_scheduler": {"scheduler": scheduler_adversary}},
                        {'optimizer': optimizer_dosers, "lr_scheduler": {"scheduler": scheduler_dosers}},
                        {'optimizer': optimizer_knockout_effects, "lr_scheduler": {"scheduler": scheduler_knockout_effects}}
                    ]
            else: 
                return [{'optimizer': optimizer_autoencoder, "lr_scheduler": {"scheduler": scheduler_autoencoder}},
                        {'optimizer': optimizer_adversaries, "lr_scheduler": {"scheduler": scheduler_adversary}},
                        {'optimizer': optimizer_dosers, "lr_scheduler": {"scheduler": scheduler_dosers}}
                    ]
        else:
            if optimizer_knockout_effects is not None:
                return [{'optimizer': optimizer_autoencoder, "lr_scheduler": {"scheduler": scheduler_autoencoder}},
                        {'optimizer': optimizer_adversaries, "lr_scheduler": {"scheduler": scheduler_adversary}},
                        {'optimizer': optimizer_knockout_effects, "lr_scheduler": {"scheduler": scheduler_knockout_effects}}
                    ]
            else: 
                return [{'optimizer': optimizer_autoencoder, "lr_scheduler": {"scheduler": scheduler_autoencoder}},
                        {'optimizer': optimizer_adversaries, "lr_scheduler": {"scheduler": scheduler_adversary}}
                    ]


    def set_hparams_(self, seed, hparams):
        """
        Set hyper-parameters to (i) default values if `seed=0`, (ii) random
        values if `seed != 0`, or (iii) values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        default = seed == 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        hyperparameters = {
            "dim": 256 if default else int(np.random.choice([128, 256, 512])),
            "dosers_width": 64 if default else int(np.random.choice([32, 64, 128])),
            "dosers_depth": 2 if default else int(np.random.choice([1, 2, 3])),
            "dosers_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "dosers_wd": 1e-7 if default else float(10 ** np.random.uniform(-8, -5)),
            "knockout_effects_width": 64 if default else int(np.random.choice([32, 64, 128])),
            "knockout_effects_depth": 2 if default else int(np.random.choice([1, 2, 3])),
            "knockout_effects_lr": 1e-3 if default else float(10 ** np.random.uniform(-4, -2)),
            "knockout_effects_wd": 1e-7 if default else float(10 ** np.random.uniform(-8, -5)),
            "autoencoder_width": 512
            if default
            else int(np.random.choice([256, 512, 1024])),
            "autoencoder_depth": 4 if default else int(np.random.choice([3, 4, 5])),
            "adversary_width": 128
            if default
            else int(np.random.choice([64, 128, 256])),
            "adversary_depth": 3 if default else int(np.random.choice([2, 3, 4])),
            "reg_adversary_drug": 5 if default else float(10 ** np.random.uniform(-2, 2)),
            "reg_adversary_knockout": 5 if default else float(10 ** np.random.uniform(-2, 2)),
            "reg_adversary_cov": 5 if default else float(10 ** np.random.uniform(-2, 2)),
            "penalty_adversary": 3
            if default
            else float(10 ** np.random.uniform(-2, 1)),
            "autoencoder_lr": 1e-3
            if default
            else float(10 ** np.random.uniform(-4, -2)),
            "adversary_lr": 3e-4 if default else float(10 ** np.random.uniform(-5, -3)),
            "autoencoder_wd": 1e-6
            if default
            else float(10 ** np.random.uniform(-8, -4)),
            "adversary_wd": 1e-4 if default else float(10 ** np.random.uniform(-6, -3)),
            "adversary_steps": 3 if default else int(np.random.choice([1, 2, 3, 4, 5])),
            "batch_size": 128
            if default
            else int(np.random.choice([64, 128, 256, 512])),
            "step_size_lr": 45 if default else int(np.random.choice([15, 25, 45])),
            "embedding_encoder_width": 512,
            "embedding_encoder_depth": 0,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                hyperparameters.update(json.loads(hparams))
            else:
                hyperparameters.update(hparams)

        return hyperparameters


    def compute_drug_embeddings_(self, drugs_idx, dosages, drugs_embeddings):
        """
        Compute sum of drug embeddings, each of them multiplied by its
        dose-response curve.

        @param drugs_idx: an array or list of tensor containing drug indices for each selected cell, and the indices of each cell is 
                        of dim [num_drugs_in_cell]
        @param dosages: an array or list of tensor containing drug dosages for each selected cell, and the dosage of each cell is 
                        of dim [num_drugs_in_cell]
        @param drugs_embeddings: an array or list of tensor containing drug embeddings for each selected cell, and the drug embedding is 
                                of dim [num_drugs_in_cell, drug_embedding_dim]
        @return: a tensor of shape [batch_size, drug_embedding_dimension]
        """
        assert (drugs_idx is not None and dosages is not None)

        stacked_idx = []
        for i, dosage in enumerate(dosages):
            if i==0:
                stacked_idx.append(len(dosage))
            else:
                stacked_idx.append(len(dosage) + stacked_idx[-1])
        
        scaled_dosages = []
        if self.doser_type == "mlp":
            for idx, dosage in zip(drugs_idx, dosages):
                scaled_dosage = []
                for i in range(dosage.shape[0]):
                    scaled_dosage.append(
                        self.dosers[idx[i]](dosage[i].unsqueeze(0)).sigmoid() #why sigmoid??
                    )
                scaled_dosage = torch.stack(scaled_dosage, dim=1)
                scaled_dosages.append(scaled_dosage.squeeze(0))


        elif self.doser_type == "amortized":
            total_stack = [torch.concat([embedding, dosage.view(-1, 1)], dim=1) for 
                           dosage, embedding in zip(dosages, drugs_embeddings)]
            total_stack = torch.cat(total_stack, dim=0)
            stacked_dosages = self.dosers(total_stack)
            for i in range(len(dosages)):
                if i == 0:
                    scaled_dosages.append(stacked_dosages[0:stacked_idx[0]].view(-1))
                else:
                    scaled_dosages.append(stacked_dosages[stacked_idx[i-1]:stacked_idx[i]].view(-1))

        else:
            assert self.doser_type in ("sigm", "logsigm", "original")
            for idx, dosage in zip(drugs_idx, dosages):
                scaled_dosages.append(self.dosers(dosage, idx))

        
        # Transform and adjust dimension to latent dims
        # need to concatenate input since the embedding encoder performs batch norm
        latent_stacked = self.drug_embedding_encoder(torch.concat(drugs_embeddings, dim=0))
        latent_drugs = []
        for i in range(len(drugs_embeddings)):
            if i == 0:
                latent_drugs.append(latent_stacked[0: stacked_idx[0]])
            else:
                latent_drugs.append(latent_stacked[stacked_idx[i-1]:stacked_idx[i]])
        latent_drugs = [torch.einsum("b,bd->bd", [scaled_dosage, latent_drug]) for 
                        scaled_dosage, latent_drug in zip(scaled_dosages, latent_drugs)]
        return torch.stack([latent_drug.sum(dim=0) for latent_drug in latent_drugs])


    def compute_knockout_embeddings_(self, knockouts_idx, knockouts_embeddings):
        """
        Compute sum of knockout embeddingss, each of them multiplied by its effects curve.

        @param knockouts_idx: an array or list of tensor containing knockout indices for each selected cell, and the indices of each cell is 
                        of dim [num_knockouts_in_cell]
    
        @param knockouts_embeddings: an arraay or list of tensor containing knockout embeddings for each selected cell, and the knockout embedding is 
                                of dim [num_knockouts_in_cell, knockout_embedding_dim]
        @return: a tensor of shape [batch_size, knockout_embedding_dimension]
        """
        
        effects = [torch.ones_like(idx, dtype=torch.float32) for idx in knockouts_idx]
        stacked_idx = []
        for i, effect in enumerate(effects):
            if i==0:
                stacked_idx.append(len(effect))
            else:
                stacked_idx.append(len(effect) + stacked_idx[-1])
        scaled_effects = []
        if self.knockout_effect_type == "mlp":
            for idx, effect in zip(knockouts_idx, effects):
                scaled_effect = []
                for i in range(effect.shape[0]):
                    scaled_effect.append(
                        self.knockout_effects[idx[i]](effect[i].unsqueeze(0)).sigmoid() 
                    )
                scaled_effect = torch.stack(scaled_effect, dim=1)
                scaled_effects.append(scaled_effect.squeeze(0))
        elif self.knockout_effect_type == "amortized":
            total_stack = [torch.concat([embedding, effect.view(-1, 1)], dim=1) for 
                           effect, embedding in zip(effects, knockouts_embeddings)]
            
            total_stack = torch.cat(total_stack, dim=0)
            stacked_effects = self.knockout_effects(total_stack)
            for i in range(len(effects)):
                if i == 0:
                    scaled_effects.append(stacked_effects[0:stacked_idx[0]].view(-1))
                else:
                    scaled_effects.append(stacked_effects[stacked_idx[i-1]:stacked_idx[i]].view(-1))


        else:
            assert self.knockout_effect_type in ("sigm", "logsigm", "original")
            for idx, effect in zip(knockouts_idx, effects):
                scaled_effects.append(self.knockout_effects(effect, idx))

        
        # Transform and adjust dimension to latent dims
        # need to concatenate input since the embedding encoder performs batch norm
        latent_stacked = self.knockout_embedding_encoder(torch.concat(knockouts_embeddings, dim=0))
        latent_knockouts = []
        for i in range(len(knockouts_embeddings)):
            if i == 0:
                latent_knockouts.append(latent_stacked[0: stacked_idx[0]])
            else:
                latent_knockouts.append(latent_stacked[stacked_idx[i-1]:stacked_idx[i]])
        latent_knockouts = [torch.einsum("b,bd->bd", [scaled_effect, latent_knockout]) for 
                            scaled_effect, latent_knockout in zip(scaled_effects, latent_knockouts)]
        return torch.stack([latent_knockout.sum(dim=0) for latent_knockout in latent_knockouts])


    def predict(
        self,
        genes,
        drugs_idx=None,
        dosages=None,
        drugs_embeddings=None,
        knockouts_idx=None,
        knockouts_embeddings=None,
        covariates_idx=None,
        return_latent_basal=False,
    ):
        """
        Predict "what would have the gene expression `genes` been, had the
        cells in `genes` with covariates been treated with perturbations
        """
        
        latent_basal = self.encoder(genes)

        latent_treated = latent_basal
        if self.num_drugs > 0:
            drug_embedding = self.compute_drug_embeddings_(
                drugs_idx=drugs_idx, dosages=dosages, drugs_embeddings=drugs_embeddings
            )
            latent_treated = latent_treated + drug_embedding
        if self.num_knockouts > 0:
            knockout_embedding = self.compute_knockout_embeddings_(
                knockouts_idx=knockouts_idx, knockouts_embeddings=knockouts_embeddings
            )
            latent_treated = latent_treated + knockout_embedding
        if self.num_covariates[0] > 0:
            for cov_type, emb_cov in enumerate(self.covariates_embeddings):
                cov_idx = covariates_idx[cov_type]
                latent_treated = latent_treated + emb_cov(cov_idx)

        gene_reconstructions = self.decoder(latent_treated)

        # convert variance estimates to a positive value in [0, \infty)
        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = F.softplus(gene_reconstructions[:, dim:])
        normalized_reconstructions = torch.concat([mean, var], dim=1)

        if return_latent_basal:
            return normalized_reconstructions, latent_basal

        return normalized_reconstructions

    
    def training_step(
        self,
        batch,
        batch_idx
        ):

        genes, drugs_idx, dosages, drugs_embeddings, knockouts_idx, knockouts_embeddings, covariates_idx = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            batch[5],
            batch[6:]
        )
        gene_reconstructions, latent_basal = self.predict(
            genes=genes,
            drugs_idx=drugs_idx,
            dosages=dosages,
            drugs_embeddings=drugs_embeddings,
            knockouts_idx=knockouts_idx,
            knockouts_embeddings=knockouts_embeddings,
            covariates_idx=covariates_idx,
            return_latent_basal=True,
        )

        dim = gene_reconstructions.size(1) // 2
        mean = gene_reconstructions[:, :dim]
        var = gene_reconstructions[:, dim:]
        reconstruction_loss = self.loss_autoencoder(input=mean, target=genes, var=var)


        if (self.num_drugs > 0) and (self.doser_type != "original"):
            if (self.num_knockouts > 0) and (self.knockout_effect_type != "original"):
                optimizer_autoencoder, optimizer_adversaries, optimizer_dosers, optimizer_knockout_effects = self.optimizers()
                scheduler_autoencoder, scheduler_adversary, scheduler_dosers, scheduler_knockout_effects = self.lr_schedulers()
            else:
                optimizer_autoencoder, optimizer_adversaries, optimizer_dosers = self.optimizers()
                scheduler_autoencoder, scheduler_adversary, scheduler_dosers = self.lr_schedulers()
        else:
            if (self.num_knockouts > 0) and (self.knockout_effect_type != "original"):
                optimizer_autoencoder, optimizer_adversaries, optimizer_knockout_effects = self.optimizers()
                scheduler_autoencoder, scheduler_adversary, scheduler_knockout_effects = self.lr_schedulers()
            else:
                optimizer_autoencoder, optimizer_adversaries = self.optimizers()
                scheduler_autoencoder, scheduler_adversary = self.lr_schedulers()

        adversary_drugs_loss = torch.tensor([0.0], device=_device)
        if self.num_drugs > 0:
            adversary_drugs_predictions = self.adversary_drugs(latent_basal)
            adversary_drugs_loss = self.loss_adversary_drugs(
                adversary_drugs_predictions, drugs_idx
            )
        adversary_knockouts_loss = torch.tensor([0.0], device=_device)
        if self.num_knockouts > 0:
            adversary_knockouts_predictions = self.adversary_knockouts(latent_basal)
            adversary_knockouts_loss = self.loss_adversary_knockout(
                adversary_knockouts_predictions, knockouts_idx
            )

        adversary_covariates_loss = torch.tensor([0.0], device=_device)
        if self.num_covariates[0] > 0:
            adversary_covariate_predictions = []
            for i, adv in enumerate(self.adversary_covariates):
                adversary_covariate_predictions.append(adv(latent_basal))
                adversary_covariates_loss += self.loss_adversary_covariates[i](
                    adversary_covariate_predictions[-1], covariates_idx[i]
                )


        # place-holders for when adversary is not executed
        adv_drugs_grad_penalty = torch.tensor([0.0], device=_device)
        adv_knockouts_grad_penalty = torch.tensor([0.0], device=_device)
        adv_covs_grad_penalty = torch.tensor([0.0], device=_device)

        if ((self.iteration) % self.hparams.hparams['adversary_steps']) == 0:
            def compute_gradient_penalty(output, input):
                grads = torch.autograd.grad(output, input, create_graph=True)
                grads = grads[0].pow(2).mean()
                return grads

            if self.num_drugs > 0:
                adv_drugs_grad_penalty = compute_gradient_penalty(
                    adversary_drugs_predictions.sum(), latent_basal
                )
            if self.num_knockouts > 0:
                adv_knockouts_grad_penalty = compute_gradient_penalty(
                    adversary_knockouts_predictions.sum(), latent_basal
                )
            if self.num_covariates[0] > 0:
                adv_covs_grad_penalty = torch.tensor([0.0], device=_device)
                for pred in adversary_covariate_predictions:
                    adv_covs_grad_penalty += compute_gradient_penalty(
                        pred.sum(), latent_basal
                    )
            optimizer_adversaries.zero_grad()
            self.manual_backward(
                                    adversary_drugs_loss
                                    + self.hparams.hparams['penalty_adversary'] * adv_drugs_grad_penalty
                                    + adversary_knockouts_loss
                                    + self.hparams.hparams['penalty_adversary'] * adv_knockouts_grad_penalty
                                    + adversary_covariates_loss
                                    + self.hparams.hparams['penalty_adversary'] * adv_covs_grad_penalty
            )
            optimizer_adversaries.step()
        else:
            optimizer_autoencoder.zero_grad()
            if (self.num_drugs > 0) and (self.doser_type != "original"):
                optimizer_dosers.zero_grad()
            if (self.num_knockouts > 0) and (self.knockout_effect_type != "original"):
                optimizer_knockout_effects.zero_grad()
            self.manual_backward(
                                reconstruction_loss
                                - self.hparams.hparams['reg_adversary_drug'] * adversary_drugs_loss
                                - self.hparams.hparams['reg_adversary_knockout'] * adversary_knockouts_loss
                                - self.hparams.hparams['reg_adversary_cov'] * adversary_covariates_loss * (self.num_covariates != [1])
                                #turn off the covariate loss if only 1 cell line
                                #the implementation may be different when there are multiple covariates
            )
            optimizer_autoencoder.step()
            if (self.num_drugs > 0) and (self.doser_type != "original"):
                optimizer_dosers.step()
            if (self.num_knockouts > 0) and (self.knockout_effect_type != "original"):
                optimizer_knockout_effects.step()

        if self.trainer.is_last_batch:
            scheduler_autoencoder.step()
            scheduler_adversary.step()
            if (self.num_drugs > 0) and (self.doser_type != "original"):
                scheduler_dosers.step()
            if (self.num_knockouts > 0) and (self.knockout_effect_type != "original"):
                scheduler_knockout_effects.step()


        self.iteration += 1
        train_stats = {
            "loss_reconstruction": reconstruction_loss.item(),
            "loss_adv_drugs": adversary_drugs_loss.item(),
            "loss_adv_knockouts": adversary_knockouts_loss.item(),
            "loss_adv_covariates": adversary_covariates_loss.item(),
            "penalty_adv_drugs": adv_drugs_grad_penalty.item(),
            "penalty_adv_knockouts": adv_knockouts_grad_penalty.item(),
            "penalty_adv_covariates": adv_covs_grad_penalty.item(),
        }
        self.log_dict(train_stats, on_step=True, on_epoch=True, prog_bar=True, 
                      logger=True, batch_size=self.hparams.hparams['batch_size'])
        

    def validation_step(self, batch, batch_idx):
        if not self.hparams.training_params['full_eval_during_train']:
            logging.info(f"Running accuracy evaluation (Epoch:{self.current_epoch})")
            dataset_test_treated, dataset_test_control_genes = batch
            evaluation_r2_scores = evaluate_r2(
                    self,
                    dataset_test_treated,
                    dataset_test_control_genes,
                )
            """
            evaluation_r2_sc_scores = evaluate_r2_sc(
                 self,
                 dataset_test_treated
                )
            """
            self.log_dict({
                'mean_r2_score': evaluation_r2_scores[0],
                'mean_r2_score_de': evaluation_r2_scores[1],
                'var_score': evaluation_r2_scores[2],
                'var_score_de': evaluation_r2_scores[3],
                'average_r2_score': np.mean(evaluation_r2_scores),
                #'mean_r2_sc_score': evaluation_r2_sc_scores[0],
                #'mean_r2_sc_score_de': evaluation_r2_sc_scores[1],
                #'var_sc_score': evaluation_r2_sc_scores[2],
                #'var_sc_score_de': evaluation_r2_sc_scores[3],
                #'average_r2_sc_score': np.mean(evaluation_r2_sc_scores)
            }, batch_size=1)
        else:
            logging.info(f"Running the full evaluation (Epoch:{self.current_epoch})")
            datasets = batch
            evaluation_stats = evaluate(
                                        self,
                                        datasets,
                                        run_disentangle=self.hparams.training_params['run_eval_disentangle'],
                                        run_r2=self.hparams.training_params['run_eval_r2'],
                                        run_r2_sc=self.hparams.training_params['run_eval_r2_sc'],
                                        run_logfold=self.hparams.training_params['run_eval_logfold'],
                        )
            self.log_dict(evaluation_stats, batch_size=1)
            if self.hparams.training_params['run_eval_r2']:
                self.log('average_r2_score', 
                        np.mean([evaluation_stats['test_mean_score'],
                        evaluation_stats['test_mean_score_de'],
                        evaluation_stats['test_var_score'],
                        evaluation_stats['test_var_score_de']]),
                        batch_size=1)


    def on_save_checkpoint(self, checkpoint):
        #need to save the states of covariates separately
        checkpoint['state_dict_covs'] = {}
        checkpoint['state_dict_covs']['adversary_covariates'] = [
                                adversary_covariates.state_dict()
                                for adversary_covariates in self.adversary_covariates
                                ]
        checkpoint['state_dict_covs']['covariates_embeddings'] = [
                                covariate_embedding.state_dict()
                                for covariate_embedding in self.covariates_embeddings
                            ]
        

    def on_load_checkpoint(self, checkpoint):
        #need to load the states of covariates separately
        if self.num_covariates[0] > 0:
                for embedding, state_dict in zip(
                    self.covariates_embeddings, checkpoint['state_dict_covs']['covariates_embeddings']
                ):
                    embedding.load_state_dict(state_dict)
                for adv, state_dict in zip(
                    self.adversary_covariates, checkpoint['state_dict_covs']['adversary_covariates']
                ):
                    adv.load_state_dict(state_dict)


    def test_step(self, batch, batch_idx):
        datasets = batch
        evaluation_stats = evaluate(
                                    self,
                                    datasets,
                                    run_disentangle=self.hparams.test_params['run_eval_disentangle'],
                                    run_r2=self.hparams.test_params['run_eval_r2'],
                                    run_r2_sc=self.hparams.test_params['run_eval_r2_sc'],
                                    run_logfold=self.hparams.test_params['run_eval_logfold'],
                    )
        self.log_dict(evaluation_stats, batch_size=1)
        if self.hparams.test_params['run_eval_r2']:
            self.log('average_r2_score', 
                    np.mean([evaluation_stats['test_mean_score'],
                                                    evaluation_stats['test_mean_score_de'],
                                                    evaluation_stats['test_var_score'],
                                                    evaluation_stats['test_var_score_de']]),
                    batch_size=1)
        

    @classmethod
    def defaults(self):
        """
        Returns the list of default hyper-parameters for ComPert
        """

        return self.set_hparams_(self, 0, "")
