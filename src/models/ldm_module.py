"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
import os
import random
import time
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.eval.catalyst_generation import CatalystGenerationEvaluator
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modelling of 3D atomic systems.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        sampling: DictConfig,
        conditional_generation: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # autoencoder models (first-stage model)
        self.autoencoder_ckpt = autoencoder_ckpt
        log.info(f"Loading Autoencoder ckpt: {autoencoder_ckpt}")
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location="cpu"
        )
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        # Condition warmup: gradually increase condition scale from 0 to 1
        # This helps prevent catastrophic forgetting when fine-tuning with new conditions
        self.warmup_epochs = getattr(conditional_generation, 'warmup_epochs', 0)

        # evaluator objects for computing metrics
        data_dir = self.hparams.sampling.data_dir
        self.dataset_configs = {
            "oc20": {
                "train_dataset_path": os.path.join(data_dir, "oc20/is2re/processed/train.pt"),
                "gt_catalyst_path": os.path.join(data_dir, "oc20/s2ef/val_10k_catalyst.pkl"),
                "num_nodes_bincount_path": os.path.join(data_dir, "oc20/num_nodes_bincount.pt"),
            },
            "oc22": {
                "train_dataset_path": os.path.join(data_dir, "oc22/is2re/processed/train.pt"),
                "gt_catalyst_path": os.path.join(data_dir, "oc22/val_10k_catalyst.pkl"),
                "num_nodes_bincount_path": os.path.join(data_dir, "oc22/num_nodes_bincount.pt"),
            },
        }

        # evaluator objects for computing metrics
        self.val_generation_evaluators = {
            dataset: CatalystGenerationEvaluator(
                train_dataset_path=config["train_dataset_path"],
                stol=0.3,
                angle_tol=5,
                ltol=0.2,
                compute_novelty=self.hparams.sampling.compute_novelty,
            )
            for dataset, config in self.dataset_configs.items()
        }
        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)
        self.num_nodes_bincount = {}
        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "oc20": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "oc22": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # num nodes bincount for sampling
        self.num_nodes_bincount = None

    def forward(self, batch: Data, sample_posterior: bool = True):
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            if sample_posterior:
                encoded_batch["x"] = encoded_batch["posterior"].sample()
            else:
                encoded_batch["x"] = encoded_batch["posterior"].mode()
            x_1 = encoded_batch["x"]

            # Convert from PyG batch to dense batch with padding
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])
            dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}

        # corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        # Prepare conditioning inputs to forward pass
        # binding_energy: continuous variable (-9.9~9.9eV), use learnable null embedding
        if self.hparams.conditional_generation.binding_energy.use and hasattr(batch, 'binding_energy'):
            binding_energy = batch.binding_energy
        else:
            binding_energy = torch.zeros(mask.shape[0], dtype=torch.float32, device=mask.device)

        # ads_id: discrete classes (0~81)
        ads_id = batch.ads_id + 1
        if not self.hparams.conditional_generation.ads_id.use:
            ads_id = torch.zeros_like(ads_id)  # null class: 0

        # cat_class: discrete classes (0~4)
        cat_class = batch.cat_class + 1
        if not self.hparams.conditional_generation.cat_class.use:
            cat_class = torch.zeros_like(cat_class)

        # Compute condition scale for warmup (0 → 1 over warmup_epochs)
        if self.warmup_epochs > 0 and self.training:
            cond_scale = min(1.0, self.current_epoch / self.warmup_epochs)
        else:
            cond_scale = 1.0

        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    binding_energy=binding_energy,
                    ads_id=ads_id,
                    cat_class=cat_class,
                    mask=mask,
                    x_sc=None,
                    cond_scale=cond_scale,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            binding_energy=binding_energy,
            ads_id=ads_id,
            cat_class=cat_class,
            mask=mask,
            x_sc=x_sc,
            cond_scale=cond_scale,
        )

        return pred_x, noisy_dense_encoded_batch

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute MSE loss w/ masking for padded tokens
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # log relative proportions of datasets in batch
        loss_dict["dataset_idx"] = batch.dataset_idx.detach().flatten()

        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        # Log condition scale for warmup monitoring
        if self.warmup_epochs > 0:
            cond_scale = min(1.0, self.current_epoch / self.warmup_epochs)
            self.log("train/cond_scale", cond_scale, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch: Data, batch_idx: int, dataloader_idx: int=0) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch: Data, batch_idx: int, dataloader_idx: int=0) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch starts."
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")
        for dataset in generation_evaluators.keys():
            generation_evaluators[dataset].clear()  # clear lists for next epoch

    def evaluation_step(self, batch, batch_idx, dataloader_idx, stage):
        dataset_name = 'oc20' if batch.dataset_idx[0].item() == 0 else 'oc22'
        
        metrics = getattr(self, f"{stage}_metrics")[dataset_name]
        generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[dataset_name]
        generation_evaluator.device = metrics["loss"].device

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # update and log metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{dataset_name}/{k}",  # val_oc20/loss, val_oc22/loss
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        for dataset in metrics.keys():
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            t_start = time.time()
            for samples_so_far in tqdm(
                range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
                desc=f"    Sampling",
            ):
                # Perform sampling and decoding to catalyst structures
                out, batch, samples = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount[dataset],
                    batch_size=self.hparams.sampling.batch_size,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                    dataset=dataset,
                )
                # Save predictions for metrics and visualisation
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    _atom_types = (out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1))  # take argmax
                    _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
                    _tags = (out["tags"].narrow(0, start_idx, num_atom).argmax(dim=1))
                    _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
                    _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (
                        1 / 3
                    )  # unscale lengths
                    _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees
                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            "tags": _tags.detach().cpu().numpy(),
                            "frac_coords": _frac_coords.detach().cpu().numpy(),
                            "lengths": _lengths.detach().cpu().numpy(),
                            "angles": _angles.detach().cpu().numpy(),
                            "sample_idx": samples_so_far
                            + self.global_rank * len(batch["num_atoms"])
                            + idx_in_batch,
                        }
                    )
                    start_idx = start_idx + num_atom
            t_end = time.time()

            # Compute generation metrics
            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=self.hparams.sampling.save_dir + f"/{dataset}_{stage}_{self.global_rank}",
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            for k, v in gen_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "struct_valid_rate" else True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            if self.hparams.sampling.visualize and type(self.logger) == WandbLogger:
                pred_table = generation_evaluators[dataset].get_wandb_table(
                    current_epoch=self.current_epoch,
                    save_dir=self.hparams.sampling.save_dir
                    + f"/{dataset}_{stage}_{self.global_rank}",
                )
                self.logger.experiment.log(
                    {f"{dataset}_{stage}_samples_table_device{self.global_rank}": pred_table}
                )

    #####################################################################################################

    def sample_and_decode(
        self,
        num_nodes_bincount,
        batch_size,
        cfg_scale=4.0,
        dataset=None,  # "oc20", "oc22", or "both" (defaults to config value)
    ):
        # sample random lengths from distribution: (B, 1)
        sample_lengths = torch.multinomial(
            num_nodes_bincount.float(),
            batch_size,
            replacement=True,
        ).to(self.device)

        # NOTE 0 -> null class within DiT, while 0 represents certain class or variable in data, so increment by 1
        # create adsorbate ID tensor
        if self.hparams.conditional_generation.ads_id.use:
            if self.hparams.conditional_generation.ads_id.value is not None:
                ads_id = torch.full(
                    (batch_size,),
                    self.hparams.conditional_generation.ads_id.value + 1,
                    dtype=torch.int64,
                   device=self.device
                )
            else:  #TODO -> sample from "ads_id_bincount.pt"
                ads_id = torch.randint(1, 83, (batch_size,), device=self.device)
        else:
            ads_id = torch.zeros((batch_size,), dtype=torch.int64, device=self.device)

        # create catalyst class tensor
        if self.hparams.conditional_generation.cat_class.use:
            if self.hparams.conditional_generation.cat_class.value is not None:
                cat_class = torch.full(
                    (batch_size,),
                    self.hparams.conditional_generation.cat_class.value + 1,
                    dtype=torch.int64,
                    device=self.device
                )
            else:
                # OC20: class 0~3, OC22: class 4 (oxides)
                # Use dataset parameter if provided, otherwise fall back to config
                _dataset = dataset if dataset is not None else self.hparams.sampling.dataset
                if _dataset == "oc20":
                    cat_class = torch.randint(1, 5, (batch_size,), device=self.device)  # 1~4 → class 0~3
                elif _dataset == "oc22":
                    cat_class = torch.full((batch_size,), 5, dtype=torch.int64, device=self.device)  # 5 → class 4
                else:  # both - sample from all classes (0~4)
                    cat_class = torch.randint(1, 6, (batch_size,), device=self.device)  # 1~5 → class 0~4
        else:
            cat_class = torch.zeros((batch_size,), dtype=torch.int64, device=self.device)

        # create binding energy tensor
        if self.hparams.conditional_generation.binding_energy.use:
            if self.hparams.conditional_generation.binding_energy.value is not None:
                binding_energy = torch.full(
                    (batch_size,),
                    self.hparams.conditional_generation.binding_energy.value,
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                binding_energy = torch.normal(
                    mean=-1.334,
                    std=2.016,
                    size=(batch_size,),
                    device=self.device
                ) # TODO-> sample from distribution file, DO NOT hardcoding
                binding_energy = torch.clamp(binding_energy, min=-10, max=10)
        else:
            binding_energy = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)

        # create token mask for visualization
        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True

        # create new samples from interpolant
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max(sample_lengths),
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            ads_id=ads_id,
            cat_class=cat_class,
            binding_energy=binding_energy,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
        )
        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1][token_mask]

        batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch)
        return out, batch, samples

    #####################################################################################################

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        try:
            # Clear cache for Equiformer SO3 embeddings
            self.autoencoder.encoder.mappingReduced.device = self.device
            self.autoencoder.encoder.mappingReduced.mask_indices_cache = None
            self.autoencoder.encoder.mappingReduced.rotate_inv_rescale_cache = None
            for rotation_module in self.autoencoder.encoder.SO3_rotation:
                rotation_module.mapping.device = self.device
                rotation_module.mapping.mask_indices_cache = None
                rotation_module.mapping.rotate_inv_rescale_cache = None
            log.info("Clear Equiformer checkpoint SO3 rotation mapping cache.")
        except AttributeError:
            pass

        if self.num_nodes_bincount is None:
            self.num_nodes_bincount = {}

        if not self.num_nodes_bincount:
            log.info("Loading num_nodes_bincount files...")
            for dataset, config in self.dataset_configs.items():
                path = config["num_nodes_bincount_path"]
                if os.path.exists(path):
                    self.num_nodes_bincount[dataset] = torch.nn.Parameter(
                        torch.load(path, map_location=self.device),
                        requires_grad=False,
                    )
                    log.info(f"Loaded {dataset} num_nodes_bincount from {path}")
                else:
                    log.warning(f"{dataset} num_nodes_bincount not found at {path}")

            # Create "both" bincount by combining oc20 and oc22
            if "oc20" in self.num_nodes_bincount and "oc22" in self.num_nodes_bincount:
                oc20_bc = self.num_nodes_bincount["oc20"]
                oc22_bc = self.num_nodes_bincount["oc22"]
                # Pad to same length if needed
                max_len = max(len(oc20_bc), len(oc22_bc))
                oc20_padded = torch.nn.functional.pad(oc20_bc, (0, max_len - len(oc20_bc)))
                oc22_padded = torch.nn.functional.pad(oc22_bc, (0, max_len - len(oc22_bc)))
                self.num_nodes_bincount["both"] = torch.nn.Parameter(
                    oc20_padded + oc22_padded,
                    requires_grad=False,
                )
                log.info("Created 'both' num_nodes_bincount by combining oc20 and oc22")

        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_oc20/struct_valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}