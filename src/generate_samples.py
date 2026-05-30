"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig
from tqdm import tqdm
import os

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.catalyst_generation import CatalystGenerationEvaluator
from src.models.ldm_module import LatentDiffusionLitModule
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    task_wrapper,
)
from torchmetrics import MeanMetric

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def generate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generates new samples from the given checkpoint."""

    assert cfg.ckpt_path

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Load checkpoint
    log.info(f"Loading model from {cfg.ckpt_path}")
    model = LatentDiffusionLitModule.load_from_checkpoint(
        cfg.ckpt_path,
        autoencoder_ckpt=cfg.autoencoder_ckpt,
        sampling=cfg.sampling,
        conditional_generation=cfg.conditional_generation,
        map_location="cuda",
        strict=False  # Allow missing/extra keys
    )
    model.eval()  # for sampling and inference

    model.setup(stage="test")

    if cfg.sampling.dataset == "oc20":
        train_dataset_path = os.path.join(
            cfg.sampling.data_dir,
            "oc20/is2re/processed/train.pt"
        )
        gt_catalyst_path = os.path.join(
            cfg.sampling.data_dir,
            f"{cfg.sampling.dataset}/s2ef/val_10k_catalyst.pkl"
        )

    elif cfg.sampling.dataset == "oc22":
        train_dataset_path = os.path.join(
            cfg.sampling.data_dir,
            "oc22/is2re/processed/train.pt"
        )
        gt_catalyst_path = None  # OC22: skip coverage / EMD
    else:
        train_dataset_path = None
        gt_catalyst_path = None

    generation_evaluator = CatalystGenerationEvaluator(
        train_dataset_path=train_dataset_path,
        gt_catalyst_path=gt_catalyst_path,
        stol=0.3,
        angle_tol=5,
        ltol=0.2,
        compute_novelty=cfg.sampling.compute_novelty)

    generation_metrics = {
        "struct_valid_rate": MeanMetric(),
        "unique_rate": MeanMetric(),
        "novel_rate": MeanMetric(),
        "sampling_time": MeanMetric(),
        "cov_recall": MeanMetric(),
        "cov_precision": MeanMetric(),
        "wdist_density": MeanMetric(),
        "wdist_num_elems": MeanMetric(),
    }

    # generate samples
    log.info(f"Generating {cfg.sampling.num_samples} samples from {cfg.sampling.dataset} num_nodes_bincounts...")

    for samples_so_far in tqdm(
        range(0, cfg.sampling.num_samples, cfg.sampling.batch_size),
        desc="Generating samples"
    ):
        # Calculate actual batch size (handle last batch)
        current_batch_size = min(
            cfg.sampling.batch_size,
            cfg.sampling.num_samples - samples_so_far
        )

        # Sample and decode
        out, batch, samples = model.sample_and_decode(
            num_nodes_bincount=model.num_nodes_bincount[cfg.sampling.dataset],
            batch_size=current_batch_size,
            cfg_scale=cfg.sampling.cfg_scale,
        )

        # Process each sample in batch
        start_idx = 0
        for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
            _atom_types = out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
            _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
            _tags = out["tags"].narrow(0, start_idx, num_atom).argmax(dim=1)
            _frac_coords = out["frac_coords"].narrow(0, start_idx, num_atom)
            _lengths = out["lengths"][idx_in_batch] * float(num_atom) ** (1 / 3)  # unscale lengths
            _angles = torch.rad2deg(out["angles"][idx_in_batch])  # convert to degrees

            generation_evaluator.append_pred_array({
                "atom_types": _atom_types.detach().cpu().numpy(),
                "tags": _tags.detach().cpu().numpy(),
                "frac_coords": _frac_coords.detach().cpu().numpy(),
                "lengths": _lengths.detach().cpu().numpy(),
                "angles": _angles.detach().cpu().numpy(),
                "sample_idx": samples_so_far + idx_in_batch,
            })
            start_idx = start_idx + num_atom

    # Computing metrics (AFTER all samples are generated)
    log.info("Computing metrics...")
    metrics_dict = generation_evaluator.get_metrics(
        save=cfg.sampling.visualize,
        save_dir=cfg.sampling.save_dir
    )

    # Update metrics with MeanMetric
    for k, v in metrics_dict.items():
        if k in generation_metrics:
            generation_metrics[k](v)

    # Log to wandb if available
    if logger and isinstance(logger[0], WandbLogger):
        pred_table = generation_evaluator.get_wandb_table()
        logger[0].experiment.log({"generated_samples": pred_table})
        for k, metric in generation_metrics.items():
            logger[0].experiment.log({f"generation/{k}": metric.compute().item()})

    # Print results
    log.info(f"Struct valid rate: {generation_metrics['struct_valid_rate'].compute():.3f}")
    log.info(f"Unique rate: {generation_metrics['unique_rate'].compute():.3f}")
    log.info(f"Novel rate: {generation_metrics['novel_rate'].compute():.3f}")
    log.info(f"Coverage Recall: {generation_metrics['cov_recall'].compute():.3f}")
    log.info(f"Coverage Precision: {generation_metrics['cov_precision'].compute():.3f}")
    log.info(f"Wasserstein (density): {generation_metrics['wdist_density'].compute():.3f}")
    log.info(f"Wasserstein (num_elems): {generation_metrics['wdist_num_elems'].compute():.3f}")

    object_dict = {
        "cfg": cfg,
        "model": model,
        "logger": logger,
        "evaluator": generation_evaluator,
    }

    return metrics_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_samples.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for generation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    generate(cfg)


if __name__ == "__main__":
    main()