"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from src.eval.catalyst import Catalyst
from src.tools.ase_notebook import AseView
from src.utils import joblib_map
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

ase_view = AseView(
    rotations="-75x, 45y, 10z",
    atom_font_size=16,
    axes_length=30,
    canvas_size=(400, 400),
    zoom=1.2,
    show_bonds=False,
    # uc_dash_pattern=(.6, .4),
    atom_show_label=True,
    canvas_background_opacity=0.0,
)
# ase_view.add_miller_plane(1, 0, 0, color="green")

# TODO-> True tags vs. Pred tags
# TODO-> tags\=2: composition check,RMSD(StructureMatcher) / structure validity

class CatalystReconstructionEvaluator:
    """Evaluator for catalyst reconstruction tasks. Can be used within a Lightning module, appending
    predictions and ground truths during training and computing metrics at the end of an epoch, or
    can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        stol (float): StructureMatcher tolerance for matching sites.
        angle_tol (float): StructureMatcher tolerance for matching angles.
        ltol (float): StructureMatcher tolerance for matching lengths.
    """

    def __init__(self, stol=0.5, angle_tol=10, ltol=0.3, device="cpu"):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []  # list of Dict[str, np.array] predictions
        self.gt_arrays_list = []  # list of Dict[str, np.array] ground truths
        self.pred_cat_list = []  # list of Catalyst predictions
        self.gt_cat_list = []  # list of Catalyst ground truths
        self.device = device

    def append_pred_array(self, pred: Dict[str, np.array]):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.array]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)

    def clear(self):
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_cat_list = []
        self.gt_cat_list = []

    def _arrays_to_catalysts(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Catalyst objects for evaluation."""
        self.pred_cat_list = joblib_map(
            partial(
                array_dict_to_catalyst,
                save=save,
                save_dir_name=f"{save_dir}/pred",
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Catalyst",
            total=len(self.pred_arrays_list),
        )
        self.gt_cat_list = joblib_map(
            partial(
                array_dict_to_catalyst,
                save=save,
                save_dir_name=f"{save_dir}/gt",
            ),
            self.gt_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    G.T. to Catalyst",
            total=len(self.gt_arrays_list),
        )

    def _get_metrics(self, pred, gt, is_valid):
        if not is_valid:
            return float("inf")
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            rms_dist = float("inf") if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return float("inf")

    def get_metrics(self, save: bool = False, save_dir: str = "") -> Dict[str, Any]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: self.rms_dists can be used to access RMSD per sample but is not returned.

        Returns:
            Dict: Dictionary of metrics, including match rate and avg. RMSD.
        """
        assert len(self.pred_arrays_list) == len(
            self.gt_arrays_list
        ), "Number of predictions and ground truths must match."

        # Convert predictions and ground truths to Catalyst objects
        self._arrays_to_catalysts(save, save_dir)

        # Check validity of predictions and ground truths
        validity = [
            c1.struct_valid and c2.struct_valid for c1, c2 in zip(self.pred_cat_list, self.gt_cat_list)
        ]

        self.rms_dists = []
        for i in tqdm(range(len(self.pred_cat_list)), desc="   Reconstruction eval"):
            self.rms_dists.append(
                self._get_metrics(self.pred_cat_list[i], self.gt_cat_list[i], validity[i])
            )
        self.rms_dists = torch.tensor(self.rms_dists, device=self.device)
        match_rate = (~torch.isinf(self.rms_dists)).long()
        if match_rate.sum() == 0:
            # No valid predictions --> return large RMSD for logging purposes
            return {
                "match_rate": match_rate,
                "rms_dist": torch.tensor([10.0] * len(match_rate), device=self.device),
            }
        else:
            return {
                "match_rate": match_rate,
                "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)],
            }

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = "") -> wandb.Table:
        """Create a wandb.Table object with the results of the evaluation."""
        log.info(f"Creating wandb table for {save_dir} with {len(self.pred_cat_list)} samples")
        pred_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample idx",
                "Num atoms",
                "RMSD",
                "Match?",
                "Struct valid?",
                "True atom types",
                "Pred atom types",
                "True tags",
                "Pred tags",
                "True lengths",
                "Pred lengths",
                "True angles",
                "Pred angles",
                "True 2D",
                "Pred 2D",
            ]
        )
        for idx in range(len(self.pred_cat_list)):
            sample_idx = self.gt_cat_list[idx].sample_idx
            assert sample_idx == self.pred_cat_list[idx].sample_idx

            num_atoms = len(self.gt_cat_list[idx].atom_types)

            rmsd = self.rms_dists[idx]

            match = rmsd != float("inf")

            true_atom_types = " ".join([str(int(t)) for t in self.gt_cat_list[idx].atom_types])

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_cat_list[idx].atom_types])

            true_tags = " ".join([str(int(t)) for t in self.gt_cat_list[idx].tags])

            pred_tags = " ".join([str(int(t)) for t in self.pred_cat_list[idx].tags])

            true_lengths = " ".join([f"{l:.2f}" for l in self.gt_cat_list[idx].lengths])

            true_angles = " ".join([f"{a:.2f}" for a in self.gt_cat_list[idx].angles])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_cat_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_cat_list[idx].angles])

            # 2D structures
            try:
                true_2d = ase_view.make_wandb_image(
                    self.gt_cat_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for true sample {sample_idx}.")
                true_2d = None
            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_cat_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                self.pred_cat_list[idx].struct_valid,
                true_atom_types,
                pred_atom_types,
                true_tags,
                pred_tags,
                true_lengths,
                pred_lengths,
                true_angles,
                pred_angles,
                true_2d,
                pred_2d,
            )
            # TODO What if Structures were to be saved as PDB, too? (for 3D)

        return pred_table


def array_dict_to_catalyst(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Catalyst:
    """Method to convert a dictionary of numpy arrays to a Catalyst object which is compatible with
    StructureMatcher (used for evaluations). Previously called 'safe_catalyst', as it return a
    generic catalyst if the input is invalid.

    Adapted from: https://github.com/facebookresearch/flowmm

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'frac_coords': Fractional coordinates of atoms.
            - 'atom_types': Atomic numbers of atoms.
            - 'tags': Atomic tag information: 0 - Fixed, sub-surface atoms, 1 - Free, surface atoms 2 - Free, adsorbate atoms
            - 'lengths': Lengths of the lattice vectors.
            - 'angles': Angles between the lattice vectors.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the catalyst as a CIF file.
        save_dir_name: Directory to save the CIF file.

    Returns:
        Catalyst: Catalyst object, optionally saved as a CIF file.
    """

    cat = Catalyst(x)
    if cat.struct_valid and save:
        os.makedirs(save_dir_name, exist_ok=True)
        cat.structure.to(os.path.join(save_dir_name, f"catalyst_{x['sample_idx']}.cif"))
    else:
        # returns an absurd catalyst
        cat = Catalyst(
            {
                "frac_coords": np.zeros_like(x["frac_coords"]),
                "atom_types": np.zeros_like(x["atom_types"]),
                "tags": np.zeros_like(x["tags"]),
                "lengths": 100 * np.ones_like(x["lengths"]),
                "angles": np.ones_like(x["angles"]) * 90,
                "sample_idx": x["sample_idx"],
            }
        )
    return cat
