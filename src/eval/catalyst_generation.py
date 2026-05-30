"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from functools import partial
from typing import Any, Dict, Literal, Tuple

import numpy as np
import torch
import wandb
import pickle
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from collections import defaultdict
from tqdm import tqdm

from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from src.eval.catalyst import Catalyst
from src.tools.ase_notebook import AseView
from src.utils import joblib_map, pylogger

warnings.filterwarnings("ignore", category=UserWarning)

log = pylogger.RankedLogger(__name__)

ase_view = AseView(
    rotations="-75x, 45y, 10z",
    atom_font_size=14,
    axes_length=30,
    canvas_size=(400, 400),
    zoom=1.2,
    show_bonds=False,
    # uc_dash_pattern=(.6, .4),
    atom_show_label=True,
    canvas_background_opacity=0.0,
)
# ase_view.add_miller_plane(1, 0, 0, color="green")
class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

CompScalerMeans = [21.194441759304013, 58.20212663122281, 37.0076848719188, 36.52738520455582, 13.350626389725019, 29.468922184630255, 28.71735137747704, 78.8868535524408, 50.16950217496375, 59.56764743604155, 19.020429484306277, 61.335572740454325, 47.14515893344343, 141.75135923307818, 94.60620029962553, 85.95794070476977, 34.07300576173523, 68.06189371516912, 637.9862061297893, 1817.2394155466848, 1179.2532094169414, 1127.2743149568837, 431.51034284549826, 909.1060025135899, 3.7744320927984534, 13.673707104881585, 9.899275012083132, 9.620186927095652, 3.8426065581251856, 9.96950217496375, 3.305461575640406, 5.483035282745288, 2.1775737071048815, 4.215114560306594, 0.8206087101824266, 3.732092798453359, 109.16732721121315, 179.5570323827936, 70.38970517158047, 136.0978305229613, 27.027545809538527, 119.16713388110198, 1.2721433060967857, 2.4614001837260617, 1.1892568776289631, 1.9844483610247092, 0.4691462290494881, 2.100143582306204, 1.4829869502174964, 1.9899951667472209, 0.5070082165297245, 1.7956250375970633, 0.2056251946617602, 1.745867568873852, 0.05650072498791687, 2.3618656355727405, 2.3053649105848235, 1.2829636137262992, 0.9995555685850794, 1.5150314161430642, 0.7731271145480909, 7.4648139197680035, 6.691686805219913, 4.010677272036105, 2.612307566507693, 3.303528274528758, 0.2739487675205413, 5.889753504108265, 5.615804736587724, 2.3244356612494683, 2.1426251769710905, 1.4464475592073465, 4.739246012566457, 14.578395360077332, 9.839149347510874, 9.413701584608935, 3.537059747455868, 8.550410826486225, 0.008119864668922184, 0.43286611889801835, 0.4247462542290962, 0.16687837041055423, 0.17139889490813626, 0.10898985016916385, 0.06283228612856452, 2.6573707104881583, 2.594538424359594, 1.219602938224228, 1.0596390454742999, 1.1120831319478008, 0.14842919284678588, 3.8473658772353794, 3.6989366843885936, 1.4541605082183982, 1.3862277372859781, 0.8018849685838569, 0.03542774287095215, 2.4474625422909617, 2.4120347994200095, 0.7745217539010397, 0.9145812330586208, 0.3198646689221846, 1.552730787820203, 6.910681488641856, 5.357950700821653, 3.615163570754227, 1.9072256165179793, 2.6702271628806185, 14.608536589568727, 34.83222477045747, 20.223688180890715, 22.47901710732293, 7.17674504190757, 18.641837024143584, 0.009066988883518605, 0.9185191396809959, 0.9094521507974755, 0.4368550481994018, 0.38905942883427047, 0.48375558240695804, 0.0012985909686158003, 0.21708593995837092, 0.21578734898975546, 0.08167977375391729, 0.08155386250705281, 0.06036340747305611, 116.32010633156113, 217.5905751570807, 101.27046882551957, 162.87154200548844, 41.920624308665566, 136.4664572257129]
CompScalerStds = [16.35781741152948, 20.189540126474725, 20.516298414514758, 16.816765336550194, 7.966591328222124, 22.270791076753067, 21.802116630115243, 12.804546460581966, 24.756629388687983, 13.930306216047477, 10.214535652334533, 27.801612936980938, 39.74031558353379, 54.269739685575814, 53.70466607591569, 42.852342044453444, 20.78341194242935, 56.28783510219931, 563.8004405882157, 732.0722574247563, 736.2122907972664, 606.351603075103, 272.62646060896407, 810.6156779688841, 3.0362262146833428, 3.2075174256751606, 4.0633818989245665, 2.9738244769894764, 1.7805586029644034, 5.643243225066782, 1.1994336274579853, 0.8939013979423364, 1.2297581799896975, 1.0066021334519983, 0.49129747526397105, 1.4159553146070951, 31.754756468836774, 28.054241463256226, 38.16336054795611, 25.83485338379922, 15.388376641904662, 39.67137484594156, 0.31988340032011076, 0.6833658037760536, 0.7464197945553585, 0.4881349085029781, 0.3176591553643101, 0.8601748146737138, 0.5864801661863596, 0.10048913710210677, 0.5836289120986499, 0.2811748167435902, 0.2468696279341553, 0.5007375747433073, 0.37237566669029587, 1.7235989187720187, 1.7058836077743305, 1.1558859351244697, 0.7677842566598179, 1.9203550253462733, 2.1289400248865182, 3.5326064169848332, 3.708508303762512, 2.8709941136664567, 1.6110681295257014, 4.310192504023775, 1.6644182118209292, 6.228287671164213, 6.1200848808512305, 3.1986202996110302, 2.4492978142248867, 4.030497343977163, 3.662028270049814, 6.8192125550358345, 6.614243783887738, 4.334987449618594, 2.568319610320196, 5.9494890200106925, 0.08974370432893491, 0.4954725441517777, 0.494304434278516, 0.2309340434963803, 0.2072873961103969, 0.31162647950590266, 0.39805702757060923, 1.8111691089355726, 1.7973395144505941, 0.9486995373104102, 0.7538753151875139, 1.5233177017753785, 0.7952606701778913, 3.711190225170556, 3.638721437232604, 1.7171165424006831, 1.4307904413917036, 2.1047820817622904, 0.49193748323158065, 4.064840532426175, 4.035286619587313, 1.4858577214526643, 1.5799117659864677, 1.6130080156145745, 1.555249156140194, 4.776932951077492, 4.569790780459629, 2.224617778217326, 1.7217507416156546, 2.5969733650703763, 7.215001918238936, 19.252513469778584, 18.775394044177858, 9.447222764774764, 6.7467931836261235, 11.106825644766616, 0.27206794253092115, 1.6449321034573106, 1.6236282792648686, 0.8506917026741503, 0.7020945355184042, 1.2281895279350408, 0.04134438177238229, 0.5508855867341717, 0.5486095551438679, 0.24239297524046477, 0.2127779137935831, 0.3036750942874694, 80.06063945615361, 21.345794811194104, 80.16475677581042, 52.58533928558554, 35.40836791039412, 85.980205895116]

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

class CatalystGenerationEvaluator:
    """Evaluator for catalyst generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        train_dataset_path=None,
        gt_catalyst_path=None,
        stol=0.3,
        angle_tol=5,
        ltol=0.2,
        device="cpu",
        compute_novelty=False,
    ):
        self.train_dataset_path = train_dataset_path
        self.dataset_struct_list = None  # lazy loading
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []
        self.pred_cat_list = []
        self.device = device
        self.compute_novelty = compute_novelty
        self.gt_cat_list = None
        if gt_catalyst_path is not None:
            with open(gt_catalyst_path, 'rb') as f:
                self.gt_cat_list = pickle.load(f)

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_cat_list = []

    def _arrays_to_catalysts(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Catalyst objects for evaluation."""
        self.pred_cat_list = joblib_map(
            partial(
                array_dict_to_catalyst,
                save=save,
                save_dir_name=save_dir,
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Catalyst",
            total=len(self.pred_arrays_list),
        )

    def _load_train_structures(self):
        """Load training slab-only structures from .pt file for novelty evaluation."""
        if self.dataset_struct_list is None and self.train_dataset_path is not None:
            # Go up 4 levels: train.pt -> processed -> is2re -> oc20 -> data
            _data_root = self.train_dataset_path
            for _ in range(4):
                _data_root = os.path.dirname(_data_root)
            structures_path = os.path.join(_data_root, "fps", "train_oc20_structures.pkl")

            if os.path.exists(structures_path):
                log.info(f"Loading pre-processed slab structures from {structures_path}...")
                with open(structures_path, 'rb') as f:
                    data = pickle.load(f)
                self.dataset_struct_list = data["structures"]
                self.train_by_comp = defaultdict(list, data["train_by_comp"])
                log.info(f"Loaded {len(self.dataset_struct_list)} structures, {len(self.train_by_comp)} element groups")

            else:
                log.info(f"Converting from {self.train_dataset_path}...")
                train_data = torch.load(self.train_dataset_path, map_location='cpu')

                self.dataset_struct_list = joblib_map(
                    self._data_to_slab_structure,
                    train_data,
                    n_jobs=-4,
                    inner_max_num_threads=1,
                    desc="    Converting train data to slab structures",
                    total=len(train_data),
                )
                self.dataset_struct_list = [s for s in self.dataset_struct_list if s is not None]

                self.train_by_comp = defaultdict(list)
                for i, struct in tqdm(enumerate(self.dataset_struct_list), desc="    Indexing by element set", total=len(self.dataset_struct_list)):
                    comp_key = frozenset(str(el) for el in struct.composition.elements)
                    self.train_by_comp[comp_key].append(i)

                log.info(f"Saving to {structures_path}...")
                with open(structures_path, 'wb') as f:
                    pickle.dump({"structures": self.dataset_struct_list, "train_by_comp": dict(self.train_by_comp)}, f)

    @staticmethod
    def _data_to_slab_structure(data):
        """Convert single PyG data object to slab-only PyMatGen Structure (exclude adsorbate, tag=2)."""
        from pymatgen.core.structure import Structure
        try:
            lattice = data['lattices'].squeeze().numpy() if hasattr(data['lattices'], 'numpy') else data['lattices']
            atom_types = data['atom_types'].numpy() if hasattr(data['atom_types'], 'numpy') else data['atom_types']
            frac_coords = data['frac_coords'].numpy() if hasattr(data['frac_coords'], 'numpy') else data['frac_coords']

            if 'tags' in data:
                tags = data['tags'].numpy() if hasattr(data['tags'], 'numpy') else data['tags']
                slab_mask = tags != 2
                atom_types = atom_types[slab_mask]
                frac_coords = frac_coords[slab_mask]

            if len(atom_types) == 0:
                return None

            struct = Structure(
                lattice=lattice,
                species=atom_types,
                coords=frac_coords,
                coords_are_cartesian=False,
            )
            return struct
        except Exception:
            return None

    def _get_novelty(self, struct):
        """Quick novelty check by comparing ONLY with training structures of the same element set."""
        comp_key = frozenset(str(el) for el in struct.composition.elements)
        candidates = self.train_by_comp.get(comp_key, [])

        if len(candidates) == 0:
            return True

        for idx in candidates:
            try:
                if self.matcher.fit(struct, self.dataset_struct_list[idx], skip_structure_reduction=True):
                    return False
            except Exception:
                continue
        return True
    
    def _get_coverage(self, struc_cutoff=0.4, comp_cutoff=10.0):
        """Coverage (precision/recall)"""
        pred_struct_fps = [c.struct_fp for c in self.pred_cat_list if c.struct_valid and c.struct_fp is not None]
        pred_comp_fps = [c.comp_fp for c in self.pred_cat_list if c.struct_valid and c.comp_fp is not None]
        gt_struct_fps = [c.struct_fp for c in self.gt_cat_list]
        gt_comp_fps = [c.comp_fp for c in self.gt_cat_list]

        # filtering 'None'
        num_gen = len(pred_struct_fps)
        valid_pairs = [(s, c) for s, c in zip(pred_struct_fps, pred_comp_fps) 
                   if s is not None and c is not None]
        pred_struct_fps = [p[0] for p in valid_pairs]
        pred_comp_fps = [p[1] for p in valid_pairs]

        pred_comp_fps = CompScaler.transform(pred_comp_fps)
        gt_comp_fps = CompScaler.transform(gt_comp_fps)
        
        pred_struct_fps = np.array(pred_struct_fps)
        pred_comp_fps = np.array(pred_comp_fps)
        gt_struct_fps = np.array(gt_struct_fps)
        gt_comp_fps = np.array(gt_comp_fps)
        
        struct_dist = cdist(pred_struct_fps, gt_struct_fps)
        comp_dist = cdist(pred_comp_fps, gt_comp_fps)
        
        # Recall: fraction of ground-truth structures covered
        struct_recall_dist = struct_dist.min(axis=0)
        comp_recall_dist = comp_dist.min(axis=0)
        cov_recall = np.mean(
            (struct_recall_dist <= struc_cutoff) & (comp_recall_dist <= comp_cutoff)
        )
        
        # Precision: fraction of generated structures matched to ground truth
        struct_precision_dist = struct_dist.min(axis=1)
        comp_precision_dist = comp_dist.min(axis=1)
        cov_precision = np.sum(
            (struct_precision_dist <= struc_cutoff) & (comp_precision_dist <= comp_cutoff)
        ) / num_gen
        
        return {
            'cov_recall': cov_recall,
            'cov_precision': cov_precision,
        }
    
    def _get_emd(self):
        """Wasserstein distance for density and num_elems distributions"""
        pred_densities = [c.structure.density for c in self.pred_cat_list if c.struct_valid]
        pred_num_elems = [len(set(c.structure.species)) for c in self.pred_cat_list if c.struct_valid]
        
        gt_densities = [c.structure.density for c in self.gt_cat_list]
        gt_num_elems = [len(set(c.structure.species)) for c in self.gt_cat_list]
        
        return {
            'wdist_density': wasserstein_distance(pred_densities, gt_densities),
            'wdist_num_elems': wasserstein_distance(pred_num_elems, gt_num_elems),
        }

    def get_metrics(self, save: bool = False, save_dir: str = ""):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."

        # Convert predictions and ground truths to Catalyst objects
        self._arrays_to_catalysts(save, save_dir)

        # Compute validity metrics
        metrics_dict = {
            "struct_valid_rate": torch.tensor(
                [c.struct_valid for c in self.pred_cat_list], device=self.device
            ),
        }

        # Compute uniqueness (slab-only, excluding adsorbate tag=2)
        valid_slab_structs = []
        for c in self.pred_cat_list:
            if c.struct_valid:
                slab_sites = [site for site in c.structure if site.properties.get('tags', 0) != 2]
                if slab_sites:
                    slab_struct = Structure.from_sites(slab_sites)
                    valid_slab_structs.append(slab_struct)
        unique_struct_groups = self.matcher.group_structures(valid_slab_structs)
        valid_structs = valid_slab_structs
        if len(valid_structs) > 0:
            metrics_dict["unique_rate"] = torch.tensor(
                len(unique_struct_groups) / len(valid_structs), device=self.device
            )
        else:
            metrics_dict["unique_rate"] = torch.tensor(0.0, device=self.device)

        # Compute novelty (slow to compute)
        if self.compute_novelty:
            self._load_train_structures()
            struct_is_novel = joblib_map(
                self._get_novelty,
                [group[0] for group in unique_struct_groups],
                n_jobs=-4,
                inner_max_num_threads=1,
                desc="    Computing novelty",  # tqdm desc
                total=len(unique_struct_groups),
            )
            
            metrics_dict["novel_rate"] = torch.tensor(
                sum(struct_is_novel) / len(struct_is_novel), device=self.device
            )

        # Compute coverage
        if self.gt_cat_list is not None:
            cov_metrics = self._get_coverage()
            metrics_dict.update({
                'cov_recall': torch.tensor(cov_metrics['cov_recall'], device=self.device),
                'cov_precision': torch.tensor(cov_metrics['cov_precision'], device=self.device),
            })
            
            # Compute EMD
            emd_metrics = self._get_emd()
            metrics_dict.update({
                'wdist_density': torch.tensor(emd_metrics['wdist_density'], device=self.device),
                'wdist_num_elems': torch.tensor(emd_metrics['wdist_num_elems'], device=self.device),
            })

        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log catalyst structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Struct valid?",
                "Pred atom types",
                "Pred tags",
                "Pred lengths",
                "Pred angles",
                "Pred 2D",
            ]
        )

        for idx in range(len(self.pred_cat_list)):
            sample_idx = self.pred_cat_list[idx].sample_idx

            num_atoms = len(self.pred_cat_list[idx].atom_types)

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_cat_list[idx].atom_types])

            pred_tags = " ".join([str(int(t)) for t in self.pred_cat_list[idx].tags])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_cat_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_cat_list[idx].angles])

            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_cat_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                self.pred_cat_list[idx].struct_valid,
                pred_atom_types,
                pred_tags,
                pred_lengths,
                pred_angles,
                pred_2d,
            )

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
            - 'tags': Tags of atoms.
            - 'lengths': Lengths of the lattice vectors.
            - 'angles': Angles between the lattice vectors.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the catalyst as a CIF file.
        save_dir_name: Directory to save the CIF file.

    Returns:
        Catalyst: Catalyst object, optionally saved as a CIF file.
    """
    cat = Catalyst(x)

    if save and cat.struct_valid:
        os.makedirs(save_dir_name, exist_ok=True)

        atoms = AseAtomsAdaptor.get_atoms(cat.structure)
        atoms.set_tags(cat.tags)
        write(
            os.path.join(save_dir_name, f"catalyst_{x['sample_idx']}.extxyz"),
            atoms,
            format='extxyz'
        )

    return cat
