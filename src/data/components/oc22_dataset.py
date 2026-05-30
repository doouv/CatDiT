import os, subprocess, warnings, pickle, torch, gc, lzma, glob
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from typing import Optional, Callable, List
from torch_geometric.data import Dataset, Data
from collections import defaultdict
from fairchem.core.datasets.oc22_lmdb_dataset import OC22LmdbDataset
from ase.io import read as ase_read
from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from src.data.components.preprocessing_utils import cart_to_frac_coords, lattice_matrix_to_params, \
    lattice_params_to_matrix, compute_lattice_info, create_base_data_object
from src.data.components.adsorbate_mapping import ADS_SYMBOLS_TO_ID, ADS_SYMBOLS_ALIAS
from src.utils import pylogger

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def check_dissociation(pos, tags, atomic_numbers, cell, mult=1.2):
    """Check if multi-atom adsorbate (tag=2) is dissociated.
    Returns True if dissociated."""
    ads_mask = tags == 2
    n_adsorbate = int(ads_mask.sum())
    if n_adsorbate <= 1:
        return False

    ads_indices = np.where(ads_mask)[0]
    cell = np.array(cell).reshape(3, 3)
    from ase.data import chemical_symbols
    symbols = [chemical_symbols[int(z)] for z in atomic_numbers[ads_indices]]
    ads_atoms = Atoms(symbols=symbols, positions=pos[ads_indices],
                      cell=cell, pbc=True)

    cutoffs = [c * mult for c in natural_cutoffs(ads_atoms)]
    i_list, j_list, d_list = neighbor_list('ijd', ads_atoms, cutoff=max(cutoffs) * 2)

    n_ads = len(ads_atoms)
    adj = np.zeros((n_ads, n_ads), dtype=bool)
    for i, j, d in zip(i_list, j_list, d_list):
        if d <= cutoffs[i] + cutoffs[j]:
            adj[i, j] = True
            adj[j, i] = True

    n_fragments, _ = connected_components(csr_matrix(adj), directed=False)
    return n_fragments > 1


# =============================================================================
# Shared Helper Functions
# =============================================================================

def _download_and_extract(
        data_url: str,
        metadata_url: str,
        raw_dir: str,
        metadata_path: str,
        tar_filename: str,
        unstructured_dirname: str,
        structured_dirname: str,
        dataset_type: str,
        data_subpath: str,
        readme_name: str
) -> None:
    """Shared download logic for OC22 datasets"""
    tar_path = os.path.join(raw_dir, tar_filename)
    structured_path = os.path.join(raw_dir, structured_dirname)
    unstructured_path = os.path.join(raw_dir, unstructured_dirname)

    # Download shared metadata
    if not os.path.exists(metadata_path):
        log.info("[↓] Downloading OC22 metadata (shared)...")
        subprocess.run(["wget", "--progress=bar:force:noscroll", metadata_url,
                        "-O", metadata_path], check=True)

    if not os.path.exists(unstructured_path) or os.path.exists(structured_path):
        if not os.path.exists(tar_path):
            log.info(f"[↓] Downloading OC22-{dataset_type} LMDBs...")
            subprocess.run(["wget", "--progress=bar:force:noscroll", data_url,
                            "-O", tar_path], check=True)
        else:
            log.info(".tar file already exists. Skipping download.")

        log.info(f"Uncompressing OC22-{dataset_type} LMDBs...")
        subprocess.run(["tar", "-xvzf", tar_path, "-C", raw_dir], check=True)
        os.remove(tar_path)
    else:
        log.info(f"OC22-{dataset_type} LMDBs already exist. Skipping download and extraction.")

    # Restructure directory
    if not os.path.exists(structured_path):
        subprocess.run(
            f"mv {os.path.join(unstructured_path, data_subpath)}/* {raw_dir}",
            shell=True, check=True
        )
        subprocess.run(
            f"mv {os.path.join(unstructured_path, readme_name)} {raw_dir}",
            shell=True, check=True
        )
        subprocess.run(["rm", "-r", unstructured_path], check=True)


def _process_lmdb_split(
        lmdb_path: str,
        metadata: dict,
        split_name: str,
        process_fn: Callable,
        desc: str
) -> List[Data]:
    """Shared LMDB processing logic"""
    data_list = []

    if os.path.exists(lmdb_path):
        dataset_config = {'format': 'oc22_lmdb', 'src': lmdb_path}
        lmdb_dataset = OC22LmdbDataset(dataset_config)

        for idx in tqdm(range(len(lmdb_dataset)), desc=desc):
            try:
                data = process_fn(lmdb_dataset[idx], metadata, split_name)
                if data is not None:
                    data_list.append(data)
            except Exception as e:
                log.warning(f"Error processing {split_name}[{idx}]: {e}")
                continue

        del lmdb_dataset
        gc.collect()

    return data_list


# =============================================================================
# S2EF Dataset
# =============================================================================

class OC22s2ef(Dataset):
    """
    The OC22-s2ef dataset from Fairchem.
    Saves as single .pt files (train.pt, val.pt, test.pt) for faster loading.
    """
    data_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/s2ef_total_train_val_test_lmdbs.tar.gz"
    metadata_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl"

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.data_list = self._load_split()

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "train/data.0000.lmdb",
            "val_id/data.0000.lmdb",
            "val_ood/data.0000.lmdb",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def metadata_path(self) -> str:
        """Metadata is shared at oc22/ level (for nads filtering)"""
        return os.path.join(os.path.dirname(self.root), "oc22_metadata.pkl")

    def download(self) -> None:
        """Download S2EF LMDBs and shared metadata"""
        _download_and_extract(
            data_url=self.data_url,
            metadata_url=self.metadata_url,
            raw_dir=self.raw_dir,
            metadata_path=self.metadata_path,
            tar_filename="s2ef_total_train_val_test_lmdbs.tar.gz",
            unstructured_dirname="s2ef_total_train_val_test_lmdbs",
            structured_dirname="train",
            dataset_type="S2EF",
            data_subpath="data/oc22/s2ef-total",
            readme_name="README_s2ef_total.md"
        )

    def process(self) -> None:
        """
        Process S2EF LMDB files into 3 splits:
        - train.pt: train (10 snapshots per trajectory)
        - val.pt: val_id (5 snapshots per trajectory)
        - test.pt: val_ood (5 snapshots per trajectory)

        Open with OC22LmdbDataset, deprecated in FairChem v2.
        """
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        log.info(f"Loaded metadata: {len(metadata):,} entries")

        os.makedirs(self.processed_dir, exist_ok=True)
        train_file = os.path.join(self.processed_dir, "train.pt")

        if os.path.exists(train_file):
            log.info("Processed file (train.pt) already exists. Skipping train processing.\n")
        else:
            # Process train → train.pt
            log.info("Processing train split (train)...")
            train_data = self._process_split_with_sampling(
                lmdb_path=os.path.join(self.raw_dir, 'train'),
                metadata=metadata,
                split_name='train',
                max_snapshots_per_trajectory=10
            )
            torch.save(train_data, train_file)
            log.info(f"Train: {len(train_data):,} samples")
            del train_data
            gc.collect()

        # Process val_id → val.pt
        log.info("Processing val split (val_id)...")
        val_data = self._process_split_with_sampling(
            lmdb_path=os.path.join(self.raw_dir, 'val_id'),
            metadata=metadata,
            split_name='val_id',
            max_snapshots_per_trajectory=5
        )
        torch.save(val_data, os.path.join(self.processed_dir, "val.pt"))
        log.info(f"Val:   {len(val_data):,} samples")
        del val_data
        gc.collect()

        # Process val_ood → test.pt
        log.info("Processing test split (val_ood)...")
        test_data = self._process_split_with_sampling(
            lmdb_path=os.path.join(self.raw_dir, 'val_ood'),
            metadata=metadata,
            split_name='val_ood',
            max_snapshots_per_trajectory=5
        )
        torch.save(test_data, os.path.join(self.processed_dir, "test.pt"))
        log.info(f"Test:  {len(test_data):,} samples")
        del test_data
        gc.collect()

        log.info("Processing Complete!\n")

    def _process_split_with_sampling(
        self,
        lmdb_path: str,
        metadata: dict,
        split_name: str,
        max_snapshots_per_trajectory: int
    ) -> List[Data]:
        """Process LMDB split with trajectory sampling"""
        if not os.path.exists(lmdb_path):
            log.warning(f"{split_name} LMDB not found: {lmdb_path}")
            return []

        dataset_config = {'format': 'oc22_lmdb', 'src': lmdb_path}
        lmdb_dataset = OC22LmdbDataset(dataset_config)

        # Collect trajectories (only store indices)
        trajectories = defaultdict(list)
        for idx in tqdm(range(len(lmdb_dataset)), desc=f"  Filtering {split_name}"):
            try:
                dd = lmdb_dataset[idx]
                if dd['nads'] in [0, 1]:
                    trajectories[dd['sid']].append(idx)
            except:
                continue

        log.info(f"  Found {len(trajectories):,} trajectories in {split_name}")

        # Sample indices
        selected_indices = []
        for idxs in trajectories.values():
            if len(idxs) <= max_snapshots_per_trajectory:
                selected_indices.extend(idxs)
            else:
                s = np.linspace(0, len(idxs) - 1, max_snapshots_per_trajectory, dtype=int)
                selected_indices.extend([idxs[i] for i in s])

        log.info(f"  Selected {max_snapshots_per_trajectory} snapshots per trajectory; {len(selected_indices):,} samples")

        # Process selected samples
        data_list = []
        for idx in tqdm(selected_indices, desc=f"  Processing {split_name}"):
            try:
                data = self._process_single_item(lmdb_dataset[idx], metadata, split_name)
                if data is not None:
                    data_list.append(data)
            except Exception as e:
                log.warning(f"Error processing {split_name}[{idx}]: {e}")
                continue

        del lmdb_dataset
        gc.collect()

        return data_list

    def _process_single_item(self, datadict, metadata, split):
        """Process single item from LMDB"""
        if datadict['nads'] not in [0, 1]:
            return None

        # Extract basic attributes
        atom_types = datadict['atomic_numbers']
        cell = datadict['cell']
        num_atoms = datadict['natoms']
        tags = datadict['tags']
        pos = datadict['pos']

        # Filter dissociated adsorbates
        if check_dissociation(pos.numpy(), tags.numpy(), atom_types.numpy(), cell.numpy()):
            return None

        # Compute lattice info
        lattice_info = compute_lattice_info(cell, num_atoms, pos)

        # Create base Data object with common fields
        data = create_base_data_object(
            atom_types, pos, lattice_info['frac_coords'], tags,
            cell, lattice_info, num_atoms,
        )

        # Total energy of the structure
        data.energy = torch.tensor([datadict['y']], dtype=torch.float)  # TODO-> binding energy
        data.sid = datadict['sid']
        return data

    def _load_split(self):
        """Load the requested split"""
        split_file = os.path.join(self.processed_dir, f"{self.split}.pt")
        return torch.load(split_file)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# =============================================================================
# IS2RE Dataset
# =============================================================================

class OC22is2re(Dataset):
    """
    The OC22-is2re dataset from Fairchem.
    Saves as single .pt files (train.pt, val.pt, test.pt) for faster loading.
    """
    data_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz"
    metadata_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl"

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            force_reload: bool = False,
    ) -> None:
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.data_list = self._load_split()

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "train/data.0000.lmdb",
            "val_id/data.0000.lmdb",
            "val_ood/data.0000.lmdb",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def metadata_path(self) -> str:
        """Metadata is shared at oc22/ level (for nads filtering)"""
        return os.path.join(os.path.dirname(self.root), "oc22_metadata.pkl")

    def download(self) -> None:
        """Download IS2RE LMDBs and shared metadata"""
        _download_and_extract(
            data_url=self.data_url,
            metadata_url=self.metadata_url,
            raw_dir=self.raw_dir,
            metadata_path=self.metadata_path,
            tar_filename="is2res_total_train_val_test_lmdbs.tar.gz",
            unstructured_dirname="is2res_total_train_val_test_lmdbs",
            structured_dirname="train",
            dataset_type="IS2RE",
            data_subpath="data/oc22/is2re-total",
            readme_name="README_is2re_total.md"
        )

    def process(self) -> None:
        """
        Process IS2RE LMDB files into 3 splits:
        - train.pt: train
        - val.pt: val_id
        - test.pt: val_ood

        Open with OC22LmdbDataset, deprecated in FairChem v2.
        """
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        log.info(f"Loaded metadata: {len(metadata):,} entries")
        os.makedirs(self.processed_dir, exist_ok=True)

        # Process train → train.pt (use shared function)
        log.info("Processing train split (train)...")
        train_data = _process_lmdb_split(
            lmdb_path=os.path.join(self.raw_dir, 'train'),
            metadata=metadata,
            split_name='train',
            process_fn=self._process_single_item,
            desc=f"Processing train"
        )
        torch.save(train_data, os.path.join(self.processed_dir, "train.pt"))
        log.info(f"Train: {len(train_data):,} samples")
        del train_data
        gc.collect()

        # Process val_id → val.pt (use shared function)
        log.info("Processing val split (val_id)...")
        val_data = _process_lmdb_split(
            lmdb_path=os.path.join(self.raw_dir, 'val_id'),
            metadata=metadata,
            split_name='val_id',
            process_fn=self._process_single_item,
            desc="Processing val_id"
        )
        torch.save(val_data, os.path.join(self.processed_dir, "val.pt"))
        log.info(f"Val:   {len(val_data):,} samples")
        del val_data
        gc.collect()

        # Process val_ood → test.pt (use shared function)
        log.info("Processing test split (val_ood)...")
        test_data = _process_lmdb_split(
            lmdb_path=os.path.join(self.raw_dir, 'val_ood'),
            metadata=metadata,
            split_name='val_ood',
            process_fn=self._process_single_item,
            desc="Processing val_ood"
        )
        torch.save(test_data, os.path.join(self.processed_dir, "test.pt"))
        log.info(f"Test:  {len(test_data):,} samples")
        del test_data
        gc.collect()

        log.info("Processing Complete!\n")

    def _process_single_item(self, datadict, metadata, split):
        """Process single item from LMDB"""

        # Remove multiple adsorbates systems and clean slab
        if datadict['nads'] == 1:
            # Extract basic attributes
            atom_types = datadict['atomic_numbers']
            cell = datadict['cell']
            num_atoms = datadict['natoms']
            tags = datadict['tags']
            pos = datadict['pos']

            # Filter dissociated adsorbates
            if check_dissociation(pos.numpy(), tags.numpy(), atom_types.numpy(), cell.numpy()):
                return None

            # Compute lattice info
            lattice_info = compute_lattice_info(cell, num_atoms, pos)

            # Create base Data object with common fields
            data = create_base_data_object(
                atom_types, pos, lattice_info['frac_coords'], tags,
                cell, lattice_info, num_atoms,
            )

            # Add conditioning info from metadata
            sid = datadict['sid']
            meta_info = metadata.get(sid)
            # TODO -> binding energy
            ads_symbols = meta_info['ads_symbols']
            ads_symbols = ADS_SYMBOLS_ALIAS.get(ads_symbols, ads_symbols)
            ads_id = ADS_SYMBOLS_TO_ID.get(ads_symbols, -1)
            data.ads_id = torch.tensor([ads_id], dtype=torch.long)
            data.surface = torch.tensor(meta_info['miller_index'], dtype=torch.float)
            data.cat_class = torch.tensor([4], dtype=torch.long)  # 4 for oxides
            return data
        else:
            return None

    def _load_split(self):
        """Load the requested split"""
        split_file = os.path.join(self.processed_dir, f"{self.split}.pt")
        return torch.load(split_file)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]