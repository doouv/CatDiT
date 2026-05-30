import os, subprocess, warnings, pickle, torch, gc, lzma, glob
from tqdm import tqdm
import multiprocessing as mp
from typing import Optional, Callable, List
from torch_geometric.data import Dataset, Data
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from ase.io import read as ase_read
from src.data.components.preprocessing_utils import cart_to_frac_coords, lattice_matrix_to_params, \
    lattice_params_to_matrix,compute_lattice_info, create_base_data_object
from src.utils import pylogger

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

"""     Expected structure:
        ${paths.data_dir}/oc20/
        ├── oc20_metadata.pkl              
        ├── oc20_num_nodes_bincount.pt     # is2re
        ├── s2ef/
        │   ├── raw/
        │   │   ├── train_2M/              # 2M structures
        │   │   ├── val_id/                # 1M structures  
        │   │   ├── val_ood_both/          # 1M structures
        │   │   └── README*.md
        │   └── processed/
        │       ├── train.pt               # 2M (train_2M + val_ood_both)
        │       ├── val.pt                 # 325K (half of val_id)
        │       └── test.pt                # 325K (half of val_id)
        └── is2re/
            ├── raw/
            │   ├── train/
            │   ├── val_id/
            │   ├── val_ood_ads/
            │   ├── val_ood_cat/
            │   ├── val_ood_both/
            │   ├── test_id/
            │   ├── test_ood_ads/
            │   ├── test_ood_cat/
            │   └── test_ood_both/
            └── processed/
                ├── train.pt               # train + val_ood_ads + val_ood_cat
                ├── val.pt                 # val_id
                └── test.pt                # val_ood_both

    Metadata is shared at: root/../oc20_metadata.pkl """

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


# =============================================================================
# S2EF Dataset
# =============================================================================

class OC20s2ef(Dataset):
    """
    The OC20-s2ef-2M dataset from Fairchem.
    Saves as single .pt files (train.pt, val.pt, test.pt) for faster loading.
    """

    metadata_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"

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
            "train_2M/0.extxyz",
            "train_2M/0.txt",
            "val_id/0.extxyz",
            "val_id/0.txt",
            "val_ood_both/0.extxyz",
            "val_ood_both/0.txt",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def metadata_path(self) -> str:
        """Metadata is shared at oc20/ level (for anomaly filtering)"""
        return os.path.join(os.path.dirname(self.root), "oc20_metadata.pkl")

    def download(self) -> None:
        """Download S2EF data and shared metadata"""

        # Download shared metadata
        if not os.path.exists(self.metadata_path):
            log.info("[↓] Downloading OC20 metadata (shared)...")
            subprocess.run([
                "wget", "--progress=bar:force:noscroll",
                self.metadata_url, "-O", self.metadata_path
            ], check=True)

        # Download S2EF datasets
        datasets = [
            {
                "name": "train_2M",
                "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar"
            },
            {
                "name": "val_id",
                "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar"
            },
            {
                "name": "val_ood_both",
                "url": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar"
            },
        ]

        for ds in datasets:
            tar_path = os.path.join(self.raw_dir, f"s2ef_{ds['name']}.tar")
            structured_path = os.path.join(self.raw_dir, ds['name'])
            unstructured_path = os.path.join(self.raw_dir, f"s2ef_{ds['name']}")
            readme = f"README_s2ef_{ds['name']}.md"

            if not os.path.exists(structured_path) and not os.path.exists(unstructured_path):
                if not os.path.exists(tar_path):
                    log.info(f"[↓] Downloading OC20-S2EF-{ds['name']}...")
                    subprocess.run([
                        "wget", "--progress=bar:force:noscroll",
                        ds['url'], "-O", tar_path
                    ], check=True)
                else:
                    log.info(f".tar file already exists. Skipping download.")

                log.info(f"Uncompressing OC20-S2EF-{ds['name']}...")
                subprocess.run(["tar", "-xvf", tar_path, "-C", self.raw_dir], check=True)
                os.remove(tar_path)
            else:
                log.info(f"OC20-S2EF-{ds['name']} already exist. Skipping download and extraction.")

            # Restructure directory
            if not os.path.exists(structured_path):
                os.makedirs(structured_path, exist_ok=True)
                subprocess.run(
                    f"mv {os.path.join(unstructured_path, readme)} {self.raw_dir}",
                    shell=True, check=True
                )
                subprocess.run(
                    f"mv {unstructured_path}/s2ef_{ds['name']}/* {structured_path}/",
                    shell=True, check=True
                )
                subprocess.run(["rm", "-r", unstructured_path], check=True)

            if os.path.exists(structured_path):
                self._uncompress_parallel(structured_path, num_workers=64)

    def _uncompress_parallel(self, split_dir: str, num_workers: int = 64):
        """Decompress all .xz files in parallel"""
        xz_files = glob.glob(os.path.join(split_dir, "*.xz"))

        if not xz_files:
            log.info(f"No .xz files in {split_dir}, skipping uncompress")
            return

        files_to_decompress = []
        for xz_file in xz_files:
            uncompressed = xz_file[:-3]
            if not os.path.exists(uncompressed):
                files_to_decompress.append(xz_file)

        if not files_to_decompress:
            log.info(f"All files already uncompressed in {split_dir}")
            return

        log.info(f"Uncompressing {len(files_to_decompress)} files using {num_workers} workers...")

        with mp.Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(self._decompress_single_file, files_to_decompress),
                total=len(files_to_decompress),
                desc=f"Uncompressing {os.path.basename(split_dir)}"
            ))

    @staticmethod
    def _decompress_single_file(xz_file: str):
        """Decompress single .xz file"""
        output_file = xz_file[:-3]

        if os.path.exists(output_file):
            return

        try:
            with open(xz_file, 'rb') as f:
                contents = lzma.decompress(f.read())
            with open(output_file, 'wb') as f:
                f.write(contents)
            os.remove(xz_file)
        except FileNotFoundError:
            pass

    def process(self) -> None:
        """
        Process S2EF extxyz files into 3 single .pt files.

        Final split strategy:
        - train.pt: train_2M + val_ood_both (~2M)
        - val.pt:   val_id first 50% (~325K)
        - test.pt:  val_id last 50% (~325K)
        """

        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        log.info(f"Loaded metadata: {len(metadata):,} entries")

        os.makedirs(self.processed_dir, exist_ok=True)

        # Process train_2M + val_ood_both → train.pt
        log.info("Processing train split (train_2M + val_ood_both)...")

        train_data = []
        for source in ['train_2M', 'val_ood_both']:
            split_dir = os.path.join(self.raw_dir, source)
            extxyz_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.extxyz')])

            for file_idx in tqdm(range(len(extxyz_files)), desc=f"Processing {source}"):
                extxyz_path = os.path.join(split_dir, f"{file_idx}.extxyz")
                txt_path = os.path.join(split_dir, f"{file_idx}.txt")

                try:
                    atoms_list = ase_read(extxyz_path, index=':', format='extxyz')
                    with open(txt_path, 'r') as f:
                        txt_lines = f.readlines()

                    for atoms, txt_line in zip(atoms_list, txt_lines):
                        try:
                            parts = txt_line.strip().split(',')
                            sid = parts[0]
                            reference_energy = float(parts[2])

                            # Anomaly filtering
                            if sid not in metadata:
                                continue

                            if metadata[sid].get('anomaly', 0) != 0:
                                continue

                            # Process to Data object
                            data = self._process_single_item(atoms, sid, reference_energy, metadata)

                            if data is not None:
                                train_data.append(data)

                        except Exception:
                            continue

                except Exception as e:
                    log.warning(f"Error reading {source}[{file_idx}]: {e}")
                    continue

                del atoms_list
                gc.collect()

        torch.save(train_data, os.path.join(self.processed_dir, "train.pt"))
        log.info(f"Train: {len(train_data):,} samples")
        del train_data
        gc.collect()

        # Process val_id → val.pt, test.pt
        log.info("Processing val_id (split into val + test)...")

        val_id_data = []
        split_dir = os.path.join(self.raw_dir, 'val_id')
        extxyz_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.extxyz')])

        for file_idx in tqdm(range(len(extxyz_files)), desc="Processing val_id"):
            extxyz_path = os.path.join(split_dir, f"{file_idx}.extxyz")
            txt_path = os.path.join(split_dir, f"{file_idx}.txt")

            try:
                atoms_list = ase_read(extxyz_path, index=':', format='extxyz')
                with open(txt_path, 'r') as f:
                    txt_lines = f.readlines()

                for atoms, txt_line in zip(atoms_list, txt_lines):
                    try:
                        parts = txt_line.strip().split(',')
                        sid = parts[0]
                        reference_energy = float(parts[2])

                        if sid not in metadata:
                            continue

                        if metadata[sid].get('anomaly', 0) != 0:
                            continue

                        data = self._process_single_item(atoms, sid, reference_energy, metadata)

                        if data is not None:
                            val_id_data.append(data)

                    except Exception:
                        continue

            except Exception as e:
                log.warning(f"Error reading val_id[{file_idx}]: {e}")
                continue

            del atoms_list
            gc.collect()

        # Split val_id 50:50
        split_point = len(val_id_data) // 2
        val_data = val_id_data[:split_point]
        test_data = val_id_data[split_point:]

        torch.save(val_data, os.path.join(self.processed_dir, "val.pt"))
        log.info(f"Val:   {len(val_data):,} samples")

        torch.save(test_data, os.path.join(self.processed_dir, "test.pt"))
        log.info(f"Test:  {len(test_data):,} samples")

        del val_id_data, val_data, test_data
        gc.collect()

        log.info("Processing Complete!\n")

    def _process_single_item(
            self,
            atoms,
            sid: str,
            reference_energy: float,
            metadata: dict,
    ) -> Optional[Data]:
        """Process single structure from atoms object"""
        try:
            # Extract structure info
            atom_types = torch.from_numpy(atoms.get_atomic_numbers()).long()
            pos = torch.from_numpy(atoms.get_positions()).float()
            cell = torch.from_numpy(atoms.get_cell().array).float()
            num_atoms = len(atoms)
            tags = torch.from_numpy(atoms.arrays['tags']).long()

            # Compute lattice info
            lattice_info = compute_lattice_info(cell, num_atoms, pos)

            # Create base Data object with common fields
            data = create_base_data_object(
                atom_types, pos, lattice_info['frac_coords'], tags,
                cell, lattice_info, num_atoms
            )

            # Add S2EF-specific fields: energy from calc.results
            energy = atoms.calc.results.get('energy')
            free_energy = atoms.calc.results.get('free_energy')

            data.energy = torch.tensor([energy], dtype=torch.float32)
            data.free_energy = torch.tensor([free_energy if free_energy is not None else energy], dtype=torch.float32)
            data.reference_energy = torch.tensor([reference_energy], dtype=torch.float32)

            return data

        except Exception as e:
            log.warning(f"Error processing structure {sid}: {e}")
            return None

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

class OC20is2re(Dataset):
    """
    The OC20-is2re dataset from Fairchem, convert PyG object .lmdb to .pt.
    All dataset is adapted from: https://fair-chem.github.io/catalysts/datasets/summary.html
    """

    data_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz"
    metadata_url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"

    def __init__(
            self,
            root: str,
            split: str = "train",
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
            "train/data.lmdb",
            "val_id/data.lmdb",
            "val_ood_ads/data.lmdb",
            "val_ood_cat/data.lmdb",
            "val_ood_both/data.lmdb",
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def metadata_path(self) -> str:
        return os.path.join(os.path.dirname(self.root), "oc20_metadata.pkl")

    def download(self) -> None:
        """Download IS2RE LMDBs and shared metadata"""
        tar_path = os.path.join(self.raw_dir, "oc20_is2re.tar.gz")
        structured_path = os.path.join(self.raw_dir, "train")
        unstructured_path = os.path.join(self.raw_dir, "is2res_train_val_test_lmdbs")

        # Download shared metadata
        if not os.path.exists(self.metadata_path):
            log.info("[↓] Downloading OC20 metadata (shared)...")
            subprocess.run(["wget", "--progress=bar:force:noscroll", self.metadata_url,
                            "-O", self.metadata_path], check=True)

        if not os.path.exists(unstructured_path) or os.path.exists(structured_path):
            if not os.path.exists(tar_path):
                log.info("[↓] Downloading OC20-IS2RE LMDBs...")
                subprocess.run(["wget", "--progress=bar:force:noscroll", self.data_url,
                                "-O", tar_path], check=True)
            else:
                log.info(".tar file already exists. Skipping download.")

            log.info("Uncompressing OC20-IS2RE LMDBs...")
            subprocess.run(["tar", "-xvzf", tar_path, "-C", self.raw_dir], check=True)
            os.remove(tar_path)
        else:
            log.info("OC20-IS2RE LMDBs already exist. Skipping download and extraction.")

        # Restructure directory
        if not os.path.exists(structured_path):
            subprocess.run(
                f"mv {os.path.join(unstructured_path, 'data', 'is2re', 'all')}/* {self.raw_dir}",
                shell=True, check=True
            )
            subprocess.run(
                f"mv {os.path.join(unstructured_path, 'README_is2res.md')} {self.raw_dir}",
                shell=True, check=True
            )
            subprocess.run(["rm", "-r", unstructured_path], check=True)

    def process(self) -> None:
        """
        Process IS2RE LMDB files into 3 splits:
        - train.pt: train + val_ood_ads + val_ood_cat
        - val.pt: val_id
        - test.pt: val_ood_both

        Open with LmdbDataset, deprecated in FairChem v2.
        """
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        log.info(f"Loaded metadata: {len(metadata):,} entries")

        os.makedirs(self.processed_dir, exist_ok=True)

        # Process train + val_ood_ads + val_ood_cat → train.pt
        log.info("Processing train split (train + val_ood_ads + val_ood_cat)...")

        train_data = []
        for split in ['train', 'val_ood_ads', 'val_ood_cat']:
            lmdb_path = os.path.join(self.raw_dir, split)

            if not os.path.exists(lmdb_path):
                log.warning(f"Split {split} not found, skipping...")
                continue

            dataset_config = {'format': 'oc20_lmdb', 'src': lmdb_path}
            lmdb_dataset = LmdbDataset(dataset_config)

            for idx in tqdm(range(len(lmdb_dataset)), desc=f"Processing {split}"):
                try:
                    data = self._process_single_item(lmdb_dataset[idx], metadata, split)
                    if data is not None:
                        train_data.append(data)
                except Exception as e:
                    log.warning(f"Error processing {split}[{idx}]: {e}")
                    continue

            del lmdb_dataset
            gc.collect()

        torch.save(train_data, os.path.join(self.processed_dir, "train.pt"))
        log.info(f"Train: {len(train_data):,} samples")
        del train_data
        gc.collect()

        # Process val_id → val.pt
        log.info("Processing val split (val_id)...")

        val_data = []
        lmdb_path = os.path.join(self.raw_dir, 'val_id')

        if os.path.exists(lmdb_path):
            dataset_config = {'format': 'oc20_lmdb', 'src': lmdb_path}
            lmdb_dataset = LmdbDataset(dataset_config)

            for idx in tqdm(range(len(lmdb_dataset)), desc="Processing val_id"):
                try:
                    data = self._process_single_item(lmdb_dataset[idx], metadata, 'val_id')
                    if data is not None:
                        val_data.append(data)
                except Exception as e:
                    log.warning(f"Error processing val_id[{idx}]: {e}")
                    continue

            del lmdb_dataset
            gc.collect()

        torch.save(val_data, os.path.join(self.processed_dir, "val.pt"))
        log.info(f"Val:   {len(val_data):,} samples")
        del val_data
        gc.collect()

        # Process val_ood_both → test.pt
        log.info("Processing test split (val_ood_both)...")

        test_data = []
        lmdb_path = os.path.join(self.raw_dir, 'val_ood_both')

        if os.path.exists(lmdb_path):
            dataset_config = {'format': 'oc20_lmdb', 'src': lmdb_path}
            lmdb_dataset = LmdbDataset(dataset_config)

            for idx in tqdm(range(len(lmdb_dataset)), desc="Processing val_ood_both"):
                try:
                    data = self._process_single_item(lmdb_dataset[idx], metadata, 'val_ood_both')
                    if data is not None:
                        test_data.append(data)
                except Exception as e:
                    log.warning(f"Error processing val_ood_both[{idx}]: {e}")
                    continue

            del lmdb_dataset
            gc.collect()

        torch.save(test_data, os.path.join(self.processed_dir, "test.pt"))
        log.info(f"Test:  {len(test_data):,} samples")
        del test_data
        gc.collect()

        log.info("Processing Complete!\n")

    def _process_single_item(self, datadict, metadata, split):
        """Process single item from LMDB"""
        sid = datadict['sid']
        meta_key = f"random{sid}"

        # Anomaly filtering
        if meta_key not in metadata:
            return None

        meta_info = metadata[meta_key]

        if meta_info.get('anomaly', 0) != 0:
            return None

        # Extract basic attributes
        atom_types = datadict['atomic_numbers']
        cell = datadict['cell']
        num_atoms = datadict['natoms']
        tags = datadict['tags']
        pos = datadict['pos']

        # Compute lattice info
        lattice_info = compute_lattice_info(cell, num_atoms, pos)

        # Create base Data object with common fields
        data = create_base_data_object(
            atom_types, pos, lattice_info['frac_coords'], tags,
            cell, lattice_info, num_atoms
        )

        # Add IS2RE-specific fields: conditioning info from metadata
        data.cat_class = torch.tensor([meta_info['class']], dtype=torch.long)
        data.ads_id = torch.tensor([meta_info['ads_id']], dtype=torch.long)
        data.surface = torch.tensor(meta_info['miller_index'], dtype=torch.float)
        data.binding_energy = torch.tensor([datadict['y_relaxed']], dtype=torch.float)

        return data

    def _load_split(self):
        """Load the requested split"""
        split_file = os.path.join(self.processed_dir, f"{self.split}.pt")
        return torch.load(split_file)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int):
        return self.data_list[idx]