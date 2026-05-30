import os
import torch
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from omegaconf import DictConfig
from typing import Optional
from torch.utils.data import ConcatDataset
from typing import Sequence

from src.data.components.oc20_dataset import OC20s2ef, OC20is2re
from src.data.components.oc22_dataset import OC22s2ef, OC22is2re
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class JointDataModule(LightningDataModule):
    """
    Unified DataModule for OC20/OC22 × S2EF/IS2RE datasets.

    Dataset structure:
    - oc20s2ef, oc20is2re → dataset_idx=0 (share loss weights)
    - oc22s2ef, oc22is2re → dataset_idx=1 (share loss weights)

    Each dataset loads from pre-processed .pt files:
    - train.pt, val.pt, test.pt

    num_nodes_bincount: Only computed for IS2RE train splits (for LDM sampling)
    """

    def __init__(
            self,
            datasets: DictConfig,
            num_workers: DictConfig,
            batch_size: DictConfig,
            pin_memory: bool = False,
            prefetch_factor: Optional[int] = 2,
            persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: str) -> None:
        # Dataset configurations
        dataset_configs = [
            ('OC20s2ef', OC20s2ef),
            ('OC20is2re', OC20is2re),
            ('OC22s2ef', OC22s2ef),
            ('OC22is2re', OC22is2re),
        ]

        keep_keys = {'atom_types', 'pos', 'frac_coords', 'tags', 'cell', 'lattices',
                     'lattices_scaled', 'lengths', 'lengths_scaled', 'angles',
                     'angles_radians', 'num_atoms', 'num_nodes', 'token_idx', 'dataset_idx',
                     'cat_class', 'ads_id', 'binding_energy' }

        def get_dataset_idx(name: str) -> int:
            return 0 if 'OC20' in name else 1

        for dataset_name, DatasetClass in dataset_configs:

            idx = get_dataset_idx(dataset_name)
            proportion = self.hparams.datasets[dataset_name].proportion

            if proportion == 0.0:
                log.info(f"Skipping {dataset_name} (proportion=0.0)")
                continue

            log.info(f"Setting up {dataset_name}...")

            # === Train dataset ===
            if stage in [None, "fit"]:
                train_dataset = DatasetClass(
                    root=self.hparams.datasets[dataset_name].root,
                    split='train'
                )

                for data in train_dataset.data_list:
                    data.dataset_idx = torch.tensor([idx], dtype=torch.long)
                    if not hasattr(data, 'binding_energy'):
                        data.binding_energy = torch.zeros(1, dtype=torch.float)
                    for key in list(data.keys()):
                        if key not in keep_keys:
                            delattr(data, key)

                if proportion < 1.0:
                    original_len = len(train_dataset.data_list)
                    train_dataset.data_list = train_dataset.data_list[:int(original_len * proportion)]
                    log.info(
                        f"{dataset_name} training dataset: {original_len:,} → {len(train_dataset.data_list):,} samples, (proportion={proportion})")
                else:
                    log.info(f"{dataset_name} training dataset: {len(train_dataset.data_list):,} samples")

                setattr(self, f'{dataset_name}_train', train_dataset)

                # num_nodes_bincount (IS2RE-only)
                if 'is2re' in dataset_name.lower():
                    num_nodes_path = os.path.join(os.path.dirname(
                        self.hparams.datasets[dataset_name].root),
                        "num_nodes_bincount.pt"
                    )
                    if not os.path.exists(num_nodes_path):
                        log.info(f"Computing num_nodes_bincount for {dataset_name}...")
                        num_nodes = torch.tensor([data.num_nodes.item() for data in train_dataset.data_list])
                        torch.save(torch.bincount(num_nodes), num_nodes_path)

            # === Val dataset ===
            if stage in [None, "fit", "validate"]:
                val_dataset = DatasetClass(
                    root=self.hparams.datasets[dataset_name].root,
                    split='val'
                )

                for data in val_dataset.data_list:
                    data.dataset_idx = torch.tensor([idx], dtype=torch.long)
                    if not hasattr(data, 'binding_energy'):
                        data.binding_energy = torch.zeros(1, dtype=torch.float)
                    for key in list(data.keys()):
                        if key not in keep_keys:
                            delattr(data, key)

                if proportion < 1.0:
                    original_len = len(val_dataset.data_list)
                    val_dataset.data_list = val_dataset.data_list[:int(original_len * proportion)]
                    log.info(f"{dataset_name} validataion dataset: {original_len:,} → {len(val_dataset.data_list):,} samples")
                else:
                    log.info(f"{dataset_name} validataion dataset: {len(val_dataset.data_list):,} samples")

                setattr(self, f'{dataset_name}_val', val_dataset)

            # === Test dataset ===
            if stage in [None, "test"]:
                test_dataset = DatasetClass(
                    root=self.hparams.datasets[dataset_name].root,
                    split='test'
                )

                for data in test_dataset.data_list:
                    data.dataset_idx = torch.tensor([idx], dtype=torch.long)
                    if not hasattr(data, 'binding_energy'):
                        data.binding_energy = torch.zeros(1, dtype=torch.float)
                    for key in list(data.keys()):
                        if key not in keep_keys:
                            delattr(data, key)
                if proportion < 1.0:
                    original_len = len(test_dataset.data_list)
                    test_dataset.data_list = test_dataset.data_list[:int(original_len * proportion)]
                    log.info(f"{dataset_name} test dataset: {original_len:,} → {len(test_dataset.data_list):,} samples")
                else:
                    log.info(f"{dataset_name} test dataset: {len(test_dataset.data_list):,} samples")

                setattr(self, f'{dataset_name}_test', test_dataset)

        # === ConcatDataset for training ===
        if stage in [None, "fit"]:
            train_datasets = []

            for dataset_name, _ in dataset_configs:
                train_attr = f'{dataset_name}_train'
                if hasattr(self, train_attr):
                    train_datasets.append(getattr(self, train_attr))
                    log.info(f"  - Added {dataset_name} to training set")

            if len(train_datasets) == 0:
                raise ValueError("No training datasets loaded! Check your proportion settings.")

            self.train_dataset = ConcatDataset(train_datasets)
            log.info(f"Total training set: {len(self.train_dataset):,} samples")


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        val_loaders = []

        for dataset_name in ['OC20s2ef', 'OC20is2re', 'OC22s2ef', 'OC22is2re']:
            val_attr = f'{dataset_name}_val'
            if hasattr(self, val_attr):
                val_loaders.append(
                    DataLoader(
                        dataset=getattr(self, val_attr),
                        batch_size=self.hparams.batch_size.val,
                        num_workers=self.hparams.num_workers.val,
                        pin_memory=self.hparams.pin_memory,
                        prefetch_factor=self.hparams.prefetch_factor,
                        persistent_workers=self.hparams.persistent_workers,
                        shuffle=False,
                    )
                )
        return val_loaders

    def test_dataloader(self) -> Sequence[DataLoader]:
        test_loaders = []

        for dataset_name in ['OC20s2ef', 'OC20is2re', 'OC22s2ef', 'OC22is2re']:
            test_attr = f'{dataset_name}_test'
            if hasattr(self, test_attr):
                test_loaders.append(
                    DataLoader(
                        dataset=getattr(self, test_attr),
                        batch_size=self.hparams.batch_size.test,
                        num_workers=self.hparams.num_workers.test,
                        pin_memory=self.hparams.pin_memory,
                        prefetch_factor=self.hparams.prefetch_factor,
                        persistent_workers=self.hparams.persistent_workers,
                        shuffle=False,
                    )
                )
        return test_loaders