# coding=utf-8
"""
PyTorch Lightning DataModule for WiFo (Wireless Foundation Model)

This module handles data loading, preprocessing, and train/validation/test splitting
for WiFo training following the paper's specifications (Liu et al., 2025).

Data format from the paper:
- 16 pre-training datasets (D1-D16) with 9k train / 1k val / 2k test split
- Each dataset: 12,000 samples total
- Input: Complex CSI tensor H in C^(T x K x N)
- Processed: Real-valued tensor H~ in R^(2 x T x K x N)
- Noise injection: 20 dB SNR during training and inference
"""

import os
import logging
from typing import Optional, List, Tuple
from tqdm import tqdm

import torch
import numpy as np
import hdf5storage
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CSIDataset(Dataset):
    """
    PyTorch Dataset for Channel State Information (CSI) data.

    Args:
        data: Preprocessed tensor data of shape [N, 2, T, H, W]
            where 2 channels represent real and imaginary parts
    """

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample from the dataset.

        Returns:
            Tensor of shape [1, 2, T, H, W] with real and imaginary channels
        """
        return self.data[idx].unsqueeze(0)


class WiFoDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for WiFo training.

    This DataModule handles:
    - Loading .mat files containing CSI data
    - Splitting data into train/validation/test sets (9k/1k/2k per dataset)
    - Normalizing data using per-dataset statistics
    - Converting complex tensors to real-valued tensors (real + imaginary)
    - Creating DataLoaders with appropriate settings

    Args:
        args: Configuration namespace containing:
            - dataset: Dataset names separated by '*' (e.g., 'D1*D2*D3')
            - data_path: Base path to dataset directory (default: 'dataset/')
            - batch_size: Batch size (default: 128)
            - num_workers: Number of data loading workers (default: 32)
            - pin_memory: Whether to pin memory for GPU transfer (default: True)
            - prefetch_factor: Number of batches to prefetch (default: 4)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_names = args.dataset.split('*') if hasattr(args, 'dataset') else ['DS1']
        self.data_path = getattr(args, 'data_path', 'dataset/')
        self.batch_size = getattr(args, 'batch_size', 128)
        self.num_workers = getattr(args, 'num_workers', 32)
        self.pin_memory = getattr(args, 'pin_memory', True)
        self.prefetch_factor = getattr(args, 'prefetch_factor', 4)

        # Data storage
        self.train_datasets: List[CSIDataset] = []
        self.val_datasets: List[CSIDataset] = []
        self.test_datasets: List[CSIDataset] = []

        # Normalization statistics per dataset
        self.dataset_stats: dict = {}

        logger.info(f"Initialized WiFoDataModule with datasets: {self.dataset_names}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and prepare data for training, validation, and testing.

        This method:
        1. Loads .mat files for each dataset
        2. Splits into train (9k), val (1k), test (2k) - total 12k per dataset
        3. Normalizes using per-dataset mean and variance
        4. Converts complex to real-valued tensors

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'predict' or stage is None:
            logger.info("Setting up data...")

        for dataset_name in tqdm(self.dataset_names, desc="Loading datasets", unit="set", ncols=100):
            # Load data for this dataset
            train_data, val_data, test_data = self._load_single_dataset(dataset_name)

            self.train_datasets.append(train_data)
            self.val_datasets.append(val_data)
            self.test_datasets.append(test_data)

        # For training, we combine all datasets
        # For validation/test, we keep them separate for evaluation
        logger.info(f"Loaded {len(self.train_datasets)} datasets")
        logger.info(f"Train samples: {sum(len(d) for d in self.train_datasets)}")
        logger.info(f"Val samples: {sum(len(d) for d in self.val_datasets)}")
        logger.info(f"Test samples: {sum(len(d) for d in self.test_datasets)}")

    def _load_single_dataset(self, dataset_name: str) -> Tuple[CSIDataset, CSIDataset, CSIDataset]:
        """
        Load a single dataset and split into train/val/test.

        From the paper:
        - 12,000 samples per dataset
        - Train: 9,000 samples
        - Validation: 1,000 samples
        - Test: 2,000 samples

        Args:
            dataset_name: Name of the dataset (e.g., 'D1', 'D2')

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Try to load test data (inference data)
        test_path = os.path.join(self.data_path, dataset_name, 'X_test.mat')
        train_path = os.path.join(self.data_path, dataset_name, 'X_train.mat')
        val_path = os.path.join(self.data_path, dataset_name, 'X_val.mat')

        # Load available data files
        data_list = []

        for path in [train_path, val_path, test_path]:
            if os.path.exists(path):
                try:
                    mat_data = hdf5storage.loadmat(path)
                    if 'X_train' in mat_data:
                        X = mat_data['X_train']
                    elif 'X_val' in mat_data:
                        X = mat_data['X_val']
                    elif 'X_test' in mat_data:
                        X = mat_data['X_test']
                    else:
                        logger.warning(f"Unknown data keys in {path}")
                        continue

                    # Convert to complex tensor
                    X_complex = torch.tensor(np.array(X, dtype=complex)).unsqueeze(1)

                    # Convert to real + imaginary channels: [N, 2, T, H, W]
                    X_real = torch.cat((X_complex.real, X_complex.imag), dim=1).float()

                    data_list.append(X_real)
                    logger.debug(f"Loaded {path}: shape {X_real.shape}")

                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        if not data_list:
            raise FileNotFoundError(f"No data files found for dataset {dataset_name}")

        # Concatenate all available data
        all_data = torch.cat(data_list, dim=0)
        total_samples = all_data.shape[0]

        logger.info(f"Dataset {dataset_name}: loaded {total_samples} samples")

        # Normalize data using per-dataset statistics
        all_data, mean, std = self._normalize_data(all_data, dataset_name)

        # Split according to paper: 9k train / 1k val / 2k test
        # If we have less than 12k samples, adjust proportions
        if total_samples >= 12000:
            train_size = 9000
            val_size = 1000
            test_size = 2000
        else:
            # Use proportional split
            train_size = int(total_samples * 0.75)
            val_size = int(total_samples * 0.083)
            test_size = total_samples - train_size - val_size

        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:train_size + val_size + test_size]

        logger.info(f"Dataset {dataset_name}: train={train_data.shape[0]}, "
                   f"val={val_data.shape[0]}, test={test_data.shape[0]}")

        return (
            CSIDataset(train_data),
            CSIDataset(val_data),
            CSIDataset(test_data)
        )

    def _normalize_data(self, data: torch.Tensor, dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize data using per-dataset mean and variance.

        From the paper: "Normalize CSI samples using the mean and variance
        calculated per dataset."

        Args:
            data: Input tensor [N, 2, T, H, W]
            dataset_name: Name of the dataset for storing stats

        Returns:
            Tuple of (normalized_data, mean, std)
        """
        # Compute statistics over the dataset
        mean = data.mean(dim=(0, 2, 3, 4), keepdim=True)  # [1, 2, 1, 1, 1]
        std = data.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8  # [1, 2, 1, 1, 1]

        # Normalize
        normalized_data = (data - mean) / std

        # Store statistics for later use
        self.dataset_stats[dataset_name] = {'mean': mean, 'std': std}

        logger.debug(f"Dataset {dataset_name} - mean: {mean.squeeze()}, std: {std.squeeze()}")

        return normalized_data, mean, std

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create training dataloader combining all training datasets.

        Returns:
            DataLoader for training
        """
        # Combine all training datasets
        if len(self.train_datasets) == 1:
            combined_data = self.train_datasets[0]
        else:
            # Concatenate all datasets
            all_train_data = torch.cat([d.data for d in self.train_datasets], dim=0)
            combined_data = CSIDataset(all_train_data)

        dataloader = DataLoader(
            combined_data,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True
        )

        logger.info(f"Train dataloader: {len(combined_data)} samples, "
                   f"batch_size={self.batch_size}, {len(dataloader)} batches")

        return dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create validation dataloader(s).

        Returns:
            Single DataLoader or list of DataLoaders (one per dataset)
        """
        if len(self.val_datasets) == 1:
            return [
                DataLoader(
                    self.val_datasets[0],
                    batch_size=self.batch_size,
                    shuffle=False,  # No shuffle for validation
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
                )
            ]
        else:
            # Return list of dataloaders, one per dataset
            dataloaders = []
            for dataset in self.val_datasets:
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
                    )
                )
            return dataloaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create test dataloader(s) for zero-shot evaluation.

        Returns:
            List of DataLoaders, one per dataset for separate evaluation
        """
        dataloaders = []

        for i, dataset in enumerate(self.test_datasets):
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No shuffle for testing
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
            )
            dataloaders.append(dataloader)
            logger.info(f"Test dataloader {i}: {len(dataset)} samples, {len(dataloader)} batches")

        return dataloaders


def create_wifo_data_module(args) -> WiFoDataModule:
    """
    Factory function to create WiFo DataModule with standard configuration.

    Args:
        args: Configuration namespace

    Returns:
        Initialized WiFoDataModule
    """
    return WiFoDataModule(args)
