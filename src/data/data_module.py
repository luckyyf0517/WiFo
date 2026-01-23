# coding=utf-8
"""
PyTorch Lightning DataModule for WiFo (Wireless Foundation Model)

This module handles data loading and preprocessing for WiFo training
following the paper's specifications (Liu et al., 2025).

Data format from the paper:
- 16 pre-training datasets (D1-D16) with 9k train / 1k val / 2k test split
- Each dataset: 12,000 samples total
- Input: Complex CSI tensor H in C^(T x K x N)
- Processed: Real-valued tensor H~ in R^(2 x T x K x N)
- Noise injection: 20 dB SNR during training and inference

Data files:
- X_train.mat contains training data (key: 'X_train')
- X_val.mat contains validation data (key: 'X_val')
- X_test.mat contains test data (key: 'X_val' - dataset quirk)
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
    - Data is already split: X_train.mat, X_val.mat, X_test.mat
    - Normalizing data using per-dataset statistics
    - Converting complex tensors to real-valued tensors (real + imaginary)
    - Creating DataLoaders with appropriate settings

    Args:
        args: Configuration namespace containing:
            - data.dataset: Dataset names separated by '*' (e.g., 'D1*D2*D3')
            - data.data_path: Base path to dataset directory
            - training.batch_size: Batch size
            - data.num_workers: Number of data loading workers
            - data.pin_memory: Whether to pin memory for GPU transfer
            - data.prefetch_factor: Number of batches to prefetch

    Note:
        All parameters MUST come from config. No default values are used.
        Missing required parameters will raise AttributeError.
    """

    @staticmethod
    def _get_attr(args, *path):
        """
        Safely get attribute from nested namespace.
        NO DEFAULT VALUES - all parameters must come from config.

        Args:
            args: Configuration namespace
            *path: Attribute path (e.g., 'data', 'dataset')

        Returns:
            Attribute value

        Raises:
            AttributeError: If any attribute in the path is not found
        """
        current = args
        for attr in path:
            if not hasattr(current, attr):
                raise AttributeError(
                    f"Required config parameter '{'.'.join(path)}' not found. "
                    f"Missing '{attr}' in path. Please check your configuration."
                )
            current = getattr(current, attr)
        return current

    def __init__(self, args):
        super().__init__()
        self.args = args

        # All parameters MUST come from config, no defaults
        dataset = self._get_attr(args, 'data', 'dataset')
        self.dataset_names = dataset.split('*') if dataset else []
        self.data_path = self._get_attr(args, 'data', 'data_path')
        self.batch_size = self._get_attr(args, 'training', 'batch_size')
        self.num_workers = self._get_attr(args, 'data', 'num_workers')
        self.pin_memory = self._get_attr(args, 'data', 'pin_memory')
        self.prefetch_factor = self._get_attr(args, 'data', 'prefetch_factor')

        # Data storage
        self.train_datasets: List[CSIDataset] = []
        self.val_datasets: List[CSIDataset] = []
        self.test_datasets: List[CSIDataset] = []

        # Normalization statistics per dataset
        self.dataset_stats: dict = {}

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and prepare data for training, validation, and testing.

        This method:
        1. Loads .mat files for each dataset (X_train.mat, X_val.mat, X_test.mat)
        2. Normalizes using per-dataset mean and variance
        3. Converts complex to real-valued tensors

        For different stages:
        - 'fit': loads train and val data only
        - 'test': loads test data only
        - 'validate': loads val data only
        - 'predict': loads test data only

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Clear existing data to prevent duplication when setup is called multiple times
        self.train_datasets.clear()
        self.val_datasets.clear()
        self.test_datasets.clear()

        if stage == 'predict' or stage is None:
            logger.info("Setting up data...")

        for dataset_name in self.dataset_names:
            # Load data for this dataset based on stage
            train_data, val_data, test_data = self._load_single_dataset(dataset_name, stage)

            # Only append non-empty datasets based on stage
            if stage == 'test' or stage == 'predict':
                self.test_datasets.append(test_data)
            elif stage == 'validate':
                self.val_datasets.append(val_data)
            else:  # 'fit' or None - load train and val only
                self.train_datasets.append(train_data)
                self.val_datasets.append(val_data)

        # Log summary
        train_count = sum(len(d) for d in self.train_datasets)
        val_count = sum(len(d) for d in self.val_datasets)
        test_count = sum(len(d) for d in self.test_datasets)

        if train_count > 0:
            logger.info(f"Train samples: {train_count}")
        if val_count > 0:
            logger.info(f"Val samples: {val_count}")
        if test_count > 0:
            logger.info(f"Test samples: {test_count}")

    def _load_single_dataset(self, dataset_name: str, stage: Optional[str] = None) -> Tuple[CSIDataset, CSIDataset, CSIDataset]:
        """
        Load a single dataset's train/val/test splits from separate files.

        The data is already split in the .mat files:
        - X_train.mat contains training data (key: 'X_train')
        - X_val.mat contains validation data (key: 'X_val')
        - X_test.mat contains test data (key: 'X_val' - dataset quirk)

        From the paper:
        - 12,000 samples per dataset
        - Train: 9,000 samples
        - Validation: 1,000 samples
        - Test: 2,000 samples

        Args:
            dataset_name: Name of the dataset (e.g., 'D1', 'D2')
            stage: 'fit', 'validate', 'test', or 'predict'

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            Returns empty datasets for splits not loaded based on stage
        """
        test_path = os.path.join(self.data_path, dataset_name, 'X_test.mat')
        train_path = os.path.join(self.data_path, dataset_name, 'X_train.mat')
        val_path = os.path.join(self.data_path, dataset_name, 'X_val.mat')

        # Determine which splits to load based on stage
        if stage == 'test' or stage == 'predict':
            splits_to_load = [('test', test_path)]
        elif stage == 'validate':
            splits_to_load = [('val', val_path)]
        else:  # 'fit' or None
            splits_to_load = [('train', train_path), ('val', val_path)]

        # Load each split with progress bar
        split_data = {}
        for split_name, split_path in tqdm(splits_to_load, desc=f"  {dataset_name}", unit="file", ncols=80):
            if os.path.exists(split_path):
                try:
                    mat_data = hdf5storage.loadmat(split_path)
                    if split_name == 'train' and 'X_train' in mat_data:
                        X = mat_data['X_train']
                    elif split_name == 'val' and 'X_val' in mat_data:
                        X = mat_data['X_val']
                    elif split_name == 'test' and 'X_val' in mat_data:
                        X = mat_data['X_val']
                    else:
                        logger.warning(f"Unknown data keys in {split_path}")
                        continue

                    X_complex = torch.tensor(np.array(X, dtype=complex)).unsqueeze(1)
                    split_data[split_name] = torch.cat((X_complex.real, X_complex.imag), dim=1).float()
                    logger.debug(f"Loaded {split_path}: shape {split_data[split_name].shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {split_path}: {e}")

        # Check if we loaded any data
        if not split_data:
            raise FileNotFoundError(f"No data files found for dataset {dataset_name}")

        # Get individual splits (use None if not loaded)
        train_data = split_data.get('train')
        val_data = split_data.get('val')
        test_data = split_data.get('test')

        # Check if we loaded any data
        if train_data is None and val_data is None and test_data is None:
            raise FileNotFoundError(f"No data files found for dataset {dataset_name}")

        # Log dataset sizes
        train_size = train_data.shape[0] if train_data is not None else 0
        val_size = val_data.shape[0] if val_data is not None else 0
        test_size = test_data.shape[0] if test_data is not None else 0
        logger.info(f"Dataset {dataset_name}: train={train_size}, val={val_size}, test={test_size}")

        # Normalize each split independently and create datasets
        train_dataset = self._normalize_and_create_dataset(train_data, dataset_name, 'train') if train_data is not None else CSIDataset(torch.empty(0, 2, 1, 1, 1))
        val_dataset = self._normalize_and_create_dataset(val_data, dataset_name, 'val') if val_data is not None else CSIDataset(torch.empty(0, 2, 1, 1, 1))
        test_dataset = self._normalize_and_create_dataset(test_data, dataset_name, 'test') if test_data is not None else CSIDataset(torch.empty(0, 2, 1, 1, 1))

        return train_dataset, val_dataset, test_dataset

    def _normalize_and_create_dataset(self, data: torch.Tensor, dataset_name: str, split: str) -> CSIDataset:
        """
        Normalize data and create CSIDataset.

        Args:
            data: Input tensor [N, 2, T, H, W]
            dataset_name: Name of the dataset
            split: Split name ('train', 'val', 'test')

        Returns:
            CSIDataset with normalized data
        """
        # Compute statistics
        mean = data.mean(dim=(0, 2, 3, 4), keepdim=True)
        std = data.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8

        # Normalize
        normalized_data = (data - mean) / std

        # Store statistics
        key = f"{dataset_name}_{split}"
        self.dataset_stats[key] = {'mean': mean, 'std': std}

        return CSIDataset(normalized_data)

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
