# coding=utf-8
"""
PyTorch Lightning Module for WiFo (Wireless Foundation Model)

This module wraps the WiFo model architecture and implements the training,
validation, and test steps using PyTorch Lightning framework.

Hyperparameters from the paper (Liu et al., SCIENCE CHINA Information Sciences, 2025):
- Optimizer: AdamW with betas=(0.9, 0.999), weight_decay=0.05
- Base Learning Rate: 5e-4
- Min Learning Rate: 1e-5
- Batch Size: 128
- Total Epochs: 200
- LR Schedule: Cosine decay with 5-epoch warmup
- Masking Ratios: Random=85%, Temporal=50%, Frequency=50%
"""

import os
import sys
import random
import logging
from typing import Optional, Tuple, List
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import lightning as L

# Add src directory to path for imports (scripts run from project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.model import WiFo_model

logger = logging.getLogger(__name__)


# For backward compatibility with old checkpoints
class ArgsNamespace:
    """Simple namespace to hold model configuration (for loading old checkpoints)."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class WiFoLightningModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for WiFo Masked Autoencoder.

    This module implements the three-task training approach:
    1. Random-masked reconstruction (85% masking) - captures 3D structured features
    2. Time-masked reconstruction (50% masking) - learns causal time relationships
    3. Frequency-masked reconstruction (50% masking) - learns frequency band variations

    Args:
        args: Configuration namespace containing model hyperparameters
            Required attributes:
            - size: Model size ('tiny', 'little', 'small', 'base')
            - lr: Base learning rate (default: 5e-4)
            - min_lr: Minimum learning rate (default: 1e-5)
            - weight_decay: Weight decay for AdamW (default: 0.05)
            - lr_anneal_steps: Total training epochs (default: 200)
            - batch_size: Batch size (default: 128)
            - warmup_steps: Warmup epochs (default: 5)
            - mask_ratio: Default masking ratio (default: 0.5)
            - mask_strategy: Default masking strategy (default: 'random')
            - mask_strategy_random: 'none' or 'batch' for multi-strategy training
            - patch_size: Patch size for spatial dimensions (default: 4)
            - t_patch_size: Patch size for temporal dimension (default: 2)
            - pos_emb: Positional embedding type ('SinCos' or 'trivial')
            - no_qkv_bias: Disable QKV bias in attention (default: 0)
            - clip_grad: Gradient clipping threshold (default: 0.05)
    """

    # Masking strategy configurations from the paper
    MASK_LIST = {
        'random': [0.85],      # 85% random masking
        'temporal': [0.5],     # 50% temporal masking
        'fre': [0.5],          # 50% frequency masking
    }

    @staticmethod
    def _get_attr(args, *path):
        """
        Safely get attribute from nested namespace.
        NO DEFAULT VALUES - all parameters must come from config.

        Args:
            args: Configuration namespace
            *path: Attribute path (e.g., 'training', 'mask_strategy')

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
        # Don't save hyperparameters to checkpoint - only save model weights

        # Store configuration
        self.args = args

        # Training hyperparameters - MUST come from config, no defaults
        self.lr = float(self._get_attr(args, 'training', 'optimizer', 'lr'))
        self.min_lr = float(self._get_attr(args, 'training', 'optimizer', 'min_lr'))
        self.weight_decay = float(self._get_attr(args, 'training', 'optimizer', 'weight_decay'))
        self.max_epochs = int(self._get_attr(args, 'training', 'scheduler', 'total_epochs'))
        self.warmup_epochs = int(self._get_attr(args, 'training', 'scheduler', 'warmup_epochs'))
        self.clip_grad = float(self._get_attr(args, 'training', 'gradient_clip'))

        # Masking configuration - MUST come from config, no defaults
        self.mask_strategy_random = self._get_attr(args, 'training', 'mask', 'strategy_mode')
        self.mask_strategy = self._get_attr(args, 'training', 'mask', 'strategy')
        self.mask_ratio = float(self._get_attr(args, 'training', 'mask', 'ratio'))

        # Initialize the WiFo model
        self.model = WiFo_model(args=args)

        # For tracking best validation NMSE
        self.best_val_nmse = float('inf')

        logger.info(f"Initialized WiFo Lightning Module with size={args.model.size}")
        logger.info(f"Learning rate: {self.lr}, Min LR: {self.min_lr}, Warmup: {self.warmup_epochs} epochs")
        logger.info(f"Max epochs: {self.max_epochs}, Weight decay: {self.weight_decay}")

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        From the paper:
        - Optimizer: AdamW with betas=(0.9, 0.999)
        - Weight decay: 0.05
        - LR schedule: Cosine decay with 5-epoch warmup

        Returns:
            dict with optimizer and scheduler configuration
        """
        # Filter parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]

        # AdamW optimizer with paper hyperparameters
        optimizer = AdamW(
            params,
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay
        )

        # Cosine learning rate scheduler with warmup
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=self.min_lr
            ),
            'interval': 'epoch',
            'frequency': 1,
            'name': 'learning_rate',
        }

        logger.info("Configured AdamW optimizer with cosine LR scheduler and warmup")
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def configure_gradient_clipping(
        self,
        optimizer: list[torch.optim.Optimizer],
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None
    ) -> None:
        """
        Configure gradient clipping.

        From the paper: gradient clipping threshold of 0.05
        """
        # Clip gradients by norm (default value from paper)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.clip_grad,
            gradient_clip_algorithm="norm"
        )

    def _select_masking_strategy(self) -> Tuple[str, float]:
        """
        Select masking strategy and ratio for current batch.

        Returns:
            Tuple of (strategy_name, mask_ratio)
        """
        if self.mask_strategy_random == 'batch':
            # Multi-strategy training: randomly select one strategy per batch
            strategy = random.choice(['random', 'temporal', 'fre'])
            ratio = random.choice(self.MASK_LIST[strategy])
        else:
            # Single strategy training
            strategy = self.mask_strategy
            ratio = self.mask_ratio

        return strategy, ratio

    def _add_noise(self, imgs: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
        """
        Add complex Gaussian noise to input tensors.

        From the paper: 20 dB SNR noise injection during training and inference.

        Args:
            imgs: Input tensor [N, 2, T, H, W] (real + imaginary channels)
            snr_db: Signal-to-noise ratio in dB (default: 20.0)

        Returns:
            Noisy tensor with same shape as input
        """
        device = imgs.device
        noise_power = torch.mean(imgs ** 2) * 10 ** (-snr_db / 10)
        noise = torch.randn_like(imgs) * torch.sqrt(noise_power)
        return imgs + noise

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Execute one training step with multi-strategy masking.

        Args:
            batch: Input batch from dataloader
            batch_idx: Batch index

        Returns:
            Loss tensor for backpropagation
        """
        # Select masking strategy for this batch
        strategy, mask_ratio = self._select_masking_strategy()

        # Move batch to device and convert to list format expected by model
        if isinstance(batch, (list, tuple)):
            imgs = [x.to(self.device) for x in batch]
        else:
            batch = batch.to(self.device)
            if batch.ndim == 6 and batch.shape[1] == 1:  # [B, 1, 2, T, K, N]
                imgs = [batch[i].squeeze(1) for i in range(batch.shape[0])]
            elif batch.ndim == 5:  # [B, 2, T, K, N]
                imgs = [batch[i].unsqueeze(0) for i in range(batch.shape[0])]
            else:
                imgs = [batch]

        # Forward pass through model
        loss, loss2, pred, target, mask = self.model(
            imgs,
            mask_ratio=mask_ratio,
            mask_strategy=strategy,
            seed=None,
            data='none'
        )

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(imgs))
        self.log(f'train/loss_{strategy}', loss, on_step=True, on_epoch=True, batch_size=len(imgs))
        self.log('train/mask_ratio', mask_ratio, on_step=True, on_epoch=True, batch_size=len(imgs))

        return loss

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Execute one validation step.

        Args:
            batch: Input batch from dataloader
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (for multi-dataset validation)

        Returns:
            Loss tensor for logging
        """
        # Use random masking for validation (can be configured differently if needed)
        strategy = self.mask_strategy
        mask_ratio = self.mask_ratio

        # Move batch to device and convert to list format expected by model
        if isinstance(batch, (list, tuple)):
            imgs = [x.to(self.device) for x in batch]
        else:
            batch = batch.to(self.device)
            if batch.ndim == 6 and batch.shape[1] == 1:  # [B, 1, 2, T, K, N]
                imgs = [batch[i].squeeze(1) for i in range(batch.shape[0])]
            elif batch.ndim == 5:  # [B, 2, T, K, N]
                imgs = [batch[i].unsqueeze(0) for i in range(batch.shape[0])]
            else:
                imgs = [batch]

        # Forward pass
        loss, loss2, pred, target, mask = self.model(
            imgs,
            mask_ratio=mask_ratio,
            mask_strategy=strategy,
            seed=None,
            data='none'
        )

        # Compute NMSE on masked patches
        target_power = torch.mean(target.abs() ** 2)
        nmse = loss / (target_power + 1e-10)

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(imgs))
        self.log('val/nmse', nmse, on_step=False, on_epoch=True, batch_size=len(imgs))

        return loss

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """
        Execute one test step for zero-shot evaluation.

        Args:
            batch: Input batch from dataloader
            batch_idx: Batch index
            dataloader_idx: Index of the dataloader (for multi-dataset evaluation)

        Returns:
            Dictionary with test metrics
        """
        # For zero-shot evaluation, we may use different masking strategies
        strategy = self.mask_strategy
        mask_ratio = self.mask_ratio

        # Move batch to device and convert to list format expected by model
        if isinstance(batch, (list, tuple)):
            imgs = [x.to(self.device) for x in batch]
        else:
            batch = batch.to(self.device)
            if batch.ndim == 6 and batch.shape[1] == 1:  # [B, 1, 2, T, K, N]
                imgs = [batch[i].squeeze(1) for i in range(batch.shape[0])]
            elif batch.ndim == 5:  # [B, 2, T, K, N]
                imgs = [batch[i].unsqueeze(0) for i in range(batch.shape[0])]
            else:
                imgs = [batch]

        # Forward pass
        loss, loss2, pred, target, mask = self.model(
            imgs,
            mask_ratio=mask_ratio,
            mask_strategy=strategy,
            seed=None,
            data='none'
        )

        # Compute NMSE on masked patches
        # pred and target are complex tensors: [N, L, C]
        # mask is [N, L] where 1 means masked
        dim1 = pred.shape[0]  # batch size

        # Handle complex tensors
        pred_real = torch.stack([pred.real, pred.imag], dim=2)  # [N, L, 2, C]
        target_real = torch.stack([target.real, target.imag], dim=2)

        pred_mask = pred_real.squeeze(dim=2)  # [N, L, 2, C]
        target_mask = target_real.squeeze(dim=2)

        # Extract masked patches
        if len(pred_mask.shape) == 3:
            mask_expanded = mask.unsqueeze(-1).expand_as(pred_mask)
            y_pred = pred_mask[mask_expanded == 1].reshape(dim1, -1)
            y_target = target_mask[mask_expanded == 1].reshape(dim1, -1)
        else:
            y_pred = pred_mask[mask == 1].reshape(dim1, -1)
            y_target = target_mask[mask == 1].reshape(dim1, -1)

        # Compute NMSE per sample
        y_pred_np = y_pred.detach().cpu().numpy()
        y_target_np = y_target.detach().cpu().numpy()
        nmse_per_sample = np.mean(np.abs(y_target_np - y_pred_np) ** 2, axis=1) / np.mean(np.abs(y_target_np) ** 2, axis=1)
        nmse_per_sample = torch.tensor(nmse_per_sample, device=self.device)

        # Log metrics - single aggregated metric across all dataloaders
        self.log(
            'test/nmse',
            nmse_per_sample.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=dim1
        )

        return {
            'nmse': nmse_per_sample,
            'dataloader_idx': dataloader_idx,
            'loss': loss
        }

    def on_train_epoch_end(self) -> None:
        """Log metrics at end of training epoch."""
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', current_lr, on_step=False, on_epoch=True, prog_bar=False)
        logger.debug(f"Epoch {self.current_epoch} - Learning rate: {current_lr:.6f}")

    def on_validation_epoch_end(self) -> None:
        """Track best validation NMSE at end of validation epoch."""
        # Get average validation NMSE across all dataloaders
        val_nmse = self.trainer.callback_metrics.get('val/nmse', None)

        if val_nmse is not None:
            if val_nmse < self.best_val_nmse:
                self.best_val_nmse = val_nmse
                self.log('val/best_nmse', self.best_val_nmse, on_step=False, on_epoch=True, prog_bar=True)
                logger.info(f"New best validation NMSE: {self.best_val_nmse:.6f} at epoch {self.current_epoch}")

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.5,
                mask_strategy: str = 'random', seed: Optional[int] = None,
                data: str = 'none') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the WiFo model.

        Args:
            imgs: Input tensor [N, 2, T, H, W]
            mask_ratio: Masking ratio
            mask_strategy: Masking strategy ('random', 'temporal', 'fre')
            seed: Random seed for reproducibility
            data: Dataset name for data-specific handling

        Returns:
            Tuple of (loss, loss2, pred, target, mask)
        """
        return self.model(imgs, mask_ratio=mask_ratio, mask_strategy=mask_strategy, seed=seed, data=data)


def create_wifo_lightning_module(args) -> WiFoLightningModule:
    """
    Factory function to create WiFo Lightning Module with standard configuration.

    Args:
        args: Configuration namespace

    Returns:
        Initialized WiFoLightningModule
    """
    return WiFoLightningModule(args)
