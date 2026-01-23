# coding=utf-8
"""
PyTorch Lightning Evaluation Script for WiFo (Wireless Foundation Model)

This script provides the evaluation/inference entry point for the WiFo model,
following the specifications from the paper:
"Liu et al., WiFo: wireless foundation model for channel prediction,
SCIENCE CHINA Information Sciences, 2025"

Usage:
    python src/eval.py --dataset "D1*D2*D3" --size base --checkpoint_path weights/lightning/wifo_base.ckpt

Evaluation scenarios:
- Zero-shot evaluation on unseen datasets
- Multi-strategy masking evaluation (random, temporal, frequency)
- NMSE computation on masked patches
"""

import argparse
import os
import sys
import random
import logging
from typing import Optional
from tqdm import tqdm

# Add src directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from training.lightning_module import WiFoLightningModule, create_wifo_lightning_module
from data.data_module import WiFoDataModule, create_wifo_data_module

logger = logging.getLogger(__name__)


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def setup_init(seed: int = 100) -> None:
    """
    Initialize random seeds for reproducibility.

    Args:
        seed: Random seed value (default: 100)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info(f"Set random seed to {seed}")


def create_argparser() -> argparse.ArgumentParser:
    """
    Create argument parser with all configuration options.

    Returns:
        Configured ArgumentParser
    """
    defaults = dict(
        # Experimental settings
        note='',
        task='short',
        dataset='DS1',
        data_path='dataset/',
        process_name='wifo_eval',
        his_len=6,
        pred_len=6,
        few_ratio=0.0,  # Default to 0.0 for zero-shot evaluation
        stage=0,

        # Model settings
        mask_ratio=0.5,
        patch_size=4,
        t_patch_size=2,
        size='base',
        no_qkv_bias=0,
        pos_emb='SinCos',
        conv_num=3,

        # Evaluation settings
        mask_strategy='temporal',  # Default strategy for evaluation
        mask_strategy_random='none',  # 'none' or 'batch' - use 'none' for specific strategy
        checkpoint_path='weights/lightning/wifo_base.ckpt',  # Path to Lightning checkpoint

        # Device settings
        device_id='0',
        accelerator='auto',
        devices='auto',

        # Paths
        output_dir='./experiments',
        log_dir='./logs',
    )

    parser = argparse.ArgumentParser(description='WiFo Evaluation with PyTorch Lightning')

    # Add all arguments
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true' if not value else 'store_false',
                             default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser


def load_legacy_weights(args):
    """
    Load legacy .pkl weights and convert to Lightning checkpoint format.

    Args:
        args: Configuration namespace with file_load_path pointing to .pkl file

    Returns:
        Path to converted Lightning checkpoint
    """
    # This function is a fallback for loading legacy .pkl weights
    # Users should pre-convert weights using convert_weights.py
    legacy_path = getattr(args, 'file_load_path', '')
    if legacy_path and not legacy_path.endswith('.ckpt'):
        logger.warning("Legacy .pkl weights detected. Please run convert_weights.py first.")
        logger.warning(f"Expected .ckpt file, got: {legacy_path}")
        raise ValueError("Please convert legacy weights to Lightning format first using convert_weights.py")

    return None


def main():
    """Main evaluation function."""
    logger = setup_logging()
    parser = create_argparser()
    args = parser.parse_args()

    # Initialize random seeds
    setup_init(100)

    # Create output folder name
    args.folder = f'Dataset_{args.dataset}_Task_{args.task}_FewRatio_{args.few_ratio}_{args.size}_{args.note}/'
    args.folder = f'Eval_{args.folder}'

    if args.mask_strategy_random != 'batch':
        args.folder = f'{args.mask_strategy}_{args.mask_ratio}_{args.folder}'

    # Create output directories
    model_path = os.path.join(args.output_dir, args.folder)
    os.makedirs(model_path, exist_ok=True)

    logger.info(f"Output directory: {model_path}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint not found: {args.checkpoint_path}")
        logger.info("Please ensure the checkpoint exists or run convert_weights.py to convert legacy weights.")
        sys.exit(1)

    # Initialize DataModule
    logger.info("Initializing DataModule...")
    data_module = create_wifo_data_module(args)
    data_module.setup(stage='test')

    # Initialize Lightning Module (will load weights from checkpoint)
    logger.info("Loading WiFo Lightning Module from checkpoint...")
    model = WiFoLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        args=args,
        map_location='cpu'  # Load to CPU first, then move to GPU
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create Trainer for evaluation
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices if args.devices != 'auto' else 'auto',
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    logger.info("Starting evaluation...")
    logger.info(f"Evaluation strategy: {args.mask_strategy}")
    logger.info(f"Mask ratio: {args.mask_ratio}")

    # Run evaluation
    if len(data_module.test_datasets) > 0:
        test_results = trainer.test(model, datamodule=data_module)

        # Parse and log results
        logger.info("=" * 80)
        logger.info("Evaluation Results:")
        for i, dataset_name in enumerate(data_module.dataset_names):
            # Get results for this dataset
            result_key = f'test/dataloader_{i}/nmse'
            if result_key in test_results[0]:
                nmse = test_results[0][result_key]
                logger.info(f"  {dataset_name:10s} | NMSE = {nmse:.6f}")

        # Calculate average NMSE
        nmse_values = [
            test_results[0][f'test/dataloader_{i}/nmse']
            for i in range(len(data_module.dataset_names))
            if f'test/dataloader_{i}/nmse' in test_results[0]
        ]
        if nmse_values:
            avg_nmse = np.mean(nmse_values)
            logger.info("=" * 80)
            logger.info(f"Average NMSE: {avg_nmse:.6f}")
            logger.info("=" * 80)

        # Save results to file
        result_file = os.path.join(model_path, 'result.txt')
        with open(result_file, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Strategy: {args.mask_strategy}\n")
            f.write(f"Mask ratio: {args.mask_ratio}\n")
            f.write(f"Datasets: {args.dataset}\n")
            f.write(f"\nPer-dataset results:\n")
            for i, dataset_name in enumerate(data_module.dataset_names):
                result_key = f'test/dataloader_{i}/nmse'
                if result_key in test_results[0]:
                    nmse = test_results[0][result_key]
                    f.write(f"  {dataset_name}: {nmse:.6f}\n")
            if nmse_values:
                f.write(f"\nAverage NMSE: {avg_nmse:.6f}\n")

        logger.info(f"Results saved to {result_file}")
    else:
        logger.warning("No test datasets available for evaluation.")


if __name__ == "__main__":
    main()
