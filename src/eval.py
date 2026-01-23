# coding=utf-8
"""
PyTorch Lightning Evaluation Script for WiFo (Wireless Foundation Model)

This script provides the evaluation/inference entry point for the WiFo model,
following the specifications from the paper:
"Liu et al., WiFo: wireless foundation model for channel prediction,
SCIENCE CHINA Information Sciences, 2025"

Usage:
    # With config file
    python src/eval.py --config configs/evaluation/zero_shot.yaml

    # With config file + override
    python src/eval.py --config configs/evaluation/zero_shot.yaml --data.dataset D1

    # Without config (backward compatible)
    python src/eval.py --data.dataset D1 --model.size base --paths.checkpoint_path weights/lightning/wifo_base.ckpt

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

# Add src directory to path to allow imports when running from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import numpy as np
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from config import load_config, save_config, parse_cli_overrides, validate_config
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
    Create argument parser for configuration.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='WiFo Evaluation with PyTorch Lightning',
        add_help=False
    )

    # Config file argument
    parser.add_argument(
        '--config',
        type=str,
        default='',
        help='Path to YAML configuration file'
    )

    # Add help manually so we can parse other args first
    parser.add_argument('-h', '--help', action='store_true',
                       help='show this help message and exit')

    return parser


def parse_config_and_args(known_args: list) -> argparse.Namespace:
    """
    Parse configuration file and CLI overrides into a namespace.

    Args:
        known_args: List of known command-line arguments

    Returns:
        Namespace with all configuration parameters
    """
    # Check if config file is specified
    config_path = None
    for i, arg in enumerate(known_args):
        if arg.startswith('--config=') or arg.startswith('--config '):
            config_path = arg.split('=', 1)[1] if '=' in arg else known_args[i + 1]
            break

    # Load configuration
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        from config.loader import get_default_config
        config = get_default_config()

    # Parse CLI overrides (args that are not --config)
    cli_args = [arg for arg in known_args if arg not in ['--config', '-h', '--help']]
    if cli_args:
        logger.info("Applying CLI overrides...")
        cli_overrides = parse_cli_overrides(cli_args)
        from config.loader import merge_configs
        config = merge_configs(config, cli_overrides)

    # Validate configuration
    errors = validate_config(config)
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    # Convert flattened config back to namespace-like structure
    from config.parser import config_to_namespace
    args = config_to_namespace(config)

    return args


def main():
    """Main evaluation function."""
    logger = setup_logging()
    parser = create_argparser()
    known_args, unknown_args = parser.parse_known_args()

    # Show help and exit if requested
    if known_args.help:
        parser.print_help()
        sys.exit(0)

    # Parse config and CLI overrides
    args = parse_config_and_args(known_args + unknown_args)

    # Set process title
    setproctitle = __import__('setproctitle').setproctitle
    setproctitle.setproctitle(f"{args.system.process_name}-{args.system.device_id}")

    # Initialize random seeds
    setup_init(args.system.seed)

    # Create output folder name
    args.folder = f'Dataset_{args.data.dataset}_Task_{args.task}_FewRatio_{args.training.few_ratio}_{args.model.size}_{args.experiment.note}/'
    args.folder = f'Eval_{args.folder}'

    if args.training.mask_strategy_random != 'batch':
        args.folder = f'{args.training.mask_strategy}_{args.training.mask_ratio}_{args.folder}'

    # Create output directories
    model_path = os.path.join(args.paths.output_dir, args.folder)
    os.makedirs(model_path, exist_ok=True)

    logger.info(f"Output directory: {model_path}")
    logger.info(f"Checkpoint: {args.paths.checkpoint_path}")

    # Save active config to output directory for reproducibility
    save_config(args.to_dict(), os.path.join(model_path, 'config.yaml'))

    # Check if checkpoint exists
    if not os.path.exists(args.paths.checkpoint_path):
        logger.error(f"Checkpoint not found: {args.paths.checkpoint_path}")
        logger.info("Please ensure the checkpoint exists or run convert_weights.py to convert legacy weights.")
        sys.exit(1)

    # Initialize DataModule
    logger.info("Initializing DataModule...")
    data_module = create_wifo_data_module(args)
    data_module.setup(stage='test')

    # Initialize Lightning Module (will load weights from checkpoint)
    logger.info("Loading WiFo Lightning Module from checkpoint...")
    model = WiFoLightningModule.load_from_checkpoint(
        args.paths.checkpoint_path,
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
        accelerator=args.trainer.accelerator,
        devices=args.trainer.devices if args.trainer.devices != 'auto' else 'auto',
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    logger.info("Starting evaluation...")
    logger.info(f"Evaluation strategy: {args.training.mask_strategy}")
    logger.info(f"Mask ratio: {args.training.mask_ratio}")
    logger.info(f"Few-shot ratio: {args.training.few_ratio}")

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
            f.write(f"Checkpoint: {args.paths.checkpoint_path}\n")
            f.write(f"Strategy: {args.training.mask_strategy}\n")
            f.write(f"Mask ratio: {args.training.mask_ratio}\n")
            f.write(f"Few-shot ratio: {args.training.few_ratio}\n")
            f.write(f"Datasets: {args.data.dataset}\n")
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
