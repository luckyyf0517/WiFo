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


def parse_config_and_args(cli_args: list) -> argparse.Namespace:
    """
    Parse configuration file and CLI overrides into a namespace.

    Args:
        cli_args: List of command-line arguments (e.g., from sys.argv[1:])

    Returns:
        Namespace with all configuration parameters
    """
    # Check for help flag first
    if '-h' in cli_args or '--help' in cli_args:
        parser = create_argparser()
        parser.print_help()
        sys.exit(0)

    # Check if config file is specified
    config_path = None
    for i, arg in enumerate(cli_args):
        if arg.startswith('--config='):
            config_path = arg.split('=', 1)[1]
            break
        elif arg == '--config' and i + 1 < len(cli_args):
            config_path = cli_args[i + 1]
            break

    # Load configuration
    if config_path:
        logger.info(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        from config.loader import get_default_config
        config = get_default_config()

    # Parse CLI overrides (args that are not --config)
    override_args = []
    skip_next = False
    for i, arg in enumerate(cli_args):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('--config'):
            # Skip --config and its value
            if '=' not in arg and i + 1 < len(cli_args) and not cli_args[i + 1].startswith('--'):
                skip_next = True
            continue
        override_args.append(arg)

    if override_args:
        logger.info("Applying CLI overrides...")
        cli_overrides = parse_cli_overrides(override_args)
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

    # Parse config and CLI overrides from sys.argv
    args = parse_config_and_args(sys.argv[1:])

    # Show help and exit if requested
    if '--help' in sys.argv or '-h' in sys.argv:
        return  # parse_config_and_args already showed help

    # Set process title
    import setproctitle as spt
    spt.setproctitle(f"{args.system.process_name}-{args.system.device_id}")

    # Initialize random seeds
    setup_init(args.system.seed)

    # Create output folder name
    args.folder = f'{args.experiment.name}/'

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

    # Initialize Lightning Module (will load weights from checkpoint)
    logger.info("Loading WiFo Lightning Module from checkpoint...")

    # Load checkpoint manually to support simple state_dict format
    checkpoint = torch.load(args.paths.checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Create model and load state dict
    model = create_wifo_lightning_module(args)
    model.load_state_dict(state_dict, strict=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Initialize DataModule
    logger.info("Initializing DataModule...")
    data_module = create_wifo_data_module(args)

    # Create Trainer for evaluation
    trainer = L.Trainer(
        accelerator=args.trainer.accelerator,
        devices=int(args.trainer.devices) if args.trainer.devices != 'auto' else 1,
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
    )

    logger.info("Starting evaluation...")
    logger.info(f"Evaluation strategy: {args.training.mask_strategy}")
    logger.info(f"Mask ratio: {args.training.mask_ratio}")
    logger.info(f"Few-shot ratio: {args.training.few_ratio}")

    # Run evaluation
    test_results = trainer.test(model, datamodule=data_module)

    # Calculate average NMSE across all dataloaders
    nmse_values = []
    for key in test_results[0].keys():
        if 'nmse' in key:
            nmse_values.append(test_results[0][key])

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
            f.write(f"\nAverage NMSE: {avg_nmse:.6f}\n")

        logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
