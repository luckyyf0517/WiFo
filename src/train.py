# coding=utf-8
"""
PyTorch Lightning Training Entry Point for WiFo (Wireless Foundation Model)

This script provides the main entry point for training the WiFo model using
PyTorch Lightning framework, following the specifications from the paper:
"Liu et al., WiFo: wireless foundation model for channel prediction,
SCIENCE CHINA Information Sciences, 2025"

Usage:
    # With config file
    python src/train.py --config configs/base_training.yaml

    # With config file + override
    python src/train.py --config configs/base.yaml --model.size tiny --data.dataset DS2

    # Without config (backward compatible)
    python src/train.py --data.dataset DS1 --model.size base

Hyperparameters from the paper:
- Optimizer: AdamW with betas=(0.9, 0.999), weight_decay=0.05
- Base Learning Rate: 5e-4
- Min Learning Rate: 1e-5
- Batch Size: 128
- Total Epochs: 200
- Warmup Epochs: 5
- LR Schedule: Cosine decay
- Masking Ratios: Random=85%, Temporal=50%, Frequency=50%
"""

import argparse
import os
import sys
import random
import logging
from typing import Optional

# Add src directory to path to allow imports when running from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import numpy as np
import setproctitle
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
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
        description='WiFo Training with PyTorch Lightning',
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


def create_callbacks(args) -> list:
    """
    Create PyTorch Lightning callbacks.

    Args:
        args: Configuration namespace

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint - save best model based on validation NMSE
    checkpoint_callback = ModelCheckpoint(
        monitor='val/nmse',
        filename='best-{epoch:02d}-{val/nmse:.6f}',
        save_top_k=1,
        mode='min',
        save_last=True,
        dirpath=os.path.join(args.output_dir, args.folder, 'checkpoints')
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Early stopping (optional, disabled by default as paper trains for fixed epochs)
    if args.early_stop > 0:
        early_stop = EarlyStopping(
            monitor='val/nmse',
            patience=args.early_stop,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)

    logger.info(f"Created {len(callbacks)} callbacks")
    return callbacks


def main():
    """Main training function."""
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
    setproctitle.setproctitle(f"{args.system.process_name}-{args.system.device_id}")

    # Initialize random seeds
    setup_init(args.system.seed)

    # Create output folder name
    args.folder = f'Dataset_{args.data.dataset}_Task_{args.task}_FewRatio_{args.training.few_ratio}_{args.model.size}_{args.experiment.note}/'
    args.folder = f'Train_{args.folder}'

    if args.training.mask_strategy_random != 'batch':
        args.folder = f'{args.training.mask_strategy}_{args.training.mask_ratio}_{args.folder}'

    # Create output directories
    model_path = os.path.join(args.paths.output_dir, args.folder)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Output directory: {model_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Save active config to output directory for reproducibility
    save_config(args.to_dict(), os.path.join(model_path, 'config.yaml'))

    # Initialize DataModule
    logger.info("Initializing DataModule...")
    data_module = create_wifo_data_module(args)
    data_module.setup(stage='fit')

    # Initialize Lightning Module
    logger.info("Initializing WiFo Lightning Module...")
    model = create_wifo_lightning_module(args)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load pretrained weights if specified
    if args.paths.checkpoint_path:
        # Check if it's a Lightning checkpoint (.ckpt) or legacy pickle (.pkl)
        if args.paths.checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint
            logger.info(f"Loading Lightning checkpoint from: {args.paths.checkpoint_path}")
            model = WiFoLightningModule.load_from_checkpoint(
                args.paths.checkpoint_path,
                args=args,
                map_location='cpu'
            )
        elif args.paths.checkpoint_path or args.file_load_path:
            # Legacy pickle
            ckpt_path = args.paths.checkpoint_path or f"{args.file_load_path}.pkl"
            if os.path.exists(ckt_path):
                state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                model.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded legacy weights from {ckpt_path}")
            else:
                logger.warning(f"Checkpoint not found: {ckpt_path}")

    # Create TensorBoard logger
    log_dir = os.path.join(args.paths.log_dir, args.folder)
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=None,
        version=None,
        log_graph=False
    )
    logger.info(f"TensorBoard logs: {log_dir}")

    # Create callbacks
    callbacks = create_callbacks(args)

    # Initialize Trainer with paper hyperparameters
    trainer = L.Trainer(
        max_epochs=args.trainer.max_epochs,
        accelerator=args.trainer.accelerator,
        devices=args.trainer.devices if args.trainer.devices != 'auto' else 'auto',
        precision=args.trainer.precision,
        callbacks=callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=args.trainer.log_every_n_steps,
        check_val_every_n_epoch=args.trainer.check_val_every_n_epoch,
        deterministic=True,
        gradient_clip_val=args.training.clip_grad,
        gradient_clip_algorithm='norm',
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    logger.info("Starting training...")
    logger.info(f"Hyperparameters: lr={args.training.optimizer.lr}, batch_size={args.training.batch_size}, "
               f"epochs={args.trainer.max_epochs}, weight_decay={args.training.optimizer.weight_decay}")

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best validation NMSE: {model.best_val_nmse:.6f}")

    # Save final model
    final_model_path = os.path.join(model_path, 'model_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Run evaluation on test set if available
    try:
        logger.info("Running test evaluation...")
        test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')
        logger.info(f"Test results: {test_results}")
    except Exception as e:
        logger.warning(f"Test evaluation failed: {e}")

    # Save results to file
    result_file = os.path.join(model_path, 'result.txt')
    with open(result_file, 'w') as f:
        f.write(f"Config: {args.experiment.name}\n")
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Epoch: {trainer.current_epoch}\n")
        f.write(f"Best validation NMSE: {model.best_val_nmse:.6f}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")

    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
