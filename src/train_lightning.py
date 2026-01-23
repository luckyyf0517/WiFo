# coding=utf-8
"""
PyTorch Lightning Training Entry Point for WiFo (Wireless Foundation Model)

This script provides the main entry point for training the WiFo model using
PyTorch Lightning framework, following the specifications from the paper:
"Liu et al., WiFo: wireless foundation model for channel prediction,
SCIENCE CHINA Information Sciences, 2025"

Usage:
    python src/train_lightning.py --dataset DS1 --size base --epochs 200

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
import random
import logging
from typing import Optional

import torch
import numpy as np
import setproctitle
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from lightning_module import WiFoLightningModule, create_wifo_lightning_module
from data_module import WiFoDataModule, create_wifo_data_module

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
        file_load_path='',
        dataset='DS1',
        data_path='dataset/',
        process_name='wifo_lightning',
        his_len=6,
        pred_len=6,
        few_ratio=0.5,
        stage=0,

        # Model settings
        mask_ratio=0.5,
        patch_size=4,
        t_patch_size=2,
        size='base',
        no_qkv_bias=0,
        pos_emb='SinCos',
        conv_num=3,

        # Pretrain settings
        random=True,
        mask_strategy='random',
        mask_strategy_random='batch',  # 'none' or 'batch'

        # Training hyperparameters (from paper)
        lr=5e-4,  # Base learning rate from paper
        min_lr=1e-5,  # Minimum learning rate from paper
        early_stop=5,
        weight_decay=0.05,  # Weight decay from paper
        batch_size=128,  # Batch size from paper
        log_interval=5,
        total_epoches=200,  # Total epochs from paper
        warmup_steps=5,  # Warmup epochs from paper
        device_id='0',
        machine='localhost',
        clip_grad=0.05,  # Gradient clipping from paper
        lr_anneal_steps=200,  # LR scheduler steps from paper

        # Lightning-specific settings
        num_workers=32,
        pin_memory=True,
        prefetch_factor=4,
        precision='32-true',
        accelerator='auto',
        devices='auto',
        max_epochs=200,  # From paper
        check_val_every_n_epoch=1,
        log_every_n_steps=10,

        # Paths
        output_dir='./experiments',
        log_dir='./logs',
    )

    parser = argparse.ArgumentParser(description='WiFo Training with PyTorch Lightning')

    # Add all arguments
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true' if not value else 'store_false',
                             default=value)
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser


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
    args = parser.parse_args()

    # Set process title
    setproctitle.setproctitle(f"{args.process_name}-{args.device_id}")

    # Initialize random seeds
    setup_init(100)

    # Create output folder name
    args.folder = f'Dataset_{args.dataset}_Task_{args.task}_FewRatio_{args.few_ratio}_{args.size}_{args.note}/'
    args.folder = f'Train_{args.folder}'

    if args.mask_strategy_random != 'batch':
        args.folder = f'{args.mask_strategy}_{args.mask_ratio}_{args.folder}'

    # Create output directories
    model_path = os.path.join(args.output_dir, args.folder)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Output directory: {model_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

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
    if args.file_load_path:
        checkpoint_path = f"{args.file_load_path}.pkl"
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            model.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained model from {checkpoint_path}")
        else:
            logger.warning(f"Pretrained model not found: {checkpoint_path}")

    # Create TensorBoard logger
    log_dir = os.path.join(args.log_dir, args.folder)
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
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices if args.devices != 'auto' else 'auto',
        precision=args.precision,
        callbacks=callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=True,
        gradient_clip_val=args.clip_grad,
        gradient_clip_algorithm='norm',
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    logger.info("Starting training...")
    logger.info(f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, "
               f"epochs={args.max_epochs}, weight_decay={args.weight_decay}")

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
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Epoch: {trainer.current_epoch}\n")
        f.write(f"Best validation NMSE: {model.best_val_nmse:.6f}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")

    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
