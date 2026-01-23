# coding=utf-8
"""
Weight Conversion Script for WiFo (Wireless Foundation Model)

This script converts legacy .pkl model weights to PyTorch Lightning checkpoint format.

Usage:
    python scripts/convert_weights.py --size base --input weights/release/wifo_base.pkl

The converted checkpoint will be saved to: weights/lightning/wifo_{size}.ckpt

Lightning checkpoint format includes:
- model state dict
- hyperparameters
- optimizer state (empty for inference checkpoints)
- epoch number (0 for inference checkpoints)
"""

import argparse
import os
import sys
import logging

# Add project root to path for proper imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import lightning as L

# Import after path is set
from src.training.lightning_module import WiFoLightningModule


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def create_argparser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description='Convert WiFo weights to Lightning checkpoint format')

    parser.add_argument('--size', type=str, default='base',
                       choices=['tiny', 'little', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input .pkl file (e.g., weights/release/wifo_base.pkl)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for .ckpt file (default: weights/lightning/wifo_{size}.ckpt)')
    parser.add_argument('--patch_size', type=int, default=4,
                       help='Patch size (default: 4)')
    parser.add_argument('--t_patch_size', type=int, default=2,
                       help='Temporal patch size (default: 2)')
    parser.add_argument('--pos_emb', type=str, default='SinCos',
                       help='Positional embedding type (default: SinCos)')

    return parser


class ArgsNamespace:
    """Simple namespace to hold model configuration."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def convert_weights(input_path: str, output_path: str, model_size: str,
                   patch_size: int, t_patch_size: int, pos_emb: str):
    """
    Convert legacy .pkl weights to Lightning checkpoint format.

    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .ckpt file
        model_size: Model size (tiny, little, small, base, large)
        patch_size: Spatial patch size
        t_patch_size: Temporal patch size
        pos_emb: Positional embedding type
    """
    logger = setup_logging()

    # Check input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Converting weights from {input_path}")
    logger.info(f"Model size: {model_size}")
    logger.info(f"Output: {output_path}")

    # Load legacy weights
    logger.info("Loading legacy weights...")
    try:
        legacy_state_dict = torch.load(input_path, map_location='cpu', weights_only=True)
        logger.info(f"Loaded {len(legacy_state_dict)} parameters from legacy checkpoint")
    except Exception as e:
        logger.error(f"Failed to load legacy weights: {e}")
        sys.exit(1)

    # Create model configuration
    args = ArgsNamespace(
        size=model_size,
        patch_size=patch_size,
        t_patch_size=t_patch_size,
        pos_emb=pos_emb,
        no_qkv_bias=0,
        conv_num=3,
        mask_ratio=0.5,
        mask_strategy='random',
        mask_strategy_random='batch',
        lr=5e-4,
        min_lr=1e-5,
        weight_decay=0.05,
        warmup_steps=5,
        lr_anneal_steps=200,
        clip_grad=0.05,
    )

    # Initialize Lightning module (this creates the model architecture)
    logger.info("Initializing Lightning module...")
    try:
        lightning_module = WiFoLightningModule(args)
        logger.info(f"Model initialized with {model_size} configuration")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load legacy weights into the model
    logger.info("Loading legacy weights into model...")
    try:
        missing_keys, unexpected_keys = lightning_module.model.load_state_dict(
            legacy_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys}")

        logger.info("Successfully loaded weights into model")
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Count parameters
    total_params = sum(p.numel() for p in lightning_module.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Save as Lightning checkpoint
    logger.info("Saving Lightning checkpoint...")
    try:
        # Save checkpoint directly using torch.save with Lightning format
        checkpoint = {
            'state_dict': lightning_module.state_dict(),
            'hyper_parameters': lightning_module.hparams,
            'epoch': 0,
            'global_step': 0,
            'pytorch-lightning_version': L.__version__,
        }
        torch.save(checkpoint, output_path)

        logger.info(f"Successfully saved checkpoint to: {output_path}")

        # Verify the checkpoint can be loaded
        logger.info("Verifying checkpoint...")
        loaded_module = WiFoLightningModule.load_from_checkpoint(
            output_path,
            args=args,
            map_location='cpu'
        )
        logger.info("Checkpoint verification successful!")

        # Show file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Checkpoint file size: {file_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("Conversion complete!")


def main():
    """Main conversion function."""
    parser = create_argparser()
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_dir = 'weights/lightning'
        model_name = f"wifo_{args.size}"
        output_path = os.path.join(output_dir, f"{model_name}.ckpt")
    else:
        output_path = args.output

    # Run conversion
    convert_weights(
        input_path=args.input,
        output_path=output_path,
        model_size=args.size,
        patch_size=args.patch_size,
        t_patch_size=args.t_patch_size,
        pos_emb=args.pos_emb
    )


if __name__ == "__main__":
    main()
