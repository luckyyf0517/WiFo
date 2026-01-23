# coding=utf-8
"""
Weight Conversion Script for WiFo (Wireless Foundation Model)

This script converts legacy .pkl model weights to PyTorch Lightning checkpoint format.

Usage:
    python scripts/convert_weights.py --input weights/release/wifo_base.pkl

The converted checkpoint will be saved to: weights/lightning/wifo_{size}.ckpt

Lightning checkpoint format includes:
- model state dict only (no hyperparameters)
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

    parser.add_argument('--input', type=str, required=True,
                       help='Path to input .pkl file (e.g., weights/release/wifo_base.pkl)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for .ckpt file (default: weights/lightning/{filename}.ckpt)')

    return parser


def auto_detect_size(input_path: str) -> str:
    """
    Auto-detect model size from filename.

    Args:
        input_path: Path to input .pkl file

    Returns:
        Model size (tiny, little, small, base)
    """
    filename = os.path.basename(input_path).lower()

    if 'tiny' in filename:
        return 'tiny'
    elif 'little' in filename:
        return 'little'
    elif 'small' in filename:
        return 'small'
    elif 'base' in filename:
        return 'base'
    elif 'large' in filename:
        return 'large'
    else:
        raise ValueError(f"Could not auto-detect model size from filename: {filename}")


def convert_weights(input_path: str, output_path: str = None):
    """
    Convert legacy .pkl weights to Lightning checkpoint format.

    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .ckpt file
    """
    logger = setup_logging()

    # Check input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Auto-detect size
    size = auto_detect_size(input_path)
    logger.info(f"Auto-detected model size: {size}")

    # Determine output path
    if output_path is None:
        output_dir = 'weights/lightning'
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.ckpt")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Converting weights from {input_path}")
    logger.info(f"Output: {output_path}")

    # Load legacy weights
    logger.info("Loading legacy weights...")
    try:
        legacy_state_dict = torch.load(input_path, map_location='cpu', weights_only=False)
        logger.info(f"Loaded {len(legacy_state_dict)} parameters from legacy checkpoint")
    except Exception as e:
        logger.error(f"Failed to load legacy weights: {e}")
        sys.exit(1)

    # Add 'model.' prefix to all keys for WiFoLightningModule compatibility
    logger.info("Adding 'model.' prefix to all keys for LightningModule compatibility...")
    lightning_state_dict = {}
    for k, v in legacy_state_dict.items():
        lightning_state_dict[f'model.{k}'] = v
    logger.info(f"Converted {len(lightning_state_dict)} parameter keys")

    # Save as Lightning checkpoint
    logger.info("Saving Lightning checkpoint...")
    try:
        checkpoint = {
            'state_dict': lightning_state_dict,
            'epoch': 0,
            'global_step': 0,
        }
        torch.save(checkpoint, output_path)

        logger.info(f"Successfully saved checkpoint to: {output_path}")

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

    # Run conversion
    convert_weights(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
