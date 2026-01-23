# coding=utf-8
"""
YAML Configuration Loader for WiFo

This module handles loading and saving YAML configuration files.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str, required: bool = True) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file
        required: Whether the config file is required (default: True)

    Returns:
        Dictionary containing the loaded configuration

    Raises:
        FileNotFoundError: If config file not found and required=True
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(config_path):
        if required:
            logger.error(f"Configuration file not found: {config_path}")
            # Suggest available config files
            config_dir = os.path.dirname(config_path)
            if os.path.exists(config_dir):
                available = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
                if available:
                    logger.info(f"Available config files in {config_dir}:")
                    for f in sorted(available):
                        logger.info(f"  - {f}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.info(f"Successfully loaded configuration with {len(config)} top-level keys")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file: {e}")
        raise


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where the YAML file should be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to: {output_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration with all hardcoded values.

    Returns:
        Dictionary containing default configuration
    """
    return {
        'experiment': {
            'name': 'wifo_training',
            'tags': [],
            'note': '',
        },
        'model': {
            'size': 'base',
            'patch_size': 4,
            't_patch_size': 2,
            'pos_emb': 'SinCos',
            'no_qkv_bias': 0,
            'conv_num': 3,
        },
        'training': {
            'optimizer': {
                'name': 'adamw',
                'lr': 5e-4,
                'min_lr': 1e-5,
                'weight_decay': 0.05,
                'betas': [0.9, 0.999],
            },
            'scheduler': {
                'name': 'cosine',
                'warmup_epochs': 5,
                'total_epochs': 200,
            },
            'batch_size': 128,
            'gradient_clip': 0.05,
            'early_stop': 5,
            'few_ratio': 0.5,
            'mask': {
                'strategy': 'random',
                'strategy_mode': 'batch',
                'ratio': 0.5,
            },
            # Legacy aliases for backward compatibility with code
            'mask_strategy': 'random',
            'mask_strategy_random': 'batch',
            'mask_ratio': 0.5,
            # Legacy flat parameters for backward compatibility
            'lr': 5e-4,
            'min_lr': 1e-5,
            'weight_decay': 0.05,
            'warmup_steps': 5,
            'lr_anneal_steps': 200,
        },
        'data': {
            'dataset': 'DS1',
            'data_path': 'dataset/',
            'train_split': 9000,
            'val_split': 1000,
            'test_split': 2000,
            'num_workers': 32,
            'pin_memory': True,
            'prefetch_factor': 4,
        },
        'trainer': {
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': '32-true',
            'max_epochs': 200,
            'check_val_every_n_epoch': 1,
            'log_every_n_steps': 10,
        },
        'paths': {
            'output_dir': './experiments',
            'log_dir': './logs',
            'checkpoint_path': '',
        },
        'system': {
            'seed': 100,
            'device_id': '0',
            'process_name': 'wifo_training',
        },
        # Legacy flat parameters for backward compatibility
        'note': '',
        'task': 'short',
        'file_load_path': '',
        'his_len': 6,
        'pred_len': 6,
        'few_ratio': 0.5,
        'stage': 0,
        'mask_ratio': 0.5,
        'mask_strategy': 'random',
        'mask_strategy_random': 'batch',
        'log_interval': 5,
        'total_epoches': 200,
        'machine': 'localhost',
        'clip_grad': 0.05,
    }


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base config (deep merge).

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    def deep_merge(base_dict: Dict, override_dict: Dict) -> Dict:
        """Recursively merge override_dict into base_dict."""
        result = base_dict.copy()

        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    return deep_merge(base, override)
