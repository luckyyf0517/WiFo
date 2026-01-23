# coding=utf-8 -*-
"""
Configuration Validator for WiFo

This module validates configuration parameters to ensure they meet requirements.
"""

import logging
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)


# Validation rules
VALIDATION_RULES: Dict[str, Callable[[Any], bool]] = {
    # Model size validation
    'model.size': lambda x: x in ['tiny', 'little', 'small', 'base'],

    # Positional embedding validation
    'model.pos_emb': lambda x: x in ['SinCos', 'trivial', 'SinCos_3D'],

    # Masking strategy validation
    'training.mask.strategy': lambda x: x in ['random', 'temporal', 'frequency', 'fre'],
    'training.mask.strategy_mode': lambda x: x in ['batch', 'none'],

    # Learning rate validation
    'training.optimizer.lr': lambda x: isinstance(x, (int, float)) and x > 0,
    'training.optimizer.min_lr': lambda x: isinstance(x, (int, float)) and x >= 0,
    'training.lr': lambda x: isinstance(x, (int, float)) and x > 0,
    'training.min_lr': lambda x: isinstance(x, (int, float)) and x >= 0,

    # Batch size validation
    'training.batch_size': lambda x: isinstance(x, int) and x > 0,
    'data.num_workers': lambda x: isinstance(x, int) and x >= 0,

    # Epoch validation
    'training.scheduler.total_epochs': lambda x: isinstance(x, int) and x > 0,
    'trainer.max_epochs': lambda x: isinstance(x, int) and x > 0,

    # Seed validation
    'system.seed': lambda x: isinstance(x, int) and x >= 0,
}


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration against validation rules.

    Args:
        config: Configuration dictionary (can be nested)

    Returns:
        List of validation error messages (empty if valid)

    Examples:
        >>> validate_config({'model': {'size': 'base'}})
        []
        >>> validate_config({'model': {'size': 'invalid'}})
        ["Invalid model.size: 'invalid'. Must be one of: ['tiny', 'little', 'small', 'base']"]
    """
    errors = []
    flat_config = _flatten_nested_dict(config)

    for key, value in flat_config.items():
        # Skip validation for None values
        if value is None:
            continue

        # Check if there's a validation rule for this key
        for rule_key, rule_func in VALIDATION_RULES.items():
            if key == rule_key or key.endswith('.' + rule_key):
                try:
                    if not rule_func(value):
                        # Get valid options for better error messages
                        if rule_key == 'model.size':
                            valid_opts = ['tiny', 'little', 'small', 'base']
                        elif rule_key == 'model.pos_emb':
                            valid_opts = ['SinCos', 'trivial', 'SinCos_3D']
                        elif rule_key == 'training.mask.strategy':
                            valid_opts = ['random', 'temporal', 'frequency', 'fre']
                        elif rule_key == 'training.mask.strategy_mode':
                            valid_opts = ['batch', 'none']
                        else:
                            valid_opts = None

                        if valid_opts:
                            errors.append(f"Invalid {key}: '{value}'. Must be one of: {valid_opts}")
                        else:
                            errors.append(f"Invalid {key}: '{value}'")
                except Exception as e:
                    errors.append(f"Error validating {key}: {e}")

    return errors


def _flatten_nested_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten nested dictionary to dot-notation keys for validation.

    Args:
        d: Nested dictionary
        parent_key: Current key prefix for recursion

    Returns:
        Flattened dictionary with dot-notation keys
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_nested_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def check_required_params(config: Dict[str, Any]) -> List[str]:
    """
    Check if all required parameters are present in configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of missing required parameters
    """
    required = [
        'model.size',
        'data.dataset',
    ]

    flat_config = _flatten_nested_dict(config)
    missing = [p for p in required if p not in flat_config or flat_config[p] is None]

    return missing
