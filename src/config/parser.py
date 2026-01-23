# coding=utf-8
"""
CLI Override Parser for WiFo Configuration

This module handles parsing command-line arguments with dot notation
for overriding nested configuration parameters.
"""

import os
import sys
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI arguments with dot notation into nested dictionary.

    Args:
        args: List of command-line arguments (e.g., ['--model.size', 'tiny', '--training.lr', '1e-3'])

    Returns:
        Dictionary with overridden values (nested structure)

    Examples:
        >>> parse_cli_overrides(['--model.size', 'tiny', '--training.lr', '1e-3'])
        {'model': {'size': 'tiny'}, 'training': {'lr': 0.001}}
    """
    overrides = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if not arg.startswith('--'):
            i += 1
            continue

        # Remove the '--' prefix
        key = arg[2:]

        # Check if there's a value following
        if i + 1 < len(args) and not args[i + 1].startswith('--'):
            value = args[i + 1]
            i += 2
        else:
            # Boolean flag, treat as True
            value = 'true'
            i += 1

        # Parse dot notation keys
        override = _set_nested_dict(overrides, key, _convert_value(value))
        overrides = merge_dicts(overrides, override)

    return overrides


def _set_nested_dict(d: Dict, key: str, value: Any) -> Dict:
    """
    Set a value in a nested dictionary using dot notation key.

    Args:
        d: Dictionary to update
        key: Dot notation key (e.g., 'model.size' or 'training.optimizer.lr')
        value: Value to set

    Returns:
        Updated dictionary
    """
    keys = key.split('.')
    current = d

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Type conversion for the final value
    current[keys[-1]] = _convert_value(value)
    return d


def _convert_value(value: str) -> Any:
    """
    Convert string value to appropriate type.

    Args:
        value: String value from CLI

    Returns:
        Converted value (int, float, bool, str, or list)
    """
    # Try boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try list (comma-separated)
    if ',' in value:
        return [_convert_value(v.strip()) for v in value.split(',')]

    # Default to string
    return value


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def flatten_config(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested dictionary to dot-notation keys.

    Args:
        config: Nested configuration dictionary
        prefix: Current key prefix (for recursion)

    Returns:
        Flattened dictionary with dot-notation keys

    Examples:
        >>> flatten_config({'model': {'size': 'base'}})
        {'model.size': 'base'}
    """
    result = {}

    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_config(value, new_key))
        else:
            result[new_key] = value

    return result


def config_to_namespace(config: Dict[str, Any]) -> Any:
    """
    Convert config dictionary to a namespace-like object.

    Args:
        config: Configuration dictionary (can be nested)

    Returns:
        Namespace object with dot access to all nested keys

    Examples:
        >>> cfg = {'model': {'size': 'base'}}
        >>> ns = config_to_namespace(cfg)
        >>> ns.model.size
        'base'
    """
    class ConfigNamespace:
        def __init__(self, config_dict: Dict[str, Any]):
            self._config = config_dict
            # Flatten for attribute access
            flat = flatten_config(config_dict)
            for key, value in flat.items():
                setattr(self, key, value)

        def to_dict(self) -> Dict[str, Any]:
            """Convert back to dictionary."""
            return self._config.copy()

        def __contains__(self, key: str) -> bool:
            return key in flatten_config(self._config)

    return ConfigNamespace(config)
