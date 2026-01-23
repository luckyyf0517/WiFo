# coding=utf-8
"""
WiFo Configuration Management

This module provides YAML-based configuration management for training and evaluation,
with support for CLI parameter overrides using dot notation.
"""

# Import with path setup for scripts running from project root
import os
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config.loader import load_config, save_config
from config.parser import parse_cli_overrides
from config.validator import validate_config

__all__ = ['load_config', 'save_config', 'parse_cli_overrides', 'validate_config']
