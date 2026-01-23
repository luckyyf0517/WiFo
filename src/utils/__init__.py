# coding=utf-8
"""
WiFo Utility Functions

This module contains utility functions and helper classes for WiFo training.
"""

# Import with path setup for scripts running from project root
import os
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils.mask_strategy import random_masking, causal_masking, fre_masking

__all__ = ['random_masking', 'causal_masking', 'fre_masking']
