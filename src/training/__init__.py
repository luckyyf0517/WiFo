# coding=utf-8
"""
WiFo Training Components

This module contains PyTorch Lightning training components for WiFo.
"""

# Import with path setup for scripts running from project root
import os
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.lightning_module import WiFoLightningModule, create_wifo_lightning_module

__all__ = ['WiFoLightningModule', 'create_wifo_lightning_module']
