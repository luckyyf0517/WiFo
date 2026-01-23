# coding=utf-8
"""
WiFo Training Components

This module contains PyTorch Lightning training components for WiFo.
"""

from .lightning_module import WiFoLightningModule, create_wifo_lightning_module

__all__ = ['WiFoLightningModule', 'create_wifo_lightning_module']
