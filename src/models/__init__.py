# coding=utf-8
"""
WiFo Model Architecture Components

This module contains the WiFo (Wireless Foundation Model) architecture
including embedding layers and the main model.
"""

# Import with path setup for scripts running from project root
import os
import sys
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.model import WiFo_model
from models.embed import DataEmbedding

__all__ = ['WiFo_model', 'DataEmbedding']
