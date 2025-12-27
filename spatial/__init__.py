"""
Utilities for preparing the Spatial galaxy dataset and shared constants.
Inference helpers are available in `spatial.inference`.
"""

from .data import CLASS_NAMES, prepare_dataset

__all__ = ["CLASS_NAMES", "prepare_dataset"]
