"""
LLSF: Learning Label-Specific Features for Multi-Label Classification.

A scikit-learn compatible implementation of the LLSF algorithm.
"""

from .llsf import LLSFClassifier
from .version import __version__

__all__ = ['LLSFClassifier', '__version__']
