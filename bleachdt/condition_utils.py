"""
Condition name extraction utilities.

This module provides utilities for extracting experimental condition
names from file and folder names.

Example
-------
    >>> from bleachdt.condition_utils import extract_condition_key
    >>> extract_condition_key("WT_ADP_001_results")
    'WT_ADP'
"""

import sys
import os

# Ensure parent directory is in path for importing original modules
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from original script
try:
    from condition_utils import extract_condition_key
except ImportError as e:
    raise ImportError(
        f"Could not import condition_utils module. Error: {e}"
    )

__all__ = ["extract_condition_key"]
