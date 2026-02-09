"""
FRAP curve combination and filtering module.

This module combines FRAP curves from multiple experiments, applies
arithmetic filtering, and generates summary statistics.

Example
-------
From command line:
    $ bleachdt-combine /path/to/results --max-drop 0.05 --max-jump 0.05

From Python:
    >>> from bleachdt import combine
    >>> combine.main()
"""

import sys
import os

# Ensure parent directory is in path for importing original modules
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from original script
try:
    from combine_frap_curves import (
        main,
        find_all_frap_curves_csvs,
        read_and_process_csv,
        transpose_and_label,
        combine_condition_csvs,
        filter_curves_arithmetic,
        pool_filtered_by_condition,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import combine_frap_curves module. Error: {e}"
    )

__all__ = [
    "main",
    "find_all_frap_curves_csvs",
    "read_and_process_csv",
    "transpose_and_label",
    "combine_condition_csvs",
    "filter_curves_arithmetic",
    "pool_filtered_by_condition",
]

if __name__ == "__main__":
    main()
