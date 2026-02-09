"""
PDE-based FRAP analysis module.

This module provides the core analysis engine for FRAP experiments,
including PDE-based diffusion coefficient fitting and exponential
plateau fitting.

Example
-------
From command line:
    $ bleachdt-analysis /path/to/movies --pixel-size 0.11 --frame-interval 10

From Python:
    >>> from bleachdt import crop_pde
    >>> crop_pde.main()
"""

import sys
import os

# Ensure parent directory is in path for importing original modules
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from original script
try:
    from Crop_PDE_Cursor_V7 import main
except ImportError as e:
    raise ImportError(
        f"Could not import Crop_PDE_Cursor_V7 module. Error: {e}"
    )

__all__ = ["main"]

if __name__ == "__main__":
    main()
