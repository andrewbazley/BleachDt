"""
BleachDt GUI module.

This module provides the graphical user interface for the BleachDt
FRAP Diffusion Analysis Pipeline, including the Analysis tab for 
running analyses and the Filtering tab for interactive scatter plot filtering.

Example
-------
From command line:
    $ bleachdt-gui

From Python:
    >>> from bleachdt import gui
    >>> gui.main()

Requirements
------------
This module requires a Qt backend (PySide6, PyQt6, or PyQt5) and pyqtgraph.
Install with:
    pip install bleachdt[gui]
"""

import sys
import os

# Ensure parent directory is in path for importing original modules
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from original script
try:
    from combine_frap_curves_gui import main, MainWindow
except ImportError as e:
    raise ImportError(
        f"Could not import combine_frap_curves_gui module. "
        f"Make sure Qt (PySide6, PyQt6, or PyQt5) is installed. Error: {e}"
    )

__all__ = ["main", "MainWindow"]

if __name__ == "__main__":
    main()
