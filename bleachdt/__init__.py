"""
BleachDt - FRAP Diffusion Analysis Pipeline

A comprehensive toolkit for analyzing Fluorescence Recovery After Photobleaching
(FRAP) experiments in phase-separated droplets. The name "BleachDt" (pronounced
"bleached") is a nod to the PDE-based diffusion coefficient (D) fitting at the
heart of this analysis.

Modules
-------
droplet_crop : Droplet detection, cropping, and stabilization
crop_pde : PDE-based diffusion coefficient fitting
combine : FRAP curve combination and filtering
gui : Graphical user interface
condition_utils : Condition name extraction utilities

Example
-------
Launch the GUI:
    >>> from bleachdt import gui
    >>> gui.main()

Or from command line:
    $ bleachdt-gui
    $ bleachdt-analysis /path/to/movies --pixel-size 0.11
    $ bleachdt-combine /path/to/results --max-drop 0.05
"""

__version__ = "3.0.0"
__author__ = "Andrew Bazley"
__email__ = "andrew.bazley@nyulangone.org"

# Lazy imports to avoid loading heavy dependencies on import
def __getattr__(name):
    """Lazy import modules to speed up initial import."""
    if name == "droplet_crop":
        from . import droplet_crop
        return droplet_crop
    elif name == "crop_pde":
        from . import crop_pde
        return crop_pde
    elif name == "combine":
        from . import combine
        return combine
    elif name == "gui":
        from . import gui
        return gui
    elif name == "condition_utils":
        from . import condition_utils
        return condition_utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "droplet_crop",
    "crop_pde", 
    "combine",
    "gui",
    "condition_utils",
]
