"""
BleachDt - FRAP Diffusion Analysis Pipeline

A comprehensive toolkit for analyzing Fluorescence Recovery After Photobleaching
(FRAP) experiments in phase-separated droplets. The name "BleachDt" (pronounced
"bleached") is a nod to the PDE-based diffusion coefficient (D) fitting at the
heart of this analysis.

Modules
-------
droplet_crop       : Droplet detection, cropping, and stabilization
crop_pde           : PDE-based diffusion coefficient fitting (core analysis engine)
combine            : FRAP curve combination and filtering
gui                : Graphical user interface (orchestrates the full pipeline)
condition_utils    : Condition / replicate name extraction utilities
r2_scatter         : R^2 summaries for FRAP scatter relationships
pre_mean_vs_radius : Pre-bleach mean intensity vs droplet radius analysis

Example
-------
Launch the GUI:
    >>> from bleachdt import gui
    >>> gui.main()

Or from the command line:
    $ bleachdt-gui
    $ bleachdt-analysis /path/to/movies --pixel-size 0.11
    $ bleachdt-combine /path/to/results --max-drop 0.05
    $ bleachdt-crop /path/to/movies --stabilize
"""

__version__ = "4.1.0"
__author__ = "Andrew Bazley"
__email__ = "andrew.bazley@nyulangone.org"

# Lazy imports so that simply importing `bleachdt` does not pull in heavy
# dependencies (numba, Qt, etc.) until a module is actually used.
_LAZY_MODULES = (
    "droplet_crop",
    "crop_pde",
    "combine",
    "gui",
    "condition_utils",
    "r2_scatter",
    "pre_mean_vs_radius",
)


def __getattr__(name):
    """Lazy import submodules to speed up `import bleachdt`."""
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_MODULES))


__all__ = ["__version__", *_LAZY_MODULES]
