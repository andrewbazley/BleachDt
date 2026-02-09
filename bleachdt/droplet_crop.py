"""
Droplet detection, cropping, and stabilization module.

This module detects droplets in FRAP image stacks, classifies them as
FRAP or NoBleach based on intensity changes, crops individual droplets,
and optionally stabilizes them using drift correction.

Example
-------
From command line:
    $ bleachdt-crop /path/to/movies --stabilize --min-area 500

From Python:
    >>> from bleachdt import droplet_crop
    >>> droplet_crop.main()
"""

import sys
import os

# Ensure parent directory is in path for importing original modules
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from original script - this makes all functions available
try:
    from Droplet_Crop import (
        main,
        process_movie,
        crop_and_stabilize_droplets,
        stabilize_stack_ecc,
        detect_objects,
        read_image_stack,
        save_tiff,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import Droplet_Crop module. Ensure Droplet_Crop.py is in the package directory. Error: {e}"
    )

__all__ = [
    "main",
    "process_movie",
    "crop_and_stabilize_droplets",
    "stabilize_stack_ecc",
    "detect_objects",
    "read_image_stack",
    "save_tiff",
]

if __name__ == "__main__":
    main()
