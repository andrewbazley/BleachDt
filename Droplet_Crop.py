#!/usr/bin/env python3
"""
Crop and stabilize FRAP / no-bleach droplets from image stacks.

Features
--------
- Detects droplets in the pre-bleach frame (frame 0).
- Computes frame0/frame1 ratio and classifies droplets:
      FRAP if mean(frame0/frame1) > bleach_threshold
      NoBleach otherwise
- Computes partition coefficient (PC) per droplet:
      PC = droplet_mean_prebleach / background_mean_prebleach
- Keeps *NoBleach* droplets only if:
      equivalent_diameter >= min_nobleach_diameter
      partition_coefficient >= min_nobleach_pc
      AND no droplet touches the border of the cropped image.
- Crops a square stack around each droplet (all frames).
- Stabilizes each cropped stack using OpenCV ECC with a translation
  model, tuned to mimic Fiji's Image Stabilizer settings:

      transformation=Translation
      maximum_pyramid_levels=4
      template_update_coefficient=0.90
      maximum_iterations=1000
      error_tolerance=1e-8

- Saves each movie's droplets into a results folder structure:

    For file: /path/to/movies/ADP_1.nd2
    Output (default):
        /path/to/movies/Analysis Results/ADP_1_results/droplet_stacks/
            Drift_ADP_1_NoBleach_droplet001.tif
            Drift_ADP_1_FRAP_droplet002.tif
            ...

Usage
-----
    python Droplet_Crop.py INPUT_PATH [--output-root OUTPUT_ROOT]

INPUT_PATH:
  - A single .tif/.tiff/.nd2 file, OR
  - A directory containing such files.

If --output-root is omitted, creates 'Analysis Results' folder alongside inputs.
Output structure: [output-root]/[movie]_results/droplet_stacks/

Example
-------
    python Droplet_Crop.py "/path/to/movies"

    python Droplet_Crop.py "/path/to/movies" \
        --output-root "/path/to/output" \
        --min-nobleach-diameter 10 \
        --min-nobleach-pc 10

Dependencies
------------
    pip install tifffile scikit-image numpy scipy nd2 opencv-python
    
Optional (for broader format support):
    pip install aicsimageio          # For CZI, LIF, OME-TIFF, and more
    pip install aicsimageio[czi]     # Zeiss CZI support
    pip install aicsimageio[lif]     # Leica LIF support
    pip install aicsimageio[bioformats]  # Full BioFormats support (requires Java)
"""

import argparse
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import ndimage as ndi
import cv2

try:
    import tifffile
except ImportError:
    tifffile = None

# Optional: aicsimageio for broad format support (Zeiss, Leica, OME-TIFF, etc.)
try:
    from aicsimageio import AICSImage
    AICSIMAGEIO_AVAILABLE = True
except ImportError:
    AICSIMAGEIO_AVAILABLE = False

from skimage import filters, morphology, measure, segmentation

# Numba JIT compilation support (optional)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ---------------------- IMAGE FORMAT SUPPORT ---------------------- #

# Supported file extensions by reader
TIFFFILE_EXTENSIONS = {".tif", ".tiff"}
ND2_EXTENSIONS = {".nd2"}
AICSIMAGEIO_EXTENSIONS = {
    ".czi",      # Zeiss
    ".lif",      # Leica
    ".ome.tif",  # OME-TIFF
    ".ome.tiff", # OME-TIFF
    ".lsm",      # Zeiss LSM
    ".oib",      # Olympus
    ".oif",      # Olympus
    ".dv",       # DeltaVision
    ".r3d",      # DeltaVision
    ".ims",      # Imaris
}

# All supported extensions
ALL_SUPPORTED_EXTENSIONS = TIFFFILE_EXTENSIONS | ND2_EXTENSIONS | AICSIMAGEIO_EXTENSIONS


class ImageImportError(Exception):
    """Custom exception for image import errors with helpful messages."""
    pass


def get_supported_formats_message() -> str:
    """Return a message listing all supported formats and how to enable them."""
    msg = "\nSupported image formats:\n"
    msg += "  Always available:\n"
    msg += "    - TIFF/TIF (via tifffile)\n"
    msg += "  With 'nd2' package (pip install nd2):\n"
    msg += "    - ND2 (Nikon)\n"
    
    if AICSIMAGEIO_AVAILABLE:
        msg += "  With 'aicsimageio' (installed):\n"
    else:
        msg += "  With 'aicsimageio' (pip install aicsimageio):\n"
    msg += "    - CZI (Zeiss) - requires: pip install aicsimageio[czi]\n"
    msg += "    - LIF (Leica) - requires: pip install aicsimageio[lif]\n"
    msg += "    - OME-TIFF\n"
    msg += "    - LSM (Zeiss), OIB/OIF (Olympus), DV (DeltaVision)\n"
    msg += "    - Many more via BioFormats: pip install aicsimageio[bioformats]\n"
    
    return msg


def detect_file_format(path: Path) -> str:
    """
    Detect the image file format and return the appropriate reader type.
    
    Returns one of: 'tifffile', 'nd2', 'aicsimageio', or raises ImageImportError.
    """
    # Handle double extensions like .ome.tif
    name_lower = path.name.lower()
    ext = path.suffix.lower()
    
    # Check for OME-TIFF specifically
    if name_lower.endswith(".ome.tif") or name_lower.endswith(".ome.tiff"):
        if AICSIMAGEIO_AVAILABLE:
            return "aicsimageio"
        else:
            # Fall back to tifffile (will read but may lose metadata)
            warnings.warn(
                f"OME-TIFF detected but aicsimageio not available. "
                f"Reading with tifffile (metadata may be incomplete). "
                f"For full OME-TIFF support: pip install aicsimageio"
            )
            return "tifffile"
    
    # Standard TIFF
    if ext in TIFFFILE_EXTENSIONS:
        return "tifffile"
    
    # Nikon ND2
    if ext in ND2_EXTENSIONS:
        return "nd2"
    
    # Formats requiring aicsimageio
    if ext in AICSIMAGEIO_EXTENSIONS:
        if AICSIMAGEIO_AVAILABLE:
            return "aicsimageio"
        else:
            format_name = {
                ".czi": "Zeiss CZI",
                ".lif": "Leica LIF",
                ".lsm": "Zeiss LSM",
                ".oib": "Olympus OIB",
                ".oif": "Olympus OIF",
                ".dv": "DeltaVision",
                ".r3d": "DeltaVision",
                ".ims": "Imaris",
            }.get(ext, ext.upper())
            
            raise ImageImportError(
                f"\n{'='*60}\n"
                f"IMAGE IMPORT ERROR: {format_name} format not supported\n"
                f"{'='*60}\n"
                f"File: {path.name}\n\n"
                f"To enable {format_name} support, install aicsimageio:\n"
                f"    pip install aicsimageio\n\n"
                f"For Zeiss CZI specifically:\n"
                f"    pip install aicsimageio[czi]\n\n"
                f"For Leica LIF specifically:\n"
                f"    pip install aicsimageio[lif]\n\n"
                f"For full BioFormats support (requires Java):\n"
                f"    pip install aicsimageio[bioformats]\n"
                f"{'='*60}\n"
            )
    
    # Unknown format - try aicsimageio if available, otherwise error
    if AICSIMAGEIO_AVAILABLE:
        return "aicsimageio"
    else:
        raise ImageImportError(
            f"\n{'='*60}\n"
            f"IMAGE IMPORT ERROR: Unsupported file format\n"
            f"{'='*60}\n"
            f"File: {path.name}\n"
            f"Extension: {ext}\n\n"
            f"This file format is not recognized.\n"
            f"{get_supported_formats_message()}\n"
            f"{'='*60}\n"
        )


def load_stack(path: Path, verbose: bool = True) -> np.ndarray:
    """
    Load an image stack as a 3D array (T, Y, X), float32.

    Supports multiple microscopy formats:
      - TIFF/TIF (via tifffile) - always available
      - ND2 (Nikon) - requires 'nd2' package
      - CZI (Zeiss) - requires 'aicsimageio[czi]'
      - LIF (Leica) - requires 'aicsimageio[lif]'
      - OME-TIFF - requires 'aicsimageio' or falls back to tifffile
      - Many more via aicsimageio with BioFormats
    
    Parameters
    ----------
    path : Path
        Path to the image file.
    verbose : bool
        If True, print warnings about format support.
    
    Returns
    -------
    np.ndarray
        3D array (T, Y, X) as float32.
    
    Raises
    ------
    ImageImportError
        If the file cannot be read or format is not supported.
    """
    path = Path(path)
    
    if not path.exists():
        raise ImageImportError(
            f"\n{'='*60}\n"
            f"IMAGE IMPORT ERROR: File not found\n"
            f"{'='*60}\n"
            f"File: {path}\n"
            f"{'='*60}\n"
        )
    
    try:
        reader_type = detect_file_format(path)
    except ImageImportError:
        raise
    except Exception as e:
        raise ImageImportError(
            f"\n{'='*60}\n"
            f"IMAGE IMPORT ERROR: Could not detect file format\n"
            f"{'='*60}\n"
            f"File: {path.name}\n"
            f"Error: {str(e)}\n"
            f"{get_supported_formats_message()}\n"
            f"{'='*60}\n"
        )
    
    data = None
    
    try:
        if reader_type == "tifffile":
            if tifffile is None:
                raise ImageImportError(
                    f"\n{'='*60}\n"
                    f"IMAGE IMPORT ERROR: tifffile not installed\n"
                    f"{'='*60}\n"
                    f"Install with: pip install tifffile\n"
                    f"{'='*60}\n"
                )
            data = tifffile.imread(str(path))
            
        elif reader_type == "nd2":
            nd2_success = False
            
            # First, try the nd2 library
            try:
                import nd2
                with nd2.ND2File(str(path)) as f:
                    # Check if the file has time dimension
                    sizes = f.sizes
                    if verbose:
                        print(f"  ND2 dimensions: {sizes}")
                    
                    data = f.asarray()
                    nd2_success = True
                    
                    # Validate: if T dimension is missing or file has Z but no T,
                    # the data might be misinterpreted
                    if 'T' not in sizes and 'Z' in sizes:
                        if verbose:
                            print(f"  Warning: ND2 file has Z={sizes.get('Z')} but no T dimension.")
                            print(f"  Interpreting Z as time dimension.")
                        # Data is already in correct order from f.asarray()
                        
            except ImportError:
                if verbose:
                    print(f"  nd2 library not installed, trying aicsimageio...")
            except Exception as e:
                if verbose:
                    print(f"  nd2 library failed ({type(e).__name__}), trying aicsimageio fallback...")
            
            # Fallback to aicsimageio if nd2 failed
            if not nd2_success:
                if AICSIMAGEIO_AVAILABLE:
                    try:
                        img = AICSImage(path)
                        if verbose:
                            print(f"  Loading ND2 via aicsimageio: dims={img.dims}")
                        
                        # AICSImage handles dimension ordering better
                        # Get data with explicit dimension request
                        # Try to get as time series first
                        try:
                            # Request TYX if available
                            data = img.get_image_data("TYX")
                            if verbose:
                                print(f"  Got TYX data: shape={data.shape}")
                        except Exception:
                            # Fall back to full data
                            data = img.data
                            if verbose:
                                print(f"  Got raw data: shape={data.shape}")
                    except Exception as e2:
                        raise ImageImportError(
                            f"\n{'='*60}\n"
                            f"IMAGE IMPORT ERROR: ND2 file could not be read\n"
                            f"{'='*60}\n"
                            f"File: {path.name}\n\n"
                            f"Both nd2 and aicsimageio failed to read this file.\n"
                            f"nd2 error: {str(e) if 'e' in dir() else 'not installed'}\n"
                            f"aicsimageio error: {str(e2)}\n\n"
                            f"This may be a corrupted or unsupported ND2 variant.\n"
                            f"Try opening in Fiji and re-saving as TIFF.\n"
                            f"{'='*60}\n"
                        )
                else:
                    raise ImageImportError(
                        f"\n{'='*60}\n"
                        f"IMAGE IMPORT ERROR: ND2 support not available\n"
                        f"{'='*60}\n"
                        f"File: {path.name}\n\n"
                        f"ND2 (Nikon) files require either:\n"
                        f"  - pip install nd2\n"
                        f"  - pip install aicsimageio\n"
                        f"{'='*60}\n"
                    )
            
        elif reader_type == "aicsimageio":
            if not AICSIMAGEIO_AVAILABLE:
                raise ImageImportError(
                    f"\n{'='*60}\n"
                    f"IMAGE IMPORT ERROR: aicsimageio not installed\n"
                    f"{'='*60}\n"
                    f"File: {path.name}\n\n"
                    f"This format requires aicsimageio.\n"
                    f"Install with: pip install aicsimageio\n"
                    f"{'='*60}\n"
                )
            
            try:
                img = AICSImage(path)
                # AICSImage returns data in TCZYX order
                # Get the first scene if multiple scenes
                data = img.get_image_data("TCZYX", T=0)  # All timepoints, first Z-stack approach
                
                # Actually, let's get all timepoints properly
                # Shape is (T, C, Z, Y, X)
                data = img.data
                
                if verbose:
                    print(f"  Loaded via aicsimageio: {img.dims}, shape={data.shape}")
                    if hasattr(img, 'physical_pixel_sizes') and img.physical_pixel_sizes.X:
                        print(f"  Pixel size: {img.physical_pixel_sizes.X:.4f} µm")
                        
            except Exception as e:
                # Check for specific missing dependencies
                ext = path.suffix.lower()
                if ext == ".czi" and "aicspylibczi" in str(e).lower():
                    raise ImageImportError(
                        f"\n{'='*60}\n"
                        f"IMAGE IMPORT ERROR: CZI reader not installed\n"
                        f"{'='*60}\n"
                        f"File: {path.name}\n\n"
                        f"Zeiss CZI files require additional dependencies.\n"
                        f"Install with: pip install aicsimageio[czi]\n"
                        f"{'='*60}\n"
                    )
                elif ext == ".lif" and "readlif" in str(e).lower():
                    raise ImageImportError(
                        f"\n{'='*60}\n"
                        f"IMAGE IMPORT ERROR: LIF reader not installed\n"
                        f"{'='*60}\n"
                        f"File: {path.name}\n\n"
                        f"Leica LIF files require additional dependencies.\n"
                        f"Install with: pip install aicsimageio[lif]\n"
                        f"{'='*60}\n"
                    )
                else:
                    raise ImageImportError(
                        f"\n{'='*60}\n"
                        f"IMAGE IMPORT ERROR: Failed to read file\n"
                        f"{'='*60}\n"
                        f"File: {path.name}\n"
                        f"Reader: aicsimageio\n"
                        f"Error: {str(e)}\n\n"
                        f"Try installing additional format support:\n"
                        f"  pip install aicsimageio[all]\n"
                        f"{'='*60}\n"
                    )
        else:
            raise ImageImportError(
                f"\n{'='*60}\n"
                f"IMAGE IMPORT ERROR: Unknown reader type\n"
                f"{'='*60}\n"
                f"File: {path.name}\n"
                f"Reader: {reader_type}\n"
                f"{'='*60}\n"
            )
            
    except ImageImportError:
        raise
    except Exception as e:
        raise ImageImportError(
            f"\n{'='*60}\n"
            f"IMAGE IMPORT ERROR: Failed to read file\n"
            f"{'='*60}\n"
            f"File: {path.name}\n"
            f"Error type: {type(e).__name__}\n"
            f"Error: {str(e)}\n\n"
            f"This could be due to:\n"
            f"  - Corrupted file\n"
            f"  - Unsupported file variant\n"
            f"  - Missing dependencies\n"
            f"{get_supported_formats_message()}\n"
            f"{'='*60}\n"
        )
    
    if data is None:
        raise ImageImportError(
            f"\n{'='*60}\n"
            f"IMAGE IMPORT ERROR: No data loaded\n"
            f"{'='*60}\n"
            f"File: {path.name}\n"
            f"The file was opened but no image data was found.\n"
            f"{'='*60}\n"
        )
    
    data = np.asarray(data)
    
    # Normalize to (T, Y, X) - handle various dimension orders
    if data.ndim == 2:
        # Single 2D frame
        data = data[np.newaxis, ...]
    elif data.ndim == 3:
        # Could be (T, Y, X) or (Z, Y, X) or (C, Y, X)
        # Assume it's already (T, Y, X)
        pass
    elif data.ndim == 4:
        # Could be (T, C, Y, X) or (T, Z, Y, X) or (C, Z, Y, X)
        # Take first channel/Z
        if data.shape[1] <= 4:  # Likely channels
            data = data[:, 0]
        else:  # Likely Z-stack
            # Max project Z or take middle slice
            data = data[:, data.shape[1] // 2]
    elif data.ndim == 5:
        # (T, C, Z, Y, X) - take first channel, max project Z or middle slice
        data = data[:, 0]  # First channel
        if data.shape[1] > 1:
            data = data[:, data.shape[1] // 2]  # Middle Z-slice
    elif data.ndim > 5:
        # Flatten extra dimensions
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
    
    return data.astype(np.float32)


def save_stack_tiff(path: Path, stack: np.ndarray) -> None:
    """
    Save a 3D stack (T, Y, X) as a TIF.
    """
    if tifffile is None:
        raise RuntimeError("tifffile is required to save TIFF outputs (pip install tifffile).")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), stack, imagej=True)


# ------------------- DROPLET DETECTION ------------------- #

def detect_droplets_prebleach(
    frame0: np.ndarray,
    min_area: float = 50.0,
    morph_open_radius: int = 1,
    smooth_sigma: float = 1.0,
    clear_borders: bool = True,
):
    """
    Detect droplets on the pre-bleach frame (frame 0).

    Returns:
      labels  : 2D int array
      regions : list of skimage.measure.RegionProperties
    """
    img = frame0.astype(np.float32)

    # Optional Gaussian smoothing before threshold
    if smooth_sigma > 0:
        img_smooth = filters.gaussian(img, sigma=smooth_sigma)
    else:
        img_smooth = img

    # Threshold for bright droplets
    thresh = filters.threshold_otsu(img_smooth)
    mask = img_smooth > thresh

    # Morphological opening to clean speckles / thin bridges
    if morph_open_radius > 0:
        selem = morphology.disk(morph_open_radius)
        mask = morphology.opening(mask, selem)

    # Remove border-touching droplets on the full frame, if requested
    if clear_borders:
        mask = segmentation.clear_border(mask)

    # Remove tiny objects
    if min_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_area))

    labels = measure.label(mask)
    if labels.max() == 0:
        return labels, []

    regions = measure.regionprops(labels, intensity_image=img)
    return labels, regions


# ---------------- FRAP CLASSIFICATION (RATIO) ---------------- #

@njit(cache=True, fastmath=True)
def compute_ratio_frame_jit(frame0: np.ndarray, frame1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Optimized ratio computation with Numba."""
    f0 = frame0.astype(np.float32)
    f1 = frame1.astype(np.float32)
    return f0 / (f1 + eps)


def compute_ratio_frame(frame0: np.ndarray, frame1: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute ratio image = frame0 / frame1, safeguarding division.
    """
    # Ensure contiguous arrays for better performance
    f0 = np.ascontiguousarray(frame0, dtype=np.float32)
    f1 = np.ascontiguousarray(frame1, dtype=np.float32)
    
    if NUMBA_AVAILABLE:
        return compute_ratio_frame_jit(f0, f1, eps)
    else:
        return f0 / (f1 + eps)


def classify_droplets_frap_vs_nobleach(
    labels: np.ndarray,
    regions,
    ratio_image: np.ndarray,
    ratio_blur_sigma: float = 1.0,
    bleach_threshold: float = 1.2,
) -> dict[int, str]:
    """
    For each labeled droplet, compute mean(frame0/frame1) on a blurred ratio image.
    Classify as "FRAP" if mean > bleach_threshold, else "NoBleach".
    """
    # Smooth ratio for robustness
    if ratio_blur_sigma > 0:
        ratio_smooth = filters.gaussian(ratio_image, sigma=ratio_blur_sigma)
    else:
        ratio_smooth = ratio_image

    props_ratio = measure.regionprops(labels, intensity_image=ratio_smooth)
    ratio_means = {r.label: float(r.mean_intensity) for r in props_ratio}

    classes = {}
    for reg in regions:
        lab = reg.label
        rmean = ratio_means.get(lab, np.nan)
        if np.isnan(rmean):
            classes[lab] = "NoBleach"
        else:
            classes[lab] = "FRAP" if rmean > bleach_threshold else "NoBleach"
    return classes


# --------------- PARTITION COEFFICIENT ----------------- #

def compute_partition_coefficients(
    frame0: np.ndarray,
    labels: np.ndarray,
    regions,
    eps: float = 1e-6,
) -> dict[int, float]:
    """
    Compute partition coefficient per droplet label:

        PC = mean_intensity_droplet / mean_intensity_background

    Background is defined as the mean of all pixels with label == 0.
    """
    mask_bg = labels == 0
    bg_vals = frame0[mask_bg]
    if bg_vals.size == 0:
        bg_mean = float(frame0.mean())
    else:
        bg_mean = float(bg_vals.mean())

    pcs = {}
    for reg in regions:
        lab = reg.label
        droplet_mean = float(reg.mean_intensity)
        pcs[lab] = droplet_mean / (bg_mean + eps)

    return pcs


# ---------------------- CROPPING ---------------------- #

@njit(cache=True)
def check_border_labels_jit(labels_crop: np.ndarray) -> bool:
    """Optimized border label check with Numba."""
    H, W = labels_crop.shape
    if H == 0 or W == 0:
        return False
    
    # Check top border
    for x in range(W):
        if labels_crop[0, x] > 0:
            return True
    
    # Check bottom border
    for x in range(W):
        if labels_crop[H-1, x] > 0:
            return True
    
    # Check left border
    for y in range(H):
        if labels_crop[y, 0] > 0:
            return True
    
    # Check right border
    for y in range(H):
        if labels_crop[y, W-1] > 0:
            return True
    
    return False


def crop_stack_around_region(
    stack: np.ndarray,
    region,
    crop_multiplier: float = 2.0,
    labels: np.ndarray | None = None,
):
    """
    Crop a square stack around the region's centroid.

    stack  : (T, Y, X)
    region : RegionProperties from frame0
    labels : full-frame label image (same Y,X shape as frame0); if provided,
             we will also return a flag indicating whether any droplet label
             touches the border of the crop in the label image.

    Returns:
      cropped_stack      : (T, Hc, Wc)
      border_has_labels  : bool (True if any labeled droplet touches crop border)
    """
    T, H, W = stack.shape
    cy, cx = region.centroid  # (row, col)
    equiv_d = float(region.equivalent_diameter)

    side = max(4, int(np.floor(crop_multiplier * equiv_d)))
    half = side / 2.0

    y0 = int(np.round(cy - half))
    x0 = int(np.round(cx - half))
    y1 = y0 + side
    x1 = x0 + side

    # Clamp to bounds
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)

    # Ensure stack is contiguous for better performance
    stack = np.ascontiguousarray(stack, dtype=np.float32)
    cropped = stack[:, y0:y1, x0:x1]

    # Border-label check on the *label image*
    border_has_labels = False
    if labels is not None:
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        labels_crop = labels[y0:y1, x0:x1]
        if labels_crop.size > 0:
            if NUMBA_AVAILABLE:
                border_has_labels = check_border_labels_jit(labels_crop)
            else:
                # Fallback to original method
                top    = labels_crop[0, :]
                bottom = labels_crop[-1, :]
                left   = labels_crop[:, 0]
                right  = labels_crop[:, -1]
                if (np.any(top > 0) or np.any(bottom > 0) or
                    np.any(left > 0) or np.any(right > 0)):
                    border_has_labels = True

    # Pad if near edges so all crops have the same shape
    ch, cw = cropped.shape[1:]
    pad_y = side - ch
    pad_x = side - cw

    if pad_y > 0 or pad_x > 0:
        pad_before_y = pad_y // 2
        pad_after_y = pad_y - pad_before_y
        pad_before_x = pad_x // 2
        pad_after_x = pad_x - pad_before_x
        cropped = np.pad(
            cropped,
            ((0, 0),
             (pad_before_y, pad_after_y),
             (pad_before_x, pad_after_x)),
            mode="edge",
        )

    return cropped, border_has_labels


# ----------------- IMAGE STABILIZATION ----------------- #
# 
# Two algorithms are available:
#   1. ECC (Enhanced Correlation Coefficient) - OpenCV's findTransformECC
#   2. Lucas-Kanade - Matches ImageJ's Image Stabilizer plugin algorithm
#
# The ImageJ Image Stabilizer uses Lucas-Kanade optical flow for motion estimation.
# Reference: K. Li, "The image stabilizer plugin for ImageJ", CMU, 2008
#            http://www.cs.cmu.edu/~kangli/code/Image_Stabilizer.html

def stabilize_stack_lucas_kanade(
    stack: np.ndarray,
    template_update_coeff: float = 0.90,
    pyramid_levels: int = 4,
    max_iterations: int = 1000,
    epsilon: float = 1e-8,
    pre_smooth_sigma: float = 0.5,
    max_shift_pixels: float = 10.0,
) -> np.ndarray:
    """
    Stabilize a stack using Lucas-Kanade optical flow - matches ImageJ Image Stabilizer.
    
    This implementation mimics the ImageJ Image Stabilizer plugin by Kang Li (CMU),
    which uses the Lucas-Kanade algorithm for motion estimation with pyramid levels
    and template updating.
    
    Parameters
    ----------
    stack : (T, Y, X), float32
        Input image stack.
    template_update_coeff : float
        Template update coefficient 'a' where:
        new_template = a * old_template + (1-a) * aligned_slice
        Higher values (closer to 1) make template update slower.
        ImageJ default: 0.90
    pyramid_levels : int
        Number of pyramid levels for coarse-to-fine alignment.
        ImageJ default: 4
    max_iterations : int
        Maximum iterations for Lucas-Kanade at each pyramid level.
        ImageJ default: 1000
    epsilon : float
        Convergence threshold (error tolerance).
        ImageJ default: 1e-8
    pre_smooth_sigma : float
        Gaussian smoothing before alignment (0 = none).
    max_shift_pixels : float
        Maximum allowed shift - larger shifts are rejected.
        
    Returns
    -------
    stabilized : (T, Y, X), float32
        Stabilized stack.
    """
    T, H, W = stack.shape
    stabilized = np.empty_like(stack, dtype=np.float32)
    
    # Initialize template with first frame
    if pre_smooth_sigma > 0:
        template = filters.gaussian(stack[0], sigma=pre_smooth_sigma).astype(np.float32)
    else:
        template = stack[0].astype(np.float32)
    stabilized[0] = template.copy()
    
    alpha = float(template_update_coeff)
    
    # Lucas-Kanade parameters matching ImageJ settings
    lk_params = dict(
        winSize=(21, 21),  # Window size for optical flow
        maxLevel=max(0, pyramid_levels - 1),  # Pyramid levels (0-indexed)
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                  max_iterations, epsilon)
    )
    
    prev_tx, prev_ty = 0.0, 0.0
    
    for t in range(1, T):
        frame = stack[t].astype(np.float32)
        
        if pre_smooth_sigma > 0:
            frame_smooth = filters.gaussian(frame, sigma=pre_smooth_sigma).astype(np.float32)
        else:
            frame_smooth = frame
        
        # Convert to 8-bit for Lucas-Kanade (OpenCV requirement)
        template_8u = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_8u = cv2.normalize(frame_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detect feature points in template using grid sampling
        # (ImageJ uses dense sampling; we use a grid for efficiency)
        grid_step = max(10, min(H, W) // 20)
        points = []
        for y in range(grid_step, H - grid_step, grid_step):
            for x in range(grid_step, W - grid_step, grid_step):
                points.append([x, y])
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        if len(points) < 10:
            # Fall back to corner detection if grid is too sparse
            points = cv2.goodFeaturesToTrack(template_8u, maxCorners=500, 
                                              qualityLevel=0.01, minDistance=10)
        
        if points is None or len(points) < 4:
            # Cannot track - use previous shift
            shift = (prev_ty, prev_tx)
            aligned = ndi.shift(frame, shift=shift, order=1, mode="nearest")
            stabilized[t] = aligned
            template = alpha * template + (1.0 - alpha) * aligned
            continue
        
        # Calculate optical flow (Lucas-Kanade)
        try:
            new_points, status, err = cv2.calcOpticalFlowPyrLK(
                template_8u, frame_8u, points, None, **lk_params
            )
        except cv2.error:
            shift = (prev_ty, prev_tx)
            aligned = ndi.shift(frame, shift=shift, order=1, mode="nearest")
            stabilized[t] = aligned
            template = alpha * template + (1.0 - alpha) * aligned
            continue
        
        if new_points is None:
            shift = (prev_ty, prev_tx)
            aligned = ndi.shift(frame, shift=shift, order=1, mode="nearest")
            stabilized[t] = aligned
            template = alpha * template + (1.0 - alpha) * aligned
            continue
        
        # Filter good points
        good_old = points[status.flatten() == 1]
        good_new = new_points[status.flatten() == 1]
        
        if len(good_old) < 4:
            shift = (prev_ty, prev_tx)
            aligned = ndi.shift(frame, shift=shift, order=1, mode="nearest")
            stabilized[t] = aligned
            template = alpha * template + (1.0 - alpha) * aligned
            continue
        
        # Calculate translation as median of point displacements (robust)
        displacements = good_new - good_old
        tx = float(np.median(displacements[:, 0, 0]))
        ty = float(np.median(displacements[:, 0, 1]))
        
        # Check for excessive shift
        shift_magnitude = np.sqrt(tx**2 + ty**2)
        if shift_magnitude > max_shift_pixels:
            tx, ty = prev_tx, prev_ty
        
        # Apply inverse translation to align frame to template
        shift = (-ty, -tx)  # Negative because we want to align frame TO template
        aligned = ndi.shift(frame, shift=shift, order=1, mode="nearest")
        
        stabilized[t] = aligned
        
        # Update template (ImageJ formula: new = a*old + (1-a)*aligned)
        template = alpha * template + (1.0 - alpha) * aligned
        
        prev_tx, prev_ty = -tx, -ty  # Store the correction applied
    
    return stabilized


@njit(cache=True, fastmath=True)
def _to_ecc_img_jit(img: np.ndarray) -> np.ndarray:
    """Optimized ECC image normalization with Numba."""
    img = img.astype(np.float32)
    m = np.float32(img.mean())
    s = np.float32(img.std())
    if s > 0.0:
        img = (img - m) / (np.float32(3.0) * s)
    # Rescale to [0,1]
    img_min = np.float32(img.min())
    img_max = np.float32(img.max())
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img.fill(np.float32(0.5))
    return img


def _to_ecc_img(img: np.ndarray) -> np.ndarray:
    """Normalize image to float32 [0,1] for ECC."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    if NUMBA_AVAILABLE:
        return _to_ecc_img_jit(img)
    else:
        # Fallback to original
        m, s = img.mean(), img.std()
        if s > 0:
            img = (img - m) / (3 * s)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img[:] = 0.5
        return img


def stabilize_stack_translation_ecc(
    stack: np.ndarray,
    max_iterations: int = 1000,
    epsilon: float = 1e-8,
    template_update_coeff: float = 0.90,
    pyramid_levels: int = 4,
    pre_smooth_sigma: float = 0.5,
    max_shift_pixels: float = 10.0,
    min_correlation: float = 0.7,
    refine_iterations: int = 1,
) -> np.ndarray:
    """
    Stabilize a stack using OpenCV's ECC algorithm with a translation model.
    Enhanced with stricter quality checks and better alignment strategies.

    Parameters
    ----------
    stack : (T, Y, X), float32
    max_iterations : int
        Maximum number of ECC iterations per frame.
    epsilon : float
        Convergence threshold (analogous to error_tolerance). Lower = stricter.
    template_update_coeff : float
        Exponential moving-average coefficient for template (0-1, higher = slower update).
    pyramid_levels : int
        Number of resolution levels (1 = full-res only; >1 = coarse-to-fine).
    pre_smooth_sigma : float
        Gaussian smoothing sigma before alignment (0 = no smoothing). Reduces noise.
    max_shift_pixels : float
        Maximum allowed shift in pixels. Rejects larger shifts as likely errors.
    min_correlation : float
        Minimum correlation coefficient (0-1) to accept alignment. Higher = stricter.
    refine_iterations : int
        Number of refinement passes at full resolution (1 = single pass, >1 = iterative refinement).

    Returns
    -------
    stabilized : (T, Y, X), float32
    """
    T, H, W = stack.shape
    stabilized = np.empty_like(stack, dtype=np.float32)

    # Pre-smooth first frame for better template
    if pre_smooth_sigma > 0:
        template = filters.gaussian(stack[0], sigma=pre_smooth_sigma).astype(np.float32)
    else:
        template = stack[0].astype(np.float32)
    stabilized[0] = template.copy()

    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iterations, epsilon)

    alpha = float(template_update_coeff)
    
    # Track previous shift for better initialization
    prev_tx, prev_ty = 0.0, 0.0

    def init_warp(tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
        return np.array([[1.0, 0.0, tx],
                         [0.0, 1.0, ty]], dtype=np.float32)
    
    def compute_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute normalized correlation coefficient between two images."""
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        mean1 = img1_flat.mean()
        mean2 = img2_flat.mean()
        std1 = img1_flat.std()
        std2 = img2_flat.std()
        if std1 == 0 or std2 == 0:
            return 0.0
        corr = ((img1_flat - mean1) * (img2_flat - mean2)).mean() / (std1 * std2)
        return float(corr)

    for t in range(1, T):
        frame = stack[t].astype(np.float32)
        
        # Pre-smooth frame if requested
        if pre_smooth_sigma > 0:
            frame_smooth = filters.gaussian(frame, sigma=pre_smooth_sigma).astype(np.float32)
        else:
            frame_smooth = frame

        tmpl_ecc = _to_ecc_img(template)
        frm_ecc  = _to_ecc_img(frame_smooth)

        tx, ty = prev_tx, prev_ty  # Initialize with previous shift
        alignment_quality = 0.0

        if pyramid_levels <= 1:
            # Single resolution alignment
            warp_matrix = init_warp(tx, ty)
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    tmpl_ecc, frm_ecc, warp_matrix, warp_mode, criteria
                )
                tx = float(warp_matrix[0, 2])
                ty = float(warp_matrix[1, 2])
                alignment_quality = float(cc)
            except cv2.error:
                tx, ty = prev_tx, prev_ty
                alignment_quality = 0.0
        else:
            # Coarse-to-fine: start at low resolution
            down_factor = 2 ** (pyramid_levels - 1)
            Hc = max(1, H // down_factor)
            Wc = max(1, W // down_factor)

            tmpl_coarse = cv2.resize(tmpl_ecc, (Wc, Hc), interpolation=cv2.INTER_AREA)
            frm_coarse  = cv2.resize(frm_ecc,  (Wc, Hc), interpolation=cv2.INTER_AREA)

            # Initialize with previous shift scaled down
            warp_matrix_coarse = init_warp(tx / down_factor, ty / down_factor)
            try:
                _, warp_matrix_coarse = cv2.findTransformECC(
                    tmpl_coarse, frm_coarse,
                    warp_matrix_coarse, warp_mode, criteria
                )
                tx0 = float(warp_matrix_coarse[0, 2]) * down_factor
                ty0 = float(warp_matrix_coarse[1, 2]) * down_factor
            except cv2.error:
                tx0, ty0 = prev_tx, prev_ty

            # Refine at full resolution with multiple passes if requested
            for refine_pass in range(refine_iterations):
                warp_matrix_full = init_warp(tx0, ty0)
                try:
                    cc, warp_matrix_full = cv2.findTransformECC(
                        tmpl_ecc, frm_ecc, warp_matrix_full, warp_mode, criteria
                    )
                    tx = float(warp_matrix_full[0, 2])
                    ty = float(warp_matrix_full[1, 2])
                    alignment_quality = float(cc)
                    tx0, ty0 = tx, ty  # Use as starting point for next refinement
                except cv2.error:
                    if refine_pass == 0:
                        tx, ty = tx0, ty0
                        alignment_quality = 0.0
                    break

        # Quality checks: reject large shifts or poor correlations
        shift_magnitude = np.sqrt(tx**2 + ty**2)
        if shift_magnitude > max_shift_pixels:
            # Reject: shift too large, likely error
            tx, ty = prev_tx, prev_ty
            alignment_quality = 0.0
        
        # Apply translation shift (ty, tx) in image coordinates
        shift = (ty, tx)
        aligned = ndi.shift(frame, shift=shift, order=1, mode="nearest")
        
        # Verify alignment quality with correlation
        if alignment_quality < min_correlation:
            # Check actual correlation of aligned frame
            aligned_ecc = _to_ecc_img(aligned)
            actual_corr = compute_correlation(tmpl_ecc, aligned_ecc)
            if actual_corr < min_correlation:
                # Poor alignment: use previous shift or no shift
                if shift_magnitude > 1.0:  # Only reject if we tried to shift
                    aligned = ndi.shift(frame, shift=(prev_ty, prev_tx), order=1, mode="nearest")
                    tx, ty = prev_tx, prev_ty
        
        stabilized[t] = aligned
        
        # Update template with stricter quality control
        if alignment_quality >= min_correlation or shift_magnitude < 1.0:
            # Good alignment or minimal shift: update template
            template = alpha * template + (1.0 - alpha) * aligned
        # Otherwise keep template unchanged to avoid accumulating errors
        
        prev_tx, prev_ty = tx, ty

    return stabilized


# ----------------- MAIN PROCESSING ----------------- #

def process_stack_file(
    input_path: Path,
    movie_out_dir: Path,
    min_area: float = 50.0,
    morph_open_radius: int = 1,
    smooth_sigma: float = 1.0,
    clear_borders: bool = True,
    ratio_blur_sigma: float = 1.0,
    bleach_threshold: float = 1.2,
    crop_multiplier: float = 2.0,
    max_iterations: int = 1000,
    epsilon: float = 1e-8,
    template_update_coeff: float = 0.90,
    pyramid_levels: int = 4,
    pre_smooth_sigma: float = 0.5,
    max_shift_pixels: float = 10.0,
    min_correlation: float = 0.7,
    refine_iterations: int = 1,
    min_nobleach_diameter: float = 10.0,
    min_nobleach_pc: float = 10.0,
    no_stabilization: bool = False,
    use_lucas_kanade: bool = False,
    verbose: bool = True,
) -> int:
    """
    Process a single movie file:
      - detect droplets
      - compute partition coefficients
      - classify FRAP / NoBleach via frame0/frame1 ratio
      - filter NoBleach droplets by:
            size, partition coefficient, and border-label check
      - crop around droplets
      - stabilize each crop (ECC or Lucas-Kanade algorithm)
      - save outputs into movie_out_dir

    Parameters
    ----------
    use_lucas_kanade : bool
        If True, use Lucas-Kanade algorithm (matches ImageJ Image Stabilizer).
        If False (default), use OpenCV ECC algorithm.

    Returns
    -------
    int
        Number of droplets processed (cropped & saved).
    """
    if verbose:
        print(f"\nProcessing: {input_path.name}")

    stack = load_stack(input_path)  # (T, Y, X)
    T, H, W = stack.shape
    if T < 2:
        if verbose:
            print("  Skipping: needs at least 2 frames for frame0/frame1 ratio.")
        return 0

    # Step 1: Stabilize whole movie first (if enabled), then save stabilized stack in movie folder
    if no_stabilization:
        stack_work = stack
    else:
        if verbose:
            print("  Stabilizing whole movie...")
        if use_lucas_kanade:
            stack_work = stabilize_stack_lucas_kanade(
                stack,
                template_update_coeff=template_update_coeff,
                pyramid_levels=pyramid_levels,
                max_iterations=max_iterations,
                epsilon=epsilon,
                pre_smooth_sigma=pre_smooth_sigma,
                max_shift_pixels=max_shift_pixels,
            )
        else:
            stack_work = stabilize_stack_translation_ecc(
                stack,
                max_iterations=max_iterations,
                epsilon=epsilon,
                template_update_coeff=template_update_coeff,
                pyramid_levels=pyramid_levels,
                pre_smooth_sigma=pre_smooth_sigma,
                max_shift_pixels=max_shift_pixels,
                min_correlation=min_correlation,
                refine_iterations=refine_iterations,
            )
        # Save stabilized whole-movie TIF in movie folder (parent of droplet_stacks)
        movie_stem = input_path.stem
        movie_results_dir = movie_out_dir.parent  # e.g. .../movie_stem_results/
        movie_results_dir.mkdir(parents=True, exist_ok=True)
        stabilized_whole_path = movie_results_dir / f"stabilized_{movie_stem}.tif"
        save_stack_tiff(stabilized_whole_path, stack_work)
        if verbose:
            print(f"  Saved stabilized whole-movie stack: {stabilized_whole_path.name}")

    frame0 = stack_work[0]
    frame1 = stack_work[1]

    # Detect droplets on pre-bleach frame (from stabilized stack)
    labels, regions = detect_droplets_prebleach(
        frame0,
        min_area=min_area,
        morph_open_radius=morph_open_radius,
        smooth_sigma=smooth_sigma,
        clear_borders=clear_borders,
    )

    if len(regions) == 0:
        if verbose:
            print("  No droplets detected.")
        return 0

    if verbose:
        print(f"  Detected {len(regions)} droplet candidate(s).")

    # Partition coefficients (signal/noise)
    pcs = compute_partition_coefficients(frame0, labels, regions)

    # Ratio-based FRAP classification (frame0/frame1)
    ratio_img = compute_ratio_frame(frame0, frame1)
    classes = classify_droplets_frap_vs_nobleach(
        labels,
        regions,
        ratio_img,
        ratio_blur_sigma=ratio_blur_sigma,
        bleach_threshold=bleach_threshold,
    )

    movie_stem = input_path.stem
    movie_out_dir.mkdir(parents=True, exist_ok=True)

    droplet_count = 0

    for idx, region in enumerate(regions, start=1):
        label = region.label
        class_label = classes.get(label, "NoBleach")
        pc = pcs.get(label, np.nan)
        equiv_d = float(region.equivalent_diameter)

        # Crop from stabilized (or original) stack - no per-crop stabilization
        cropped, border_has_labels = crop_stack_around_region(
            stack_work,
            region,
            crop_multiplier=crop_multiplier,
            labels=labels,
        )

        # Apply NoBleach filters: size, partition coefficient, border labels
        if class_label == "NoBleach":
            if np.isnan(pc) or pc < min_nobleach_pc or equiv_d < min_nobleach_diameter:
                if verbose:
                    print(
                        f"    Skipping NoBleach droplet {idx:03d} "
                        f"(equiv_d={equiv_d:.2f}, PC={pc:.2f})"
                    )
                continue
            if border_has_labels:
                if verbose:
                    print(
                        f"    Skipping NoBleach droplet {idx:03d} "
                        f"(droplet(s) touching crop border)"
                    )
                continue

        # Crops are already from stabilized stack; no per-droplet stabilization
        stabilized = cropped.copy()

        index_str = f"{idx:03d}"
        out_name = f"Drift_{movie_stem}_{class_label}_droplet{index_str}.tif"
        out_path = movie_out_dir / out_name

        save_stack_tiff(out_path, stabilized)

        droplet_count += 1
        if verbose:
            print(
                f"    Saved {out_name}  "
                f"(class={class_label}, equiv_d={equiv_d:.2f}, PC={pc:.2f}, "
                f"border_labels={border_has_labels})"
            )

    return droplet_count


def _process_file_wrapper(args_tuple):
    """Wrapper function for parallel processing (must be top-level for pickling)."""
    (
        input_path_str,
        movie_out_dir_str,
        min_area,
        morph_open_radius,
        smooth_sigma,
        clear_borders,
        ratio_blur_sigma,
        bleach_threshold,
        crop_multiplier,
        max_iterations,
        epsilon,
        template_update_coeff,
        pyramid_levels,
        pre_smooth_sigma,
        max_shift_pixels,
        min_correlation,
        refine_iterations,
        min_nobleach_diameter,
        min_nobleach_pc,
        no_stabilization,
        use_lucas_kanade,
        verbose,
    ) = args_tuple
    
    input_path = Path(input_path_str)
    movie_out_dir = Path(movie_out_dir_str)
    
    return process_stack_file(
        input_path,
        movie_out_dir,
        min_area=min_area,
        morph_open_radius=morph_open_radius,
        smooth_sigma=smooth_sigma,
        clear_borders=clear_borders,
        ratio_blur_sigma=ratio_blur_sigma,
        bleach_threshold=bleach_threshold,
        crop_multiplier=crop_multiplier,
        max_iterations=max_iterations,
        epsilon=epsilon,
        template_update_coeff=template_update_coeff,
        pyramid_levels=pyramid_levels,
        pre_smooth_sigma=pre_smooth_sigma,
        max_shift_pixels=max_shift_pixels,
        min_correlation=min_correlation,
        refine_iterations=refine_iterations,
        min_nobleach_diameter=min_nobleach_diameter,
        min_nobleach_pc=min_nobleach_pc,
        no_stabilization=no_stabilization,
        use_lucas_kanade=use_lucas_kanade,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Crop and stabilize FRAP / no-bleach droplets from image stacks."
    )
    parser.add_argument("input_path", type=str,
                        help="Input file (.tif/.tiff/.nd2) or directory of such files.")

    parser.add_argument("--output-root", type=str, default=None,
                        help="Optional root directory for Analysis Results. "
                             "If omitted, creates 'Analysis Results' folder next to input. "
                             "Output: [output-root]/[movie]_results/droplet_stacks/")
    parser.add_argument("--droplet-images-root", type=str, default=None,
                        help="Override output root for droplet stacks. Default: uses --output-root")

    parser.add_argument("--min-area", type=float, default=50.0,
                        help="Minimum droplet area (pixels) for detection on pre-bleach frame.")
    parser.add_argument("--morph-open-radius", type=int, default=1,
                        help="Radius of disk for morphological opening on pre-bleach mask (0 to disable).")
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="Gaussian sigma for smoothing pre-bleach frame before thresholding.")
    parser.add_argument("--clear-borders", action="store_true",
                        help="Remove droplets touching image borders in the full frame.")
    parser.add_argument("--no-clear-borders", action="store_true",
                        help="Do NOT remove border-touching droplets in the full frame.")
    parser.add_argument("--ratio-blur-sigma", type=float, default=0.7,
                        help="Gaussian sigma for smoothing frame0/frame1 ratio image.")
    parser.add_argument("--bleach-threshold", type=float, default=1.2,
                        help="FRAP if mean(frame0/frame1) > bleach_threshold.")
    parser.add_argument("--crop-multiplier", type=float, default=2.0,
                        help="Square side = crop_multiplier * equivalent_diameter.")

    # ECC stabilizer parameters (match Fiji-style controls)
    parser.add_argument("--max-iterations", type=int, default=2000,
                        help="Maximum ECC iterations per frame (like Fiji's maximum_iterations).")
    parser.add_argument("--epsilon", type=float, default=1e-10,
                        help="ECC convergence threshold (like Fiji's error_tolerance). Lower = stricter.")
    parser.add_argument("--pyramid-levels", type=int, default=4,
                        help="Number of resolution levels (like Fiji's maximum_pyramid_levels).")
    parser.add_argument("--template-update-coeff", type=float, default=0.99,
                        help="Template update coefficient (alpha) for stabilization (0–1, higher = slower update).")
    parser.add_argument("--pre-smooth-sigma", type=float, default=0.5,
                        help="Gaussian smoothing sigma before alignment (0 = no smoothing). Reduces noise.")
    parser.add_argument("--max-shift-pixels", type=float, default=25.0,
                        help="Maximum allowed shift in pixels. Rejects larger shifts as likely errors.")
    parser.add_argument("--min-correlation", type=float, default=0.8,
                        help="Minimum correlation coefficient (0-1) to accept alignment. Higher = stricter.")
    parser.add_argument("--refine-iterations", type=int, default=20,
                        help="Number of refinement passes at full resolution (1 = single pass, >1 = iterative refinement).")

    parser.add_argument("--min-nobleach-diameter", type=float, default=10.0,
                        help="Minimum equivalent diameter (pixels) for NoBleach droplets to keep.")
    parser.add_argument("--min-nobleach-pc", type=float, default=5.0,
                        help="Minimum partition coefficient (droplet/background) for NoBleach droplets to keep.")

    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Number of parallel workers for processing multiple files (default: 8).")

    parser.add_argument("--no-stabilize", action="store_true",
                        help="Disable image stabilization. By default stabilization is ON.")
    
    parser.add_argument("--stabilize-algorithm", type=str, default="lucas-kanade",
                        choices=["ecc", "lucas-kanade", "lk"],
                        help="Stabilization algorithm: 'lucas-kanade'/'lk' (matches ImageJ "
                             "Image Stabilizer plugin, default) or 'ecc' (OpenCV ECC).")

    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-droplet logging.")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_root = Path(args.output_root) if args.output_root is not None else None
    droplet_root_arg = Path(args.droplet_images_root) if args.droplet_images_root else None
    no_stabilization_flag = args.no_stabilize  # default: stabilization ON
    use_lucas_kanade = args.stabilize_algorithm in ["lucas-kanade", "lk"]

    clear_borders_flag = True
    if args.no_clear_borders:
        clear_borders_flag = False
    elif args.clear_borders:
        clear_borders_flag = True

    # Collect files
    # Supported file extensions
    supported_extensions = {".tif", ".tiff", ".nd2"}
    if AICSIMAGEIO_AVAILABLE:
        supported_extensions.update({".czi", ".lif", ".lsm", ".oib", ".oif", ".dv", ".r3d", ".ims"})
    
    if input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in supported_extensions
            or f.name.lower().endswith(".ome.tif")
            or f.name.lower().endswith(".ome.tiff")
        )
    elif input_path.is_file():
        files = [input_path]
    else:
        raise SystemExit(f"Input path does not exist: {input_path}")

    if not files:
        msg = f"No supported image files found in: {input_path}\n"
        msg += f"Supported extensions: {', '.join(sorted(supported_extensions))}\n"
        if not AICSIMAGEIO_AVAILABLE:
            msg += "\nFor additional format support (CZI, LIF, OME-TIFF, etc.):\n"
            msg += "  pip install aicsimageio[all]\n"
        raise SystemExit(msg)

    # Resolve output root for results
    if output_root is None:
        # default: Analysis Results folder alongside inputs
        results_root = input_path.parent / "Analysis Results"
    else:
        results_root = output_root
    if droplet_root_arg:
        results_root = droplet_root_arg

    # Prepare arguments for parallel processing
    # Output structure: [results_root]/[movie_stem]_results/droplet_stacks/
    worker_args = []
    for f in files:
        movie_out_dir = results_root / f"{f.stem}_results" / "droplet_stacks"
        
        worker_args.append((
            str(f),
            str(movie_out_dir),
            args.min_area,
            args.morph_open_radius,
            args.smooth_sigma,
            clear_borders_flag,
            args.ratio_blur_sigma,
            args.bleach_threshold,
            args.crop_multiplier,
            args.max_iterations,
            args.epsilon,
            args.template_update_coeff,
            args.pyramid_levels,
            args.pre_smooth_sigma,
            args.max_shift_pixels,
            args.min_correlation,
            args.refine_iterations,
            args.min_nobleach_diameter,
            args.min_nobleach_pc,
            no_stabilization_flag,
            use_lucas_kanade,
            not args.quiet,
        ))

    total = 0
    had_errors = False
    if args.n_jobs > 1 and len(files) > 1:
        # Parallel processing
        if not args.quiet:
            print(f"Processing {len(files)} files with {args.n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {executor.submit(_process_file_wrapper, worker_arg): i 
                      for i, worker_arg in enumerate(worker_args)}
            
            for future in as_completed(futures):
                i = futures[future]
                try:
                    n = future.result()
                    total += n
                except Exception as e:
                    had_errors = True
                    f = files[i]
                    print(f"Error processing {f.name}: {e}", file=sys.stderr)
                    if not args.quiet:
                        import traceback
                        traceback.print_exc()
    else:
        # Sequential processing (original behavior)
        for i, f in enumerate(files):
            try:
                n = _process_file_wrapper(worker_args[i])
                total += n
            except Exception as e:
                had_errors = True
                print(f"Error processing {f.name}: {e}", file=sys.stderr)
                if not args.quiet:
                    import traceback
                    traceback.print_exc()
                continue

    if not args.quiet:
        print(f"\nDone. Total droplets processed: {total}")

    # Exit with code 1 if any file failed so the GUI reports failure
    if had_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
