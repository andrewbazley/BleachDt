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
- Stabilizes the whole movie BEFORE cropping. The default algorithm is a faithful
  Python port of Fiji's Image Stabilizer (Kang Li) in Translation mode: dense,
  gradient-based, pyramidal Lucas-Kanade on float intensities. Settings match the
  Fiji macro:

      transformation=Translation
      maximum_pyramid_levels=4
      template_update_coefficient=0.999
      maximum_iterations=1000
      error_tolerance=1e-8

  Drift is estimated at subpixel precision but APPLIED at whole-pixel precision by
  default (--subpixel-stabilization to opt in), so pixel intensities inside droplets
  are preserved exactly - only their position is corrected. This matters for the
  RawIntDen / total-fluorescence measurements downstream. ECC remains available via
  --stabilize-algorithm ecc.

- Saves each movie's droplets into a results folder structure:

    For file: /path/to/movies/ADP_1.nd2
    Output (default):
        /path/to/movies/Analysis Results/ADP_1_results/droplet_stacks/
            Drift_ADP_1_NoBleach_droplet001.tif
            Drift_ADP_1_FRAP_droplet002.tif
            ...

Usage
-----
    bleachdt-crop INPUT_PATH [--output-root OUTPUT_ROOT]
    # or, equivalently:
    python -m bleachdt.droplet_crop INPUT_PATH [--output-root OUTPUT_ROOT]

INPUT_PATH:
  - A single .tif/.tiff/.nd2 file, OR
  - A directory containing such files.

If --output-root is omitted, creates 'Analysis Results' folder alongside inputs.
Output structure: [output-root]/[movie]_results/droplet_stacks/

Example
-------
    bleachdt-crop "/path/to/movies"

    bleachdt-crop "/path/to/movies" \
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

# Defer annotation evaluation so PEP 604 union syntax (e.g. ``np.ndarray | None``) does
# not error on Python 3.9 (where ``type | None`` is not yet a runtime operation).
from __future__ import annotations

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
try:
    from skimage.registration import phase_cross_correlation as _phase_cross_correlation
except Exception:  # pragma: no cover - very old skimage
    _phase_cross_correlation = None

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


def _read_stack_array(path: Path, verbose: bool = True) -> np.ndarray:
    """
    Load raw image array from disk (before channel/Z normalization).

    Raises ImageImportError on failure.
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
    
    return np.asarray(data)


def load_stack(path: Path, verbose: bool = True) -> np.ndarray:
    """
    Load an image stack as a 3D array (T, Y, X), float32.

    Multichannel inputs use the first channel (or middle Z slice) for backward
    compatibility. Use :func:`load_stack_multichannel` for full (T, C, Y, X) data.
    """
    data = _read_stack_array(path, verbose=verbose)
    return to_tyx_single_channel(data)


def to_tc_yx(data: np.ndarray) -> np.ndarray:
    """
    Normalize microscopy array to (T, C, Y, X).

    Heuristic for 4D: if the second axis looks like a small channel dimension (<=4)
    and Y/X are larger, treat as (T, C, Y, X). Otherwise treat as (T, Z, Y, X) and
    take the middle Z slice as a single channel.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        return data[np.newaxis, np.newaxis, ...]
    if data.ndim == 3:
        return data[:, np.newaxis, :, :]
    if data.ndim == 4:
        t, a1, a2, a3 = data.shape
        # (T, C, Y, X): channel index before spatial when C is small
        if a1 <= 4 and a2 > 8 and a3 > 8:
            return data
        # (T, Y, X, C): last axis is channels
        if a3 <= 4 and a1 > 8 and a2 > 8:
            return np.moveaxis(data, 3, 1)
        # (T, Z, Y, X): take middle Z as one channel
        zi = a1 // 2
        return data[:, zi : zi + 1, :, :]
    if data.ndim == 5:
        # (T, C, Z, Y, X)
        zmid = data.shape[2] // 2
        return data[:, :, zmid, :, :]
    if data.ndim > 5:
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
        return data[:, np.newaxis, :, :]
    return data.astype(np.float32)


def to_tyx_single_channel(data: np.ndarray) -> np.ndarray:
    """Backward-compatible (T, Y, X): first channel of multichannel or middle Z."""
    tc = to_tc_yx(data)
    return tc[:, 0, :, :].astype(np.float32)


def load_stack_multichannel(path: Path, verbose: bool = True) -> np.ndarray:
    """Load image as (T, C, Y, X), float32 (same readers as :func:`load_stack`)."""
    data = _read_stack_array(path, verbose=verbose)
    return to_tc_yx(data)


def save_stack_tiff(path: Path, stack: np.ndarray) -> None:
    """
    Save a grayscale (T, Y, X) or multichannel (T, C, Y, X) stack as ImageJ TIFF.
    """
    if tifffile is None:
        raise RuntimeError("tifffile is required to save TIFF outputs (pip install tifffile).")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = stack.astype(np.float32)
    if arr.ndim == 4:
        tifffile.imwrite(
            str(path),
            arr,
            imagej=True,
            metadata={"axes": "TCYX"},
        )
    else:
        tifffile.imwrite(str(path), arr, imagej=True)


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


def crop_stack_around_region_tc(
    stack_tc: np.ndarray,
    region,
    crop_multiplier: float = 2.0,
    labels: np.ndarray | None = None,
):
    """
    Crop a (T, C, Y, X) stack around ``region`` (same geometry as
    :func:`crop_stack_around_region` on one channel).
    """
    T, C, H, W = stack_tc.shape
    cy, cx = region.centroid
    equiv_d = float(region.equivalent_diameter)

    side = max(4, int(np.floor(crop_multiplier * equiv_d)))
    half = side / 2.0

    y0 = int(np.round(cy - half))
    x0 = int(np.round(cx - half))
    y1 = y0 + side
    x1 = x0 + side

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)

    stack_tc = np.ascontiguousarray(stack_tc, dtype=np.float32)
    cropped = stack_tc[:, :, y0:y1, x0:x1]

    border_has_labels = False
    if labels is not None:
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        labels_crop = labels[y0:y1, x0:x1]
        if labels_crop.size > 0:
            if NUMBA_AVAILABLE:
                border_has_labels = check_border_labels_jit(labels_crop)
            else:
                top = labels_crop[0, :]
                bottom = labels_crop[-1, :]
                left = labels_crop[:, 0]
                right = labels_crop[:, -1]
                if (
                    np.any(top > 0)
                    or np.any(bottom > 0)
                    or np.any(left > 0)
                    or np.any(right > 0)
                ):
                    border_has_labels = True

    ch, cw = cropped.shape[2:]
    pad_y = side - ch
    pad_x = side - cw

    if pad_y > 0 or pad_x > 0:
        pad_before_y = pad_y // 2
        pad_after_y = pad_y - pad_before_y
        pad_before_x = pad_x // 2
        pad_after_x = pad_x - pad_before_x
        cropped = np.pad(
            cropped,
            (
                (0, 0),
                (0, 0),
                (pad_before_y, pad_after_y),
                (pad_before_x, pad_after_x),
            ),
            mode="edge",
        )

    return cropped, border_has_labels


def apply_shifts_to_tc(
    stack_tc: np.ndarray,
    shifts: np.ndarray,
    subpixel: bool = False,
) -> np.ndarray:
    """Apply the same (dy, dx) drift correction (from the analysis channel) to every channel.

    Pixel-preservation policy (important for FRAP / RawIntDen):
      - ``subpixel=False`` (default): the estimated subpixel drift is rounded to the
        nearest whole pixel and applied with nearest-neighbour shifting (``order=0``).
        This RELOCATES pixels without interpolating, so every intensity value inside a
        droplet is preserved exactly - only its position changes. Residual drift is at
        most +/-0.5 px, which is negligible for multi-pixel droplets and far preferable
        to the intensity blurring that bilinear interpolation introduces.
      - ``subpixel=True``: applies the full subpixel drift with bilinear interpolation
        (``order=1``). This matches the Fiji Image Stabilizer's warp exactly but blends
        neighbouring pixels, slightly altering per-pixel intensities. Opt in only if you
        need subpixel registration more than exact pixel fidelity.

    Exposed borders are filled with edge values (``mode="nearest"``); intensities are
    never wrapped around or zero-filled.
    """
    T, C, H, W = stack_tc.shape
    out = np.empty_like(stack_tc, dtype=np.float32)
    out[0] = stack_tc[0]
    for t in range(1, T):
        dy, dx = float(shifts[t, 0]), float(shifts[t, 1])
        if subpixel:
            sy, sx, order = dy, dx, 1
        else:
            sy, sx, order = float(round(dy)), float(round(dx)), 0
        for c in range(C):
            out[t, c] = ndi.shift(
                stack_tc[t, c].astype(np.float32),
                shift=(sy, sx),
                order=order,
                mode="nearest",
            )
    return out


# Backwards-compatible alias (older name); the default is now pixel-preserving.
apply_subpixel_shifts_to_tc = apply_shifts_to_tc


# ----------------- IMAGE STABILIZATION ----------------- #
#
# Two algorithms are available:
#   1. Lucas-Kanade - faithful port of ImageJ's Image Stabilizer (Kang Li) translation
#      mode: dense, gradient-based, pyramidal Lucas-Kanade on float intensities.
#   2. ECC (Enhanced Correlation Coefficient) - OpenCV's findTransformECC (alternative).
#
# Reference: K. Li, "The image stabilizer plugin for ImageJ", CMU, 2008
#            http://www.cs.cmu.edu/~kangli/code/Image_Stabilizer.html
#
# NOTE ON PIXEL PRESERVATION: these functions only *estimate* the per-frame drift
# (returned as the ``shifts`` array). The drift is applied to the saved/measured movie
# by ``apply_shifts_to_tc``, which by default rounds to whole-pixel shifts so droplet
# intensities are never interpolated. Subpixel estimation here only improves the
# whole-pixel decision (and is available for opt-in subpixel application).


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient-magnitude image.

    Fiji's Image Stabilizer estimates motion on the gradient magnitude, not raw
    intensity. This makes registration robust to absolute-intensity changes such as
    photobleaching and the FRAP bleach event (which are NOT motion). Estimating drift
    on this image is the single most important fix for FRAP stabilization.
    """
    a = np.asarray(img, dtype=np.float32)
    gy = ndi.sobel(a, axis=0, mode="reflect")
    gx = ndi.sobel(a, axis=1, mode="reflect")
    return np.hypot(gx, gy).astype(np.float32)


def _refine_translation_lk(ref_g, mov_g, weight, dy, dx, max_iter=12, eps=1e-3):
    """Masked gradient-domain Gauss-Newton subpixel refinement of a translation,
    starting from (dy, dx). ``weight`` is a float mask (1 inside the valid region)."""
    w = np.asarray(weight, dtype=np.float32)
    for _ in range(int(max_iter)):
        warp = ndi.shift(mov_g, (dy, dx), order=1, mode="reflect")
        gy, gx = np.gradient(warp)
        e = (ref_g - warp) * w
        gx = gx * w
        gy = gy * w
        Axx = float(np.sum(gx * gx))
        Axy = float(np.sum(gx * gy))
        Ayy = float(np.sum(gy * gy))
        bx = float(np.sum(gx * e))
        by = float(np.sum(gy * e))
        det = Axx * Ayy - Axy * Axy
        if abs(det) < 1e-9:
            break
        ddx = -(Ayy * bx - Axy * by) / det
        ddy = -(Axx * by - Axy * bx) / det
        dy += ddy
        dx += ddx
        if (ddx * ddx + ddy * ddy) < (eps * eps):
            break
    return dy, dx


def estimate_drift_gradient(ref_frame, mov_frame, valid_mask=None, refine_iters=12):
    """Subpixel translation that aligns ``mov_frame`` onto ``ref_frame``.

    Done in the gradient-magnitude domain with an optional validity mask that excludes
    the bleach spot / FRAP droplets. A bleach-proof integer estimate from masked phase
    cross-correlation is refined to subpixel by a few masked gradient Gauss-Newton steps.
    Returns ``((dy, dx), residual_rms)``. The returned shift is applied to the moving
    frame via ``ndi.shift(frame, (dy, dx))`` (same convention as ``apply_shifts_to_tc``).
    """
    ref_g = gradient_magnitude(ref_frame)
    mov_g = gradient_magnitude(mov_frame)
    if valid_mask is None:
        valid = np.ones(ref_g.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.sum() < 0.02 * valid.size:
            valid = np.ones(ref_g.shape, dtype=bool)

    dy = dx = 0.0
    if _phase_cross_correlation is not None:
        try:
            res = _phase_cross_correlation(
                ref_g, mov_g, reference_mask=valid, moving_mask=valid
            )
            shift = res[0] if isinstance(res, tuple) else res
            dy, dx = float(shift[0]), float(shift[1])
        except Exception:
            dy = dx = 0.0

    dy, dx = _refine_translation_lk(
        ref_g, mov_g, valid.astype(np.float32), dy, dx, max_iter=refine_iters
    )
    warp = ndi.shift(mov_g, (dy, dx), order=1, mode="reflect")
    resid = float(np.sqrt(np.mean(((ref_g - warp) * valid) ** 2)))
    return (dy, dx), resid


def build_registration_mask(frame0, frame1, min_area, morph_open_radius, smooth_sigma,
                            clear_borders, ratio_blur_sigma, bleach_threshold, verbose=False):
    """Boolean validity mask for drift estimation that EXCLUDES bleached (FRAP) droplets.

    A bleach event violates the brightness-constancy assumption behind registration, so
    drift is estimated from non-bleached droplets + background structure only. Falls back
    to all-True if FRAP droplets cannot be identified or exclusion would remove too much.
    """
    try:
        labels, regions = detect_droplets_prebleach(
            frame0, min_area=min_area, morph_open_radius=morph_open_radius,
            smooth_sigma=smooth_sigma, clear_borders=clear_borders,
        )
        if len(regions) == 0:
            return np.ones(np.asarray(frame0).shape, dtype=bool)
        ratio_img = compute_ratio_frame(frame0, frame1)
        classes = classify_droplets_frap_vs_nobleach(
            labels, regions, ratio_img,
            ratio_blur_sigma=ratio_blur_sigma, bleach_threshold=bleach_threshold,
        )
        exclude = np.zeros(np.asarray(frame0).shape, dtype=bool)
        n_frap = 0
        for region in regions:
            if classes.get(region.label, "NoBleach") == "FRAP":
                exclude[labels == region.label] = True
                n_frap += 1
        if exclude.any():
            exclude = ndi.binary_dilation(exclude, iterations=3)
        valid = ~exclude
        if valid.sum() < 0.05 * valid.size:
            return np.ones(np.asarray(frame0).shape, dtype=bool)
        if verbose:
            print(f"  Stabilization mask: excluded {n_frap} FRAP droplet(s); "
                  f"using {100.0 * valid.mean():.1f}% of the frame for drift estimation.")
        return valid
    except Exception:
        return np.ones(np.asarray(frame0).shape, dtype=bool)


def stabilize_stack_gradient(stack, valid_mask=None, refine_iters=12,
                             max_shift_pixels=25.0, verbose=False):
    """Estimate per-frame drift for a (T, Y, X) stack by registering every frame to
    frame 0 in the masked gradient-magnitude domain (bleach-robust, subpixel).

    Returns
    -------
    shifts : (T, 2) float32   subpixel (dy, dx) to apply to each frame (row 0 = (0, 0))
    resids : (T,)   float32   residual RMS per frame (QC metric)
    """
    stack = np.asarray(stack, dtype=np.float32)
    T = stack.shape[0]
    shifts = np.zeros((T, 2), dtype=np.float32)
    resids = np.zeros(T, dtype=np.float32)
    ref = stack[0]
    prev = (0.0, 0.0)
    for t in range(1, T):
        (dy, dx), resid = estimate_drift_gradient(
            ref, stack[t], valid_mask, refine_iters=refine_iters
        )
        if (not np.isfinite(dy)) or (not np.isfinite(dx)) or (np.hypot(dy, dx) > max_shift_pixels):
            dy, dx = prev
        shifts[t, 0] = dy
        shifts[t, 1] = dx
        resids[t] = resid
        prev = (dy, dx)
    return shifts, resids


def _gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Build a Gaussian image pyramid. Index 0 = finest (full resolution)."""
    pyr = [np.asarray(img, dtype=np.float32)]
    for _ in range(1, max(1, int(levels))):
        prev = pyr[-1]
        if min(prev.shape) < 8:
            break
        smoothed = filters.gaussian(prev, sigma=1.0, preserve_range=True)
        pyr.append(smoothed[::2, ::2].astype(np.float32))
    return pyr


def _estimate_translation_lk(
    template: np.ndarray,
    frame: np.ndarray,
    max_iterations: int,
    epsilon: float,
    pyramid_levels: int,
) -> tuple:
    """Dense gradient-based Lucas-Kanade translation estimate (Kang Li style).

    Forward-additive Gauss-Newton minimisation of the sum of squared intensity
    differences between ``template`` and a translated ``frame``, solved coarse-to-fine
    over a Gaussian pyramid. Returns ``(dy, dx)`` - the shift to apply to ``frame`` to
    align it onto ``template`` (same convention as ``ndi.shift``).
    """
    t_pyr = _gaussian_pyramid(template, pyramid_levels)
    f_pyr = _gaussian_pyramid(frame, pyramid_levels)
    n_levels = min(len(t_pyr), len(f_pyr))

    dy = dx = 0.0
    for lvl in range(n_levels - 1, -1, -1):  # coarse -> fine
        T_l = t_pyr[lvl]
        F_l = f_pyr[lvl]
        for _ in range(max(1, int(max_iterations))):
            warped = ndi.shift(F_l, shift=(dy, dx), order=1, mode="nearest")
            gy, gx = np.gradient(warped)
            err = T_l - warped
            Axx = float(np.sum(gx * gx))
            Axy = float(np.sum(gx * gy))
            Ayy = float(np.sum(gy * gy))
            bx = float(np.sum(gx * err))
            by = float(np.sum(gy * err))
            det = Axx * Ayy - Axy * Axy
            if abs(det) < 1e-12:
                break
            # Gauss-Newton step: Delta p = -H^{-1} b  (descent on SSD). With
            # warped(x)=F(x-p), d(err)/dp = +grad(warped), so the update is negated.
            ddx = -(Ayy * bx - Axy * by) / det
            ddy = -(Axx * by - Axy * bx) / det
            dx += ddx
            dy += ddy
            if (ddx * ddx + ddy * ddy) < (float(epsilon) ** 2):
                break
        if lvl > 0:  # propagate estimate to the next finer level
            dy *= 2.0
            dx *= 2.0
    return dy, dx


def stabilize_stack_lucas_kanade(
    stack: np.ndarray,
    template_update_coeff: float = 0.999,
    pyramid_levels: int = 4,
    max_iterations: int = 1000,
    epsilon: float = 1e-8,
    pre_smooth_sigma: float = 0.5,
    max_shift_pixels: float = 25.0,
) -> np.ndarray:
    """
    Stabilize a stack with dense gradient-based Lucas-Kanade - a faithful port of the
    ImageJ Image Stabilizer plugin by Kang Li (CMU) in Translation mode.

    Unlike sparse feature-tracking, this estimates a single subpixel translation per
    frame by minimising the sum of squared intensity differences against a running
    template, using image gradients and a Gaussian pyramid (coarse-to-fine). It works
    on float intensities (no 8-bit conversion), matching the plugin's numerics.

    Parameters
    ----------
    stack : (T, Y, X), float32
        Input image stack.
    template_update_coeff : float
        Template update coefficient 'a': new_template = a * old + (1-a) * aligned.
        Higher (closer to 1) keeps the template closer to frame 0. Fiji macro default: 0.999.
    pyramid_levels : int
        Number of Gaussian-pyramid levels for coarse-to-fine alignment. Fiji default: 4.
    max_iterations : int
        Maximum Gauss-Newton iterations per pyramid level. Fiji default: 1000.
    epsilon : float
        Convergence threshold on the per-iteration update magnitude. Fiji default: 1e-8.
    pre_smooth_sigma : float
        Gaussian smoothing applied before *estimation* only (0 = none). Does not affect
        the intensities that get saved/measured.
    max_shift_pixels : float
        Maximum allowed per-frame shift magnitude; larger estimates are rejected
        (held at the previous frame's shift).

    Returns
    -------
    stabilized : (T, Y, X), float32
        Subpixel-aligned preview (used only for the running template; the saved/measured
        movie is produced by ``apply_shifts_to_tc`` from ``shifts``).
    shifts : (T, 2), float32
        Subpixel shift ``(dy, dx)`` estimated for each frame (row 0 is (0, 0)).
    """
    T, H, W = stack.shape
    stabilized = np.empty_like(stack, dtype=np.float32)
    shifts = np.zeros((T, 2), dtype=np.float32)

    def _prep(img):
        if pre_smooth_sigma and pre_smooth_sigma > 0:
            return filters.gaussian(
                img, sigma=pre_smooth_sigma, preserve_range=True
            ).astype(np.float32)
        return np.asarray(img, dtype=np.float32)

    # Running template, initialised from frame 0 (kept close to frame 0 when alpha->1).
    template = _prep(stack[0])
    stabilized[0] = stack[0].astype(np.float32)
    alpha = float(template_update_coeff)

    prev_dy, prev_dx = 0.0, 0.0

    for t in range(1, T):
        frame = stack[t].astype(np.float32)
        frame_s = _prep(frame)

        try:
            dy, dx = _estimate_translation_lk(
                template, frame_s, max_iterations, epsilon, pyramid_levels
            )
        except Exception:
            dy, dx = prev_dy, prev_dx

        # Reject non-finite or excessive shifts; hold the previous estimate instead.
        if (not np.isfinite(dy)) or (not np.isfinite(dx)) or (np.hypot(dy, dx) > max_shift_pixels):
            dy, dx = prev_dy, prev_dx

        # Subpixel-aligned preview, used ONLY to advance the running template.
        aligned = ndi.shift(frame, shift=(dy, dx), order=1, mode="nearest")
        stabilized[t] = aligned
        shifts[t, 0] = dy
        shifts[t, 1] = dx

        aligned_s = ndi.shift(frame_s, shift=(dy, dx), order=1, mode="nearest")
        template = alpha * template + (1.0 - alpha) * aligned_s
        prev_dy, prev_dx = dy, dx

    return stabilized, shifts


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
    template_update_coeff: float = 0.999,
    pyramid_levels: int = 4,
    pre_smooth_sigma: float = 0.5,
    max_shift_pixels: float = 25.0,
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
    shifts : (T, 2), float32
        Shift ``(dy, dx)`` passed to ``ndi.shift`` for each frame (frame 0 is (0, 0)).
    """
    T, H, W = stack.shape
    stabilized = np.empty_like(stack, dtype=np.float32)
    shifts = np.zeros((T, 2), dtype=np.float32)

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
        shifts[t, 0] = ty
        shifts[t, 1] = tx
        
        # Update template with stricter quality control
        if alignment_quality >= min_correlation or shift_magnitude < 1.0:
            # Good alignment or minimal shift: update template
            template = alpha * template + (1.0 - alpha) * aligned
        # Otherwise keep template unchanged to avoid accumulating errors
        
        prev_tx, prev_ty = tx, ty

    return stabilized, shifts


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
    template_update_coeff: float = 0.999,
    pyramid_levels: int = 4,
    pre_smooth_sigma: float = 0.5,
    max_shift_pixels: float = 25.0,
    min_correlation: float = 0.7,
    refine_iterations: int = 1,
    min_nobleach_diameter: float = 10.0,
    min_nobleach_pc: float = 10.0,
    no_stabilization: bool = False,
    use_lucas_kanade: bool = False,
    verbose: bool = True,
    analysis_channel: int = 1,
    subpixel_apply: bool = True,
    exclude_bleach_from_stab: bool = False,
    pre_bleach_frames: int = 1,
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

    stack_tc = load_stack_multichannel(input_path)
    T, nch, H, W = stack_tc.shape
    ach = int(np.clip(analysis_channel, 1, nch)) - 1
    if verbose and nch > 1:
        print(
            f"  Multi-channel stack: {nch} channel(s); "
            f"droplet detection / FRAP class / stabilization ref = channel {ach + 1}"
        )

    if T < 2:
        if verbose:
            print("  Skipping: needs at least 2 frames for frame0/frame1 ratio.")
        return 0

    # Step 1: Stabilize whole movie (shifts from analysis channel, applied to all channels)
    if no_stabilization:
        stack_work = stack_tc
    else:
        if verbose:
            print("  Stabilizing whole movie (gradient-domain, bleach-excluded; "
                  "estimated on analysis channel, applied to all channels)...")
        ref = stack_tc[:, ach, :, :].astype(np.float32)
        stab_method = "ecc" if not use_lucas_kanade else "gradient"
        resids = None
        if use_lucas_kanade:
            # Default path: whole-image gradient-magnitude registration to frame 0,
            # exactly like Fiji's Image Stabilizer (verified to agree with the ImageJ
            # plugin's logged shifts to ~0.03 px). The gradient-magnitude domain is what
            # makes this robust to the bleach event, so no droplet masking is needed.
            # (Optional bleach exclusion is available via exclude_bleach_from_stab for
            # sparse-droplet movies; off by default.)
            reg_mask = None
            if exclude_bleach_from_stab:
                reg_mask = build_registration_mask(
                    ref[0], ref[1],
                    min_area=min_area, morph_open_radius=morph_open_radius,
                    smooth_sigma=smooth_sigma, clear_borders=clear_borders,
                    ratio_blur_sigma=ratio_blur_sigma, bleach_threshold=bleach_threshold,
                    verbose=verbose,
                )
            shifts, resids = stabilize_stack_gradient(
                ref, valid_mask=reg_mask,
                refine_iters=max(4, min(int(max_iterations), 30)),
                max_shift_pixels=max_shift_pixels, verbose=verbose,
            )
        else:
            _, shifts = stabilize_stack_translation_ecc(
                ref,
                max_iterations=max_iterations,
                epsilon=epsilon,
                template_update_coeff=template_update_coeff,
                pyramid_levels=pyramid_levels,
                pre_smooth_sigma=pre_smooth_sigma,
                max_shift_pixels=max_shift_pixels,
                min_correlation=min_correlation,
                refine_iterations=refine_iterations,
            )
        # Apply the SAME shift to every channel, once, with subpixel bilinear
        # interpolation by default (smooth registration faithful to intensity changes).
        stack_work = apply_shifts_to_tc(stack_tc, shifts, subpixel=subpixel_apply)
        movie_stem = input_path.stem
        movie_results_dir = movie_out_dir.parent
        movie_results_dir.mkdir(parents=True, exist_ok=True)
        stabilized_whole_path = movie_results_dir / f"stabilized_{movie_stem}.tif"
        save_stack_tiff(stabilized_whole_path, stack_work)

        # Document the stabilization: per-frame shift log for QC.
        try:
            import csv as _csv
            shift_csv = movie_results_dir / f"stabilization_shifts_{movie_stem}.csv"
            with open(shift_csv, "w", newline="") as f_sh:
                w = _csv.writer(f_sh)
                w.writerow(["frame", "dy_px", "dx_px", "shift_magnitude_px",
                            "residual_rms", "method", "application"])
                for t in range(shifts.shape[0]):
                    dy, dx = float(shifts[t, 0]), float(shifts[t, 1])
                    rr = float(resids[t]) if resids is not None else ""
                    w.writerow([t, f"{dy:.4f}", f"{dx:.4f}", f"{np.hypot(dy, dx):.4f}",
                                (f"{rr:.4f}" if rr != "" else ""), stab_method,
                                "subpixel" if subpixel_apply else "whole-pixel"])
            if verbose:
                mags = np.hypot(shifts[:, 0], shifts[:, 1])
                print(f"  Saved stabilized whole-movie stack: {stabilized_whole_path.name} "
                      f"(max drift {mags.max():.2f} px; shift log: {shift_csv.name})")
        except Exception as e:
            if verbose:
                print(f"  Saved stabilized whole-movie stack: {stabilized_whole_path.name} "
                      f"(could not write shift log: {e})")

    # Pre-bleach = last pre-bleach frame; post-bleach = first post-bleach frame.
    first_post = max(1, min(int(pre_bleach_frames), T - 1))
    frame0 = stack_work[first_post - 1, ach]
    frame1 = stack_work[first_post, ach]

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
    # Per-droplet location + geometry records (centroids are in full-frame, post-
    # stabilization pixel coordinates). Written to droplet_locations.csv so the
    # analysis stage can report droplet X/Y and geometry without extra image reads.
    geometry_records = []

    for idx, region in enumerate(regions, start=1):
        label = region.label
        class_label = classes.get(label, "NoBleach")
        pc = pcs.get(label, np.nan)
        equiv_d = float(region.equivalent_diameter)

        # Crop from stabilized (or original) stack - no per-crop stabilization
        cropped, border_has_labels = crop_stack_around_region_tc(
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

        # Record droplet location + geometry (full-frame, post-stabilization coords).
        cy, cx = (float(region.centroid[0]), float(region.centroid[1]))
        area_px = float(region.area)
        try:
            perimeter_px = float(region.perimeter)
        except Exception:
            perimeter_px = float("nan")
        circularity = (
            float(4.0 * np.pi * area_px / (perimeter_px ** 2))
            if perimeter_px and perimeter_px > 0
            else float("nan")
        )
        try:
            eccentricity = float(region.eccentricity)
        except Exception:
            eccentricity = float("nan")
        geometry_records.append({
            "droplet_file": out_name,
            "droplet_index": idx,
            "class": class_label,
            "centroid_x_px": round(cx, 3),
            "centroid_y_px": round(cy, 3),
            "area_px": round(area_px, 3),
            "equiv_diameter_px": round(equiv_d, 3),
            "effective_radius_px": round(float(np.sqrt(area_px / np.pi)), 3),
            "perimeter_px": round(perimeter_px, 3) if np.isfinite(perimeter_px) else "nan",
            "circularity": round(circularity, 4) if np.isfinite(circularity) else "nan",
            "eccentricity": round(eccentricity, 4) if np.isfinite(eccentricity) else "nan",
            "partition_coefficient": round(float(pc), 4) if np.isfinite(pc) else "nan",
        })

        droplet_count += 1
        if verbose:
            print(
                f"    Saved {out_name}  "
                f"(class={class_label}, equiv_d={equiv_d:.2f}, PC={pc:.2f}, "
                f"border_labels={border_has_labels})"
            )

    # Write the per-movie droplet location / geometry sidecar into the results folder
    # (parent of droplet_stacks/). The analysis stage merges this into droplet_metrics.csv.
    if geometry_records:
        import csv as _csv
        loc_dir = movie_out_dir.parent
        loc_dir.mkdir(parents=True, exist_ok=True)
        loc_path = loc_dir / "droplet_locations.csv"
        fieldnames = list(geometry_records[0].keys())
        try:
            with open(loc_path, "w", newline="") as f_loc:
                w = _csv.DictWriter(f_loc, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(geometry_records)
            if verbose:
                print(f"  Saved droplet locations/geometry: {loc_path.name} ({len(geometry_records)} droplet(s))")
        except OSError as e:
            if verbose:
                print(f"  Warning: could not write droplet_locations.csv: {e}")

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
        analysis_channel,
        subpixel_apply,
        exclude_bleach_from_stab,
        pre_bleach_frames,
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
        analysis_channel=analysis_channel,
        subpixel_apply=subpixel_apply,
        exclude_bleach_from_stab=exclude_bleach_from_stab,
        pre_bleach_frames=pre_bleach_frames,
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
    parser.add_argument("--max-iterations", type=int, default=500,
                        help="Maximum iterations per frame (like Fiji's maximum_iterations).")
    parser.add_argument("--epsilon", type=float, default=1e-7,
                        help="Convergence threshold (like Fiji's error_tolerance). Lower = stricter.")
    parser.add_argument("--pyramid-levels", type=int, default=4,
                        help="Number of resolution levels (like Fiji's maximum_pyramid_levels).")
    parser.add_argument("--template-update-coeff", type=float, default=0.95,
                        help="Template update coefficient (alpha) for stabilization (0-1, higher = slower update).")
    parser.add_argument("--pre-smooth-sigma", type=float, default=1.0,
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

    # Subpixel application is now the DEFAULT: it gives smooth, accurate registration so a
    # fixed ROI measures the true droplet/bleach area across all frames. Pass
    # --whole-pixel-stabilization to instead snap to whole-pixel shifts (preserves exact
    # intensities but leaves up to +/-0.5 px residual drift).
    parser.add_argument("--pre-bleach-frames", type=int, default=1,
                        help="Number of leading pre-bleach frames. The bleach event is taken to occur "
                             "between frame N-1 and frame N (default 1: frame 0 pre, frame 1 post).")
    parser.add_argument("--subpixel-stabilization", action="store_true",
                        help="(Default behaviour) Apply drift at subpixel precision with bilinear "
                             "interpolation. Kept for backward compatibility; subpixel is now the default.")
    parser.add_argument("--whole-pixel-stabilization", action="store_true",
                        help="DISCOURAGED. Apply drift at whole-pixel precision (nearest-neighbour). This "
                             "ROUNDS smooth subpixel shifts to integers and produces visible stair-step "
                             "'shaky' stabilization. Only use if exact pixel-value preservation matters more "
                             "than smooth registration. Subpixel application is the default.")
    parser.add_argument("--exclude-bleach-from-stabilization", action="store_true",
                        help="Estimate drift only from non-bleached droplets + background (mask out FRAP "
                             "droplets). Off by default: whole-image gradient-magnitude registration matches "
                             "Fiji and is already bleach-robust. Useful only for very sparse-droplet movies.")
    
    parser.add_argument("--stabilize-algorithm", type=str, default="lucas-kanade",
                        choices=["ecc", "lucas-kanade", "lk"],
                        help="Stabilization algorithm: 'lucas-kanade'/'lk' (matches ImageJ "
                             "Image Stabilizer plugin, default) or 'ecc' (OpenCV ECC).")

    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-droplet logging.")

    parser.add_argument(
        "--analysis-channel",
        type=int,
        default=1,
        choices=(1, 2, 3, 4),
        help="1-based channel for droplet detection, FRAP vs NoBleach ratio, and "
        "stabilization reference. Cropped stacks include all channels.",
    )

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
            args.analysis_channel,
            (not args.whole_pixel_stabilization),
            args.exclude_bleach_from_stabilization,
            args.pre_bleach_frames,
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
