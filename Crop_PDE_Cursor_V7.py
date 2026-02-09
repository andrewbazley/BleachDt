#!/usr/bin/env python3
"""
FRAP PDE fitting for droplet movies (explicit FD + Numba, grid search over D),
with optional photobleach correction and single-exponential plateau fitting.

Supports two analysis modes:
- "separate" model: PDE-only early + unconstrained exponential plateau (original).
- "combined" model: PDE-only early + constrained exponential tail stitched at T_switch.

Key conventions:
- Frame 0: pre-bleach (used only for normalization).
- Frame 1: first post-bleach frame; this is the PDE initial condition at t=0.
- PDE time axis is always relative to the first PDE frame (Δt between frames).
- Plotting time axis is frame_index * frame_interval.

Defaults assume: 0.11 µm/pixel, 10 s frame interval.
"""

import os
import glob
import argparse
import logging
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
from PIL import Image

from scipy.optimize import curve_fit
from scipy import stats
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    binary_dilation,
    disk,
)
from skimage.segmentation import clear_border
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
from numba import njit

logger = logging.getLogger("frap_pde_fit")


def setup_logging(verbosity: int):
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ------------------------------------------------------------
# Image I/O & masks
# ------------------------------------------------------------

def load_movie(path):
    logger.info(f"Loading movie: {path}")
    stack = tiff.imread(path).astype(np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    logger.debug(f"Movie shape: {stack.shape}")
    return stack


def make_droplet_mask(pre_frame, min_size=50):
    logger.info("Generating droplet mask (Otsu on pre-bleach)...")
    blur = gaussian(pre_frame, sigma=1.0)
    thr = threshold_otsu(blur)
    mask = blur > thr
    mask = clear_border(mask)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = binary_closing(mask, disk(2))
    mask = binary_fill_holes(mask)
    logger.debug(
        f"Droplet mask: {int(mask.sum())} pixels, min_size={min_size}, thr={thr:.3f}"
    )
    return mask.astype(bool)


def make_bleach_mask(pre_frame, post_frame, droplet_mask, ratio_percentile=90.0):
    logger.info("Generating bleach mask from pre/post ratio...")
    eps = 1e-6
    ratio = pre_frame / (post_frame + eps)
    ratio = gaussian(ratio, sigma=1.0)
    ratio[~droplet_mask] = 0.0

    vals = ratio[droplet_mask]
    if vals.size == 0:
        logger.warning("No droplet pixels when building bleach mask.")
        return np.zeros_like(droplet_mask, dtype=bool)

    thr = np.percentile(vals, ratio_percentile)
    mask = ratio >= thr
    mask = binary_opening(mask, disk(1))
    mask = binary_dilation(mask, disk(1))
    mask &= droplet_mask

    logger.debug(
        f"Bleach mask: {int(mask.sum())} pixels, percentile={ratio_percentile}, thr={thr:.3f}"
    )
    return mask.astype(bool)


def estimate_background(pre_frame, droplet_mask):
    bg_pixels = pre_frame[~droplet_mask]
    if bg_pixels.size == 0:
        bg = float(np.median(pre_frame))
        logger.warning("No background pixels outside droplet; using global median.")
    else:
        bg = float(np.median(bg_pixels))
    logger.debug(f"Estimated background: {bg:.6g}")
    return bg


def normalize_stack(stack, droplet_mask):
    logger.info("Background subtraction and normalization (to droplet mean)...")
    pre = stack[0]
    bg = estimate_background(pre, droplet_mask)
    stack_bs = stack - bg
    stack_bs[stack_bs < 0] = 0.0

    pre_mean = stack_bs[0][droplet_mask].mean()
    if pre_mean <= 0:
        logger.warning("Pre-bleach mean <=0; forcing pre_mean=1.0 for normalization.")
        pre_mean = 1.0

    norm_stack = stack_bs / (pre_mean + 1e-6)
    logger.debug(
        f"Pre-bleach mean in droplet: {pre_mean:.6g}, "
        f"norm_stack min/max: {norm_stack.min():.3f}/{norm_stack.max():.3f}"
    )
    return norm_stack.astype(np.float32), bg, pre_mean


def crop_to_droplet(norm_stack, droplet_mask, bleach_mask, margin=5):
    ys, xs = np.where(droplet_mask)
    if ys.size == 0 or xs.size == 0:
        logger.warning("Droplet mask empty, skipping crop.")
        H, W = droplet_mask.shape
        return norm_stack, droplet_mask, bleach_mask, (0, H, 0, W)

    H, W = droplet_mask.shape
    y0 = max(ys.min() - margin, 0)
    y1 = min(ys.max() + 1 + margin, H)
    x0 = max(xs.min() - margin, 0)
    x1 = min(xs.max() + 1 + margin, W)

    logger.info(f"Cropping to droplet bbox: y[{y0}:{y1}], x[{x0}:{x1}], margin={margin}")
    norm_crop = norm_stack[:, y0:y1, x0:x1]
    droplet_crop = droplet_mask[y0:y1, x0:x1]
    bleach_crop = bleach_mask[y0:y1, x0:x1]
    return norm_crop, droplet_crop, bleach_crop, (y0, y1, x0, x1)


def downsample_stack_and_masks(stack, droplet_mask, bleach_mask, factor):
    if factor <= 1:
        return stack, droplet_mask, bleach_mask

    logger.info(f"Downsampling stack and masks by factor={factor}...")
    T, H, W = stack.shape
    new_H = H // factor
    new_W = W // factor
    if new_H < 2 or new_W < 2:
        logger.warning("Downsample factor too large for crop size; skipping downsample.")
        return stack, droplet_mask, bleach_mask

    stack_ds = np.empty((T, new_H, new_W), dtype=np.float32)
    for i in range(T):
        stack_ds[i] = resize(
            stack[i],
            (new_H, new_W),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    droplet_ds = resize(
        droplet_mask.astype(float),
        (new_H, new_W),
        order=0,
        mode="constant",
        anti_aliasing=False,
        preserve_range=True,
    ) >= 0.5

    bleach_ds = resize(
        bleach_mask.astype(float),
        (new_H, new_W),
        order=0,
        mode="constant",
        anti_aliasing=False,
        preserve_range=True,
    ) >= 0.5

    logger.debug(f"Downsampled to shape: {stack_ds.shape}")
    return stack_ds, droplet_ds, bleach_ds


# ------------------------------------------------------------
# Photobleach correction & noise / FRAP curve
# ------------------------------------------------------------

def build_photobleach_region(norm_stack, droplet_ds, bleach_ds,
                             pb_exclude_radius=2, pb_ratio_tol=0.1):
    """
    Build a 'safe' region for photobleach estimation:
    - exclude a dilation of the bleach mask
    - keep only pixels whose frame0/frame1 ratio is ~1 (within tolerance)
    """
    logger.info("Building photobleach/noise region...")
    T = norm_stack.shape[0]
    if T < 2:
        logger.warning("Movie has <2 frames, using whole droplet for PB region.")
        return droplet_ds.copy()

    if pb_exclude_radius > 0:
        pb_exclude = binary_dilation(bleach_ds, disk(pb_exclude_radius)) & droplet_ds
    else:
        pb_exclude = bleach_ds & droplet_ds

    base_region = droplet_ds & ~pb_exclude
    if not np.any(base_region):
        logger.warning("Base PB region empty; using full droplet.")
        base_region = droplet_ds.copy()

    eps = 1e-6
    frame0 = norm_stack[0]
    frame1 = norm_stack[1]
    ratio = frame0 / (frame1 + eps)
    vals = ratio[base_region]
    if vals.size == 0:
        logger.warning("No valid pixels in base PB region; using full droplet.")
        return droplet_ds.copy()

    r0 = np.median(vals)
    lower = r0 * (1.0 - pb_ratio_tol)
    upper = r0 * (1.0 + pb_ratio_tol)
    good = (ratio >= lower) & (ratio <= upper)
    region = base_region & good

    if not np.any(region):
        logger.warning("PB ratio filter removed all pixels; falling back to base region.")
        region = base_region

    logger.debug(f"PB region: {int(region.sum())} pixels "
                 f"(exclude radius={pb_exclude_radius}, tol={pb_ratio_tol})")
    return region


def apply_photobleach_correction(norm_stack, pb_region):
    logger.info("Applying photobleach correction from PB region...")
    T = norm_stack.shape[0]
    curve = np.zeros(T, dtype=np.float32)

    for i in range(T):
        vals = norm_stack[i][pb_region]
        curve[i] = vals.mean() if vals.size > 0 else np.nan

    valid = np.isfinite(curve) & (curve > 0)
    if not np.any(valid):
        logger.warning("Photobleach curve invalid; skipping correction.")
        factors = np.ones(T, dtype=np.float32)
        return norm_stack, curve, factors

    ref = curve[0]
    if ref <= 0:
        ref = np.nanmax(curve[valid])
    factors = np.ones(T, dtype=np.float32)
    factors[valid] = curve[valid] / ref
    factors[factors <= 0] = 1.0

    corrected = norm_stack / factors[:, None, None]
    logger.debug(
        f"Photobleach correction factors: min={factors.min():.3f}, max={factors.max():.3f}"
    )
    return corrected.astype(np.float32), curve, factors


def estimate_noise(norm_stack, droplet_mask, bleach_mask, region=None):
    logger.info("Estimating noise from droplet pixels outside bleach ROI...")
    if region is None:
        region = droplet_mask & ~bleach_mask
        if not np.any(region):
            logger.warning("No droplet pixels outside bleach; using entire droplet for noise.")
            region = droplet_mask.copy()

    T = norm_stack.shape[0]
    sigmas = np.zeros(T, dtype=np.float32)
    for i in range(T):
        vals = norm_stack[i][region]
        sigmas[i] = np.std(vals) if vals.size > 0 else 0.0

    sigma_global = float(np.median(sigmas[sigmas > 0])) if np.any(sigmas > 0) else 0.0
    logger.debug(
        f"Noise sigma global={sigma_global:.4g}, per-frame range=({sigmas.min():.4g},{sigmas.max():.4g})"
    )
    return sigma_global, sigmas, region


def compute_frap_curve(norm_stack, roi_mask):
    T = norm_stack.shape[0]
    frap = np.empty(T, dtype=np.float32)
    for i in range(T):
        vals = norm_stack[i][roi_mask]
        frap[i] = vals.mean() if vals.size > 0 else np.nan
    return frap


# ------------------------------------------------------------
# Numba PDE + cost
# ------------------------------------------------------------

@njit(cache=True, fastmath=True)
def diffusion_step_jit(c, mask, alpha):
    """Optimized diffusion step with fastmath and caching."""
    ny, nx = c.shape
    out = np.empty_like(c)
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x]:
                out[y, x] = 0.0
                continue

            center = c[y, x]

            yy = y - 1
            if yy >= 0 and mask[yy, x]:
                up = c[yy, x]
            else:
                up = center

            yy = y + 1
            if yy < ny and mask[yy, x]:
                down = c[yy, x]
            else:
                down = center

            xx = x - 1
            if xx >= 0 and mask[y, xx]:
                left = c[y, xx]
            else:
                left = center

            xx = x + 1
            if xx < nx and mask[y, xx]:
                right = c[y, xx]
            else:
                right = center

            lap = up + down + left + right - 4.0 * center
            out[y, x] = center + alpha * lap
    return out


@njit(cache=True, fastmath=True)
def simulate_diffusion_jit(D, c0, mask, times, dt_sim, dx):
    """Optimized diffusion simulation with fastmath and caching."""
    ny, nx = c0.shape
    nT = times.shape[0]
    snaps = np.empty((nT, ny, nx), dtype=np.float32)

    alpha = D * dt_sim / (dx * dx)

    c = c0.copy()
    t = 0.0
    target = 0
    eps = 1e-12
    while target < nT:
        t_target = times[target]
        while t < t_target - eps:
            c = diffusion_step_jit(c, mask, alpha)
            t += dt_sim
        snaps[target, :, :] = c
        target += 1

    return snaps


@njit(cache=True, fastmath=True)
def cost_D_jit(D, c0, data_stack, domain_mask, fit_mask, times, weights, dt_sim, dx):
    """Optimized cost calculation with fastmath and caching."""
    snaps = simulate_diffusion_jit(D, c0, domain_mask, times, dt_sim, dx)
    nT, ny, nx = snaps.shape
    total = 0.0

    for k in range(nT):
        w = weights[k]
        if w <= 0.0:
            continue
        # Optimized: check fit_mask once per pixel, compute diff inline
        for y in range(ny):
            for x in range(nx):
                if fit_mask[y, x]:
                    diff = snaps[k, y, x] - data_stack[k, y, x]
                    total += w * diff * diff
    return total


def compute_cost_for_D(D, c0, data_stack, domain_mask, fit_mask, times, weights, dt_sim, dx):
    """Compute cost for a given D value. Optimized for Numba JIT compilation."""
    # Ensure all inputs are properly typed and contiguous for Numba
    return float(
        cost_D_jit(
            float(D),
            c0,
            data_stack,
            domain_mask,
            fit_mask,
            times.astype(np.float32),
            weights.astype(np.float32),
            float(dt_sim),
            float(dx),
        )
    )


def _worker_D(args):
    """Top-level worker function for parallel D evaluation (pickle-safe)."""
    D, c0, data_stack, domain_mask, fit_mask, times, weights, dt_sim, dx = args
    return D, compute_cost_for_D(D, c0, data_stack, domain_mask, fit_mask, times, weights, dt_sim, dx)


def grid_search_D(
    D_min,
    D_max,
    n_grid,
    n_grid_refine,
    c0,
    data_stack,
    domain_mask,
    fit_mask,
    times,
    weights,
    dt_sim,
    dx,
    n_jobs_D,
):
    """
    Grid search for optimal D value using two-stage refinement.
    
    Uses ThreadPoolExecutor instead of ProcessPoolExecutor because:
    1. Numba-compiled functions release the GIL, enabling true parallelism with threads
    2. Avoids pickling overhead and issues with closures
    3. Faster for CPU-bound Numba code
    """
    # Ensure arrays are contiguous for better performance
    c0 = np.ascontiguousarray(c0, dtype=np.float32)
    data_stack = np.ascontiguousarray(data_stack, dtype=np.float32)
    domain_mask = np.ascontiguousarray(domain_mask, dtype=bool)
    fit_mask = np.ascontiguousarray(fit_mask, dtype=bool)
    times = np.ascontiguousarray(times, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    
    Ds1 = np.linspace(D_min, D_max, n_grid, dtype=np.float64)
    logger.info(f"Stage 1 D grid: {n_grid} points in [{D_min}, {D_max}]")

    # Prepare arguments for worker (avoid closure for better performance)
    worker_args = [(D, c0, data_stack, domain_mask, fit_mask, times, weights, dt_sim, dx) for D in Ds1]
    
    costs1 = np.empty_like(Ds1)
    if n_jobs_D > 1:
        logger.info(f"Parallelizing stage 1 D evaluations over {n_jobs_D} workers (threading)...")
        with ThreadPoolExecutor(max_workers=n_jobs_D) as ex:
            futures = {ex.submit(_worker_D, args): i for i, args in enumerate(worker_args)}
            for fut in as_completed(futures):
                i = futures[fut]
                D, c = fut.result()
                costs1[i] = c
    else:
        for i, args in enumerate(worker_args):
            _, c = _worker_D(args)
            costs1[i] = c

    idx_best1 = int(np.argmin(costs1))
    D_best1 = float(Ds1[idx_best1])
    cost_best1 = float(costs1[idx_best1])
    logger.info(f"Stage 1 best D ~ {D_best1:.4g} (cost={cost_best1:.4g})")

    span = (D_max - D_min) / 4.0
    D_lo = max(D_min, D_best1 - span)
    D_hi = min(D_max, D_best1 + span)
    Ds2 = np.linspace(D_lo, D_hi, n_grid_refine, dtype=np.float64)
    logger.info(f"Stage 2 D grid: {n_grid_refine} points in [{D_lo}, {D_hi}]")

    # Prepare arguments for refinement stage
    worker_args2 = [(D, c0, data_stack, domain_mask, fit_mask, times, weights, dt_sim, dx) for D in Ds2]
    
    costs2 = np.empty_like(Ds2)
    if n_jobs_D > 1:
        logger.info(f"Parallelizing stage 2 D evaluations over {n_jobs_D} workers (threading)...")
        with ThreadPoolExecutor(max_workers=n_jobs_D) as ex:
            futures = {ex.submit(_worker_D, args): i for i, args in enumerate(worker_args2)}
            for fut in as_completed(futures):
                i = futures[fut]
                D, c = fut.result()
                costs2[i] = c
    else:
        for i, args in enumerate(worker_args2):
            _, c = _worker_D(args)
            costs2[i] = c

    idx_best2 = int(np.argmin(costs2))
    D_best2 = float(Ds2[idx_best2])
    cost_best2 = float(costs2[idx_best2])
    logger.info(f"Stage 2 best D ~ {D_best2:.4g} (cost={cost_best2:.4g})")

    return D_best2, cost_best2


# ------------------------------------------------------------
# Single-exponential plateau (separate model)
# ------------------------------------------------------------

def single_exp_plateau(t, YM, Y0, k):
    return YM - (YM - Y0) * np.exp(-k * t)


def fit_single_exp_plateau(times, intensities, constrain_Y0=None):
    """
    Fit single exponential plateau.
    
    Parameters
    ----------
    times : array-like
        Time points
    intensities : array-like
        Intensity values
    constrain_Y0 : float, optional
        If provided, fix Y0 to this value and only fit YM and k.
    """
    mask = np.isfinite(times) & np.isfinite(intensities)
    t = np.asarray(times)[mask]
    y = np.asarray(intensities)[mask]
    
    if constrain_Y0 is not None:
        # Constrained fit: Y0 is fixed, only fit YM and k
        if t.size < 2:
            return {"success": False}
        
        Y0_fixed = float(constrain_Y0)
        YM0 = float(np.nanmax(y))
        k0 = 0.01
        
        def constrained_exp(t, YM, k):
            return YM - (YM - Y0_fixed) * np.exp(-k * t)
        
        p0 = [YM0, k0]
        
        try:
            popt, _ = curve_fit(constrained_exp, t, y, p0=p0, maxfev=20000)
            YM, k = popt
            Y0 = Y0_fixed
            y_pred = constrained_exp(t, YM, k)
            
            mobile_fraction = YM - Y0
            immobile_fraction = 1.0 - mobile_fraction
            
            return {
                "success": True,
                "YM": float(YM),
                "Y0": float(Y0),
                "k": float(k),
                "mobile_fraction": float(mobile_fraction),
                "immobile_fraction": float(immobile_fraction),
                "t": t,
                "y_pred": y_pred,
            }
        except Exception as e:
            logger.warning(f"Constrained single-exp plateau fit failed: {e}")
            return {"success": False}
    else:
        # Unconstrained fit (original behavior)
        if t.size < 3:
            return {"success": False}

        YM0 = float(np.nanmax(y))
        Y0_0 = float(y[0])
        k0 = 0.01
        p0 = [YM0, Y0_0, k0]

        try:
            popt, _ = curve_fit(single_exp_plateau, t, y, p0=p0, maxfev=20000)
            YM, Y0, k = popt
            y_pred = single_exp_plateau(t, *popt)
        except Exception as e:
            logger.warning(f"Single-exp plateau fit failed: {e}")
            return {"success": False}

        mobile_fraction = YM - Y0
        immobile_fraction = 1.0 - mobile_fraction

        return {
            "success": True,
            "YM": float(YM),
            "Y0": float(Y0),
            "k": float(k),
            "mobile_fraction": float(mobile_fraction),
            "immobile_fraction": float(immobile_fraction),
            "t": t,
            "y_pred": y_pred,
        }


# ------------------------------------------------------------
# Combined model helpers (PDE + constrained exponential tail)
# ------------------------------------------------------------

def compute_pde_mean_curve(
    D,
    c0,
    domain_mask,
    fit_mask,
    frame_ids,
    frame_interval,
    dt_sim,
    dx,
):
    """
    Return (t_plot, mean_intensity) where t_plot is aligned with the data axis.

    - PDE time axis starts at t_rel = 0 at the first PDE frame in `frame_ids`.
    - We simulate at times_rel = (frame_ids - frame_ids[0]) * frame_interval.
    - We return t_plot = frame_ids * frame_interval (so frame 1 is at Δt, etc.).
    """
    if frame_ids.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    times_rel = (frame_ids - frame_ids[0]).astype(np.float32) * frame_interval
    snaps = simulate_diffusion_jit(
        D,
        c0.astype(np.float32),
        domain_mask,
        times_rel.astype(np.float32),
        float(dt_sim),
        float(dx),
    )
    pred_curve = np.empty(len(frame_ids), dtype=np.float32)
    for k in range(len(frame_ids)):
        vals = snaps[k][fit_mask]
        pred_curve[k] = vals.mean() if vals.size > 0 else np.nan

    t_plot = frame_ids.astype(np.float32) * frame_interval
    return t_plot, pred_curve


def fit_constrained_exp_tail(
    frap_curve,
    frame_interval,
    switch_frame,
    I_switch,
    last_frame_index,
):
    """
    Fit a constrained exponential tail that *starts from a fixed value* I_switch:

        I(t) = Y_M - (Y_M - I_switch) * exp(-k * (t - t_switch))

    where t_switch is the time of frame `switch_frame` and I_switch is usually the
    PDE-predicted mean at that frame. We fit Y_M and k using frames >= switch_frame.
    """
    if last_frame_index <= switch_frame:
        return {"success": False}

    frame_ids_tail = np.arange(switch_frame, last_frame_index + 1, dtype=int)
    t_local = (frame_ids_tail - switch_frame).astype(np.float32) * frame_interval
    y_tail = frap_curve[frame_ids_tail]

    def constrained_exp_local(t_loc, YM_local, k_local):
        return YM_local - (YM_local - I_switch) * np.exp(-k_local * t_loc)

    p0 = [float(np.nanmax(y_tail)), 0.01]
    try:
        popt, _ = curve_fit(constrained_exp_local, t_local, y_tail, p0=p0, maxfev=20000)
        YM_c, k_c = popt
        y_pred_tail = constrained_exp_local(t_local, YM_c, k_c)

        Y0_c = float(I_switch)
        mobile_fraction = YM_c - Y0_c
        immobile_fraction = 1.0 - mobile_fraction

        return {
            "success": True,
            "YM": float(YM_c),
            "Y0": float(Y0_c),
            "k": float(k_c),
            "mobile_fraction": float(mobile_fraction),
            "immobile_fraction": float(immobile_fraction),
            "frame_ids_tail": frame_ids_tail,
            "t_local": t_local,
            "y_pred_tail": y_pred_tail,
        }
    except Exception as e:
        logger.warning(f"Constrained exponential tail fit failed: {e}")
        return {"success": False}


def build_combined_curve_and_cost(
    frap_curve,
    frame_interval,
    frame_ids_pde,
    pde_curve_all,
    switch_frame,
    tail_fit,
    exp_last_eff,
):
    """
    Given:
      - frap_curve (data),
      - PDE ROI-mean curve over frame_ids_pde,
      - chosen switch_frame,
      - constrained exponential tail fit (tail_fit),
    build the combined model curve (PDE early + exp tail late) and
    return (cost, combined_curve).

    Cost is SSE vs data over frames from frame_ids_pde[0] .. exp_last_eff.
    """
    T = len(frap_curve)
    combined_curve = np.full(T, np.nan, dtype=np.float32)

    # Map PDE values into global frame indices
    for f, val in zip(frame_ids_pde, pde_curve_all):
        if f <= switch_frame:
            combined_curve[f] = val

    # Exponential tail from switch_frame forward
    if tail_fit is not None and tail_fit.get("success", False):
        YM_c = tail_fit["YM"]
        k_c = tail_fit["k"]
        I_switch = tail_fit["Y0"]

        frames_tail_all = np.arange(switch_frame, exp_last_eff + 1, dtype=int)
        t_tail_all = (frames_tail_all - switch_frame).astype(np.float32) * frame_interval

        def constrained_exp_all(t_loc):
            return YM_c - (YM_c - I_switch) * np.exp(-k_c * t_loc)

        combined_tail = constrained_exp_all(t_tail_all)
        for frame_idx, val in zip(frames_tail_all, combined_tail):
            combined_curve[frame_idx] = val

    # Cost over the combined time window
    f0 = int(frame_ids_pde[0])
    f1 = int(exp_last_eff)
    frames_eval = np.arange(f0, f1 + 1, dtype=int)

    y_data = frap_curve[frames_eval]
    y_model = combined_curve[frames_eval]
    mask = np.isfinite(y_data) & np.isfinite(y_model)
    if not np.any(mask):
        return np.inf, combined_curve

    diff = y_data[mask] - y_model[mask]
    cost = float(np.sum(diff * diff))
    return cost, combined_curve


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

def fig_to_array(fig, dpi=100):
    """Convert matplotlib figure to numpy array (RGB)."""
    # Set figure DPI before rendering to reduce file size
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert RGBA to RGB
    if img.shape[2] == 4:
        # Remove alpha channel
        img = img[:, :, :3]
    return img


class ImageCollector:
    """Collect images by droplet type and plot type for stacking."""
    def __init__(self):
        self.images = {}  # {(droplet_type, plot_type): [list of image arrays]}
        self.filenames = {}  # {(droplet_type, plot_type): [list of filenames]}
    
    def add_image(self, droplet_type, plot_type, image_array, filename):
        """Add an image to the collection."""
        key = (droplet_type, plot_type)
        if key not in self.images:
            self.images[key] = []
            self.filenames[key] = []
        self.images[key].append(image_array)
        self.filenames[key].append(filename)
    
    def save_stacks(self, output_dir):
        """Save all collected images as TIF stacks."""
        os.makedirs(output_dir, exist_ok=True)
        
        for (droplet_type, plot_type), img_list in self.images.items():
            if not img_list:
                continue
            
            # Ensure all images have the same size (resize if needed)
            # Get target size from first image
            target_h, target_w = img_list[0].shape[:2]
            
            # Resize all images to match
            resized_images = []
            for img in img_list:
                if img.shape[:2] != (target_h, target_w):
                    # Resize using PIL
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    img = np.array(pil_img)
                resized_images.append(img)
            
            # Stack images: shape will be (N, H, W, 3) for RGB
            stack = np.stack(resized_images, axis=0)
            
            # Save as TIF
            stack_name = f"{droplet_type}_{plot_type}.tif"
            stack_path = os.path.join(output_dir, stack_name)
            tiff.imwrite(stack_path, stack, imagej=True)
            logger.info(f"Saved {len(img_list)} images to {stack_path}")


def plot_masks(frame_for_display, droplet_mask_full, bleach_mask_full, return_array=False):
    """Plot droplet and bleach masks. Returns image array if return_array=True, else saves to file."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(frame_for_display, cmap="gray")
    ax.contour(droplet_mask_full, levels=[0.5], colors="cyan", linewidths=1.5)
    ax.contour(bleach_mask_full, levels=[0.5], colors="red", linewidths=1.5)
    ax.set_title("Droplet (cyan) and bleach ROI (red)")
    ax.axis("off")
    fig.tight_layout()
    
    if return_array:
        img_array = fig_to_array(fig)
        plt.close(fig)
        return img_array
    else:
        # Legacy: save to file (for backward compatibility during transition)
        import tempfile
        temp_mask_path = os.path.join(tempfile.gettempdir(), "temp_mask.png")
        fig.savefig(temp_mask_path, dpi=200)
        plt.close(fig)
        return None


def plot_frap_separate(
    frap_curve,
    D_best,
    domain_mask,
    fit_mask,
    norm_stack_crop_ds,
    frame_interval,
    pixel_size_eff,
    D_min,
    D_max,
    out_path=None,
    exp_fit=None,
    frame_ids_pde=None,
    frame_ids_exp=None,
    return_array=False,
    droplet_label=None,
):
    """
    Plot FRAP data, PDE fit (only over PDE frame window), and exponential plateau fit
    for the original "separate" model.
    Returns image array if return_array=True, else saves to file.
    """
    T = len(frap_curve)
    t_full = np.arange(T, dtype=np.float32) * frame_interval

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_full, frap_curve, "ko-", label="Data (fit ROI)")

    # PDE curve only over the PDE fit frames
    if (
        np.isfinite(D_best)
        and frame_ids_pde is not None
        and len(frame_ids_pde) > 0
    ):
        dx = pixel_size_eff
        dt_stable = 0.24 * dx * dx / max(D_max, 1e-6)
        dt_sim = dt_stable

        c0 = norm_stack_crop_ds[1] * domain_mask
        times_pde, pred_curve = compute_pde_mean_curve(
            D_best,
            c0,
            domain_mask,
            fit_mask,
            frame_ids_pde,
            frame_interval,
            dt_sim,
            dx,
        )
        ax.plot(
            times_pde,
            pred_curve,
            "r-",
            label=f"PDE fit (D={D_best:.3f} µm²/s)",
        )

    # Exponential plateau curve over its fit window
    # Calculate time axis from frame indices to ensure it starts at frame 1
    if exp_fit is not None and exp_fit.get("success", False):
        y_exp = exp_fit["y_pred"]
        # Use frame_ids_exp if provided, otherwise calculate from exp_fit["t"]
        if frame_ids_exp is not None and len(frame_ids_exp) == len(y_exp):
            # Calculate time directly from frame indices: frame 1 is at frame_interval
            t_exp_abs = frame_ids_exp.astype(np.float32) * frame_interval
        else:
            # Fallback: exp_fit["t"] is relative to first post-bleach frame (t=0 at frame 1)
            t_exp_rel = exp_fit["t"]
            # Convert to absolute time: frame 1 is at frame_interval
            t_exp_abs = t_exp_rel + frame_interval
        ax.plot(
            t_exp_abs,
            y_exp,
            "b--",
            label=f"Exp plateau (k={exp_fit['k']:.3g} s$^{{-1}}$)",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized intensity (fit ROI)")
    title_lbl = f"{droplet_label}" if droplet_label else "FRAP fit (separate)"
    ax.set_title(title_lbl)
    ax.legend()
    fig.tight_layout()
    
    if return_array:
        img_array = fig_to_array(fig)
        plt.close(fig)
        return img_array
    else:
        if out_path:
            fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None


def plot_frap_combined(
    frap_curve,
    frame_interval,
    D_best,
    c0,
    domain_mask,
    fit_mask,
    pixel_size_eff,
    D_max,
    frame_ids_pde,
    combined_curve,
    T_switch_frame,
    out_path=None,
    return_array=False,
    droplet_label=None,
):
    """
    Plot FRAP data, PDE-only curve, and combined (PDE + constrained tail) model.

    - PDE-only curve is shown in red over the PDE window.
    - Combined model (PDE early + constrained exponential tail) is shown in green.
    - Pre-bleach frame (frame 0) is *not* modeled; only the data point is shown at t=0.
    Returns image array if return_array=True, else saves to file.
    """
    T = len(frap_curve)
    t_full = np.arange(T, dtype=np.float32) * frame_interval

    dx = pixel_size_eff
    dt_stable = 0.24 * dx * dx / max(D_max, 1e-6)
    dt_sim = dt_stable

    # PDE prediction over PDE window (for plotting)
    times_pde, pde_curve = compute_pde_mean_curve(
        D_best,
        c0,
        domain_mask,
        fit_mask,
        frame_ids_pde,
        frame_interval,
        dt_sim,
        dx,
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    # Data (including pre-bleach point at t=0)
    ax.plot(t_full, frap_curve, "ko-", label="Data (fit ROI)")

    # PDE-only (red), no model at pre-bleach frame
    if times_pde.size > 0:
        ax.plot(times_pde, pde_curve, "r-", label="PDE-only (early)")

    # Combined model (green)
    if combined_curve is not None:
        ax.plot(
            t_full,
            combined_curve,
            "g-",
            linewidth=2.0,
            label="Combined PDE+exp",
        )

    # T_switch vertical line at the *data* time for that frame
    ax.axvline(T_switch_frame * frame_interval, color="gray",
               linestyle=":", label="T_switch")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized intensity (fit ROI)")
    title_lbl = f"{droplet_label}" if droplet_label else "FRAP fit (combined)"
    ax.set_title(title_lbl)
    ax.legend()
    fig.tight_layout()
    
    if return_array:
        img_array = fig_to_array(fig)
        plt.close(fig)
        return img_array
    else:
        if out_path:
            fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None


def plot_noise_pixels(
    norm_stack_crop,
    noise_region_crop,
    sigma_per_frame,
    out_path=None,
    max_points_per_frame=500,
    weights_pde=None,
    frame_ids_pde=None,
    return_array=False,
    droplet_label=None,
):
    logger.info(f"Saving noise pixel plot to {out_path}")
    T = norm_stack_crop.shape[0]

    frames_all, values_all, colors_all = [], [], []
    mean_per_frame = np.full(T, np.nan, dtype=np.float32)

    for i in range(T):
        vals_full = norm_stack_crop[i][noise_region_crop]
        if vals_full.size == 0:
            continue

        mean_per_frame[i] = vals_full.mean()

        vals = vals_full
        if vals.size > max_points_per_frame:
            idx = np.random.choice(vals.size, max_points_per_frame, replace=False)
            vals = vals[idx]
        frames_all.append(np.full(vals.shape, i))
        values_all.append(vals)
        colors_all.append(np.full(vals.shape, sigma_per_frame[i]))

    if not frames_all:
        logger.warning("No noise-region pixels found; skipping noise plot.")
        return

    frames_all = np.concatenate(frames_all)
    values_all = np.concatenate(values_all)
    colors_all = np.concatenate(colors_all)

    eps = 1e-6
    relative_sigma = sigma_per_frame / (mean_per_frame + eps)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    sc = axes[0].scatter(frames_all, values_all, c=colors_all, s=4, alpha=0.5)
    axes[0].set_ylabel("Noise-region pixel intensity")
    cbar = fig.colorbar(sc, ax=axes[0])
    cbar.set_label("Per-frame σ (noise)")

    frames = np.arange(T)
    axes[1].plot(frames, sigma_per_frame, "o-", label="σ (absolute)", color='blue')
    axes[1].plot(frames, relative_sigma, "o--", label="σ / mean (relative)", color='green')
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Noise σ", color='blue')
    axes[1].tick_params(axis='y', labelcolor='blue')
    
    # Add second Y axis for PDE weights if provided
    if weights_pde is not None and frame_ids_pde is not None:
        ax2 = axes[1].twinx()
        # Map weights to full frame array
        weights_full = np.full(T, np.nan, dtype=np.float32)
        for idx, fid in enumerate(frame_ids_pde):
            if 0 <= fid < T:
                weights_full[fid] = weights_pde[idx]
        ax2.plot(frames, weights_full, "s-", label="PDE weight", color='red', markersize=4)
        ax2.set_ylabel("PDE model weight", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # Combine legends
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        axes[1].legend()

    if droplet_label:
        fig.suptitle(droplet_label, y=1.02)
    fig.tight_layout()
    
    if return_array:
        img_array = fig_to_array(fig)
        plt.close(fig)
        return img_array
    else:
        if out_path:
            fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None



# ------------------------------------------------------------
# Control (NoBleach) movies: boundary growth & global photobleach
# ------------------------------------------------------------

def is_control_movie(basename: str) -> bool:
    """Heuristic to detect non-FRAP control droplets.

    Currently treats any file whose basename contains 'NoBleach' (case-insensitive)
    as a control. You can extend this by renaming files (e.g. 'control_*.tif').
    """
    name = basename.lower()
    return "nobleach" in name


def apply_photobleach_correction_from_curve(norm_stack, pb_curve_external):
    """Apply photobleach correction using an externally supplied curve.

    Parameters
    ----------
    norm_stack : (T, Y, X) float32
        Background-subtracted, pre-bleach-normalized movie.
    pb_curve_external : 1D array-like, length >= 1
        Mean droplet intensity vs frame index from a NoBleach control movie,
        already normalized so frame 0 ~ 1.0.

    Returns
    -------
    corrected_stack : (T, Y, X) float32
    pb_curve_used   : 1D float32, length T
    factors         : 1D float32, length T (multiplicative correction factors)
    """
    T = norm_stack.shape[0]
    pb_curve_external = np.asarray(pb_curve_external, dtype=np.float32).ravel()
    if pb_curve_external.size < 1:
        # Degenerate; no correction
        logger.warning("External photobleach curve empty; skipping correction.")
        factors = np.ones(T, dtype=np.float32)
        return norm_stack.astype(np.float32), factors.copy(), factors

    # Use as many points as we have frames, pad with last value if needed
    pb_used = np.empty(T, dtype=np.float32)
    L = min(T, pb_curve_external.size)
    pb_used[:L] = pb_curve_external[:L]
    if L < T:
        pb_used[L:] = pb_curve_external[-1]

    # Normalize relative to first frame
    ref = pb_used[0]
    if ref <= 0:
        ref = 1.0
    factors = pb_used / ref
    factors[factors <= 0] = 1.0

    corrected = norm_stack / factors[:, None, None]
    return corrected.astype(np.float32), pb_used, factors


def analyze_control_movie(
    stack,
    norm_full,
    pixel_size,
    frame_interval,
    droplet_min_size,
):
    """Quantify growth/exchange in a NoBleach control droplet.

    For each frame we:
    - segment the droplet (Otsu + morphology),
    - measure droplet and background mean intensity from the normalized stack,
    - measure droplet area and an effective radius,
    - compute a partition coefficient K = I_droplet / I_background.

    Returns
    -------
    metrics : dict of 1D arrays (time_s, I_droplet, I_background, K, area_um2, radius_um)
    pb_curve : 1D array, droplet intensity normalized to frame 0 (for PB reference)
    """
    T, H, W = stack.shape
    times = np.arange(T, dtype=np.float32) * frame_interval

    I_droplet = np.full(T, np.nan, dtype=np.float32)
    I_bg = np.full(T, np.nan, dtype=np.float32)
    K = np.full(T, np.nan, dtype=np.float32)
    area_px = np.zeros(T, dtype=np.float32)

    baseline_mask = make_droplet_mask(stack[0], min_size=droplet_min_size)

    for i in range(T):
        frame = stack[i]
        try:
            mask_t = make_droplet_mask(frame, min_size=droplet_min_size)
            if not np.any(mask_t):
                mask_t = baseline_mask
        except Exception:
            mask_t = baseline_mask

        if not np.any(mask_t):
            continue

        area_px[i] = float(mask_t.sum())

        vals_drop = norm_full[i][mask_t]
        vals_bg = norm_full[i][~mask_t]

        if vals_drop.size > 0:
            I_droplet[i] = float(np.mean(vals_drop))
        if vals_bg.size > 0:
            I_bg[i] = float(np.mean(vals_bg))
            if I_bg[i] > 0:
                K[i] = I_droplet[i] / I_bg[i]

    area_um2 = area_px * (pixel_size ** 2)
    radius_um = np.sqrt(area_um2 / (np.pi + 1e-12))

    # Photobleach curve for use as reference: droplet intensity normalized to frame 0
    valid = np.isfinite(I_droplet) & (I_droplet > 0)
    if np.any(valid):
        I0 = I_droplet[0] if I_droplet[0] > 0 else np.nanmax(I_droplet[valid])
    else:
        I0 = 1.0
    if I0 <= 0:
        I0 = 1.0
    pb_curve = I_droplet / I0

    metrics = {
        "time_s": times,
        "I_droplet": I_droplet,
        "I_background": I_bg,
        "K_partition": K,
        "area_um2": area_um2,
        "radius_um": radius_um,
    }
    return metrics, pb_curve


def plot_control_metrics(basename, metrics, out_dir, return_array=False, image_collector=None):
    """Plot control droplet metrics (intensity, partition coefficient, size)."""
    t = metrics["time_s"]
    I_d = metrics["I_droplet"]
    I_bg = metrics["I_background"]
    K = metrics["K_partition"]
    radius_um = metrics["radius_um"]

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    axes[0].plot(t, I_d, "o-", label="Droplet")
    axes[0].plot(t, I_bg, "o--", label="Background")
    axes[0].set_ylabel("Normalized intensity")
    axes[0].legend()

    axes[1].plot(t, K, "o-")
    axes[1].set_ylabel("Partition coeff. (dense/dilute)")

    axes[2].plot(t, radius_um, "o-")
    axes[2].set_ylabel("Effective radius (µm)")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    
    if return_array or image_collector is not None:
        img_array = fig_to_array(fig)
        plt.close(fig)
        if image_collector is not None:
            image_collector.add_image("NoBleach", "control_metrics", img_array, basename)
        return img_array if return_array else None
    else:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{basename}_control_metrics.png"), dpi=100)
        plt.close(fig)
        return None




def plot_control_boundaries(stack, droplet_min_size, out_dir, basename, return_array=False, image_collector=None):
    """Save PNG showing droplet boundaries at a few key timepoints.

    We segment the droplet independently at t0, t_mid, and t_end using
    the same Otsu-based method as in `make_droplet_mask`, then overlay
    the contours on the raw frames.
    """
    T = stack.shape[0]
    if T == 0:
        return None

    indices = sorted({0, T // 2, T - 1})
    n_panels = len(indices)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
    axes = axes[0]

    for ax, idx in zip(axes, indices):
        frame = stack[idx]
        mask = make_droplet_mask(frame, min_size=droplet_min_size)
        ax.imshow(frame, cmap="gray")
        if np.any(mask):
            ax.contour(mask, levels=[0.5], colors="cyan", linewidths=1.5)
        ax.set_title(f"t = {idx}")
        ax.axis("off")

    fig.suptitle("Control droplet boundaries (cyan)")
    fig.tight_layout()
    
    if return_array or image_collector is not None:
        img_array = fig_to_array(fig)
        plt.close(fig)
        if image_collector is not None:
            image_collector.add_image("NoBleach", "control_boundaries", img_array, basename)
        return img_array if return_array else None
    else:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{basename}_control_boundaries.png"), dpi=100)
        plt.close(fig)
        return None


def plot_control_deltaK_deltaR(control_metrics_list, input_dir, dK_rel_all, dR_rel_all, mean_dK_dt, mean_dR_dt, return_array=False):
    """Plot deltaK_rel and deltaR_rel together with average rates.
    
    Parameters
    ----------
    control_metrics_list : list of (name, metrics_dict)
    input_dir : str
    dK_rel_all : list of deltaK_rel values
    dR_rel_all : list of deltaR_rel values
    mean_dK_dt : float, average dK_dt rate
    mean_dR_dt : float, average dR_dt rate
    return_array : bool
        If True, return image array instead of saving file.
    """
    if not control_metrics_list:
        return
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Collect deltaK and deltaR values
    deltaK_vals = []
    deltaR_vals = []
    names = []
    
    for name, metrics in control_metrics_list:
        t = np.asarray(metrics["time_s"], dtype=np.float64)
        K = np.asarray(metrics["K_partition"], dtype=np.float64)
        R = np.asarray(metrics["radius_um"], dtype=np.float64)
        
        mask_K = np.isfinite(t) & np.isfinite(K)
        mask_R = np.isfinite(t) & np.isfinite(R)
        
        if np.any(mask_K) and np.any(mask_R):
            tK = t[mask_K]
            KK = K[mask_K]
            tR = t[mask_R]
            RR = R[mask_R]
            
            if tK.size >= 2 and tR.size >= 2:
                K0 = float(KK[0])
                Kend = float(KK[-1])
                R0 = float(RR[0])
                Rend = float(RR[-1])
                
                if K0 != 0 and R0 != 0:
                    deltaK_rel = float((Kend - K0) / K0)
                    deltaR_rel = float((Rend - R0) / R0)
                    if np.isfinite(deltaK_rel) and np.isfinite(deltaR_rel):
                        deltaK_vals.append(deltaK_rel)
                        deltaR_vals.append(deltaR_rel)
                        names.append(name)
    
    if deltaK_vals and deltaR_vals:
        # Plot deltaK and deltaR
        ax.plot(deltaK_vals, deltaR_vals, 'o', alpha=0.6, markersize=6)
        
        # Add average rates as text annotation
        ax.text(0.05, 0.95, f'Mean dK/dt = {mean_dK_dt:.4g} 1/s\nMean dR/dt = {mean_dR_dt:.4g} µm/s',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('deltaK_rel (dimensionless)')
        ax.set_ylabel('deltaR_rel (dimensionless)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if return_array:
        img_array = fig_to_array(fig)
        plt.close(fig)
        return img_array
    else:
        out_path = os.path.join(input_dir, "NoBleach_K_r_curves.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return None


def plot_all_frap_curves_from_csv(csv_path, output_dir, frame_interval=10.0):
    """
    Plot all FRAP curves from all_frap_curves.csv starting from post-bleach frame (frame 1).
    Similar to NoBleach_r_curves.tif but for FRAP recovery curves.
    
    Parameters
    ----------
    csv_path : str
        Path to all_frap_curves.csv file (rows are frames, columns are curves)
    output_dir : str
        Directory to save the plot
    frame_interval : float
        Time interval between frames in seconds (default: 10.0)
    """
    if not os.path.exists(csv_path):
        return
    
    try:
        # Read CSV: rows are frames, columns are curve names
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return  # Need at least header and one data row
        
        # Parse header (first line)
        header = lines[0].strip().split(',')
        curve_names = [name.strip() for name in header if name.strip()]
        if not curve_names:
            return
        
        # Parse data rows (skip header, start from row 1 which is frame 1)
        # Row 0 (first data row) is frame 0 (pre-bleach), Row 1 is frame 1 (first post-bleach)
        if len(lines) < 3:
            return  # Need at least frame 0 and frame 1
        
        # Get all curves from row 1 onwards (post-bleach frames)
        curves_list = []
        labels_list = []
        
        for col_idx, col_name in enumerate(curve_names):
            # Extract values from row 1 onwards (post-bleach frames)
            curve_values = []
            for row_idx in range(1, len(lines)):  # Start from row 1 (frame 1)
                parts = lines[row_idx].strip().split(',')
                if col_idx >= len(parts):
                    break
                val_str = parts[col_idx].strip()
                if not val_str:
                    break
                try:
                    val = float(val_str)
                    if not np.isfinite(val):
                        break
                    curve_values.append(val)
                except (ValueError, TypeError):
                    break
            
            if len(curve_values) > 0:
                curves_list.append(np.array(curve_values, dtype=np.float32))
                labels_list.append(col_name)
        
        if not curves_list:
            return
        
        # Find minimum length for alignment
        min_len = min(len(c) for c in curves_list)
        if min_len == 0:
            return
        
        # Truncate all curves to minimum length
        curves_mat = np.stack([c[:min_len] for c in curves_list], axis=0)
        
        # Time axis: frame 1 is at frame_interval, frame 2 is at 2*frame_interval, etc.
        # Frame indices: 1, 2, 3, ... (post-bleach frames)
        frame_indices = np.arange(1, min_len + 1, dtype=int)
        time_axis = frame_indices.astype(np.float32) * frame_interval
        
        # Calculate mean curve
        mean_curve = np.nanmean(curves_mat, axis=0)
        
        # Plot
        plt.figure(figsize=(6, 4))
        
        # Plot individual curves
        for i, curve in enumerate(curves_mat):
            plt.plot(time_axis, curve, "o-", alpha=0.4, markersize=3)
        
        # Plot mean curve
        plt.plot(time_axis, mean_curve, "k-", linewidth=2.0, label="Mean")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized intensity")
        plt.title("FRAP recovery curves (post-bleach)")
        plt.legend()
        plt.tight_layout()
        
        # Save as TIF
        output_path = os.path.join(output_dir, "FRAP_curves.tif")
        img_array = fig_to_array(plt.gcf())
        plt.close()
        
        # Save as single-image TIF stack
        tiff.imwrite(output_path, img_array[None, :, :, :], imagej=True)  # Add dimension for stack
        logger.info(f"Saved FRAP curves plot: {output_path}")
    except Exception as e:
        logger.warning(f"Error generating FRAP curves plot: {e}")


def plot_all_control_r_curves(control_metrics_list, input_dir, return_array=False):
    """Overlay radius(t) from all control droplets and plot the mean curve.
    Filters outliers based on deviation from the mean curve.

    Parameters
    ----------
    control_metrics_list : list of (name, metrics_dict)
        Each metrics_dict must contain 'time_s' and 'radius_um'.
    input_dir : str
        Directory to save the figure (NoBleach_r_curves.png).
    return_array : bool
        If True, return image array instead of saving file.
    """
    if not control_metrics_list:
        return

    # First pass: collect all curves and compute mean
    R_list = []
    t_common = None
    names_list = []

    for name, metrics in control_metrics_list:
        t = np.asarray(metrics["time_s"], dtype=np.float32)
        R = np.asarray(metrics["radius_um"], dtype=np.float32)

        if t_common is None:
            t_common = t.copy()
        else:
            # Use the minimum overlapping length
            L = min(len(t_common), len(t))
            t_common = t_common[:L]
            R = R[:L]
        R_list.append(R)
        names_list.append(name)

    if not R_list or t_common is None:
        return

    # Truncate all R arrays to min length for averaging
    min_len = min(len(r) for r in R_list)
    if min_len == 0:
        return

    R_mat = np.stack([r[:min_len] for r in R_list], axis=0)
    t_mean = t_common[:min_len]
    R_mean = np.nanmean(R_mat, axis=0)

    # Filter outliers: compute deviation from mean for each curve
    deviations = []
    for r in R_mat:
        # Compute RMS deviation from mean
        valid = np.isfinite(r) & np.isfinite(R_mean)
        if np.any(valid):
            rms_dev = np.sqrt(np.nanmean((r[valid] - R_mean[valid])**2))
            deviations.append(rms_dev)
        else:
            deviations.append(np.inf)

    deviations = np.array(deviations)
    # Use IQR-based outlier detection
    q25, q75 = np.nanpercentile(deviations, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        threshold = q75 + 1.5 * iqr
    else:
        threshold = np.nanmax(deviations) if np.any(np.isfinite(deviations)) else np.inf

    # Plot filtered curves (no legend)
    plt.figure(figsize=(6, 4))
    
    for i, (name, r) in enumerate(zip(names_list, R_mat)):
        if deviations[i] <= threshold:
            plt.plot(t_mean, r, "o-", alpha=0.4, markersize=3)

    # Plot mean curve
    plt.plot(
        t_mean,
        R_mean,
        "k-",
        linewidth=2.0,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Effective radius (µm)")
    plt.title("NoBleach control droplets: R(t)")
    plt.tight_layout()
    
    if return_array:
        fig = plt.gcf()
        img_array = fig_to_array(fig)
        plt.close()
        return img_array
    else:
        out_path = os.path.join(input_dir, "NoBleach_r_curves.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return None


def plot_all_control_K_curves(control_metrics_list, input_dir, return_array=False):
    """Overlay K(t) from all control droplets and plot the mean curve.
    Filters outliers based on deviation from the mean curve.

    Parameters
    ----------
    control_metrics_list : list of (name, metrics_dict)
        Each metrics_dict must contain 'time_s' and 'K_partition'.
    input_dir : str
        Directory to save the figure (NoBleach_K_curves.png).
    return_array : bool
        If True, return image array instead of saving file.
    """
    if not control_metrics_list:
        return

    # First pass: collect all curves and compute mean
    K_list = []
    t_common = None
    names_list = []

    for name, metrics in control_metrics_list:
        t = np.asarray(metrics["time_s"], dtype=np.float32)
        K = np.asarray(metrics["K_partition"], dtype=np.float32)

        if t_common is None:
            t_common = t.copy()
        else:
            # Use the minimum overlapping length
            L = min(len(t_common), len(t))
            t_common = t_common[:L]
            K = K[:L]
        K_list.append(K)
        names_list.append(name)

    if not K_list or t_common is None:
        return

    # Truncate all K arrays to min length for averaging
    min_len = min(len(k) for k in K_list)
    if min_len == 0:
        return

    K_mat = np.stack([k[:min_len] for k in K_list], axis=0)
    t_mean = t_common[:min_len]
    K_mean = np.nanmean(K_mat, axis=0)

    # Filter outliers: compute deviation from mean for each curve
    deviations = []
    for k in K_mat:
        # Compute RMS deviation from mean
        valid = np.isfinite(k) & np.isfinite(K_mean)
        if np.any(valid):
            rms_dev = np.sqrt(np.nanmean((k[valid] - K_mean[valid])**2))
            deviations.append(rms_dev)
        else:
            deviations.append(np.inf)

    deviations = np.array(deviations)
    # Use IQR-based outlier detection
    q25, q75 = np.nanpercentile(deviations, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        threshold = q75 + 1.5 * iqr
    else:
        threshold = np.nanmax(deviations) if np.any(np.isfinite(deviations)) else np.inf

    # Plot filtered curves (no legend)
    plt.figure(figsize=(6, 4))
    
    for i, (name, k) in enumerate(zip(names_list, K_mat)):
        if deviations[i] <= threshold:
            plt.plot(t_mean, k, "o-", alpha=0.4, markersize=3)

    # Plot mean curve
    plt.plot(
        t_mean,
        K_mean,
        "k-",
        linewidth=2.0,
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Partition coeff. K (dense/dilute)")
    plt.title("NoBleach control droplets: K(t)")
    plt.tight_layout()
    
    if return_array:
        fig = plt.gcf()
        img_array = fig_to_array(fig)
        plt.close()
        return img_array
    else:
        out_path = os.path.join(input_dir, "NoBleach_K_curves.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        return None

def process_control_movie(
    path,
    input_dir,
    pixel_size,
    frame_interval,
    droplet_min_size,
    summary_dir=None,
    image_collector=None,
):
    """Process a NoBleach control movie.

    This does *not* attempt any FRAP/PDE fitting. Instead it:
    - quantifies droplet growth/exchange metrics (see analyze_control_movie),
    - saves a per-frame CSV and summary plots,
    - returns a photobleach reference curve based on droplet intensity.
    """
    basename = os.path.splitext(os.path.basename(path))[0]
    if summary_dir is None:
        summary_dir = input_dir

    logger.info(f"Processing control movie (NoBleach): {basename}")

    stack = load_movie(path)
    T, H, W = stack.shape
    if T < 2:
        logger.error("Control movie has fewer than 2 frames; skipping.")
        return {"filename": basename, "success": False}, None

    # Use droplet mask on first frame for background subtraction / normalization
    droplet_mask_full = make_droplet_mask(stack[0], min_size=droplet_min_size)
    norm_full, bg, pre_mean = normalize_stack(stack, droplet_mask_full)

    metrics, pb_curve = analyze_control_movie(
        stack,
        norm_full,
        pixel_size,
        frame_interval,
        droplet_min_size,
    )

    # Save metrics as CSV
    csv_path = os.path.join(summary_dir, f"{basename}_control_metrics.csv")
    with open(csv_path, "w") as f:
        f.write(
            "time_s,I_droplet,I_background,K_partition,area_um2,radius_um,background_raw,pre_mean_raw\n"
        )
        for (t, Id, Ib, Kp, A, R) in zip(
            metrics["time_s"],
            metrics["I_droplet"],
            metrics["I_background"],
            metrics["K_partition"],
            metrics["area_um2"],
            metrics["radius_um"],
        ):
            f.write(
                f"{t:.6g},{Id:.6g},{Ib:.6g},{Kp:.6g},{A:.6g},{R:.6g},{bg:.6g},{pre_mean:.6g}\n"
            )

    # Plots - add to image collector for stacking
    plot_control_metrics(basename, metrics, summary_dir, return_array=(image_collector is not None), image_collector=image_collector)
    plot_control_boundaries(stack, droplet_min_size, summary_dir, basename, return_array=(image_collector is not None), image_collector=image_collector)

    summary = {
        "filename": basename,
        "success": True,
        "background": bg,
        "pre_mean": pre_mean,
    }
    return summary, pb_curve, metrics

# ------------------------------------------------------------
# Per-movie processing
# ------------------------------------------------------------

def process_movie(
    path,
    input_dir,
    pixel_size,
    frame_interval,
    D_min,
    D_max,
    droplet_min_size,
    ratio_percentile,
    max_frames,
    downsample_factor,
    D_grid_size,
    D_refine_grid_size,
    n_jobs_D,
    crop_margin,
    fit_region,
    noise_points_per_frame,
    photobleach_correct,
    exp_plateau_fit,
    pb_exclude_radius,
    pb_ratio_tol,
    pde_first_frame,
    pde_last_frame,
    exp_first_frame,
    exp_last_frame,
    combined_model,
    T_switch_frame_arg,
    control_pb_curve=None,
    image_collector=None,
    summary_dir=None,
):
    basename = os.path.splitext(os.path.basename(path))[0]
    if summary_dir is None:
        summary_dir = input_dir

    logger.info(f"Processing movie: {basename}")
    summary_sep = {
        "filename": basename,
        "D_um2_per_s": np.nan,
        "success": False,
        "cost": np.nan,
        "n_frames_fit": 0,
        "frame_interval_s": frame_interval,
        "pixel_size_um": pixel_size,
        "background": np.nan,
        "pre_mean": np.nan,
        "k_single_exp": np.nan,
        "mobile_fraction": np.nan,
        "immobile_fraction": np.nan,
        "YM_single_exp": np.nan,
        "Y0_single_exp": np.nan,
        "K_pre": np.nan,
        "bleach_fraction": np.nan,
    }
    summary_comb = {
        "filename": basename,
        "D_um2_per_s": np.nan,
        "success": False,
        "cost": np.nan,
        "n_frames_fit": 0,
        "frame_interval_s": frame_interval,
        "pixel_size_um": pixel_size,
        "background": np.nan,
        "pre_mean": np.nan,
        "k_single_exp": np.nan,
        "mobile_fraction": np.nan,
        "immobile_fraction": np.nan,
        "YM_single_exp": np.nan,
        "Y0_single_exp": np.nan,
        "K_pre": np.nan,
        "bleach_fraction": np.nan,
        "T_switch_frame": -1,
        "droplet_radius_um": np.nan,
    }

    try:
        stack = load_movie(path)
        T, H, W = stack.shape
        if T < 2:
            logger.error("Movie has fewer than 2 frames; cannot perform FRAP.")
            return summary_sep, None

        pre = stack[0]
        post = stack[1]

        droplet_mask_full = make_droplet_mask(pre, min_size=droplet_min_size)
        bleach_mask_full = make_bleach_mask(pre, post, droplet_mask_full, ratio_percentile)

        # Bleach area fraction in pre-bleach frame (full resolution)
        droplet_area_px = float(droplet_mask_full.sum())
        bleach_area_px = float(bleach_mask_full.sum())
        if droplet_area_px > 0 and bleach_area_px > 0:
            bleach_fraction = bleach_area_px / droplet_area_px
        else:
            bleach_fraction = float("nan")
        summary_sep["bleach_fraction"] = bleach_fraction
        summary_comb["bleach_fraction"] = bleach_fraction
        
        # Calculate droplet radius (effective radius from area)
        droplet_area_um2 = droplet_area_px * (pixel_size ** 2)
        droplet_radius_um = np.sqrt(droplet_area_um2 / (np.pi + 1e-12))
        summary_sep["droplet_radius_um"] = droplet_radius_um
        summary_comb["droplet_radius_um"] = droplet_radius_um

        norm_full, bg, pre_mean = normalize_stack(stack, droplet_mask_full)
        summary_sep["background"] = bg
        summary_sep["pre_mean"] = pre_mean
        summary_comb["background"] = bg
        summary_comb["pre_mean"] = pre_mean

        # Partition coefficient from droplet minus bleach ROI (full resolution)
        dense_nb_mask_full = droplet_mask_full & ~bleach_mask_full
        if not np.any(dense_nb_mask_full):
            logger.warning("No non-bleached dense-phase pixels; using full droplet for K_FRAP.")
            dense_nb_mask_full = droplet_mask_full.copy()
        bg_region_full = ~droplet_mask_full

        T_total = norm_full.shape[0]
        I_dense_nb = np.full(T_total, np.nan, dtype=np.float32)
        I_bg_full = np.full(T_total, np.nan, dtype=np.float32)
        K_FRAP = np.full(T_total, np.nan, dtype=np.float32)

        for i in range(T_total):
            frame_norm = norm_full[i]
            vals_dense = frame_norm[dense_nb_mask_full]
            vals_bg = frame_norm[bg_region_full]
            if vals_dense.size > 0:
                I_dense_nb[i] = float(vals_dense.mean())
            if vals_bg.size > 0:
                I_bg_full[i] = float(vals_bg.mean())
                if I_bg_full[i] > 0:
                    K_FRAP[i] = I_dense_nb[i] / I_bg_full[i]

        K_pre = float(K_FRAP[0]) if np.isfinite(K_FRAP[0]) else float("nan")
        summary_sep["K_pre"] = K_pre
        summary_comb["K_pre"] = K_pre

        # Masks plot – show post-bleach frame
        mask_img = plot_masks(
            stack[1],
            droplet_mask_full,
            bleach_mask_full,
            return_array=(image_collector is not None),
        )
        if image_collector is not None and mask_img is not None:
            image_collector.add_image("FRAP", "droplet_masks", mask_img, basename)

        # Crop and downsample
        norm_crop, droplet_crop, bleach_crop, bbox = crop_to_droplet(
            norm_full, droplet_mask_full, bleach_mask_full, margin=crop_margin
        )
        norm_crop_ds, droplet_ds, bleach_ds = downsample_stack_and_masks(
            norm_crop,
            droplet_crop,
            bleach_crop,
            factor=downsample_factor,
        )
        pixel_size_eff = pixel_size * downsample_factor

        # Photobleach region based on pre-correction stack
        pb_region = build_photobleach_region(
            norm_crop_ds,
            droplet_ds,
            bleach_ds,
            pb_exclude_radius=pb_exclude_radius,
            pb_ratio_tol=pb_ratio_tol,
        )

        # Photobleach correction
        if photobleach_correct:
            if control_pb_curve is not None:
                logger.info("Using external NoBleach control curve for photobleach correction.")
                norm_crop_ds, pb_curve, pb_factors = apply_photobleach_correction_from_curve(
                    norm_crop_ds, control_pb_curve
                )
            else:
                norm_crop_ds, pb_curve, pb_factors = apply_photobleach_correction(
                    norm_crop_ds, pb_region
                )
        else:
            logger.info("Photobleach correction disabled by user.")
            pb_curve = None
            pb_factors = np.ones(norm_crop_ds.shape[0], dtype=np.float32)

        # Noise (after PB correction) – use pb_region as noise region
        sigma_global, sigma_per_frame, _ = estimate_noise(
            norm_crop_ds, droplet_ds, bleach_ds, region=pb_region
        )

        # Fit region
        if fit_region == "bleach":
            fit_mask = bleach_ds.copy()
            logger.info("Using BLEACH ROI as fit region.")
        else:
            fit_mask = droplet_ds.copy()
            logger.info("Using DROPLET as fit region.")

        # FRAP curve, then renormalize so frame 0 in fit ROI = 1
        frap_curve = compute_frap_curve(norm_crop_ds, fit_mask)
        pre_fit = frap_curve[0]
        if pre_fit > 0:
            scale = 1.0 / pre_fit
            logger.info(f"Rescaling intensities so pre-bleach fit ROI = 1 (scale={scale:.3f}).")
            frap_curve *= scale
            sigma_per_frame *= scale
            sigma_global *= scale
            norm_crop_ds *= scale
        else:
            logger.warning("Pre-bleach fit ROI mean <= 0; skipping ROI-based rescale.")

        # Save per-frame FRAP + partition coefficient timeseries for this movie
        times_full = np.arange(T, dtype=np.float32) * frame_interval
        ts_csv = os.path.join(summary_dir, f"{basename}_frap_timeseries.csv")
        with open(ts_csv, "w") as f_ts:
            f_ts.write(
                "frame_index,time_s,FRAP_fit_ROI,K_dense_nonbleach,I_dense_nonbleach,I_background\n"
            )
            for i in range(T):
                frap_val = frap_curve[i] if i < frap_curve.size else float("nan")
                K_val = K_FRAP[i] if i < K_FRAP.size else float("nan")
                Id_val = I_dense_nb[i] if i < I_dense_nb.size else float("nan")
                Ib_val = I_bg_full[i] if i < I_bg_full.size else float("nan")
                f_ts.write(
                    f"{i},{times_full[i]:.6g},{frap_val:.6g},{K_val:.6g},{Id_val:.6g},{Ib_val:.6g}\n"
                )


        # ---------- PDE WINDOW ----------
        pde_first = max(1, pde_first_frame)
        pde_last_eff = min(T - 1, max_frames, pde_last_frame)
        if pde_last_eff < pde_first:
            logger.error("Invalid PDE frame window; skipping movie.")
            return summary_sep, None

        frame_ids_pde = np.arange(pde_first, pde_last_eff + 1, dtype=int)

        # Weights for PDE window
        I = frap_curve
        dI = np.abs(np.diff(I))
        dynamic_full = np.concatenate([dI, dI[-1:]])
        dynamic = dynamic_full[frame_ids_pde]
        sigma_used = sigma_per_frame[frame_ids_pde]

        eps = 1e-6
        dyn_norm = dynamic / (np.nanmax(dynamic) + eps)
        noise_norm = sigma_used / (np.nanmax(sigma_used) + eps)
        weights_pde = dyn_norm / (noise_norm + 1e-3)
        weights_pde = weights_pde / (np.nanmax(weights_pde) + eps)
        weights_pde = np.maximum(weights_pde, 0.05).astype(np.float32)

        logger.info(
            f"PDE fit frames {pde_first}-{pde_last_eff} "
            f"({len(frame_ids_pde)} post-bleach frames); "
            f"weight range=({weights_pde.min():.3g},{weights_pde.max():.3g})"
        )

        data_stack = np.ascontiguousarray(
            norm_crop_ds[frame_ids_pde].astype(np.float32)
        )
        domain_mask = np.ascontiguousarray(droplet_ds.astype(bool))
        fit_mask_bool = np.ascontiguousarray(fit_mask.astype(bool))

        # PDE times for fitting: relative to first PDE frame (t_rel=0 there)
        times_pde_rel = np.ascontiguousarray(
            (frame_ids_pde - frame_ids_pde[0]).astype(np.float32) * frame_interval
        )

        dx = float(pixel_size_eff)
        dt_stable = 0.24 * dx * dx / max(D_max, 1e-6)
        dt_sim = dt_stable
        logger.info(
            f"Using dt_sim={dt_sim:.4g}s for PDE (stable for D_max={D_max}, dx={dx})"
        )

        # Initial condition: post-bleach frame 1, masked
        c0 = np.ascontiguousarray(
            (norm_crop_ds[1] * domain_mask).astype(np.float32)
        )
        weights_pde = np.ascontiguousarray(weights_pde)

        # Fit D via grid search
        D_best, cost_best = grid_search_D(
            D_min,
            D_max,
            D_grid_size,
            D_refine_grid_size,
            c0,
            data_stack,
            domain_mask,
            fit_mask_bool,
            times_pde_rel,
            weights_pde,
            dt_sim,
            dx,
            n_jobs_D=n_jobs_D,
        )

        summary_sep["D_um2_per_s"] = D_best
        summary_sep["success"] = True
        summary_sep["cost"] = cost_best
        summary_sep["n_frames_fit"] = int(len(frame_ids_pde))

        summary_comb["D_um2_per_s"] = D_best
        summary_comb["success"] = True
        summary_comb["cost"] = cost_best
        summary_comb["n_frames_fit"] = int(len(frame_ids_pde))

        # ---------- EXPONENTIAL WINDOW (SEPARATE MODEL) ----------
        exp_fit = None
        if exp_plateau_fit:
            # Use all post-bleach frames (frames 1-31, excluding frame 0)
            exp_first = 1  # First post-bleach frame
            exp_last_eff = min(T - 1, max_frames)  # Up to max_frames or end of movie
            if exp_last_eff >= exp_first:
                frame_ids_exp = np.arange(exp_first, exp_last_eff + 1, dtype=int)
                # Time axis: frame 1 is at t=0, frame 2 is at frame_interval, etc.
                t_exp = (frame_ids_exp - exp_first).astype(np.float32) * frame_interval
                y_exp = frap_curve[frame_ids_exp]
                # Constrain Y0 to the first post-bleach frame (frame 1)
                Y0_constraint = float(frap_curve[exp_first])
                logger.info(
                    f"Exp plateau fit frames {exp_first}-{exp_last_eff} "
                    f"({len(frame_ids_exp)} post-bleach frames), Y0 constrained to frame {exp_first}"
                )
                exp_fit = fit_single_exp_plateau(t_exp, y_exp, constrain_Y0=Y0_constraint)
                if exp_fit is not None and exp_fit.get("success", False):
                    summary_sep["k_single_exp"] = exp_fit["k"]
                    summary_sep["mobile_fraction"] = exp_fit["mobile_fraction"]
                    summary_sep["immobile_fraction"] = exp_fit["immobile_fraction"]
                    summary_sep["YM_single_exp"] = exp_fit["YM"]
                    summary_sep["Y0_single_exp"] = exp_fit["Y0"]
            else:
                logger.warning("Invalid exponential frame window; skipping exp fit.")

        # ---------- PLOTS: SEPARATE MODEL ----------
        # Get frame_ids_exp for exponential plot time axis
        frame_ids_exp_for_plot = None
        if exp_fit is not None and exp_fit.get("success", False):
            exp_first = 1
            exp_last_eff = min(T - 1, max_frames)
            if exp_last_eff >= exp_first:
                frame_ids_exp_for_plot = np.arange(exp_first, exp_last_eff + 1, dtype=int)
        
        frap_sep_img = plot_frap_separate(
            frap_curve,
            D_best,
            domain_mask,
            fit_mask_bool,
            norm_crop_ds,
            frame_interval,
            pixel_size_eff,
            D_min,
            D_max,
            return_array=(image_collector is not None),
            exp_fit=exp_fit,
            frame_ids_pde=frame_ids_pde,
            frame_ids_exp=frame_ids_exp_for_plot,
            droplet_label=basename,
        )
        if image_collector is not None and frap_sep_img is not None:
            image_collector.add_image("FRAP", "frap_fit_separate", frap_sep_img, basename)

        # Note: weights_pde and frame_ids_pde are defined in the PDE window section above
        noise_img = plot_noise_pixels(
            norm_crop_ds,
            pb_region,
            sigma_per_frame,
            return_array=(image_collector is not None),
            max_points_per_frame=noise_points_per_frame,
            weights_pde=weights_pde,
            frame_ids_pde=frame_ids_pde,
            droplet_label=basename,
        )
        if image_collector is not None and noise_img is not None:
            image_collector.add_image("FRAP", "noise_pixels", noise_img, basename)

        # Note: Per-movie CSV summaries are now written to a combined file at the end

        # ---------- COMBINED MODEL (PDE + CONSTRAINED EXPONENTIAL) ----------
        combined_curve_best = None
        T_switch_best = None
        if combined_model and exp_plateau_fit and summary_sep["success"]:
            exp_last_eff = min(T - 1, max_frames, exp_last_frame)
            if exp_last_eff > pde_first:
                # PDE mean curve over full PDE window (for all candidates)
                times_pde_all, pde_curve_all = compute_pde_mean_curve(
                    D_best,
                    c0,
                    domain_mask,
                    fit_mask_bool,
                    frame_ids_pde,
                    frame_interval,
                    dt_sim,
                    dx,
                )

                # Candidate T_switch frames:

                if T_switch_frame_arg is not None:
                    # user-specified; clamp into valid range
                    T_switch_candidates = np.array(
                        [max(frame_ids_pde[0] + 1,
                             min(frame_ids_pde[-1], T_switch_frame_arg))]
                    )
                else:
                    # automatic: use PDE window only, but require we still have
                    # at least a couple of frames after T_switch for the tail fit
                    lower = frame_ids_pde[0] + 1
                    upper = min(frame_ids_pde[-1], exp_last_eff - 2)

                    if upper < lower:
                        # fallback: just use the last PDE frame as switch
                        T_switch_candidates = np.array([frame_ids_pde[-1]])
                    else:
                        T_switch_candidates = np.arange(lower, upper + 1, dtype=int)

                logger.info(f"Combined model: candidate T_switch frames = {T_switch_candidates}")

                best_cost_comb = None
                best_tail_fit = None

                for T_sw in T_switch_candidates:
                    # PDE intensity at switch frame
                    if T_sw in frame_ids_pde:
                        idx_sw = np.where(frame_ids_pde == T_sw)[0][0]
                        I_switch = float(pde_curve_all[idx_sw])
                    else:
                        continue

                    tail_fit = fit_constrained_exp_tail(
                        frap_curve,
                        frame_interval,
                        T_sw,
                        I_switch,
                        last_frame_index=exp_last_eff,
                    )
                    if not tail_fit.get("success", False):
                        continue

                    cost_comb, combined_curve_candidate = build_combined_curve_and_cost(
                        frap_curve,
                        frame_interval,
                        frame_ids_pde,
                        pde_curve_all,
                        T_sw,
                        tail_fit,
                        exp_last_eff,
                    )

                    logger.debug(
                        f"T_switch={T_sw}: combined cost={cost_comb:.4g}, "
                        f"k={tail_fit['k']:.3g}, YM={tail_fit['YM']:.3f}"
                    )

                    if (best_cost_comb is None) or (cost_comb < best_cost_comb):
                        best_cost_comb = cost_comb
                        combined_curve_best = combined_curve_candidate
                        best_tail_fit = tail_fit
                        T_switch_best = int(T_sw)

                if best_tail_fit is not None:
                    summary_comb["k_single_exp"] = best_tail_fit["k"]
                    summary_comb["mobile_fraction"] = best_tail_fit["mobile_fraction"]
                    summary_comb["immobile_fraction"] = best_tail_fit["immobile_fraction"]
                    summary_comb["YM_single_exp"] = best_tail_fit["YM"]
                    summary_comb["Y0_single_exp"] = best_tail_fit["Y0"]
                    summary_comb["cost"] = best_cost_comb

                    # Plot combined model
                    frap_comb_img = plot_frap_combined(
                        frap_curve,
                        frame_interval,
                        D_best,
                        c0,
                        domain_mask,
                        fit_mask_bool,
                        pixel_size_eff,
                        D_max,
                        frame_ids_pde,
                        combined_curve_best,
                        T_switch_best,
                        return_array=(image_collector is not None),
                        droplet_label=basename,
                    )
                    if image_collector is not None and frap_comb_img is not None:
                        image_collector.add_image("FRAP", "frap_fit_combined", frap_comb_img, basename)

                    # Store T_switch in summary_comb for combined CSV writing
                    summary_comb["T_switch_frame"] = T_switch_best if T_switch_best is not None else -1
                    # Note: Per-movie CSV summaries are now written to a combined file at the end
                else:
                    logger.warning("Combined constrained exponential fit failed for all candidates.")
            else:
                logger.warning("Combined model: invalid exp window; skipping.")
        else:
            logger.info("Combined model disabled or exp fit disabled; skipping.")

        logger.info(f"Finished movie: {basename} (D={D_best:.4g} µm²/s)")

    except Exception as e:
        logger.error(f"Error processing movie {basename}: {e}", exc_info=True)

    return summary_sep, (summary_comb if combined_model else None)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def process_single_condition(
    condition_dir,
    pixel_size,
    frame_interval,
    D_min,
    D_max,
    droplet_min_size,
    ratio_percentile,
    max_frames,
    downsample_factor,
    D_grid_size,
    D_refine_grid_size,
    n_jobs_D,
    crop_margin,
    fit_region,
    noise_points_per_frame,
    photobleach_correct,
    exp_plateau_fit,
    pb_exclude_radius,
    pb_ratio_tol,
    pde_first_frame,
    pde_last_frame,
    exp_first_frame,
    exp_last_frame,
    combined_model,
    T_switch_frame_arg,
    results_root,
    verbose_level,
):
    """
    Process a single movie folder (subfolder containing TIF files from one movie).
    This function is designed to be called in parallel for multiple movies.
    
    Note: "condition_dir" parameter name kept for backward compatibility, but
    represents a single movie's folder containing its cropped droplet TIF files.
    """
    # Set up logging for this movie (use global logger)
    setup_logging(verbose_level)
    
    movie_dir = condition_dir  # Renamed internally for clarity
    logger.info(f"Processing movie: {movie_dir}")
    
    # Find TIF files in this movie folder
    tif_paths = sorted(
        glob.glob(os.path.join(movie_dir, "*.tif"))
        + glob.glob(os.path.join(movie_dir, "*.tiff"))
    )
    
    if not tif_paths:
        logger.warning(f"No TIF/TIFF files found in {movie_dir}")
        return movie_dir, False

    logger.info(f"Found {len(tif_paths)} TIF files in {movie_dir}")

    # Split into NoBleach control movies and FRAP movies
    control_paths = []
    frap_paths = []
    for path in tif_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        if is_control_movie(basename):
            control_paths.append(path)
        else:
            frap_paths.append(path)

    if control_paths:
        logger.info(f"Identified {len(control_paths)} control (NoBleach) movies.")
    if frap_paths:
        logger.info(f"Identified {len(frap_paths)} FRAP movies.")

    # Initialize image collector for this movie
    image_collector = ImageCollector()
    
    # Handle both old and new folder structures
    # New: movie_dir is [results_root]/[movie]_results/droplet_stacks/
    # Old: movie_dir is [input_dir]/[movie]/
    if os.path.basename(movie_dir) == "droplet_stacks":
        # New structure: results_dir is the parent of droplet_stacks
        results_dir = os.path.dirname(movie_dir)
        movie_name = os.path.basename(results_dir).replace("_results", "")
    else:
        # Old structure: create results_dir in results_root
        movie_name = os.path.basename(movie_dir)
        results_dir = os.path.join(results_root, f"{movie_name}_results")
    
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # First pass: analyze control movies to obtain global photobleach curve
    control_summaries = []
    control_pb_curves = []
    control_metrics_list = []

    for path in control_paths:
        summary_ctrl, pb_curve, metrics = process_control_movie(
            path,
            movie_dir,
            pixel_size,
            frame_interval,
            droplet_min_size,
            summary_dir=summary_dir,
            image_collector=image_collector,
        )
        control_summaries.append(summary_ctrl)
        control_metrics_list.append((summary_ctrl["filename"], metrics))
        if pb_curve is not None:
            control_pb_curves.append(pb_curve)

    global_pb_curve = None
    if control_pb_curves:
        # Align curves by minimum length and average
        min_len = min(len(c) for c in control_pb_curves)
        pb_mat = np.stack([c[:min_len] for c in control_pb_curves], axis=0)
        global_pb_curve = np.nanmean(pb_mat, axis=0).astype(np.float32)

        # Save global photobleach curve to CSV for inspection
        pb_csv = os.path.join(results_dir, "NoBleach_photobleach_curve.csv")
        logger.info(f"Saving averaged NoBleach photobleach curve to {pb_csv}")
        with open(pb_csv, "w") as f:
            f.write("frame_index,time_s,mean_droplet_intensity_normalized\n")
            for i, val in enumerate(global_pb_curve):
                f.write(f"{i},{i * frame_interval:.6g},{val:.6g}\n")

    # NoBleach per-droplet summary metrics (dK/dt, dR/dt, etc.)
    if control_metrics_list:
        summary_csv = os.path.join(results_dir, "NoBleach_summary.csv")
        logger.info(f"Saving NoBleach per-droplet summary to {summary_csv}")

        # Precompute per-droplet stats for filtering
        droplet_stats = []
        for name, metrics in control_metrics_list:
            t = np.asarray(metrics["time_s"], dtype=np.float64)
            K = np.asarray(metrics["K_partition"], dtype=np.float64)
            R = np.asarray(metrics["radius_um"], dtype=np.float64)

            mask_K = np.isfinite(t) & np.isfinite(K)
            mask_R = np.isfinite(t) & np.isfinite(R)

            K0 = Kend = deltaK_rel = dK_dt = np.nan
            R0 = Rend = deltaR_rel = dR_dt = np.nan
            mean_K = np.nan

            if np.any(mask_K):
                tK = t[mask_K]
                KK = K[mask_K]
                K0 = float(KK[0])
                Kend = float(KK[-1])
                mean_K = float(np.nanmean(KK))
                deltaK_rel = float((Kend - K0) / K0) if K0 != 0 else np.nan
                if tK.size >= 2:
                    pK = np.polyfit(tK, KK, 1)
                    dK_dt = float(pK[0])
            if np.any(mask_R):
                tR = t[mask_R]
                RR = R[mask_R]
                R0 = float(RR[0])
                Rend = float(RR[-1])
                deltaR_rel = float((Rend - R0) / R0) if R0 != 0 else np.nan
                if tR.size >= 2:
                    pR = np.polyfit(tR, RR, 1)
                    dR_dt = float(pR[0])

            droplet_stats.append({
                "name": name,
                "metrics": metrics,
                "mean_K": mean_K,
                "deltaK_rel": deltaK_rel,
                "deltaR_rel": deltaR_rel,
                "K0": K0, "Kend": Kend, "dK_dt": dK_dt,
                "R0": R0, "Rend": Rend, "dR_dt": dR_dt,
            })

        # Compute SD filtering across mean_K, deltaK_rel, deltaR_rel
        def sd_bounds(values):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return -np.inf, np.inf
            m = np.nanmean(arr)
            s = np.nanstd(arr)
            return m - 2 * s, m + 2 * s

        mean_K_lower, mean_K_upper = sd_bounds([d["mean_K"] for d in droplet_stats])
        dK_lower, dK_upper = sd_bounds([d["deltaK_rel"] for d in droplet_stats])
        dR_lower, dR_upper = sd_bounds([d["deltaR_rel"] for d in droplet_stats])

        filtered_stats = []
        filtered_names = []
        for d in droplet_stats:
            keep = True
            if np.isfinite(d["mean_K"]) and not (mean_K_lower <= d["mean_K"] <= mean_K_upper):
                keep = False
            if np.isfinite(d["deltaK_rel"]) and not (dK_lower <= d["deltaK_rel"] <= dK_upper):
                keep = False
            if np.isfinite(d["deltaR_rel"]) and not (dR_lower <= d["deltaR_rel"] <= dR_upper):
                keep = False
            if keep:
                filtered_stats.append(d)
            else:
                filtered_names.append(d["name"])

        if filtered_names:
            logger.info(f"Filtered NoBleach droplets (outside ±2 SD in K/ΔK/ΔR): {filtered_names}")

        control_metrics_list = [(d["name"], d["metrics"]) for d in filtered_stats]

        dK_rel_all = []
        dR_rel_all = []

        with open(summary_csv, "w") as f:
            f.write("filename,K0,K_end,deltaK_rel (dimensionless),dK_dt (1/s),R0 (um),R_end (um),deltaR_rel (dimensionless),dR_dt (um/s)\n")
            for d in filtered_stats:
                name = d["name"]
                K0 = d["K0"]; Kend = d["Kend"]; deltaK_rel = d["deltaK_rel"]; dK_dt = d["dK_dt"]
                R0 = d["R0"]; Rend = d["Rend"]; deltaR_rel = d["deltaR_rel"]; dR_dt = d["dR_dt"]

                if np.isfinite(deltaK_rel):
                    dK_rel_all.append(deltaK_rel)
                if np.isfinite(deltaR_rel):
                    dR_rel_all.append(deltaR_rel)

                f.write(
                    f"{name},{K0:.6g},{Kend:.6g},{deltaK_rel:.6g},"
                    f"{dK_dt:.6g},{R0:.6g},{Rend:.6g},{deltaR_rel:.6g},{dR_dt:.6g}\n"
                )

        # Compute average rates (dK_dt and dR_dt)
        dK_dt_all = []
        dR_dt_all = []
        for name, metrics in control_metrics_list:
            t = np.asarray(metrics["time_s"], dtype=np.float64)
            K = np.asarray(metrics["K_partition"], dtype=np.float64)
            R = np.asarray(metrics["radius_um"], dtype=np.float64)
            
            mask_K = np.isfinite(t) & np.isfinite(K)
            mask_R = np.isfinite(t) & np.isfinite(R)
            
            if np.any(mask_K) and t[mask_K].size >= 2:
                tK = t[mask_K]
                KK = K[mask_K]
                pK = np.polyfit(tK, KK, 1)
                if np.isfinite(pK[0]):
                    dK_dt_all.append(pK[0])
            
            if np.any(mask_R) and t[mask_R].size >= 2:
                tR = t[mask_R]
                RR = R[mask_R]
                pR = np.polyfit(tR, RR, 1)
                if np.isfinite(pR[0]):
                    dR_dt_all.append(pR[0])
        
        mean_dK_dt = float(np.nanmean(dK_dt_all)) if dK_dt_all else float("nan")
        mean_dR_dt = float(np.nanmean(dR_dt_all)) if dR_dt_all else float("nan")

        # Simple movie-level classification (very conservative)
        cond_summary_path = os.path.join(results_dir, "NoBleach_summary.txt")
        mean_dK_rel = float(np.nanmean(dK_rel_all)) if dK_rel_all else float("nan")
        mean_dR_rel = float(np.nanmean(dR_rel_all)) if dR_rel_all else float("nan")

        status = "uncertain"
        if np.isfinite(mean_dK_rel) and np.isfinite(mean_dR_rel):
            if abs(mean_dK_rel) < 0.10 and abs(mean_dR_rel) < 0.05:
                status = "stable_reflective"
            else:
                status = "possible_exchange_or_growth"

        with open(cond_summary_path, "w") as fstat:
            fstat.write(
                f"Mean deltaK_rel = {mean_dK_rel:.6g}\n"
                f"Mean deltaR_rel = {mean_dR_rel:.6g}\n"
                f"Mean dK_dt = {mean_dK_dt:.6g} 1/s\n"
                f"Mean dR_dt = {mean_dR_dt:.6g} µm/s\n"
                f"Classification (heuristic) = {status}\n"
            )

        # Overlay plot of all K(t) curves
        k_curves_img = plot_all_control_K_curves(control_metrics_list, results_dir, return_array=True)
        if k_curves_img is not None:
            image_collector.add_image("NoBleach", "K_curves", k_curves_img, "all_K_curves")
        
        # Overlay plot of all R(t) curves
        r_curves_img = plot_all_control_r_curves(control_metrics_list, results_dir, return_array=True)
        if r_curves_img is not None:
            image_collector.add_image("NoBleach", "r_curves", r_curves_img, "all_r_curves")
        
        # Combined deltaK/deltaR plot
        delta_img = plot_control_deltaK_deltaR(control_metrics_list, results_dir, dK_rel_all, dR_rel_all, mean_dK_dt, mean_dR_dt, return_array=True)
        if delta_img is not None:
            image_collector.add_image("NoBleach", "K_r_curves", delta_img, "deltaK_deltaR")
    
    # Second pass: run FRAP/PDE analysis, optionally using global NoBleach PB curve
    all_summaries_sep = []
    all_summaries_comb = []

    for path in frap_paths:
        summary_sep, summary_comb = process_movie(
            path,
            movie_dir,
            pixel_size,
            frame_interval,
            D_min,
            D_max,
            droplet_min_size,
            ratio_percentile,
            max_frames,
            downsample_factor,
            D_grid_size,
            D_refine_grid_size,
            n_jobs_D,
            crop_margin,
            fit_region,
            noise_points_per_frame,
            photobleach_correct,
            exp_plateau_fit,
            pb_exclude_radius,
            pb_ratio_tol,
            pde_first_frame,
            pde_last_frame,
            exp_first_frame,
            exp_last_frame,
            combined_model,
            T_switch_frame_arg,
            control_pb_curve=global_pb_curve,
            image_collector=image_collector,
            summary_dir=summary_dir,
        )
        all_summaries_sep.append(summary_sep)
        if combined_model and summary_comb is not None:
            all_summaries_comb.append(summary_comb)

    # Filter out K_exp outliers (+/- 2 SD from mean) from all summaries
    # This filtering is applied early so outliers are excluded from all outputs
    if all_summaries_sep:
        # Extract K_exp values for filtering
        k_exp_values = []
        valid_indices = []
        for i, s in enumerate(all_summaries_sep):
            k_exp = s.get("k_single_exp", np.nan)
            if s.get("success", False) and np.isfinite(k_exp):
                k_exp_values.append(k_exp)
                valid_indices.append(i)
        
        if len(k_exp_values) > 0:
            k_exp_array = np.array(k_exp_values)
            k_exp_mean = np.nanmean(k_exp_array)
            k_exp_std = np.nanstd(k_exp_array)
            k_exp_lower = k_exp_mean - 2 * k_exp_std
            k_exp_upper = k_exp_mean + 2 * k_exp_std
            
            # Create mask for valid summaries (within +/- 2 SD)
            keep_mask = np.ones(len(all_summaries_sep), dtype=bool)
            filtered_count = 0
            
            for i, s in enumerate(all_summaries_sep):
                k_exp = s.get("k_single_exp", np.nan)
                if s.get("success", False) and np.isfinite(k_exp):
                    if k_exp < k_exp_lower or k_exp > k_exp_upper:
                        keep_mask[i] = False
                        filtered_count += 1
                        filename = s.get("filename", "unknown")
                        logger.info(f"Filtering outlier: {filename} (K_exp={k_exp:.6f} outside [{k_exp_lower:.6f}, {k_exp_upper:.6f}])")
            
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} FRAP movies with K_exp outside +/- 2 SD range [{k_exp_lower:.6f}, {k_exp_upper:.6f}]")
                # Filter summaries
                all_summaries_sep = [s for i, s in enumerate(all_summaries_sep) if keep_mask[i]]
                
                # Also filter corresponding combined summaries if they exist
                if combined_model and all_summaries_comb:
                    # Match by filename
                    sep_filenames = {s.get("filename", "") for s in all_summaries_sep}
                    all_summaries_comb = [s for s in all_summaries_comb if s.get("filename", "") in sep_filenames]

    # Write combined CSV with Models column (Combined/Separate)
    if all_summaries_sep or (combined_model and all_summaries_comb):
        combined_csv = os.path.join(results_dir, "FRAP_condition_summary.csv")
        logger.info(f"Writing combined summary CSV to {combined_csv}")
        
        with open(combined_csv, "w") as f:
            # Header
            f.write("Models,filename,D_um2_per_s,success,cost,n_frames_fit,"
                    "frame_interval_s,pixel_size_um,background,pre_mean,"
                    "k_single_exp,mobile_fraction,immobile_fraction,"
                    "YM_single_exp,Y0_single_exp,K_pre,bleach_fraction,T_switch_frame,droplet_radius_um\n")
            
            # Write rows: Combined and Separate for each movie
            # Match summaries by filename
            summaries_by_name = {}
            for s in all_summaries_sep:
                name = s.get("filename", "")
                summaries_by_name[name] = {"separate": s}

            if combined_model and all_summaries_comb:
                for s in all_summaries_comb:
                    name = s.get("filename", "")
                    if name not in summaries_by_name:
                        summaries_by_name[name] = {}
                    summaries_by_name[name]["combined"] = s
            
            # Write in order: Combined then Separate for each movie
            for name in sorted(summaries_by_name.keys()):
                entry = summaries_by_name[name]
                
                # Combined row
                if "combined" in entry:
                    s = entry["combined"]
                    f.write(f"Combined,{s.get('filename', '')},{s.get('D_um2_per_s', np.nan):.6g},"
                            f"{int(s.get('success', False))},{s.get('cost', np.nan):.6g},"
                            f"{s.get('n_frames_fit', 0)},{s.get('frame_interval_s', np.nan):.6g},"
                            f"{s.get('pixel_size_um', np.nan):.6g},{s.get('background', np.nan):.6g},"
                            f"{s.get('pre_mean', np.nan):.6g},"
                            f"{s.get('k_single_exp', np.nan):.6g},"
                            f"{s.get('mobile_fraction', np.nan):.6g},"
                            f"{s.get('immobile_fraction', np.nan):.6g},"
                            f"{s.get('YM_single_exp', np.nan):.6g},"
                            f"{s.get('Y0_single_exp', np.nan):.6g},"
                            f"{s.get('K_pre', np.nan):.6g},"
                            f"{s.get('bleach_fraction', np.nan):.6g},"
                            f"{s.get('T_switch_frame', -1)},"
                            f"{s.get('droplet_radius_um', np.nan):.6g}\n")
                else:
                    # Write empty Combined row if no combined model
                    s_sep = entry.get("separate", {})
                    f.write(f"Combined,{s_sep.get('filename', '')},,,,,,,,,,,,,,,,\n")
                
                # Separate row
                if "separate" in entry:
                    s = entry["separate"]
                    f.write(f"Separate,{s.get('filename', '')},{s.get('D_um2_per_s', np.nan):.6g},"
                            f"{int(s.get('success', False))},{s.get('cost', np.nan):.6g},"
                            f"{s.get('n_frames_fit', 0)},{s.get('frame_interval_s', np.nan):.6g},"
                            f"{s.get('pixel_size_um', np.nan):.6g},{s.get('background', np.nan):.6g},"
                            f"{s.get('pre_mean', np.nan):.6g},"
                            f"{s.get('k_single_exp', np.nan):.6g},"
                            f"{s.get('mobile_fraction', np.nan):.6g},"
                            f"{s.get('immobile_fraction', np.nan):.6g},"
                            f"{s.get('YM_single_exp', np.nan):.6g},"
                            f"{s.get('Y0_single_exp', np.nan):.6g},"
                            f"{s.get('K_pre', np.nan):.6g},"
                            f"{s.get('bleach_fraction', np.nan):.6g},,"
                            f"{s.get('droplet_radius_um', np.nan):.6g}\n")

    # Combined FRAP curves (normalized, photobleach-corrected) side-by-side
    # Only include curves from filtered summaries (outliers excluded)
    if frap_paths:
        # Get set of filenames from filtered summaries
        filtered_filenames = {s.get("filename", "") for s in all_summaries_sep}
        
        frap_curves = []
        frap_names = []
        min_len = None

        for path in frap_paths:
            basename = os.path.splitext(os.path.basename(path))[0]
            # Skip if this movie was filtered out as an outlier
            if basename not in filtered_filenames:
                continue
            basename = os.path.splitext(os.path.basename(path))[0]
            ts_csv = os.path.join(summary_dir, f"{basename}_frap_timeseries.csv")
            if not os.path.exists(ts_csv):
                logger.warning(f"FRAP timeseries CSV not found for {basename}: {ts_csv}")
                continue

            # Read FRAP_fit_ROI column
            try:
                with open(ts_csv, "r") as f_ts:
                    header_line = f_ts.readline().strip()
                    if not header_line:
                        logger.warning(f"Empty timeseries file for {basename}: {ts_csv}")
                        continue
                    headers = header_line.split(",")
                    try:
                        idx_frap = headers.index("FRAP_fit_ROI")
                    except ValueError:
                        logger.warning(f"FRAP_fit_ROI column not found in {ts_csv}")
                        continue

                    vals = []
                    for line in f_ts:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) <= idx_frap:
                            continue
                        try:
                            vals.append(float(parts[idx_frap]))
                        except ValueError:
                            vals.append(np.nan)
            except OSError as e:
                logger.warning(f"Could not read FRAP timeseries for {basename}: {e}")
                continue

            if not vals:
                continue

            arr = np.asarray(vals, dtype=float)
            if min_len is None:
                min_len = arr.size
            else:
                min_len = min(min_len, arr.size)

            frap_curves.append(arr)
            frap_names.append(basename)

        if frap_curves and min_len is not None and min_len > 0:
            n_cols = len(frap_curves)
            mat = np.empty((min_len, n_cols), dtype=float)
            for j, arr in enumerate(frap_curves):
                mat[:, j] = arr[:min_len]

            out_csv = os.path.join(results_dir, "all_frap_curves.csv")
            logger.info(f"Saving combined FRAP curves to {out_csv}")
            with open(out_csv, "w") as f_out:
                # Header: one column per FRAP movie (basename)
                f_out.write(",".join(frap_names) + "\n")
                for i in range(min_len):
                    row = ",".join(f"{mat[i, j]:.6g}" for j in range(n_cols))
                    f_out.write(row + "\n")
            
            # Generate FRAP curves plot starting from post-bleach frame (frame 1)
            plot_all_frap_curves_from_csv(out_csv, results_dir, frame_interval)

    # Create scatter plots for K_exp vs various parameters
    if all_summaries_sep:
        write_r2_scatter_summary(all_summaries_sep, results_dir, movie_name)
        plot_k_exp_scatters(all_summaries_sep, results_dir, image_collector, condition_name=movie_name)
    
    # Save all collected images as TIF stacks
    image_collector.save_stacks(results_dir)
    
    logger.info(f"Finished processing movie: {movie_dir}")
    return movie_dir, True


def write_global_csv(path, summaries):
    logger.info(f"Saving combined summary to {path}")
    with open(path, "w") as f:
        f.write(
            "filename,D_um2_per_s,success,cost,n_frames_fit,"
            "frame_interval_s,pixel_size_um,background,pre_mean,"
            "k_single_exp,mobile_fraction,immobile_fraction,"
            "YM_single_exp,Y0_single_exp,K_pre,bleach_fraction\n"
        )
        for s in summaries:
            f.write(
                f"{s['filename']},{s['D_um2_per_s']:.6g},"
                f"{int(s['success'])},{s['cost']:.6g},"
                f"{s['n_frames_fit']},{s['frame_interval_s']},"
                f"{s['pixel_size_um']},{s['background']:.6g},"
                f"{s['pre_mean']:.6g},"
                f"{s['k_single_exp']:.6g},"
                f"{s['mobile_fraction']:.6g},"
                f"{s['immobile_fraction']:.6g},"
                f"{s['YM_single_exp']:.6g},"
                f"{s['Y0_single_exp']:.6g},"
                f"{s['K_pre']:.6g},"
                f"{s['bleach_fraction']:.6g}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Fit diffusion coefficients from droplet FRAP movies (explicit FD + Numba)."
    )
    parser.add_argument("input_dir", help="Folder with TIF movies")
    parser.add_argument("--droplet-images-root", type=str, default=None,
                        help="Optional root for droplet crops if ND2 files are detected. Default: <input_dir>/Droplet Images")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Optional root directory for analysis results. Default: <input_dir>/Analysis Results")

    # Core imaging (defaults for your setup)
    parser.add_argument("--pixel-size", type=float, default=0.11,
                        help="Pixel size (µm, default 0.11)")
    parser.add_argument("--frame-interval", type=float, default=10.0,
                        help="Frame interval (s, default 10.0)")

    # D search
    parser.add_argument("--D-min", type=float, default=0.001, help="Min D (µm²/s)")
    parser.add_argument("--D-max", type=float, default=2.0, help="Max D (µm²/s)")
    parser.add_argument("--D-grid-size", type=int, default=25, help="#points in coarse D grid")
    parser.add_argument("--D-refine-grid-size", type=int, default=100, help="#points in refine D grid")
    parser.add_argument("--n-jobs-D", type=int, default=8, help="#parallel workers over D grid")

    # Masking
    parser.add_argument("--droplet-min-size", type=int, default=50, help="Min droplet size (pix)")
    parser.add_argument("--ratio-percentile", type=float, default=85.0,
                        help="Percentile of pre/post ratio for bleach ROI")
    parser.add_argument("--crop-margin", type=int, default=5,
                        help="Margin around droplet box before PDE (pixels)")

    # Frames & downsampling
    parser.add_argument("--max-frames", type=int, default=30,
                        help="Upper bound on post-bleach frames considered at all")
    parser.add_argument("--downsample-factor", type=int, default=2,
                        help="Downsample factor for PDE grid (default: 2)")

    # Fit region & noise plot
    parser.add_argument("--fit-region", choices=["bleach", "droplet"], default="bleach",
                        help="Region for FRAP curve / fitting (default: bleach)")
    parser.add_argument("--noise-points-per-frame", type=int, default=1000,
                        help="Max noise pixels per frame in plot")

    # Photobleach options
    parser.add_argument("--no-photobleach-correct", action="store_true",
                        help="Disable photobleach correction.")
    parser.add_argument("--pb-exclude-radius", type=int, default=3,
                        help="Radius (pixels) to dilate bleach mask when defining PB region.")
    parser.add_argument("--pb-ratio-tolerance", type=float, default=0.05,
                        help="Tolerance on frame0/frame1 ratio when defining PB region "
                             "(fractional, default 0.05 = ±5%).")

    # PDE / exponential frame windows
    parser.add_argument("--pde-first-frame", type=int, default=1,
                        help="First post-bleach frame index for PDE fit (>=1).")
    parser.add_argument("--pde-last-frame", type=int, default=7,
                        help="Last post-bleach frame index for PDE fit.")
    parser.add_argument("--exp-first-frame", type=int, default=8,
                        help="First post-bleach frame index for exp plateau fit.")
    parser.add_argument("--exp-last-frame", type=int, default=30,
                        help="Last post-bleach frame index for exp plateau fit.")

    # Exponential plateau toggle
    parser.add_argument("--no-exp-plateau-fit", action="store_true",
                        help="Disable single-exponential plateau fit.")

    # Combined model options
    parser.add_argument("--no-combined-model", action="store_true",
                        help="Disable combined PDE + constrained exponential model output.")
    parser.add_argument("--T-switch-frame", type=int, default=None,
                        help=("Post-bleach frame index at which combined model switches "
                              "from PDE to exponential. "
                              "Default: automatic search over a valid range."))

    # Parallel processing options
    parser.add_argument("--n-jobs-conditions", type=int, default=8,
                        help="Number of parallel workers for processing multiple movie folders (default: 8).")

    # Optional: crop pass-through (for ND2 -> cropped TIFFs via Droplet_Crop.py)
    parser.add_argument("--crop-min-area", type=float, default=50.0)
    parser.add_argument("--crop-morph-open-radius", type=int, default=1)
    parser.add_argument("--crop-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--crop-clear-borders", action="store_true")
    parser.add_argument("--crop-no-clear-borders", action="store_true")
    parser.add_argument("--crop-ratio-blur-sigma", type=float, default=0.7)
    parser.add_argument("--crop-bleach-threshold", type=float, default=1.2)
    parser.add_argument("--crop-crop-multiplier", type=float, default=2.0)
    parser.add_argument("--crop-max-iterations", type=int, default=2000)
    parser.add_argument("--crop-epsilon", type=float, default=1e-10)
    parser.add_argument("--crop-pyramid-levels", type=int, default=4)
    parser.add_argument("--crop-template-update-coeff", type=float, default=0.99)
    parser.add_argument("--crop-pre-smooth-sigma", type=float, default=0.5)
    parser.add_argument("--crop-max-shift-pixels", type=float, default=25.0)
    parser.add_argument("--crop-min-correlation", type=float, default=0.8)
    parser.add_argument("--crop-refine-iterations", type=int, default=20)
    parser.add_argument("--crop-min-nobleach-diameter", type=float, default=10.0)
    parser.add_argument("--crop-min-nobleach-pc", type=float, default=5.0)
    parser.add_argument("--crop-n-jobs", type=int, default=8)
    parser.add_argument("--crop-stabilize", action="store_true",
                        help="Enable stabilization during cropping (default off).")
    parser.add_argument("--crop-quiet", action="store_true",
                        help="Suppress per-droplet logging during cropping.")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="count", default=1,
                        help="-v=info, -vv=debug, default=warning")

    args = parser.parse_args()
    setup_logging(args.verbose)

    input_dir = os.path.abspath(args.input_dir)
    pixel_size = args.pixel_size
    frame_interval = args.frame_interval

    photobleach_correct = not args.no_photobleach_correct
    exp_plateau_fit = not args.no_exp_plateau_fit
    combined_model = not args.no_combined_model
    T_switch_frame_arg = args.T_switch_frame

    def maybe_run_cropping(base_dir):
        # Determine droplet images root
        droplet_root = args.droplet_images_root
        if droplet_root is None:
            droplet_root = os.path.join(base_dir, "Droplet Images")
        droplet_root = os.path.abspath(droplet_root)

        # Detect ND2 files
        nd2_files = sorted(
            glob.glob(os.path.join(base_dir, "*.nd2"))
        )
        tif_files = sorted(
            glob.glob(os.path.join(base_dir, "*.tif")) + glob.glob(os.path.join(base_dir, "*.tiff"))
        )
        if tif_files:
            return base_dir  # nothing to do
        if not nd2_files:
            return base_dir  # no data

        # Build Droplet_Crop command
        droplet_crop_path = os.path.join(os.path.dirname(__file__), "Droplet_Crop.py")
        cmd = [
            "python", droplet_crop_path, base_dir,
            "--output-root", droplet_root,
            "--droplet-images-root", droplet_root,
            "--min-area", str(args.crop_min_area),
            "--morph-open-radius", str(args.crop_morph_open_radius),
            "--smooth-sigma", str(args.crop_smooth_sigma),
            "--ratio-blur-sigma", str(args.crop_ratio_blur_sigma),
            "--bleach-threshold", str(args.crop_bleach_threshold),
            "--crop-multiplier", str(args.crop_crop_multiplier),
            "--max-iterations", str(args.crop_max_iterations),
            "--epsilon", str(args.crop_epsilon),
            "--pyramid-levels", str(args.crop_pyramid_levels),
            "--template-update-coeff", str(args.crop_template_update_coeff),
            "--pre-smooth-sigma", str(args.crop_pre_smooth_sigma),
            "--max-shift-pixels", str(args.crop_max_shift_pixels),
            "--min-correlation", str(args.crop_min_correlation),
            "--refine-iterations", str(args.crop_refine_iterations),
            "--min-nobleach-diameter", str(args.crop_min_nobleach_diameter),
            "--min-nobleach-pc", str(args.crop_min_nobleach_pc),
            "--n-jobs", str(args.crop_n_jobs),
        ]
        if args.crop_clear_borders:
            cmd.append("--clear-borders")
        if args.crop_no_clear_borders:
            cmd.append("--no-clear-borders")
        if args.crop_stabilize:
            cmd.append("--stabilize")
        if args.crop_quiet:
            cmd.append("--quiet")

        logger.info(f"No TIFFs found; running Droplet_Crop on ND2 files in {base_dir}")
        subprocess.run(cmd, check=True)

        return droplet_root

    # If no TIFFs, try cropping ND2 to generate them
    input_dir = maybe_run_cropping(input_dir)

    # Determine results root directory
    results_root = args.results_root
    if results_root is None:
        # Default: create Analysis Results folder at the same level as input_dir
        # But ensure it's not inside "Droplet Images"
        results_root = os.path.join(os.path.dirname(input_dir), "Analysis Results")
        # If input_dir itself is inside "Droplet Images", go up one more level
        if "Droplet Images" in input_dir:
            # Find the parent directory that's not "Droplet Images"
            parts = input_dir.split(os.sep)
            droplet_idx = None
            for i, part in enumerate(parts):
                if part == "Droplet Images":
                    droplet_idx = i
                    break
            if droplet_idx is not None and droplet_idx > 0:
                parent_dir = os.sep.join(parts[:droplet_idx])
                results_root = os.path.join(parent_dir, "Analysis Results")
    results_root = os.path.abspath(results_root)
    logger.info(f"Analysis results will be saved to: {results_root}")

    # After possible cropping, re-scan for movie folders
    # Supports two folder structures:
    # 1. New: [input_dir]/[movie]_results/droplet_stacks/ containing TIF files
    # 2. Old: [input_dir]/[movie]/ containing TIF files directly
    
    subdirs = [d for d in os.listdir(input_dir) 
               if os.path.isdir(os.path.join(input_dir, d)) 
               and not d.startswith('.')]
    
    tif_files_direct = sorted(
        glob.glob(os.path.join(input_dir, "*.tif"))
        + glob.glob(os.path.join(input_dir, "*.tiff"))
    )
    
    movie_folders = []
    
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(input_dir, subdir)
            
            # Check new structure: [movie]_results/droplet_stacks/
            if subdir.endswith('_results'):
                droplet_stacks_path = os.path.join(subdir_path, "droplet_stacks")
                if os.path.isdir(droplet_stacks_path):
                    tifs_in_stacks = sorted(
                        glob.glob(os.path.join(droplet_stacks_path, "*.tif"))
                        + glob.glob(os.path.join(droplet_stacks_path, "*.tiff"))
                    )
                    if tifs_in_stacks:
                        movie_folders.append(droplet_stacks_path)
                        continue
            
            # Check old structure: TIF files directly in subfolder
            tifs_in_subdir = sorted(
                glob.glob(os.path.join(subdir_path, "*.tif"))
                + glob.glob(os.path.join(subdir_path, "*.tiff"))
            )
            if tifs_in_subdir:
                movie_folders.append(subdir_path)
        
        if movie_folders:
            logger.info(f"Found {len(movie_folders)} movie folders with TIF files")
            logger.info(f"Movie folders: {[os.path.basename(os.path.dirname(f)) if f.endswith('droplet_stacks') else os.path.basename(f) for f in movie_folders]}")
        elif tif_files_direct:
            # TIF files in root, treat as single movie
            logger.info(f"Found TIF files directly in {input_dir}, treating as single movie")
            movie_folders = [input_dir]
        else:
            logger.error(f"No TIF files found in {input_dir} or its subdirectories")
            return
    elif tif_files_direct:
        # Backward compatibility: TIF files directly in input_dir
        logger.info(f"Found TIF files directly in {input_dir}, treating as single movie")
        movie_folders = [input_dir]
    else:
        logger.error(f"No TIF files found in {input_dir}")
        return
    
    # Prepare arguments for parallel processing
    worker_args = []
    for movie_dir in movie_folders:
        worker_args.append((
            movie_dir,
            pixel_size,
            frame_interval,
            args.D_min,
            args.D_max,
            args.droplet_min_size,
            args.ratio_percentile,
            args.max_frames,
            args.downsample_factor,
            args.D_grid_size,
            args.D_refine_grid_size,
            args.n_jobs_D,
            args.crop_margin,
            args.fit_region,
            args.noise_points_per_frame,
            photobleach_correct,
            exp_plateau_fit,
            args.pb_exclude_radius,
            args.pb_ratio_tolerance,
            args.pde_first_frame,
            args.pde_last_frame,
            args.exp_first_frame,
            args.exp_last_frame,
            combined_model,
            T_switch_frame_arg,
            results_root,
            args.verbose,
        ))
    
    # Process movies in parallel or sequentially
    if args.n_jobs_conditions > 1 and len(movie_folders) > 1:
        logger.info(f"Processing {len(movie_folders)} movies with {args.n_jobs_conditions} workers...")
        with ProcessPoolExecutor(max_workers=args.n_jobs_conditions) as executor:
            futures = {executor.submit(process_single_condition, *args): i 
                      for i, args in enumerate(worker_args)}
            
            completed = 0
            failed = []
            for future in as_completed(futures):
                i = futures[future]
                try:
                    movie_dir, success = future.result()
                    if success:
                        completed += 1
                        logger.info(f"Completed movie {completed}/{len(movie_folders)}: {os.path.basename(movie_dir)}")
                    else:
                        failed.append(movie_dir)
                except Exception as e:
                    movie_dir = movie_folders[i]
                    failed.append(movie_dir)
                    logger.error(f"Error processing movie {os.path.basename(movie_dir)}: {e}", exc_info=True)
            
            if failed:
                logger.warning(f"Failed to process {len(failed)} movie(s): {[os.path.basename(f) for f in failed]}")
            logger.info(f"Processed {completed}/{len(movie_folders)} movies successfully")
    else:
        # Sequential processing
        logger.info(f"Processing {len(movie_folders)} movie(s) sequentially...")
        for i, worker_arg in enumerate(worker_args):
            movie_dir = movie_folders[i]
            try:
                _, success = process_single_condition(*worker_arg)
                if success:
                    logger.info(f"Completed movie {i+1}/{len(movie_folders)}: {os.path.basename(movie_dir)}")
                else:
                    logger.warning(f"Movie {os.path.basename(movie_dir)} completed with warnings")
            except Exception as e:
                logger.error(f"Error processing movie {os.path.basename(movie_dir)}: {e}", exc_info=True)

    # Write combined outputs at results_root (all movies aggregated, no condition grouping)
    if len(movie_folders) > 0:
        write_combined_outputs_at_root(results_root, frame_interval)

    logger.info("All done.")


def write_combined_outputs_at_root(results_root, frame_interval):
    """
    After all movies are processed, write combined output files at the top level
    of results_root (Analysis Results). Does not use condition grouping; combines
    all movie-specific data into single files.
    """
    import csv as csv_module
    results_root = os.path.abspath(results_root)
    if not os.path.isdir(results_root):
        return

    # Find all *_results folders that contain FRAP_condition_summary.csv
    subdirs = [d for d in os.listdir(results_root)
               if os.path.isdir(os.path.join(results_root, d)) and d.endswith("_results")]
    summary_files = []
    for d in subdirs:
        path = os.path.join(results_root, d, "FRAP_condition_summary.csv")
        if os.path.isfile(path):
            summary_files.append((d.replace("_results", ""), path))

    if not summary_files:
        logger.info("No per-movie FRAP_condition_summary.csv found; skipping combined outputs.")
        return

    logger.info(f"Writing combined outputs at {results_root} from {len(summary_files)} movies.")

    # 1. Concatenate FRAP_condition_summary.csv with a movie column
    combined_summary_path = os.path.join(results_root, "FRAP_condition_summary.csv")
    header_written = False
    with open(combined_summary_path, "w", newline="") as f_out:
        writer = csv_module.writer(f_out)
        for movie_name, path in summary_files:
            with open(path, "r") as f_in:
                reader = csv_module.reader(f_in)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer.writerow(["movie"] + header)
                    header_written = True
                for row in reader:
                    if len(row) > 0:
                        writer.writerow([movie_name] + row)
    if header_written:
        logger.info(f"Wrote combined FRAP_condition_summary.csv to {combined_summary_path}")

    # 2. Build combined all_frap_curves.csv (columns = movie_curvename, aligned by frame)
    all_curve_dfs = []
    all_movie_names = []
    for movie_name, _ in summary_files:
        movie_results_dir = os.path.join(results_root, movie_name + "_results")
        curves_path = os.path.join(movie_results_dir, "all_frap_curves.csv")
        if not os.path.isfile(curves_path):
            continue
        with open(curves_path, "r") as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            col_names = [c.strip() for c in first_line.split(",")]
            # Prefix columns with movie name to avoid clashes
            prefixed = [f"{movie_name}_{c}" for c in col_names]
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append([float(x) if x.strip() else np.nan for x in line.split(",")])
            if not rows:
                continue
            all_curve_dfs.append((prefixed, rows))
            all_movie_names.append(movie_name)

    if all_curve_dfs:
        max_rows = max(len(rows) for _, rows in all_curve_dfs)
        all_cols = [c for prefixed, _ in all_curve_dfs for c in prefixed]
        combined_curves_path = os.path.join(results_root, "all_frap_curves.csv")
        with open(combined_curves_path, "w") as f_out:
            f_out.write(",".join(all_cols) + "\n")
            for i in range(max_rows):
                row_parts = []
                for prefixed, rows in all_curve_dfs:
                    ncol = len(prefixed)
                    if i < len(rows):
                        for x in rows[i]:
                            if isinstance(x, float) and np.isnan(x):
                                row_parts.append("")
                            else:
                                row_parts.append(f"{x:.6g}")
                    else:
                        row_parts.extend([""] * ncol)
                f_out.write(",".join(row_parts) + "\n")
        logger.info(f"Wrote combined all_frap_curves.csv to {combined_curves_path}")

        # 3. Generate FRAP curves plot at top level
        try:
            plot_all_frap_curves_from_csv(combined_curves_path, results_root, frame_interval)
        except Exception as e:
            logger.warning(f"Could not generate combined FRAP curves plot: {e}")

    # 4. Collect Separate rows from all FRAP_condition_summary.csv into summary dicts (with movie key)
    all_summaries_sep = []
    for movie_name, path in summary_files:
        with open(path, "r") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if row.get("Models", "") != "Separate":
                    continue
                try:
                    s = {
                        "filename": row.get("filename", ""),
                        "movie": movie_name,
                        "D_um2_per_s": float(row["D_um2_per_s"]) if row.get("D_um2_per_s") else np.nan,
                        "success": bool(int(row["success"])) if row.get("success") else False,
                        "cost": float(row["cost"]) if row.get("cost") else np.nan,
                        "n_frames_fit": int(row["n_frames_fit"]) if row.get("n_frames_fit") else 0,
                        "frame_interval_s": float(row["frame_interval_s"]) if row.get("frame_interval_s") else np.nan,
                        "pixel_size_um": float(row["pixel_size_um"]) if row.get("pixel_size_um") else np.nan,
                        "background": float(row["background"]) if row.get("background") else np.nan,
                        "pre_mean": float(row["pre_mean"]) if row.get("pre_mean") else np.nan,
                        "k_single_exp": float(row["k_single_exp"]) if row.get("k_single_exp") else np.nan,
                        "mobile_fraction": float(row["mobile_fraction"]) if row.get("mobile_fraction") else np.nan,
                        "immobile_fraction": float(row["immobile_fraction"]) if row.get("immobile_fraction") else np.nan,
                        "YM_single_exp": float(row["YM_single_exp"]) if row.get("YM_single_exp") else np.nan,
                        "Y0_single_exp": float(row["Y0_single_exp"]) if row.get("Y0_single_exp") else np.nan,
                        "K_pre": float(row["K_pre"]) if row.get("K_pre") else np.nan,
                        "bleach_fraction": float(row["bleach_fraction"]) if row.get("bleach_fraction") else np.nan,
                        "droplet_radius_um": float(row["droplet_radius_um"]) if row.get("droplet_radius_um") else np.nan,
                    }
                    all_summaries_sep.append(s)
                except (ValueError, KeyError):
                    continue

    if all_summaries_sep:
        write_r2_scatter_summary(all_summaries_sep, results_root, "all")
        plot_k_exp_scatters(all_summaries_sep, results_root, None, condition_name="all")

    # 5. Run combine_frap_curves to create stacked_all.csv and averaged_all.csv at results_root
    try:
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        combine_script = os.path.join(script_dir, "combine_frap_curves.py")
        if os.path.isfile(combine_script):
            result = subprocess.run(
                [sys.executable, combine_script, results_root, "--frame-interval", str(frame_interval)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                logger.info("Combined stacked_all.csv and averaged_all.csv written to results root.")
            else:
                logger.warning(f"combine_frap_curves.py exited with code {result.returncode}: {result.stderr[:500] if result.stderr else ''}")
    except Exception as e:
        logger.warning(f"Could not run combine_frap_curves for stacked_all/averaged_all: {e}")


def calculate_r2_pair(x_vals, y_vals):
    """Compute R^2 and p-value for x vs y using linear regression."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    if x_clean.size < 2:
        return np.nan, np.nan
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_clean, y_clean)
    return r_val ** 2, p_val


def write_r2_scatter_summary(all_summaries_sep, results_dir, movie_name):
    """Save R^2 summary and raw data for k_single_exp scatter relationships."""
    data = []
    for s in all_summaries_sep:
        if s.get("success", False) and np.isfinite(s.get("k_single_exp", np.nan)):
            data.append({
                "movie": s.get("movie", movie_name),
                "filename": s.get("filename", ""),
                "k_single_exp": s.get("k_single_exp", np.nan),
                "pre_mean": s.get("pre_mean", np.nan),
                "K_pre": s.get("K_pre", np.nan),
                "bleach_fraction": s.get("bleach_fraction", np.nan),
            })
    if not data:
        logger.warning("No valid Separate-model data for R^2 summary.")
        return

    # Arrays for regression
    k_exp = [d["k_single_exp"] for d in data]
    pre_mean = [d["pre_mean"] for d in data]
    k_pre = [d["K_pre"] for d in data]
    bleach_frac = [d["bleach_fraction"] for d in data]

    r2_pre_mean, p_pre_mean = calculate_r2_pair(pre_mean, k_exp)
    r2_k_pre, p_k_pre = calculate_r2_pair(k_pre, k_exp)
    r2_bleach, p_bleach = calculate_r2_pair(bleach_frac, k_exp)

    summary_rows = [
        ("k_single_exp vs pre_mean", r2_pre_mean, p_pre_mean, len(data)),
        ("k_single_exp vs K_pre", r2_k_pre, p_k_pre, len(data)),
        ("k_single_exp vs bleach_fraction", r2_bleach, p_bleach, len(data)),
    ]

    # Write summary CSV
    summary_path = os.path.join(results_dir, "scatter_plot_R2_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movie", "relationship", "R_squared", "p_value", "n_points"])
        for rel, r2, p, n in summary_rows:
            r2_str = f"{r2:.6f}" if np.isfinite(r2) else "nan"
            p_str = f"{p:.6e}" if np.isfinite(p) else "nan"
            writer.writerow([movie_name, rel, r2_str, p_str, n])

    # Write raw data CSV
    raw_path = os.path.join(results_dir, "scatter_plot_raw_data.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movie", "filename", "k_single_exp", "pre_mean", "K_pre", "bleach_fraction"])
        for d in data:
            writer.writerow([
                d["movie"],
                d["filename"],
                f"{d['k_single_exp']:.6f}" if np.isfinite(d["k_single_exp"]) else "nan",
                f"{d['pre_mean']:.6f}" if np.isfinite(d["pre_mean"]) else "nan",
                f"{d['K_pre']:.6f}" if np.isfinite(d["K_pre"]) else "nan",
                f"{d['bleach_fraction']:.6f}" if np.isfinite(d["bleach_fraction"]) else "nan",
            ])

    logger.info(f"Saved R^2 summary to {summary_path} and raw data to {raw_path}")


def plot_k_exp_scatters(all_summaries_sep, input_dir, image_collector=None, condition_name=None):
    """Create combined 4x1 scatter plots showing K_exp vs various parameters.
    Note: Summaries should already be filtered for outliers before calling this function.
    Includes linear regression with R^2 and p-value on each plot.
    """
    # Extract data (summaries are already filtered for outliers)
    k_exp = []
    pre_mean_vals = []
    k_pre_vals = []
    bleach_frac_vals = []
    radius_vals = []
    
    for s in all_summaries_sep:
        if s.get("success", False) and np.isfinite(s.get("k_single_exp", np.nan)):
            k_exp.append(s.get("k_single_exp"))
            pre_mean_vals.append(s.get("pre_mean"))
            k_pre_vals.append(s.get("K_pre"))
            bleach_frac_vals.append(s.get("bleach_fraction"))
            radius_vals.append(s.get("droplet_radius_um", np.nan))
    
    if not k_exp:
        logger.warning("No valid K_exp values found for scatter plots.")
        return
    
    k_exp = np.array(k_exp)
    pre_mean_vals = np.array(pre_mean_vals)
    k_pre_vals = np.array(k_pre_vals)
    bleach_frac_vals = np.array(bleach_frac_vals)
    radius_vals = np.array(radius_vals)
    
    if len(k_exp) < 2:
        logger.warning("Too few K_exp values for scatter plots.")
        return
    
    # Get condition name for title
    if condition_name is None:
        condition_name = os.path.basename(input_dir)
    
    # Create combined 4x1 plot with K_exp on X-axis
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    
    # Helper function to calculate and plot linear regression
    # Note: K_exp is on X-axis, other parameters on Y-axis
    def plot_with_regression(ax, x, y, y_label, title):
        # Filter out NaN/inf values
        mask_valid = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask_valid]
        y_clean = y[mask_valid]
        
        if len(x_clean) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_ylabel(y_label)
            ax.set_title(title)
            return
        
        # Scatter plot (K_exp on X, parameter on Y)
        ax.scatter(x_clean, y_clean, alpha=0.6, s=50)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        r_squared = r_value ** 2
        
        # Plot regression line
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=1.5)
        
        # Add R^2 and p-value text
        text_str = f"R² = {r_squared:.4f}, p = {p_value:.4e}"
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # Plot 1: K_exp (X) vs pre_mean (Y)
    plot_with_regression(axes[0], k_exp, pre_mean_vals,
                        "Average droplet pre-bleach intensity",
                        f"{condition_name}: K_exp vs Pre-Bleach Intensity")
    
    # Plot 2: K_exp (X) vs K_pre (Y)
    plot_with_regression(axes[1], k_exp, k_pre_vals,
                        "Partition Coefficient",
                        f"{condition_name}: K_exp vs Partition Coefficient")
    
    # Plot 3: K_exp (X) vs bleach_fraction (Y)
    plot_with_regression(axes[2], k_exp, bleach_frac_vals,
                        "bleach_fraction",
                        f"{condition_name}: K_exp vs bleach_fraction")
    
    # Plot 4: K_exp (X) vs droplet radius (Y)
    plot_with_regression(axes[3], k_exp, radius_vals,
                        "Droplet radius (µm)",
                        f"{condition_name}: K_exp vs Droplet Radius")
    
    # Set common X-axis label
    axes[3].set_xlabel("K_exp (1/s)")
    
    plt.tight_layout()
    
    if image_collector is not None:
        img_array = fig_to_array(fig)
        image_collector.add_image("FRAP", "K_exp_scatters_combined", img_array, "K_exp_scatters_combined")
    else:
        out_path = os.path.join(input_dir, "K_exp_scatters_combined.png")
        plt.savefig(out_path, dpi=200)
    plt.close(fig)
    
    logger.info(f"Saved combined K_exp scatter plots")


if __name__ == "__main__":
    main()
