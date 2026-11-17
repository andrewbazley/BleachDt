#!/usr/bin/env python3
"""
GUI for combining and filtering FRAP curves.

Features:
- Tabbed interface for analysis and filtering
- Interactive scatter plot filtering with draggable cutoff lines
- Real-time visualization of filter effects
"""

from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from datetime import datetime

# Local helpers. Support running both as a package module (``bleachdt.gui``)
# and as a standalone script (``python bleachdt/gui.py``).
try:
    from .condition_utils import extract_condition_key
except ImportError:
    from condition_utils import extract_condition_key

# Try PySide6 first, fall back to PyQt6, then PyQt5
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
        QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
        QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar,
        QSplitter, QScrollArea, QFrame
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
    QT_VERSION = "PySide6"
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
            QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
            QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar,
            QSplitter, QScrollArea, QFrame
        )
        from PyQt6.QtCore import Qt, QThread, Signal, QTimer
        from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
        QT_VERSION = "PyQt6"
    except ImportError:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QTabWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
            QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
            QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar,
            QSplitter, QScrollArea, QFrame
        )
        from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QTimer
        from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
        QT_VERSION = "PyQt5"

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("Warning: PyQtGraph not available. Filtering tab will be limited.")

###############################################################################
# Branding assets (BleachDt logo / wordmark)
###############################################################################

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def asset_path(name):
    """Absolute path to a bundled asset in ``bleachdt/assets`` (empty string if missing)."""
    p = os.path.join(_ASSETS_DIR, name)
    return p if os.path.isfile(p) else ""


def app_icon():
    """Return the BleachDt QIcon assembled from the bundled multi-resolution PNGs.

    Setting this on the QApplication makes it the taskbar/dock icon at runtime; setting it
    on each window makes it the window/title-bar icon. Returns an empty QIcon if the assets
    or the platform image plugins are unavailable, so it is always safe to call.
    """
    icon = QIcon()
    try:
        added = False
        for name in ("icon_256.png", "icon_512.png", "icon_1024.png", "favicon.ico"):
            p = asset_path(name)
            if p:
                pm = QPixmap(p)
                if not pm.isNull():
                    icon.addPixmap(pm)
                    added = True
        if not added:
            return QIcon()
    except Exception:
        return QIcon()
    return icon


def wordmark_pixmap(height=34):
    """Render the BleachDt wordmark SVG (logo + name) to a QPixmap at the given height.
    Returns None if QtSvg or the asset is unavailable (callers fall back to text)."""
    svg = asset_path("bleachDt-wordmark.svg")
    if not svg:
        return None
    try:
        if QT_VERSION == "PySide6":
            from PySide6.QtSvg import QSvgRenderer
        elif QT_VERSION == "PyQt6":
            from PyQt6.QtSvg import QSvgRenderer
        else:
            from PyQt5.QtSvg import QSvgRenderer
        from_qtgui = __import__(
            {"PySide6": "PySide6.QtGui", "PyQt6": "PyQt6.QtGui"}.get(QT_VERSION, "PyQt5.QtGui"),
            fromlist=["QPainter"])
        QPainter = from_qtgui.QPainter
        renderer = QSvgRenderer(svg)
        size = renderer.defaultSize()
        if size.height() <= 0:
            return None
        width = int(size.width() * (height / size.height()))
        pm = QPixmap(width, height)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        renderer.render(painter)
        painter.end()
        return pm
    except Exception:
        return None

###############################################################################
# Agent debug logging helper (for runtime instrumentation)
###############################################################################


def _agent_debug_log(message, data=None, hypothesis_id="?", run_id="pre-fix"):
    """
    Lightweight, opt-in debug logger for runtime instrumentation.

    Disabled by default. To capture NDJSON debug lines, set the environment
    variable ``BLEACHDT_DEBUG_LOG`` to a writable file path. Never raises.
    """
    log_path = os.environ.get("BLEACHDT_DEBUG_LOG")
    if not log_path:
        return
    try:
        import time

        timestamp_ms = int(time.time() * 1000)
        log_entry = {
            "id": f"log_{timestamp_ms}",
            "timestamp": timestamp_ms,
            "location": "bleachdt/gui.py",
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        # Never let debug logging break the app
        pass


# Import curve-combining helpers from the sibling ``combine`` module. Support
# both package-relative and standalone-script execution.
try:
    from .combine import (
        find_all_frap_curves_csvs,
        read_and_process_csv,
        transpose_and_label,
        apply_filters,
        filter_jump_drop,
        filter_value_range,
        process_single_csv,
        plot_all_frap_curves,
    )
except ImportError:
    from combine import (
        find_all_frap_curves_csvs,
        read_and_process_csv,
        transpose_and_label,
        apply_filters,
        filter_jump_drop,
        filter_value_range,
        process_single_csv,
        plot_all_frap_curves,
    )


class DropletCropWorker(QThread):
    """Worker thread for running droplet_crop.py to crop and save droplet stacks."""
    finished = Signal(bool, str)
    progress = Signal(str)
    
    def __init__(self, input_dir, params):
        super().__init__()
        self.input_dir = input_dir
        self.params = params
    
    def run(self):
        try:
            import subprocess
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, "droplet_crop.py")
            
            if not os.path.exists(script_path):
                self.finished.emit(False, f"droplet_crop.py not found at {script_path}")
                return
            
            p = self.params
            # New structure: droplet stacks go inside Analysis Results/[movie]_results/droplet_stacks/
            results_root = os.path.join(self.input_dir, "Analysis Results")
            droplet_root = p.get('droplet_images_root') or results_root
            
            cmd = [
                sys.executable, script_path,
                self.input_dir,
                "--output-root", droplet_root,
                "--min-area", str(p.get('min_area', 50.0)),
                "--morph-open-radius", str(p.get('morph_open_radius', 1)),
                "--smooth-sigma", str(p.get('smooth_sigma', 1.0)),
                "--ratio-blur-sigma", str(p.get('ratio_blur_sigma', 2.0)),
                "--bleach-threshold", str(p.get('bleach_threshold', 1.2)),
                "--crop-multiplier", str(p.get('crop_multiplier', 2.0)),
                "--max-iterations", str(p.get('max_iterations', 2000)),
                "--epsilon", str(p.get('epsilon', 1e-10)),
                "--pyramid-levels", str(p.get('pyramid_levels', 4)),
                "--template-update-coeff", str(p.get('template_update_coeff', 0.99)),
                "--pre-smooth-sigma", str(p.get('pre_smooth_sigma', 0.5)),
                "--max-shift-pixels", str(p.get('max_shift_pixels', 25.0)),
                "--min-correlation", str(p.get('min_correlation', 0.8)),
                "--refine-iterations", str(p.get('refine_iterations', 20)),
                "--min-diameter", str(p.get('min_nobleach_diameter', 10.0)),
                "--min-pc", str(p.get('min_nobleach_pc', 5.0)),
                "--min-bleach-depth", str(p.get('min_bleach_depth', 0.3)),
                "--pixel-size", str(p.get('pixel_size', 0.11)),
                "--detect-blur-um", str(p.get('detect_blur_um', 1.0)),
                "--n-jobs", str(p.get('n_jobs', 8)),
            ]
            
            if p.get('clear_borders', True):
                cmd.append("--clear-borders")
            else:
                cmd.append("--no-clear-borders")
            
            if not p.get('stabilize', True):
                cmd.append("--no-stabilize")

            if p.get('save_threshold_masks', False):
                cmd.append("--save-threshold-masks")

            if p.get('quiet', False):
                cmd.append("--quiet")
            
            self.progress.emit("Running Droplet Crop pipeline...")
            self.progress.emit(f"Command: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                if line:
                    self.progress.emit(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.progress.emit("\nDroplet cropping completed successfully!")
                self.finished.emit(True, droplet_root)
            else:
                self.finished.emit(False, f"Droplet cropping failed with return code {process.returncode}")
                
        except Exception as e:
            self.finished.emit(False, f"Error running droplet crop: {str(e)}\n{traceback.format_exc()}")


class FullAnalysisWorker(QThread):
    """Worker thread for running full FRAP analysis pipeline (crop_pde.py)."""
    finished = Signal(bool, str)
    progress = Signal(str)
    progress_update = Signal(int, int)  # (current, total) for progress bar
    
    def __init__(self, input_dir, params):
        super().__init__()
        self.input_dir = input_dir
        self.params = params
    
    def _count_movies(self):
        """Count the number of movie folders to process (for progress bar)."""
        import glob
        # If input_dir looks like Analysis Results root, count *_results/droplet_stacks folders
        subdirs = [d for d in os.listdir(self.input_dir)
                   if os.path.isdir(os.path.join(self.input_dir, d)) and not d.startswith('.')]
        movie_count = 0
        for d in subdirs:
            droplet_stacks = os.path.join(self.input_dir, d, "droplet_stacks")
            if os.path.isdir(droplet_stacks):
                tifs = (glob.glob(os.path.join(droplet_stacks, "*.tif")) +
                        glob.glob(os.path.join(droplet_stacks, "*.tiff")))
                if tifs:
                    movie_count += 1
        if movie_count > 0:
            return movie_count
        # Fallback: count TIF files under input_dir
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            movie_count += len(glob.glob(os.path.join(self.input_dir, '**', ext), recursive=True))
        return max(1, movie_count)
    
    def run(self):
        try:
            import subprocess
            import glob
            import re
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            p = self.params
            
            # Set results root to be in the same folder as input_dir
            results_root = p.get('results_root') or os.path.join(self.input_dir, "Analysis Results")
            os.makedirs(results_root, exist_ok=True)
            
            # Check if we need to process raw movie files first
            if p.get('has_raw_movies', False):
                self.progress.emit("Step 1: Processing raw movie files...\n")
                raw_files = p.get('raw_movie_files', [])
                total_movies = len(raw_files)
                self.progress_update.emit(0, total_movies)
                
                # Run droplet_crop.py to stabilize and crop raw movies
                crop_script_path = os.path.join(script_dir, "droplet_crop.py")
                
                if not os.path.exists(crop_script_path):
                    self.finished.emit(False, f"droplet_crop.py not found at {crop_script_path}")
                    return
                
                # Build droplet_crop.py command
                crop_cmd = [
                    sys.executable, crop_script_path,
                    self.input_dir,
                    "--output-root", results_root,
                    "--min-area", str(p.get('crop_min_area', 50.0)),
                    "--pre-bleach-frames", str(p.get('pre_bleach_frames', 1)),
                    "--analysis-channel", str(p.get('analysis_channel', 1)),
                    "--morph-open-radius", str(p.get('crop_morph_open_radius', 1)),
                    "--smooth-sigma", str(p.get('crop_smooth_sigma', 1.0)),
                    # 8-bit detection image + physically-sized (micron) pre-threshold blur
                    "--pixel-size", str(p.get('pixel_size', 0.11)),
                    "--detect-blur-um", str(p.get('detect_blur_um', 1.0)),
                    "--ratio-blur-sigma", str(p.get('crop_ratio_blur_sigma', 2.0)),
                    "--bleach-threshold", str(p.get('crop_bleach_threshold', 1.2)),
                    "--crop-multiplier", str(p.get('crop_crop_multiplier', 2.0)),
                    "--min-diameter", str(p.get('crop_min_nobleach_diameter', 0.0)),
                    "--min-pc", str(p.get('crop_min_nobleach_pc', 0.0)),
                    "--bleach-spot-min-area", str(p.get('crop_bleach_spot_min_area', 5.0)),
                    "--min-bleach-depth", str(p.get('crop_min_bleach_depth', 0.3)),
                    "--frame-interval", str(p.get('frame_interval', 10.0)),
                    "--n-jobs", str(p.get('crop_n_jobs', 8)),
                    # Frame-range stabilization (aligned to frame 0, applied to whole stack)
                    "--stabilize-frame-start", str(p.get('stabilize_frame_start', 0)),
                    "--stabilize-frame-end", str(p.get('stabilize_frame_end', -1)),
                    "--stabilize-extrapolation", str(p.get('stabilize_extrapolation', 'hold')),
                    # Stabilization parameters
                    "--pyramid-levels", str(p.get('stab_pyramid_levels', 4)),
                    "--max-iterations", str(p.get('stab_max_iterations', 1000)),
                    "--epsilon", str(p.get('stab_epsilon', 1e-8)),
                    "--template-update-coeff", str(p.get('stab_template_update_coeff', 0.90)),
                    "--pre-smooth-sigma", str(p.get('stab_pre_smooth_sigma', 0.5)),
                    "--max-shift-pixels", str(p.get('stab_max_shift_pixels', 25.0)),
                    "--min-correlation", str(p.get('stab_min_correlation', 0.8)),
                    "--refine-iterations", str(p.get('stab_refine_iterations', 20)),
                ]
                
                if p.get('save_threshold_masks', False):
                    crop_cmd.append("--save-threshold-masks")

                # Stabilization options
                if not p.get('stabilize_enabled', True):
                    crop_cmd.append("--no-stabilize")
                else:
                    stab_algo = p.get('stabilize_algorithm', 'lucas-kanade')
                    crop_cmd.extend(["--stabilize-algorithm", stab_algo])
                    # Subpixel drift application is the default; opt in to whole-pixel
                    # (exact-intensity-preserving) application only when requested.
                    if p.get('whole_pixel_stabilization', False):
                        crop_cmd.append("--whole-pixel-stabilization")

                # Clear borders (default OFF now: keep every droplet, including edge-touching)
                if p.get('crop_clear_borders', False):
                    crop_cmd.append("--clear-borders")
                else:
                    crop_cmd.append("--no-clear-borders")

                # Variable acquisition timing (e.g. "300x0.1,35x10"); metadata is tried first.
                _timing = str(p.get('timing_segments', '') or '').strip()
                if _timing:
                    crop_cmd.extend(["--timing-segments", _timing])

                self.progress.emit(f"Running droplet_crop.py on {total_movies} movies...")
                self.progress.emit(f"Command: {' '.join(crop_cmd)}\n")
                
                # Run droplet_crop.py
                process = subprocess.Popen(
                    crop_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                current_movie = 0
                from collections import deque
                last_lines = deque(maxlen=50)
                for line in process.stdout:
                    if line:
                        line_stripped = line.strip()
                        last_lines.append(line_stripped)
                        self.progress.emit(line_stripped)
                        # Track progress: Droplet_Crop prints "Processing: filename" once per file
                        if line_stripped.startswith("Processing: ") or "Completed" in line_stripped:
                            current_movie += 1
                            self.progress_update.emit(min(current_movie, total_movies), total_movies)
                
                process.wait()
                
                # Check that at least one movie produced droplet TIFs (otherwise Crop_PDE
                # will fail with "No TIF files found").
                import glob
                any_droplets = False
                for name in os.listdir(results_root):
                    if not name.endswith("_results"):
                        continue
                    droplet_dir = os.path.join(results_root, name, "droplet_stacks")
                    if os.path.isdir(droplet_dir):
                        tifs = glob.glob(os.path.join(droplet_dir, "*.tif")) + glob.glob(os.path.join(droplet_dir, "*.tiff"))
                        if tifs:
                            any_droplets = True
                            break

                # A non-zero return code means AT LEAST ONE movie failed (e.g. a corrupt
                # ND2). That must not sink the whole run: abort only if NOTHING was
                # produced; otherwise warn and continue with the movies that succeeded.
                if not any_droplets:
                    self.progress.emit("\n--- Last lines of Droplet_Crop output ---")
                    for L in last_lines:
                        self.progress.emit(L)
                    if process.returncode != 0:
                        self.finished.emit(False, f"Droplet cropping failed for every movie (return code {process.returncode}) and produced no droplet TIF files. Check the log above for the per-file errors.")
                    else:
                        self.finished.emit(False, "Droplet cropping produced no droplet TIF files (no droplets detected in any movie). Check the log above. Common causes: wrong bleach depth, no droplets in frames, or image format/dimensions not (T, Y, X).")
                    return
                elif process.returncode != 0:
                    self.progress.emit(
                        f"\nWARNING: droplet cropping reported errors (return code {process.returncode}) "
                        f"for one or more movies (e.g. a corrupt/unreadable file). Continuing with the "
                        f"movies that succeeded.\n")
                
                self.progress.emit("\nStep 1 complete. Cropped droplet stacks created.\n")
                self.progress.emit("Step 2: Running PDE fitting analysis...\n")
                
                # Update input_dir to point to the results root where cropped files are
                self.input_dir = results_root
            
            # Count movies for progress tracking
            total_movies = self._count_movies()
            current_movie = 0
            self.progress_update.emit(0, total_movies)
            
            # Find the crop_pde.py script
            script_path = os.path.join(script_dir, "crop_pde.py")
            
            if not os.path.exists(script_path):
                self.finished.emit(False, f"crop_pde.py not found at {script_path}")
                return
            
            # Build command with all parameters
            cmd = [
                sys.executable, script_path,
                self.input_dir,
                "--results-root", results_root,
                # Core imaging
                "--pixel-size", str(p.get('pixel_size', 0.11)),
                "--frame-interval", str(p.get('frame_interval', 10.0)),
                "--pre-bleach-frames", str(p.get('pre_bleach_frames', 1)),
                "--analysis-channel", str(p.get('analysis_channel', 1)),
                # D search
                "--D-min", str(p.get('d_min', 0.001)),
                "--D-max", str(p.get('d_max', 2.0)),
                "--D-grid-size", str(p.get('d_grid_size', 25)),
                "--D-refine-grid-size", str(p.get('d_refine_grid_size', 100)),
                "--n-jobs-D", str(p.get('n_jobs_d', 8)),
                # Masking
                "--droplet-min-size", str(p.get('droplet_min_size', 50)),
                "--ratio-percentile", str(p.get('ratio_percentile', 85.0)),
                "--bleach-depth", str(p.get('bleach_depth', 0.0)),
                "--crop-margin", str(p.get('crop_margin', 5)),
                # Frames & downsampling
                "--max-frames", str(p.get('max_frames', -1)),
                "--downsample-factor", str(p.get('downsample_factor', 2)),
                # Fit region
                "--fit-region", p.get('fit_region', 'bleach'),
                # Normalization method for the reported recovery curve. Both
                # normalizations + raw intensities are always saved regardless.
                "--normalization-method", p.get('normalization_method', 'both'),
                "--noise-points-per-frame", str(p.get('noise_points_per_frame', 1000)),
                # Photobleach options
                "--pb-exclude-radius", str(p.get('pb_exclude_radius', 3)),
                "--pb-ratio-tolerance", str(p.get('pb_ratio_tolerance', 0.05)),
                # PDE / exponential frame windows
                "--pde-first-frame", str(p.get('pde_first_frame', 1)),
                "--pde-last-frame", str(p.get('pde_last_frame', 7)),
                "--exp-first-frame", str(p.get('exp_first_frame', 8)),
                "--exp-last-frame", str(p.get('exp_last_frame', 30)),
                # Parallel processing
                "--n-jobs-conditions", str(p.get('n_jobs_conditions', 8)),
            ]
            
            # Optional flags
            if p.get('no_photobleach_correct', False):
                cmd.append("--no-photobleach-correct")
            if p.get('no_exp_plateau_fit', False):
                cmd.append("--no-exp-plateau-fit")
            # PDE fitting disabled -> skip the D grid search entirely (single-exp only).
            if not p.get('pde_fitting_enabled', True):
                cmd.append("--no-pde")
            if p.get('no_combined_model', False):
                cmd.append("--no-combined-model")
            if p.get('t_switch_frame') is not None:
                cmd.extend(["--T-switch-frame", str(p.get('t_switch_frame'))])
            # Diagnostic-plot save toggles
            if p.get('save_frap_fit_separate', False):
                cmd.append("--save-frap-fit-separate")
            if p.get('save_k_exp_scatters', False):
                cmd.append("--save-k-exp-scatters")
            if p.get('save_noise_pixels', False):
                cmd.append("--save-noise-pixels")
            if not p.get('save_nobleach_k_curves', True):
                cmd.append("--no-nobleach-k-curves")
            if p.get('two_channel', False):
                cmd.append("--two-channel")
            
            # Droplet crop pass-through options
            if p.get('run_droplet_crop', False):
                cmd.extend([
                    "--crop-min-area", str(p.get('crop_min_area', 50.0)),
                    "--crop-morph-open-radius", str(p.get('crop_morph_open_radius', 1)),
                    "--crop-smooth-sigma", str(p.get('crop_smooth_sigma', 1.0)),
                    "--crop-ratio-blur-sigma", str(p.get('crop_ratio_blur_sigma', 2.0)),
                    "--crop-bleach-threshold", str(p.get('crop_bleach_threshold', 1.2)),
                    "--crop-crop-multiplier", str(p.get('crop_crop_multiplier', 2.0)),
                    "--crop-max-iterations", str(p.get('crop_max_iterations', 2000)),
                    "--crop-epsilon", str(p.get('crop_epsilon', 1e-10)),
                    "--crop-pyramid-levels", str(p.get('crop_pyramid_levels', 4)),
                    "--crop-template-update-coeff", str(p.get('crop_template_update_coeff', 0.99)),
                    "--crop-pre-smooth-sigma", str(p.get('crop_pre_smooth_sigma', 0.5)),
                    "--crop-max-shift-pixels", str(p.get('crop_max_shift_pixels', 25.0)),
                    "--crop-min-correlation", str(p.get('crop_min_correlation', 0.8)),
                    "--crop-refine-iterations", str(p.get('crop_refine_iterations', 20)),
                    "--crop-min-nobleach-diameter", str(p.get('crop_min_nobleach_diameter', 10.0)),
                    "--crop-min-nobleach-pc", str(p.get('crop_min_nobleach_pc', 5.0)),
                    "--crop-n-jobs", str(p.get('crop_n_jobs', 8)),
                ])
                if p.get('stabilize_enabled', True):
                    cmd.append("--crop-stabilize")
                    cmd.extend(["--crop-stabilize-algorithm", p.get('stabilize_algorithm', 'lucas-kanade')])
                if p.get('crop_clear_borders', True):
                    cmd.append("--crop-clear-borders")
                else:
                    cmd.append("--crop-no-clear-borders")
                if p.get('crop_quiet', False):
                    cmd.append("--crop-quiet")
            
            # Verbosity
            verbosity = p.get('verbosity', 1)
            if verbosity >= 2:
                cmd.append("-vv")
            elif verbosity >= 1:
                cmd.append("-v")
            
            self.progress.emit(f"Running full analysis pipeline ({total_movies} files detected)...")
            self.progress.emit(f"Command: {' '.join(cmd)}\n")
            
            # Run the script and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Patterns to detect movie processing progress (Crop_PDE and Droplet_Crop log output)
            completed_movie_re = re.compile(r'Completed movie\s+(\d+)/(\d+)', re.IGNORECASE)
            processing_movie_re = re.compile(r'Processing movie:', re.IGNORECASE)
            processing_file_re = re.compile(r'^Processing:\s+', re.IGNORECASE)  # Droplet_Crop
            bracket_re = re.compile(r'\[(\d+)/(\d+)\]')
            
            from collections import deque
            last_lines = deque(maxlen=50)
            # Read output line by line
            for line in process.stdout:
                if line:
                    line_stripped = line.strip()
                    last_lines.append(line_stripped)
                    self.progress.emit(line_stripped)
                    
                    # Completed movie X/Y (Crop_PDE)
                    m = completed_movie_re.search(line_stripped)
                    if m:
                        try:
                            current_movie = int(m.group(1))
                            total_movies = int(m.group(2))
                            self.progress_update.emit(current_movie, total_movies)
                        except ValueError:
                            pass
                        continue
                    # Processing movie: or Processing: filename
                    if processing_movie_re.search(line_stripped) or processing_file_re.search(line_stripped):
                        current_movie += 1
                        self.progress_update.emit(min(current_movie, total_movies), total_movies)
                        continue
                    # [X/Y] style
                    m = bracket_re.search(line_stripped)
                    if m:
                        try:
                            current_movie = int(m.group(1))
                            total_movies = int(m.group(2))
                            self.progress_update.emit(current_movie, total_movies)
                        except ValueError:
                            pass
            
            process.wait()
            
            if process.returncode == 0:
                # If user unchecked "Save droplet TIF stacks" and we processed raw movies, remove droplet_stacks
                if p.get('has_raw_movies', False) and not p.get('run_droplet_crop', False):
                    try:
                        import shutil
                        for name in os.listdir(results_root):
                            droplet_dir = os.path.join(results_root, name, "droplet_stacks")
                            if os.path.isdir(droplet_dir):
                                shutil.rmtree(droplet_dir)
                                self.progress.emit(f"Removed droplet stacks (Save droplet TIF stacks was unchecked): {droplet_dir}")
                    except Exception as cleanup_err:
                        self.progress.emit(f"Note: could not remove droplet_stacks: {cleanup_err}")
                self.progress.emit("\nFull analysis pipeline completed successfully!")
                self.finished.emit(True, "Full analysis pipeline completed successfully!")
            else:
                self.progress.emit("\n--- Last lines of Crop_PDE output ---")
                for L in last_lines:
                    self.progress.emit(L)
                self.finished.emit(False, f"Analysis pipeline failed with return code {process.returncode}. Check the log above for the error.")
                
        except Exception as e:
            self.finished.emit(False, f"Error running analysis pipeline: {str(e)}\n{traceback.format_exc()}")


class AnalysisWorker(QThread):
    """Worker thread for running analysis to avoid freezing GUI."""
    finished = Signal(bool, str)
    progress = Signal(str)
    
    def __init__(self, input_dir, max_drop, max_jump, ignore_first_n, 
                 min_val, max_val, no_filter, n_jobs=None,
                 allowed_filenames=None, output_prefix="", frame_interval=10.0,
                 pool_filtered=False):
        super().__init__()
        self.input_dir = input_dir
        self.max_drop = max_drop
        self.max_jump = max_jump
        self.ignore_first_n = ignore_first_n
        self.min_val = min_val
        self.max_val = max_val
        self.no_filter = no_filter
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        self.allowed_filenames = set(allowed_filenames) if allowed_filenames else None
        self.output_prefix = output_prefix
        self.frame_interval = frame_interval
        self.pool_filtered = pool_filtered
    
    def run(self):
        try:
            import os
            from pathlib import Path
            
            self.progress.emit(f"Searching for all_frap_curves.csv files in {self.input_dir}...")
            csv_files = find_all_frap_curves_csvs(self.input_dir)
            
            if not csv_files:
                self.finished.emit(False, f"No all_frap_curves.csv files found in {self.input_dir}")
                return
            
            self.progress.emit(f"Found {len(csv_files)} all_frap_curves.csv files")
            
            # Determine number of workers
            n_jobs = max(1, min(self.n_jobs, len(csv_files)))
            self.progress.emit(f"Processing with {n_jobs} parallel worker(s)...")
            
            all_stacked_data = []
            all_averaged_data = {}
            
            if n_jobs == 1:
                # Sequential processing
                for i, csv_path in enumerate(csv_files):
                    result = process_single_csv(csv_path, self.max_drop, self.max_jump,
                                               self.ignore_first_n, self.min_val, self.max_val,
                                               self.no_filter, allowed_filenames=self.allowed_filenames)
                    if result is not None:
                        movie_name, stacked_rows, avg_curve = result
                        all_stacked_data.extend(stacked_rows)
                        all_averaged_data[movie_name] = avg_curve
                        self.progress.emit(f"Processed {i+1}/{len(csv_files)}: {movie_name}")
            else:
                # Parallel processing
                process_func = partial(process_single_csv,
                                      max_drop=self.max_drop,
                                      max_jump=self.max_jump,
                                      ignore_first_n=self.ignore_first_n,
                                      min_val=self.min_val,
                                      max_val=self.max_val,
                                      no_filter=self.no_filter,
                                      allowed_filenames=self.allowed_filenames)
                
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_csv = {executor.submit(process_func, csv_path): csv_path 
                                    for csv_path in csv_files}
                    
                    completed = 0
                    for future in as_completed(future_to_csv):
                        csv_path = future_to_csv[future]
                        try:
                            result = future.result()
                            if result is not None:
                                movie_name, stacked_rows, avg_curve = result
                                all_stacked_data.extend(stacked_rows)
                                all_averaged_data[movie_name] = avg_curve
                                completed += 1
                                self.progress.emit(f"Processed {completed}/{len(csv_files)}: {movie_name}")
                        except Exception as e:
                            self.progress.emit(f"Error processing {csv_path}: {str(e)}")
                
                self.progress.emit(f"Successfully processed {completed}/{len(csv_files)} files")
            
            if not all_stacked_data:
                self.finished.emit(False, "No data to write!")
                return
            
            # Find Analysis Results folder (parent of first results folder)
            first_csv_dir = os.path.dirname(csv_files[0])
            # Navigate up to find Analysis Results folder
            analysis_results_dir = None
            current_dir = first_csv_dir
            for _ in range(5):  # Limit search depth
                parent = os.path.dirname(current_dir)
                if os.path.basename(parent) == "Analysis Results" or os.path.basename(current_dir) == "Analysis Results":
                    analysis_results_dir = parent if os.path.basename(parent) == "Analysis Results" else current_dir
                    break
                current_dir = parent
            
            if analysis_results_dir is None:
                # Fallback: use input_dir/Analysis Results
                analysis_results_dir = os.path.join(os.path.dirname(os.path.dirname(first_csv_dir)), "Analysis Results")
                if not os.path.exists(analysis_results_dir):
                    analysis_results_dir = os.path.join(self.input_dir, "Analysis Results")
            
            os.makedirs(analysis_results_dir, exist_ok=True)
            
            # Determine if this is a filtered run
            is_filtered_run = (self.allowed_filenames is not None) or (self.output_prefix == "filtered_")
            
            # Always generate unfiltered outputs first (process without filtering)
            self.progress.emit("Generating unfiltered outputs...")
            # Process data without filtering for unfiltered outputs
            all_stacked_data_unfiltered = []
            all_averaged_data_unfiltered = {}
            
            if n_jobs == 1:
                for csv_path in csv_files:
                    result = process_single_csv(csv_path, self.max_drop, self.max_jump,
                                               self.ignore_first_n, self.min_val, self.max_val,
                                               no_filter=True, allowed_filenames=None)  # No filtering
                    if result is not None:
                        movie_name, stacked_rows, avg_curve = result
                        all_stacked_data_unfiltered.extend(stacked_rows)
                        all_averaged_data_unfiltered[movie_name] = avg_curve
            else:
                process_func_unfiltered = partial(process_single_csv,
                                                 max_drop=self.max_drop,
                                                 max_jump=self.max_jump,
                                                 ignore_first_n=self.ignore_first_n,
                                                 min_val=self.min_val,
                                                 max_val=self.max_val,
                                                 no_filter=True,  # No filtering
                                                 allowed_filenames=None)
                
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    future_to_csv = {executor.submit(process_func_unfiltered, csv_path): csv_path 
                                    for csv_path in csv_files}
                    
                    for future in as_completed(future_to_csv):
                        try:
                            result = future.result()
                            if result is not None:
                                movie_name, stacked_rows, avg_curve = result
                                all_stacked_data_unfiltered.extend(stacked_rows)
                                all_averaged_data_unfiltered[movie_name] = avg_curve
                        except Exception as e:
                            pass
            
            # Generate unfiltered outputs
            if all_stacked_data_unfiltered:
                self._generate_combined_outputs(
                    all_stacked_data_unfiltered, all_averaged_data_unfiltered, 
                    analysis_results_dir, output_prefix="", csv_files=csv_files
                )
            
            # Generate filtered outputs if filtering is applied
            if is_filtered_run and all_stacked_data:
                self.progress.emit("Generating filtered outputs...")
                self._generate_combined_outputs(
                    all_stacked_data, all_averaged_data, analysis_results_dir,
                    output_prefix="filtered_", csv_files=csv_files
                )
                
                # Save filtered data in movie-level _filtered folders
                self._save_filtered_movie_outputs(csv_files, all_stacked_data, all_averaged_data)
                
                # Optionally pool filtered tracks across conditions
                if self.pool_filtered:
                    try:
                        self._pool_filtered_by_condition(analysis_results_dir)
                    except Exception as e:
                        self.progress.emit(f"Warning: Error pooling filtered tracks by condition: {str(e)}")
            
            
            # Calculate and display summary statistics table
            self.progress.emit("\n" + "="*70)
            self.progress.emit("SUMMARY STATISTICS")
            self.progress.emit("="*70)
            
            # Extract all curve values for statistics (from all_stacked_data)
            all_values = []
            curve_lengths = []
            for row_dict in all_stacked_data:
                frame_cols = sorted([k for k in row_dict.keys() if k.startswith('Mean')], 
                                  key=lambda x: int(x.replace('Mean', '')))
                curve_vals = []
                for col in frame_cols:
                    val = row_dict.get(col)
                    if pd.notna(val):
                        try:
                            curve_vals.append(float(val))
                        except (ValueError, TypeError):
                            pass
                if curve_vals:
                    all_values.extend(curve_vals)
                    curve_lengths.append(len(curve_vals))
            
            # Calculate statistics
            if all_values:
                all_values_arr = np.array(all_values)
                valid_values = all_values_arr[np.isfinite(all_values_arr)]
                
                if len(valid_values) > 0:
                    stats_data = []
                    
                    # Basic statistics
                    stats_data.append(("Total curves processed", f"{len(all_stacked_data)}"))
                    stats_data.append(("Total data points", f"{len(valid_values)}"))
                    stats_data.append(("Average frames per curve", f"{np.mean(curve_lengths):.2f}" if curve_lengths else "N/A"))
                    stats_data.append(("Min frames per curve", f"{min(curve_lengths)}" if curve_lengths else "N/A"))
                    stats_data.append(("Max frames per curve", f"{max(curve_lengths)}" if curve_lengths else "N/A"))
                    
                    # Intensity statistics
                    stats_data.append(("", ""))  # Separator
                    stats_data.append(("Average intensity", f"{np.mean(valid_values):.6f}"))
                    stats_data.append(("Median intensity", f"{np.median(valid_values):.6f}"))
                    stats_data.append(("Min intensity", f"{np.min(valid_values):.6f}"))
                    stats_data.append(("Max intensity", f"{np.max(valid_values):.6f}"))
                    stats_data.append(("Std deviation", f"{np.std(valid_values):.6f}"))
                    
                    # Detection parameters used
                    stats_data.append(("", ""))  # Separator
                    stats_data.append(("Detection Parameters:", ""))
                    stats_data.append(("  Max Drop", f"{self.max_drop:.3f}" if not self.no_filter else "N/A (disabled)"))
                    stats_data.append(("  Max Jump", f"{self.max_jump:.3f}" if not self.no_filter else "N/A (disabled)"))
                    stats_data.append(("  Ignore First N Frames", f"{self.ignore_first_n}" if not self.no_filter else "N/A (disabled)"))
                    stats_data.append(("  Min Value", f"{self.min_val:.3f}" if not self.no_filter else "N/A (filtering disabled)"))
                    stats_data.append(("  Max Value", f"{self.max_val:.3f}" if not self.no_filter else "N/A (filtering disabled)"))
                    
                    # Processing parameters
                    stats_data.append(("", ""))  # Separator
                    stats_data.append(("Processing Parameters:", ""))
                    stats_data.append(("  Frame Interval (s)", f"{self.frame_interval:.1f}"))
                    stats_data.append(("  Total movies", f"{len(all_averaged_data)}"))
                    
                    # Format as table
                    max_label_len = max(len(str(label)) for label, _ in stats_data)
                    for label, value in stats_data:
                        if label == "":
                            self.progress.emit("")
                        else:
                            padding = " " * (max_label_len - len(label) + 2)
                            self.progress.emit(f"{label}:{padding}{value}")
            
            self.progress.emit("="*70 + "\n")
            
            self.finished.emit(True, f"Successfully processed {len(csv_files)} files!")
            
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}\n{traceback.format_exc()}")
    
    def _generate_combined_outputs(self, all_stacked_data, all_averaged_data, analysis_results_dir, 
                                   output_prefix, csv_files):
        """Generate combined outputs (stacked_all.csv, averaged_all.csv, plots, etc.)"""
        try:
            # Create stacked DataFrame with NEW orientation: columns=movies, rows=frames
            # Collect all curves by movie
            movie_curves = {}  # {movie_name: [[frame_values], ...]}
            
            for row_dict in all_stacked_data:
                movie_name = row_dict.get('label', 'unknown')
                if movie_name not in movie_curves:
                    movie_curves[movie_name] = []
                
                # Extract frame values
                frame_values = []
                frame_cols = sorted([k for k in row_dict.keys() if k.startswith('Mean')], 
                                  key=lambda x: int(x.replace('Mean', '')))
                for col in frame_cols:
                    val = row_dict.get(col, np.nan)
                    frame_values.append(float(val) if pd.notna(val) else np.nan)
                
                movie_curves[movie_name].append(frame_values)
            
            # Find max frames across all curves
            max_frames = 0
            for movie_data in movie_curves.values():
                for curve_data in movie_data:
                    max_frames = max(max_frames, len(curve_data))
            
            # Create stacked CSV: columns are movies (one column per movie), rows are frames
            # For each movie, average all its curves at each frame
            stacked_dict = {}
            for movie_name in sorted(movie_curves.keys()):
                all_curves = movie_curves[movie_name]
                if not all_curves:
                    continue
                
                # Average curves frame by frame
                avg_curve = []
                for frame_idx in range(max_frames):
                    frame_values = []
                    for curve in all_curves:
                        if frame_idx < len(curve) and pd.notna(curve[frame_idx]):
                            frame_values.append(curve[frame_idx])
                    if frame_values:
                        avg_curve.append(np.mean(frame_values))
                    else:
                        avg_curve.append(np.nan)
                
                stacked_dict[movie_name] = avg_curve
            
            # Create DataFrame: columns are movies, rows are frames
            stacked_df_new = pd.DataFrame(stacked_dict)
            stacked_df_new.index = [f'Frame{i+1}' for i in range(len(stacked_df_new))]
            
            stacked_output = os.path.join(analysis_results_dir, f'{output_prefix}stacked_all.csv')
            stacked_df_new.to_csv(stacked_output)
            self.progress.emit(f"Saved stacked CSV: {stacked_output}")
            
            # Create averaged CSV (movies as columns, frames as rows)
            if all_averaged_data:
                max_len = max(len(v) for v in all_averaged_data.values())
                averaged_df = pd.DataFrame()
                for movie_name, avg_curve in sorted(all_averaged_data.items()):
                    padded_curve = np.pad(avg_curve, (0, max_len - len(avg_curve)), 
                                          constant_values=np.nan)
                    averaged_df[movie_name] = padded_curve
                
                averaged_df.index = [f'Mean{i+1}' for i in range(len(averaged_df))]
                averaged_output = os.path.join(analysis_results_dir, f'{output_prefix}averaged_all.csv')
                averaged_df.to_csv(averaged_output)
                self.progress.emit(f"Saved averaged CSV: {averaged_output}")
            
            # Generate FRAP curves plot (need to convert back to old format for plotting)
            # Create temporary stacked_df in old format for plotting
            old_format_data = []
            for row_dict in all_stacked_data:
                old_format_data.append(row_dict)
            old_stacked_df = pd.DataFrame(old_format_data)
            
            self.progress.emit("Generating FRAP curves plot...")
            try:
                plot_all_frap_curves(old_stacked_df, analysis_results_dir, 
                                    frame_interval=self.frame_interval, 
                                    output_prefix=output_prefix)
                self.progress.emit(f"Saved FRAP curves plot: {os.path.join(analysis_results_dir, f'{output_prefix}FRAP_curves.tif')}")
            except Exception as e:
                self.progress.emit(f"Warning: Could not generate FRAP curves plot: {str(e)}")
            
            # Generate scatter plots (pre_mean_vs_radius, scatter_plot_raw_data.csv, scatter_plot_R2_summary.csv)
            self._generate_scatter_plots(csv_files, analysis_results_dir, output_prefix)
            
        except Exception as e:
            self.progress.emit(f"Error generating combined outputs: {str(e)}\n{traceback.format_exc()}")
    
    def _generate_scatter_plots(self, csv_files, analysis_results_dir, output_prefix):
        """Generate scatter plots and related CSVs."""
        try:
            # Collect FRAP summary data
            all_summaries = []
            for csv_path in csv_files:
                results_dir = os.path.dirname(csv_path)
                summary_path = os.path.join(results_dir, "FRAP_condition_summary.csv")
                if os.path.exists(summary_path):
                    try:
                        df = pd.read_csv(summary_path)
                        # Filter to "Separate" models if Models column exists
                        if "Models" in df.columns:
                            df = df[df["Models"] == "Separate"]
                        all_summaries.append(df)
                    except Exception as e:
                        self.progress.emit(f"Warning: Could not read {summary_path}: {str(e)}")
            
            if not all_summaries:
                return
            
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            
            # Filter by allowed_filenames if provided
            if self.allowed_filenames:
                allowed_set = set(self.allowed_filenames)
                combined_summary = combined_summary[combined_summary['filename'].isin(allowed_set)]
            
            if combined_summary.empty:
                return
            
            # Generate scatter_plot_raw_data.csv
            raw_data_path = os.path.join(analysis_results_dir, f'{output_prefix}scatter_plot_raw_data.csv')
            combined_summary.to_csv(raw_data_path, index=False)
            self.progress.emit(f"Saved scatter plot raw data: {raw_data_path}")
            
            # Generate scatter_plot_R2_summary.csv (simplified version)
            # This would require calculating R² values, which is complex
            # For now, just create a placeholder
            r2_summary_path = os.path.join(analysis_results_dir, f'{output_prefix}scatter_plot_R2_summary.csv')
            # TODO: Calculate actual R² values
            self.progress.emit(f"Note: R² summary generation not yet implemented")
            
            # Generate pre_mean_vs_radius_conditions.png
            # This is complex and would require grouping by condition
            # For now, skip this or create a simplified version
            self.progress.emit(f"Note: pre_mean_vs_radius plot generation not yet fully implemented")
            
        except Exception as e:
            self.progress.emit(f"Error generating scatter plots: {str(e)}\n{traceback.format_exc()}")
    
    def _save_filtered_movie_outputs(self, csv_files, all_stacked_data, all_averaged_data):
        """Save filtered data in movie results folder with _filtered suffix in filenames."""
        if not self.allowed_filenames:
            return
        
        self.progress.emit("Saving filtered outputs to movie results folders...")
        
        for csv_path in csv_files:
            try:
                # Find the movie's results folder
                results_dir = os.path.dirname(csv_path)
                if not os.path.exists(results_dir):
                    continue
                
                # Read original CSV
                df = read_and_process_csv(csv_path)
                if df is None or df.empty:
                    continue
                
                # Filter to allowed filenames
                allowed_set = set(self.allowed_filenames)
                existing_cols = [c for c in df.columns if c in allowed_set]
                df_filtered = df[existing_cols]
                
                if df_filtered.empty:
                    continue
                
                # Apply detection thresholds
                if not self.no_filter:
                    df_filtered = apply_filters(df_filtered,
                                               max_drop=self.max_drop,
                                               max_jump=self.max_jump,
                                               ignore_first_n=self.ignore_first_n,
                                               min_val=self.min_val,
                                               max_val=self.max_val)
                
                if df_filtered.empty:
                    continue
                
                # Save filtered all_frap_curves.csv with _filtered suffix (replace existing if present)
                filtered_csv_path = os.path.join(results_dir, "all_frap_curves_filtered.csv")
                df_filtered.to_csv(filtered_csv_path, index=False)
                self.progress.emit(f"Saved filtered CSV: {filtered_csv_path}")
                
            except Exception as e:
                self.progress.emit(f"Error saving filtered outputs for {csv_path}: {str(e)}")

    def _extract_condition_from_movie_name(self, movie_name):
        """
        Extract experimental condition from a movie folder name.

        Historically movie folders were named like:
          - WT_ADP_001_results
          - VF_ATP_401

        New experiments may encode conditions differently (e.g. Ca titrations like
        '0.1mMCa_001_results'). We therefore use a general heuristic:
        strip replicate suffixes and return the remaining prefix.
        """
        return extract_condition_key(movie_name)

    def _pool_filtered_by_condition(self, analysis_results_dir):
        """
        Pool filtered tracks across conditions.
        
        For each condition (e.g., WT_ADP, WT_ATP, VF_ADP, VF_ATP, G226D_ADP, G226D_ATP,
        T229G_ADP, T229G_ATP), find all per-movie 'all_frap_curves_filtered.csv' files,
        and combine their columns into a single CSV per condition.
        
        Output files are written to analysis_results_dir as:
            filtered_all_frap_curves_<CONDITION>.csv
        where CONDITION is e.g. 'WT_ADP'.
        """
        import os
        import pandas as pd
        
        self.progress.emit("Pooling filtered tracks across conditions...")
        
        # Map condition -> list of (movie_name, DataFrame)
        condition_map = {}
        
        for root, dirs, files in os.walk(self.input_dir):
            if "all_frap_curves_filtered.csv" in files:
                filtered_path = os.path.join(root, "all_frap_curves_filtered.csv")
                try:
                    # Determine movie folder name
                    csv_dir = root
                    csv_dir_basename = os.path.basename(csv_dir)
                    if csv_dir_basename == "results":
                        # V2/V3-style structure: parent folder is movie name
                        movie_name = os.path.basename(os.path.dirname(csv_dir))
                    else:
                        movie_name = csv_dir_basename
                    
                    condition = self._extract_condition_from_movie_name(movie_name)
                    if not condition:
                        self.progress.emit(f"Warning: Could not determine condition for movie '{movie_name}'")
                        continue
                    
                    df = pd.read_csv(filtered_path)
                    if df is None or df.empty:
                        continue
                    
                    if condition not in condition_map:
                        condition_map[condition] = []
                    condition_map[condition].append((movie_name, df))
                except Exception as e:
                    self.progress.emit(f"Warning: Failed to read {filtered_path}: {str(e)}")
        
        if not condition_map:
            self.progress.emit("No per-movie filtered CSVs found for pooling.")
            return
        
        # Build pooled CSV per condition
        for condition, movie_dfs in condition_map.items():
            try:
                pooled_df = pd.DataFrame()
                for movie_name, df in movie_dfs:
                    # Prefix columns with movie name to keep them unique
                    new_cols = {}
                    for col in df.columns:
                        new_cols[col] = f"{movie_name}__{col}"
                    df_renamed = df.rename(columns=new_cols)
                    # Concatenate columns
                    pooled_df = pd.concat([pooled_df, df_renamed], axis=1)
                
                if pooled_df.empty:
                    continue
                
                out_name = f"filtered_all_frap_curves_{condition}.csv"
                out_path = os.path.join(analysis_results_dir, out_name)
                pooled_df.to_csv(out_path, index=False)
                self.progress.emit(f"Pooled filtered CSV saved for condition '{condition}': {out_path}")
            except Exception as e:
                self.progress.emit(f"Warning: Failed to pool condition '{condition}': {str(e)}")


class DraggableLine(pg.InfiniteLine):
    """Draggable line for setting cutoffs in scatter plots."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMovable(True)
        self.setHoverPen({'color': 'r', 'width': 3})


class FilteringTab(QWidget):
    """Tab for interactive filtering with scatter plots."""
    
    # Signal emitted when filters are modified
    filters_changed = Signal()
    
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window  # Reference to main window to access analysis tab
        self.frap_data = None
        self.nobleach_data = None
        self.frap_paths = []
        self.nobleach_paths = []
        self.saved_filters = {}  # {param_name: {'min': val, 'max': val}}
        self.current_plot = None
        self.cutoff_lines = []
        self.scatter_plot_item = None
        self.box_roi = None  # Rectangular selection ROI
        self.current_valid_filenames = None
        self.hover_tooltip = None  # Tooltip label for hover
        self.scatter_items = []  # Store scatter plot items for hover detection
        self.source_directory = None  # Track where data is coming from
        self._plot_handlers_connected = False  # Avoid duplicate scene connections (fixes blank GUI)
        self._mouse_move_proxy = None  # Throttled sigMouseMoved (prevents interaction crash)
        self._mouse_click_proxy = None
        self._in_hover = False  # Reentrancy guard for the hover handler
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Source directory label (shown when auto-loaded from analysis)
        self.source_label = QLabel("")
        self.source_label.setStyleSheet("color: #1976D2; font-style: italic; padding: 5px;")
        self.source_label.setVisible(False)
        layout.addWidget(self.source_label)
        
        # Top section: File selection
        file_group = QGroupBox("Load Summary Files")
        file_layout = QVBoxLayout()
        
        # FRAP files
        frap_layout = QHBoxLayout()
        self.frap_label = QLabel("FRAP Summary: Not loaded")
        self.frap_btn = QPushButton("Load FRAP Summary File")
        self.frap_btn.clicked.connect(self.load_frap_summary)
        self.frap_dir_btn = QPushButton("Load FRAP Summaries (Recursive)")
        self.frap_dir_btn.clicked.connect(self.load_frap_summaries_recursive)
        frap_layout.addWidget(self.frap_label)
        frap_layout.addWidget(self.frap_btn)
        frap_layout.addWidget(self.frap_dir_btn)
        frap_layout.addStretch()
        file_layout.addLayout(frap_layout)
        
        # NoBleach files
        nobleach_layout = QHBoxLayout()
        self.nobleach_label = QLabel("NoBleach Summary: Not loaded")
        self.nobleach_btn = QPushButton("Load NoBleach Summary File")
        self.nobleach_btn.clicked.connect(self.load_nobleach_summary)
        self.nobleach_dir_btn = QPushButton("Load NoBleach Summaries (Recursive)")
        self.nobleach_dir_btn.clicked.connect(self.load_nobleach_summaries_recursive)
        nobleach_layout.addWidget(self.nobleach_label)
        nobleach_layout.addWidget(self.nobleach_btn)
        nobleach_layout.addWidget(self.nobleach_dir_btn)
        nobleach_layout.addStretch()
        file_layout.addLayout(nobleach_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Middle section: Plot controls (compact)
        controls_group = QGroupBox("Plot Controls")
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        controls_layout.addWidget(QLabel("Data:"))
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["FRAP", "NoBleach"])
        self.data_type_combo.currentTextChanged.connect(self.update_param_combos)
        controls_layout.addWidget(self.data_type_combo)
        
        controls_layout.addWidget(QLabel("X:"))
        self.x_param_combo = QComboBox()
        controls_layout.addWidget(self.x_param_combo)
        
        controls_layout.addWidget(QLabel("Y:"))
        self.y_param_combo = QComboBox()
        controls_layout.addWidget(self.y_param_combo)
        
        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self.create_plot)
        controls_layout.addWidget(plot_btn)
        controls_layout.addStretch()
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Plot area with legend container below
        if PYQTGRAPH_AVAILABLE:
            plot_container = QWidget()
            plot_layout = QVBoxLayout()
            plot_container.setLayout(plot_layout)
            plot_layout.setContentsMargins(0, 0, 0, 0)
            plot_layout.setSpacing(0)
            
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('black')  # Dark background for contrast
            self.plot_widget.setLabel('left', 'Y', color='white')
            self.plot_widget.setLabel('bottom', 'X', color='white')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            # Set axis text color to white
            self.plot_widget.getAxis('left').setPen(pg.mkPen('white'))
            self.plot_widget.getAxis('left').setTextPen(pg.mkPen('white'))
            self.plot_widget.getAxis('bottom').setPen(pg.mkPen('white'))
            self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen('white'))
            plot_layout.addWidget(self.plot_widget, stretch=1)
            
            # Legend container below plot
            self.legend_container = QWidget()
            self.legend_container.setStyleSheet("background-color: black;")
            self.legend_layout = QHBoxLayout()
            self.legend_container.setLayout(self.legend_layout)
            self.legend_layout.setContentsMargins(10, 5, 10, 5)
            self.legend_layout.setAlignment(Qt.AlignCenter)
            plot_layout.addWidget(self.legend_container)
            
            layout.addWidget(plot_container, stretch=1)
        else:
            no_graph_label = QLabel("PyQtGraph not available. Please install it for interactive plotting.")
            no_graph_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_graph_label)
            self.plot_widget = None
        
        # Filter controls (compact)
        filter_group = QGroupBox("Filter Controls")
        filter_layout = QVBoxLayout()
        filter_layout.setSpacing(3)
        filter_layout.setContentsMargins(5, 5, 5, 5)
        
        # Line buttons (compact)
        line_btn_layout = QHBoxLayout()
        line_btn_layout.setSpacing(3)
        self.add_h_line_btn = QPushButton("Add Horizontal Line")
        self.add_h_line_btn.clicked.connect(lambda: self.add_cutoff_line('horizontal'))
        self.add_h_line_btn.setEnabled(False)
        line_btn_layout.addWidget(self.add_h_line_btn)
        
        self.add_v_line_btn = QPushButton("Add Vertical Line")
        self.add_v_line_btn.clicked.connect(lambda: self.add_cutoff_line('vertical'))
        self.add_v_line_btn.setEnabled(False)
        line_btn_layout.addWidget(self.add_v_line_btn)
        
        self.remove_line_btn = QPushButton("Remove Line")
        self.remove_line_btn.clicked.connect(self.remove_last_line)
        self.remove_line_btn.setEnabled(False)
        line_btn_layout.addWidget(self.remove_line_btn)

        self.box_select_btn = QPushButton("Box Select")
        self.box_select_btn.clicked.connect(self.add_box_roi)
        self.box_select_btn.setEnabled(False)
        line_btn_layout.addWidget(self.box_select_btn)
        
        self.remove_box_btn = QPushButton("Remove Box")
        self.remove_box_btn.clicked.connect(self.remove_box_roi)
        self.remove_box_btn.setEnabled(False)
        line_btn_layout.addWidget(self.remove_box_btn)
        
        self.save_filter_btn = QPushButton("Save Filter")
        self.save_filter_btn.clicked.connect(self.save_current_filter)
        self.save_filter_btn.setEnabled(False)
        line_btn_layout.addWidget(self.save_filter_btn)
        line_btn_layout.addStretch()
        filter_layout.addLayout(line_btn_layout)
        
        # Save/load filter buttons (compact)
        save_layout = QHBoxLayout()
        save_layout.setSpacing(3)
        self.save_config_btn = QPushButton("Save current filters")
        self.save_config_btn.clicked.connect(self.save_filter_config)
        save_layout.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton("Load previous filters")
        self.load_config_btn.clicked.connect(self.load_filter_config)
        save_layout.addWidget(self.load_config_btn)
        
        self.clear_filters_btn = QPushButton("Clear All")
        self.clear_filters_btn.clicked.connect(self.clear_all_filters)
        save_layout.addWidget(self.clear_filters_btn)
        
        self.show_filtered_check = QCheckBox("Show filtered")
        self.show_filtered_check.setChecked(True)
        self.show_filtered_check.stateChanged.connect(self.update_plot_with_filters)
        save_layout.addWidget(self.show_filtered_check)
        
        self.color_by_condition_check = QCheckBox("Color by condition")
        self.color_by_condition_check.setChecked(True)
        self.color_by_condition_check.stateChanged.connect(self.update_plot_with_filters)
        save_layout.addWidget(self.color_by_condition_check)
        
        save_layout.addStretch()
        filter_layout.addLayout(save_layout)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Saved filters display (collapsible with splitter)
        saved_group = QGroupBox("Saved Filters")
        saved_layout = QVBoxLayout()
        saved_layout.setContentsMargins(5, 5, 5, 5)
        
        self.saved_filters_table = QTableWidget()
        self.saved_filters_table.setColumnCount(3)
        self.saved_filters_table.setHorizontalHeaderLabels(["Parameter", "Range", "Movies Included"])
        self.saved_filters_table.horizontalHeader().setStretchLastSection(True)
        self.saved_filters_table.setMaximumHeight(150)  # Limit height
        self.saved_filters_table.setMinimumHeight(80)
        saved_layout.addWidget(self.saved_filters_table)
        
        # Movies status display
        movies_status_label = QLabel("Movies Status:")
        movies_status_label.setStyleSheet("font-weight: bold;")
        saved_layout.addWidget(movies_status_label)
        
        self.movies_status_text = QTextEdit()
        self.movies_status_text.setReadOnly(True)
        self.movies_status_text.setMaximumHeight(100)
        self.movies_status_text.setFont(QFont("Arial", 9))
        self.movies_status_text.setStyleSheet("background-color: #f9f9f9; border: 1px solid #cccccc;")
        saved_layout.addWidget(self.movies_status_text)
        
        saved_group.setLayout(saved_layout)
        layout.addWidget(saved_group)
        
        self.setLayout(layout)
    
    def _clear_legend_below(self):
        """Clear the legend container below the plot."""
        while self.legend_layout.count():
            child = self.legend_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _update_legend_below(self, legend_items):
        """Update the legend displayed below the plot."""
        self._clear_legend_below()
        
        if not legend_items:
            return
        
        # Create legend items with labels and colors
        for label, plot_item in legend_items.items():
            # Get color from plot item (handle both PlotDataItem and ScatterPlotItem)
            if isinstance(plot_item, pg.ScatterPlotItem):
                brush = plot_item.opts.get('brush', (0, 0, 255, 255))
            else:
                brush = plot_item.opts.get('symbolBrush', 'b')
            
            if isinstance(brush, tuple):
                color = brush[:3]  # RGB tuple
            else:
                color = brush
            
            # Create a label widget with colored symbol
            label_widget = QLabel()
            # Set symbol color using a colored circle
            if isinstance(color, tuple):
                color_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
            else:
                color_str = str(color)
            label_widget.setText(f"<span style='color: {color_str}; font-size: 16px;'>●</span> <span style='color: white;'>{label}</span>")
            self.legend_layout.addWidget(label_widget)
        
        self.legend_layout.addStretch()
    
    def load_frap_summary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FRAP Summary CSV", "", "CSV Files (*.csv)"
        )
        if path:
            try:
                df = pd.read_csv(path)
                if df.empty:
                    QMessageBox.warning(self, "Warning", "The CSV file is empty.")
                    return
                self.frap_data = df
                self.frap_paths = [path]
                self.frap_label.setText(f"FRAP Summary: {os.path.basename(path)} ({len(self.frap_data)} rows)")
                self.update_param_combos()
                QMessageBox.information(self, "Success", f"Loaded {len(self.frap_data)} rows from FRAP summary.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load FRAP summary: {str(e)}")
    
    def load_frap_summaries_from_paths(self, csv_files):
        """Load FRAP summaries from a list of file paths (used for auto-population)."""
        try:
            dfs = []
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        dfs.append(df)
                except Exception:
                    continue  # Skip problematic files
            
            if not dfs:
                return  # No valid data
            
            self.frap_data = pd.concat(dfs, ignore_index=True)
            self.frap_paths = csv_files
            self.frap_label.setText(f"FRAP Summary: {len(csv_files)} files, {len(self.frap_data)} total rows")
            self.update_param_combos()
        except Exception as e:
            pass  # Silently fail for auto-population
    
    def load_frap_summaries_recursive(self):
        """Recursively load all FRAP_condition_summary.csv files from a directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory to Search for FRAP Summaries"
        )
        if not dir_path:
            return
        
        csv_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file == "FRAP_condition_summary.csv":
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            QMessageBox.warning(self, "Warning", "No FRAP_condition_summary.csv files found.")
            return
        
        try:
            self.load_frap_summaries_from_paths(csv_files)
            QMessageBox.information(self, "Success", 
                                  f"Loaded {len(csv_files)} FRAP summary files with {len(self.frap_data)} total rows.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load FRAP summaries: {str(e)}")
    
    def load_nobleach_summary(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load NoBleach Summary CSV", "", "CSV Files (*.csv)"
        )
        if path:
            try:
                df = pd.read_csv(path)
                if df.empty:
                    QMessageBox.warning(self, "Warning", "The CSV file is empty.")
                    return
                self.nobleach_data = df
                self.nobleach_paths = [path]
                self.nobleach_label.setText(f"NoBleach Summary: {os.path.basename(path)} ({len(self.nobleach_data)} rows)")
                self.update_param_combos()
                QMessageBox.information(self, "Success", f"Loaded {len(self.nobleach_data)} rows from NoBleach summary.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load NoBleach summary: {str(e)}")
    
    def load_nobleach_summaries_from_paths(self, csv_files):
        """Load NoBleach summaries from a list of file paths (used for auto-population)."""
        try:
            dfs = []
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        dfs.append(df)
                except Exception:
                    continue  # Skip problematic files
            
            if not dfs:
                return  # No valid data
            
            self.nobleach_data = pd.concat(dfs, ignore_index=True)
            self.nobleach_paths = csv_files
            self.nobleach_label.setText(f"NoBleach Summary: {len(csv_files)} files, {len(self.nobleach_data)} total rows")
            self.update_param_combos()
        except Exception as e:
            pass  # Silently fail for auto-population
    
    def load_nobleach_summaries_recursive(self):
        """Recursively load all NoBleach_summary.csv files from a directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory to Search for NoBleach Summaries"
        )
        if not dir_path:
            return
        
        csv_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file == "NoBleach_summary.csv":
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            QMessageBox.warning(self, "Warning", "No NoBleach_summary.csv files found.")
            return
        
        try:
            self.load_nobleach_summaries_from_paths(csv_files)
            QMessageBox.information(self, "Success", 
                                  f"Loaded {len(csv_files)} NoBleach summary files with {len(self.nobleach_data)} total rows.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load NoBleach summaries: {str(e)}")
    
    def auto_load_from_directory(self, directory_path, show_source=True):
        """
        Automatically load FRAP and NoBleach summary CSVs from a directory after analysis.
        
        Args:
            directory_path: Path to search for summary CSVs
            show_source: If True, show the source directory label
        """
        if not directory_path or not os.path.isdir(directory_path):
            return
        
        self.source_directory = directory_path
        
        # Find FRAP summary CSVs
        frap_files = []
        nobleach_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file == "FRAP_condition_summary.csv":
                    frap_files.append(os.path.join(root, file))
                elif file == "NoBleach_summary.csv":
                    nobleach_files.append(os.path.join(root, file))
        
        loaded_any = False
        
        # Load FRAP summaries if found
        if frap_files:
            try:
                self.load_frap_summaries_from_paths(frap_files)
                loaded_any = True
            except Exception:
                pass
        
        # Load NoBleach summaries if found
        if nobleach_files:
            try:
                self.load_nobleach_summaries_from_paths(nobleach_files)
                loaded_any = True
            except Exception:
                pass
        
        # Update source label
        if show_source and loaded_any:
            self.source_label.setText(f"Sourcing from: {directory_path}")
            self.source_label.setVisible(True)
        elif not show_source:
            self.source_label.setVisible(False)
        
        return loaded_any
    
    def notify_filters_changed(self):
        """Notify that filters have been modified - enables Apply Filtering Settings button."""
        self.filters_changed.emit()
        # Also directly enable the button if we have access to the analysis tab
        if self.main_window and hasattr(self.main_window, 'analysis_tab'):
            analysis_tab = self.main_window.analysis_tab
            if hasattr(analysis_tab, 'filter_analysis_btn'):
                analysis_tab.filter_analysis_btn.setEnabled(True)
    
    def update_param_combos(self):
        """Update parameter combo boxes based on selected data type and PDE fitting status."""
        try:
            data_type = self.data_type_combo.currentText()
            
            # Check if PDE fitting is enabled via the analysis tab
            pde_enabled = True  # Default to true if we can't check
            if self.main_window and hasattr(self.main_window, 'analysis_tab'):
                analysis_tab = self.main_window.analysis_tab
                if hasattr(analysis_tab, 'is_pde_fitting_enabled'):
                    pde_enabled = analysis_tab.is_pde_fitting_enabled()
            
            # PDE-specific parameters to exclude when PDE fitting is disabled
            pde_params = ['D_um2_per_s', 'cost', 'T_switch_frame']
            
            if data_type == "FRAP" and self.frap_data is not None and not self.frap_data.empty:
                # Filter to only "Separate" rows if Models column exists
                if "Models" in self.frap_data.columns:
                    available_data = self.frap_data[self.frap_data["Models"] == "Separate"]
                else:
                    available_data = self.frap_data
                
                if available_data.empty:
                    available_data = self.frap_data  # Fallback to all data
                
                # Get numeric columns only
                numeric_cols = available_data.select_dtypes(include=[np.number]).columns.tolist()
                # Remove filename and other non-parameter columns
                exclude_cols = ['filename', 'success', 'n_frames_fit']
                numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                # Exclude PDE parameters if PDE fitting is disabled
                if not pde_enabled:
                    numeric_cols = [c for c in numeric_cols if c not in pde_params]
                
                self.x_param_combo.clear()
                self.x_param_combo.addItems(numeric_cols)
                self.y_param_combo.clear()
                self.y_param_combo.addItems(numeric_cols)
                
            elif data_type == "NoBleach" and self.nobleach_data is not None and not self.nobleach_data.empty:
                numeric_cols = self.nobleach_data.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['filename']
                numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
                
                self.x_param_combo.clear()
                self.x_param_combo.addItems(numeric_cols)
                self.y_param_combo.clear()
                self.y_param_combo.addItems(numeric_cols)
            else:
                self.x_param_combo.clear()
                self.y_param_combo.clear()
        except Exception as e:
            print(f"Error updating param combos: {e}")
            self.x_param_combo.clear()
            self.y_param_combo.clear()
    
    def create_plot(self):
        """Create scatter plot with selected parameters."""
        try:
            # NOTE: do NOT reset self._plot_handlers_connected here. The PlotWidget scene
            # persists across replots, so the throttled hover/click proxies only need to be
            # connected once for the tab's lifetime. Re-connecting on every plot stacked
            # duplicate handlers (N calls per mouse move) and was a cause of the crash.
            _agent_debug_log(
                "create_plot_called",
                data={
                    "data_type": self.data_type_combo.currentText(),
                    "x_param": self.x_param_combo.currentText(),
                    "y_param": self.y_param_combo.currentText(),
                },
                hypothesis_id="H1",
            )
            if not PYQTGRAPH_AVAILABLE:
                QMessageBox.warning(self, "Warning", "PyQtGraph not available for plotting.")
                return
            
            data_type = self.data_type_combo.currentText()
            x_param = self.x_param_combo.currentText()
            y_param = self.y_param_combo.currentText()
            
            if not x_param or not y_param:
                QMessageBox.warning(self, "Warning", "Please select both X and Y parameters.")
                return
            
            # Get data
            if data_type == "FRAP":
                if self.frap_data is None or self.frap_data.empty:
                    QMessageBox.warning(self, "Warning", "Please load FRAP summary first.")
                    return
                if "Models" in self.frap_data.columns:
                    data = self.frap_data[self.frap_data["Models"] == "Separate"]
                    if data.empty:
                        data = self.frap_data  # Fallback
                else:
                    data = self.frap_data
            else:  # NoBleach
                if self.nobleach_data is None or self.nobleach_data.empty:
                    QMessageBox.warning(self, "Warning", "Please load NoBleach summary first.")
                    return
                data = self.nobleach_data
            
            # Check if columns exist
            if x_param not in data.columns or y_param not in data.columns:
                QMessageBox.warning(self, "Warning", f"Column '{x_param}' or '{y_param}' not found in data.")
                return
            
            # Extract values and filenames for color coding
            x_vals = data[x_param].values
            y_vals = data[y_param].values
            filenames = data['filename'].values if 'filename' in data.columns else None
            
            # Filter out NaN/Inf
            mask = np.isfinite(x_vals.astype(float)) & np.isfinite(y_vals.astype(float))
            x_clean = x_vals[mask].astype(float)
            y_clean = y_vals[mask].astype(float)
            if filenames is not None:
                filenames_clean = filenames[mask]
            else:
                filenames_clean = None
            
            if len(x_clean) == 0:
                QMessageBox.warning(self, "Warning", "No valid data points for selected parameters.")
                return
            
            # Clear previous cutoff lines when creating new plot
            for orientation, line in self.cutoff_lines:
                try:
                    self.plot_widget.removeItem(line)
                except Exception:
                    pass
            self.cutoff_lines.clear()

            # Remove existing box ROI if any
            if self.box_roi is not None:
                try:
                    self.plot_widget.removeItem(self.box_roi)
                except Exception:
                    pass
                self.box_roi = None
            
            # Clear legend container
            self._clear_legend_below()
            
            # Store current plot info (before filtering)
            self.current_plot = {
                'data_type': data_type,
                'x_param': x_param,
                'y_param': y_param,
                'x_vals': x_clean,
                'y_vals': y_clean,
                'filenames': filenames_clean,
                'data': data  # Store full data for filtering
            }
            _agent_debug_log(
                "create_plot_data_prepared",
                data={
                    "n_points_total": int(len(x_vals)),
                    "n_points_clean": int(len(x_clean)),
                    "has_filenames": bool(filenames_clean is not None),
                },
                hypothesis_id="H1",
            )
            
            # Update plot with filters applied
            self.update_plot_with_filters()
            
            # Enable controls
            self.add_h_line_btn.setEnabled(True)
            self.add_v_line_btn.setEnabled(True)
            self.remove_line_btn.setEnabled(True)
            self.box_select_btn.setEnabled(True)
            self.remove_box_btn.setEnabled(True)
            self.save_filter_btn.setEnabled(True)
            self.box_select_btn.setEnabled(True)
        except Exception as e:
            _agent_debug_log(
                "create_plot_exception",
                data={"error": repr(e), "traceback": traceback.format_exc()},
                hypothesis_id="H1",
            )
            QMessageBox.critical(self, "Error", f"Failed to create plot: {str(e)}")
    
    def add_cutoff_line(self, orientation):
        """Add a draggable cutoff line to the plot."""
        if not PYQTGRAPH_AVAILABLE or self.current_plot is None:
            return
        
        if orientation == 'horizontal':
            # Get current Y range
            y_range = self.plot_widget.viewRange()[1]
            y_pos = (y_range[0] + y_range[1]) / 2
            line = DraggableLine(pos=y_pos, angle=0, pen=pg.mkPen('r', width=2))
            line.sigPositionChangeFinished.connect(self.update_plot_with_filters)
            self.plot_widget.addItem(line)
            self.cutoff_lines.append(('horizontal', line))
        else:  # vertical
            x_range = self.plot_widget.viewRange()[0]
            x_pos = (x_range[0] + x_range[1]) / 2
            line = DraggableLine(pos=x_pos, angle=90, pen=pg.mkPen('r', width=2))
            line.sigPositionChangeFinished.connect(self.update_plot_with_filters)
            self.plot_widget.addItem(line)
            self.cutoff_lines.append(('vertical', line))
        
        # Update plot to show filtered points
        self.update_plot_with_filters()

    def add_box_roi(self):
        """Add or reset a rectangular ROI to keep points inside the box."""
        if not PYQTGRAPH_AVAILABLE or self.current_plot is None:
            return
        
        # Remove existing ROI
        if self.box_roi is not None:
            try:
                self.plot_widget.removeItem(self.box_roi)
            except Exception:
                pass
            self.box_roi = None
        
        # Use current view range to initialize ROI
        x_range, y_range = self.plot_widget.viewRange()
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        width = (x_range[1] - x_range[0]) * 0.5
        height = (y_range[1] - y_range[0]) * 0.5
        roi_pos = [x_center - width / 2, y_center - height / 2]
        roi_size = [width, height]
        
        self.box_roi = pg.RectROI(roi_pos, roi_size, pen=pg.mkPen('y', width=2))
        self.box_roi.sigRegionChangeFinished.connect(self.update_plot_with_filters)
        self.plot_widget.addItem(self.box_roi)
        
        # Update plot to apply box filter
        self.update_plot_with_filters()
    
    def remove_box_roi(self):
        """Remove the box ROI selection."""
        if not PYQTGRAPH_AVAILABLE or self.box_roi is None:
            return
        
        try:
            self.plot_widget.removeItem(self.box_roi)
        except Exception:
            pass
        self.box_roi = None
        
        # Update plot to remove box filter
        self.update_plot_with_filters()
    
    def remove_last_line(self):
        """Remove the last added cutoff line."""
        if not PYQTGRAPH_AVAILABLE or not self.cutoff_lines:
            return
        
        orientation, line = self.cutoff_lines.pop()
        self.plot_widget.removeItem(line)
        self.update_plot_with_filters()
    
    def update_plot_with_filters(self):
        """Update the plot showing filtered vs unfiltered points."""
        try:
            _agent_debug_log(
                "update_plot_with_filters_called",
                data={
                    "has_current_plot": self.current_plot is not None,
                    "n_cutoff_lines": len(self.cutoff_lines),
                    "has_box_roi": self.box_roi is not None,
                },
                hypothesis_id="H2",
            )
            if not PYQTGRAPH_AVAILABLE or self.current_plot is None:
                return
            
            if self.plot_widget is None:
                return
            
            x_param = self.current_plot.get('x_param')
            y_param = self.current_plot.get('y_param')
            x_vals = self.current_plot.get('x_vals')
            y_vals = self.current_plot.get('y_vals')
            
            if x_vals is None or y_vals is None or len(x_vals) == 0 or len(y_vals) == 0:
                return
            
            # Clear previous scatter plot items and legend (keep lines)
            items_to_remove = []
            try:
                for item in self.plot_widget.listDataItems():
                    # Keep InfiniteLine items (cutoff lines)
                    if not isinstance(item, pg.InfiniteLine):
                        items_to_remove.append(item)
                
                for item in items_to_remove:
                    self.plot_widget.removeItem(item)
            except Exception:
                pass
            
            # Clear scatter items list
            self.scatter_items = []
            
            # Hide tooltip if visible
            if self.hover_tooltip is not None:
                try:
                    self.hover_tooltip.hide()
                except Exception:
                    pass
            
            # Clear legend container
            try:
                self._clear_legend_below()
            except Exception:
                pass
            
            # Start with all points valid
            x_filtered_mask = np.ones(len(x_vals), dtype=bool)
            y_filtered_mask = np.ones(len(y_vals), dtype=bool)
            
            # Apply saved filters for X parameter
            if x_param in self.saved_filters:
                filt = self.saved_filters[x_param]
                if filt.get('min') is not None:
                    x_filtered_mask &= (x_vals >= filt['min'])
                if filt.get('max') is not None:
                    x_filtered_mask &= (x_vals <= filt['max'])
            
            # Apply saved filters for Y parameter
            if y_param in self.saved_filters:
                filt = self.saved_filters[y_param]
                if filt.get('min') is not None:
                    y_filtered_mask &= (y_vals >= filt['min'])
                if filt.get('max') is not None:
                    y_filtered_mask &= (y_vals <= filt['max'])
            
            # Apply current cutoff lines (temporary filters for current plot)
            x_lines = []
            y_lines = []
            for orientation, line in self.cutoff_lines:
                pos = line.value()
                if orientation == 'vertical':
                    x_lines.append(pos)
                else:  # horizontal
                    y_lines.append(pos)
            
            # Apply line-based filters
            if len(x_lines) == 1:
                x_filtered_mask &= (x_vals >= x_lines[0])
            elif len(x_lines) >= 2:
                x_min_line = min(x_lines)
                x_max_line = max(x_lines)
                x_filtered_mask &= (x_vals >= x_min_line) & (x_vals <= x_max_line)
            
            if len(y_lines) == 1:
                y_filtered_mask &= (y_vals >= y_lines[0])
            elif len(y_lines) >= 2:
                y_min_line = min(y_lines)
                y_max_line = max(y_lines)
                y_filtered_mask &= (y_vals >= y_min_line) & (y_vals <= y_max_line)
            
            # Combined mask
            valid_mask = x_filtered_mask & y_filtered_mask
            _agent_debug_log(
                "update_plot_masks_computed",
                data={
                    "n_points": int(len(x_vals)),
                    "n_valid": int(np.sum(valid_mask)),
                },
                hypothesis_id="H2",
            )

            # Apply box ROI if present (keep points inside the box)
            if self.box_roi is not None:
                try:
                    # Get ROI bounds in data coordinates
                    state = self.box_roi.getState()
                    pos = state['pos']
                    size = state['size']
                    x0 = pos[0]
                    x1 = pos[0] + size[0]
                    y0 = pos[1]
                    y1 = pos[1] + size[1]
                    # Ensure correct min/max
                    x_min_box = min(x0, x1)
                    x_max_box = max(x0, x1)
                    y_min_box = min(y0, y1)
                    y_max_box = max(y0, y1)
                    valid_mask &= (x_vals >= x_min_box) & (x_vals <= x_max_box) & (y_vals >= y_min_box) & (y_vals <= y_max_box)
                except Exception:
                    # Fallback to parentBounds if getState fails
                    bounds = self.box_roi.parentBounds()
                    if bounds is not None:
                        x0 = min(bounds.left(), bounds.right())
                        x1 = max(bounds.left(), bounds.right())
                        y0 = min(bounds.top(), bounds.bottom())
                        y1 = max(bounds.top(), bounds.bottom())
                        valid_mask &= (x_vals >= x0) & (x_vals <= x1) & (y_vals >= y0) & (y_vals <= y1)
            
            # Color code by condition if filenames are available
            filenames = self.current_plot.get('filenames', None)
            self.current_valid_filenames = None
            color_by_condition = self.color_by_condition_check.isChecked() if hasattr(self, 'color_by_condition_check') else True
            
            if filenames is not None and len(filenames) > 0:
                # Consistent, general condition extraction (no hard-coded mutants)
                def get_condition(filename):
                    if pd.isna(filename):
                        return "Unknown"
                    key = extract_condition_key(filename)
                    return key if key else "Unknown"

                # Contrasting palette (tab10-like) for arbitrary numbers of conditions
                # (Cycles if there are more conditions than colors)
                palette = [
                    (31, 119, 180, 255),   # blue
                    (255, 127, 14, 255),   # orange
                    (44, 160, 44, 255),    # green
                    (214, 39, 40, 255),    # red
                    (148, 103, 189, 255),  # purple
                    (140, 86, 75, 255),    # brown
                    (227, 119, 194, 255),  # pink
                    (127, 127, 127, 255),  # gray
                    (188, 189, 34, 255),   # olive
                    (23, 190, 207, 255),   # cyan
                ]
                unknown_color = (170, 170, 170, 255)
                
                # Plot valid points
                if np.any(valid_mask):
                    valid_filenames = filenames[valid_mask]
                    self.current_valid_filenames = set(valid_filenames)
                    valid_x = x_vals[valid_mask]
                    valid_y = y_vals[valid_mask]
                    
                    if color_by_condition:
                        # Group by condition for color coding
                        conditions_data = {}
                        for i, filename in enumerate(valid_filenames):
                            condition = get_condition(filename)
                            if condition not in conditions_data:
                                conditions_data[condition] = {'x': [], 'y': [], 'filenames': []}
                            conditions_data[condition]['x'].append(valid_x[i])
                            conditions_data[condition]['y'].append(valid_y[i])
                            conditions_data[condition]['filenames'].append(filename)
                        
                        # Assign stable colors to discovered conditions
                        known_conditions = sorted([c for c in conditions_data.keys() if c != "Unknown"])
                        condition_colors = {
                            cond: palette[i % len(palette)] for i, cond in enumerate(known_conditions)
                        }
                        condition_colors["Unknown"] = unknown_color

                        # Plot each condition with its color using ScatterPlotItem
                        legend_items = {}
                        ordered_conditions = known_conditions + (["Unknown"] if "Unknown" in conditions_data else [])
                        for condition in ordered_conditions:
                            data = conditions_data[condition]
                            color = condition_colors.get(condition, unknown_color)
                            
                            # Create ScatterPlotItem with hover support
                            scatter = pg.ScatterPlotItem(
                                x=np.array(data['x']),
                                y=np.array(data['y']),
                                pen='w',
                                brush=color,
                                size=8,
                                hoverable=True,
                                tip=None  # We'll handle tooltips manually
                            )
                            # Store filenames in point data
                            point_data = [{'filename': fn} for fn in data['filenames']]
                            scatter.setData(
                                x=np.array(data['x']),
                                y=np.array(data['y']),
                                data=point_data
                            )
                            self.plot_widget.addItem(scatter)
                            self.scatter_items.append(scatter)
                            
                            if condition not in legend_items:
                                legend_items[condition] = scatter
                        
                        # Add legend below plot
                        if legend_items:
                            self._update_legend_below(legend_items)
                    else:
                        # Plot all points in blue without color coding
                        scatter = pg.ScatterPlotItem(
                            x=valid_x,
                            y=valid_y,
                            pen='w',
                            brush=(0, 0, 255, 255),  # Blue
                            size=8,
                            hoverable=True,
                            tip=None
                        )
                        # Store filenames in point data
                        point_data = [{'filename': fn} for fn in valid_filenames]
                        scatter.setData(
                            x=valid_x,
                            y=valid_y,
                            data=point_data
                        )
                        self.plot_widget.addItem(scatter)
                        self.scatter_items.append(scatter)
                        self.scatter_plot_item = scatter
                        
                        # Add simple legend
                        self._update_legend_below({'Data': scatter})
            else:
                # No filenames - plot valid points in blue
                if np.any(valid_mask):
                    scatter = pg.ScatterPlotItem(
                        x=x_vals[valid_mask],
                        y=y_vals[valid_mask],
                        pen='w',
                        brush=(0, 0, 255, 255),  # Blue
                        size=8,
                        hoverable=True,
                        tip=None
                    )
                    # Create dummy filenames
                    dummy_filenames = [f"Point {i+1}" for i in range(np.sum(valid_mask))]
                    point_data = [{'filename': fn} for fn in dummy_filenames]
                    scatter.setData(
                        x=x_vals[valid_mask],
                        y=y_vals[valid_mask],
                        data=point_data
                    )
                    self.plot_widget.addItem(scatter)
                    self.scatter_items.append(scatter)
                    self.scatter_plot_item = scatter
                    
                    # Add simple legend below plot
                    self._update_legend_below({'Data': scatter})
            
            # Plot filtered points in gray if enabled
            if self.show_filtered_check.isChecked() and np.any(~valid_mask):
                filtered_filenames = filenames[~valid_mask] if filenames is not None else None
                if filtered_filenames is None:
                    filtered_filenames = [f"Point {i+1}" for i in range(np.sum(~valid_mask))]
                
                scatter_filtered = pg.ScatterPlotItem(
                    x=x_vals[~valid_mask],
                    y=y_vals[~valid_mask],
                    pen='w',
                    brush=(128, 128, 128, 128),
                    size=6,
                    hoverable=True,
                    tip=None
                )
                point_data = [{'filename': fn} for fn in filtered_filenames]
                scatter_filtered.setData(
                    x=x_vals[~valid_mask],
                    y=y_vals[~valid_mask],
                    data=point_data
                )
                self.plot_widget.addItem(scatter_filtered)
                self.scatter_items.append(scatter_filtered)
        
            # Update labels with white color
            self.plot_widget.setLabel('bottom', x_param, color='white')
            self.plot_widget.setLabel('left', y_param, color='white')
            n_valid = np.sum(valid_mask)
            n_total = len(x_vals)
            self.plot_widget.setTitle(
                f"{self.current_plot['data_type']}: {y_param} vs {x_param} "
                f"({n_valid}/{n_total} points)", color='white'
            )
            
            # Connect hover and click events for tooltips
            try:
                self._setup_hover_handlers()
            except Exception:
                pass
            
            # Update saved filters table to reflect current movie status
            try:
                self.update_saved_filters_table()
            except Exception:
                pass
        except Exception as e:
            _agent_debug_log(
                "update_plot_with_filters_exception",
                data={"error": repr(e), "traceback": traceback.format_exc()},
                hypothesis_id="H2",
            )
    
    def save_current_filter(self):
        """Save filter cutoffs for current parameters, including box ROI bounds."""
        if self.current_plot is None:
            return
        
        x_param = self.current_plot['x_param']
        y_param = self.current_plot['y_param']
        
        # Extract cutoff values from lines
        x_lines = []
        y_lines = []
        
        for orientation, line in self.cutoff_lines:
            pos = line.value()
            if orientation == 'vertical':
                x_lines.append(pos)
            else:  # horizontal
                y_lines.append(pos)
        
        # Determine min/max from lines
        x_min = x_max = None
        y_min = y_max = None
        
        if len(x_lines) == 1:
            # Single line: assume it's a minimum
            x_min = x_lines[0]
        elif len(x_lines) >= 2:
            x_min = min(x_lines)
            x_max = max(x_lines)
        
        if len(y_lines) == 1:
            y_min = y_lines[0]
        elif len(y_lines) >= 2:
            y_min = min(y_lines)
            y_max = max(y_lines)
        
        # Extract box ROI bounds if present
        if self.box_roi is not None:
            try:
                state = self.box_roi.getState()
                pos = state['pos']
                size = state['size']
                x0 = pos[0]
                x1 = pos[0] + size[0]
                y0 = pos[1]
                y1 = pos[1] + size[1]
                box_x_min = min(x0, x1)
                box_x_max = max(x0, x1)
                box_y_min = min(y0, y1)
                box_y_max = max(y0, y1)
                
                # Merge box bounds with line-based filters (box takes precedence if both exist)
                if x_min is None or box_x_min > x_min:
                    x_min = box_x_min
                if x_max is None or box_x_max < x_max:
                    x_max = box_x_max
                if y_min is None or box_y_min > y_min:
                    y_min = box_y_min
                if y_max is None or box_y_max < y_max:
                    y_max = box_y_max
            except Exception:
                # Fallback if getState fails
                bounds = self.box_roi.parentBounds()
                if bounds is not None:
                    box_x_min = min(bounds.left(), bounds.right())
                    box_x_max = max(bounds.left(), bounds.right())
                    box_y_min = min(bounds.top(), bounds.bottom())
                    box_y_max = max(bounds.top(), bounds.bottom())
                    if x_min is None or box_x_min > x_min:
                        x_min = box_x_min
                    if x_max is None or box_x_max < x_max:
                        x_max = box_x_max
                    if y_min is None or box_y_min > y_min:
                        y_min = box_y_min
                    if y_max is None or box_y_max < y_max:
                        y_max = box_y_max
        
        # Save filters (merge with existing if any)
        if x_min is not None or x_max is not None:
            if x_param not in self.saved_filters:
                self.saved_filters[x_param] = {}
            if x_min is not None:
                self.saved_filters[x_param]['min'] = x_min
            if x_max is not None:
                self.saved_filters[x_param]['max'] = x_max
        
        if y_min is not None or y_max is not None:
            if y_param not in self.saved_filters:
                self.saved_filters[y_param] = {}
            if y_min is not None:
                self.saved_filters[y_param]['min'] = y_min
            if y_max is not None:
                self.saved_filters[y_param]['max'] = y_max
        
        # Update table and plot
        self.update_saved_filters_table()
        self.update_plot_with_filters()
        
        # Notify that filters have changed
        self.notify_filters_changed()
    
    def save_filter_config(self):
        """Save filter configuration to JSON file."""
        if not self.saved_filters:
            QMessageBox.warning(self, "Warning", "No filters to save.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Filter Configuration", "", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    json.dump(self.saved_filters, f, indent=2)
                QMessageBox.information(self, "Success", f"Filter configuration saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save filter config: {str(e)}")
    
    def load_filter_config(self):
        """Load filter configuration from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Filter Configuration", "", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'r') as f:
                    self.saved_filters = json.load(f)
                self.update_saved_filters_table()
                if self.current_plot is not None:
                    self.update_plot_with_filters()
                self.notify_filters_changed()
                QMessageBox.information(self, "Success", f"Filter configuration loaded from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load filter config: {str(e)}")
    
    def clear_all_filters(self):
        """Clear all saved filters."""
        reply = QMessageBox.question(
            self, "Confirm", "Clear all saved filters?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.saved_filters = {}
            self.update_saved_filters_table()
            if self.current_plot is not None:
                self.update_plot_with_filters()
            QMessageBox.information(self, "Success", "All filters cleared.")
    
    def _extract_movie_id(self, filename):
        """Extract movie/replicate identifier from filename."""
        if pd.isna(filename):
            return None
        filename_str = str(filename)
        # Try to extract movie identifier - could be folder name, prefix, etc.
        # Common patterns: movie name before extension, folder name, etc.
        basename = os.path.basename(filename_str)
        # Remove extension
        name_no_ext = os.path.splitext(basename)[0]
        # Try to extract meaningful identifier (e.g., everything before first underscore or last part)
        parts = name_no_ext.split('_')
        if len(parts) > 1:
            # Return first few parts as movie identifier
            return '_'.join(parts[:min(3, len(parts))])
        return name_no_ext
    
    def _get_movies_affected_by_filters(self):
        """Calculate which movies are affected by current filters."""
        if self.current_plot is None:
            return {}, {}
        
        filenames = self.current_plot.get('filenames', None)
        if filenames is None or len(filenames) == 0:
            return {}, {}
        
        x_param = self.current_plot['x_param']
        y_param = self.current_plot['y_param']
        x_vals = self.current_plot['x_vals']
        y_vals = self.current_plot['y_vals']
        
        # Group data by movie
        movies_data = {}
        for i, filename in enumerate(filenames):
            movie_id = self._extract_movie_id(filename)
            if movie_id is None:
                movie_id = f"Unknown_{i}"
            if movie_id not in movies_data:
                movies_data[movie_id] = {
                    'x': [], 'y': [], 'filenames': [], 'indices': []
                }
            movies_data[movie_id]['x'].append(x_vals[i])
            movies_data[movie_id]['y'].append(y_vals[i])
            movies_data[movie_id]['filenames'].append(filename)
            movies_data[movie_id]['indices'].append(i)
        
        # Apply filters to determine which movies are affected
        x_filtered_mask = np.ones(len(x_vals), dtype=bool)
        y_filtered_mask = np.ones(len(y_vals), dtype=bool)
        
        # Apply saved filters for X parameter
        if x_param in self.saved_filters:
            filt = self.saved_filters[x_param]
            if filt.get('min') is not None:
                x_filtered_mask &= (x_vals >= filt['min'])
            if filt.get('max') is not None:
                x_filtered_mask &= (x_vals <= filt['max'])
        
        # Apply saved filters for Y parameter
        if y_param in self.saved_filters:
            filt = self.saved_filters[y_param]
            if filt.get('min') is not None:
                y_filtered_mask &= (y_vals >= filt['min'])
            if filt.get('max') is not None:
                y_filtered_mask &= (y_vals <= filt['max'])
        
        valid_mask = x_filtered_mask & y_filtered_mask

        # Apply cutoff lines
        x_lines = []
        y_lines = []
        for orientation, line in self.cutoff_lines:
            pos = line.value()
            if orientation == 'vertical':
                x_lines.append(pos)
            else:
                y_lines.append(pos)

        if len(x_lines) == 1:
            valid_mask &= (x_vals >= x_lines[0])
        elif len(x_lines) >= 2:
            x_min_line = min(x_lines)
            x_max_line = max(x_lines)
            valid_mask &= (x_vals >= x_min_line) & (x_vals <= x_max_line)

        if len(y_lines) == 1:
            valid_mask &= (y_vals >= y_lines[0])
        elif len(y_lines) >= 2:
            y_min_line = min(y_lines)
            y_max_line = max(y_lines)
            valid_mask &= (y_vals >= y_min_line) & (y_vals <= y_max_line)

        # Apply box ROI if present
        if self.box_roi is not None:
            try:
                state = self.box_roi.getState()
                pos = state['pos']
                size = state['size']
                x0 = pos[0]
                x1 = pos[0] + size[0]
                y0 = pos[1]
                y1 = pos[1] + size[1]
                x_min_box = min(x0, x1)
                x_max_box = max(x0, x1)
                y_min_box = min(y0, y1)
                y_max_box = max(y0, y1)
                valid_mask &= (x_vals >= x_min_box) & (x_vals <= x_max_box) & (y_vals >= y_min_box) & (y_vals <= y_max_box)
            except Exception:
                bounds = self.box_roi.parentBounds()
                if bounds is not None:
                    x0 = min(bounds.left(), bounds.right())
                    x1 = max(bounds.left(), bounds.right())
                    y0 = min(bounds.top(), bounds.bottom())
                    y1 = max(bounds.top(), bounds.bottom())
                    valid_mask &= (x_vals >= x0) & (x_vals <= x1) & (y_vals >= y0) & (y_vals <= y1)
        
        # Determine which movies are completely excluded vs partially included
        movies_included = set()
        movies_excluded = set()
        
        for movie_id, data in movies_data.items():
            movie_indices = np.array(data['indices'])
            movie_valid = valid_mask[movie_indices]
            
            if np.any(movie_valid):
                movies_included.add(movie_id)
            else:
                movies_excluded.add(movie_id)
        
        return movies_included, movies_excluded
    
    def update_saved_filters_table(self):
        """Update the saved filters display table - one line per parameter."""
        self.saved_filters_table.setRowCount(len(self.saved_filters))
        
        # Get movies affected by filters
        movies_included, movies_excluded = self._get_movies_affected_by_filters()
        
        included_list = sorted(movies_included)
        
        for i, (param, filters) in enumerate(self.saved_filters.items()):
            self.saved_filters_table.setItem(i, 0, QTableWidgetItem(param))
            min_val = filters.get('min', None)
            max_val = filters.get('max', None)
            
            # Format range string
            if min_val is not None and max_val is not None:
                range_str = f"[{min_val:.6f}, {max_val:.6f}]"
            elif min_val is not None:
                range_str = f"≥ {min_val:.6f}"
            elif max_val is not None:
                range_str = f"≤ {max_val:.6f}"
            else:
                range_str = "None"
            
            self.saved_filters_table.setItem(i, 1, QTableWidgetItem(range_str))
            
            # Show which movies are currently included after filtering
            if included_list:
                movies_str = ", ".join(included_list)
            else:
                movies_str = "None"
            self.saved_filters_table.setItem(i, 2, QTableWidgetItem(movies_str))
        
        # Update movies status text
        self._update_movies_status(movies_included, movies_excluded)
    
    def _count_movies_affected_by_param(self, param, filters):
        """Count how many movies are affected by a specific parameter filter."""
        if self.current_plot is None:
            return 0
        
        filenames = self.current_plot.get('filenames', None)
        if filenames is None or len(filenames) == 0:
            return 0
        
        # Get values for this parameter
        if param == self.current_plot['x_param']:
            vals = self.current_plot['x_vals']
        elif param == self.current_plot['y_param']:
            vals = self.current_plot['y_vals']
        else:
            return 0
        
        # Group by movie
        movies_data = {}
        for i, filename in enumerate(filenames):
            movie_id = self._extract_movie_id(filename)
            if movie_id is None:
                movie_id = f"Unknown_{i}"
            if movie_id not in movies_data:
                movies_data[movie_id] = []
            movies_data[movie_id].append(vals[i])
        
        # Check which movies are affected
        affected_movies = set()
        for movie_id, movie_vals in movies_data.items():
            movie_vals_arr = np.array(movie_vals)
            movie_mask = np.ones(len(movie_vals_arr), dtype=bool)
            
            if filters.get('min') is not None:
                movie_mask &= (movie_vals_arr >= filters['min'])
            if filters.get('max') is not None:
                movie_mask &= (movie_vals_arr <= filters['max'])
            
            # If any point in this movie is filtered out, the movie is affected
            if not np.all(movie_mask):
                affected_movies.add(movie_id)
        
        return len(affected_movies)
    
    def _update_movies_status(self, movies_included, movies_excluded):
        """Update the movies status display."""
        if not movies_included and not movies_excluded:
            self.movies_status_text.setText("No movie data available.")
            return
        
        status_lines = []
        status_lines.append(f"Total movies: {len(movies_included) + len(movies_excluded)}")
        status_lines.append(f"Included: {len(movies_included)}")
        if movies_included:
            status_lines.append("Included movies:")
            for movie in sorted(movies_included):
                status_lines.append(f"  - {movie}")
        status_lines.append(f"Excluded: {len(movies_excluded)}")
        
        if movies_excluded:
            status_lines.append("Excluded movies:")
            for movie in sorted(movies_excluded):
                status_lines.append(f"  - {movie}")
        
        self.movies_status_text.setText("\n".join(status_lines))

    def _create_tooltip_label(self):
        """Create a tooltip label for displaying point information on hover."""
        if self.hover_tooltip is None:
            self.hover_tooltip = QLabel(self.plot_widget)
            self.hover_tooltip.setStyleSheet("""
                QLabel {
                    background-color: rgba(255, 255, 255, 240);
                    border: 2px solid #333;
                    border-radius: 4px;
                    padding: 6px;
                    font-size: 11px;
                    color: black;
                    font-weight: bold;
                }
            """)
            self.hover_tooltip.hide()
            self.hover_tooltip.setWordWrap(True)
            self.hover_tooltip.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        return self.hover_tooltip
    
    def _setup_hover_handlers(self):
        """Set up hover and click handlers for scatter plot items (once per plot session).

        The mouse-move signal is routed through a pyqtgraph ``SignalProxy`` that rate-limits
        it to ~60 Hz. Connecting a heavy handler directly to ``sigMouseMoved`` (which fires
        on every pixel of motion) is what made the Filtering tab crash: rapid re-entrant
        callbacks that touched scatter items being rebuilt/deleted would dereference freed
        C++ objects and segfault. Throttling + the guards in the handlers fix that.
        """
        if not PYQTGRAPH_AVAILABLE or self.plot_widget is None:
            return
        if self._plot_handlers_connected:
            return
        try:
            scene = self.plot_widget.scene()
            if scene is not None:
                # Keep references on self so the proxies aren't garbage-collected.
                self._mouse_move_proxy = pg.SignalProxy(
                    scene.sigMouseMoved, rateLimit=60, slot=self._on_plot_hover)
                self._mouse_click_proxy = pg.SignalProxy(
                    scene.sigMouseClicked, rateLimit=60, slot=self._on_plot_click)
                self._plot_handlers_connected = True
        except Exception:
            pass  # Silently fail if scene not available

    def _live_scatter_items(self):
        """Return only scatter items whose underlying C++ object is still alive and still
        attached to the current scene. Guards against segfaults from stale references left
        behind when the plot is cleared/rebuilt."""
        live = []
        for item in list(self.scatter_items):
            try:
                if item is not None and item.scene() is not None:
                    live.append(item)
            except (RuntimeError, Exception):
                # RuntimeError: wrapped C/C++ object has been deleted -> drop it.
                continue
        return live
    
    @staticmethod
    def _scene_pos_from_event(evt):
        """Extract a scene position from either a raw signal arg or a SignalProxy tuple."""
        # SignalProxy delivers the signal args wrapped in a tuple: (pos,) for sigMouseMoved.
        if isinstance(evt, (tuple, list)):
            evt = evt[0] if evt else None
        return evt

    def _on_plot_hover(self, evt):
        """Handle mouse hover to show a point tooltip. Fully guarded: any access to a
        pyqtgraph item that was deleted during a plot rebuild raises RuntimeError, which we
        swallow so the app can never crash on hover."""
        if self._in_hover:
            return  # Drop re-entrant calls (belt-and-suspenders with the SignalProxy).
        self._in_hover = True
        try:
            if (not PYQTGRAPH_AVAILABLE or self.plot_widget is None
                    or self.current_plot is None):
                self._hide_tooltip()
                return

            live_items = self._live_scatter_items()
            if not live_items:
                self._hide_tooltip()
                return

            pos = self._scene_pos_from_event(evt)
            if pos is None:
                self._hide_tooltip()
                return

            vb = self.plot_widget.plotItem.vb
            view_pos = vb.mapSceneToView(pos)
            # Compute the hit threshold once (not per point).
            x_range, y_range = self.plot_widget.viewRange()
            x_threshold = (x_range[1] - x_range[0]) * 0.03
            y_threshold = (y_range[1] - y_range[0]) * 0.03

            x_param = self.current_plot['x_param']
            y_param = self.current_plot['y_param']

            min_dist = float('inf')
            closest_info = None
            for scatter_item in live_items:
                try:
                    points = scatter_item.points()
                except (RuntimeError, Exception):
                    continue
                if points is None or len(points) == 0:
                    continue
                for point in points:
                    try:
                        px, py = point.pos()
                    except (RuntimeError, Exception):
                        continue
                    if (abs(view_pos.x() - px) < x_threshold
                            and abs(view_pos.y() - py) < y_threshold):
                        dist = ((view_pos.x() - px) ** 2 + (view_pos.y() - py) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            pdata = point.data()
                            if pdata and 'filename' in pdata:
                                closest_info = {'filename': pdata['filename'],
                                                'x': px, 'y': py}

            tooltip = self._create_tooltip_label()
            if closest_info is not None:
                tooltip.setText(
                    f"{closest_info['filename']}\n"
                    f"{x_param}: {closest_info['x']:.4f}\n"
                    f"{y_param}: {closest_info['y']:.4f}")
                cursor_pos = self.plot_widget.mapFromGlobal(
                    QApplication.instance().cursor().pos())
                tooltip.move(cursor_pos.x() + 15, cursor_pos.y() + 15)
                tooltip.show()
                tooltip.raise_()
            else:
                tooltip.hide()
        except (RuntimeError, Exception):
            # Deleted C++ object or any transient plot-state error: never crash on hover.
            self._hide_tooltip()
        finally:
            self._in_hover = False

    def _hide_tooltip(self):
        """Hide the hover tooltip if it exists, ignoring any teardown errors."""
        try:
            if self.hover_tooltip is not None:
                self.hover_tooltip.hide()
        except (RuntimeError, Exception):
            pass
    
    def _on_plot_click(self, evt):
        """Handle a mouse click to show point info. Guarded the same way as the hover
        handler so a click during a plot rebuild can't dereference a deleted item."""
        try:
            if (not PYQTGRAPH_AVAILABLE or self.plot_widget is None
                    or self.current_plot is None):
                return
            live_items = self._live_scatter_items()
            if not live_items:
                return

            # SignalProxy wraps the args in a tuple: (MouseClickEvent,).
            event = evt[0] if isinstance(evt, (tuple, list)) and evt else evt
            try:
                scene_pos = event.scenePos()
            except (AttributeError, RuntimeError, Exception):
                return

            vb = self.plot_widget.plotItem.vb
            view_pos = vb.mapSceneToView(scene_pos)
            x_range, y_range = self.plot_widget.viewRange()
            x_threshold = (x_range[1] - x_range[0]) * 0.03
            y_threshold = (y_range[1] - y_range[0]) * 0.03

            x_param = self.current_plot['x_param']
            y_param = self.current_plot['y_param']

            min_dist = float('inf')
            closest_info = None
            for scatter_item in live_items:
                try:
                    points = scatter_item.points()
                except (RuntimeError, Exception):
                    continue
                if points is None or len(points) == 0:
                    continue
                for point in points:
                    try:
                        px, py = point.pos()
                    except (RuntimeError, Exception):
                        continue
                    if (abs(view_pos.x() - px) < x_threshold
                            and abs(view_pos.y() - py) < y_threshold):
                        dist = ((view_pos.x() - px) ** 2 + (view_pos.y() - py) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            pdata = point.data()
                            if pdata and 'filename' in pdata:
                                closest_info = {'filename': pdata['filename'],
                                                'x': px, 'y': py}

            if closest_info is not None:
                QMessageBox.information(
                    self, "Point Details",
                    f"Point Information:\n\n"
                    f"Filename: {closest_info['filename']}\n"
                    f"{x_param}: {closest_info['x']:.6f}\n"
                    f"{y_param}: {closest_info['y']:.6f}")
        except (RuntimeError, Exception):
            pass  # Never crash on click.
    
    def get_included_filenames(self):
        """Return set of filenames that survived current filters (if available)."""
        if self.current_valid_filenames is not None:
            return set(self.current_valid_filenames)
        return None


class CollapsibleSection(QWidget):
    """A collapsible section widget for organizing parameters."""
    
    def __init__(self, title, parent=None, initially_collapsed=True):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not initially_collapsed)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #e8e8e8;
                font-weight: bold;
                color: #000000;
            }
            QPushButton:checked {
                background-color: #d0d0d0;
                color: #000000;
            }
            QPushButton:hover {
                background-color: #d8d8d8;
                color: #000000;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_content)
        
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(10, 5, 10, 5)
        self.content_area.setLayout(self.content_layout)
        self.content_area.setVisible(not initially_collapsed)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)
        
        self._update_button_text()
    
    def toggle_content(self):
        self.content_area.setVisible(self.toggle_button.isChecked())
        self._update_button_text()
    
    def _update_button_text(self):
        text = self.toggle_button.text()
        if text.startswith("▼ ") or text.startswith("▶ "):
            text = text[2:]
        if self.toggle_button.isChecked():
            self.toggle_button.setText("▼ " + text)
        else:
            self.toggle_button.setText("▶ " + text)
    
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


class AnalysisTab(QWidget):
    """Tab for running FRAP curve combination analysis."""
    
    def __init__(self, main_window=None):
        super().__init__()
        self.worker = None
        self.crop_worker = None
        self.main_window = main_window  # Reference to main window to access filtering tab
        self.init_ui()
    
    def init_ui(self):
        # Use a scroll area for all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Input directory
        input_group = QGroupBox("Input Directory")
        input_layout = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.clicked.connect(self.browse_input_dir)
        input_layout.addWidget(QLabel("Directory:"))
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(input_browse_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # ============ IMAGING PARAMETERS ============
        imaging_section = CollapsibleSection("Imaging Parameters", initially_collapsed=False)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Pixel Size (µm):"))
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.001, 10.0)
        self.pixel_size_spin.setSingleStep(0.01)
        self.pixel_size_spin.setValue(0.11)
        self.pixel_size_spin.setDecimals(3)
        self.pixel_size_spin.setToolTip("Physical size of one pixel in micrometers")
        row1.addWidget(self.pixel_size_spin)

        row1.addWidget(QLabel("Frame Interval (s):"))
        self.frame_interval_spin = QDoubleSpinBox()
        self.frame_interval_spin.setRange(0.1, 1000.0)
        self.frame_interval_spin.setSingleStep(1.0)
        self.frame_interval_spin.setValue(10.0)
        self.frame_interval_spin.setDecimals(1)
        self.frame_interval_spin.setToolTip("Time between frames in seconds")
        row1.addWidget(self.frame_interval_spin)

        row1.addWidget(QLabel("Pre-bleach frames:"))
        self.pre_bleach_frames_spin = QSpinBox()
        self.pre_bleach_frames_spin.setRange(1, 50)
        self.pre_bleach_frames_spin.setValue(1)
        self.pre_bleach_frames_spin.setToolTip(
            "Number of pre-bleach frames (averaged for the baseline). The bleach event is "
            "taken to occur between frame N-1 and frame N. Default 1."
        )
        row1.addWidget(self.pre_bleach_frames_spin)
        row1.addStretch()
        imaging_section.addLayout(row1)

        row_ch = QHBoxLayout()
        row_ch.addWidget(QLabel("Analysis channel:"))
        self.analysis_channel_combo = QComboBox()
        self.analysis_channel_combo.addItems(["Channel 1", "Channel 2"])
        self.analysis_channel_combo.setToolTip(
            "The channel used to detect droplets and build the bleach/droplet ROIs "
            "(1-based). Single-channel movies always use channel 1."
        )
        row_ch.addWidget(self.analysis_channel_combo)

        self.two_channel_check = QCheckBox("Two-channel analysis")
        self.two_channel_check.setChecked(False)
        self.two_channel_check.setToolTip(
            "Also analyze the other channel(s) of 2-channel movies, measured with the "
            "analysis-channel ROIs (outputs suffixed _ch1/_ch2). Mixed 1-/2-channel "
            "datasets are handled automatically."
        )
        row_ch.addWidget(self.two_channel_check)
        row_ch.addStretch()
        imaging_section.addLayout(row_ch)

        # Timing segments (moved here from the dissolved Droplet Cropping section).
        row_timing = QHBoxLayout()
        row_timing.addWidget(QLabel("Timing segments:"))
        self.timing_segments_edit = QLineEdit()
        self.timing_segments_edit.setPlaceholderText("e.g. 300x0.1,35x10  (optional)")
        self.timing_segments_edit.setToolTip(
            "Variable acquisition cadence used when the movie has no timing metadata: "
            "COUNTxINTERVAL segments, e.g. '300x0.1,35x10' = 300 frames at 0.1 s then 35 "
            "at 10 s. Leave blank to use metadata or the constant Frame Interval."
        )
        row_timing.addWidget(self.timing_segments_edit, 1)
        imaging_section.addLayout(row_timing)

        layout.addWidget(imaging_section)

        # ============ IMAGE STABILIZATION ============
        stab_section = CollapsibleSection("Image Stabilization", initially_collapsed=True)

        stab_row_en = QHBoxLayout()
        self.stabilize_check = QCheckBox("Enable Stabilization")
        self.stabilize_check.setChecked(True)  # Default ON
        self.stabilize_check.setToolTip("Stabilize entire movie before cropping (drift correction)")
        stab_row_en.addWidget(self.stabilize_check)
        stab_row_en.addStretch()
        stab_section.addLayout(stab_row_en)

        # Frame-range stabilization (moved here, ABOVE the algorithm). The defaults
        # [start=0, end=-1, extrapolate=hold] cover the whole stack, so the pipeline defers
        # to the traditional per-frame (full-range) stabilization; narrow the window only to
        # estimate drift on a sub-range.
        srange_layout = QHBoxLayout()
        srange_layout.addWidget(QLabel("Stabilize frames [start:"))
        self.stab_frame_start_spin = QSpinBox()
        self.stab_frame_start_spin.setRange(0, 1000000)
        self.stab_frame_start_spin.setValue(0)
        self.stab_frame_start_spin.setToolTip("First frame of the drift-estimation window.")
        srange_layout.addWidget(self.stab_frame_start_spin)
        srange_layout.addWidget(QLabel("end:"))
        self.stab_frame_end_spin = QSpinBox()
        self.stab_frame_end_spin.setRange(-1, 1000000)
        self.stab_frame_end_spin.setValue(-1)
        self.stab_frame_end_spin.setToolTip(
            "Last frame of the drift-estimation window (-1 = last). Drift is estimated on "
            "frame 0 + [start,end] and applied to the whole stack, aligned to frame 0 (no "
            "tail jump). Defaults (0, -1) = traditional whole-stack stabilization."
        )
        srange_layout.addWidget(self.stab_frame_end_spin)
        srange_layout.addWidget(QLabel("] extrapolate:"))
        self.stab_extrapolation_combo = QComboBox()
        self.stab_extrapolation_combo.addItems(["hold", "linear", "poly"])
        self.stab_extrapolation_combo.setToolTip(
            "How to fill shifts outside the window: hold (jump-free), linear (end slope), "
            "or poly (low-order drift model). Only used when the window is a sub-range."
        )
        srange_layout.addWidget(self.stab_extrapolation_combo)
        srange_layout.addStretch()
        stab_section.addLayout(srange_layout)

        stab_row1 = QHBoxLayout()
        stab_row1.addWidget(QLabel("Algorithm:"))
        self.stabilize_algorithm_combo = QComboBox()
        self.stabilize_algorithm_combo.addItems(["Gradient (recommended)", "ECC (OpenCV)"])
        self.stabilize_algorithm_combo.setToolTip(
            "Gradient: registers each frame to the pre-bleach frame in the gradient-magnitude "
            "domain (robust to bleaching), excludes FRAP droplets from the estimate, and applies "
            "a single subpixel shift to all channels. ECC is an intensity-correlation alternative."
        )
        stab_row1.addWidget(self.stabilize_algorithm_combo)
        stab_row1.addStretch()
        stab_section.addLayout(stab_row1)

        # FRAP normalization + photobleach correction (photobleach moved here from fitting).
        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("FRAP normalization:"))
        self.normalization_method_combo = QComboBox()
        self.normalization_method_combo.addItems([
            "Both (pre-bleach + whole-droplet)",
            "Pre-bleach frame",
            "Whole-droplet ratio",
        ])
        self.normalization_method_combo.setToolTip(
            "Pre-bleach frame: bleach spot / its own pre-bleach value.\n"
            "Whole-droplet ratio: (bleach / whole droplet), pre-bleach-normalized.\n"
            "Both normalizations and the raw RawIntDen values are always saved either way."
        )
        norm_row.addWidget(self.normalization_method_combo)
        self.no_photobleach_check = QCheckBox("No Photobleach Correction")
        self.no_photobleach_check.setToolTip("Skip photobleach correction")
        norm_row.addWidget(self.no_photobleach_check)
        norm_row.addStretch()
        stab_section.addLayout(norm_row)

        stab_row2 = QHBoxLayout()
        stab_row2.addWidget(QLabel("Pyramid Levels:"))
        self.stab_pyramid_spin = QSpinBox()
        self.stab_pyramid_spin.setRange(1, 10)
        self.stab_pyramid_spin.setValue(4)
        self.stab_pyramid_spin.setToolTip("Number of pyramid levels for multi-scale stabilization")
        stab_row2.addWidget(self.stab_pyramid_spin)

        stab_row2.addWidget(QLabel("Max Iterations:"))
        self.stab_iterations_spin = QSpinBox()
        self.stab_iterations_spin.setRange(10, 5000)
        self.stab_iterations_spin.setValue(500)
        self.stab_iterations_spin.setToolTip("Maximum iterations per frame")
        stab_row2.addWidget(self.stab_iterations_spin)

        stab_row2.addWidget(QLabel("Error Tolerance:"))
        self.stab_epsilon_spin = QDoubleSpinBox()
        self.stab_epsilon_spin.setRange(1e-12, 1e-4)
        self.stab_epsilon_spin.setValue(1e-7)
        self.stab_epsilon_spin.setDecimals(10)
        self.stab_epsilon_spin.setSingleStep(1e-9)
        self.stab_epsilon_spin.setToolTip("Convergence threshold (lower = stricter)")
        stab_row2.addWidget(self.stab_epsilon_spin)

        stab_row2.addWidget(QLabel("Template Update:"))
        self.stab_template_coeff_spin = QDoubleSpinBox()
        self.stab_template_coeff_spin.setRange(0.0, 1.0)
        self.stab_template_coeff_spin.setValue(0.95)
        self.stab_template_coeff_spin.setDecimals(3)
        self.stab_template_coeff_spin.setToolTip(
            "Template update coefficient (0=fixed template, 1=full update). Fiji macro default: 0.999."
        )
        stab_row2.addWidget(self.stab_template_coeff_spin)
        stab_row2.addStretch()
        stab_section.addLayout(stab_row2)

        stab_row3 = QHBoxLayout()
        stab_row3.addWidget(QLabel("Pre-smooth Sigma:"))
        self.stab_pre_smooth_spin = QDoubleSpinBox()
        self.stab_pre_smooth_spin.setRange(0.0, 5.0)
        self.stab_pre_smooth_spin.setValue(1.0)
        self.stab_pre_smooth_spin.setDecimals(2)
        self.stab_pre_smooth_spin.setToolTip("Gaussian smoothing before alignment (0 = none)")
        stab_row3.addWidget(self.stab_pre_smooth_spin)

        stab_row3.addWidget(QLabel("Max Shift (px):"))
        self.stab_max_shift_spin = QDoubleSpinBox()
        self.stab_max_shift_spin.setRange(1.0, 100.0)
        self.stab_max_shift_spin.setValue(25.0)
        self.stab_max_shift_spin.setDecimals(1)
        self.stab_max_shift_spin.setToolTip("Maximum allowed shift in pixels")
        stab_row3.addWidget(self.stab_max_shift_spin)

        stab_row3.addWidget(QLabel("Min Correlation:"))
        self.stab_min_corr_spin = QDoubleSpinBox()
        self.stab_min_corr_spin.setRange(0.0, 1.0)
        self.stab_min_corr_spin.setValue(0.8)
        self.stab_min_corr_spin.setDecimals(2)
        self.stab_min_corr_spin.setToolTip("Minimum correlation to accept alignment (ECC)")
        stab_row3.addWidget(self.stab_min_corr_spin)

        stab_row3.addWidget(QLabel("Refine Iterations:"))
        self.stab_refine_spin = QSpinBox()
        self.stab_refine_spin.setRange(1, 100)
        self.stab_refine_spin.setValue(20)
        self.stab_refine_spin.setToolTip("Refinement passes at full resolution (ECC)")
        stab_row3.addWidget(self.stab_refine_spin)
        stab_row3.addStretch()
        stab_section.addLayout(stab_row3)

        layout.addWidget(stab_section)

        # ============ DETECTION AND FILTERING ============
        filter_section = CollapsibleSection("Detection and Filtering", initially_collapsed=False)

        # Detection sizes (min droplet area = same for FRAP and NoBleach).
        det_layout = QHBoxLayout()
        det_layout.addWidget(QLabel("Minimum droplet area (px):"))
        self.crop_min_area_spin = QDoubleSpinBox()
        self.crop_min_area_spin.setRange(1.0, 10000.0)
        self.crop_min_area_spin.setValue(50.0)
        self.crop_min_area_spin.setToolTip(
            "Minimum droplet area (px) for detection on the pre-bleach frame. Applies to "
            "both FRAP and NoBleach droplets."
        )
        det_layout.addWidget(self.crop_min_area_spin)

        det_layout.addWidget(QLabel("Min mask size (px):"))
        self.droplet_min_size_spin = QSpinBox()
        self.droplet_min_size_spin.setRange(1, 10000)
        self.droplet_min_size_spin.setValue(50)
        self.droplet_min_size_spin.setToolTip(
            "Minimum droplet size (px) for the per-crop Otsu droplet mask used in the "
            "fitting stage."
        )
        det_layout.addWidget(self.droplet_min_size_spin)
        det_layout.addStretch()
        filter_section.addLayout(det_layout)

        # Bleach detection: single merged "Bleach depth" control + spot min area. The same
        # depth defines the contiguous bleach ROI whose pixels are measured for recovery.
        bleach_layout = QHBoxLayout()
        bleach_layout.addWidget(QLabel("Bleach depth:"))
        self.crop_min_bleach_depth_spin = QDoubleSpinBox()
        self.crop_min_bleach_depth_spin.setRange(0.0, 1.0)
        self.crop_min_bleach_depth_spin.setSingleStep(0.05)
        self.crop_min_bleach_depth_spin.setDecimals(2)
        self.crop_min_bleach_depth_spin.setValue(0.3)
        self.crop_min_bleach_depth_spin.setToolTip(
            "Minimum fractional intensity drop (1 - post/pre) that defines a bleach spot. A "
            "droplet carrying such a spot is FRAP; the same threshold defines the contiguous "
            "bleach ROI whose pixels are measured. Replaces the old bleach-threshold + "
            "ratio-percentile controls."
        )
        bleach_layout.addWidget(self.crop_min_bleach_depth_spin)

        bleach_layout.addWidget(QLabel("Bleach spot min area (px):"))
        self.crop_bleach_spot_min_area_spin = QDoubleSpinBox()
        self.crop_bleach_spot_min_area_spin.setRange(0.0, 100000.0)
        self.crop_bleach_spot_min_area_spin.setValue(5.0)
        self.crop_bleach_spot_min_area_spin.setToolTip(
            "Minimum area (px) of a bleach spot in the pre/post ratio image; a droplet "
            "overlapping a spot is FRAP, else NoBleach."
        )
        bleach_layout.addWidget(self.crop_bleach_spot_min_area_spin)
        bleach_layout.addStretch()
        filter_section.addLayout(bleach_layout)

        # General droplet quality filters (apply to ALL droplets: FRAP + NoBleach).
        pc_layout = QHBoxLayout()
        pc_layout.addWidget(QLabel("Minimum partition coefficient:"))
        self.crop_min_nb_pc_spin = QDoubleSpinBox()
        self.crop_min_nb_pc_spin.setRange(0.0, 1000.0)
        self.crop_min_nb_pc_spin.setValue(0.0)
        self.crop_min_nb_pc_spin.setToolTip(
            "Minimum partition coefficient (droplet/background) for ALL droplets (FRAP and "
            "NoBleach). 0 keeps everything."
        )
        pc_layout.addWidget(self.crop_min_nb_pc_spin)

        pc_layout.addWidget(QLabel("Minimum droplet diameter (px):"))
        self.crop_min_nb_diam_spin = QDoubleSpinBox()
        self.crop_min_nb_diam_spin.setRange(0.0, 1000.0)
        self.crop_min_nb_diam_spin.setValue(0.0)
        self.crop_min_nb_diam_spin.setToolTip(
            "Minimum equivalent diameter (px) for ALL droplets (FRAP and NoBleach). "
            "0 keeps everything."
        )
        pc_layout.addWidget(self.crop_min_nb_diam_spin)
        pc_layout.addStretch()
        filter_section.addLayout(pc_layout)

        # Save / QC options.
        save_layout = QHBoxLayout()
        self.crop_clear_borders_check = QCheckBox("Clear Borders")
        self.crop_clear_borders_check.setChecked(False)
        self.crop_clear_borders_check.setToolTip(
            "Remove droplets whose body touches the image edge. OFF by default so large "
            "and edge-touching droplets are kept (flagged via touches_image_edge instead)."
        )
        save_layout.addWidget(self.crop_clear_borders_check)

        self.run_droplet_crop_check = QCheckBox("Save droplet TIF stacks")
        self.run_droplet_crop_check.setToolTip("Save individual droplet TIF stacks to droplet_stacks/ subfolder")
        save_layout.addWidget(self.run_droplet_crop_check)

        self.save_threshold_qc_check = QCheckBox("Save threshold QC stacks")
        self.save_threshold_qc_check.setChecked(False)
        self.save_threshold_qc_check.setToolTip(
            "Save a whole-field-of-view threshold check per movie "
            "(threshold_masks_<movie>.tif: detection 8-bit + droplet mask + bleach-spot "
            "mask). Distinct from the per-droplet FRAP_droplet_masks.tif."
        )
        save_layout.addWidget(self.save_threshold_qc_check)
        save_layout.addStretch()
        filter_section.addLayout(save_layout)

        # Curve-acceptance filters (arithmetic thresholds on the recovery curves).
        drop_layout = QHBoxLayout()
        drop_layout.addWidget(QLabel("Max Drop:"))
        self.max_drop_spin = QDoubleSpinBox()
        self.max_drop_spin.setRange(0.0, 1.0)
        self.max_drop_spin.setSingleStep(0.01)
        self.max_drop_spin.setValue(0.1)
        self.max_drop_spin.setDecimals(3)
        self.max_drop_spin.setToolTip("Maximum intensity drop between frames to accept a curve")
        drop_layout.addWidget(self.max_drop_spin)

        drop_layout.addWidget(QLabel("Max Jump:"))
        self.max_jump_spin = QDoubleSpinBox()
        self.max_jump_spin.setRange(0.0, 1.0)
        self.max_jump_spin.setSingleStep(0.01)
        self.max_jump_spin.setValue(0.1)
        self.max_jump_spin.setDecimals(3)
        self.max_jump_spin.setToolTip("Maximum intensity jump between frames to accept a curve")
        drop_layout.addWidget(self.max_jump_spin)

        drop_layout.addWidget(QLabel("Ignore First N:"))
        self.ignore_first_n_spin = QSpinBox()
        self.ignore_first_n_spin.setRange(0, 100)
        self.ignore_first_n_spin.setValue(4)
        self.ignore_first_n_spin.setToolTip("Frames to ignore in jump/drop analysis")
        drop_layout.addWidget(self.ignore_first_n_spin)
        drop_layout.addStretch()
        filter_section.addLayout(drop_layout)

        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Min Value:"))
        self.min_val_spin = QDoubleSpinBox()
        self.min_val_spin.setRange(-10.0, 10.0)
        self.min_val_spin.setSingleStep(0.1)
        self.min_val_spin.setValue(0.1)
        self.min_val_spin.setDecimals(3)
        self.min_val_spin.setToolTip("Minimum allowed intensity value")
        val_layout.addWidget(self.min_val_spin)

        val_layout.addWidget(QLabel("Max Value:"))
        self.max_val_spin = QDoubleSpinBox()
        self.max_val_spin.setRange(-10.0, 10.0)
        self.max_val_spin.setSingleStep(0.1)
        self.max_val_spin.setValue(1.2)
        self.max_val_spin.setDecimals(3)
        self.max_val_spin.setToolTip("Maximum allowed intensity value")
        val_layout.addWidget(self.max_val_spin)

        self.no_filter_check = QCheckBox("Skip detection thresholds")
        self.no_filter_check.setToolTip("Include all curves without applying detection thresholds")
        val_layout.addWidget(self.no_filter_check)
        val_layout.addStretch()
        filter_section.addLayout(val_layout)

        # Retired controls kept instantiated (not shown) for backward-compatible wiring:
        #   crop_bleach_thresh_spin -> fixed ratio floor; the bleach call is driven by Bleach depth.
        #   crop_multiplier_spin    -> fixed crop-window multiplier (removed from the UI).
        #   ratio_percentile_spin   -> legacy bleach-ROI selector (superseded by Bleach depth).
        self.crop_bleach_thresh_spin = QDoubleSpinBox()
        self.crop_bleach_thresh_spin.setRange(0.1, 10.0)
        self.crop_bleach_thresh_spin.setValue(1.0)
        self.crop_multiplier_spin = QDoubleSpinBox()
        self.crop_multiplier_spin.setRange(1.0, 10.0)
        self.crop_multiplier_spin.setValue(2.0)
        self.ratio_percentile_spin = QDoubleSpinBox()
        self.ratio_percentile_spin.setRange(0.0, 100.0)
        self.ratio_percentile_spin.setValue(85.0)

        layout.addWidget(filter_section)

        # ============ CURVE FITTING ============
        pde_section = CollapsibleSection("Curve Fitting", initially_collapsed=True)

        # Diffusion (PDE) search bounds. These bound the PDE-CALCULATED diffusion
        # coefficient D; they have no effect when PDE fitting is disabled.
        d_layout = QHBoxLayout()
        d_layout.addWidget(QLabel("PDE D Min (µm²/s):"))
        self.d_min_spin = QDoubleSpinBox()
        self.d_min_spin.setRange(0.0001, 10.0)
        self.d_min_spin.setSingleStep(0.001)
        self.d_min_spin.setValue(0.001)
        self.d_min_spin.setDecimals(4)
        self.d_min_spin.setToolTip("Minimum PDE-calculated diffusion coefficient to search")
        d_layout.addWidget(self.d_min_spin)

        d_layout.addWidget(QLabel("PDE D Max (µm²/s):"))
        self.d_max_spin = QDoubleSpinBox()
        self.d_max_spin.setRange(0.001, 100.0)
        self.d_max_spin.setSingleStep(0.1)
        self.d_max_spin.setValue(2.0)
        self.d_max_spin.setDecimals(3)
        self.d_max_spin.setToolTip("Maximum PDE-calculated diffusion coefficient to search")
        d_layout.addWidget(self.d_max_spin)
        d_layout.addStretch()
        pde_section.addLayout(d_layout)

        d_grid_layout = QHBoxLayout()
        d_grid_layout.addWidget(QLabel("D Grid Size:"))
        self.d_grid_size_spin = QSpinBox()
        self.d_grid_size_spin.setRange(5, 200)
        self.d_grid_size_spin.setValue(25)
        self.d_grid_size_spin.setToolTip("Points in coarse D grid (lower = faster)")
        d_grid_layout.addWidget(self.d_grid_size_spin)

        d_grid_layout.addWidget(QLabel("D Refine Grid:"))
        self.d_refine_grid_spin = QSpinBox()
        self.d_refine_grid_spin.setRange(10, 500)
        self.d_refine_grid_spin.setValue(100)
        self.d_refine_grid_spin.setToolTip("Points in refined D grid (lower = faster)")
        d_grid_layout.addWidget(self.d_refine_grid_spin)

        d_grid_layout.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 10)
        self.downsample_spin.setValue(2)
        self.downsample_spin.setToolTip("Downsample factor for PDE grid (higher = faster)")
        d_grid_layout.addWidget(self.downsample_spin)
        d_grid_layout.addStretch()
        pde_section.addLayout(d_grid_layout)

        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("Max Frames:"))
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(-1, 1000)
        self.max_frames_spin.setValue(-1)
        self.max_frames_spin.setToolTip(
            "Upper bound on post-bleach frames for fitting. -1 (default) = use the whole "
            "movie (last frame)."
        )
        mask_layout.addWidget(self.max_frames_spin)
        mask_layout.addStretch()
        pde_section.addLayout(mask_layout)

        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("PDE Frames:"))
        self.pde_first_frame_spin = QSpinBox()
        self.pde_first_frame_spin.setRange(1, 100)
        self.pde_first_frame_spin.setValue(1)
        self.pde_first_frame_spin.setToolTip("First post-bleach frame for PDE")
        frame_layout.addWidget(self.pde_first_frame_spin)
        frame_layout.addWidget(QLabel("-"))
        self.pde_last_frame_spin = QSpinBox()
        self.pde_last_frame_spin.setRange(1, 100)
        self.pde_last_frame_spin.setValue(7)
        self.pde_last_frame_spin.setToolTip("Last frame for PDE")
        frame_layout.addWidget(self.pde_last_frame_spin)

        frame_layout.addWidget(QLabel("Exp Frames:"))
        self.exp_first_frame_spin = QSpinBox()
        self.exp_first_frame_spin.setRange(1, 100)
        self.exp_first_frame_spin.setValue(8)
        self.exp_first_frame_spin.setToolTip("First frame for exponential fit (mixed model only)")
        frame_layout.addWidget(self.exp_first_frame_spin)
        frame_layout.addWidget(QLabel("-"))
        self.exp_last_frame_spin = QSpinBox()
        self.exp_last_frame_spin.setRange(1, 100)
        self.exp_last_frame_spin.setValue(30)
        self.exp_last_frame_spin.setToolTip("Last frame for exponential fit (mixed model only)")
        frame_layout.addWidget(self.exp_last_frame_spin)
        frame_layout.addStretch()
        pde_section.addLayout(frame_layout)

        model_layout = QHBoxLayout()
        self.pde_fitting_check = QCheckBox("Enable PDE Fitting")
        self.pde_fitting_check.setChecked(True)  # Default ON
        self.pde_fitting_check.setToolTip("Enable PDE diffusion fitting. Uncheck to use only single exponential fits.")
        self.pde_fitting_check.stateChanged.connect(self._on_pde_fitting_changed)
        model_layout.addWidget(self.pde_fitting_check)

        self.mixed_model_check = QCheckBox("Mixed Model")
        self.mixed_model_check.setChecked(False)  # Default OFF
        self.mixed_model_check.setToolTip(
            "Fit the combined PDE + constrained-exponential (mixed) model. "
            "Requires PDE fitting to be enabled; used only when checked."
        )
        model_layout.addWidget(self.mixed_model_check)

        self.no_exp_plateau_check = QCheckBox("No Exp Plateau")
        self.no_exp_plateau_check.setToolTip("Skip exponential plateau fitting")
        model_layout.addWidget(self.no_exp_plateau_check)
        model_layout.addStretch()
        pde_section.addLayout(model_layout)

        # Retired here (moved to Image Stabilization / next to Run buttons) but kept
        # instantiated for backward-compatible wiring:
        #   n_jobs_conditions_spin / crop_n_jobs_spin -> driven by the single Process number.
        self.n_jobs_conditions_spin = QSpinBox()
        self.n_jobs_conditions_spin.setRange(1, 64)
        self.n_jobs_conditions_spin.setValue(8)
        self.crop_n_jobs_spin = QSpinBox()
        self.crop_n_jobs_spin.setRange(1, 64)
        self.crop_n_jobs_spin.setValue(8)

        # Diagnostic-plot save toggles (apply to per-movie and combined outputs).
        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Save plots:"))
        self.save_frap_fit_check = QCheckBox("FRAP fit (separate)")
        self.save_frap_fit_check.setChecked(False)
        self.save_frap_fit_check.setToolTip("Save FRAP_frap_fit_separate.tif (default off)")
        plot_row.addWidget(self.save_frap_fit_check)

        self.save_k_exp_scatters_check = QCheckBox("K-exp scatters")
        self.save_k_exp_scatters_check.setChecked(False)
        self.save_k_exp_scatters_check.setToolTip("Save FRAP_K_exp_scatters_combined.tif/.png (default off)")
        plot_row.addWidget(self.save_k_exp_scatters_check)

        self.save_noise_pixels_check = QCheckBox("Noise pixels")
        self.save_noise_pixels_check.setChecked(False)
        self.save_noise_pixels_check.setToolTip("Save FRAP_noise_pixels.tif (default off)")
        plot_row.addWidget(self.save_noise_pixels_check)

        self.save_nobleach_k_check = QCheckBox("NoBleach K curves")
        self.save_nobleach_k_check.setChecked(True)
        self.save_nobleach_k_check.setToolTip("Save NoBleach_K_curves.tif (default on)")
        plot_row.addWidget(self.save_nobleach_k_check)
        plot_row.addStretch()
        pde_section.addLayout(plot_row)

        layout.addWidget(pde_section)
        
        # ============ OUTPUT OPTIONS ============
        output_layout = QHBoxLayout()
        self.pool_filtered_check = QCheckBox("Generate combined output files")
        self.pool_filtered_check.setToolTip("Generate combined CSV/XLSX files across all movies")
        output_layout.addWidget(self.pool_filtered_check)
        output_layout.addStretch()
        layout.addLayout(output_layout)
        
        # Save/Load analysis config buttons
        config_layout = QHBoxLayout()
        self.save_config_btn = QPushButton("Save Configuration File")
        self.save_config_btn.clicked.connect(self.save_analysis_config)
        self.load_config_btn = QPushButton("Load Configuration File")
        self.load_config_btn.clicked.connect(self.load_analysis_config)
        config_layout.addWidget(self.save_config_btn)
        config_layout.addWidget(self.load_config_btn)
        
        # Auto-save checkbox (enabled by default)
        self.auto_save_config_check = QCheckBox("Auto-save run config")
        self.auto_save_config_check.setChecked(True)
        self.auto_save_config_check.setToolTip(
            "Automatically save all parameters and log output to a JSON file\n"
            "in the input directory when analysis completes"
        )
        config_layout.addWidget(self.auto_save_config_check)
        
        config_layout.addStretch()
        layout.addLayout(config_layout)
        
        # Run buttons
        run_layout = QHBoxLayout()
        
        # Process number: single control driving all parallel workers (crop + fitting +
        # D-grid). Last user input, next to the Run buttons.
        run_layout.addWidget(QLabel("Process number:"))
        self.process_number_spin = QSpinBox()
        self.process_number_spin.setRange(1, 64)
        self.process_number_spin.setValue(8)
        self.process_number_spin.setToolTip(
            "Number of parallel worker processes used across cropping and fitting.")
        def _sync_process_number(v):
            self.crop_n_jobs_spin.setValue(v)
            self.n_jobs_conditions_spin.setValue(v)
        self.process_number_spin.valueChanged.connect(_sync_process_number)
        _sync_process_number(self.process_number_spin.value())
        run_layout.addWidget(self.process_number_spin)

        self.crop_only_btn = QPushButton("Crop Only")
        self.crop_only_btn.clicked.connect(self.run_crop_only)
        self.crop_only_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.crop_only_btn.setToolTip("Run droplet detection and cropping only (no PDE fitting)")
        run_layout.addWidget(self.crop_only_btn)
        
        self.run_btn = QPushButton("Run Full Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        run_layout.addWidget(self.run_btn)
        
        self.filter_analysis_btn = QPushButton("Apply Filtering Settings")
        self.filter_analysis_btn.clicked.connect(self.filter_analysis)
        self.filter_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.filter_analysis_btn.setToolTip("Apply filters from the Filtering tab (requires filtering changes after analysis)")
        self.filter_analysis_btn.setEnabled(False)  # Disabled until filtering changes made
        run_layout.addWidget(self.filter_analysis_btn)
        run_layout.addStretch()
        layout.addLayout(run_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Output log
        log_group = QGroupBox("Output Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setMinimumHeight(120)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group, stretch=1)
        
        scroll_widget.setLayout(layout)
        scroll.setWidget(scroll_widget)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
    
    def _get_crop_params(self):
        """Get parameters for droplet_crop.py."""
        return {
            'min_area': self.crop_min_area_spin.value(),
            'bleach_threshold': self.crop_bleach_thresh_spin.value(),
            'crop_multiplier': self.crop_multiplier_spin.value(),
            'min_nobleach_diameter': self.crop_min_nb_diam_spin.value(),
            'min_nobleach_pc': self.crop_min_nb_pc_spin.value(),
            'min_bleach_depth': self.crop_min_bleach_depth_spin.value(),
            'save_threshold_masks': self.save_threshold_qc_check.isChecked(),
            'pixel_size': self.pixel_size_spin.value(),
            'n_jobs': self.crop_n_jobs_spin.value(),
            'stabilize': self.stabilize_check.isChecked(),
            'clear_borders': self.crop_clear_borders_check.isChecked(),
            'quiet': False,
        }
    
    def run_crop_only(self):
        """Run only the droplet cropping step."""
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Error", "Please select a valid input directory.")
            return
        
        if self.crop_worker and self.crop_worker.isRunning():
            QMessageBox.warning(self, "Error", "Cropping is already running.")
            return
        
        params = self._get_crop_params()
        self.crop_worker = DropletCropWorker(input_dir, params)
        self.crop_worker.progress.connect(self.log_message)
        self.crop_worker.finished.connect(self.crop_finished)
        
        self.crop_only_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log_text.append("Starting droplet cropping...\n")
        
        self.crop_worker.start()
    
    def crop_finished(self, success, message):
        self.crop_only_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.log_text.append(f"\nDroplet cropping completed! Output: {message}\n")
            # Auto-save run configuration if enabled
            if self.auto_save_config_check.isChecked():
                self._auto_save_run_config(success, "crop_only")
            QMessageBox.information(self, "Success", f"Droplet cropping completed!\nOutput: {message}")
        else:
            self.log_text.append(f"\nError: {message}\n")
            # Still save config on failure so user can see what went wrong
            if self.auto_save_config_check.isChecked():
                self._auto_save_run_config(success, "crop_only")
            QMessageBox.critical(self, "Error", message)
    
    def browse_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir_path:
            self.input_dir_edit.setText(dir_path)
    
    def _on_pde_fitting_changed(self, state):
        """Called when PDE fitting checkbox state changes. Updates filtering tab dropdowns."""
        # The mixed model needs a PDE fit; disable (and visually clear) it when PDE is off.
        if hasattr(self, 'mixed_model_check'):
            pde_on = self.pde_fitting_check.isChecked()
            self.mixed_model_check.setEnabled(pde_on)
            if not pde_on:
                self.mixed_model_check.setChecked(False)
        # Notify the filtering tab to update its parameter combos
        if self.main_window and hasattr(self.main_window, 'filtering_tab'):
            filtering_tab = self.main_window.filtering_tab
            if hasattr(filtering_tab, 'update_param_combos'):
                filtering_tab.update_param_combos()
    
    def is_pde_fitting_enabled(self):
        """Return whether PDE fitting is enabled."""
        return self.pde_fitting_check.isChecked()
    
    def run_analysis(self):
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Error", "Please select a valid input directory.")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Error", "Analysis is already running.")
            return
        
        # Check if we're processing CSVs (existing data), TIF files (cropped droplets), or raw movies
        csv_files = find_all_frap_curves_csvs(input_dir)
        is_csv_processing = len(csv_files) > 0
        
        # Supported raw movie extensions
        raw_movie_extensions = ['.nd2', '.czi', '.lif', '.lsm', '.oib', '.oif', '.dv', '.r3d', '.ims']
        tif_extensions = ['.tif', '.tiff']
        
        # Check for TIF files in subdirectories (cropped droplets)
        has_tif_files = False
        has_raw_movies = False
        raw_movie_files = []
        
        if not is_csv_processing:
            import glob
            
            # First check for raw movie files in root directory
            for ext in raw_movie_extensions:
                found = glob.glob(os.path.join(input_dir, f"*{ext}"))
                raw_movie_files.extend(found)
            
            has_raw_movies = len(raw_movie_files) > 0
            
            if not has_raw_movies:
                # Look for TIF files in subdirectories (cropped droplets from previous run)
                subdirs = [d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d)) 
                          and not d.startswith('.')]
                
                for subdir in subdirs:
                    subdir_path = os.path.join(input_dir, subdir)
                    # Check for cropped droplet TIFs (start with "Drift_")
                    tif_files = glob.glob(os.path.join(subdir_path, "Drift_*.tif")) + \
                               glob.glob(os.path.join(subdir_path, "Drift_*.tiff"))
                    if tif_files:
                        has_tif_files = True
                        break
                    # Also check for any TIF files
                    tif_files = glob.glob(os.path.join(subdir_path, "*.tif")) + \
                               glob.glob(os.path.join(subdir_path, "*.tiff"))
                    if tif_files:
                        has_tif_files = True
                        break
                
                # Also check root directory for TIF movies (not cropped droplets)
                if not has_tif_files:
                    tif_files = glob.glob(os.path.join(input_dir, "*.tif")) + \
                               glob.glob(os.path.join(input_dir, "*.tiff"))
                    if tif_files:
                        # Check if these look like raw movies (not small cropped droplets)
                        # Raw movies are typically larger files
                        for f in tif_files:
                            if os.path.getsize(f) > 1_000_000:  # > 1MB likely a movie
                                has_raw_movies = True
                                raw_movie_files.append(f)
                        if not has_raw_movies:
                            has_tif_files = True
        
        # Get parameters
        max_drop = self.max_drop_spin.value()
        max_jump = self.max_jump_spin.value()
        ignore_first_n = self.ignore_first_n_spin.value()
        min_val = self.min_val_spin.value()
        max_val = self.max_val_spin.value()
        no_filter = self.no_filter_check.isChecked()
        n_jobs = multiprocessing.cpu_count()  # Use all available cores by default
        
        # Get droplet detection parameters
        pixel_size = self.pixel_size_spin.value()
        frame_interval = self.frame_interval_spin.value()
        droplet_min_size = self.droplet_min_size_spin.value()
        ratio_percentile = self.ratio_percentile_spin.value()
        d_min = self.d_min_spin.value()
        d_max = self.d_max_spin.value()
        
        # For CSV processing, check if filtering tab has filters to apply
        allowed_filenames = None
        if is_csv_processing and self.main_window and hasattr(self.main_window, 'filtering_tab'):
            filtering_tab = self.main_window.filtering_tab
            included_filenames = filtering_tab.get_included_filenames()
            if included_filenames is not None:
                allowed_filenames = included_filenames
        
        # Start appropriate worker
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        
        pool_filtered = self.pool_filtered_check.isChecked()
        
        if is_csv_processing:
            # Process existing CSV files
            self.log_text.append("Starting analysis on existing CSV files...\n")
            self.worker = AnalysisWorker(input_dir, max_drop, max_jump, ignore_first_n,
                                         min_val, max_val, no_filter, n_jobs,
                                         allowed_filenames=allowed_filenames,
                                         frame_interval=frame_interval,
                                         pool_filtered=pool_filtered)
        elif has_raw_movies or has_tif_files:
            # Process raw movie files or cropped droplet TIF files
            if has_raw_movies:
                self.log_text.append(f"Detected {len(raw_movie_files)} raw movie file(s). Starting full pipeline...\n")
                self.log_text.append("Pipeline: Stabilize -> Crop droplets -> Analyze\n\n")
            else:
                self.log_text.append("Detected cropped TIF files. Starting analysis pipeline...\n")
            
            # Build params dict for the worker using all GUI parameters
            # Determine stabilization algorithm
            stab_algo = self.stabilize_algorithm_combo.currentText()
            # "Gradient (recommended)" routes through the gradient-domain registration
            # (use_lucas_kanade flag); only "ECC" selects the ECC path.
            use_lucas_kanade = not stab_algo.strip().upper().startswith("ECC")
            
            params = {
                'pixel_size': pixel_size,
                'frame_interval': frame_interval,
                'pre_bleach_frames': self.pre_bleach_frames_spin.value(),
                'analysis_channel': self.analysis_channel_combo.currentIndex() + 1,
                'two_channel': self.two_channel_check.isChecked(),
                'save_frap_fit_separate': self.save_frap_fit_check.isChecked(),
                'save_k_exp_scatters': self.save_k_exp_scatters_check.isChecked(),
                'save_noise_pixels': self.save_noise_pixels_check.isChecked(),
                'save_nobleach_k_curves': self.save_nobleach_k_check.isChecked(),
                'droplet_min_size': droplet_min_size,
                'ratio_percentile': ratio_percentile,
                'd_min': d_min,
                'd_max': d_max,
                'n_jobs_conditions': self.n_jobs_conditions_spin.value(),
                # PDE fitting parameters from GUI
                'd_grid_size': self.d_grid_size_spin.value(),
                'd_refine_grid_size': self.d_refine_grid_spin.value(),
                'n_jobs_d': self.n_jobs_conditions_spin.value(),
                'crop_margin': 5,
                'max_frames': self.max_frames_spin.value(),
                'downsample_factor': self.downsample_spin.value(),
                'fit_region': 'bleach',
                'normalization_method': (
                    {
                        'Both (pre-bleach + whole-droplet)': 'both',
                        'Pre-bleach frame': 'prebleach',
                        'Whole-droplet ratio': 'wholedroplet',
                    }.get(self.normalization_method_combo.currentText(), 'both')
                    if hasattr(self, 'normalization_method_combo') else 'both'
                ),
                'pde_first_frame': self.pde_first_frame_spin.value(),
                'pde_last_frame': self.pde_last_frame_spin.value(),
                'exp_first_frame': self.exp_first_frame_spin.value(),
                'exp_last_frame': self.exp_last_frame_spin.value(),
                'pb_exclude_radius': 3,
                'pb_ratio_tolerance': 0.05,
                'no_photobleach_correct': self.no_photobleach_check.isChecked(),
                'no_exp_plateau_fit': self.no_exp_plateau_check.isChecked(),
                # PDE fitting on/off drives whether the D grid search runs at all.
                'pde_fitting_enabled': self.pde_fitting_check.isChecked(),
                # Mixed model is independent, but only when PDE is enabled AND selected.
                'no_combined_model': not (
                    self.pde_fitting_check.isChecked() and self.mixed_model_check.isChecked()
                ),
                'verbosity': 1,
                # Stabilization parameters
                'stabilize_enabled': self.stabilize_check.isChecked(),
                'stabilize_algorithm': 'lucas-kanade' if use_lucas_kanade else 'ecc',
                # Stabilization always uses subpixel application from the GUI.
                'whole_pixel_stabilization': False,
                'stab_pyramid_levels': self.stab_pyramid_spin.value(),
                'stab_max_iterations': self.stab_iterations_spin.value(),
                'stab_epsilon': self.stab_epsilon_spin.value(),
                'stab_template_update_coeff': self.stab_template_coeff_spin.value(),
                'stab_pre_smooth_sigma': self.stab_pre_smooth_spin.value(),
                'stab_max_shift_pixels': self.stab_max_shift_spin.value(),
                'stab_min_correlation': self.stab_min_corr_spin.value(),
                'stab_refine_iterations': self.stab_refine_spin.value(),
                # Droplet crop pass-through
                'run_droplet_crop': self.run_droplet_crop_check.isChecked(),
                'crop_clear_borders': self.crop_clear_borders_check.isChecked(),
                'crop_min_area': self.crop_min_area_spin.value(),
                'crop_bleach_threshold': self.crop_bleach_thresh_spin.value(),
                'crop_crop_multiplier': self.crop_multiplier_spin.value(),
                'crop_min_nobleach_diameter': self.crop_min_nb_diam_spin.value(),
                'crop_min_nobleach_pc': self.crop_min_nb_pc_spin.value(),
                'crop_bleach_spot_min_area': self.crop_bleach_spot_min_area_spin.value(),
                'crop_min_bleach_depth': self.crop_min_bleach_depth_spin.value(),
                'bleach_depth': self.crop_min_bleach_depth_spin.value(),
                'save_threshold_masks': self.save_threshold_qc_check.isChecked(),
                'crop_n_jobs': self.crop_n_jobs_spin.value(),
                # Variable timing + frame-range stabilization
                'timing_segments': self.timing_segments_edit.text().strip(),
                'stabilize_frame_start': self.stab_frame_start_spin.value(),
                'stabilize_frame_end': self.stab_frame_end_spin.value(),
                'stabilize_extrapolation': self.stab_extrapolation_combo.currentText(),
                # Flag to indicate raw movies need full processing
                'has_raw_movies': has_raw_movies,
                'raw_movie_files': raw_movie_files if has_raw_movies else [],
            }
            self.worker = FullAnalysisWorker(input_dir, params)
            # Connect progress_update for determinate progress bar
            self.worker.progress_update.connect(self.update_progress_bar)
        else:
            QMessageBox.warning(
                self, "Error",
                "No CSV files or TIF files found in the input directory.\n\n"
                "Please select a directory containing either:\n"
                "- all_frap_curves.csv files (for combining/filtering), or\n"
                "- Subfolders with cropped droplet TIF files (for full analysis)"
            )
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        
        self.worker.progress.connect(self.log_message)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()
    
    def _auto_save_run_config(self, success, run_type):
        """Automatically save run configuration and log output to a JSON file.
        
        Args:
            success: Whether the run completed successfully
            run_type: Type of run ('full_analysis', 'crop_only', 'filter_analysis')
        """
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_config_{run_type}_{timestamp}.json"
        log_path = os.path.join(input_dir, log_filename)
        
        # Collect comprehensive configuration
        run_config = {
            'run_info': {
                'timestamp': timestamp,
                'timestamp_readable': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'run_type': run_type,
                'success': success,
                'input_directory': input_dir,
            },
            'imaging_parameters': {
                'pixel_size': self.pixel_size_spin.value(),
                'frame_interval': self.frame_interval_spin.value(),
            },
            'stabilization_parameters': {
                'stabilize': self.stabilize_check.isChecked(),
                'algorithm': 'ecc' if self.stabilize_algorithm_combo.currentText().strip().upper().startswith('ECC') else 'lucas-kanade',
                'pyramid_levels': self.stab_pyramid_spin.value(),
                'max_iterations': self.stab_iterations_spin.value(),
                'epsilon': self.stab_epsilon_spin.value(),
                'template_update_coeff': self.stab_template_coeff_spin.value(),
                'pre_smooth_sigma': self.stab_pre_smooth_spin.value(),
                'max_shift_pixels': self.stab_max_shift_spin.value(),
                'min_correlation': self.stab_min_corr_spin.value(),
                'refine_iterations': self.stab_refine_spin.value(),
            },
            'crop_parameters': {
                'run_droplet_crop': self.run_droplet_crop_check.isChecked(),
                'stabilize': self.stabilize_check.isChecked(),
                'clear_borders': self.crop_clear_borders_check.isChecked(),
                'min_area': self.crop_min_area_spin.value(),
                'bleach_threshold': self.crop_bleach_thresh_spin.value(),
                'crop_multiplier': self.crop_multiplier_spin.value(),
                'min_nobleach_diameter': self.crop_min_nb_diam_spin.value(),
                'min_nobleach_pc': self.crop_min_nb_pc_spin.value(),
                'n_jobs': self.crop_n_jobs_spin.value(),
            },
            'pde_parameters': {
                'd_min': self.d_min_spin.value(),
                'd_max': self.d_max_spin.value(),
                'd_grid_size': self.d_grid_size_spin.value(),
                'd_refine_grid_size': self.d_refine_grid_spin.value(),
                'droplet_min_size': self.droplet_min_size_spin.value(),
                'ratio_percentile': self.ratio_percentile_spin.value(),
                'max_frames': self.max_frames_spin.value(),
                'downsample_factor': self.downsample_spin.value(),
                'pde_first_frame': self.pde_first_frame_spin.value(),
                'pde_last_frame': self.pde_last_frame_spin.value(),
                'exp_first_frame': self.exp_first_frame_spin.value(),
                'exp_last_frame': self.exp_last_frame_spin.value(),
                'no_photobleach_correct': self.no_photobleach_check.isChecked(),
                'no_exp_plateau_fit': self.no_exp_plateau_check.isChecked(),
                'pde_fitting_enabled': self.pde_fitting_check.isChecked(),
                'mixed_model_enabled': self.mixed_model_check.isChecked(),
                'n_jobs_conditions': self.n_jobs_conditions_spin.value(),
            },
            'arithmetic_filter_parameters': {
                'max_drop': self.max_drop_spin.value(),
                'max_jump': self.max_jump_spin.value(),
                'ignore_first_n': self.ignore_first_n_spin.value(),
                'min_val': self.min_val_spin.value(),
                'max_val': self.max_val_spin.value(),
                'no_filter': self.no_filter_check.isChecked()
            },
            'output_parameters': {
                'pool_filtered': self.pool_filtered_check.isChecked(),
            },
            'visual_filter_parameters': {},
            'log_output': self.log_text.toPlainText()
        }
        
        # Include visual filtering tab filters if available
        if self.main_window and hasattr(self.main_window, 'filtering_tab'):
            filtering_tab = self.main_window.filtering_tab
            run_config['visual_filter_parameters'] = filtering_tab.saved_filters.copy()
        
        try:
            with open(log_path, 'w') as f:
                json.dump(run_config, f, indent=2)
            self.log_text.append(f"\nRun configuration saved to: {log_path}\n")
        except Exception as e:
            self.log_text.append(f"\nWarning: Could not save run config: {str(e)}\n")
    
    def save_analysis_config(self):
        """Save comprehensive analysis configuration including all parameters."""
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "Warning", "Please set an input directory first.")
            return
        
        # Collect all parameters
        config = {
            'input_directory': input_dir,
            'imaging_parameters': {
                'pixel_size': self.pixel_size_spin.value(),
                'frame_interval': self.frame_interval_spin.value(),
            },
            'stabilization_parameters': {
                'stabilize': self.stabilize_check.isChecked(),
                'algorithm': 'ecc' if self.stabilize_algorithm_combo.currentText().strip().upper().startswith('ECC') else 'lucas-kanade',
                'pyramid_levels': self.stab_pyramid_spin.value(),
                'max_iterations': self.stab_iterations_spin.value(),
                'epsilon': self.stab_epsilon_spin.value(),
                'template_update_coeff': self.stab_template_coeff_spin.value(),
                'pre_smooth_sigma': self.stab_pre_smooth_spin.value(),
                'max_shift_pixels': self.stab_max_shift_spin.value(),
                'min_correlation': self.stab_min_corr_spin.value(),
                'refine_iterations': self.stab_refine_spin.value(),
            },
            'crop_parameters': {
                'run_droplet_crop': self.run_droplet_crop_check.isChecked(),
                'stabilize': self.stabilize_check.isChecked(),
                'clear_borders': self.crop_clear_borders_check.isChecked(),
                'min_area': self.crop_min_area_spin.value(),
                'bleach_threshold': self.crop_bleach_thresh_spin.value(),
                'crop_multiplier': self.crop_multiplier_spin.value(),
                'min_nobleach_diameter': self.crop_min_nb_diam_spin.value(),
                'min_nobleach_pc': self.crop_min_nb_pc_spin.value(),
                'n_jobs': self.crop_n_jobs_spin.value(),
            },
            'pde_parameters': {
                'd_min': self.d_min_spin.value(),
                'd_max': self.d_max_spin.value(),
                'd_grid_size': self.d_grid_size_spin.value(),
                'd_refine_grid_size': self.d_refine_grid_spin.value(),
                'droplet_min_size': self.droplet_min_size_spin.value(),
                'ratio_percentile': self.ratio_percentile_spin.value(),
                'max_frames': self.max_frames_spin.value(),
                'downsample_factor': self.downsample_spin.value(),
                'pde_first_frame': self.pde_first_frame_spin.value(),
                'pde_last_frame': self.pde_last_frame_spin.value(),
                'exp_first_frame': self.exp_first_frame_spin.value(),
                'exp_last_frame': self.exp_last_frame_spin.value(),
                'no_photobleach_correct': self.no_photobleach_check.isChecked(),
                'no_exp_plateau_fit': self.no_exp_plateau_check.isChecked(),
                'pde_fitting_enabled': self.pde_fitting_check.isChecked(),
                'mixed_model_enabled': self.mixed_model_check.isChecked(),
                'n_jobs_conditions': self.n_jobs_conditions_spin.value(),
            },
            'filter_parameters': {
                'max_drop': self.max_drop_spin.value(),
                'max_jump': self.max_jump_spin.value(),
                'ignore_first_n': self.ignore_first_n_spin.value(),
                'min_val': self.min_val_spin.value(),
                'max_val': self.max_val_spin.value(),
                'no_filter': self.no_filter_check.isChecked()
            },
            'output_parameters': {
                'pool_filtered': self.pool_filtered_check.isChecked(),
            },
            'filtering_tab_filters': {}
        }
        
        # Include filtering tab filters if available
        if self.main_window and hasattr(self.main_window, 'filtering_tab'):
            filtering_tab = self.main_window.filtering_tab
            config['filtering_tab_filters'] = filtering_tab.saved_filters.copy()
        
        # Save to JSON
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Configuration", "", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'w') as f:
                    json.dump(config, f, indent=2)
                QMessageBox.information(self, "Success", f"Analysis configuration saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save config: {str(e)}")
    
    def load_analysis_config(self):
        """Load comprehensive analysis configuration and restore all parameters."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Analysis Configuration", "", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                
                # Restore input directory
                if 'input_directory' in config:
                    self.input_dir_edit.setText(config['input_directory'])
                
                # Restore imaging parameters
                if 'imaging_parameters' in config:
                    ip = config['imaging_parameters']
                    self.pixel_size_spin.setValue(ip.get('pixel_size', 0.11))
                    self.frame_interval_spin.setValue(ip.get('frame_interval', 10.0))
                
                # Restore stabilization parameters
                if 'stabilization_parameters' in config:
                    sp = config['stabilization_parameters']
                    self.stabilize_check.setChecked(sp.get('stabilize', True))
                    algo = sp.get('algorithm', 'lucas-kanade')
                    idx = 0 if algo in ('lucas-kanade', 'lk') else 1
                    self.stabilize_algorithm_combo.setCurrentIndex(idx)
                    self.stab_pyramid_spin.setValue(sp.get('pyramid_levels', 4))
                    self.stab_iterations_spin.setValue(sp.get('max_iterations', 500))
                    self.stab_epsilon_spin.setValue(sp.get('epsilon', 1e-7))
                    self.stab_template_coeff_spin.setValue(sp.get('template_update_coeff', 0.95))
                    self.stab_pre_smooth_spin.setValue(sp.get('pre_smooth_sigma', 1.0))
                    self.stab_max_shift_spin.setValue(sp.get('max_shift_pixels', 25.0))
                    self.stab_min_corr_spin.setValue(sp.get('min_correlation', 0.8))
                    self.stab_refine_spin.setValue(sp.get('refine_iterations', 20))
                
                # Restore crop parameters
                if 'crop_parameters' in config:
                    cp = config['crop_parameters']
                    self.run_droplet_crop_check.setChecked(cp.get('run_droplet_crop', False))
                    self.stabilize_check.setChecked(cp.get('stabilize', True))
                    self.crop_clear_borders_check.setChecked(cp.get('clear_borders', True))
                    self.crop_min_area_spin.setValue(cp.get('min_area', 50.0))
                    self.crop_bleach_thresh_spin.setValue(cp.get('bleach_threshold', 1.2))
                    self.crop_multiplier_spin.setValue(cp.get('crop_multiplier', 2.0))
                    self.crop_min_nb_diam_spin.setValue(cp.get('min_nobleach_diameter', 10.0))
                    self.crop_min_nb_pc_spin.setValue(cp.get('min_nobleach_pc', 5.0))
                    self.crop_n_jobs_spin.setValue(cp.get('n_jobs', 8))
                
                # Restore PDE parameters
                if 'pde_parameters' in config:
                    pp = config['pde_parameters']
                    self.d_min_spin.setValue(pp.get('d_min', 0.001))
                    self.d_max_spin.setValue(pp.get('d_max', 2.0))
                    self.d_grid_size_spin.setValue(pp.get('d_grid_size', 25))
                    self.d_refine_grid_spin.setValue(pp.get('d_refine_grid_size', 100))
                    self.droplet_min_size_spin.setValue(pp.get('droplet_min_size', 50))
                    self.ratio_percentile_spin.setValue(pp.get('ratio_percentile', 85.0))
                    self.max_frames_spin.setValue(pp.get('max_frames', -1))
                    self.downsample_spin.setValue(pp.get('downsample_factor', 2))
                    self.pde_first_frame_spin.setValue(pp.get('pde_first_frame', 1))
                    self.pde_last_frame_spin.setValue(pp.get('pde_last_frame', 7))
                    self.exp_first_frame_spin.setValue(pp.get('exp_first_frame', 8))
                    self.exp_last_frame_spin.setValue(pp.get('exp_last_frame', 30))
                    self.no_photobleach_check.setChecked(pp.get('no_photobleach_correct', False))
                    self.no_exp_plateau_check.setChecked(pp.get('no_exp_plateau_fit', False))
                    # Handle both old config format (no_combined_model) and new (pde_fitting_enabled)
                    if 'pde_fitting_enabled' in pp:
                        self.pde_fitting_check.setChecked(pp.get('pde_fitting_enabled', True))
                    elif 'no_combined_model' in pp:
                        self.pde_fitting_check.setChecked(not pp.get('no_combined_model', False))
                    # Mixed model: prefer explicit new flag; fall back to legacy no_combined_model.
                    if 'mixed_model_enabled' in pp:
                        self.mixed_model_check.setChecked(pp.get('mixed_model_enabled', False))
                    elif 'no_combined_model' in pp:
                        self.mixed_model_check.setChecked(not pp.get('no_combined_model', False))
                    # Keep the mixed-model checkbox enabled state in sync with PDE.
                    self._on_pde_fitting_changed(None)
                    self.n_jobs_conditions_spin.setValue(pp.get('n_jobs_conditions', 8))
                    if hasattr(self, 'process_number_spin'):
                        self.process_number_spin.setValue(pp.get('n_jobs_conditions', 8))
                
                # Restore filter parameters (also check legacy format)
                if 'filter_parameters' in config:
                    fp = config['filter_parameters']
                    self.max_drop_spin.setValue(fp.get('max_drop', 0.1))
                    self.max_jump_spin.setValue(fp.get('max_jump', 0.1))
                    self.ignore_first_n_spin.setValue(fp.get('ignore_first_n', 4))
                    self.min_val_spin.setValue(fp.get('min_val', 0.1))
                    self.max_val_spin.setValue(fp.get('max_val', 1.2))
                    self.no_filter_check.setChecked(fp.get('no_filter', False))
                
                # Legacy format support for droplet_detection_parameters
                if 'droplet_detection_parameters' in config and 'imaging_parameters' not in config:
                    ddp = config['droplet_detection_parameters']
                    self.pixel_size_spin.setValue(ddp.get('pixel_size', 0.11))
                    self.frame_interval_spin.setValue(ddp.get('frame_interval', 10.0))
                    self.droplet_min_size_spin.setValue(ddp.get('droplet_min_size', 50))
                    self.ratio_percentile_spin.setValue(ddp.get('ratio_percentile', 85.0))
                    self.d_min_spin.setValue(ddp.get('d_min', 0.001))
                    self.d_max_spin.setValue(ddp.get('d_max', 2.0))
                
                # Restore output parameters
                if 'output_parameters' in config:
                    op = config['output_parameters']
                    self.pool_filtered_check.setChecked(op.get('pool_filtered', False))
                
                # Restore filtering tab filters
                if 'filtering_tab_filters' in config and self.main_window and hasattr(self.main_window, 'filtering_tab'):
                    filtering_tab = self.main_window.filtering_tab
                    filtering_tab.saved_filters = config['filtering_tab_filters'].copy()
                    filtering_tab.update_saved_filters_table()
                    if filtering_tab.current_plot is not None:
                        filtering_tab.update_plot_with_filters()
                
                QMessageBox.information(self, "Success", f"Analysis configuration loaded from {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")
    
    def log_message(self, message):
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_progress_bar(self, current, total):
        """Update progress bar with current/total values."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"Processing: {current}/{total} files (%p%)")
        else:
            # Fall back to indeterminate
            self.progress_bar.setRange(0, 0)
    
    def analysis_finished(self, success, message):
        self.run_btn.setEnabled(True)
        self.crop_only_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_text.append(f"\n{message}\n")
        
        # Auto-save run configuration if enabled
        if self.auto_save_config_check.isChecked():
            self._auto_save_run_config(success, "full_analysis")
        
        if success:
            # Auto-load summary CSVs into filtering tab
            input_dir = self.input_dir_edit.text().strip()
            if input_dir and self.main_window and hasattr(self.main_window, 'filtering_tab'):
                filtering_tab = self.main_window.filtering_tab
                loaded = filtering_tab.auto_load_from_directory(input_dir, show_source=True)
                if loaded:
                    self.log_text.append("Summary CSVs auto-loaded into Filtering tab.\n")
                    # Disable the apply button until user makes filtering changes
                    self.filter_analysis_btn.setEnabled(False)
            
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def filter_analysis(self):
        """Apply filters from filtering tab to the analysis."""
        if not self.main_window or not hasattr(self.main_window, 'filtering_tab'):
            QMessageBox.warning(self, "Warning", "Filtering tab not available.")
            return
        
        filtering_tab = self.main_window.filtering_tab
        saved_filters = filtering_tab.saved_filters
        included_filenames = filtering_tab.get_included_filenames()

        input_dir = self.input_dir_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Error", "Please select a valid input directory first.")
            return

        if included_filenames is None:
            QMessageBox.warning(self, "Warning", "Please create a plot and apply filters in the Filtering tab to determine which curves to keep.")
            return
        
        # Get parameters (apply to curves individually)
        max_drop = self.max_drop_spin.value()
        max_jump = self.max_jump_spin.value()
        ignore_first_n = self.ignore_first_n_spin.value()
        min_val = self.min_val_spin.value()
        max_val = self.max_val_spin.value()
        no_filter = self.no_filter_check.isChecked()
        n_jobs = multiprocessing.cpu_count()

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Error", "Another analysis is already running.")
            return

        # Get frame interval for plotting
        frame_interval = self.frame_interval_spin.value()
        pool_filtered = self.pool_filtered_check.isChecked()
        
        self.worker = AnalysisWorker(
            input_dir, max_drop, max_jump, ignore_first_n,
            min_val, max_val, no_filter, n_jobs,
            allowed_filenames=included_filenames,
            output_prefix="filtered_",
            frame_interval=frame_interval,
            pool_filtered=pool_filtered
        )
        self.worker.progress.connect(self.log_message)
        self.worker.finished.connect(self.analysis_finished)

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log_text.append("Starting filtered analysis...\n")

        self.worker.start()


class VisualizationTab(QWidget):
    """Tab for visualizing individual FRAP curves from a selected movie."""
    
    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.current_data = None
        self.current_csv_path = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File selection section
        file_group = QGroupBox("Select Movie / CSV File")
        file_layout = QVBoxLayout()
        
        # Directory and file selection
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Input Directory:"))
        self.dir_label = QLabel("Not selected")
        self.dir_label.setStyleSheet("color: #666;")
        dir_row.addWidget(self.dir_label, 1)
        
        self.browse_dir_btn = QPushButton("Browse Directory")
        self.browse_dir_btn.clicked.connect(self.browse_directory)
        dir_row.addWidget(self.browse_dir_btn)
        file_layout.addLayout(dir_row)
        
        # Movie/file selector
        movie_row = QHBoxLayout()
        movie_row.addWidget(QLabel("Select Movie:"))
        self.movie_combo = QComboBox()
        self.movie_combo.setMinimumWidth(300)
        self.movie_combo.currentTextChanged.connect(self.load_selected_movie)
        movie_row.addWidget(self.movie_combo, 1)
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_movie_list)
        movie_row.addWidget(self.refresh_btn)
        file_layout.addLayout(movie_row)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Filtering options
        filter_group = QGroupBox("Filtering Options")
        filter_layout = QHBoxLayout()
        
        self.show_filtered_check = QCheckBox("Show only curves passing detection thresholds")
        self.show_filtered_check.setToolTip("Display only curves that pass the detection threshold parameters")
        self.show_filtered_check.stateChanged.connect(self.update_plot)
        filter_layout.addWidget(self.show_filtered_check)
        
        filter_layout.addWidget(QLabel("Max Drop:"))
        self.viz_max_drop = QDoubleSpinBox()
        self.viz_max_drop.setRange(0.0, 1.0)
        self.viz_max_drop.setValue(0.1)
        self.viz_max_drop.setDecimals(3)
        self.viz_max_drop.valueChanged.connect(self.update_plot)
        filter_layout.addWidget(self.viz_max_drop)
        
        filter_layout.addWidget(QLabel("Max Jump:"))
        self.viz_max_jump = QDoubleSpinBox()
        self.viz_max_jump.setRange(0.0, 1.0)
        self.viz_max_jump.setValue(0.1)
        self.viz_max_jump.setDecimals(3)
        self.viz_max_jump.valueChanged.connect(self.update_plot)
        filter_layout.addWidget(self.viz_max_jump)
        
        filter_layout.addWidget(QLabel("Ignore First N:"))
        self.viz_ignore_n = QSpinBox()
        self.viz_ignore_n.setRange(0, 100)
        self.viz_ignore_n.setValue(4)
        self.viz_ignore_n.valueChanged.connect(self.update_plot)
        filter_layout.addWidget(self.viz_ignore_n)
        
        filter_layout.addStretch()
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Plot area
        plot_group = QGroupBox("FRAP Curves")
        plot_layout = QVBoxLayout()
        
        if PYQTGRAPH_AVAILABLE:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('w')
            self.plot_widget.setLabel('left', 'Normalized Intensity')
            self.plot_widget.setLabel('bottom', 'Frame')
            self.plot_widget.addLegend()
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            no_plot_label = QLabel("PyQtGraph not available. Install with: pip install pyqtgraph")
            no_plot_label.setStyleSheet("color: red; padding: 20px;")
            plot_layout.addWidget(no_plot_label)
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group, 1)  # Give it stretch
        
        # Status bar
        self.status_label = QLabel("Select a directory and movie to visualize FRAP curves.")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def browse_directory(self):
        """Browse for input directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory with FRAP Data"
        )
        if dir_path:
            self.dir_label.setText(dir_path)
            self.refresh_movie_list()
    
    def use_analysis_directory(self):
        """Use the directory from the analysis tab if available."""
        if self.main_window and hasattr(self.main_window, 'analysis_tab'):
            analysis_dir = self.main_window.analysis_tab.input_dir_edit.text().strip()
            if analysis_dir and os.path.isdir(analysis_dir):
                self.dir_label.setText(analysis_dir)
                self.refresh_movie_list()
                return True
        return False
    
    def refresh_movie_list(self):
        """Scan the Analysis Results directory for the curve CSVs written by the
        Analysis-run tab and populate the movie selector.

        Prefers the new per-normalization stacked CSVs (``Pre-bleach_stacked_all.csv`` /
        ``Whole-droplet_stacked_all.csv``), both the filtered main files and the raw
        ``*_unfiltered.csv`` copies, per movie folder and at the results root. Falls back
        to the legacy ``all_frap_curves.csv`` if that's all that's present.
        """
        self.movie_combo.clear()

        dir_path = self.dir_label.text()
        if not dir_path or dir_path == "Not selected" or not os.path.isdir(dir_path):
            return

        # Recognized curve-file base names -> friendly normalization label.
        stacked_labels = {
            "Pre-bleach_stacked_all.csv": "Pre-bleach",
            "Whole-droplet_stacked_all.csv": "Whole-droplet",
            "Pre-bleach_stacked_all_unfiltered.csv": "Pre-bleach (unfiltered)",
            "Whole-droplet_stacked_all_unfiltered.csv": "Whole-droplet (unfiltered)",
        }

        entries = []  # (label, path)
        dir_abs = os.path.abspath(dir_path)
        for root, dirs, files in os.walk(dir_path):
            is_root = os.path.abspath(root) == dir_abs
            # Movie name from the *_results folder (root aggregates all movies).
            folder = os.path.basename(root)
            movie_name = folder[:-8] if folder.endswith("_results") else folder
            for fname, norm in stacked_labels.items():
                if fname in files:
                    scope = "All movies (combined)" if is_root else movie_name
                    entries.append((f"{scope}  —  {norm}",
                                    os.path.join(root, fname)))
            # Legacy fallback.
            if "all_frap_curves.csv" in files:
                scope = "All movies (combined)" if is_root else movie_name
                entries.append((f"{scope}  —  legacy curves",
                                os.path.join(root, "all_frap_curves.csv")))

        if not entries:
            self.status_label.setText(
                "No FRAP curve CSVs found. Run the Analysis tab first (looks for "
                "*_stacked_all.csv / all_frap_curves.csv).")
            return

        # Combined-root entries first, then per-movie, alphabetical; filtered before raw.
        entries.sort(key=lambda e: (not e[0].startswith("All movies"),
                                    "unfiltered" in e[0], e[0]))
        for label, path in entries:
            self.movie_combo.addItem(label, path)

        self.status_label.setText(f"Found {len(entries)} curve set(s).")

    @staticmethod
    def _load_curves_dataframe(csv_path):
        """Load any of the supported curve CSVs into a frames-x-curves DataFrame
        (columns are individual curves, one row per frame)."""
        df = pd.read_csv(csv_path)
        cols_lower = [str(c).lower() for c in df.columns]

        # New transposed stacked format: a 'bleach_point' column + frame_N columns.
        if "bleach_point" in cols_lower:
            bp_col = df.columns[cols_lower.index("bleach_point")]
            frame_cols = [c for c in df.columns if str(c).lower().startswith("frame")]
            wide = df.set_index(bp_col)[frame_cols].T
            wide.index = range(len(wide))  # frame index
            return wide

        # Long format: FRAP_droplet_timeseries.csv (droplet + FRAP_norm_* columns).
        if "droplet" in cols_lower:
            drop_col = df.columns[cols_lower.index("droplet")]
            norm_col = None
            for cand in ("FRAP_norm_prebleach", "FRAP_norm_wholedroplet", "FRAP_fit_ROI"):
                if cand in df.columns:
                    norm_col = cand
                    break
            if norm_col is not None:
                pivoted = df.pivot_table(index=df.groupby(drop_col).cumcount(),
                                         columns=drop_col, values=norm_col)
                return pivoted

        # Legacy all_frap_curves.csv: columns already curves (maybe a leading frame col).
        return df

    def load_selected_movie(self, movie_name):
        """Load the selected curve set into the plot."""
        if not movie_name:
            return

        csv_path = self.movie_combo.currentData()
        if not csv_path or not os.path.exists(csv_path):
            return

        try:
            self.current_data = self._load_curves_dataframe(csv_path)
            self.current_csv_path = csv_path
            n_curves = self.current_data.shape[1]
            self.status_label.setText(f"Loaded {movie_name}: {n_curves} curves")
            self.update_plot()
        except Exception as e:
            self.status_label.setText(f"Error loading {movie_name}: {str(e)}")
            self.current_data = None
    
    def update_plot(self):
        """Update the plot with current data and filter settings."""
        if self.plot_widget is None or self.current_data is None:
            return
        
        self.plot_widget.clear()
        
        df = self.current_data.copy()
        
        # Check if first column is a frame index (only for legacy/wide inputs that carry
        # one). The stacked/long loaders already return a clean frames-x-curves frame.
        first_col = str(df.columns[0]).lower()
        if 'frame' in first_col or 'time' in first_col or df.columns[0] == 'Unnamed: 0':
            df = df.set_index(df.columns[0])
        
        # Get filter parameters
        apply_filter = self.show_filtered_check.isChecked()
        max_drop = self.viz_max_drop.value()
        max_jump = self.viz_max_jump.value()
        ignore_n = self.viz_ignore_n.value()
        
        # Colors for curves
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        total_curves = 0
        filtered_curves = 0
        
        for i, col in enumerate(df.columns):
            try:
                y_vals = df[col].values.astype(float)
                x_vals = np.arange(len(y_vals))
                
                # Check if curve passes filter
                passes_filter = True
                if apply_filter:
                    # Apply jump/drop filter logic
                    for j in range(ignore_n, len(y_vals) - 1):
                        if j >= len(y_vals):
                            break
                        diff = y_vals[j + 1] - y_vals[j]
                        if diff < -max_drop or diff > max_jump:
                            passes_filter = False
                            break
                
                total_curves += 1
                
                if not apply_filter or passes_filter:
                    filtered_curves += 1
                    color = colors[i % len(colors)]
                    pen = pg.mkPen(color=color, width=1.5)
                    self.plot_widget.plot(x_vals, y_vals, pen=pen, name=str(col)[:20])
                    
            except (ValueError, TypeError):
                continue
        
        # Update status
        if apply_filter:
            self.status_label.setText(
                f"Showing {filtered_curves}/{total_curves} curves (passed filter)"
            )
        else:
            self.status_label.setText(f"Showing all {total_curves} curves")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BleachDt: FRAP Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Window/title-bar icon (matches the app/dock icon).
        _win_icon = app_icon()
        if not _win_icon.isNull():
            self.setWindowIcon(_win_icon)

        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                color: #000000;
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 3px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                color: #000000;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                color: #000000;
                selection-background-color: #2196F3;
                selection-color: white;
            }
            QComboBox {
                padding-right: 20px;
            }
            QComboBox::drop-down {
                border: 1px solid #cccccc;
                border-left: none;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                width: 20px;
                background-color: #f0f0f0;
            }
            QComboBox::drop-down:hover {
                background-color: #e0e0e0;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #666666;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #333333;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            QTableWidget {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
                gridline-color: #e0e0e0;
            }
        """)
        
        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Branding header: wordmark (logo + package name), falling back to styled text.
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        wm = wordmark_pixmap(height=34)
        brand_label = QLabel()
        if wm is not None and not wm.isNull():
            brand_label.setPixmap(wm)
        else:
            # Fallback: an icon glyph next to a text wordmark.
            _ic = app_icon()
            if not _ic.isNull():
                brand_label.setPixmap(_ic.pixmap(28, 28))
            text_label = QLabel("bleachDτ")
            text_label.setStyleSheet("font-size: 20px; font-weight: 600; color: #1a1a1a;")
            header.addWidget(text_label)
        header.addWidget(brand_label)
        header.addStretch()
        layout.addLayout(header)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #2196F3;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
        """)
        
        # Add tabs (pass main_window reference to both tabs for cross-communication)
        self.filtering_tab = FilteringTab(main_window=self)
        self.analysis_tab = AnalysisTab(main_window=self)
        self.visualization_tab = VisualizationTab(main_window=self)
        
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.filtering_tab, "Filtering")
        self.tabs.addTab(self.visualization_tab, "Visualization")

        layout.addWidget(self.tabs)

        # Ensure no numeric field is capped below the range the user needs. Some spinboxes
        # were built with low maxima (e.g. PDE/exp frame fields at 100), which blocked
        # legitimate values like a 300-frame fit window. Sweep every spinbox in the app and
        # raise its maximum: integer fields (frame indices, counts, iterations) to a very
        # large value; float fields to at least 999. Minimums and current values are left
        # untouched.
        self._raise_spinbox_limits()

    def _raise_spinbox_limits(self, int_max: int = 1_000_000, float_min_max: float = 999.0):
        """Raise the maximum of every spin box so numeric entry is never blocked below the
        needed range (see call site). Never lowers an already-higher maximum."""
        try:
            for sb in self.findChildren(QSpinBox):
                if sb.maximum() < int_max:
                    sb.setMaximum(int_max)
            for sb in self.findChildren(QDoubleSpinBox):
                if sb.maximum() < float_min_max:
                    sb.setMaximum(float_min_max)
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)

    # Branding: application name + icon. Setting the QApplication icon makes it the
    # taskbar/dock icon at runtime across platforms. (For a fully standalone macOS .app
    # bundle, also point CFBundleIconFile at assets/icon.icns in the bundle's Info.plist.)
    app.setApplicationName("BleachDt")
    app.setApplicationDisplayName("BleachDt")
    _icon = app_icon()
    if not _icon.isNull():
        app.setWindowIcon(_icon)

    # Set application style
    app.setStyle('Fusion')
    # Force a light palette so the UI stays readable in macOS dark mode
    light_palette = QPalette()
    light_palette.setColor(QPalette.Window, QColor(245, 245, 245))
    light_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    light_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.Text, QColor(0, 0, 0))
    light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    light_palette.setColor(QPalette.Highlight, QColor(33, 150, 243))
    light_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(light_palette)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

