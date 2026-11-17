# BleachDt

<p align="center">
  <strong>A comprehensive Python toolkit for analyzing FRAP experiments in phase-separated droplets</strong>
</p>

<p align="center">
  <em>The name "BleachDt" (pronounced "bleached") is a nod to the optional PDE-based diffusion coefficient (D) fitting.</em>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#gui-guide">GUI Guide</a> •
  <a href="#command-line">Command Line</a> •
  <a href="#parameters">Parameters</a> •
  <a href="#output-files">Output Files</a>
</p>

---

## Features

- **Fiji-matched Stabilization** — Registers every frame to the pre-bleach frame on the whole-image **gradient magnitude**, exactly like Fiji's Image Stabilizer plugin. Verified against the ImageJ plugin's own transformation log: shifts agree to **~0.03 px**. The gradient-magnitude domain makes this robust to the bleach event and photobleaching (which are not motion). A single **subpixel** bilinear shift is applied to **all channels** (estimated from the analysis channel only), so a fixed ROI tracks the true droplet area across all frames. A `stabilization_shifts_*.csv` log is written for QC. (Optional `--exclude-bleach-from-stabilization` masks FRAP droplets for very sparse-droplet movies; off by default.)
- **Automated Droplet Detection** — Identifies and crops individual droplets from multi-droplet image stacks, recording each droplet's X/Y location and geometry.
- **PDE-Based Diffusion Fitting** — Fits diffusion coefficients using partial differential equation model to compensate for bimodal recovery that single exponential fitting obscures.
- **Dual Normalization** — Computes BOTH the standard pre-bleach-frame normalization and the whole-droplet fluorescence per frame normalization. Raw integrated densities (RawIntDen) and both normalized curves are always saved; the GUI selects which is reported as the headline recovery curve.
- **Dual Model Support** — Choose between "Separate" (independent PDE + exponential) or "Combined" (constrained) fitting models.
- **Consolidated Outputs** — One per-movie `droplet_metrics.csv` merges FRAP, NoBleach, and droplet location/geometry.
- **Single- or Two-Channel** — Pick the analysis channel (droplet/bleach ROIs are built from it), and optionally analyze the other channel too (measured with the same ROIs, outputs suffixed `_ch1`/`_ch2`). Mixed 1-/2-channel datasets are handled per-movie, and a second channel with no signal (or that fails thresholding) is skipped automatically.
- **Interactive GUI** — User-friendly interface with real-time, re-runnable scatter-plot filtering.
- **Batch Processing** — Parallel processing across multiple conditions and experiments.
- **Automatic Configuration Saving** — Every run saves its parameters and log output for reproducibility.

### Variable-cadence acquisitions & large-droplet detection (v4.3)

Support for long movies with a mixed acquisition cadence (e.g. a continuous 100 ms burst followed by 10 s sampling) and for large, non-circular, densely packed droplets:

- **Keep-everything detection.** Border clearing is OFF by default and droplets are no longer rejected because a neighbor falls inside their crop window — every detected droplet is kept regardless of size or shape (there is no upper size limit). Droplets whose body runs off the image edge are flagged (`touches_image_edge`) rather than discarded. Optional `--min-nobleach-diameter` / `--min-nobleach-pc` (default 0) can prune the photobleach-reference pool if you want.
- **Spot-based bleach-point detection.** A bleach point is found by thresholding the pre/post ratio image (mean of the pre-bleach frames ÷ first post-bleach frame) into localized **bleach spots**; any droplet a spot overlaps is FRAP, everything else is NoBleach. This calls large droplets correctly, where a whole-droplet mean would dilute the small bleach spot to background. Controlled by `--bleach-threshold` and `--bleach-spot-min-area`.
- **Averaged pre-bleach frames.** Detection, partition coefficient, and the ratio use the **mean of the pre-bleach frames** (not just the last one).
- **Range stabilization aligned to the first frame.** Estimate drift on frame 0 plus a chosen `[start, end]` window (`--stabilize-frame-start/-end`), then apply a continuous, whole-stack trajectory referenced to frame 0 — no discontinuity where the window ends (`--stabilize-extrapolation hold|linear|poly`). Efficient when drift is concentrated in a sparse tail.
- **Variable per-frame timing.** Per-frame acquisition timestamps are read from the movie metadata (Nikon ND2, or OME-TIFF `DeltaT`) when present; otherwise a `--timing-segments "300x0.1,35x10"` spec (300 frames at 0.1 s, then 35 at 10 s) is used, then an ImageJ TIFF's constant frame interval, then the constant `--frame-interval`. (Segments outrank ImageJ's interval because Fiji writes a single average interval when it converts a variable-cadence movie.) Real timestamps or segments are written to a `[movie]_frame_times.csv` sidecar and consumed by the **time-aware fits** — the exponential, PDE and mixed-model fits and half-time run on true seconds, so a dense early burst contributes many closely-spaced points and the sparse tail few. Constant-interval movies get no sidecar and behave exactly as before.

---

## Installation

### From PyPI (Recommended)

```bash
# Install core package (command-line tools only)
pip install bleachdt

# Install with GUI support (recommended)
pip install bleachdt[gui]

# Alternative GUI backends
pip install bleachdt[gui-pyqt6]  # PyQt6 instead of PySide6
pip install bleachdt[gui-pyqt5]  # PyQt5 instead of PySide6
```

### Additional Microscopy Format Support

```bash
# Zeiss CZI files
pip install bleachdt[zeiss]

# Leica LIF files
pip install bleachdt[leica]

# Nikon ND2 files (included by default, but explicit)
pip install bleachdt[nikon]

# All formats (Zeiss, Leica, Nikon)
pip install bleachdt[all-formats]

# Everything (GUI + all formats)
pip install bleachdt[all]
```

### From Source

```bash
# Clone the repository
git clone https://github.com/andrewbazley/BleachDt.git
cd BleachDt

# Install in development mode with all features
pip install -e .[all]
```

### Requirements

- **Python**: 3.9 or higher
- **Platforms**: Windows, macOS, Linux
- **Core Dependencies**: numpy, pandas, matplotlib, scipy, scikit-image, tifffile, numba, opencv-python
- **GUI Dependencies**: pyqtgraph + PySide6 (or PyQt6/PyQt5)

### Supported Image Formats

| Format | Extension | Package Required |
|--------|-----------|------------------|
| TIFF | `.tif`, `.tiff` | (included) |
| OME-TIFF | `.ome.tif`, `.ome.tiff` | aicsimageio |
| Nikon ND2 | `.nd2` | nd2 |
| Zeiss CZI | `.czi` | aicsimageio[czi] |
| Leica LIF | `.lif` | aicsimageio[lif] |
| Zeiss LSM | `.lsm` | aicsimageio |
| Olympus | `.oib`, `.oif` | aicsimageio |
| DeltaVision | `.dv`, `.r3d` | aicsimageio |

---

## Quick Start

### Launch the GUI

```bash
# After pip install
bleachdt-gui

# Or run directly from source
python -m bleachdt.gui
```

### Basic Workflow

1. **Set Input Directory** — Point to a folder containing raw movie files
2. **Configure Parameters** — Adjust imaging settings (pixel size, frame interval) and fitting parameters
3. **Run Analysis** — Click "Run Full Analysis" to process all movies
4. **Filter Results** — Use the Filtering tab to interactively exclude outliers
5. **Export** — Filtered data and configurations are automatically saved

### Supported Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Nikon ND2 | `.nd2` | Full support with dimension auto-detection |
| TIFF/TIF | `.tif`, `.tiff` | Raw movies or pre-cropped droplet stacks |
| Zeiss CZI | `.czi` | Requires `pip install bleachdt[zeiss]` |
| Leica LIF | `.lif` | Requires `pip install bleachdt[leica]` |
| OME-TIFF | `.ome.tif` | Via aicsimageio |

The pipeline automatically detects the input type and processes accordingly:
- **Raw movies** (ND2, CZI, etc.) → Stabilize → Crop droplets → Analyze
- **Pre-cropped TIFs** → Analyze directly
- **Existing CSVs** → Combine/filter results

---

## Pipeline Components

All functionality lives in a single `bleachdt` package. The core modules are:

| Module | Console script | Purpose |
|--------|----------------|---------|
| `bleachdt.gui` | `bleachdt-gui` | Main GUI application (orchestrates the full pipeline) |
| `bleachdt.droplet_crop` | `bleachdt-crop` | Detects, crops, and stabilizes droplets from raw movies |
| `bleachdt.crop_pde` | `bleachdt-analysis` | Core analysis engine for PDE fitting and curve extraction |
| `bleachdt.combine` | `bleachdt-combine` | Combines and filters FRAP curves across experiments |
| `bleachdt.condition_utils` | — | Condition / replicate name extraction helpers |
| `bleachdt.r2_scatter` | `bleachdt-r2` | R² summaries for FRAP scatter relationships |
| `bleachdt.pre_mean_vs_radius` | `bleachdt-radius` | Pre-bleach mean intensity vs droplet radius analysis |

---

## GUI Guide

The GUI provides two tabs: **Analysis** and **Filtering**.

### Analysis Tab

The Analysis tab contains collapsible sections for organized parameter control:

#### Core Imaging Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| Pixel Size | 0.11 µm | Physical size of each pixel |
| Frame Interval | 10.0 s | Time between consecutive frames |

#### Image Stabilization
*Applied as first step to the whole movie before droplet cropping. The stabilized full-movie TIF is saved in each movie's results folder as `stabilized_[moviename].tif` (e.g. `Analysis Results/movie_stem_results/stabilized_movie_stem.tif`).*
<details>
<summary>Click to expand parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Stabilization | ✓ | Apply drift correction to entire movie before cropping |
| Algorithm | Lucas-Kanade | Lucas-Kanade (ImageJ-compatible) or ECC (OpenCV) |
| Pyramid Levels | 4 | Multi-scale pyramid levels for coarse-to-fine alignment |
| Max Iterations | 1000 | Maximum iterations per frame |
| Error Tolerance | 1e-8 | Convergence threshold (lower = stricter) |
| Template Update | 0.90 | Template update coefficient (0=fixed, 1=full update) |
| Pre-smooth Sigma | 0.5 | Gaussian smoothing before alignment (0 = none) |
| Max Shift (px) | 25 | Maximum allowed shift in pixels |
| Min Correlation | 0.8 | Minimum correlation to accept alignment (ECC) |
| Refine Iterations | 20 | Refinement passes at full resolution (ECC) |

</details>

#### Droplet Cropping
*Uses: `bleachdt.droplet_crop` (`bleachdt-crop`)*
<details>
<summary>Click to expand parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| Save Droplet TIF Stacks | | Save individual droplet TIF files to droplet_stacks/ |
| Clear Borders | ✓ | Exclude droplets touching image edges |
| Min Area | 500 px | Minimum droplet area to detect |
| Bleach Threshold | 1.2 | Ratio threshold to classify FRAP vs NoBleach |
| Crop Multiplier | 2.5 | Crop box size relative to droplet diameter |
| Min NoBleach Diameter | 5.0 µm | Minimum size for control droplets |
| Min NoBleach PC | 10.0 | Minimum partition coefficient for controls |
| N Jobs | 8 | Parallel workers for cropping |

</details>

#### PDE Fitting
*Uses: `bleachdt.crop_pde` (`bleachdt-analysis`)*
<details>
<summary>Click to expand parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| D Min / D Max | 0.001–2.0 µm²/s | Diffusion coefficient search range |
| D Grid Size | 25 | Coarse grid points for D search |
| D Refine Grid | 100 | Fine grid points for refinement |
| Min Droplet Size | 100 px | Minimum droplet area for analysis |
| Ratio Percentile | 85% | Threshold for bleach ROI detection |
| Max Frames | 0 (all) | Limit frames to process (0 = all) |
| Downsample Factor | 1 | Spatial downsampling (1 = none) |
| PDE Frames | 1–7 | Frame window for PDE fitting |
| Exp Frames | 1–30 | Frame window for exponential fitting |
| Enable PDE Fitting | ✓ | Enable PDE diffusion fitting (uncheck for exponential-only analysis) |
| No Photobleach Correct | ☐ | Skip photobleach correction |
| No Exp Plateau Fit | ☐ | Skip exponential plateau fitting |
| N Jobs | 8 | Parallel workers for processing movie folders |

</details>

#### Detection Parameters
*Uses: `bleachdt.combine` (`bleachdt-combine`)*

These parameters control initial curve quality detection (not to be confused with user-defined filtering in the Filtering tab).

<details>
<summary>Click to expand parameters</summary>

| Parameter | Default | Description |
|-----------|---------|-------------|
| Max Drop | 0.1 | Maximum intensity drop between frames to accept a curve |
| Max Jump | 0.1 | Maximum intensity jump between frames to accept a curve |
| Ignore First N | 4 | Frames to skip in drop/jump check |
| Min Value | 0.1 | Minimum allowed intensity |
| Max Value | 1.2 | Maximum allowed intensity |
| Skip Detection | ☐ | Include all curves without detection thresholds |

</details>

#### Output Options
- **Generate Combined Output Files** — Create combined CSV/XLSX files across all movies

#### Run Buttons
| Button | Action |
|--------|--------|
| **Crop Only** | Run only droplet detection and cropping |
| **Run Full Analysis** | Complete pipeline (crop → fit → combine) |
| **Apply Filtering Settings** | Re-run analysis using Filtering tab selections (enabled after filtering changes) |

#### Configuration Management
- **Save Configuration File** / **Load Configuration File** — Manually save/restore all parameters
- **Auto-save run config** ✓ — Automatically saves parameters + log output after each run

### Filtering Tab

Interactive scatter plot filtering for quality control:

1. **Load Data** — Import `FRAP_condition_summary.csv` or `NoBleach_summary.csv`
2. **Create Plot** — Select X/Y parameters (e.g., `D_um2_per_s` vs `cost`)
3. **Add Filters** — Use horizontal/vertical lines or box selection
4. **Save Filters** — Apply filters to subsequent analyses

After running a full analysis, the filtering tab automatically loads summary CSVs from the analysis directory and displays the source path.

### Visualization Tab

Preview individual FRAP curves from any movie in your dataset:

1. **Browse Directory** — Select folder containing analyzed data
2. **Select Movie** — Choose a movie from the dropdown
3. **View Curves** — See all FRAP curves plotted together
4. **Apply Filters** — Optionally show only curves that pass filtering parameters

This is useful for quality control to verify filtering parameters are working as expected.

---

## Command Line

After installing via pip, the following commands are available:

### Launch GUI

```bash
bleachdt-gui
```

### Full Analysis

```bash
bleachdt-analysis /path/to/movies \
    --pixel-size 0.11 \
    --frame-interval 10.0 \
    --D-min 0.001 --D-max 2.0 \
    --n-jobs-conditions 8
```

### Droplet Cropping Only

```bash
# Stabilization is ON by default using Lucas-Kanade algorithm
bleachdt-crop /path/to/movies \
    --min-area 500 \
    --min-nobleach-diameter 5.0 \
    --min-nobleach-pc 10.0

# To disable stabilization:
bleachdt-crop /path/to/movies --no-stabilize

# To use ECC algorithm instead of Lucas-Kanade:
bleachdt-crop /path/to/movies --stabilize-algorithm ecc
```

### Combine and Filter Curves

```bash
bleachdt-combine /path/to/results \
    --max-drop 0.05 \
    --max-jump 0.05 \
    --min-val 0.2 \
    --output-prefix filtered_
```

<details>
<summary>Running from source (without pip install)</summary>

```bash
python -m bleachdt.gui                              # GUI
python -m bleachdt.crop_pde /path/to/movies ...     # Full analysis
python -m bleachdt.droplet_crop /path/to/movies ... # Cropping
python -m bleachdt.combine /path/to/results ...     # Combine curves
```

</details>

### HPC / Cluster Usage

BleachDt is fully compatible with HPC environments. All scripts support parallel processing and can run entirely from the command line without a GUI.

```bash
# Run on 24 cores
bleachdt-analysis /path/to/movies --n-jobs-conditions 24

# Droplet cropping on 24 cores  
bleachdt-crop /path/to/movies --n-jobs 24

# Combine curves on 24 cores
bleachdt-combine /path/to/results --n-jobs 24
```

Key features for HPC:
- All processing uses Python's `ProcessPoolExecutor` for parallel execution
- No GUI dependencies required for command-line operation
- Progress output to stdout for log monitoring
- Configurable worker count (up to 64 by default)

---

## Parameters

### Primary Measurements

| Parameter | Units | Description |
|-----------|-------|-------------|
| `D_um2_per_s` | µm²/s | Diffusion coefficient from PDE fitting |
| `k_single_exp` | 1/s | Exponential recovery rate constant |
| `half_time_s` | s | Recovery half-time, t½ = ln(2)/k |
| `mobile_fraction` | — | **True** mobile fraction = (YM−Y0)/(1−Y0) computed on the whole-droplet-normalized curve, so it accounts for redistribution relative to the finite droplet pool (not just bleach depth). |
| `immobile_fraction` | — | 1 − mobile_fraction |
| `recovery_amplitude` | — | Raw recovered signal YM−Y0 (reported separately from the true mobile fraction) |
| `mobile_fraction_basis` | — | Which curve the mobile fraction was derived from (`whole_droplet` or, as fallback, `fit_ROI`) |

### Droplet Properties

| Parameter | Units | Description |
|-----------|-------|-------------|
| `pre_mean` | — | Normalized pre-bleach intensity |
| `K_pre` | — | Partition coefficient (dense/dilute) |
| `bleach_fraction` | — | Fraction of droplet area bleached |
| `droplet_radius_um` | µm | Effective droplet radius |

### Fit Quality

| Parameter | Description |
|-----------|-------------|
| `cost` | Weighted sum of squared residuals (lower = better) |
| `success` | Boolean indicating fit convergence |
| `n_frames_fit` | Number of frames used in fitting |

### NoBleach Control Parameters

| Parameter | Description |
|-----------|-------------|
| `K0`, `K_end` | Initial and final partition coefficients |
| `deltaK_rel` | Relative change in partition coefficient |
| `dK_dt` | Rate of partition coefficient change (1/s) |
| `R0`, `R_end` | Initial and final droplet radii (µm) |
| `dR_dt` | Droplet growth rate (µm/s) |

---

## Normalization & Raw Intensities

BleachDt computes two FRAP normalizations for every bleach spot, both in true
integrated-density (`RawIntDen`) space with explicit per-ROI background subtraction
(matching the Fiji total-fluorescence macro):

- **Pre-bleach frame** (`FRAP_norm_prebleach`): the classic FRAP **double normalization**.
  The bleach-spot background-subtracted integrated density is corrected for acquisition
  photobleaching using **in-field NoBleach reference droplets** (their mean intensity
  normalized to the first frame) — never the bleached droplet itself — then normalized to
  its own pre-bleach value. If no NoBleach reference is available the curve is left
  uncorrected and flagged (`pb_source`). The uncorrected ratio and per-frame
  `photobleach_factor` are also saved for transparency.

Background is subtracted **per frame** from the non-droplet region (the conventional
order: local background removed at each time point before normalization).
- **Whole-droplet ratio** (`FRAP_norm_wholedroplet`): `(bleach / whole-droplet)`
  background-subtracted, normalized to the pre-bleach ratio. Accounts for the droplet's
  own total-fluorescence dynamics (e.g. condensate growth / influx).

Both curves **and** the raw `RawIntDen` values (bleach + whole droplet) are *always*
written to each movie's consolidated `FRAP_droplet_timeseries.csv`, so you can re-derive
or re-filter anything later. In the GUI, **Image Stabilization → FRAP normalization**
selects which curve is reported as the headline recovery curve (`prebleach`,
`wholedroplet`, or `both`); on the command line use `--normalization-method`. The PDE
diffusion fit is normalization-independent (it fits the spatial relaxation), so this
choice does not change `D`.

Drift is applied at subpixel precision by default (smooth registration; intensities are
resampled once via bilinear interpolation). If you would rather preserve exact pixel
values at the cost of up to ±0.5 px residual drift, enable **Whole-pixel apply** in the
GUI or pass `--whole-pixel-stabilization` to `bleachdt-crop`.

## Output Files

### Per-Movie Outputs (`Analysis Results/[movie]_results/`)

| File | Description |
|------|-------------|
| `stabilized_[movie].tif` | Stabilized whole-movie stack (when stabilization is on). Subpixel bilinear drift correction by default (smooth, faithful intensities; `--whole-pixel-stabilization` to preserve exact values instead). |
| `stabilization_shifts_[movie].csv` | Per-frame drift log: `dy`, `dx`, shift magnitude, residual RMS, method, application mode — for QC / comparison against Fiji. |
| `droplet_metrics.csv` | **The single consolidated metrics table (quality-filtered).** Merges per-droplet FRAP fit parameters (incl. true `mobile_fraction`, `recovery_amplitude`, `half_time_s`), NoBleach control metrics, droplet X/Y location + geometry, and the scatter R²/p-values — one droplet-keyed row each. (`droplet_locations.csv` is folded in and removed.) |
| `droplet_metrics_all.csv` | Same table but for **every detected droplet, pre-filter** — the raw copy so nothing is lost. |
| `FRAP_droplet_timeseries.csv` | **Filtered** FRAP droplets' per-frame curves in one long CSV (`droplet` column). Always includes raw bleach/droplet `RawIntDen`, BG-subtracted integrated density, the `photobleach_factor`, and **both** normalized curves. |
| `FRAP_droplet_timeseries_unfiltered.csv` | Same long CSV for **all** FRAP droplets (raw copy). |
| `NoBleach_droplet_timeseries.csv` | All **NoBleach** control droplets' per-frame K(t)/R(t) in one long CSV. NoBleach controls are never quality-filtered. |
| `Pre-bleach_stacked_all.csv` / `Whole-droplet_stacked_all.csv` | **Filtered** FRAP curves per normalization method, **transposed**: one row per bleach point, one column per frame. Raw copies: `*_stacked_all_unfiltered.csv`. |
| `Pre-bleach_averaged_all.csv` / `Whole-droplet_averaged_all.csv` | **Per-movie** averaged recovery curve, per normalization method: `frame_index`, `time_s`, then one column per movie (the mean of that movie's bleach points). Raw copies: `*_averaged_all_unfiltered.csv`. |
| `FRAP_condition_summary.csv` / `NoBleach_summary.csv` | Per-droplet fit table (filtered) / control table (retained for the Filtering tab; data also in `droplet_metrics.csv`). Raw copy: `FRAP_condition_summary_unfiltered.csv`. |
| `droplet_stacks/` | (Optional) Cropped droplet TIF stacks; only written if "Save droplet TIF stacks" is checked. |

Optional diagnostic plots (toggled by GUI checkboxes / CLI flags, applied per-movie and combined): `FRAP_frap_fit_separate.tif` (off), `FRAP_K_exp_scatters_combined.tif` (off), `FRAP_noise_pixels.tif` (off), `NoBleach_K_curves.tif` (on). The per-droplet `summary/` folder is no longer created.

### Combined Outputs (`Analysis Results/`)

| File | Description |
|------|-------------|
| `droplet_metrics.csv` / `droplet_metrics_all.csv` | Combined consolidated metrics across all movies (adds a `movie` column) — filtered / raw all-droplet. |
| `Pre-bleach_stacked_all.csv` / `Whole-droplet_stacked_all.csv` | **Filtered** FRAP curves across all movies, per normalization method, transposed (rows = bleach points, columns = frames). Raw copies: `*_stacked_all_unfiltered.csv`. |
| `Pre-bleach_averaged_all.csv` / `Whole-droplet_averaged_all.csv` | Per-movie averaged curve across all movies (one column per movie), per normalization method. Raw copies: `*_averaged_all_unfiltered.csv`. |
| `FRAP_condition_summary.csv` / `FRAP_condition_summary_unfiltered.csv` | Combined per-droplet fit summary — filtered / raw all-droplet. |
| `stacked_all_Filtered.xlsx` | Interactively-filtered curves (from the Filtering tab). |

> **Automatic quality filtering.** The main CSVs above hold only curves that pass two automatic filters: the **K-exp ±2 SD outlier** filter plus an **arithmetic** jump/drop/value-range filter (`--arith-*` flags; `--no-arithmetic-filter` to disable). A safety guard skips the arithmetic filter for a condition if it would remove more than half the curves (thresholds ill-suited to the data). The raw `*_all` / `*_unfiltered` copies always keep every detected droplet, so nothing is lost. The interactive Filtering tab refines further on top.

> Eliminated: the combined `all_frap_curves.csv`, `stacked_all.csv`, `averaged_all.csv`, and the top-level `FRAP_curves.tif` overlay — superseded by the transposed per-normalization stacked + averaged CSVs above.

### Run Logs

| File | Description |
|------|-------------|
| `run_config_*_YYYYMMDD_HHMMSS.json` | Complete parameter record + log output |

---

## Tips

1. **Start broad, then filter** — Run analysis first, then use scatter plots to identify and exclude outliers
2. **Check fit quality** — Plot `D_um2_per_s` vs `cost` to identify failed fits
3. **Organize by movie** — Each movie folder generates its own `_results` output folder
4. **Save configurations** — Use "Save Configuration File" before long runs for reproducibility
5. **Monitor progress** — The log output shows real-time processing status

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Analysis failed (return code) | Open the run config JSON in the input folder; check the end of `log_output` for the actual error. Common causes: Droplet_Crop exception (e.g. image format not T,Y,X), or Crop_PDE "No TIF files found" (no droplets detected). |
| "Droplet cropping produced no droplet TIF files" | No droplets were detected in any movie. Adjust Bleach threshold, Min area, or check that stacks are (time, Y, X) and contain visible droplets. |
| "No CSV files found" | Ensure `all_frap_curves.csv` exists in result folders |
| "PyQtGraph not available" | Install with `pip install pyqtgraph` |
| Slow performance | Reduce D grid size or increase downsampling |
| Missing movies in output | Check that TIF files exist in movie folders or droplet_stacks/ |

---

## Citation

If you use BleachDt in your research, please cite:

> Bazley, A. (2026). BleachDt: A Python toolkit for FRAP diffusion analysis. GitHub repository: https://github.com/andrewbazley/BleachDt

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

Andrew Bazley
andrew.bazley@nyulangone.org