# BleachDt

<p align="center">
  <strong>A comprehensive Python toolkit for analyzing FRAP experiments in phase-separated droplets</strong>
</p>

<p align="center">
  <em>The name "BleachDt" (pronounced "bleached") is a nod to the PDE-based diffusion coefficient (D) fitting at the heart of this analysis.</em>
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

- **Automated Droplet Detection** — Identifies and crops individual droplets from multi-droplet image stacks
- **PDE-Based Diffusion Fitting** — Fits diffusion coefficients using partial differential equation models
- **Dual Model Support** — Choose between "Separate" (independent PDE + exponential) or "Combined" (constrained) fitting models
- **Interactive GUI** — User-friendly interface with real-time scatter plot filtering
- **Batch Processing** — Parallel processing across multiple conditions and experiments
- **Automatic Configuration Saving** — Every run saves its parameters and log output for reproducibility

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

- **Python**: 3.7 or higher
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
python combine_frap_curves_gui.py
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

| Script | Purpose |
|--------|---------|
| `combine_frap_curves_gui.py` | Main GUI application |
| `Droplet_Crop.py` | Detects, crops, and stabilizes droplets from raw movies |
| `Crop_PDE_Cursor_V7.py` | Core analysis engine for PDE fitting and curve extraction |
| `combine_frap_curves.py` | Combines and filters FRAP curves across experiments |

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
*Uses: `Droplet_Crop.py`*
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
*Uses: `Crop_PDE_Cursor_V7.py`*
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
*Uses: `combine_frap_curves.py`*

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
python combine_frap_curves_gui.py                    # GUI
python Crop_PDE_Cursor_V7.py /path/to/movies ...     # Full analysis
python Droplet_Crop.py /path/to/movies ...           # Cropping
python combine_frap_curves.py /path/to/results ...   # Combine curves
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
| `mobile_fraction` | — | Fraction of mobile molecules (0–1) |
| `immobile_fraction` | — | Fraction of immobile molecules (0–1) |

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

## Output Files

### Per-Movie Outputs (`Analysis Results/[movie]_results/`)

| File | Description |
|------|-------------|
| `stabilized_[movie].tif` | Stabilized whole-movie stack (when stabilization is on) |
| `all_frap_curves.csv` | Individual recovery curves (columns = droplets) |
| `FRAP_condition_summary.csv` | Fit parameters per droplet |
| `NoBleach_summary.csv` | Control droplet measurements |
| `NoBleach_summary.txt` | Movie-level classification |
| `FRAP_curves.tif` | Visual plot of all curves |
| `*_frap_fit_*.tif` | Diagnostic fit plots |
| `droplet_stacks/` | (Optional) Cropped droplet TIF stacks; omitted if "Save droplet TIF stacks" is unchecked |

### Combined Outputs (`Analysis Results/`)

| File | Description |
|------|-------------|
| `stacked_all.csv` | All curves stacked with movie labels |
| `averaged_all.csv` | Averaged curves per movie |
| `FRAP_curves.tif` | Combined curves visualization |
| `stacked_all_Filtered.xlsx` | Filtered curves (from Filtering tab) |

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