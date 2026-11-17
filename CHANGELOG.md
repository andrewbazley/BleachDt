# Changelog

## 4.3.0

### Added
- Spot-based bleach detection: the pre/post ratio image is thresholded (adaptive IsoData with a fixed floor and a minimum bleach depth) into bleach spots, and any droplet a spot overlaps is FRAP. Large droplets are no longer diluted to NoBleach.
- Per-frame timing from ND2 or OME-TIFF metadata, or from `--timing-segments` (e.g. `300x0.1,35x10`), written to a `[movie]_frame_times.csv` sidecar and used by the exponential, PDE and mixed-model fits.
- Range stabilization referenced to frame 0 (`--stabilize-frame-start/-end`, `--stabilize-extrapolation hold|linear|poly`).
- Automatic quality filtering (K-exp ±2 SD outliers plus an arithmetic jump/drop/value-range filter) of the main outputs, with `*_unfiltered` / `*_all` copies always written.
- `--no-pde` skips the diffusion grid search entirely; the mixed model is a separate option that requires PDE.
- `--bleach-depth` measurement ROI, `--min-pc` / `--min-diameter` gates for all droplets, `--save-threshold-masks` QC TIF.
- Optional second-channel analysis using the analysis-channel ROIs.
- Reorganised GUI (four sections, single process-number control), app icon and wordmark.

### Changed
- Every detected droplet is kept: border clearing is off by default, neighbour-in-window rejection is removed, edge-touching droplets are flagged (`touches_image_edge`).
- Detection, partition coefficient and the ratio image use the mean of the pre-bleach frames.
- The single-exponential plateau fit is bounded (plateau near the observed maximum, k ≥ 0), and mobile/immobile fractions are clamped to [0, 1].
- `*_averaged_all.csv` holds one averaged curve per movie.
- `--max-frames` defaults to -1 (last frame).

### Fixed
- The arithmetic filter now examines only post-bleach frames. With more than four pre-bleach frames the bleach drop used to flag every curve, and the >50% guard then switched the filter off.
- Two-channel runs lost droplet location and geometry in `droplet_metrics.csv`.
- A single dead or hot pixel could push the adaptive bleach threshold above every real spot, calling all droplets NoBleach.
- A constant-interval movie wrote a 1 s/frame timing sidecar that overrode `--frame-interval` in the analysis stage. A sidecar is now written only for real timestamps or explicit segments, and a stale one is removed.
- OME-TIFF timing used every plane (all channels and Z), not one per timepoint, and ignored `DeltaTUnit`; ImageJ `finterval` ignored its time unit.
- `--timing-segments` now overrides a constant ImageJ frame interval, which Fiji writes as an average for variable-cadence movies.
- Malformed or empty `--timing-segments` specs warn instead of failing silently or raising.
- An empty stabilization frame range warns and falls back to the full stack instead of silently disabling stabilization.
- Polynomial drift extrapolation is pinned to zero shift at frame 0.
- The mixed model used a uniform time grid for variable-cadence movies.
- `crop_pde`'s internal crop step now forwards pixel size, frame interval, pre-bleach frames, analysis channel and timing segments, and uses the current detection and stabilization defaults.
- GUI: stabilization epsilon was always 0; "Crop Only" ignored most crop settings (it now builds the same command as a full run); config save/load and the saved run config omitted the newer controls.
- When every curve is filtered out, a previous run's main stacked/averaged CSVs are removed rather than left in place.
- The ND2 read-error message always reported the `nd2` error as "not installed".

## 4.1.1

- Restored Python 3.9 compatibility (`from __future__ import annotations` in every module).

## 4.1.0

- Package restructured into `bleachdt/`; consolidated outputs, dual normalization, photobleach-corrected pre-bleach normalization, two-channel planning.
