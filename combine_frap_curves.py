#!/usr/bin/env python3
"""
Combine FRAP curves from multiple all_frap_curves.csv files.

Recursively scans an input folder for all_frap_curves.csv files, combines them,
applies detection thresholds (drop/jump checks), and generates output CSVs:
1. Stacked CSV: All curves stacked vertically with movie label column
2. Averaged CSV: Each movie averaged, columns next to each other
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import tifffile as tiff
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False


def filter_jump_drop(df: pd.DataFrame,
                     max_drop: float = 0.1,
                     max_jump: float = 0.1,
                     ignore_first_n: int = 4) -> pd.DataFrame:
    """
    Filter curves based on jump/drop detection (independent of curve fitting).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with FRAP data (columns are curves, rows are frames)
    max_drop : float, default 0.1
        Maximum allowed drop per frame
    max_jump : float, default 0.1
        Maximum allowed jump per frame
    ignore_first_n : int, default 4
        Number of initial frames to ignore in jump/drop analysis
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only valid columns
    """
    valid_columns = []
    
    for col in df.columns:
        values = df[col].values
        
        # Skip if no valid data
        if len(values) < 2 or np.all(np.isnan(values)):
            continue
        
        # Check for excessive jumps/drops (using absolute values)
        valid = True
        for i in range(ignore_first_n, len(values) - 1):
            if np.isnan(values[i]) or np.isnan(values[i+1]):
                continue
                
            diff = values[i+1] - values[i]
            abs_diff = abs(diff)
            
            # Check for excessive drop (negative change)
            if diff < 0 and abs_diff > abs(max_drop):
                valid = False
                break
            
            # Check for excessive jump (positive change)
            if diff > 0 and abs_diff > abs(max_jump):
                valid = False
                break
        
        if valid:
            valid_columns.append(col)
    
    return df[valid_columns].copy()


def filter_value_range(df: pd.DataFrame,
                      min_val: float = 0.1,
                      max_val: float = 1.2) -> pd.DataFrame:
    """
    Filter curves based on value range constraints.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    min_val : float, default 0.1
        Minimum allowed value
    max_val : float, default 1.2
        Maximum allowed value
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    valid_columns = []
    
    for col in df.columns:
        values = df[col].values
        
        # Skip if no valid data
        if len(values) == 0 or np.all(np.isnan(values)):
            continue
        
        # Check if all values are within range
        valid_mask = ~np.isnan(values)
        if np.any(valid_mask):
            valid_values = values[valid_mask]
            if np.all((valid_values >= min_val) & (valid_values <= max_val)):
                valid_columns.append(col)
    
    return df[valid_columns].copy()


def find_all_frap_curves_csvs(root_dir):
    """
    Recursively find all all_frap_curves.csv files from per-movie result folders only.
    Excludes the combined all_frap_curves.csv at the Analysis Results root so that
    stacked_all.csv compiles each curve once, labeled by the movie it came from.
    """
    root_dir_abs = os.path.abspath(root_dir)
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        if 'all_frap_curves.csv' not in files:
            continue
        # Skip the combined file at the search root (e.g. Analysis Results/all_frap_curves.csv)
        if os.path.abspath(root) == root_dir_abs:
            continue
        csv_path = os.path.join(root, 'all_frap_curves.csv')
        csv_files.append(csv_path)
    return sorted(csv_files)


def read_and_process_csv(csv_path):
    """
    Read a single all_frap_curves.csv file and return as DataFrame.
    
    Format: columns are movie names, rows are frames.
    Returns: DataFrame with columns as curves, rows as frames.
    """
    try:
        df = pd.read_csv(csv_path, index_col=None)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def transpose_and_label(df, movie_name):
    """
    Transpose dataframe and add label column.
    
    Input: columns are movie names, rows are frames
    Output: columns are frames (Mean1, Mean2, ...), rows are individual curves,
            with a 'label' column containing the movie name
    
    Note: Each column in the input represents one curve from that movie.
    """
    if df is None or df.empty:
        return None
    
    # Transpose: rows (frames) become columns, columns (curves) become rows
    df_t = df.T
    
    # Reset index
    df_t.reset_index(drop=True, inplace=True)
    
    # Add label column
    df_t.insert(0, 'label', movie_name)
    
    # Rename frame columns to Mean1, Mean2, etc.
    frame_cols = [col for col in df_t.columns if col != 'label']
    new_frame_cols = {col: f'Mean{i+1}' for i, col in enumerate(frame_cols)}
    df_t.rename(columns=new_frame_cols, inplace=True)
    
    return df_t


def apply_filters(df, max_drop=0.1, max_jump=0.1, ignore_first_n=4, 
                 min_val=0.1, max_val=1.2):
    """
    Apply detection thresholds to filter curves with abnormal intensity changes.
    
    Note: df should have columns as curves and rows as frames.
    """
    if df is None or df.empty:
        return df
    
    # Apply jump/drop filter
    df_filtered = filter_jump_drop(df, max_drop=max_drop, max_jump=max_jump, 
                                   ignore_first_n=ignore_first_n)
    
    # Apply value range filter
    df_filtered = filter_value_range(df_filtered, min_val=min_val, max_val=max_val)
    
    return df_filtered


def plot_all_frap_curves(stacked_df, output_dir, frame_interval=10.0, output_prefix=""):
    """
    Plot all FRAP curves starting from post-bleach frame (frame 1, second frame).
    Similar to NoBleach_r_curves.tif but for FRAP recovery curves.
    
    Parameters
    ----------
    stacked_df : pd.DataFrame
        DataFrame with 'label' column and 'Mean1', 'Mean2', ... columns for frames
    output_dir : str
        Directory to save the plot
    frame_interval : float
        Time interval between frames in seconds (default: 10.0)
    output_prefix : str
        Prefix for output filename (e.g., "filtered_")
    """
    if stacked_df is None or stacked_df.empty:
        return
    
    # Extract frame columns (Mean1, Mean2, ...)
    frame_cols = [col for col in stacked_df.columns if col.startswith('Mean')]
    if not frame_cols:
        return
    
    # Sort frame columns by number
    def get_frame_num(col):
        try:
            return int(col.replace('Mean', ''))
        except:
            return 0
    frame_cols = sorted(frame_cols, key=get_frame_num)
    
    # Extract curves starting from frame 1 (post-bleach, second frame)
    # Frame 0 is pre-bleach, Frame 1 is first post-bleach frame
    if 'Mean1' not in frame_cols:
        return  # Need at least frame 1
    
    # Get all curves (rows) starting from Mean1
    curves_list = []
    labels_list = []
    
    for idx, row in stacked_df.iterrows():
        label = row.get('label', f'curve_{idx}')
        # Extract values from Mean1 onwards
        curve_values = []
        for col in frame_cols:
            val = row.get(col, np.nan)
            if pd.isna(val):
                break
            curve_values.append(float(val))
        
        if len(curve_values) > 0:
            curves_list.append(np.array(curve_values, dtype=np.float32))
            labels_list.append(label)
    
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
    
    # Save as TIF if available, otherwise as PNG
    output_filename = f"{output_prefix}FRAP_curves.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    # Convert figure to array and save
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    # Convert RGBA to RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]
    plt.close(fig)
    
    if TIFF_AVAILABLE:
        # Save as TIF
        tiff.imwrite(output_path, img, imagej=True)
        print(f"Saved FRAP curves plot: {output_path}")
    else:
        # Fallback to PNG
        from PIL import Image
        pil_img = Image.fromarray(img)
        png_path = output_path.replace('.tif', '.png')
        pil_img.save(png_path, dpi=(200, 200))
        print(f"Saved FRAP curves plot: {png_path} (PNG format, tifffile not available)")


def process_single_csv(csv_path, max_drop, max_jump, ignore_first_n, 
                       min_val, max_val, no_filter, allowed_filenames=None):
    """
    Process a single CSV file. Designed for parallel execution.
    
    Returns:
        tuple: (movie_name, stacked_rows, averaged_curve) or None if failed
    """
    try:
        # Extract movie name from path
        # Handle two folder structures:
        # 1. V1/V4/V5: .../G226D_ADP_001_results/all_frap_curves.csv
        # 2. V2/V3: .../G226D_ADP_101/results/all_frap_curves.csv
        path_parts = Path(csv_path).parts
        movie_name = None
        csv_dir = os.path.dirname(csv_path)
        
        # Check if we're in a 'results' subfolder (V2/V3 structure)
        if os.path.basename(csv_dir) == 'results':
            # Go up one level to get the movie name
            movie_name = os.path.basename(os.path.dirname(csv_dir))
            # Add '_results' suffix for consistency with V1/V4/V5 naming
            if not movie_name.endswith('_results'):
                movie_name = f"{movie_name}_results"
        else:
            # Look for folder ending in '_results' (V1/V4/V5 structure)
            for i, part in enumerate(path_parts):
                if part.endswith('_results'):
                    movie_name = part
                    break
            
            # Fallback: use the directory containing the CSV
            if movie_name is None:
                movie_name = os.path.basename(csv_dir)
        
        # Read CSV
        df = read_and_process_csv(csv_path)
        if df is None or df.empty:
            return None
        
        # Optionally restrict to allowed filenames (curve names)
        if allowed_filenames:
            allowed_set = set(allowed_filenames)
            existing_cols = [c for c in df.columns if c in allowed_set]
            df = df[existing_cols]
            if df.empty:
                return None
        
        # Apply filters if requested
        if not no_filter:
            df_filtered = apply_filters(df, 
                                       max_drop=max_drop,
                                       max_jump=max_jump,
                                       ignore_first_n=ignore_first_n,
                                       min_val=min_val,
                                       max_val=max_val)
        else:
            df_filtered = df
        
        if df_filtered.empty:
            return None
        
        # Transpose and add label for stacked output
        df_labeled = transpose_and_label(df_filtered, movie_name)
        stacked_rows = []
        if df_labeled is not None:
            for _, row in df_labeled.iterrows():
                stacked_rows.append(row.to_dict())
        
        # Calculate average
        avg_curve = df_filtered.mean(axis=1).values
        
        return (movie_name, stacked_rows, avg_curve)
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Combine and filter FRAP curves from multiple all_frap_curves.csv files."
    )
    parser.add_argument("input_dir", type=str,
                       help="Root directory to recursively search for all_frap_curves.csv files. Outputs will be saved here.")
    parser.add_argument("--max-drop", type=float, default=0.1,
                       help="Maximum allowed drop per frame (default: 0.01)")
    parser.add_argument("--max-jump", type=float, default=0.1,
                       help="Maximum allowed jump per frame (default: 0.05)")
    parser.add_argument("--ignore-first-n", type=int, default=4,
                       help="Number of initial frames to ignore in jump/drop analysis (default: 4)")
    parser.add_argument("--min-val", type=float, default=0.1,
                       help="Minimum allowed value (default: 0.2)")
    parser.add_argument("--max-val", type=float, default=1.2,
                       help="Maximum allowed value (default: 1.0)")
    parser.add_argument("--no-filter", action="store_true",
                       help="Skip filtering (combine all curves without filtering)")
    parser.add_argument("--n-jobs", type=int, default=None,
                       help="Number of parallel workers for processing CSV files. "
                            "Default: number of CPU cores. Set to 1 for sequential processing.")
    parser.add_argument("--frame-interval", type=float, default=10.0,
                       help="Frame interval in seconds for plotting (default: 10.0)")
    parser.add_argument("--output-prefix", type=str, default="",
                       help="Prefix for output filenames (e.g., 'filtered_')")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    
    # Find all CSV files first
    print(f"Searching for all_frap_curves.csv files in {input_dir}...")
    csv_files = find_all_frap_curves_csvs(input_dir)
    
    # Determine output_dir: find "Analysis Results" root
    # If input_dir is already "Analysis Results", use it
    # Otherwise, look for common parent "Analysis Results" folder
    output_dir = input_dir
    if os.path.basename(input_dir) == "Analysis Results":
        output_dir = input_dir
    elif csv_files:
        # Find the "Analysis Results" folder by navigating up from first CSV
        first_csv_dir = os.path.dirname(csv_files[0])
        current = first_csv_dir
        for _ in range(5):  # Look up to 5 levels
            if os.path.basename(current) == "Analysis Results":
                output_dir = current
                break
            parent = os.path.dirname(current)
            if parent == current:  # Reached root
                break
            current = parent
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Combined outputs will be saved to: {output_dir}")
    
    if not csv_files:
        print(f"No all_frap_curves.csv files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} all_frap_curves.csv files")
    
    # Determine number of workers
    import multiprocessing
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    n_jobs = max(1, min(n_jobs, len(csv_files)))  # Don't use more workers than files
    
    print(f"Processing {len(csv_files)} files with {n_jobs} parallel worker(s)...")
    
    # Process CSV files in parallel
    all_stacked_data = []
    all_averaged_data = {}
    
    if n_jobs == 1:
        # Sequential processing (for debugging or single-core systems)
        for csv_path in csv_files:
            result = process_single_csv(csv_path, args.max_drop, args.max_jump,
                                       args.ignore_first_n, args.min_val, args.max_val,
                                       args.no_filter)
            if result is not None:
                movie_name, stacked_rows, avg_curve = result
                all_stacked_data.extend(stacked_rows)
                all_averaged_data[movie_name] = avg_curve
                print(f"Processed: {movie_name}")
    else:
        # Parallel processing
        process_func = partial(process_single_csv,
                              max_drop=args.max_drop,
                              max_jump=args.max_jump,
                              ignore_first_n=args.ignore_first_n,
                              min_val=args.min_val,
                              max_val=args.max_val,
                              no_filter=args.no_filter)
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_csv = {executor.submit(process_func, csv_path): csv_path 
                            for csv_path in csv_files}
            
            # Collect results as they complete
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
                        print(f"Processed {completed}/{len(csv_files)}: {movie_name}")
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
        
        print(f"Successfully processed {completed}/{len(csv_files)} files")
    
    if not all_stacked_data:
        print("No data to write!")
        return
    
    # Determine maximum number of frames
    max_frames = max(len(row) - 1 for row in all_stacked_data)  # -1 for label column
    
    # Create stacked DataFrame
    stacked_df = pd.DataFrame(all_stacked_data)
    
    # Ensure all Mean columns exist (pad with NaN if needed)
    for i in range(1, max_frames + 1):
        col_name = f'Mean{i}'
        if col_name not in stacked_df.columns:
            stacked_df[col_name] = np.nan
    
    # Reorder columns: label first, then Mean1, Mean2, ...
    frame_cols = [f'Mean{i}' for i in range(1, max_frames + 1)]
    stacked_df = stacked_df[['label'] + frame_cols]
    
    # Write stacked CSV
    stacked_output = os.path.join(output_dir, f'{args.output_prefix}stacked_all.csv')
    stacked_df.to_csv(stacked_output, index=False)
    print(f"\nSaved stacked CSV: {stacked_output}")
    print(f"  Total curves: {len(stacked_df)}")
    print(f"  Total frames: {max_frames}")
    
    # Create averaged DataFrame
    if all_averaged_data:
        # Find maximum length
        max_len = max(len(v) for v in all_averaged_data.values())
        
        # Create DataFrame with movies as columns, frames as rows
        averaged_df = pd.DataFrame()
        for movie_name, avg_curve in sorted(all_averaged_data.items()):
            # Pad with NaN if needed
            padded_curve = np.pad(avg_curve, (0, max_len - len(avg_curve)), 
                                  constant_values=np.nan)
            averaged_df[movie_name] = padded_curve
        
        # Rename index to frame numbers (Mean1, Mean2, ...)
        averaged_df.index = [f'Mean{i+1}' for i in range(len(averaged_df))]
        
        # Write averaged CSV (movies as columns, frames as rows)
        averaged_output = os.path.join(output_dir, f'{args.output_prefix}averaged_all.csv')
        averaged_df.to_csv(averaged_output)
        print(f"\nSaved averaged CSV: {averaged_output}")
        print(f"  Total movies: {len(averaged_df.columns)}")
        print(f"  Total frames: {max_len}")
    
    # Generate FRAP curves plot
    print("\nGenerating FRAP curves plot...")
    plot_all_frap_curves(stacked_df, output_dir, frame_interval=args.frame_interval, 
                        output_prefix=args.output_prefix)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

