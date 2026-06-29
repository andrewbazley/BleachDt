#!/usr/bin/env python3
"""
Calculate R^2 values for linear regressions from FRAP scatter plot data.
Reads all FRAP_condition_summary.csv files and calculates R^2 for:
- k_single_exp vs pre_mean
- k_single_exp vs K_pre
- k_single_exp vs bleach_fraction
"""

from __future__ import annotations

import os
import glob
import csv
import argparse
import numpy as np
from scipy import stats

def calculate_r2(x, y):
    """Calculate R^2 for linear regression between x and y."""
    # Filter out NaN and infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    if len(x_clean) < 2:
        return np.nan, np.nan  # R^2, p-value
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r_squared = r_value ** 2
    
    return r_squared, p_value

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Calculate R^2 / p-values for FRAP scatter relationships "
            "(k_single_exp vs pre_mean, K_pre, and bleach_fraction) across all "
            "FRAP_condition_summary.csv files under a results directory."
        )
    )
    parser.add_argument(
        "base_dir",
        help="Root directory containing <condition>/<*_results>/FRAP_condition_summary.csv files.",
    )
    args = parser.parse_args(argv)
    base_dir = args.base_dir

    # Find all FRAP_condition_summary.csv files (handle both old "results" and new "*_results" folder names)
    pattern1 = os.path.join(base_dir, "*/*_results/FRAP_condition_summary.csv")
    pattern2 = os.path.join(base_dir, "*/results/FRAP_condition_summary.csv")
    csv_files = sorted(glob.glob(pattern1) + glob.glob(pattern2))
    
    if not csv_files:
        print(f"No FRAP_condition_summary.csv files found in {base_dir}")
        return
    
    print(f"Found {len(csv_files)} condition summary files")
    
    # Collect all data from "Separate" rows
    all_data = []
    
    for csv_file in csv_files:
        condition_name = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
        print(f"Processing {condition_name}...")
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Models', '').strip() == 'Separate':
                        # Extract relevant columns
                        try:
                            k_exp = float(row.get('k_single_exp', 'nan'))
                            pre_mean = float(row.get('pre_mean', 'nan'))
                            k_pre = float(row.get('K_pre', 'nan'))
                            bleach_frac = float(row.get('bleach_fraction', 'nan'))
                            filename = row.get('filename', '')
                            
                            # Only include if k_exp is valid (successful fit)
                            if np.isfinite(k_exp):
                                all_data.append({
                                    'condition': condition_name,
                                    'filename': filename,
                                    'k_single_exp': k_exp,
                                    'pre_mean': pre_mean,
                                    'K_pre': k_pre,
                                    'bleach_fraction': bleach_frac,
                                })
                        except (ValueError, KeyError) as e:
                            continue
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
    
    if not all_data:
        print("No valid 'Separate' model data found!")
        return
    
    print(f"\nTotal valid data points: {len(all_data)}")
    
    # Group data by condition
    conditions_data = {}
    for d in all_data:
        cond = d['condition']
        if cond not in conditions_data:
            conditions_data[cond] = []
        conditions_data[cond].append(d)
    
    # Calculate R^2 values for each condition
    results = []
    
    print("\n" + "="*80)
    print("R^2 Values for Linear Regressions (by condition):")
    print("="*80)
    
    for condition in sorted(conditions_data.keys()):
        cond_data = conditions_data[condition]
        
        # Extract arrays for this condition
        k_exp = [d['k_single_exp'] for d in cond_data]
        pre_mean = [d['pre_mean'] for d in cond_data]
        k_pre = [d['K_pre'] for d in cond_data]
        bleach_frac = [d['bleach_fraction'] for d in cond_data]
        
        # Calculate R^2 values
        r2_pre_mean, p_pre_mean = calculate_r2(pre_mean, k_exp)
        r2_k_pre, p_k_pre = calculate_r2(k_pre, k_exp)
        r2_bleach_frac, p_bleach_frac = calculate_r2(bleach_frac, k_exp)
        
        # Store results
        results.append({
            'condition': condition,
            'relationship': 'k_single_exp vs pre_mean',
            'R_squared': r2_pre_mean,
            'p_value': p_pre_mean,
            'n_points': len(cond_data)
        })
        results.append({
            'condition': condition,
            'relationship': 'k_single_exp vs K_pre',
            'R_squared': r2_k_pre,
            'p_value': p_k_pre,
            'n_points': len(cond_data)
        })
        results.append({
            'condition': condition,
            'relationship': 'k_single_exp vs bleach_fraction',
            'R_squared': r2_bleach_frac,
            'p_value': p_bleach_frac,
            'n_points': len(cond_data)
        })
        
        # Print results for this condition
        print(f"\n{condition} (n={len(cond_data)}):")
        print(f"  k_single_exp vs pre_mean:           R^2 = {r2_pre_mean:.6f}, p = {p_pre_mean:.6e}")
        print(f"  k_single_exp vs K_pre:              R^2 = {r2_k_pre:.6f}, p = {p_k_pre:.6e}")
        print(f"  k_single_exp vs bleach_fraction:    R^2 = {r2_bleach_frac:.6f}, p = {p_bleach_frac:.6e}")
    
    print("="*80)
    
    # Save to CSV
    output_csv = os.path.join(base_dir, "scatter_plot_R2_summary.csv")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['condition', 'relationship', 'R_squared', 'p_value', 'n_points']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'condition': r['condition'],
                'relationship': r['relationship'],
                'R_squared': f"{r['R_squared']:.6f}" if np.isfinite(r['R_squared']) else 'nan',
                'p_value': f"{r['p_value']:.6e}" if np.isfinite(r['p_value']) else 'nan',
                'n_points': r['n_points']
            })
    
    print(f"\nResults saved to: {output_csv}")
    
    # Also save detailed data for reference
    detailed_csv = os.path.join(base_dir, "scatter_plot_raw_data.csv")
    with open(detailed_csv, 'w', newline='') as f:
        fieldnames = ['condition', 'filename', 'k_single_exp', 'pre_mean', 'K_pre', 'bleach_fraction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"Raw data saved to: {detailed_csv}")

if __name__ == "__main__":
    main()

