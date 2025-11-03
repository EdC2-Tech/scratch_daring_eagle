# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 19:21:25 2025

@author: ChEdw
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import random

# ============================================================================
# USER CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================

# Folder containing CSV files
INPUT_FOLDER = "016_Q8_175_5000_short"

# Number of CSV files to randomly select and plot
N_read = 10

# Column name for x-axis
X_COLUMN = "time"

# Column names for y-axis (can specify multiple columns to plot on same graph)
Y_COLUMNS = ["TL14s1"]

# X-axis label for the plot
X_AXIS_LABEL = "Time (arbitrary)"

# Optional: File pattern to match specific CSV files
FILE_PATTERN = "*.csv"

# Maximum number of files to consider (hard limit)
MAX_FILES = 5000

# Plot settings
FIGURE_SIZE = (14, 10)  # Width, height in inches
PLOT_COLUMNS = 2  # Number of columns in subplot grid

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def plot_random_csv_files(
    input_folder,
    n_read,
    x_column,
    y_columns,
    x_axis_label,
    file_pattern="*.csv",
    max_files=5000,
    figsize=(14, 10),
    plot_cols=2
):
    """
    Read random CSV files and plot specified columns.
    All files are plotted on the same plot.
    
    Args:
        input_folder: Path to folder containing CSV files
        n_read: Number of random files to plot
        x_column: Column name for x-axis
        y_columns: List of column names for y-axis
        x_axis_label: Label for x-axis
        file_pattern: Pattern to match CSV files
        max_files: Maximum number of files to consider
        figsize: Figure size as (width, height) tuple
        plot_cols: Number of columns in subplot grid (not used, kept for compatibility)
    """
    
    # Ensure y_columns is a list
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    
    # Get all CSV files from input folder
    csv_files = glob.glob(os.path.join(input_folder, file_pattern))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files in folder")
    
    # Limit to max_files
    if len(csv_files) > max_files:
        print(f"Limiting to first {max_files} files")
        csv_files = csv_files[:max_files]
    
    # Validate n_read
    if n_read > len(csv_files):
        print(f"WARNING: N_read ({n_read}) is greater than available files ({len(csv_files)})")
        n_read = len(csv_files)
        print(f"Setting N_read to {n_read}")
    
    # Randomly select n_read files
    selected_files = random.sample(csv_files, n_read)
    
    print(f"\nRandomly selected {n_read} files to plot")
    print(f"X-axis column: {x_column}")
    print(f"Y-axis columns: {y_columns}")
    print(f"X-axis label: {x_axis_label}\n")
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot each selected file
    successful_plots = 0
    
    for idx, file_path in enumerate(selected_files):
        filename = os.path.basename(file_path)
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            print(f"[{idx + 1}/{n_read}] Plotting: {filename}")
            
            # Check if x_column exists
            if x_column not in df.columns:
                print(f"  WARNING: X-column '{x_column}' not found. Available columns: {list(df.columns)}")
                continue
            
            # Get x data
            x_data = df[x_column]
            
            # Plot each y column
            plots_made = False
            for y_col in y_columns:
                if y_col in df.columns:
                    ax.plot(x_data, df[y_col], marker='o', markersize=2, 
                           label=f"{filename} - {y_col}", linewidth=1.5, alpha=0.7)
                    plots_made = True
                else:
                    print(f"  WARNING: Y-column '{y_col}' not found in {filename}")
            
            if plots_made:
                successful_plots += 1
            
        except Exception as e:
            print(f"  ERROR plotting {filename}: {e}")
    
    # Set labels and title
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"Combined Plot of {successful_plots} Random CSV Files", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Total files found: {len(csv_files)}")
    print(f"  Files plotted: {n_read}")
    print(f"  Successful plots: {successful_plots}")
    print(f"  Failed plots: {n_read - successful_plots}")
    print(f"{'='*70}")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    
    # Validate inputs
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Folder '{INPUT_FOLDER}' does not exist!")
        print("Please update INPUT_FOLDER with a valid path.")
    else:
        plot_random_csv_files(
            input_folder=INPUT_FOLDER,
            n_read=N_read,
            x_column=X_COLUMN,
            y_columns=Y_COLUMNS,
            x_axis_label=X_AXIS_LABEL,
            file_pattern=FILE_PATTERN,
            max_files=MAX_FILES,
            figsize=FIGURE_SIZE,
            plot_cols=PLOT_COLUMNS
        )