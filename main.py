# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 18:57:29 2025

@author: ChEdw
"""

import pandas as pd
import numpy as np
import random
import glob
import os
import matplotlib.pyplot as plt

from swarm import Swarm
from swarm import get_csv_files_from_folder

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Number of DataPoints to create
X = 10

# Folder containing CSV files
CSV_FOLDER = "./016_Q8_175_5000_short/"
CSV_FILE = "histories_264.csv"

# Columns to track
COLS = ["FL1"]

# Valid Columns    
# FL1 = Pump 1 flow rate
# FL6 = Pump 2 flow rate
# TL14s1 = Upper plenum temperature
# TA21s1 = Fuel centerline temperature
# PS1 = Pump 1 pump speed
# PS2 = Pump 2 pump speed
# TFL = Total core flow rate (FL1 + FL6)

# Range for random index values
INDEX_MIN = 0
INDEX_MAX = 2490  # Assuming 2000 datapoints per file

# ============================================================================
# MAIN PROGRAM
# ============================================================================



# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    
    # Setup swarm of particles
    # Get list of CSV files from folder
    csv_tags = get_csv_files_from_folder(CSV_FOLDER)
    
    if not csv_tags:
        print("No CSV files found. Exiting.")
        exit()
    
    print(f"\nCreating {X} random Particles...\n")
    
    # Create list of Particles
    pt_swarm = Swarm(
        num_points=X,
        tags_list=csv_tags,
        index_range=(INDEX_MIN, INDEX_MAX),
        path=CSV_FOLDER,
        cols=COLS
    )
    
    pt_swarm.set_threshold(0.05)
    
    # Test for tracking algorithm
    
    # Load the CSV file
    df = pd.read_csv(CSV_FOLDER+CSV_FILE)
    
    print(f"Loaded CSV file: {CSV_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Check which columns exist
    available_columns = [col for col in COLS if col in df.columns]
    missing_columns = [col for col in COLS if col not in df.columns]

    if missing_columns:
        print(f"WARNING: Missing columns: {missing_columns}")
    
    if not available_columns:
        print("ERROR: None of the specified columns found")
        exit()
    
    # Filter dataframe to only keep specified columns
    df_filtered = df[available_columns]

    store_particles = list()
    
    # Iterate over all rows and store each as a dict
    for index, row in df_filtered.iterrows():
        # Convert row to dictionary
        row_dict = row.to_dict()
        
        pt_swarm.score_swarm(row_dict)
        pt_swarm.repopulate()
        
        
        store_particles.append(pt_swarm.get_current())
        
        # Show progress
        if index%10 == 0:
            print(f"Progression: {index}/{len(df_filtered)}")
            
        # Short stop (remove when finished debugging)
        if index == 1000:
            print(f"Short stop engaged")
            break

    # Get N
    N = len(store_particles)
    
    print(f"Plotting scatter plot for {N} x-positions")
    
    # Create scatter plot
    plt.figure(figsize=(12, 6))
    
    # For each key, plot all values
    for key in COLS:
        x_values = []
        y_values = []
        
        # Iterate through each list position (x-axis)
        for x_pos, dict_list in enumerate(store_particles):
            # Extract all values for this key at this x position
            for d in dict_list:
                if key in d:
                    x_values.append(x_pos)
                    y_values.append(d[key])
        
        # Plot scatter for this key
        plt.scatter(x_values, y_values, alpha=0.6, s=5)
        
        print(f"Plotted {len(y_values)} points for key '{key}'")
    
    # Labels and title
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Scatter Plot of Values", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.scatter(np.linspace(0, 1000, num=1001), df_filtered[0:1001], c="red", s=5)
    
    # Show plot
    plt.tight_layout()
    plt.show()