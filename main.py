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

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Number of Particles to create
NUM_PARTICLES = 25

# Folder containing CSV files
CSV_FOLDER = "./016_Q8_175_5000_short/"
CSV_FILE = "histories_264.csv"

# Columns to track
COLS = ["FL1", "TL14s1"]

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

# Short stop option
SHORT_STOP = 1000

# ============================================================================
# MAIN PROGRAM
# ============================================================================



# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    
    # Setup swarm of particles

    print(f"\nCreating {NUM_PARTICLES} random Particles...\n")
    
    # Create list of Particles
    pt_swarm = Swarm(
        num_particles=NUM_PARTICLES,
        index_range=(INDEX_MIN, INDEX_MAX),
        folder_path=CSV_FOLDER,
        selected_cols=COLS
    )
    
    pt_swarm.set_threshold(0.8)
    pt_swarm.set_population_cut(0.5)
    
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
    mean_particles = list()
    std_particles = list()
    # Iterate over all rows and store each as a dict
    for index, row in df_filtered.iterrows():
        # Convert row to dictionary
        row_dict = row.to_dict()
        
        pt_swarm.predict_all()
        pt_swarm.calculate_score_all(row_dict)
        pt_swarm.repopulate()
        pt_swarm.forward_all()
        
        mean_particles.append(pt_swarm.get_mean_pred(col=COLS, cutoff=0.25))
        store_particles.append(pt_swarm.get_current())
        std_particles.append(pt_swarm.get_std_pred(col=COLS, cutoff=0.25))
        
        # Show progress
        if index%10 == 0:
            print(f"Progression: {index}/{len(df_filtered)}")
            
        # Short stop (remove when finished debugging)
        if index == SHORT_STOP:
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

    
    X = np.linspace(0, SHORT_STOP, num=SHORT_STOP+1)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Plotting mean and standard deviation
    
    
    # Extract data from list of dictionary output
    mean = list()
    std  = list()
    for col in COLS:
        mean = [d[col] for d in mean_particles]
        std  = [d[col] for d in std_particles]
    
        lower_bound = np.array(mean)-np.array(std)
        upper_bound = np.array(mean)+np.array(std)
        
        plt.figure()
        plt.plot(X, mean, color='blue', label='Mean')
        plt.plot(X, df_filtered[col][0:SHORT_STOP+1], c="red", label="True")
        plt.fill_between(X, lower_bound, upper_bound, color="gray", alpha=0.3)
    
        # Labels and legend
        plt.title('Line Plot with Mean and Standard Deviation')
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        plt.figure()
        plt.scatter(X, df_filtered[col][0:SHORT_STOP+1], c="red", s=5, alpha=0.2)
        plt.scatter(X, mean, c='g', s=5, alpha=0.3)
        plt.xlabel("Time Step", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title("Scatter Plot of Values", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.show()