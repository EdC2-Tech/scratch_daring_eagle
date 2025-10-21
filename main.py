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

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Folder containing the 5000 CSV files
INPUT_FOLDER = "016_Q8_175_5000_short"

# Path to CSV file containing new sensor values
NEW_SENSOR_DATA_FILE = INPUT_FOLDER + "/" + "histories_234.csv"

# Row index to read from the new sensor data file (can be incremented each cycle)
SENSOR_ROW_INDEX = 0

# Number of random points to sample each loop cycle
N = 5

# Refresh rate of samples
M = 0.5

# Maximum number of CSV files to load (up to 5000)
MAX_FILES = 2

# Column name containing sensor values in the CSV files
SENSOR_COLUMNS = ["FL1", "FL6"]

# Number of loop cycles to run (set to None for infinite loop)
NUM_CYCLES = 5

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def load_csv_files(input_folder, max_files=5000):
    """
    Load all CSV files and create unique tags for each.
    
    Returns:
        dict: Dictionary with unique tags as keys and DataFrames as values
    """
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"ERROR: No CSV files found in {input_folder}")
        return {}
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Limit to max_files
    if len(csv_files) > max_files:
        print(f"Loading first {max_files} files")
        csv_files = csv_files[:max_files]
    
    # Load all CSV files with unique tags
    data_dict = {}
    
    for idx, file_path in enumerate(csv_files):
        try:
            filename = os.path.basename(file_path)
            # Create unique tag (using filename without extension)
            unique_tag = os.path.splitext(filename)[0]
            
            # Load CSV
            df = pd.read_csv(file_path)
            data_dict[unique_tag] = df
            
            if (idx + 1) % 500 == 0:
                print(f"  Loaded {idx + 1}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")
    
    print(f"Successfully loaded {len(data_dict)} CSV files\n")
    return data_dict


def initialize_random_points(data_dict, n, sensor_columns):
    """
    Sample N random points from the 5000 CSV files.
    Each sample can come from any file and any row.
    
    Args:
        data_dict: Dictionary of DataFrames with unique tags
        n: Number of random points to sample
        sensor_column: Column name containing sensor values
    
    Returns:
        list: List of tuples (unique_tag, row_index, sensor_value)
    """
    samples = []
    tags = list(data_dict.keys())

    # Ensure sensor_columns is a list
    if isinstance(sensor_columns, str):
        sensor_columns = [sensor_columns]
        
    for _ in range(n):
        # Randomly select a file
        random_tag = random.choice(tags)
        df = data_dict[random_tag]
        
        # Randomly select a row (datapoint)
        random_row_idx = random.randint(0, len(df) - 1)
        
        # Extract all sensor column values
        sensor_values = {}
        for col in sensor_columns:
            if col in df.columns:
                sensor_values[col] = df.loc[random_row_idx, col]
        
        # Only add sample if at least one sensor column exists
        # 1: Unique identifier of file, 2: Index from file, 3: values, 4: MSE default=0
        if sensor_values:
            samples.append((random_tag, random_row_idx, sensor_values, 0))
    
    return samples

def sample_random_points(list_samples, data_dict, n, sensor_columns):
    """
    Sample N random points from the 5000 CSV files.
    Each sample can come from any file and any row.
    
    Args:
        data_dict: Dictionary of DataFrames with unique tags
        n: Number of random points to sample
        sensor_column: Column name containing sensor values
    
    Returns:
        list: List of tuples (unique_tag, row_index, sensor_value)
    """
    samples = []
    tags = list(data_dict.keys())

    # Ensure sensor_columns is a list
    if isinstance(sensor_columns, str):
        sensor_columns = [sensor_columns]
        
    for _ in range(n):
        # Randomly select a file
        random_tag = random.choice(tags)
        df = data_dict[random_tag]
        
        # Randomly select a row (datapoint)
        random_row_idx = random.randint(0, len(df) - 1)
        
        # Extract all sensor column values
        sensor_values = {}
        for col in sensor_columns:
            if col in df.columns:
                sensor_values[col] = df.loc[random_row_idx, col]
        
        # Only add sample if at least one sensor column exists
        if sensor_values:
            samples.append((random_tag, random_row_idx, sensor_values))
    
    return samples


def scoring_function(new_sensor_values, sampled_points):
    """
    Scoring function that calculates mean square error between
    new sensor values and N sampled points for multiple columns.
    
    Args:
        new_sensor_values: Dict of new sensor values received in this cycle
        sampled_points: List of tuples (unique_tag, row_index, dict_of_sensor_values)
    
    Returns:
        dict: Dictionary containing MSE and other metrics for each sensor column
    """
    all_scores = {}
    
    # Process each sensor column
    for col_name in new_sensor_values.keys():
        # Extract values for this column from sampled points
        sampled_values = []
        for point in sampled_points:
            sensor_dict = point[2]
            if col_name in sensor_dict:
                sampled_values.append(sensor_dict[col_name])
        
        if not sampled_values:
            continue
        
        # Calculate mean square error for this column
        new_value = new_sensor_values[col_name]
        squared_errors = [(new_value - val) ** 2 for val in sampled_values]
        mse = np.mean(squared_errors)
        
        # Calculate additional metrics
        all_scores[col_name] = {
            'mean_square_error': mse,
            'root_mean_square_error': np.sqrt(mse),
            'num_samples': len(sampled_values)
        }
    
    return all_scores


def simulate_new_sensor_values(sensor_columns, new_data_file=None, row_index=0):
    """
    Load sensor values from a CSV file.
    In real application, these would come from actual sensors.
    
    Args:
        sensor_columns: List of sensor column names
        new_data_file: Path to CSV file containing new sensor data
        row_index: Which row to read from the CSV (default: 0)
    
    Returns:
        dict: Dictionary of sensor values from the CSV file
    """
    sensor_values = {}
    
    # Ensure sensor_columns is a list
    if isinstance(sensor_columns, str):
        sensor_columns = [sensor_columns]
    
    if new_data_file is None:
        print("ERROR: new_data_file path must be specified")
        return sensor_values
    
    try:
        # Load the CSV file
        df = pd.read_csv(new_data_file)
        
        # Check if row_index is valid
        if row_index >= len(df):
            print(f"ERROR: row_index {row_index} exceeds file length {len(df)}")
            return sensor_values
        
        # Extract specified sensor columns from the row
        for col in sensor_columns:
            if col in df.columns:
                sensor_values[col] = df.loc[row_index, col]
            else:
                print(f"WARNING: Column '{col}' not found in {new_data_file}")
        
    except Exception as e:
        print(f"ERROR loading new sensor data from {new_data_file}: {e}")
    
    return sensor_values


def main_loop(data_dict, n, sensor_columns, num_cycles=None):
    """
    Main loop that processes new sensor values.
    
    Args:
        data_dict: Dictionary of loaded CSV files
        n: Number of random points to sample each cycle
        sensor_columns: List of column names containing sensor values
        num_cycles: Number of cycles to run (None for infinite)
    """
    cycle = 0
    
    # Ensure sensor_columns is a list
    if isinstance(sensor_columns, str):
        sensor_columns = [sensor_columns]
    
    print(f"{'='*70}")
    print(f"Starting main loop...")
    print(f"  Total CSV files loaded: {len(data_dict)}")
    print(f"  Random samples per cycle: {n}")
    print(f"  Sensor columns: {sensor_columns}")
    print(f"{'='*70}\n")
    
    sampled_points = initialize_random_points(data_dict, n, sensor_columns)
    
    try:
        while True:
            cycle += 1
            
            # Check if we've reached the desired number of cycles
            if num_cycles is not None and cycle > num_cycles:
                print(f"\nCompleted {num_cycles} cycles. Exiting loop.")
                break
            
            print(f"--- Cycle {cycle} ---")
            
            # Step 1: Receive new sensor values
            new_sensor_values = simulate_new_sensor_values(
                        sensor_columns, 
                        new_data_file=NEW_SENSOR_DATA_FILE, 
                        row_index=SENSOR_ROW_INDEX
            )
            
            print(f"New sensor values received:")
            for col, val in new_sensor_values.items():
                print(f"  {col}: {val:.4f}")
            
            # Step 2: Sample N random points from the 5000 CSV files
            sampled_points = sample_random_points(sampled_points, data_dict, n, sensor_columns)
            print(f"Sampled {len(sampled_points)} random points from CSV files")
            
            # Optional: Show some sample details
            if len(sampled_points) > 0:
                print(f"  Example samples:")
                for i in range(min(3, len(sampled_points))):
                    tag, row_idx, values = sampled_points[i]
                    print(f"    - File: {tag}, Row: {row_idx}")
                    for col, val in values.items():
                        print(f"      {col}: {val:.4f}")
            
            # Step 3: Calculate scores using the scoring function
            scores = scoring_function(new_sensor_values, sampled_points)
            
            # Display scoring weights
            print(f"Scores:")
            for col_name, col_scores in scores.items():
                print(f"  {col_name}:")
                for key, value in col_scores.items():
                    print(f"    {key}: {value:.4f}")
            
            # Step 4: Normalize weights
            
            print()
            
    except KeyboardInterrupt:
        print(f"\n\nLoop interrupted by user after {cycle} cycles.")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Load all 5000 CSV files (done once, outside the loop)
    print("Loading CSV files...")
    data_dict = load_csv_files(INPUT_FOLDER, max_files=MAX_FILES)
    
    if not data_dict:
        print("No data loaded. Exiting.")
        exit()
    
    # Verify data structure
    print(f"\nData loaded successfully!")
    print(f"Total files: {len(data_dict)}")
    
    # Show example of first file
    first_tag = list(data_dict.keys())[0]
    first_df = data_dict[first_tag]
    print(f"\nExample file: '{first_tag}'")
    print(f"  Shape: {first_df.shape}")
    print(f"  Columns: {list(first_df.columns)}")
    print()
    
    # Step 2: Enter the main loop
    main_loop(data_dict, N, SENSOR_COLUMNS, num_cycles=NUM_CYCLES)
    
    print("\nProgram completed.")