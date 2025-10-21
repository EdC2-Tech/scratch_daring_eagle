# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 19:21:25 2025

@author: ChEdw
"""
import pandas as pd
import os
import glob
from pathlib import Path

def filter_and_export_csvs(input_folder: str, output_folder: str, columns_to_keep: list, file_pattern: str = "*.csv", max_files=1):
    """
    Import CSV files, keep only specified columns, and export with same names.
    
    Args:
        input_folder: Path to folder containing input CSV files
        output_folder: Path to folder for output CSV files
        columns_to_keep: List of column names to keep in each file
        file_pattern: Pattern to match CSV files (default: "*.csv")
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all CSV files from input folder
    csv_files = glob.glob(os.path.join(input_folder, file_pattern))[:max_files]
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Columns to keep: {columns_to_keep}\n")
    
    successful = 0
    failed = 0
    
    for file_path in csv_files:
        try:
            # Get the filename
            filename = os.path.basename(file_path)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check which columns exist in the file
            available_columns = [col for col in columns_to_keep if col in df.columns]
            missing_columns = [col for col in columns_to_keep if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  {filename}: Missing columns {missing_columns}")
            
            if not available_columns:
                print(f"❌ {filename}: None of the specified columns found. Skipping.")
                failed += 1
                continue
            
            # Filter to keep only specified columns
            df_filtered = df[available_columns]
            
            if 'FL1' in df.columns and 'FL6' in df.columns:
                df_filtered['TFL'] = df['FL1'] + df['FL6']
            else:
                print(f"WARNING: Cannot create TFL column. Missing columns.")
                if 'FL1' not in df.columns:
                    print(f"  - 'FL1' not found")
                if 'FL6' not in df.columns:
                    print(f"  - 'FL6' not found")            
                    
            # Create output path with same filename
            output_path = os.path.join(output_folder, filename)
            
            # Export to CSV
            df_filtered.to_csv(output_path, index=False)
            
            print(f"✓ {filename}: Processed ({len(df)} rows, {len(available_columns)} columns)")
            successful += 1
            
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Output saved to: {output_folder}")


# Example usage
if __name__ == "__main__":
    
    # CONFIGURATION - Modify these variables
    INPUT_FOLDER = "016_Q8_175_5000_t"  # Folder containing your CSV files
    OUTPUT_FOLDER = "016_Q8_175_5000_short"  # Folder where filtered CSVs will be saved
    
    # Specify the column names you want to keep
    COLUMNS_TO_KEEP = [
        "time",
        "FL1",
        "FL6",
        "TL14s1",
        "TA21s1",
        "PS1",
        "PS2"
    ]
    
    # Optional: Use a specific file pattern (e.g., "sales_*.csv")
    FILE_PATTERN = "*.csv"
    
    # Run the function
    filter_and_export_csvs(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        columns_to_keep=COLUMNS_TO_KEEP,
        file_pattern=FILE_PATTERN,
        max_files=1
    )