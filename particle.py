# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:39 2025

@author: ChEdw
"""
import os
import pandas as pd

import scoring

class Particle:
    """
    A class to store information about a data point. Acts as a particle in the filter.
    
    Attributes:
        tag (str): The name of the file
        index (int): The index value
        current (dict): The determined value by this particle
        score (int): The calculated score value
    """
    
    def __init__(self, tag, index, path, columns):
        """
        Initialize a Particle object.
        
        Args:
            tag (str): The name of the file.
            index (int): The index value (integer).
            score (int): The score value (integer).
            path (str): The path where the dataset is located to distribute probes.
            columns (list): The names of the columns to track.
        """
        self.tag     = tag
        self.index   = int(index)
        self.score   = 0
        self.path    = path
        self.columns = columns
        self.current = self.load_data()
        self.max_idx = 2500
    
    def update(self, true_value):
        """
        Updates the particle's information and moves the particle forward by one.
        
        Args:
            true_value (): The sensor value to compare the particle value to
        """          
        self.calculate_score(true_value)
        
    def calculate_score(self, true_value):
        """
        Initialize a DataPoint object.
        
        Args:
            true_value (): The name of the file
        """        
        # Get values for current index and tag
        self.current = self.load_data()
        
        # Calculate score; score must be single numerical value
        self.score = scoring.calculate_normAbs_score(self.current, true_value)
        
        return self.score
        
    def load_data(self):
        """
        Lightweight function to load a single row from a CSV file.
        
        Args:
            folder_path (str): Path to folder containing the CSV file (default: current directory)
            columns (list or str): Column name(s) to retrieve. If None, retrieves all columns.
        
        Returns:
            dict: Dictionary with column names as keys and values from the specified row.
                  Returns None if loading fails.
        """
        # Construct full file path
        file_path = os.path.join(self.path, self.tag)
        
        columns = self.columns
        
        try:
            # Read only the specific row (more memory efficient than loading entire file)
            df = pd.read_csv(file_path, skiprows=range(1, self.index + 1), nrows=1)
            
            # If skiprows skipped the header, we need to reload with proper header
            if self.index > 0:
                # Read header separately
                header_df = pd.read_csv(file_path, nrows=0)
                df.columns = header_df.columns
            
            # Handle column selection
            if columns is not None:
                if isinstance(self.columns, str):
                    columns = [columns]
                
                # Filter to requested columns
                available_cols = [col for col in columns if col in df.columns]
                
                if not available_cols:
                    print(f"WARNING: None of the specified columns {columns} found in {self.tag}")
                    return None
                
                df = df[available_cols]
            
            # Convert the row to dictionary
            result = df.iloc[0].to_dict()
            return result
            
        except FileNotFoundError:
            print(f"ERROR: File '{file_path}' not found")
            return None
        except IndexError:
            print(f"ERROR: Index {self.index} out of range for file '{self.tag}'")
            return None
        except Exception as e:
            print(f"ERROR loading data from '{self.tag}': {e}")
            return None    
    
    def forward(self):
        """Moves the particle foward in the same dataset"""
        if self.index < self.max_idx:
            self.index = self.index+1
        
    def __str__(self):
        """String representation of the Particle."""
        return f"DataPoint(tag='{self.tag}', index={self.index}, score={self.score})"
    
    def __repr__(self):
        """Official string representation of the Particle."""
        return f"DataPoint('{self.tag}', {self.index}, {self.score})"
    
    def get_tag(self):
        """Get the tag value."""
        return self.tag

    def get_index(self):
        """Get the index value."""
        return self.index
    
    def get_worth(self):
        """Get the worth value."""
        return self.worth
    
    def set_tag(self, tag):
        """Set the tag value."""
        self.tag = tag
    
    def set_index(self, index):
        """Set the index value."""
        self.index = int(index)
    
    def set_worth(self, worth):
        """Set the worth value."""
        self.worth = int(worth)
    
    def to_dict(self):
        """Convert DataPoint to dictionary."""
        return {
            'tag': self.tag,
            'index': self.index,
            'worth': self.worth
        }

# Example usage
if __name__ == "__main__":
    # Generic path for all particles
    path   = "./016_Q8_175_5000_short/"
    
    # Columns to track
    cols   = ["PS1", "PS2"]
    
    # Create a DataPoint object
    point1 = Particle("histories_1.csv", 245, path, cols)
    
    test_dict = {
                 'PS1': 500,
                 'PS2': 500
                }
    print(point1.calculate_score(test_dict))
    
    print(point1)
    
