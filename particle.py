# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:16:39 2025

@author: ChEdw
"""
import os
import pandas as pd
import numpy as np

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
    
    def __init__(self, tag, index, initial_score):
        """
        Initialize a Particle object.
        
        Args:
            tag (str): The name of the file.
            index (int): The index value (integer).
            score (int): The score value (integer).
        """
        self.tag     = tag
        self.index   = int(index)
        self.score   = initial_score
        self.current = None
    
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
        # Calculate score; score must be single numerical value
        self.score = scoring.calculate_normAbs_score(self.current, true_value)
        
        return self.score
    
    def forward(self, max_idx):
        """Moves the particle foward in the same dataset"""
        if self.index < max_idx:
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
    
