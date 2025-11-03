# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:50:00 2025

@author: ChEdw
"""
import os
import glob
import random

from particle import Particle

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Number of DataPoints to create
X = 10

# Folder containing CSV files
CSV_FOLDER = "./016_Q8_175_5000_short/"

# Columns to track
COLS = ["FL1", "FL6"]
    
# Range for random index values
INDEX_MIN = 0
INDEX_MAX = 1999  # Assuming 2000 datapoints per file

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def get_csv_files_from_folder(folder_path):
      """
      Read all CSV filenames from a given folder.
        
      Args:
          folder_path (str): Path to folder containing CSV files
        
      Returns:
          list: List of CSV filenames (basenames only, not full paths)
      """
      if not os.path.exists(folder_path):
          print(f"ERROR: Folder '{folder_path}' does not exist")
          return []
        
      # Get all CSV files in the folder
      csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
      if not csv_files:
          print(f"WARNING: No CSV files found in '{folder_path}'")
          return []
        
      # Extract just the filenames (not full paths)
      filenames = [os.path.basename(file) for file in csv_files]
        
      print(f"Found {len(filenames)} CSV files in '{folder_path}'")
        
      return filenames
    
class Swarm:
    def __init__(self, num_points, tags_list, index_range, path, cols):
        """
        Create a list of Particle objects with random tags and indices.
        
        Args:
            num_points (int): Number of DataPoints to create
            tags_list (list): List of possible file tags to randomly select from
            index_range (tuple): (min, max) range for random index values
        
        Returns:
            list: List of DataPoint objects
        """
        self.particles = []
        self.threshold = 0
        
        self.tags_list   = tags_list
        self.index_range = index_range 
        self.global_path = path
        self.global_cols = cols
        
        for i in range(num_points):
            # Randomly select a tag from the list (potentially add latin hypercube sampling here)
            random_tag = random.choice(tags_list)
            
            # Generate random index
            random_index = random.randint(index_range[0], index_range[1])
            
            # Create DataPoint and add to list
            pt = Particle(random_tag, random_index, path, cols)
            self.particles.append(pt)
        
        return
    
    def repopulate(self):
        if self.threshold == 0 or self.threshold == 1:
            print("Initialize the threshold for rejection. Use set_threshold(x) to set threshold. x must be a value between 0 and 1 non-inclusive")

        for pt_idx in range(len(self.particles)):
            
            pt = self.particles[pt_idx]

            # Discard and regenerate if less than threshold
            if pt.score >= self.threshold:
                new_tag = random.choice(self.tags_list)
                new_idx = random.randint(self.index_range[0], self.index_range[1])
                self.particles[pt_idx] = Particle(new_tag, new_idx, self.global_path, self.global_cols)
                
            # Update if particle is valid
            if pt.score < self.threshold:
                self.particles[pt_idx].forward()
                
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def score_swarm(self, true_value):
        """
        Score every single Particle object.
        
        Args:
            true_value (dict): dictionary containing the true value at a timestep
        
        Returns:
            None
        """ 
        for pt in self.particles:
            pt.calculate_score(true_value)
    
    def print_score_all(self):
        
        for pt in self.particles:
            print(pt.score)
    
    def get_current(self):
        current_particles = list()
        
        for pt in self.particles:
            current_particles.append(pt.current)
            
        return current_particles

if __name__ == "__main__":
    
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
    
    test_dict = {
                 'FL1': 242,
                 'FL6': 242
                }
    
    for i in range(20):
        pt_swarm.score_swarm(test_dict)
        pt_swarm.repopulate()
        
        print("Iteration: " + str(i))
    
    # Show score of particles
    pt_swarm.print_score_all()
    