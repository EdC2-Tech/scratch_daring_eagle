# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:50:00 2025

@author: ChEdw
"""
# Default libraries
import os
import glob
import random
import math

# Install libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Explicit libraries
from particle import Particle

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Number of particles to create
NUM_PARTICLES = 20

# Folder containing CSV files
CSV_FOLDER = "./test_data/"

# Columns to track
COLS = ["FL1", "FL6"]
    
# Range for random index values
INDEX_MIN = 0
INDEX_MAX = 1999  # Assuming 2000 datapoints per file

# ============================================================================
# MAIN PROGRAM
# ============================================================================
  
class Swarm:
    def __init__(self, num_particles, folder_path, selected_cols, index_range=None, replacement_rate=0.1):
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
        self.population_cut = 0.5
        self.tags_list = list()
        self.replacement_rate = replacement_rate
        
        self.__init_source_data__(folder_path, selected_cols)
        self.__index_range_finder__(index_range, find_range=True)
        
        self.weights = np.ones(num_particles)/num_particles
        
        for i in range(num_particles):
            # Randomly select a tag from the list (potentially add latin hypercube sampling here)
            random_tag = random.choice(self.tags_list)
            
            # Generate random index
            min_range_idx = self.index_range[0]
            max_range_idx = self.index_range[1]
            random_index = random.randint(min_range_idx, max_range_idx)
            
            # Create particles and add to list
            pt = Particle(random_tag, random_index, initial_score=1)
            self.particles.append(pt)
        
        return

#%% Performance Functions
    def predict_all(self):
        """
        Get data from each particle using the source data.
        """
        for pt in self.particles:
            self.predict(pt) 
            
    def predict(self, pt):
        """
        Get data from a single particle using the source data.
        """
        pt.current = self.__load_data__(pt.get_tag(), pt.get_index())
        
    def forward_all(self):
        """
        Move each particle forward.
        """
        for pt in self.particles:
            pt.forward(self.index_range[1])
        
    def calculate_score_all(self, true_value):
        """
        Score every single Particle object.
        """ 
        for i, pt in enumerate(self.particles):
            pt.calculate_score(true_value)        
        
    def calculate_weights_all(self):
        """
        Calculate the weight of each Particle after scoring.
        """
        for i, pt in enumerate(self.particles):
            self.weights[i] = np.exp(-0.5 * pt.score**2)
        
        self.weights += 1.e-300  # Avoid divide by zero
        self.weights /= np.sum(self.weights)
    
    def sort_by_weight(self):
        """
        Sort the particles by weight.
        """
        return np.argsort(-self.weights)
    
    def sort_by_score(self):
        """
        Sort the particles by score.
        """
        return sorted(self.particles, key=lambda x: x.score)
    
    def repopulate(self, method="both"):
        if self.threshold == 0:
            print("Threshold for rejection cannot be zero (0). Use set_threshold(x) to set threshold. x must be a value between 0 and 1 non-inclusive")        
        
        if method.lower() == "both":
            self.resample_by_weight()
            self.resample_by_threshold()
        
        elif method.lower() == "weight":
            self.resample_by_weight()
            
        elif method.lower() == "threshold":
            self.resample_by_threshold()

    def resample_by_weight(self):
        # Reorder particles by score, higher is better
        sorted_particles = self.sort_by_weight()
                                  
        # Calculate the weight of all particles
        self.calculate_weights_all()
           
        # Only repopulate the lowest X number of particles
        cut_off = int(len(self.particles)*self.population_cut)
        
        # Indices of best and worst particles
        best_indices  = sorted_particles[:cut_off]
        worst_indices = sorted_particles[cut_off:]
        
        # Resample indices based on weight probability of the best indices
        resample_indices = np.random.choice(best_indices, size=len(worst_indices), p=self.weights[best_indices] / np.sum(self.weights[best_indices]))
        
        # Replace bottom particles with resampled copies of top particles
        for i, idx in zip(worst_indices, resample_indices):
            # Deep copy to avoid shared references
            copied_particle = Particle(self.particles[idx].tag, self.particles[idx].index, initial_score=1)
            
            # Update particle state information
            self.predict(copied_particle)
            
            # Replace particle with new particle
            self.particles[i] = copied_particle
    
    def resample_by_threshold(self):
        # Replace if threshold is not met either
        for pt_idx in range(len(self.particles)):
            pt = self.particles[pt_idx]
            
            # Discard and regenerate if error greater than threshold
            if pt.score >= self.threshold:
                # Randomly get a new particle location
                new_tag = random.choice(self.tags_list)
                new_idx = random.randint(self.index_range[0], self.index_range[1])
                
                # Create new particle
                new_particle = Particle(new_tag, new_idx, initial_score=1)
                
                # Update particle state information
                self.predict(new_particle)
                
                # Replace particle with new particle
                self.particles[pt_idx] = new_particle 
    
#%% Get and Set Methods            
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def set_population_cut(self, population_cut):
        self.population_cut = population_cut
    
    def get_best_score(self):
        """
        Sort the particles by score then retrieve the best particle score. 

        Returns
        -------
        float
            Score of the best particle in the swarm.

        """
        # Reorder particles by score, higher is better
        sorted_particles = self.sort_by_score()
        return sorted_particles[0].score
    
    def get_mean_pred(self, col, cutoff=-1):
        """
        Calculate the mean particle predictions for one or more columns.
    
        Parameters
        ----------
        col : str or list of str
            Column(s) to compute the mean prediction for.
        cutoff : float, optional
            If not -1, only the top fraction of particles (based on score) are used.
    
        Returns
        -------
        dict
            Dictionary of mean values for each column.
        """
        # Convert to list for uniform processing
        if isinstance(col, str):
            col = [col]
            
        current_pt_mean = {key: 0.0 for key in col}        
        sorted_particles = self.sort_by_score()
        
        # Determine if only the best particles are used or all particles based on specified cutoff limit.
        if cutoff != -1:
            CF = int(len(sorted_particles)*self.population_cut)
            for pt_idx in range(CF):
                for key in col:
                    current_pt_mean[key] += sorted_particles[pt_idx].current[key]
            
            for key in col:
                current_pt_mean[key] /= CF
        
        else:
            for key in col:
                values = [p.current[key] for p in self.particles]
                current_pt_mean[key] = np.mean(values)
            
        return current_pt_mean
    
    def get_std_pred(self, col, cutoff=-1):
        """
        Calculate the standard deviation of particle predictions for one or more columns.
    
        Parameters
        ----------
        col : str or list of str
            Column(s) to compute the standard deviation for.
        cutoff : float, optional
            If not -1, only the top fraction of particles (based on score) are used.
    
        Returns
        -------
        dict
            Dictionary of standard deviations for each column.
        """

        # Convert to list for uniform processing
        if isinstance(col, str):
            col = [col]
            
        current_pt_std = {}
        sorted_particles = self.sort_by_score()
        
        # Determine if only the best particles are used or all particles based on specified cutoff limit.
        if cutoff != -1:
            CF = int(len(sorted_particles)*cutoff)
            selected_particles = sorted_particles[:CF]
        
        else:
            selected_particles = self.particles
        
        for key in col:
            values = [pt.current[key] for pt in selected_particles]
            current_pt_std[key] = np.std(values, ddof=1)

        #     for pt_idx in range(CF):
        #         total_score += sorted_particles[pt_idx].current[col]
                
        #     mean = total_score / (CF)
        #     variance = sum((pt.current[col] - mean) ** 2 for pt in sorted_particles)
        #     current_pt_std = math.sqrt(variance / (len(sorted_particles)-1))
        
        # else:
        #     for pt in self.particles:
        #         total_score += pt.current[col]
                
        #     mean = total_score / len(self.particles)
        #     variance = sum((pt.current[col] - mean) ** 2 for pt in self.particles)
        #     current_pt_std = variance / (len(self.particles)-1)
            
        return current_pt_std        
        
    def get_current(self, cutoff=True):
        """
        Get the current state of all or some of the particles in the swarm. 

        Returns
        -------
        list
            Current state of all or some particles in the swarm.

        """
        current_particles = list()
        
        if cutoff:
            cut_off = int(len(self.particles)*self.population_cut)
            for pt_idx in range(cut_off):
                current_particles.append(self.particles[pt_idx].current)
        
        else:
            for pt in self.particles:
                current_particles.append(pt.current)
            
        return current_particles

#%% Hidden Functions    
    def __load_data__(self, tag, index):
        """
        Loads the value at the given tag and index of the particle
        """
        df  = self.csv_dict[tag]
        ret = df.iloc[index]
        
        return ret.to_dict()

    def __init_source_data__(self, folder_path, selected_cols):
        """
        Loads all CSV files in the specified folder into a dictionary of DataFrames.
        
        Parameters:
            folder_path (str): Path to the folder containing CSV files.
            
        Returns:
            no return
        """
        self.global_path = folder_path
        self.global_cols = selected_cols
        
        csv_dict = {}
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                try:
                    df             = pd.read_csv(file_path, usecols=selected_cols)
                    key            = os.path.splitext(filename)[0]
                    self.tags_list.append(key)
                    csv_dict[key]  = df
                
                except ValueError as e:
                                print(f"Skipping {filename}: {e}")

        self.csv_dict = csv_dict
        
        return
    
    def __index_range_finder__(self, index_range, find_range=True):
        """
        Determine the start and end of each data file.
        """
        if find_range:
            # Get the minimum and maximum row index for the csv files
            first_entry = next(iter(self.csv_dict.values()))
            min_idx = first_entry.index.min()
            max_idx = first_entry.index.max()
            self.index_range = (min_idx, max_idx)
        
        else:
            self.index_range = index_range


#%% MAIN FUNCTION            
if __name__ == "__main__":
       
    print(f"\nCreating {NUM_PARTICLES} random Particles...\n")
    
    # Create list of Particles
    pt_swarm = Swarm(
        num_particles=NUM_PARTICLES,
        index_range=(INDEX_MIN, INDEX_MAX),
        folder_path=CSV_FOLDER,
        selected_cols=COLS
    )
    
    pt_swarm.set_threshold(0.99)
    
    test_dict = {
                 'FL1': 242,
                 'FL6': 242
                }
    
    for i in range(20):
        pt_swarm.predict_all()
        pt_swarm.calculate_score_all(test_dict)
        pt_swarm.repopulate()
        pt_swarm.get_mean_pred("FL1")
        pt_swarm.get_std_pred("FL1")
        
        print("Iteration: " + str(i) + " Best: " + str(pt_swarm.get_best_score()))
    


  