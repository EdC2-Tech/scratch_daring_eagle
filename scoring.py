# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:21:23 2025

@author: ChEdw
"""

def calculate_mse_between_dicts(dict1, dict2):
    """
    Calculate mean square error between matching columns in two dictionaries.
    
    Args:
        dict1 (dict): First dictionary with column names as keys and values
        dict2 (dict): Second dictionary with column names as keys and values
    
    Returns:
        dict: Dictionary with column names as keys and MSE as values
    """
    squared_errors = []
    
    # Find common keys between both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    for key in common_keys:
        try:
            # Calculate squared error for this column
            squared_error = (dict1[key] - dict2[key]) ** 2
            squared_errors.append(squared_error)
        except (TypeError, ValueError) as e:
            print(f"WARNING: Cannot calculate MSE for column '{key}': {e}")
    
    # Return average MSE
    if squared_errors:
        return sum(squared_errors) / len(squared_errors)
    else:
        return None

def calculate_normAbs_score(pred_value, true_value):
    """
    Calculate normalized error between matching columns in two dictionaries.
    
    Args:
        dict1 (dict): First dictionary with column names as keys and values
        dict2 (dict): Second dictionary with column names as keys and values
    
    Returns:
        dict: Dictionary with column names as keys and MSE as values
    """
    abs_norm_errors = []
    
    # Find common keys between both dictionaries
    common_keys = set(pred_value.keys()) & set(true_value.keys())
    
    for key in common_keys:
        try:
            # Calculate squared error for this column
            abs_norm_error = abs((pred_value[key] - true_value[key])) / true_value[key]
            abs_norm_errors.append(abs_norm_error)
        except (TypeError, ValueError) as e:
            print(f"WARNING: Cannot calculate normalized absolute error for column '{key}': {e}")
    
    # Return average MSE
    if abs_norm_errors:
        return sum(abs_norm_errors) / len(abs_norm_errors)
    else:
        return None
    