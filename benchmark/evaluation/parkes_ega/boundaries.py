"""
Boundary definitions for Parkes Error Grid Analysis (Type 1 diabetes).

Converted from MATLAB implementation by Rupert Thomas, 2016.

Source:
'Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013
"""

import numpy as np
from typing import Dict, Tuple


def get_boundaries_type1(units_mg_dl: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Get boundary vertices for Parkes Error Grid Analysis (Type 1 diabetes).
    
    Parameters
    ----------
    units_mg_dl : bool, optional
        If True (default), coordinates are in mg/dl.
        If False, coordinates are converted to mM.
        
    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing region boundaries as (x, y) coordinate tuples.
        Keys are 'A', 'B', 'C', 'D', 'E' for each region.
        
    Notes
    -----
    This data is provided 'as-is', with no warranty. Don't use this for
    anything important/clinical without thoroughly double-checking the code
    and making sure it does what you expect it to do!
    """
    
    # Region A - Clinically accurate
    region_a_x = np.array([0, 50, 50, 170, 385, 550, 550, 430, 280, 140, 30, 0])
    region_a_y = np.array([0, 0, 30, 145, 300, 450, 550, 550, 380, 170, 50, 50])
    
    # Region B - Clinically acceptable
    region_b_x = np.array([0, 120, 120, 260, 550, 550, 260, 70, 50, 30, 0])
    region_b_y = np.array([0, 0, 30, 130, 250, 550, 550, 110, 80, 60, 60])
    
    # Region C - Clinically acceptable
    region_c_x = np.array([0, 250, 250, 550, 550, 125, 80, 50, 25, 0])
    region_c_y = np.array([0, 0, 40, 150, 550, 550, 215, 125, 100, 100])
    
    # Region D - Potentially dangerous
    region_d_x = np.array([0, 550, 550, 50, 35, 0])
    region_d_y = np.array([0, 0, 550, 550, 155, 150])
    
    # Region E - Erroneous (everything else in range)
    region_e_x = np.array([0, 0, 550, 550])
    region_e_y = np.array([0, 550, 550, 0])
    
    boundaries = {
        'A': (region_a_x, region_a_y),
        'B': (region_b_x, region_b_y),
        'C': (region_c_x, region_c_y),
        'D': (region_d_x, region_d_y),
        'E': (region_e_x, region_e_y)
    }
    
    # Convert to mM if requested (conversion factor: mg/dl * 0.05556 = mM)
    if not units_mg_dl:
        conversion_factor = 0.05556
        boundaries = {
            region: (x_coords * conversion_factor, y_coords * conversion_factor)
            for region, (x_coords, y_coords) in boundaries.items()
        }
    
    return boundaries


def get_region_colors() -> Dict[str, str]:
    """
    Get standard colors for each Parkes EGA region.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping region letters to matplotlib color codes.
    """
    return {
        'A': 'green',      # Clinically accurate
        'B': 'yellow',     # Clinically acceptable  
        'C': 'orange',     # Clinically acceptable
        'D': 'lightcoral', # Potentially dangerous
        'E': 'red'  # Erroneous
    }


def get_region_descriptions() -> Dict[str, str]:
    """
    Get clinical descriptions for each Parkes EGA region.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping region letters to clinical descriptions.
    """
    return {
        'A': 'Clinically accurate',
        'B': 'Clinically acceptable',
        'C': 'Clinically acceptable', 
        'D': 'Potentially dangerous',
        'E': 'Erroneous'
    }
