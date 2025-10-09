"""
Parkes Error Grid Analysis (EGA) for Python

A Python implementation of the Parkes Error Grid Analysis for performance 
assessment of blood glucose concentration measurement methods.

Original MATLAB implementation by Rupert Thomas, 2016
https://github.com/4OH4/Parkes_EGA_MATLAB
Python conversion maintaining scientific accuracy and functionality.

Based on:
'Technical Aspects of the Parkes Error Grid' - Andreas Pfützner et al., 2013
"""

from .parkes_ega import ParkesEGA, identify_regions, plot_parkes_grid
from .boundaries import get_boundaries_type1

__version__ = "1.0.0"
__author__ = "Converted from MATLAB by Rupert Thomas, 2016"

__all__ = [
    "ParkesEGA",
    "identify_regions", 
    "plot_parkes_grid",
    "get_boundaries_type1"
]
