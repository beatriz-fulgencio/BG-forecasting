"""
Clarke Error Grid Analysis for Python

A Python implementation of the Clarke Error Grid Analysis for assessing the clinical
significance of differences between glucose measurement techniques.

Original MATLAB implementation by Edgar Guevara Codina
Python conversion maintaining scientific accuracy and functionality.
"""

from .clarke_ega import ClarkeEGA, clarke_analysis, plot_clarke_grid, get_clarke_zone_colors

__version__ = "1.0.0"
__author__ = "Converted from MATLAB by Edgar Guevara Codina"

__all__ = [
    "ClarkeEGA",
    "clarke_analysis", 
    "plot_clarke_grid",
    "get_clarke_zone_colors"
]
