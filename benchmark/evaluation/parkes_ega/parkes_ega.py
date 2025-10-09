"""
Main Parkes Error Grid Analysis implementation.

Converted from MATLAB implementation by Rupert Thomas, 2016.
Optimized for use with machine learning predictions (y_pred vs y_true).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from typing import Tuple, Dict, List, Optional, Union
import warnings

from .boundaries import get_boundaries_type1, get_region_colors, get_region_descriptions


class ParkesEGA:
    """
    Parkes Error Grid Analysis for blood glucose measurement assessment.
    
    This class provides methods to analyze the clinical accuracy of glucose
    measurements using the Parkes Error Grid, designed for Type 1 diabetes.
    
    Attributes
    ----------
    units_mg_dl : bool
        Whether measurements are in mg/dl (True) or mM (False)
    boundaries : Dict
        Region boundary coordinates
    """
    
    def __init__(self, units_mg_dl: bool = True):
        """
        Initialize Parkes EGA analyzer.
        
        Parameters
        ----------
        units_mg_dl : bool, optional
            If True (default), assume measurements are in mg/dl.
            If False, assume measurements are in mM.
        """
        self.units_mg_dl = units_mg_dl
        self.boundaries = get_boundaries_type1(units_mg_dl)
        self._max_range = 550 if units_mg_dl else 550 * 0.05556
        
    def identify_regions(self, y_true: Union[float, np.ndarray], 
                        y_pred: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Identify which Parkes EGA region each measurement pair belongs to.
        
        Parameters
        ----------
        y_true : float or array-like
            Reference method measurements (true values)
        y_pred : float or array-like  
            Test method measurements (predicted values)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys 'A', 'B', 'C', 'D', 'E', 'OOR' containing
            boolean arrays indicating which points belong to each region.
            'OOR' indicates out-of-range points.
            
        Notes
        -----
        This implementation maintains the original MATLAB logic where regions
        are evaluated in reverse order (E->D->C->B->A) to handle overlapping
        boundaries correctly.
        """
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
            
        n_points = len(y_true)
        
        # Initialize result arrays
        regions = {region: np.zeros(n_points, dtype=bool) for region in ['A', 'B', 'C', 'D', 'E']}
        
        # Check if points are inside each polygon
        for region_name, (boundary_x, boundary_y) in self.boundaries.items():
            # Create path for polygon
            vertices = np.column_stack((boundary_x, boundary_y))
            path = Path(vertices)
            points = np.column_stack((y_true, y_pred))
            regions[region_name] = path.contains_points(points)
        
        # Remove overlap by processing regions in reverse order (E->D->C->B->A)
        # This matches the original MATLAB implementation
        regions['E'] = regions['E'] & ~regions['D']
        regions['D'] = regions['D'] & ~regions['C'] 
        regions['C'] = regions['C'] & ~regions['B']
        regions['B'] = regions['B'] & ~regions['A']
        
        # Points that don't belong to any region are out of range
        in_any_region = np.logical_or.reduce([regions[r] for r in ['A', 'B', 'C', 'D', 'E']])
        regions['OOR'] = ~in_any_region
        
        return regions
    
    def analyze(self, y_true: Union[float, np.ndarray], 
                y_pred: Union[float, np.ndarray]) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Perform complete Parkes EGA analysis.
        
        Parameters
        ----------
        y_true : float or array-like
            Reference method measurements (true values)
        y_pred : float or array-like
            Test method measurements (predicted values)
            
        Returns
        -------
        Dict[str, Union[int, float, np.ndarray]]
            Analysis results containing:
            - 'regions': Region assignments for each point
            - 'counts': Number of points in each region
            - 'percentages': Percentage of points in each region
            - 'total_points': Total number of points analyzed
        """
        regions = self.identify_regions(y_true, y_pred)
        total_points = len(np.atleast_1d(y_true))
        
        counts = {region: np.sum(mask) for region, mask in regions.items()}
        percentages = {region: (count / total_points) * 100 
                      for region, count in counts.items()}
        
        return {
            'regions': regions,
            'counts': counts, 
            'percentages': percentages,
            'total_points': total_points
        }
    
    def plot(self, y_true: Union[float, np.ndarray] = None, 
             y_pred: Union[float, np.ndarray] = None,
             figsize: Tuple[int, int] = (10, 10),
             show_points: bool = True,
             show_labels: bool = True,
             point_size: float = 30,
             alpha: float = 0.5,
             title: Optional[str] = None) -> plt.Figure:
        """
        Plot the Parkes Error Grid with optional data points.
        
        Parameters
        ----------
        y_true : float or array-like, optional
            Reference method measurements to plot
        y_pred : float or array-like, optional
            Test method measurements to plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        show_points : bool, optional
            Whether to show data points (default True)
        show_labels : bool, optional
            Whether to show region labels (default True)
        point_size : float, optional
            Size of data points (default 50)
        alpha : float, optional
            Transparency of boundary fills (default 0.7)
        title : str, optional
            Custom plot title
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot boundary regions with colors
        colors = get_region_colors()
        descriptions = get_region_descriptions()
        
        # Plot regions in reverse order (largest to smallest) to handle overlaps
        region_order = ['E', 'D', 'C', 'B', 'A']
        
        for region_name in region_order:
            if region_name not in self.boundaries:
                continue
                
            boundary_x, boundary_y = self.boundaries[region_name]
            
            # Skip filling Region E (it's just the outer boundary)
            if region_name != 'E':
                ax.fill(boundary_x, boundary_y, 
                       color=colors[region_name], 
                       alpha=alpha,
                       label=f"Region {region_name}: {descriptions[region_name]}")
            
            # Draw all boundary lines
            ax.plot(boundary_x, boundary_y, 'k-', linewidth=1.5)
        
        # Plot data points if provided
        if show_points and y_true is not None and y_pred is not None:
            regions = self.identify_regions(y_true, y_pred)
            y_true = np.atleast_1d(y_true)
            y_pred = np.atleast_1d(y_pred)
            
            # Plot points colored by region
            for region_name, mask in regions.items():
                if region_name == 'OOR':
                    continue
                if np.any(mask):
                    ax.scatter(y_true[mask], y_pred[mask], 
                             c=colors[region_name], 
                             s=point_size,
                             edgecolors='black',
                             linewidth=0.5,
                             alpha=0.9,
                             zorder=5)
            
            # Plot out-of-range points in gray
            if np.any(regions['OOR']):
                ax.scatter(y_true[regions['OOR']], y_pred[regions['OOR']],
                         c='gray', s=point_size, 
                         edgecolors='black', linewidth=0.5,
                         alpha=0.8, zorder=5,
                         label='Out of range')
        
        # Set up plot
        ax.set_xlim(0, self._max_range)
        ax.set_ylim(0, self._max_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Labels
        units_label = 'mg/dl' if self.units_mg_dl else 'mM'
        ax.set_xlabel(f'Reference method glucose ({units_label})', fontsize=12)
        ax.set_ylabel(f'Test method glucose ({units_label})', fontsize=12)
        
        if title is None:
            title = 'Parkes Error Grid Analysis (Type 1 Diabetes)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add diagonal reference line
        ax.plot([0, self._max_range], [0, self._max_range], 
               'k--', alpha=0.5, linewidth=1, label='Perfect agreement')
        
        # Legend
        if show_labels:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def plot_on_axes(self, ax: plt.Axes,
                     y_true: Union[float, np.ndarray] = None,
                     y_pred: Union[float, np.ndarray] = None,
                     show_points: bool = True,
                     show_labels: bool = True,
                     point_size: float = 30,
                     alpha: float = 0.5,
                     title: Optional[str] = None) -> plt.Axes:
        """
        Plot the Parkes Error Grid with optional data points.
        
        Parameters
        ----------
        y_true : float or array-like, optional
            Reference method measurements to plot
        y_pred : float or array-like, optional
            Test method measurements to plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        show_points : bool, optional
            Whether to show data points (default True)
        show_labels : bool, optional
            Whether to show region labels (default True)
        point_size : float, optional
            Size of data points (default 50)
        alpha : float, optional
            Transparency of boundary fills (default 0.7)
        title : str, optional
            Custom plot title
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
        """
        
        # Plot boundary regions with colors
        colors = get_region_colors()
        descriptions = get_region_descriptions()
        
        # Plot regions in reverse order (largest to smallest) to handle overlaps
        region_order = ['E', 'D', 'C', 'B', 'A']
        
        for region_name in region_order:
            if region_name not in self.boundaries:
                continue
                
            boundary_x, boundary_y = self.boundaries[region_name]
            
            # Skip filling Region E (it's just the outer boundary)
            if region_name != 'E':
                ax.fill(boundary_x, boundary_y, 
                       color=colors[region_name], 
                       alpha=alpha,
                       label=f"Region {region_name}: {descriptions[region_name]}")
            
            # Draw all boundary lines
            ax.plot(boundary_x, boundary_y, 'k-', linewidth=1.5)
        
        # Plot data points if provided
        if show_points and y_true is not None and y_pred is not None:
            regions = self.identify_regions(y_true, y_pred)
            y_true = np.atleast_1d(y_true)
            y_pred = np.atleast_1d(y_pred)
            
            # Plot points colored by region
            for region_name, mask in regions.items():
                if region_name == 'OOR':
                    continue
                if np.any(mask):
                    ax.scatter(y_true[mask], y_pred[mask], 
                             c=colors[region_name], 
                             s=point_size,
                             edgecolors='black',
                             linewidth=0.5,
                             alpha=0.9,
                             zorder=5)
            
            # Plot out-of-range points in gray
            if np.any(regions['OOR']):
                ax.scatter(y_true[regions['OOR']], y_pred[regions['OOR']],
                         c='gray', s=point_size, 
                         edgecolors='black', linewidth=0.5,
                         alpha=0.8, zorder=5,
                         label='Out of range')
        
        # Set up plot
        ax.set_xlim(0, self._max_range)
        ax.set_ylim(0, self._max_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Labels
        units_label = 'mg/dl' if self.units_mg_dl else 'mM'
        ax.set_xlabel(f'Reference method glucose ({units_label})', fontsize=12)
        ax.set_ylabel(f'Test method glucose ({units_label})', fontsize=12)
        
        if title is None:
            title = 'Parkes Error Grid Analysis (Type 1 Diabetes)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add diagonal reference line
        ax.plot([0, self._max_range], [0, self._max_range], 
               'k--', alpha=0.5, linewidth=1, label='Perfect agreement')
        
        # Legend
        if show_labels:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return ax
    
    def print_summary(self, y_true: Union[float, np.ndarray], 
                     y_pred: Union[float, np.ndarray]) -> None:
        """
        Print a summary of the Parkes EGA analysis results.
        
        Parameters
        ----------
        y_true : float or array-like
            Reference method measurements
        y_pred : float or array-like
            Test method measurements
        """
        results = self.analyze(y_true, y_pred)
        descriptions = get_region_descriptions()
        
        print("Parkes Error Grid Analysis Results")
        print("=" * 50)
        print(f"Total points analyzed: {results['total_points']}")
        print()
        
        for region in ['A', 'B', 'C', 'D', 'E', 'OOR']:
            count = results['counts'][region]
            percentage = results['percentages'][region]
            
            if region == 'OOR':
                desc = 'Out of range'
            else:
                desc = descriptions[region]
                
            print(f"Region {region} ({desc}): {count} points ({percentage:.1f}%)")


# Convenience functions for direct use
def identify_regions(y_true: Union[float, np.ndarray], 
                    y_pred: Union[float, np.ndarray],
                    units_mg_dl: bool = True) -> Dict[str, np.ndarray]:
    """
    Identify Parkes EGA regions for measurement pairs.
    
    This is a convenience function that creates a ParkesEGA instance
    and calls its identify_regions method.
    
    Parameters
    ----------
    y_true : float or array-like
        Reference method measurements (true values)
    y_pred : float or array-like
        Test method measurements (predicted values)
    units_mg_dl : bool, optional
        If True (default), assume measurements are in mg/dl.
        If False, assume measurements are in mM.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with boolean arrays for each region ('A', 'B', 'C', 'D', 'E', 'OOR')
    """
    ega = ParkesEGA(units_mg_dl=units_mg_dl)
    return ega.identify_regions(y_true, y_pred)


def plot_parkes_grid(y_true: Union[float, np.ndarray] = None,
                    y_pred: Union[float, np.ndarray] = None,
                    units_mg_dl: bool = True,
                    figsize: Tuple[int, int] = (10, 10),
                    **kwargs) -> plt.Figure:
    """
    Plot Parkes Error Grid with optional data points.
    
    This is a convenience function that creates a ParkesEGA instance
    and calls its plot method.
    
    Parameters
    ----------
    y_true : float or array-like, optional
        Reference method measurements to plot
    y_pred : float or array-like, optional  
        Test method measurements to plot
    units_mg_dl : bool, optional
        If True (default), assume measurements are in mg/dl.
        If False, assume measurements are in mM.
    figsize : tuple, optional
        Figure size (width, height) in inches
    **kwargs
        Additional keyword arguments passed to ParkesEGA.plot()
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    ega = ParkesEGA(units_mg_dl=units_mg_dl)
    return ega.plot(y_true, y_pred, figsize=figsize, **kwargs)
