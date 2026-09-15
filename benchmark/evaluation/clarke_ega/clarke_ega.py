"""
Clarke Error Grid Analysis implementation.

Converted from MATLAB implementation by Edgar Guevara Codina.
Optimized for use with machine learning predictions (y_pred vs y_true).

References:
[1] A. Maran et al. "Continuous Subcutaneous Glucose Monitoring in Diabetic 
    Patients" Diabetes Care, Volume 25, Number 2, February 2002
[2] B.P. Kovatchev et al. "Evaluating the Accuracy of Continuous Glucose-
    Monitoring Sensors" Diabetes Care, Volume 27, Number 8, August 2004
[3] E. Guevara and F. J. Gonzalez, "Prediction of Glucose Concentration by
    Impedance Phase Measurements," in MEDICAL PHYSICS: Tenth Mexican 
    Symposium on Medical Physics, Mexico City (Mexico), 2008, vol. 1032, pp.
    259–261. 
[4] E. Guevara and F. J. Gonzalez, "Joint optical-electrical technique for
    noninvasive glucose monitoring," REVISTA MEXICANA DE FISICA, vol. 56, 
    no. 5, pp. 430–434, Sep. 2010.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Tuple, Dict, List, Optional, Union
import warnings


def get_clarke_zone_colors() -> Dict[str, str]:
    """
    Get standard colors for each Clarke EGA zone.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping zone letters to matplotlib color codes.
    """
    return {
        'A': 'green',      # Clinically accurate
        'B': 'yellow',     # Benign errors  
        'C': 'orange',     # Overcorrection errors
        'D': 'lightcoral',        # Failure to detect errors
        'E': 'red'  # Dangerous errors
    }


class ClarkeEGA:
    """
    Clarke Error Grid Analysis for evaluating glucose measurement accuracy.
    
    This class provides methods to analyze glucose measurement pairs using
    the Clarke Error Grid, which classifies measurement errors into zones
    based on their clinical significance.
    """
    
    def _create_zone_background(self, ax, alpha: float = 0.3):
        """
        Create zone background colors using pixel-based approach.
        This ensures colors match exactly with the mathematical classification.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to add zone backgrounds to
        alpha : float
            Transparency of zone backgrounds
        """
        # Create a grid of points to classify
        x_grid, y_grid = np.meshgrid(np.arange(0, self._max_range + 1, 2), np.arange(0, self._max_range + 1, 2))
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        
        # Classify all grid points
        results = self.analyze(x_flat, y_flat)
        zone_map = results['zones'].reshape(x_grid.shape)
        
        # Get colors
        colors = get_clarke_zone_colors()
        zone_colors = ['green', 'yellow', 'orange', 'lightcoral', 'red']  # A, B, C, D, E
        
        # Create colored background using imshow
        # Convert zone numbers (1-5) to color indices (0-4)
        zone_map_colors = zone_map - 1
        
        # Create RGB array
        color_array = np.zeros((zone_map.shape[0], zone_map.shape[1], 4))
        
        for i, color_name in enumerate(zone_colors):
            mask = zone_map_colors == i
            if color_name == 'green':
                color_array[mask] = [0, 0.5, 0, alpha]
            elif color_name == 'yellow':
                color_array[mask] = [1, 1, 0, alpha]
            elif color_name == 'orange':
                color_array[mask] = [1, 0.65, 0, alpha]
            elif color_name == 'red':
                color_array[mask] = [1, 0, 0, alpha]
            elif color_name == 'lightcoral':
                color_array[mask] = [0.94, 0.5, 0.5, alpha]
        
        # Display the background
        ax.imshow(color_array, extent=[0, self._max_range, 0, self._max_range], aspect='equal', 
                 origin='lower', interpolation='nearest')
    
    def _get_zone_polygons(self) -> Dict[str, np.ndarray]:
        """
        Define zone boundaries as polygons for visualization.
        These polygons precisely match the mathematical zone classification logic.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping zone letters to polygon vertices
        """
        # Simplified version - we'll use the pixel-based approach instead
        return {
            'A': [],
            'B': [],
            'C': [], 
            'D': [],
            'E': []
        }
    
    def __init__(self):
        """Initialize Clarke EGA analyzer."""
        # The Clarke grid is published on 0-400 mg/dL and the zone rules are
        # only defined there, so the range is not a tunable. The OhioT1DM CGM
        # reports [40, 400] and saturates at both ends, so no target can fall
        # outside it; callers clip predictions to the sensor range instead.
        self._max_range = 400  # mg/dL
        
    def analyze(self, y_true: Union[float, np.ndarray], 
                y_pred: Union[float, np.ndarray]) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Perform Clarke Error Grid Analysis.
        
        Parameters
        ----------
        y_true : float or array-like
            Reference glucose values (mg/dl)
        y_pred : float or array-like  
            Predicted/estimated glucose values (mg/dl)
            
        Returns
        -------
        Dict[str, Union[int, float, np.ndarray]]
            Analysis results containing:
            - 'total': Total points per zone [A, B, C, D, E]
            - 'percentage': Percentage of data in each zone
            - 'zones': Zone assignment for each point
            - 'total_points': Total number of points analyzed
            
        Raises
        ------
        ValueError
            If input arrays have different lengths or values are out of 
            physiological range (0-600 mg/dL)
        """
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
        
        # Input validation
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
            
        if (np.max(y_true) > self._max_range) or (np.max(y_pred) > self._max_range) or \
           (np.min(y_true) < 0) or (np.min(y_pred) < 0):
            raise ValueError(f"Values must be in physiological range (0-{self._max_range} mg/dL)")
        
        n = len(y_true)
        zones = np.zeros(n, dtype=int)  # 1=A, 2=B, 3=C, 4=D, 5=E
        total = np.zeros(5, dtype=int)
        
        # Zone classification logic (converted from MATLAB)
        for i in range(n):
            y_ref = y_true[i]
            y_test = y_pred[i]
            
            # Zone A: Within ±20% or both in hypoglycemic range
            if ((y_test <= 70 and y_ref <= 70) or 
                (y_test <= 1.2 * y_ref and y_test >= 0.8 * y_ref)):
                zones[i] = 1
                total[0] += 1
                
            # Zone E: Dangerous errors
            elif ((y_ref >= 180 and y_test <= 70) or 
                  (y_ref <= 70 and y_test >= 180)):
                zones[i] = 5
                total[4] += 1
                
            # Zone C: Overcorrection errors
            elif (((y_ref >= 70 and y_ref <= 290) and (y_test >= y_ref + 110)) or
                  ((y_ref >= 130 and y_ref <= 180) and (y_test <= (7/5) * y_ref - 182))):
                zones[i] = 3
                total[2] += 1
                
            # Zone D: Failure to detect errors
            elif (((y_ref >= 240) and (y_test >= 70 and y_test <= 180)) or
                  (y_ref <= 175/3 and y_test <= 180 and y_test >= 70) or
                  ((y_ref >= 175/3 and y_ref <= 70) and y_test >= (6/5) * y_ref)):
                zones[i] = 4
                total[3] += 1
                
            # Zone B: Benign errors (everything else)
            else:
                zones[i] = 2
                total[1] += 1
        
        # Calculate percentages
        percentage = (total / n) * 100
        
        return {
            'total': total,
            'percentage': percentage,
            'zones': zones,
            'total_points': n
        }
    
    def plot(self, y_true: Union[float, np.ndarray], 
             y_pred: Union[float, np.ndarray],
             figsize: Tuple[int, int] = (8, 8),
             point_size: float = 30,
             title: Optional[str] = None,
             save_figure: bool = False,
             filename: str = 'Clarke_EGA',
             alpha: float = 0.5,
             color_zones: bool = True,
             color_points: bool = True) -> plt.Figure:
        """
        Plot Clarke Error Grid with data points.
        
        Parameters
        ----------
        y_true : float or array-like
            Reference glucose values (mg/dl)
        y_pred : float or array-like
            Predicted glucose values (mg/dl)
        figsize : tuple, optional
            Figure size (width, height) in inches
        point_size : float, optional
            Size of data points
        title : str, optional
            Custom plot title
        save_figure : bool, optional
            Whether to save the figure
        filename : str, optional
            Filename for saved figure (without extension)
        alpha : float, optional
            Transparency of zone fills (default 0.3)
        color_zones : bool, optional
            Whether to color the zones (default True)
        color_points : bool, optional
            Whether to color points by zone (default True)
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
        """
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get zone colors and polygons
        colors = get_clarke_zone_colors()
        zone_polygons = self._get_zone_polygons()
        
        # Fill zones with colors if requested
        if color_zones:
            # Use pixel-based zone background for accurate coloring
            self._create_zone_background(ax, alpha)
        
        # Set up axes first
        ax.set_xlim(0, self._max_range)
        ax.set_ylim(0, self._max_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Plot data points
        if len(y_true) > 0 and color_points:
            # Get zone classifications for each point
            results = self.analyze(y_true, y_pred)
            zone_classifications = results['zones']
            
            # Map zone numbers to letters
            zone_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

            zone_descriptions = {
                'A': 'Acceptable',
                'B': 'Borderline',
                'C': 'Critical',
                'D': 'Dangerous',
                'E': 'Erroneous'
            }

            # Plot points colored by zone
            for zone_num, zone_letter in zone_map.items():
                mask = zone_classifications == zone_num
                if np.any(mask):
                    ax.scatter(y_true[mask], y_pred[mask], 
                             s=point_size, 
                             c=colors[zone_letter], 
                             marker='o',
                             edgecolors='black',
                             linewidth=0.5,
                             alpha=0.8,
                             label=f'Zone {zone_letter}: {zone_descriptions[zone_letter]}')
        elif len(y_true) > 0:
            # Plot all points in black if color_points is False
            ax.scatter(y_true, y_pred, s=point_size, c='black', 
                      marker='o', facecolors='black', edgecolors='black', alpha=0.5)
        
        # Perfect agreement line (45° line)
        ax.plot([0, self._max_range], [0, self._max_range], 'k:', linewidth=1, alpha=0.9, 
               label='Perfect agreement')
        
        # Zone boundaries (converted from MATLAB)
        ax.plot([0, 175/3], [70, 70], 'k-', linewidth=1.5)
        ax.plot([175/3, self._max_range/1.2], [70, self._max_range], 'k-', linewidth=1.5)
        ax.plot([70, 70], [84, self._max_range], 'k-', linewidth=1.5)
        ax.plot([0, 70], [180, 180], 'k-', linewidth=1.5)
        # Upper C edge is y_test = y_ref + 110, and the rule only applies up to
        # y_ref = 290, so this segment ends at (290, 400) whatever the axis limit.
        ax.plot([70, 290], [180, 400], 'k-', linewidth=1.5)
        ax.plot([70, 70], [0, 56], 'k-', linewidth=1.5)
        # Lower A edge is y_test = 0.8 * y_ref, so both endpoints scale together.
        ax.plot([70, self._max_range], [56, 0.8 * self._max_range], 'k-', linewidth=1.5)
        ax.plot([180, 180], [0, 70], 'k-', linewidth=1.5)
        ax.plot([180, self._max_range], [70, 70], 'k-', linewidth=1.5)
        ax.plot([240, 240], [70, 180], 'k-', linewidth=1.5)
        ax.plot([240, self._max_range], [180, 180], 'k-', linewidth=1.5)
        ax.plot([130, 180], [0, 70], 'k-', linewidth=1.5)
        
        # Zone labels with colored backgrounds
        label_style = dict(boxstyle="round,pad=0.3", alpha=0.9, edgecolor='black', linewidth=1)
        
        ax.text(30, 20, 'A', fontsize=14, fontweight='bold', 
               bbox=dict(**label_style, facecolor=colors['A']))
        ax.text(30, 150, 'D', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['D']))
        ax.text(30, 380, 'E', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['E']))
        ax.text(150, 380, 'C', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['C']))
        ax.text(160, 20, 'C', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['C']))
        ax.text(380, 20, 'E', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['E']))
        ax.text(380, 120, 'D', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['D']))
        ax.text(380, 260, 'B', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['B']))
        ax.text(280, 380, 'B', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['B']))
        
        # Labels
        ax.set_xlabel('Reference Concentration [mg/dl]', fontsize=12)
        ax.set_ylabel('Predicted Concentration [mg/dl]', fontsize=12)
        
        if title is None:
            title = "Clarke's Error Grid Analysis"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='white', edgecolor='black', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_figure:
            fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Figure saved as {filename}.png")
        
        return fig
    
    def plot_on_axes(self, ax: plt.Axes, y_true: Union[float, np.ndarray], 
                    y_pred: Union[float, np.ndarray],
                    figsize: Tuple[int, int] = (8, 8),
                    point_size: float = 30,
                    title: Optional[str] = None,
                    save_figure: bool = False,
                    filename: str = 'Clarke_EGA',
                    alpha: float = 0.5,
                    color_zones: bool = True,
                    color_points: bool = True) -> plt.Axes:
        """
        Plot Clarke Error Grid on existing axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        y_true : float or array-like
            Reference glucose values (mg/dl)
        y_pred : float or array-like
            Predicted glucose values (mg/dl)
        point_size : float, optional
            Size of data points
        alpha : float, optional
            Transparency of zone fills (default 0.3)
        color_zones : bool, optional
            Whether to color the zones (default True)
        color_points : bool, optional
            Whether to color points by zone (default True)

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the Clarke Error Grid plot
        """
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)
                
        # Get zone colors and polygons
        colors = get_clarke_zone_colors()
        
        # Fill zones with colors if requested
        if color_zones:
            # Use pixel-based zone background for accurate coloring
            self._create_zone_background(ax, alpha)
        
        # Set up axes first
        ax.set_xlim(0, self._max_range)
        ax.set_ylim(0, self._max_range)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Plot data points
        if len(y_true) > 0 and color_points:
            # Get zone classifications for each point
            results = self.analyze(y_true, y_pred)
            zone_classifications = results['zones']
            
            # Map zone numbers to letters
            zone_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

            zone_descriptions = {
                'A': 'Acceptable',
                'B': 'Borderline',
                'C': 'Critical',
                'D': 'Dangerous',
                'E': 'Erroneous'
            }

            # Plot points colored by zone
            for zone_num, zone_letter in zone_map.items():
                mask = zone_classifications == zone_num
                if np.any(mask):
                    ax.scatter(y_true[mask], y_pred[mask], 
                             s=point_size, 
                             c=colors[zone_letter], 
                             marker='o',
                             edgecolors='black',
                             linewidth=0.5,
                             alpha=0.8,
                             label=f'Zone {zone_letter}: {zone_descriptions[zone_letter]}')
        elif len(y_true) > 0:
            # Plot all points in black if color_points is False
            ax.scatter(y_true, y_pred, s=point_size, c='black', 
                      marker='o', facecolors='black', edgecolors='black', alpha=0.5)
        
        # Perfect agreement line (45° line)
        ax.plot([0, self._max_range], [0, self._max_range], 'k:', linewidth=1, alpha=0.9, 
               label='Perfect agreement')
        
        # Zone boundaries (converted from MATLAB)
        ax.plot([0, 175/3], [70, 70], 'k-', linewidth=1.5)
        ax.plot([175/3, self._max_range/1.2], [70, self._max_range], 'k-', linewidth=1.5)
        ax.plot([70, 70], [84, self._max_range], 'k-', linewidth=1.5)
        ax.plot([0, 70], [180, 180], 'k-', linewidth=1.5)
        # Upper C edge is y_test = y_ref + 110, and the rule only applies up to
        # y_ref = 290, so this segment ends at (290, 400) whatever the axis limit.
        ax.plot([70, 290], [180, 400], 'k-', linewidth=1.5)
        ax.plot([70, 70], [0, 56], 'k-', linewidth=1.5)
        # Lower A edge is y_test = 0.8 * y_ref, so both endpoints scale together.
        ax.plot([70, self._max_range], [56, 0.8 * self._max_range], 'k-', linewidth=1.5)
        ax.plot([180, 180], [0, 70], 'k-', linewidth=1.5)
        ax.plot([180, self._max_range], [70, 70], 'k-', linewidth=1.5)
        ax.plot([240, 240], [70, 180], 'k-', linewidth=1.5)
        ax.plot([240, self._max_range], [180, 180], 'k-', linewidth=1.5)
        ax.plot([130, 180], [0, 70], 'k-', linewidth=1.5)
        
        # Zone labels with colored backgrounds
        label_style = dict(boxstyle="round,pad=0.3", alpha=0.9, edgecolor='black', linewidth=1)
        
        ax.text(30, 20, 'A', fontsize=14, fontweight='bold', 
               bbox=dict(**label_style, facecolor=colors['A']))
        ax.text(30, 150, 'D', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['D']))
        ax.text(30, 380, 'E', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['E']))
        ax.text(150, 380, 'C', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['C']))
        ax.text(160, 20, 'C', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['C']))
        ax.text(380, 20, 'E', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['E']))
        ax.text(380, 120, 'D', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['D']))
        ax.text(380, 260, 'B', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['B']))
        ax.text(280, 380, 'B', fontsize=14, fontweight='bold',
               bbox=dict(**label_style, facecolor=colors['B']))
        
        # Labels
        ax.set_xlabel('Reference Concentration [mg/dl]', fontsize=12)
        ax.set_ylabel('Predicted Concentration [mg/dl]', fontsize=12)
        
        if title is None:
            title = "Clarke's Error Grid Analysis"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='white', edgecolor='black', fontsize=10)
        
        return ax

    def print_summary(self, y_true: Union[float, np.ndarray],
                     y_pred: Union[float, np.ndarray]) -> None:
        """
        Print a summary of the Clarke EGA analysis results.
        
        Parameters
        ----------
        y_true : float or array-like
            Reference glucose values (mg/dl)
        y_pred : float or array-like
            Predicted glucose values (mg/dl)
        """
        results = self.analyze(y_true, y_pred)
        
        zone_descriptions = {
            'A': 'Clinically accurate',
            'B': 'Benign errors',
            'C': 'Overcorrection errors',
            'D': 'Failure to detect errors',
            'E': 'Dangerous errors'
        }
        
        print("Clarke Error Grid Analysis Results")
        print("=" * 50)
        print(f"Total points analyzed: {results['total_points']}")
        print()
        
        zones = ['A', 'B', 'C', 'D', 'E']
        for i, zone in enumerate(zones):
            count = results['total'][i]
            percentage = results['percentage'][i]
            desc = zone_descriptions[zone]
            print(f"Zone {zone} ({desc}): {count} points ({percentage:.1f}%)")
        
        # Clinical assessment
        acceptable = results['percentage'][0] + results['percentage'][1]  # A + B
        print(f"\nClinical Assessment:")
        print(f"Clinically acceptable (Zones A+B): {acceptable:.1f}%")
        
        if acceptable >= 95:
            print("✓ Excellent clinical accuracy")
        elif acceptable >= 90:
            print("✓ Good clinical accuracy")
        elif acceptable >= 80:
            print("⚠ Acceptable clinical accuracy")
        else:
            print("✗ Poor clinical accuracy - requires improvement")


# Convenience functions for direct use
def clarke_analysis(y_true: Union[float, np.ndarray], 
                   y_pred: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Clarke Error Grid Analysis.
    
    This is a convenience function that creates a ClarkeEGA instance
    and returns results in MATLAB-compatible format.
    
    Parameters
    ----------
    y_true : float or array-like
        Reference glucose values (mg/dl)
    y_pred : float or array-like
        Predicted glucose values (mg/dl)
        
    Returns
    -------
    tuple
        (total, percentage) where:
        - total: array of counts for zones A, B, C, D, E
        - percentage: array of percentages for zones A, B, C, D, E
    """
    ega = ClarkeEGA()
    results = ega.analyze(y_true, y_pred)
    return results['total'], results['percentage']


def plot_clarke_grid(y_true: Union[float, np.ndarray], 
                    y_pred: Union[float, np.ndarray],
                    figsize: Tuple[int, int] = (8, 8),
                    **kwargs) -> plt.Figure:
    """
    Plot Clarke Error Grid with data points.
    
    This is a convenience function that creates a ClarkeEGA instance
    and calls its plot method.
    
    Parameters
    ----------
    y_true : float or array-like
        Reference glucose values (mg/dl)
    y_pred : float or array-like
        Predicted glucose values (mg/dl)
    figsize : tuple, optional
        Figure size (width, height) in inches
    **kwargs
        Additional keyword arguments passed to ClarkeEGA.plot()
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    ega = ClarkeEGA()
    return ega.plot(y_true, y_pred, figsize=figsize, **kwargs)
