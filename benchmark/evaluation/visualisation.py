"""
Visualization tools for blood glucose prediction evaluation.

This module provides functions for creating various visualizations
of prediction results and error analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib not available. Visualization functions will not work.")

try:
    import pandas as pd #type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Some visualization functions will have limited functionality.")

# Import the new EGA implementations
try:
    from .clarke_ega import ClarkeEGA
    CLARKE_EGA_AVAILABLE = True
except ImportError:
    CLARKE_EGA_AVAILABLE = False
    warnings.warn("Clarke EGA implementation not available")

try:
    from .parkes_ega import ParkesEGA
    PARKES_EGA_AVAILABLE = True
except ImportError:
    PARKES_EGA_AVAILABLE = False
    warnings.warn("Parkes EGA implementation not available")

# Import BGMetrics for fallback
try:
    from .metrics import BGMetrics
    BGMETRICS_AVAILABLE = True
except ImportError:
    BGMETRICS_AVAILABLE = False


def plot_predictions(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     timestamps: Optional[np.ndarray] = None,
                     uncertainty: Optional[np.ndarray] = None, 
                     title: str = "Glucose Prediction",
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None) -> Optional[Figure]:
    """
    Plot true vs predicted glucose values with optional uncertainty.
    
    Args:
        y_true: True glucose values
        y_pred: Predicted glucose values
        timestamps: Optional timestamps for x-axis
        uncertainty: Optional prediction uncertainty (std dev)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create plot.")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = timestamps if timestamps is not None else np.arange(len(y_true))
    
    # Plot the true values
    ax.plot(x, y_true, 'b-', label='True Glucose')
    
    # Plot the predicted values
    ax.plot(x, y_pred, 'r-', label='Predicted Glucose')
    
    # Add uncertainty if provided
    if uncertainty is not None:
        ax.fill_between(x, 
                        y_pred - 2*uncertainty,
                        y_pred + 2*uncertainty,
                        color='r', alpha=0.2,
                        label='95% Confidence Interval')
    
    # Add clinical ranges
    ax.axhspan(70, 180, color='g', alpha=0.1, label='Target Range (70-180 mg/dL)')
    ax.axhspan(0, 70, color='r', alpha=0.1, label='Hypoglycemia (<70 mg/dL)')
    ax.axhspan(180, max(np.max(y_true), np.max(y_pred))*1.1, color='y', alpha=0.1, label='Hyperglycemia (>180 mg/dL)')
    
    # Configure the plot
    ax.set_title(title)
    ax.set_ylabel('Glucose (mg/dL)')
    
    if timestamps is not None and PANDAS_AVAILABLE:
        # Format the x-axis for timestamps
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlabel('Time')
        except:
            # If timestamps are not datetime objects, just use them as is
            ax.set_xlabel('Time')
    else:
        ax.set_xlabel('Sample Index')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_clarke_analysis(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         figsize: Tuple[int, int] = (10, 10),
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> Optional[Figure]:
    """
    Plot Clarke Error Grid Analysis for glucose predictions.
    
    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        figsize: Figure size
        title: Custom plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create plot.")
        return None
    
    if CLARKE_EGA_AVAILABLE:
        # Use the new implementation
        ega = ClarkeEGA()
        fig = ega.plot(y_true, y_pred, figsize=figsize, title=title)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    else:
        warnings.warn("Clarke EGA implementation not available. Cannot create Clarke Error Grid.")
        return None


def plot_parkes_analysis(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        diabetes_type: int = 1,
                        figsize: Tuple[int, int] = (10, 10),
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> Optional[Figure]:
    """
    Plot Parkes Error Grid Analysis for glucose predictions.
    
    Args:
        y_true: True glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        diabetes_type: Type of diabetes (1 or 2) - currently only Type 1 supported
        figsize: Figure size
        title: Custom plot title
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create plot.")
        return None
    
    if PARKES_EGA_AVAILABLE:
        # Use the new implementation
        ega = ParkesEGA(units_mg_dl=True)
        
        if title is None:
            title = f"Parkes Error Grid Analysis (Type {diabetes_type} Diabetes)"
        
        fig = ega.plot(y_true, y_pred, figsize=figsize, title=title)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    else:
        warnings.warn("Parkes EGA implementation not available. Cannot create Parkes Error Grid.")
        return None


def create_prediction_dashboard(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               timestamps: Optional[np.ndarray] = None,
                               metrics: Optional[Dict[str, Union[float, Dict]]] = None,
                               title: Optional[str] = None,
                               patient_id: Optional[str] = None,
                               diabetes_type: int = 1,
                               figsize: Tuple[int, int] = (16, 12),
                               save_path: Optional[str] = None) -> Optional[Figure]:
    """
    Create a comprehensive dashboard with predictions, error analysis, and error grids.
    
    Args:
        y_true: True glucose values
        y_pred: Predicted glucose values
        timestamps: Optional timestamps for time-series plot
        metrics: Optional dictionary of evaluation metrics
        patient_id: Optional patient identifier
        diabetes_type: Type of diabetes for Parkes analysis
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create dashboard.")
        return None
    
    # Calculate metrics if not provided
    if metrics is None and BGMETRICS_AVAILABLE:
        metrics = BGMetrics.calculate_comprehensive_metrics(y_true, y_pred)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title if title is not None else "Blood Glucose Prediction Dashboard", fontsize=16, fontweight='bold')
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1])
    
    # Time series plot (top row, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    x = timestamps if timestamps is not None else np.arange(len(y_true))
    ax1.plot(x, y_true, 'b-', label='True Glucose', linewidth=2)
    ax1.plot(x, y_pred, 'r-', label='Predicted Glucose', linewidth=2)
    
    # Add clinical ranges
    ax1.axhspan(70, 180, color='g', alpha=0.1, label='Target Range')
    ax1.axhspan(0, 70, color='r', alpha=0.1, label='Hypoglycemia')
    ax1.axhspan(180, 400, color='y', alpha=0.1, label='Hyperglycemia')
    
    ax1.set_title('Glucose Prediction Time Series', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_xlabel('Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Metrics table (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    if metrics is not None:
        metrics_text = f"Evaluation Metrics"
        if patient_id:
            metrics_text += f" - Patient {patient_id}"
        
        table_data = []
        if 'rmse' in metrics:
            table_data.append(['RMSE', f"{metrics['rmse']:.2f}"])
        if 'mae' in metrics:
            table_data.append(['MAE', f"{metrics['mae']:.2f}"])
        if 'mape' in metrics:
            table_data.append(['MAPE', f"{metrics['mape']:.2f}%"])
        
        # Add Clarke zones
        if 'clarke_zones' in metrics:
            table_data.append(['', ''])  # Empty row
            table_data.append(['Clarke Zone A', f"{metrics['clarke_zones']['A']:.1f}%"])
            table_data.append(['Clarke Zone B', f"{metrics['clarke_zones']['B']:.1f}%"])
        
        # Add Parkes zones
        if 'parkes_zones' in metrics:
            table_data.append(['', ''])  # Empty row
            table_data.append(['Parkes Zone A', f"{metrics['parkes_zones']['A']:.1f}%"])
            table_data.append(['Parkes Zone B', f"{metrics['parkes_zones']['B']:.1f}%"])
        
        # Create table
        table = ax2.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):  # +1 for header
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif table_data[i-1][0] == '':  # Empty rows
                    cell.set_facecolor('#f0f0f0')
                elif 'Zone A' in table_data[i-1][0]:
                    cell.set_facecolor('#90EE90')  # Light green for Zone A
                elif 'Zone B' in table_data[i-1][0]:
                    cell.set_facecolor('#FFFFE0')  # Light yellow for Zone B
        
        ax2.set_title(metrics_text, fontsize=12, fontweight='bold')
    
    # Clarke Error Grid (middle)
    ax3 = fig.add_subplot(gs[1, 0])
    if CLARKE_EGA_AVAILABLE:
        try:
            # Direct approach - draw the Clarke grid on the given axis
            clarke_ega = ClarkeEGA()
            ax3 = clarke_ega.plot_on_axes(ax3, y_true, y_pred)
            ax3.set_title('Clarke Error Grid', fontsize=11, fontweight='bold')
        except Exception as e:
            warnings.warn(f"Failed to render Clarke EGA: {e}")

    # Parkes Error Grid (bottom, spanning all columns)
    ax4 = fig.add_subplot(gs[1, 2])
    if PARKES_EGA_AVAILABLE:
        try:
            # Direct approach - draw the Parkes grid on the given axis
            parkes_ega = ParkesEGA(units_mg_dl=True)
            ax4 = parkes_ega.plot_on_axes(ax4, y_true, y_pred)
            ax4.set_title(f"Parkes Error Grid (Type {diabetes_type})", fontsize=12, fontweight='bold')
        except Exception as e:
            warnings.warn(f"Failed to render Parkes EGA: {e}")
            
    else:
        ax4.text(0.5, 0.5, "Parkes Error Grid not available", 
                 ha='center', va='center', fontsize=12)
        ax4.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig