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
    from .metrics import BGMetrics, clarke_grid_series
    BGMETRICS_AVAILABLE = True
except ImportError:
    BGMETRICS_AVAILABLE = False


def plot_temporal_prediction(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            input_sequences: Optional[np.ndarray] = None,
                            prediction_horizon: int = 6,
                            sequence_length: int = 12,
                            sample_interval_minutes: int = 5,
                            num_examples: int = 4,
                            title: str = "Temporal Glucose Prediction",
                            figsize: Tuple[int, int] = (16, 10),
                            save_path: Optional[str] = None):
    """
    Plot predictions showing the temporal relationship between history and future.
    
    This visualization shows:
    - History window (sequence_length steps back)
    - Prediction point (prediction_horizon steps ahead)
    - Different horizons will show different temporal spans
    
    Args:
        y_true: True glucose values (test set)
        y_pred: Predicted glucose values (test set)
        input_sequences: Historical sequences used as input (N, sequence_length, features) - if None, will simulate
        prediction_horizon: How many steps ahead we're predicting
        sequence_length: How many steps of history were used
        sample_interval_minutes: Minutes between samples (default 5)
        num_examples: Number of example predictions to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create plot.")
        return None
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    horizon_minutes = prediction_horizon * sample_interval_minutes
    history_minutes = sequence_length * sample_interval_minutes
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Enhanced title
    fig.suptitle(f'{title}\nPrediction Horizon: {horizon_minutes} min ({prediction_horizon} steps) | ' +
                f'MAE: {mae:.2f} mg/dL | RMSE: {rmse:.2f} mg/dL',
                fontsize=14, fontweight='bold')
    
    # Use fixed prediction indices for consistency (e.g., 3rd, 33rd, 66th, 100th)
    # This ensures the same examples are shown every time
    base_indices = [3, 33, 66, 100]  # Fixed positions in test set
    indices = [min(idx, len(y_true) - 1) for idx in base_indices[:num_examples]]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Get or simulate history
        if input_sequences is not None and idx < len(input_sequences):
            # Use actual input sequence (take first feature if multivariate)
            if input_sequences.ndim == 3:
                history_true = input_sequences[idx, :, 0]  # First feature (glucose)
            else:
                history_true = input_sequences[idx, :]
        else:
            # Simulate history - linear interpolation from a reasonable range
            history_true = np.linspace(y_true[idx] - 20, y_true[idx], sequence_length)
        
        # Create time axis
        # Negative time = history, 0 = now, positive = future
        time_history = np.arange(-history_minutes, 0, sample_interval_minutes)
        time_prediction = horizon_minutes
        
        # Plot history
        ax.plot(time_history, history_true, 'b-', linewidth=2, 
               label='Historical data', marker='o', markersize=4, alpha=0.7)
        
        # Plot "now" marker
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, 
                  label='Now (prediction time)', alpha=0.7)
        
        # Plot prediction point
        ax.plot([0, time_prediction], [history_true[-1], y_pred[idx]], 
               'r--', linewidth=2, alpha=0.5)
        ax.plot(time_prediction, y_true[idx], 'bo', markersize=10, 
               label=f'True @ +{horizon_minutes}min', zorder=5)
        ax.plot(time_prediction, y_pred[idx], 'r^', markersize=10, 
               label=f'Predicted @ +{horizon_minutes}min', zorder=5)
        
        # Add error annotation
        error = abs(y_true[idx] - y_pred[idx])
        mid_y = (y_true[idx] + y_pred[idx]) / 2
        ax.annotate(f'Error: {error:.1f} mg/dL', 
                   xy=(time_prediction, mid_y),
                   xytext=(time_prediction + 10, mid_y),
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Shade prediction region
        ax.axvspan(0, time_prediction, alpha=0.1, color='yellow', 
                  label=f'Prediction span ({horizon_minutes} min)')
        
        # Clinical ranges
        ax.axhspan(70, 180, color='g', alpha=0.05)
        ax.axhspan(0, 70, color='r', alpha=0.05)
        ax.axhspan(180, 400, color='y', alpha=0.05)
        
        # Formatting
        ax.set_xlabel('Time (minutes relative to prediction point)', fontsize=10)
        ax.set_ylabel('Glucose (mg/dL)', fontsize=10)
        ax.set_title(f'Example {i+1}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Set x-axis limits to show temporal span - THIS IS KEY!
        # Different horizons will have different x-axis ranges
        ax.set_xlim([-history_minutes - 10, time_prediction + 15])
        ax.set_ylim([50, 250])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     timestamps: Optional[np.ndarray] = None,
                     uncertainty: Optional[np.ndarray] = None, 
                     title: str = "Glucose Prediction",
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None,
                     sample_interval_minutes: int = 5):
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
        sample_interval_minutes: Minutes between samples (default 5)
        
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create plot.")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis in minutes for standardization
    if timestamps is not None and PANDAS_AVAILABLE:
        x = timestamps
        use_time_formatter = True
    else:
        # Standardize x-axis to show time in minutes
        x = np.arange(len(y_true)) * sample_interval_minutes
        use_time_formatter = False
    
    # Plot the true values
    ax.plot(x, y_true, 'b-', label='True Glucose', linewidth=2, alpha=0.7)
    
    # Plot the predicted values
    ax.plot(x, y_pred, 'r-', label='Predicted Glucose', linewidth=2, alpha=0.7)
    
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
    
    
    enhanced_title = f"{title}\n"

    ax.set_title(enhanced_title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Glucose (mg/dL)', fontsize=11)
    
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


def _note_excluded_pairs(title: Optional[str], excluded: int, total: int):
    """
    Add the excluded-pair count to a grid figure's title.

    A grid panel that quietly drops points looks like a clean result, so the
    figure carries the same admission the metric does.
    """
    base = title if title else "Clarke Error Grid"
    if not excluded:
        return base
    return (f"{base}\n({excluded} of {total} pairs outside the grid domain, "
            f"not shown)")


def plot_clarke_analysis(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         figsize: Tuple[int, int] = (10, 10),
                         title: Optional[str] = None,
                         save_path: Optional[str] = None):
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
        # Use the new implementation, on the pairs the grid can classify. The
        # caller in benchmark/experiments/configured.py does not guard this
        # call, so an out-of-domain prediction here would fail the whole seed.
        if BGMETRICS_AVAILABLE:
            grid_true, grid_pred, n_off_grid = clarke_grid_series(y_true, y_pred)
        else:
            grid_true, grid_pred, n_off_grid = y_true, y_pred, 0
        if n_off_grid:
            title = _note_excluded_pairs(title, n_off_grid, np.size(y_true))
        ega = ClarkeEGA()
        fig = ega.plot(grid_true, grid_pred, figsize=figsize, title=title)
        
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
                        save_path: Optional[str] = None):
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
                               prediction_horizon: Optional[int] = None,
                               sequence_length: int = 12,
                               input_sequences: Optional[np.ndarray] = None,
                               sample_interval_minutes: int = 5,
                               figsize: Tuple[int, int] = (16, 16),
                               save_path: Optional[str] = None):
    """
    Create a comprehensive dashboard with predictions, error analysis, error grids, and temporal visualization.
    
    Args:
        y_true: True glucose values
        y_pred: Predicted glucose values
        timestamps: Optional timestamps for time-series plot
        metrics: Optional dictionary of evaluation metrics
        patient_id: Optional patient identifier
        diabetes_type: Type of diabetes for Parkes analysis
        prediction_horizon: Optional prediction horizon in timesteps
        sequence_length: Length of input sequence used for predictions
        input_sequences: Optional input sequences for temporal visualization
        sample_interval_minutes: Minutes between samples (default 5)
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure or None if plotting is not available
    """
    if not PLOTTING_AVAILABLE:
        warnings.warn("Matplotlib not available. Cannot create dashboard.")
        return None
    
    # Calculate metrics if not provided.
    #
    # Guarded like the grid-drawing blocks below: a dashboard is a figure, and a
    # figure must never be the thing that fails a training run. The error grids
    # now exclude out-of-domain pairs themselves, so this guard is for whatever
    # the next bad input turns out to be, and it costs only the summary table
    # rather than the whole seed.
    if metrics is None and BGMETRICS_AVAILABLE:
        try:
            metrics = BGMetrics.calculate_comprehensive_metrics(y_true, y_pred)
        except Exception as e:
            warnings.warn(f"Failed to compute dashboard metrics table: {e}")
            metrics = None
    
    # Create enhanced title with prediction horizon
    if title is None:
        title = "Blood Glucose Prediction Dashboard"
    
    if prediction_horizon is not None:
        minutes = prediction_horizon * sample_interval_minutes
        title = f"{title} - {minutes} min ({prediction_horizon} steps) Prediction Horizon"
    
    # Create figure with subplots - now with 4 rows to include temporal visualization
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[2, 1, 1], hspace=0.3)
    
    # Time series plot (top row, spanning 2 columns) - WITH MINUTES ON X-AXIS
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(y_true))
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
    
    # The time-series and error panels above draw every prediction. The Clarke
    # panel draws the pairs its grid can classify and says what it left out --
    # handed raw output it raised inside the guard below, which drew the zone
    # background and nothing else: no points, no boundaries, no labels. Parkes
    # places any value, so it keeps the full series.
    grid_true, grid_pred = y_true, y_pred

    # Clarke Error Grid (row 1, left)
    ax3 = fig.add_subplot(gs[1, 0])
    if CLARKE_EGA_AVAILABLE:
        try:
            if BGMETRICS_AVAILABLE:
                clarke_true, clarke_pred, n_off_grid = clarke_grid_series(y_true, y_pred)
            else:
                clarke_true, clarke_pred, n_off_grid = y_true, y_pred, 0
            # Direct approach - draw the Clarke grid on the given axis
            clarke_ega = ClarkeEGA()
            ax3 = clarke_ega.plot_on_axes(ax3, clarke_true, clarke_pred)
            ax3.set_title(
                _note_excluded_pairs('Clarke Error Grid', n_off_grid, np.size(y_true)),
                fontsize=11, fontweight='bold',
            )
        except Exception as e:
            warnings.warn(f"Failed to render Clarke EGA: {e}")

    # Parkes Error Grid (row 1, right)
    ax4 = fig.add_subplot(gs[1, 2])
    if PARKES_EGA_AVAILABLE:
        try:
            # Direct approach - draw the Parkes grid on the given axis
            parkes_ega = ParkesEGA(units_mg_dl=True)
            ax4 = parkes_ega.plot_on_axes(ax4, grid_true, grid_pred)
            ax4.set_title(f"Parkes Error Grid (Type {diabetes_type})", fontsize=12, fontweight='bold')
        except Exception as e:
            warnings.warn(f"Failed to render Parkes EGA: {e}")
            
    else:
        ax4.text(0.5, 0.5, "Parkes Error Grid not available", 
                 ha='center', va='center', fontsize=12)
        ax4.axis('off')
    
    # Temporal Prediction Examples (rows 2-3, spanning all columns)
    # This shows the temporal relationship between history and predictions
    if prediction_horizon is not None:
        # Create 1 temporal example subplots
        ax_temp1 = fig.add_subplot(gs[2, :])
        
        horizon_minutes = prediction_horizon * sample_interval_minutes
        history_minutes = sequence_length * sample_interval_minutes
        
        # Use fixed prediction index for consistency (always show the 3rd prediction)
        num_examples = 1
        indices = [min(3, len(y_true) - 1)]  # Fixed at index 3 for reproducibility

        for ax_idx, (ax, idx) in enumerate(zip([ax_temp1], indices)):
            # Get or simulate history
            if input_sequences is not None and idx < len(input_sequences):
                # Use actual input sequence
                if input_sequences.ndim == 3:
                    history_true = input_sequences[idx, :, 0]  # First feature (glucose)
                else:
                    history_true = input_sequences[idx, :]
            else:
                # Simulate history
                history_true = np.linspace(y_true[idx] - 20, y_true[idx], sequence_length)
            
            # Create time axis
            time_history = np.arange(-history_minutes, 0, sample_interval_minutes)
            time_prediction = horizon_minutes
            
            # Plot history
            ax.plot(time_history, history_true, 'b-', linewidth=2.5, 
                   label='Historical data', marker='o', markersize=5, alpha=0.8)
            
            # Plot "now" marker
            ax.axvline(x=0, color='green', linestyle='--', linewidth=2.5, 
                      label='Now (prediction time)', alpha=0.8)
            
            # Plot prediction trajectory
            ax.plot([0, time_prediction], [history_true[-1], y_pred[idx]], 
                   'r--', linewidth=2, alpha=0.6)
            
            # Plot prediction points
            ax.plot(time_prediction, y_true[idx], 'bo', markersize=12, 
                   label=f'True @ +{horizon_minutes}min', zorder=5)
            ax.plot(time_prediction, y_pred[idx], 'r^', markersize=12, 
                   label=f'Predicted @ +{horizon_minutes}min', zorder=5)
            
            # Add error annotation
            error = abs(y_true[idx] - y_pred[idx])
            mid_y = (y_true[idx] + y_pred[idx]) / 2
            ax.annotate(f'Error: {error:.1f} mg/dL', 
                       xy=(time_prediction, mid_y),
                       xytext=(time_prediction + 10, mid_y),
                       fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Shade prediction region
            ax.axvspan(0, time_prediction, alpha=0.15, color='yellow', 
                      label=f'Prediction span ({horizon_minutes} min)')
            
            # Clinical ranges
            ax.axhspan(70, 180, color='g', alpha=0.08)
            ax.axhspan(0, 70, color='r', alpha=0.08)
            ax.axhspan(180, 400, color='y', alpha=0.08)
            
            # Formatting
            ax.set_xlabel('Time (minutes relative to prediction point)', fontsize=11)
            ax.set_ylabel('Glucose (mg/dL)', fontsize=11)
            ax.set_title(f'Temporal Prediction Example {ax_idx + 1}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=9, loc='best')
            
            # Set x-axis limits - THIS SHOWS THE TEMPORAL SPAN!
            ax.set_xlim([-history_minutes - 10, time_prediction + 15])
            ax.set_ylim([50, 250])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig