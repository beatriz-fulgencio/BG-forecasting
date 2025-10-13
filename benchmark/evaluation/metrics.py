"""
Standardized evaluation metrics for blood glucose forecasting.

This module implements domain-specific metrics including:
- Standard regression metrics (MAE, RMSE, MAPE)
- Clarke Error Grid Analysis (EGA)
- Parkes Error Grid Analysis (PEGA)
- Time in Range (TIR) metrics
- Continuous Glucose Error Grid Analysis (CG-EGA) TODO
- Clinical Trend Concurrence Analysis (CTCA) TODO
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import pandas as pd # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

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


class BGMetrics:
    """
    Comprehensive metrics calculator for blood glucose forecasting evaluation.
    
    This class provides all standard and domain-specific metrics used in
    blood glucose prediction research.
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred)) # Mean Absolute Error = (1/n) * Σ|y_true - y_pred|
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Square Error.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2)) # Root Mean Squared Error = sqrt((1/n) * Σ(y_true - y_pred)^2)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            
        Returns:
            MAPE value as percentage
        """
        mask = y_true != 0
        if not np.any(mask):
            warnings.warn("All true values are zero, MAPE is undefined")
            return float('inf')
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 # Mean Absolute Percentage Error = (1/n) * Σ(|y_true - y_pred| / y_true) * 100
    
    @staticmethod
    def mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Relative Difference - standard in CGM literature.
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            MARD value as percentage
        """
        return np.mean(np.abs(y_true - y_pred) / y_true) * 100 # Mean Absolute Relative Difference = (1/n) * Σ(|y_true - y_pred| / y_true) * 100

    @staticmethod
    def time_in_range(glucose_values: np.ndarray, 
                     lower_bound: float = 70, 
                     upper_bound: float = 180) -> float:
        """
        Time in Range (TIR) - percentage of readings within target range.
        
        Args:
            glucose_values: Glucose readings (mg/dL)
            lower_bound: Lower bound of target range (default: 70 mg/dL)
            upper_bound: Upper bound of target range (default: 180 mg/dL)
            
        Returns:
            TIR as percentage
        """
        in_range = (glucose_values >= lower_bound) & (glucose_values <= upper_bound)
        return np.mean(in_range) * 100
    
    @staticmethod
    def time_below_range(glucose_values: np.ndarray, threshold: float = 70) -> float:
        """
        Time Below Range (TBR) - percentage of readings below threshold.
        
        Args:
            glucose_values: Glucose readings (mg/dL)
            threshold: Threshold for hypoglycemia (default: 70 mg/dL)
            
        Returns:
            TBR as percentage
        """
        below_range = glucose_values < threshold
        return np.mean(below_range) * 100
    
    @staticmethod
    def time_above_range(glucose_values: np.ndarray, threshold: float = 180) -> float:
        """
        Time Above Range (TAR) - percentage of readings above threshold.
        
        Args:
            glucose_values: Glucose readings (mg/dL)
            threshold: Threshold for hyperglycemia (default: 180 mg/dL)
            
        Returns:
            TAR as percentage
        """
        above_range = glucose_values > threshold
        return np.mean(above_range) * 100
    
    @staticmethod
    def comparing_time_in_range(glucose_values: np.ndarray, reference_values: np.ndarray) -> Dict[str, float]:
        """
        Compare Time in Range (TIR) metrics between predicted and reference glucose values.

        Args:
            glucose_values: Predicted glucose readings (mg/dL)
            reference_values: True glucose readings (mg/dL)
            
        Returns:
            Difference in TIR as percentage dictionary
        """
        return {
            'time_in_range': BGMetrics.time_in_range(glucose_values) - BGMetrics.time_in_range(reference_values),
            'time_below_range': BGMetrics.time_below_range(glucose_values) - BGMetrics.time_below_range(reference_values),
            'time_above_range': BGMetrics.time_above_range(glucose_values) - BGMetrics.time_above_range(reference_values)
        }
    
    @staticmethod
    def clarke_error_grid_analysis(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Clarke Error Grid Analysis for glucose prediction evaluation.
        
        Uses the new ClarkeEGA implementation for more accurate zone classification.
        
        Zones:
        - Zone A: Clinically acceptable (within 20% or ±15 mg/dL for values <75)
        - Zone B: Benign errors (would not lead to inappropriate treatment)
        - Zone C: Overcorrection errors (unnecessary treatment)
        - Zone D: Dangerous failures to detect (missed treatment)
        - Zone E: Erroneous treatment (opposite treatment than needed)
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            Dictionary with percentage of points in each zone
        """
        if CLARKE_EGA_AVAILABLE:
            # Use the new implementation
            ega = ClarkeEGA()
            results = ega.analyze(y_true, y_pred)
            
            # Convert to the expected format
            zones = {}
            for i, zone_letter in enumerate(['A', 'B', 'C', 'D', 'E']):
                zones[zone_letter] = results['percentage'][i]
            
            return zones
        else:
           raise NotImplementedError("Clarke Error Grid Analysis not available")

    @staticmethod
    def parkes_error_grid_analysis(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Parkes Error Grid Analysis for Type 1 diabetes patients.
        
        Uses the new ParkesEGA implementation for more accurate zone classification.
        
        Based on 'Technical Aspects of the Parkes Error Grid':
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3876371/
        
        Zones:
        - Zone A: Clinically accurate (≤20% or ±15 mg/dL for values <75)
        - Zone B: Benign errors with little/no clinical effect
        - Zone C: Overcorrection errors (could lead to unnecessary treatment)
        - Zone D: Dangerous failure to detect/treat
        - Zone E: Erroneous treatment (opposite treatment than needed)
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            Dictionary with percentage of points in each zone
        """
        if PARKES_EGA_AVAILABLE:
            # Use the new implementation
            ega = ParkesEGA(units_mg_dl=True)
            results = ega.analyze(y_true, y_pred)
            
            # Convert to the expected format (exclude 'OOR' zone)
            zones = {}
            for zone_letter in ['A', 'B', 'C', 'D', 'E']:
                zones[zone_letter] = results['percentages'][zone_letter]
            
            return zones
        else:
           raise NotImplementedError("Parkes Error Grid Analysis not available")
   
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate all available metrics for blood glucose prediction evaluation.
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['rmse'] = BGMetrics.rmse(y_true, y_pred)
        metrics['mae'] = BGMetrics.mae(y_true, y_pred)
        metrics['mape'] = BGMetrics.mape(y_true, y_pred)
        metrics['mard'] = BGMetrics.mard(y_true, y_pred)
        
        # Time in range metrics
        metrics['time_in_range'] = BGMetrics.time_in_range(y_pred)
        metrics['time_below_range'] = BGMetrics.time_below_range(y_pred)
        metrics['time_above_range'] = BGMetrics.time_above_range(y_pred)
        
        # Error grid analyses
        metrics['clarke_zones'] = BGMetrics.clarke_error_grid_analysis(y_true, y_pred)
        metrics['parkes_zones'] = BGMetrics.parkes_error_grid_analysis(y_true, y_pred)
        
        return metrics

    #TODO:implement
    @staticmethod
    def continuous_glucose_error_grid_analysis(y_true: np.ndarray, 
                                             y_pred: np.ndarray) -> Optional[Dict[str, Union[float, int]]]:
        """TODO: Implement CG-EGA for continuous glucose monitoring."""
        raise NotImplementedError("Continuous Glucose Error Grid Analysis is not yet implemented.")
        return None
        

    #TODO: ADD CTCA Metric
    def clinical_trend_concurrence_analysis():
        """TODO
        https://github.com/IfDTUlm/CGM_Performance_Assessment/tree/main/Clinical%20Trend%20Concurrence%20Analysis
        """
        raise NotImplementedError("Clinical Trend Concurrence Analysis is not yet implemented.")
    
