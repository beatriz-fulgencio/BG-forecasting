"""
Standardized evaluation metrics for blood glucose forecasting.

This module implements domain-specific metrics including:
- Standard regression metrics (MAE, RMSE, MAPE)
- Clarke Error Grid Analysis (EGA)
- Parkes Error Grid Analysis (PEGA)
- Time in Range (TIR) metrics
- Continuous Glucose Error Grid Analysis (CG-EGA)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


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
    
    
    #TODO: Edit time in range to compare against reference
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
    def clarke_error_grid_analysis(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Clarke Error Grid Analysis for glucose prediction evaluation.
        
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
        zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        total_points = len(y_true)
        
        for true_val, pred_val in zip(y_true, y_pred):
            zone = BGMetrics._classify_clarke_zone(true_val, pred_val)
            zones[zone] += 1
        
        # Convert to percentages
        for zone in zones:
            zones[zone] = (zones[zone] / total_points) * 100
        
        return zones
    
    @staticmethod
    def _classify_clarke_zone(act: float, pred: float) -> str:
        """
        Classify a single point into Clarke Error Grid zone.
        
        Based on 'Evaluating clinical accuracy of systems for self-monitoring of blood glucose':
        https://care.diabetesjournals.org/content/10/5/622
        """
        # Zone A - Clinically accurate
        if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
            return 'A'
        
        # Zone E - Erroneous treatment (most dangerous)
        # Zone E - left upper: low glucose predicted as very high
        if act <= 70 and pred >= 180:
            return 'E'
        # Zone E - right lower: high glucose predicted as very low  
        if act >= 180 and pred <= 70:
            return 'E'
        
        # Zone D - Dangerous failure to detect/treat
        # Zone D - right: high glucose (≥240) predicted as moderate (70-180)
        if act >= 240 and 70 <= pred <= 180:
            return 'D'
        # Zone D - left: low glucose (≤70) predicted as moderate (70-180)
        if act <= 70 <= pred <= 180:
            return 'D'
        
        # Zone C - Overcorrection errors (unnecessary treatment)
        # Zone C - upper: moderate glucose predicted as very high
        if 70 <= act <= 290 and pred >= act + 110:
            return 'C'
        # Zone C - lower: moderate-high glucose predicted as low (leads to overcorrection)
        if 130 <= act <= 180 and pred <= (7/5) * act - 182:
            return 'C'
        
        # Zone B - Benign errors (everything else)
        # Zone B - upper: prediction higher than actual but not dangerous
        if act < pred:
            return 'B'
        # Zone B - lower: prediction lower than actual but not dangerous
        return 'B'
    
    @staticmethod
    def parkes_error_grid_analysis(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Parkes Error Grid Analysis for Type 1 diabetes patients.
        
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
        zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        total_points = len(y_true)
        
        for true_val, pred_val in zip(y_true, y_pred):
            zone = BGMetrics._classify_parkes_zone_t1(true_val, pred_val)
            zones[zone] += 1
        
        # Convert to percentages
        for zone in zones:
            zones[zone] = (zones[zone] / total_points) * 100
        
        return zones
    
    @staticmethod
    def _classify_parkes_zone_t1(act: float, pred: float) -> str:
        """Classify a single point into Parkes Error Grid zone for Type 1 diabetes."""
        
        def above_line(x_1, y_1, x_2, y_2, strict=False):
            """Check if prediction is above the line defined by two points."""
            if x_1 == x_2:
                return False
            y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
            return pred > y_line if strict else pred >= y_line

        def below_line(x_1, y_1, x_2, y_2, strict=False):
            """Check if prediction is below the line defined by two points."""
            return not above_line(x_1, y_1, x_2, y_2, not strict)
        
        # Zone E
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 'E'
        
        # Zone D - left upper
        if (pred > 100 and above_line(25, 100, 50, 125) and
                above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
            return 'D'
        
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 550, 150)):
            return 'D'
        
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 50, 80) and
                above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
            return 'C'
        
        # Zone C - right lower
        if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
            return 'C'
        
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 140, 170) and
                above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
            return 'B'
        
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 170, 145) and
                below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
            return 'B'
        
        # Zone A (default)
        return 'A'
    
    
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
    
