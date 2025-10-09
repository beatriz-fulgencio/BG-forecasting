import numpy as np
import warnings
from typing import List, Optional, Dict, Union

from .metrics import BGMetrics

class BGEvaluator:
    """
    Main evaluator class that integrates with base models.
    
    This class provides a unified interface for computing all blood glucose
    forecasting metrics and integrates seamlessly with the base model framework.
    """
    
    def __init__(self, 
                 default_metrics: Optional[List[str]] = None):
        """
        Initialize the evaluator.
        
        Args:
            default_metrics: Default list of metrics to compute
        """
        self.default_metrics = default_metrics or [
            'mae', 'rmse', 'mape', 'mard', 'tir', 'clarke'
        ]
        self.metrics_calculator = BGMetrics()
    
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       uncertainty: Optional[np.ndarray] = None,
                       metrics: Optional[List[str]] = None) -> Dict[str, Union[float, Dict]]:
        """
        Compute specified metrics for glucose prediction evaluation.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values  
            uncertainty: Optional prediction uncertainty estimates
            metrics: List of metrics to compute (uses default if None)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = metrics or self.default_metrics
        results = {}
        
        for metric in metrics:
            metric_lower = metric.lower() # Normalize to lower case for case-insensitive comparison 
            
            if metric_lower == 'mae':
                results['mae'] = self.metrics_calculator.mae(y_true, y_pred)
            elif metric_lower == 'rmse':
                results['rmse'] = self.metrics_calculator.rmse(y_true, y_pred)
            elif metric_lower == 'mape':
                results['mape'] = self.metrics_calculator.mape(y_true, y_pred)
            elif metric_lower == 'mard':
                results['mard'] = self.metrics_calculator.mard(y_true, y_pred)
            elif metric_lower in ['tir', 'time_in_range']:
                results['tir'] = self.metrics_calculator.comparing_time_in_range(y_true, y_pred)
            elif metric_lower in ['clarke', 'clarke_ega']:
                results['clarke_zones'] = self.metrics_calculator.clarke_error_grid_analysis(y_true, y_pred)
            elif metric_lower in ['parkes', 'parkes_ega']:
                results['parkes_zones'] = self.metrics_calculator.parkes_error_grid_analysis(y_true, y_pred)
            #TODO : implement
            elif metric_lower in ['cg_ega', 'continuous_glucose_ega']:
                results['cg_ega'] = self.metrics_calculator.continuous_glucose_error_grid_analysis(y_true, y_pred)
            elif metric_lower in ['ctca', 'clinical_trend_concurrence_analysis']:
                results['ctca'] = self.metrics_calculator.clinical_trend_concurrence_analysis(y_true, y_pred)
            else:
                warnings.warn(f"Unknown metric: {metric}")
        
        return results


# direct use
def evaluate_bg_prediction(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          metrics: Optional[List[str]] = None) -> Dict[str, Union[float, Dict]]:
    """
    Convenience function for evaluating blood glucose predictions.
    
    Args:
        y_true: True glucose values
        y_pred: Predicted glucose values
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    evaluator = BGEvaluator()
    return evaluator.compute_metrics(y_true, y_pred, metrics=metrics)