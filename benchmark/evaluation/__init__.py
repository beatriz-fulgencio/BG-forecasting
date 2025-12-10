"""
BG-Forecasting Evaluation Module

This module provides standardized evaluation metrics, protocols, and
reporting utilities for reproducible blood glucose forecasting model comparison.

Main Components:
- metrics: Standard and clinical glucose prediction metrics (RMSE, MAE, MAPE, etc.)
- evaluator: Framework for consistent model evaluation
- reporting: Report generation for model performance analysis
- visualisation: Visualization tools for prediction results and error analysis
- clarke_ega: Clarke Error Grid Analysis implementation
- parkes_ega: Parkes Error Grid Analysis implementation

Example Usage:
    from benchmark.evaluation import BGEvaluator, plot_predictions
    
    # Evaluate model predictions
    evaluator = BGEvaluator()
    metrics = evaluator.compute_metrics(y_true, y_pred)
    
    # Generate visual analysis
    plot_predictions(y_true, y_pred, uncertainty=None, 
                    title="30-min Glucose Prediction")
    
    # Generate full report
    report = generate_evaluation_report(model, test_data, metrics)
"""

# Import main classes and functions for easy access
try:
    from .evaluator import (
        BGEvaluator
    )
    from .metrics import (
        BGMetrics
    )
    from .reporting import (
        export_metrics_to_csv,
        export_metrics_to_json,
        generate_latex_report
    )
    from .visualisation import (
        plot_clarke_analysis,
        plot_parkes_analysis,
        create_prediction_dashboard
    )
    
    # Import EGA implementations
    try:
        from .clarke_ega import ClarkeEGA, plot_clarke_grid
        CLARKE_EGA_AVAILABLE = True
    except ImportError:
        CLARKE_EGA_AVAILABLE = False
    
    try:
        from .parkes_ega import ParkesEGA, plot_parkes_grid
        PARKES_EGA_AVAILABLE = True
    except ImportError:
        PARKES_EGA_AVAILABLE = False
    
    __all__ = [
        # Main classes
        'BGEvaluator',
        'BGMetrics',
        # Reporting functions
        'export_metrics_to_csv',
        'export_metrics_to_json', 
        'generate_latex_report',
        # Visualization functions
        'plot_clarke_analysis',
        'plot_parkes_analysis',
        'create_prediction_dashboard'
    ]
    
    # Add EGA classes if available
    if CLARKE_EGA_AVAILABLE:
        __all__.extend(['ClarkeEGA', 'plot_clarke_grid'])
    
    if PARKES_EGA_AVAILABLE:
        __all__.extend(['ParkesEGA', 'plot_parkes_grid'])
    
except ImportError as e:
    # If imports fail (e.g., missing dependencies), provide informative error
    import warnings
    warnings.warn(f"Could not import all evaluation modules: {e}. "
                 "Please ensure all dependencies are installed.")
    
    __all__ = []