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
        BGEvaluator,
        evaluate_model
    )
    from .metrics import (
        BGMetrics
    )
    from .reporting import (
        generate_evaluation_report,
        export_metrics_to_csv,
        export_metrics_to_json,
        generate_model_comparison_report,
        generate_patient_comparison_report,
        aggregate_patient_metrics
    )
    from .visualisation import (
        plot_predictions,
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
        # Evaluator functions
        'evaluate_model',
        # Reporting functions
        'generate_evaluation_report',
        'export_metrics_to_csv',
        'export_metrics_to_json', 
        'generate_model_comparison_report',
        'generate_patient_comparison_report',
        'aggregate_patient_metrics',
        # Visualization functions
        'plot_predictions',
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