"""
Blood Glucose Forecasting Benchmark

A reproducible benchmark framework for comparing blood glucose prediction models.
This package provides standardized data processing, model interfaces, evaluation
metrics, and experiment management for fair comparison of different approaches.

Usage:
    from benchmark import Experiment
    
    # Load configuration
    config = load_config('configs/my_experiment.yaml')
    
    # Run experiment
    experiment = Experiment(config)
    results = experiment.run()
    
    # Evaluate results
    metrics = experiment.evaluate(results)
"""

__version__ = "0.1.0"
__author__ = "Blood Glucose Forecasting Research Group"

# TODO: Implement main benchmark interface
# TODO: Add experiment management functions
# TODO: Implement result comparison utilities