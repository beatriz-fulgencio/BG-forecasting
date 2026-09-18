"""
Reporting and visualization utilities for evaluation results.

This module provides functions for:
- Generating standardized reports
- Creating comparison visualizations
- Statistical analysis of results
- Export to common formats (LaTeX, CSV, JSON)
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

try:
    import pandas as pd #type:ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def generate_evaluation_report(model: Any, 
                              test_data: Any, 
                              metrics: Dict[str, Union[float, Dict]],
                              prediction_horizon: int = 30,
                              patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report for a blood glucose forecasting model.
    
    Args:
        model: The forecasting model that was evaluated
        test_data: The test dataset used for evaluation
        metrics: Dictionary of evaluation metrics
        prediction_horizon: Prediction horizon in minutes
        patient_id: Optional patient identifier
    
    Returns:
        Dictionary containing the complete evaluation report
    """
    # Extract model information
    model_info = {
        'model_type': model.__class__.__name__,
        'model_parameters': getattr(model, 'get_params', lambda: {})()
    }
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Compile the report
    report = {
        'timestamp': timestamp,
        'model_info': model_info,
        'metrics': metrics,
        'prediction_horizon': prediction_horizon
    }
    
    # Add patient ID if provided
    if patient_id is not None:
        report['patient_id'] = patient_id
    
    # Add dataset summary if possible
    if hasattr(test_data, 'shape'):
        report['dataset_info'] = {
            'size': test_data.shape[0],
            'features': test_data.shape[1] if len(test_data.shape) > 1 else 1
        }
    
    return report


def export_metrics_to_csv(metrics: Dict[str, Union[float, Dict]], 
                         filepath: str,
                         model_name: Optional[str] = None,
                         append: bool = False) -> None:
    """
    Export evaluation metrics to a CSV file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        filepath: Path to save the CSV file
        model_name: Optional name of the model
        append: Whether to append to an existing file
    
    Returns:
        None
    
    Raises:
        ImportError: If pandas is not available
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required to export metrics to CSV")
    
    # Define the order of metrics: statistics, then Clarke EGA, then Parkes EGA
    metric_order = [
        # Statistical metrics
        'rmse', 'mae', 'mape', 'mse', 'r2', 'correlation',
        # Clarke EGA zones
        'clarke_zones_A', 'clarke_zones_B', 'clarke_zones_C', 'clarke_zones_D', 'clarke_zones_E',
        # Parkes EGA zones
        'parkes_zones_A', 'parkes_zones_B', 'parkes_zones_C', 'parkes_zones_D', 'parkes_zones_E',
    ]
    
    # Flatten nested dictionaries (e.g., clarke_zones)
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_metrics[f"{key}_{subkey}"] = subvalue
        else:
            flat_metrics[key] = value
    
    # Add model name if provided
    if model_name is not None:
        flat_metrics['model'] = model_name
    
    # Add timestamp
    flat_metrics['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create ordered dictionary based on defined order
    ordered_metrics = {}
    
    # Add metadata first (model and timestamp)
    if 'model' in flat_metrics:
        ordered_metrics['model'] = flat_metrics['model']
    if 'timestamp' in flat_metrics:
        ordered_metrics['timestamp'] = flat_metrics['timestamp']
    
    # Add metrics in order
    for metric_key in metric_order:
        if metric_key in flat_metrics:
            ordered_metrics[metric_key] = flat_metrics[metric_key]
    
    # Add any remaining metrics not in the order list
    for key, value in flat_metrics.items():
        if key not in ordered_metrics:
            ordered_metrics[key] = value
    
    # Create DataFrame
    metrics_df = pd.DataFrame([ordered_metrics])
    
    # Check if file exists and append mode is requested
    if append and os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    # Write to CSV
    metrics_df.to_csv(filepath, index=False)


def export_metrics_to_json(metrics: Dict[str, Union[float, Dict]],
                          filepath: str,
                          model_name: Optional[str] = None,
                          patient_id: Optional[str] = None,
                          append: bool = False) -> None:
    """
    Export evaluation metrics to a JSON file with ordered structure:
    1. Metadata (model, patient_id, timestamp)
    2. Statistical metrics (rmse, mae, mape, etc.)
    3. Clarke EGA zones
    4. Parkes EGA zones
    
    Args:
        metrics: Dictionary of evaluation metrics
        filepath: Path to save the JSON file
        model_name: Optional name of the model
        patient_id: Optional patient identifier
        append: Whether to append to an existing file
    
    Returns:
        None
    """
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        else:
            return obj
    
    # Create ordered metrics dictionary
    ordered_metrics = {}
    
    # 1. Add metadata first
    if model_name is not None:
        ordered_metrics['model'] = model_name
    if patient_id is not None:
        ordered_metrics['patient_id'] = patient_id
    ordered_metrics['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define metric categories
    stat_metrics = ['rmse', 'mae', 'mape', 'mse', 'r2', 'correlation']
    
    # 2. Add statistical metrics
    for key in stat_metrics:
        if key in metrics:
            ordered_metrics[key] = convert_numpy(metrics[key])
    
    # Add any other statistical metrics not in the predefined list
    for key, value in metrics.items():
        if key not in ['clarke_zones', 'parkes_zones'] and key not in ordered_metrics:
            if not isinstance(value, dict):
                ordered_metrics[key] = convert_numpy(value)
    
    # 3. Add Clarke EGA zones
    if 'clarke_zones' in metrics:
        ordered_metrics['clarke_zones'] = convert_numpy(metrics['clarke_zones'])
    
    # 4. Add Parkes EGA zones
    if 'parkes_zones' in metrics:
        ordered_metrics['parkes_zones'] = convert_numpy(metrics['parkes_zones'])
    
    metrics_json = ordered_metrics
    
    # Check if file exists and append mode is requested
    if append and os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            
            # If existing data is a dictionary, convert to a list
            if isinstance(existing_data, dict):
                existing_data = [existing_data]
            
            # Append new metrics
            existing_data.append(metrics_json)
            
            with open(filepath, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(f"Error appending to JSON file: {e}")
            # Fall back to overwrite
            with open(filepath, 'w') as f:
                json.dump(metrics_json, f, indent=2)
    else:
        with open(filepath, 'w') as f:
            json.dump(metrics_json, f, indent=2)


def generate_model_comparison_report(model_metrics: Dict[str, Dict[str, Union[float, Dict]]],
                                    output_format: str = 'dict') -> Union[Dict, str, pd.DataFrame]:
    """
    Generate a comparison report between multiple models.
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        output_format: Format of the output ('dict', 'markdown', 'dataframe')
    
    Returns:
        Comparison report in the specified format
    """
    if not model_metrics:
        return {} if output_format == 'dict' else pd.DataFrame() if output_format == 'dataframe' else ""
    
    # Extract common metrics across all models
    common_metrics = set()
    for model_name, metrics in model_metrics.items():
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_metrics[f"{key}_{subkey}"] = subvalue
            else:
                flat_metrics[key] = value
        
        if not common_metrics:
            common_metrics = set(flat_metrics.keys())
        else:
            common_metrics = common_metrics.intersection(set(flat_metrics.keys()))
    
    # Create comparison dict
    comparison = {metric: {} for metric in common_metrics}
    for model_name, metrics in model_metrics.items():
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_key = f"{key}_{subkey}"
                    if flat_key in common_metrics:
                        flat_metrics[flat_key] = subvalue
            else:
                if key in common_metrics:
                    flat_metrics[key] = value
        
        for metric in common_metrics:
            if metric in flat_metrics:
                comparison[metric][model_name] = flat_metrics[metric]
    
    # Convert to the requested output format
    if output_format == 'dict':
        return comparison
    
    elif output_format == 'dataframe':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for dataframe output format")
        
        # Convert to a DataFrame
        comparison_df = pd.DataFrame(comparison).transpose()
        comparison_df.index.name = 'Metric'
        return comparison_df
    
    elif output_format == 'markdown':
        # Create a markdown table
        model_names = list(model_metrics.keys())
        
        markdown =  "| Metric | " + " | ".join(model_names) + " |\n"
        markdown += "| ------ | " + " | ".join(["------" for _ in model_names]) + " |\n"
        
        for metric in sorted(comparison.keys()):
            row = f"| {metric} | "
            for model_name in model_names:
                if model_name in comparison[metric]:
                    value = comparison[metric][model_name]
                    # Format based on value type
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    row += f"{formatted_value} | "
                else:
                    row += "N/A | "
            markdown += row + "\n"
        
        return markdown
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def export_comparison_to_file(model_metrics: Dict[str, Dict[str, Union[float, Dict]]],
                             filepath: str,
                             output_format: str = None) -> None:
    """
    Export a model comparison report to a file.
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        filepath: Path to save the file
        output_format: Format of the output (auto-detected from file extension if None)
    
    Returns:
        None
    """
    # Auto-detect format from file extension if not specified
    if output_format is None:
        extension = os.path.splitext(filepath)[1].lower()
        if extension == '.csv':
            output_format = 'csv'
        elif extension == '.json':
            output_format = 'json'
        elif extension == '.md':
            output_format = 'markdown'
        else:
            raise ValueError(f"Could not determine output format from file extension: {extension}")
    
    # Generate the comparison report
    if output_format == 'csv':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV output")
        
        comparison_df = generate_model_comparison_report(model_metrics, output_format='dataframe')
        comparison_df.to_csv(filepath)
    
    elif output_format == 'json':
        comparison = generate_model_comparison_report(model_metrics, output_format='dict')
        
        # Add timestamp
        comparison['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    elif output_format == 'markdown':
        comparison_md = generate_model_comparison_report(model_metrics, output_format='markdown')
        
        with open(filepath, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(comparison_md)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def generate_patient_comparison_report(patient_metrics: Dict[str, Dict[str, Dict[str, Union[float, Dict]]]],
                                      output_format: str = 'dict') -> Union[Dict, str, pd.DataFrame]:
    """
    Generate a comparison report across multiple patients and models.
    
    Args:
        patient_metrics: Dictionary mapping patient IDs to model metrics
                        {patient_id: {model_name: {metric: value}}}
        output_format: Format of the output ('dict', 'markdown', 'dataframe')
    
    Returns:
        Comparison report in the specified format
    """
    # First, extract all unique models and metrics
    all_models = set()
    all_metrics = set()
    
    for patient_id, models in patient_metrics.items():
        all_models.update(models.keys())
        
        for model_name, metrics in models.items():
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        all_metrics.add(f"{key}_{subkey}")
                else:
                    all_metrics.add(key)
    
    # Now create the comparison structure
    if output_format == 'dict':
        # Format: {metric: {model: {patient: value}}}
        comparison = {}
        
        for metric in all_metrics:
            comparison[metric] = {}
            
            for model in all_models:
                comparison[metric][model] = {}
                
                for patient_id, models in patient_metrics.items():
                    if model in models:
                        # Extract the metric value (handle nested metrics)
                        if '_' in metric:
                            main_metric, sub_metric = metric.split('_', 1)
                            if main_metric in models[model] and isinstance(models[model][main_metric], dict) and sub_metric in models[model][main_metric]:
                                comparison[metric][model][patient_id] = models[model][main_metric][sub_metric]
                        else:
                            if metric in models[model]:
                                comparison[metric][model][patient_id] = models[model][metric]
        
        return comparison
    
    elif output_format == 'dataframe':
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for dataframe output format")
        
        # Create a DataFrame with MultiIndex
        data = []
        indices = []
        
        for metric in sorted(all_metrics):
            for model in sorted(all_models):
                row = []
                for patient_id in sorted(patient_metrics.keys()):
                    # Extract the metric value (handle nested metrics)
                    value = None
                    if patient_id in patient_metrics and model in patient_metrics[patient_id]:
                        if '_' in metric:
                            main_metric, sub_metric = metric.split('_', 1)
                            if main_metric in patient_metrics[patient_id][model] and isinstance(patient_metrics[patient_id][model][main_metric], dict) and sub_metric in patient_metrics[patient_id][model][main_metric]:
                                value = patient_metrics[patient_id][model][main_metric][sub_metric]
                        else:
                            if metric in patient_metrics[patient_id][model]:
                                value = patient_metrics[patient_id][model][metric]
                    
                    row.append(value)
                
                data.append(row)
                indices.append((metric, model))
        
        index = pd.MultiIndex.from_tuples(indices, names=['Metric', 'Model'])
        columns = sorted(patient_metrics.keys())
        
        return pd.DataFrame(data, index=index, columns=columns)
    
    elif output_format == 'markdown':
        # Generate markdown tables for each metric
        markdown = ""
        
        for metric in sorted(all_metrics):
            markdown += f"## Metric: {metric}\n\n"
            
            # Create the table header
            markdown += "| Model | " + " | ".join(sorted(patient_metrics.keys())) + " |\n"
            markdown += "| ----- | " + " | ".join(["-----" for _ in patient_metrics]) + " |\n"
            
            for model in sorted(all_models):
                row = f"| {model} | "
                
                for patient_id in sorted(patient_metrics.keys()):
                    # Extract the metric value (handle nested metrics)
                    value = "N/A"
                    if patient_id in patient_metrics and model in patient_metrics[patient_id]:
                        if '_' in metric:
                            main_metric, sub_metric = metric.split('_', 1)
                            if main_metric in patient_metrics[patient_id][model] and isinstance(patient_metrics[patient_id][model][main_metric], dict) and sub_metric in patient_metrics[patient_id][model][main_metric]:
                                raw_value = patient_metrics[patient_id][model][main_metric][sub_metric]
                                value = f"{raw_value:.4f}" if isinstance(raw_value, float) else str(raw_value)
                        else:
                            if metric in patient_metrics[patient_id][model]:
                                raw_value = patient_metrics[patient_id][model][metric]
                                value = f"{raw_value:.4f}" if isinstance(raw_value, float) else str(raw_value)
                    
                    row += f"{value} | "
                
                markdown += row + "\n"
            
            markdown += "\n\n"
        
        return markdown
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def load_metrics_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load evaluation metrics from a JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary containing the loaded metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_patient_metrics(patient_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple patients.
    
    Args:
        patient_results: Dictionary mapping patient IDs to their metrics
    
    Returns:
        Dictionary containing the aggregated metrics
    """
    # Initialize the aggregated metrics
    aggregated = {}
    
    # Count the number of patients
    num_patients = len(patient_results)
    
    # Iterate through all patient results
    for patient_id, metrics in patient_results.items():
        for metric_name, metric_value in metrics.items():
            # Skip non-numeric metrics
            if isinstance(metric_value, (str, list, dict)):
                # Handle special case for error grid zones
                if metric_name in ['clarke_zones', 'parkes_zones'] and isinstance(metric_value, dict):
                    if metric_name not in aggregated:
                        aggregated[metric_name] = {zone: 0.0 for zone in metric_value.keys()}
                    
                    for zone, percentage in metric_value.items():
                        aggregated[metric_name][zone] += percentage / num_patients
                continue
            
            # Initialize the metric in the aggregated dict if it doesn't exist
            if metric_name not in aggregated:
                aggregated[metric_name] = 0.0
            
            # Add the metric value (to calculate the average later)
            aggregated[metric_name] += metric_value / num_patients
    
    return aggregated


def generate_latex_report(model_metrics: Dict[str, Dict[str, Union[float, Dict]]],
                         y_true_dict: Dict[str, np.ndarray],
                         y_pred_dict: Dict[str, np.ndarray],
                         output_dir: str,
                         report_title: str = "Blood Glucose Forecasting Model Comparison Report",
                         author: str = "BG-Forecasting System",
                         include_visualizations: bool = True) -> str:
    """
    Generate a comprehensive LaTeX report with model comparison and visualizations.
    
    This function creates a complete LaTeX document including:
    - Model comparison tables with statistical metrics
    - Clarke and Parkes Error Grid Analysis results
    - Visualizations from the dashboard (if requested)
    - Summary and recommendations
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
                      {model_name: {metric: value}}
        y_true_dict: Dictionary mapping model names to true values
                    {model_name: array of true values}
        y_pred_dict: Dictionary mapping model names to predicted values
                    {model_name: array of predicted values}
        output_dir: Directory to save the LaTeX file and figures
        report_title: Title of the report
        author: Author name
        include_visualizations: Whether to include dashboard visualizations
    
    Returns:
        Path to the generated LaTeX file
    
    Raises:
        ImportError: If required visualization modules are not available
    """
    if include_visualizations and not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for visualization generation")
    
    # Import visualization functions
    if include_visualizations:
        try:
            from .visualisation import (
                create_prediction_dashboard,
                plot_clarke_analysis,
                plot_parkes_analysis
            )
        except ImportError:
            raise ImportError("Could not import visualization functions from benchmark.evaluation.visualisation")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Start building the LaTeX document
    latex_content = []
    
    # Document preamble
    latex_content.append(r"\documentclass[11pt,a4paper]{article}")
    latex_content.append(r"\usepackage[utf8]{inputenc}")
    latex_content.append(r"\usepackage{graphicx}")
    latex_content.append(r"\usepackage{booktabs}")
    latex_content.append(r"\usepackage{float}")
    latex_content.append(r"\usepackage{geometry}")
    latex_content.append(r"\usepackage{hyperref}")
    latex_content.append(r"\usepackage{longtable}")
    latex_content.append(r"\usepackage{array}")
    latex_content.append(r"\geometry{margin=1in}")
    latex_content.append(r"")
    latex_content.append(r"\title{" + report_title + r"}")
    latex_content.append(r"\author{" + author + r"}")
    latex_content.append(r"\date{" + timestamp + r"}")
    latex_content.append(r"")
    latex_content.append(r"\begin{document}")
    latex_content.append(r"\maketitle")
    latex_content.append(r"\tableofcontents")
    latex_content.append(r"\newpage")
    latex_content.append(r"")
    
    # Section 1: Executive Summary
    latex_content.append(r"\section{Executive Summary}")
    latex_content.append(r"This report presents a comprehensive comparison of " + 
                        str(len(model_metrics)) + r" blood glucose forecasting models. ")
    latex_content.append(r"The evaluation includes statistical metrics (RMSE, MAE, MAPE), ")
    latex_content.append(r"Clarke Error Grid Analysis, and Parkes Error Grid Analysis.")
    latex_content.append(r"")
    
    # Section 2: Model Comparison - Statistical Metrics
    latex_content.append(r"\section{Model Comparison}")
    latex_content.append(r"\subsection{Statistical Metrics}")
    latex_content.append(r"")
    latex_content.append(r"Table~\ref{tab:statistical_metrics} presents the statistical performance metrics for all models.")
    latex_content.append(r"")
    latex_content.append(r"\begin{table}[H]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{Statistical Performance Metrics}")
    latex_content.append(r"\label{tab:statistical_metrics}")
    
    # Build the statistical metrics table
    model_names = sorted(model_metrics.keys())
    stat_metrics_order = ['rmse', 'mae', 'mape', 'mse', 'r2', 'correlation']
    
    # Filter only metrics that exist
    available_stat_metrics = []
    for metric in stat_metrics_order:
        if any(metric in model_metrics[model] for model in model_names):
            available_stat_metrics.append(metric)
    
    # Create table header
    latex_content.append(r"\begin{tabular}{l" + "c" * len(model_names) + "}")
    latex_content.append(r"\toprule")
    latex_content.append(r"\textbf{Metric} & " + " & ".join([r"\textbf{" + model + r"}" for model in model_names]) + r" \\")
    latex_content.append(r"\midrule")
    
    # Add table rows
    for metric in available_stat_metrics:
        metric_display = metric.upper()
        row = metric_display + " & "
        values = []
        for model in model_names:
            if metric in model_metrics[model]:
                value = model_metrics[model][metric]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            else:
                values.append("N/A")
        row += " & ".join(values) + r" \\"
        latex_content.append(row)
    
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    latex_content.append(r"")
    
    # Section 3: Clarke Error Grid Analysis
    latex_content.append(r"\subsection{Clarke Error Grid Analysis}")
    latex_content.append(r"")
    latex_content.append(r"Clarke Error Grid Analysis categorizes glucose predictions into five zones (A-E) ")
    latex_content.append(r"based on their clinical significance:")
    latex_content.append(r"\begin{itemize}")
    latex_content.append(r"  \item \textbf{Zone A}: Clinically accurate (within 20\% or both $<$70 mg/dL)")
    latex_content.append(r"  \item \textbf{Zone B}: Benign errors (no clinical impact)")
    latex_content.append(r"  \item \textbf{Zone C}: Overcorrection errors")
    latex_content.append(r"  \item \textbf{Zone D}: Dangerous failure to detect")
    latex_content.append(r"  \item \textbf{Zone E}: Erroneous treatment")
    latex_content.append(r"\end{itemize}")
    latex_content.append(r"")
    latex_content.append(r"Table~\ref{tab:clarke_ega} shows the percentage of predictions in each zone.")
    latex_content.append(r"")
    latex_content.append(r"\begin{table}[H]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{Clarke Error Grid Analysis Results (\%)}")
    latex_content.append(r"\label{tab:clarke_ega}")
    latex_content.append(r"\begin{tabular}{l" + "c" * len(model_names) + "}")
    latex_content.append(r"\toprule")
    latex_content.append(r"\textbf{Zone} & " + " & ".join([r"\textbf{" + model + r"}" for model in model_names]) + r" \\")
    latex_content.append(r"\midrule")
    
    # Add Clarke zone rows
    for zone in ['A', 'B', 'C', 'D', 'E']:
        row = f"Zone {zone} & "
        values = []
        for model in model_names:
            if 'clarke_zones' in model_metrics[model] and zone in model_metrics[model]['clarke_zones']:
                value = model_metrics[model]['clarke_zones'][zone]
                values.append(f"{value:.2f}")
            else:
                values.append("N/A")
        row += " & ".join(values) + r" \\"
        latex_content.append(row)
    
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    latex_content.append(r"")
    
    # Section 4: Parkes Error Grid Analysis
    latex_content.append(r"\subsection{Parkes (Consensus) Error Grid Analysis}")
    latex_content.append(r"")
    latex_content.append(r"Parkes Error Grid Analysis uses similar zones but with different boundaries ")
    latex_content.append(r"based on clinical consensus for Type 1 diabetes.")
    latex_content.append(r"")
    latex_content.append(r"Table~\ref{tab:parkes_ega} shows the percentage of predictions in each zone.")
    latex_content.append(r"")
    latex_content.append(r"\begin{table}[H]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{Parkes Error Grid Analysis Results (\%)}")
    latex_content.append(r"\label{tab:parkes_ega}")
    latex_content.append(r"\begin{tabular}{l" + "c" * len(model_names) + "}")
    latex_content.append(r"\toprule")
    latex_content.append(r"\textbf{Zone} & " + " & ".join([r"\textbf{" + model + r"}" for model in model_names]) + r" \\")
    latex_content.append(r"\midrule")
    
    # Add Parkes zone rows
    for zone in ['A', 'B', 'C', 'D', 'E']:
        row = f"Zone {zone} & "
        values = []
        for model in model_names:
            if 'parkes_zones' in model_metrics[model] and zone in model_metrics[model]['parkes_zones']:
                value = model_metrics[model]['parkes_zones'][zone]
                values.append(f"{value:.2f}")
            else:
                values.append("N/A")
        row += " & ".join(values) + r" \\"
        latex_content.append(row)
    
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    latex_content.append(r"")
    
    # Section 5: Visualizations
    if include_visualizations:
        latex_content.append(r"\section{Visualizations}")
        latex_content.append(r"")
        latex_content.append(r"This section presents comprehensive visualizations for each model, ")
        latex_content.append(r"including prediction plots, error distributions, and Error Grid Analyses.")
        latex_content.append(r"")
        
        for model_name in model_names:
            if model_name in y_true_dict and model_name in y_pred_dict:
                latex_content.append(r"\subsection{" + model_name.replace('_', r'\_') + r"}")
                latex_content.append(r"")
                
                # Generate dashboard visualization
                dashboard_filename = f"dashboard_{model_name}.png"
                dashboard_path = os.path.join(output_dir, dashboard_filename)
                
                try:
                    fig = create_prediction_dashboard(
                        y_true=y_true_dict[model_name],
                        y_pred=y_pred_dict[model_name],
                        metrics=model_metrics[model_name],save_path=dashboard_path
                    )
                    if fig is not None:
                        plt.close(fig)
                    
                    latex_content.append(r"\begin{figure}[H]")
                    latex_content.append(r"\centering")
                    latex_content.append(r"\includegraphics[width=\textwidth]{" + dashboard_filename + r"}")
                    latex_content.append(r"\caption{Comprehensive dashboard for " + model_name.replace('_', r'\_') + r"}")
                    latex_content.append(r"\label{fig:dashboard_" + model_name.replace('_', '_') + r"}")
                    latex_content.append(r"\end{figure}")
                    latex_content.append(r"")
                except Exception as e:
                    latex_content.append(r"Error generating dashboard: " + str(e).replace('_', r'\_'))
                    latex_content.append(r"")
    
    # Section 6: Conclusions and Recommendations
    latex_content.append(r"\section{Conclusions and Recommendations}")
    latex_content.append(r"")
    
    # Find the best model based on RMSE and Clarke Zone A
    best_model = None
    best_rmse = float('inf')
    best_clarke_a = 0.0
    
    for model_name in model_names:
        if 'rmse' in model_metrics[model_name]:
            rmse = model_metrics[model_name]['rmse']
            clarke_a = model_metrics[model_name].get('clarke_zones', {}).get('A', 0.0)
            
            if rmse < best_rmse or (rmse == best_rmse and clarke_a > best_clarke_a):
                best_rmse = rmse
                best_clarke_a = clarke_a
                best_model = model_name
    
    if best_model:
        latex_content.append(r"Based on the comprehensive evaluation, \textbf{" + 
                           best_model.replace('_', r'\_') + 
                           r"} shows the best overall performance with:")
        latex_content.append(r"\begin{itemize}")
        latex_content.append(r"  \item RMSE: " + f"{best_rmse:.4f}" + r" mg/dL")
        latex_content.append(r"  \item Clarke Zone A: " + f"{best_clarke_a:.2f}" + r"\%")
        latex_content.append(r"\end{itemize}")
        latex_content.append(r"")
    
    latex_content.append(r"For clinical deployment, models with $>$95\% in Clarke Zone A ")
    latex_content.append(r"and RMSE $<$15 mg/dL are recommended.")
    latex_content.append(r"")
    
    # End document
    latex_content.append(r"\end{document}")
    
    # Write the LaTeX file
    latex_filename = "model_comparison_report.tex"
    latex_filepath = os.path.join(output_dir, latex_filename)
    
    with open(latex_filepath, 'w') as f:
        f.write('\n'.join(latex_content))
        
    return latex_filepath