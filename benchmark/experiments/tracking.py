"""
Experiment tracking utilities.

This module provides functions for:
- Logging experiment parameters and results
- Version control integration
- Reproducibility tracking
- Result persistence
"""

import json
import time
import hashlib
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentTracker:
    """
    Tracks experiment parameters, progress, and results for reproducibility.
    """
    
    def __init__(self, results_dir: Path, config: Dict[str, Any] = None,
                 experiment_dir: Optional[Path] = None):
        """
        Initialize experiment tracker.
        
        Args:
            results_dir: Directory to save tracking information
            config: Experiment configuration
            experiment_dir: Exact run directory. When omitted, a timestamped
                directory is created under ``results_dir``.
        """
        self.results_dir = Path(results_dir)
        self.config = config or {}
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = (
            Path(experiment_dir)
            if experiment_dir is not None
            else self.results_dir / f"experiment_{self.experiment_id}"
        )
        
        self.tracking_data = {
            'experiment_id': self.experiment_id,
            'start_time': None,
            'end_time': None,
            'config': self.config,
            'environment': self._capture_environment(),
            'data_params': {},
            'models': {},
            'results': {},
            'status': 'initialized'
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on timestamp and config."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"{timestamp}_{config_hash}_{uuid.uuid4().hex[:6]}"
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility."""
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': None,
            'git_commit': None,
            'git_branch': None,
            'working_directory': str(Path.cwd()),
            'device': None
        }
        
        try:
            # Get Python version
            import sys
            env_info['python_version'] = sys.version
        except:
            pass
        
        try:
            # Get Git information
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_commit'] = git_commit
            
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            env_info['git_branch'] = git_branch    
        except:
            pass
        
        try:
            import torch
            
            if torch.cuda.is_available():
                env_info['device'] = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                env_info['device'] = "Apple Silicon GPU (MPS)"
            else:
                env_info['device'] = "CPU"
        except:
            pass
        
        return env_info
    
    def _ensure_experiment_dir(self):
        """Create the run directory the first time something needs it."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def start_experiment(self):
        """Mark experiment start."""
        self.tracking_data['start_time'] = datetime.now().isoformat()
        self.tracking_data['status'] = 'running'
        self._save_tracking_data()
        
        print(f"[INFO] Experiment started: {self.experiment_id}")
        print(f"[INFO] Tracking directory: {self.experiment_dir}")
    
    def end_experiment(self, final_results: Dict[str, Any], status: str = 'completed'):
        """Mark experiment completion.

        Args:
            final_results: Results to persist with the run.
            status: Terminal status. Pass 'completed_with_failures' when some
                subruns failed, so a partial result is never mistaken for a
                whole one by anything reading tracking.json.
        """
        self.tracking_data['final_results'] = final_results
        self._close_out(status)
        self._save_tracking_data()
        self._generate_experiment_summary()
        
        print(f"Experiment {self.tracking_data['status']}: {self.experiment_id}")
        if 'duration_seconds' in self.tracking_data:
            print(f"  Total duration: {self.tracking_data['duration_seconds']:.1f} seconds")
    
    def mark_interrupted(self, reason: str = None):
        """Close out a run that was killed rather than finished."""
        self.tracking_data['interrupted_by'] = reason or 'interrupted before completion'
        self._close_out('interrupted')
        self._save_tracking_data()

        print(f"[WARNING] Experiment interrupted: {self.experiment_id}"
              f" ({self.tracking_data['interrupted_by']})")

    def _close_out(self, status: str):
        """Stamp the terminal status, end time and duration onto the record."""
        self.tracking_data['end_time'] = datetime.now().isoformat()
        self.tracking_data['status'] = status

        if self.tracking_data['start_time']:
            start_time = datetime.fromisoformat(self.tracking_data['start_time'])
            end_time = datetime.fromisoformat(self.tracking_data['end_time'])
            self.tracking_data['duration_seconds'] = (end_time - start_time).total_seconds()

    def log_data_params(self, params: Dict[str, Any]):
        """Log data loading parameters."""
        self.tracking_data['data_params'] = params
        self._save_tracking_data()
    
    def log_training_start(self, model_name: str, patient_id: int, hyperparameters: Dict[str, Any]):
        """Log start of model training."""
        model_key = f"{model_name}_patient_{patient_id}"
        
        self.tracking_data['models'][model_key] = {
            'model_name': model_name,
            'patient_id': patient_id,
            'hyperparameters': hyperparameters,
            'training_start': datetime.now().isoformat(),
            'training_end': None,
            'training_history': None,
            'evaluation_results': None
        }
        
        self._save_tracking_data()
        print(f"[INFO] Training started: {model_key}")
    
    def log_training_completion(self, model_name: str, patient_id: int, history: Dict[str, Any]):
        """Log completion of model training."""
        model_key = f"{model_name}_patient_{patient_id}"
        
        if model_key in self.tracking_data['models']:
            self.tracking_data['models'][model_key]['training_end'] = datetime.now().isoformat()
            self.tracking_data['models'][model_key]['training_history'] = history
            
            # Calculate training duration
            start_time = datetime.fromisoformat(self.tracking_data['models'][model_key]['training_start'])
            end_time = datetime.fromisoformat(self.tracking_data['models'][model_key]['training_end'])
            duration = (end_time - start_time).total_seconds()
            self.tracking_data['models'][model_key]['training_duration_seconds'] = duration
        
        self._save_tracking_data()
        print(f"  Training completed: {model_key}")
    
    def log_evaluation_results(self, model_name: str, patient_id: int, results: Dict[str, Any]):
        """Log model evaluation results."""
        model_key = f"{model_name}_patient_{patient_id}"
        
        if model_key in self.tracking_data['models']:
            self.tracking_data['models'][model_key]['evaluation_results'] = results
        
        self._save_tracking_data()
    
    def log_error(self, error_message: str, context: str = None):
        """Log error information."""
        if 'errors' not in self.tracking_data:
            self.tracking_data['errors'] = []
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'message': error_message,
            'context': context
        }
        
        self.tracking_data['errors'].append(error_info)
        self.tracking_data['status'] = 'error'
        self._save_tracking_data()
    
    def _save_tracking_data(self):
        """Save tracking data to file."""
        self._ensure_experiment_dir()
        tracking_file = self.experiment_dir / "tracking.json"
        
        # Convert to JSON-serializable format
        serializable_data = self._make_json_serializable(self.tracking_data)
        
        with open(tracking_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        # Preserve native JSON scalar types. In particular, Python integers
        # implement ``__float__`` and were previously written as 540.0.
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__float__'):  # numpy floats
            return float(obj)
        elif hasattr(obj, '__int__'):  # numpy ints
            return int(obj)
        else:
            return obj
    
    def _generate_experiment_summary(self):
        """Generate human-readable experiment summary."""
        summary_lines = [
            f"# Experiment Summary: {self.experiment_id}",
            f"",
            f"**Status**: {self.tracking_data['status']}",
            f"**Start Time**: {self.tracking_data['start_time']}",
            f"**End Time**: {self.tracking_data['end_time']}",
            f""
        ]
        
        if 'duration_seconds' in self.tracking_data:
            duration_min = self.tracking_data['duration_seconds'] / 60
            summary_lines.append(f"**Duration**: {duration_min:.1f} minutes")
            summary_lines.append("")
        
        # Add environment info
        if self.tracking_data['environment']:
            env = self.tracking_data['environment']
            summary_lines.extend([
                "## Environment",
                f"- **Git Branch**: {env.get('git_branch', 'N/A')}",
                f"- **Git Commit**: {env.get('git_commit', 'N/A')[:8] if env.get('git_commit') else 'N/A'}",
                f"- **Working Directory**: {env.get('working_directory', 'N/A')}",
                ""
            ])
        
        # Add data parameters
        if self.tracking_data['data_params']:
            summary_lines.extend([
                "## Data Configuration",
                f"- **Patients**: {self.tracking_data['data_params'].get('patient_ids', 'N/A')}",
                f"- **Dataset Version**: {self.tracking_data['data_params'].get('version', 'N/A')}",
                f"- **Sequence Length**: {self.tracking_data['data_params'].get('sequence_length', 'N/A')}",
                f"- **Prediction Horizon**: {self.tracking_data['data_params'].get('prediction_horizon', 'N/A')}",
                f"- **Batch Size**: {self.tracking_data['data_params'].get('batch_size', 'N/A')}",
                ""
            ])
        
        # Add model results summary
        if self.tracking_data['models']:
            summary_lines.extend([
                "## Model Results",
                ""
            ])
            
            for model_key, model_info in self.tracking_data['models'].items():
                if model_info.get('evaluation_results'):
                    results = model_info['evaluation_results']
                    summary_lines.append(f"### {model_key}")
                    summary_lines.append(
                        self._format_metric("MAE", results.get('mae'), "mg/dL"))
                    summary_lines.append(
                        self._format_metric("MARD", results.get('mard'), "%", decimals=2))
                    summary_lines.extend(self._format_range_deltas(results.get('tir')))
                    summary_lines.append("")
        
        # Add errors if any
        if 'errors' in self.tracking_data and self.tracking_data['errors']:
            summary_lines.extend([
                "## Errors",
                ""
            ])
            for error in self.tracking_data['errors']:
                summary_lines.extend([
                    f"- **{error['timestamp']}**: {error['message']}",
                    f"  - Context: {error.get('context', 'N/A')}",
                    ""
                ])
        
        # Save summary
        self._ensure_experiment_dir()
        summary_file = self.experiment_dir / "summary.md"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"  Experiment summary saved: {summary_file}")
    
    @staticmethod
    def _as_number(value: Any) -> Optional[float]:
        """Fit a metric to float, or None when it is not a number."""
        if isinstance(value, bool) or isinstance(value, str) or value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _format_metric(cls, label: str, value: Any, unit: str,
                       decimals: int = 3) -> str:
        """Render one scalar metric, or say plainly that it is not there."""
        number = cls._as_number(value)
        if number is None:
            return f"- **{label}**: N/A"
        separator = "" if unit == "%" else " "
        return f"- **{label}**: {number:.{decimals}f}{separator}{unit}"

    _RANGE_BANDS = (
        ("time_in_range", "TIR"),
        ("time_below_range", "TBR"),
        ("time_above_range", "TAR"),
    )

    @classmethod
    def _format_range_deltas(cls, deltas: Any) -> list:
        """Render the glucose-range occupancy deltas as signed percentage point"""
        if not isinstance(deltas, dict):
            return ["- **Range occupancy (predicted - reference)**: N/A"]

        lines = ["- **Range occupancy (predicted - reference), percentage points**:"]
        for key, label in cls._RANGE_BANDS:
            number = cls._as_number(deltas.get(key))
            lines.append(f"  - **{label}**: N/A" if number is None
                         else f"  - **{label}**: {number:+.2f} pp")
        return lines

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get current experiment information."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_dir': str(self.experiment_dir),
            'status': self.tracking_data['status'],
            'start_time': self.tracking_data['start_time'],
            'config': self.config
        }

    def get_experiment_dir(self) -> Path:
        """Get the experiment directory, creating it if it is not there yet."""
        self._ensure_experiment_dir()
        return self.experiment_dir


def load_experiment(experiment_dir: str) -> Dict[str, Any]:
    """
    Load experiment tracking data from directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Experiment tracking data
    """
    tracking_file = Path(experiment_dir) / "tracking.json"
    
    if not tracking_file.exists():
        raise FileNotFoundError(f"Tracking file not found: {tracking_file}")
    
    with open(tracking_file, 'r') as f:
        return json.load(f)


def compare_experiments(experiment_dirs: list) -> Dict[str, Any]:
    """
    Compare multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
        
    Returns:
        Comparison results
    """
    experiments = []
    
    for exp_dir in experiment_dirs:
        try:
            exp_data = load_experiment(exp_dir)
            experiments.append(exp_data)
        except Exception as e:
            print(f"Warning: Could not load experiment from {exp_dir}: {e}")
    
    if not experiments:
        return {'error': 'No valid experiments found'}
    
    # Extract key metrics for comparison
    comparison = {
        'experiments': len(experiments),
        'experiment_ids': [exp['experiment_id'] for exp in experiments],
        'model_results': []
    }
    
    for exp in experiments:
        for model_key, model_info in exp.get('models', {}).items():
            if model_info.get('evaluation_results'):
                results = model_info['evaluation_results']
                comparison['model_results'].append({
                    'experiment_id': exp['experiment_id'],
                    'model': model_key,
                    'mae': results.get('mae'),
                    'mard': results.get('mard'),
                    'tir': results.get('tir')
                })
    
    return comparison
