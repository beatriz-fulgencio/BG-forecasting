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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentTracker:
    """
    Tracks experiment parameters, progress, and results for reproducibility.
    """
    
    def __init__(self, results_dir: Path, config: Dict[str, Any] = None):
        """
        Initialize experiment tracker.
        
        Args:
            results_dir: Directory to save tracking information
            config: Experiment configuration
        """
        self.results_dir = Path(results_dir)
        self.config = config or {}
        self.experiment_id = self._generate_experiment_id()
        self.experiment_dir = self.results_dir / f"experiment_{self.experiment_id}"
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking data
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
        
        # Save initial tracking data
        self._save_tracking_data()
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID based on timestamp and config."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        return f"{timestamp}_{config_hash}"
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility."""
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': None,
            'git_commit': None,
            'git_branch': None,
            'working_directory': str(Path.cwd())
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
        
        return env_info
    
    def start_experiment(self):
        """Mark experiment start."""
        self.tracking_data['start_time'] = datetime.now().isoformat()
        self.tracking_data['status'] = 'running'
        self._save_tracking_data()
        
        print(f"🔬 Experiment started: {self.experiment_id}")
        print(f"📁 Tracking directory: {self.experiment_dir}")
    
    def end_experiment(self, final_results: Dict[str, Any]):
        """Mark experiment completion."""
        self.tracking_data['end_time'] = datetime.now().isoformat()
        self.tracking_data['status'] = 'completed'
        self.tracking_data['final_results'] = final_results
        
        # Calculate experiment duration
        if self.tracking_data['start_time']:
            start_time = datetime.fromisoformat(self.tracking_data['start_time'])
            end_time = datetime.fromisoformat(self.tracking_data['end_time'])
            duration = (end_time - start_time).total_seconds()
            self.tracking_data['duration_seconds'] = duration
        
        self._save_tracking_data()
        self._generate_experiment_summary()
        
        print(f"✅ Experiment completed: {self.experiment_id}")
        if 'duration_seconds' in self.tracking_data:
            print(f"⏱️  Total duration: {self.tracking_data['duration_seconds']:.1f} seconds")
    
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
        print(f"🏋️  Training started: {model_key}")
    
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
        print(f"✅ Training completed: {model_key}")
    
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
        tracking_file = self.experiment_dir / "tracking.json"
        
        # Convert to JSON-serializable format
        serializable_data = self._make_json_serializable(self.tracking_data)
        
        with open(tracking_file, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
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
                    mae = results.get('mae', 'N/A')
                    mard = results.get('mard', 'N/A')
                    tir = results.get('tir', 'N/A')
                    
                    summary_lines.extend([
                        f"### {model_key}",
                        f"- **MAE**: {mae:.3f} mg/dL" if isinstance(mae, (int, float)) else f"- **MAE**: {mae}",
                        f"- **MARD**: {mard:.2f}%" if isinstance(mard, (int, float)) else f"- **MARD**: {mard}",
                        f"- **TIR**: {tir:.1f}%" if isinstance(tir, (int, float)) else f"- **TIR**: {tir}",
                        ""
                    ])
        
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
        summary_file = self.experiment_dir / "summary.md"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"📋 Experiment summary saved: {summary_file}")
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get current experiment information."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_dir': str(self.experiment_dir),
            'status': self.tracking_data['status'],
            'start_time': self.tracking_data['start_time'],
            'config': self.config
        }


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