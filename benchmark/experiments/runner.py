"""
Experiment runner for blood glucose forecasting benchmark.

This module provides the main interface for running benchmark experiments
with standardized configurations and reproducible results.
"""

import sys
import argparse
import json
import pandas as pd #type: ignore 
import numpy as np
import torch  #type: ignore 
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader  #type: ignore

from ..data.loaders import load_ohiot1dm_data
from ..data.preprocessors import preprocess_ohiot1dm_data
from ..data.torch_dataset import prepare_multi_patient_dataset, prepare_personal_data
from ..models.rnn import RNNBGModel, LSTMBGModel, GRUBGModel
from ..evaluation.evaluator import BGEvaluator
from ..evaluation.visualisation import (
    create_prediction_dashboard,
    plot_clarke_analysis,
    plot_parkes_analysis
)
from ..evaluation.reporting import (
    export_metrics_to_csv,
    generate_latex_report,
    generate_model_comparison_report,
    export_comparison_to_file,
    generate_patient_comparison_report
)
from .tracking import ExperimentTracker

GLUCOSE_PLAUSIBLE_RANGE_MG_DL = (20.0, 600.0)

class BGForecastingDataset:
    """
    Wrapper around OhioDataset to return (input, target) pairs for training.
    """
    def __init__(self, ohio_dataset):
        self.ohio_dataset = ohio_dataset
    
    def __len__(self):
        return len(self.ohio_dataset)
    
    def __getitem__(self, idx):
        sequence = self.ohio_dataset[idx]
        target = self.ohio_dataset.get_target(idx)
        return sequence, target


class BGConcatDataset:
    """
    Wrapper around ConcatDataset for transfer learning to return (input, target) pairs.
    """
    def __init__(self, concat_dataset):
        self.concat_dataset = concat_dataset
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
        # ConcatDataset handles the indexing internally
        dataset_idx = 0
        current_idx = idx
        
        # Find the right sub-dataset
        for i, dataset in enumerate(self.concat_dataset.datasets):
            if current_idx < len(dataset):
                dataset_idx = i
                break
            current_idx -= len(dataset)
        
        # Get sequence from the concat dataset
        sequence = self.concat_dataset[idx]
        
        # Get target from the appropriate sub-dataset
        target = self.concat_dataset.datasets[dataset_idx].get_target(current_idx)
        
        return sequence, target


class ExperimentRunner:
    """
    Main experiment runner for blood glucose forecasting benchmark.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        
        # Extract paths from config
        paths_config = config.get('paths', {})
        self.data_dir = paths_config.get('data_root', 'data')
        self.results_dir = Path(paths_config.get('results', 'results'))
    
        # Create directories
        self.results_dir.mkdir(exist_ok=True)

        # Initialize evaluator (new API doesn't use default_metrics)
        self.evaluator = BGEvaluator()
        
        # Initialize tracker
        self.tracker = ExperimentTracker(
            results_dir=self.results_dir,
            config=self.config
        )
        self.experiment_dir = self.tracker.get_experiment_dir()

        # Store results
        self.experiment_results = {}
    
    def load_and_prepare_data(self, 
                             patient_ids: list = [540, 544, 552, 567, 584, 596],
                             version: str = "2020",
                             sequence_length: int = 12,
                             prediction_horizon: int = 6,
                             unimodal: bool = False,
                             batch_size: int = 32):
        """
        Load and prepare data for all specified patients using either transfer learning or regular training.
        
        Args:
            patient_ids: List of patient IDs to load
            version: Dataset version ('2018' or '2020')
            sequence_length: Input sequence length (history)
            prediction_horizon: Prediction horizon (future steps)
            unimodal: Whether to use only glucose features
            batch_size: Batch size for DataLoaders
            
        Returns:
            Dictionary with patient data and DataLoaders
        """
        print(f"\n{'='*60}")
        print("LOADING AND PREPARING DATA")
        print(f"{'='*60}")
        
        # Check if transfer learning is enabled
        transfer_learning_enabled = self.config.get('training', {}).get('transfer_learning', {}).get('enabled', True)
        
        if transfer_learning_enabled:
            print("  Using TRANSFER LEARNING approach")
            return self._load_data_for_transfer_learning(patient_ids, version, sequence_length, prediction_horizon, unimodal, batch_size)
        else:
            print("  Using REGULAR TRAINING approach")
            return self._load_data_for_regular_training(patient_ids, version, sequence_length, prediction_horizon, unimodal, batch_size)
    
    def _load_data_for_transfer_learning(self, patient_ids, version, sequence_length, prediction_horizon, unimodal, batch_size):
        """Load data for transfer learning approach."""
        if len(patient_ids) < 2:
            raise ValueError(
                f"Transfer learning requires at least 2 patients (got {len(patient_ids)}). "
                "Need at least one source patient and one target patient. "
                "Please provide 2 or more patient IDs."
            )
        
        print(f"Loading data for {len(patient_ids)} patients for transfer learning...")

        # Load and preprocess all patient data
        all_patient_data = {}
        
        for patient_id in patient_ids:
            print(f"\nProcessing patient {patient_id}...")
            
            try:
                # Load raw data
                train_data = load_ohiot1dm_data(
                    data_dir=self.data_dir,
                    patient_ids=[patient_id],
                    mode='train',
                    version=version
                )
                
                test_data = load_ohiot1dm_data(
                    data_dir=self.data_dir,
                    patient_ids=[patient_id],
                    mode='test',
                    version=version
                )
                
                # Preprocess data
                processed_train = preprocess_ohiot1dm_data(
                    train_data,
                    include_feature_engineering=True,
                )
                
                processed_test = preprocess_ohiot1dm_data(
                    test_data,
                    include_feature_engineering=True,
                )
                
                all_patient_data[patient_id] = {
                    'train': processed_train[patient_id],
                    'test': processed_test[patient_id]
                }
                
                print(f"[OK] Patient {patient_id}: {len(processed_train[patient_id])} train, {len(processed_test[patient_id])} test samples")
                
            except Exception as e:
                print(f"[ERROR] Error processing patient {patient_id}: {e}")
                continue
        
        # Create transfer learning datasets for each target patient
        patient_data = {}
        
        for target_patient_id in patient_ids:
            if target_patient_id not in all_patient_data:
                continue
                
            print(f"\n  Preparing transfer learning datasets for target patient {target_patient_id}...")
            
            # Create global dataset (all other patients) + target patient datasets
            global_dataset, target_train_dataset, target_test_dataset = prepare_multi_patient_dataset(
                patient_data=all_patient_data,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                target_patient_id=target_patient_id,
                unimodal=unimodal
            )
            
            # Wrap datasets
            global_bg_dataset = BGConcatDataset(global_dataset)  # Use BGConcatDataset for ConcatDataset
            target_train_bg_dataset = BGForecastingDataset(target_train_dataset)
            target_test_bg_dataset = BGForecastingDataset(target_test_dataset)
            
            # Create DataLoaders
            # Global dataset for pre-training
            global_loader = DataLoader(
                global_bg_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            
            # Target patient train data for fine-tuning
            target_train_loader = DataLoader(
                target_train_bg_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            
            # Target patient test data for evaluation
            target_test_loader = DataLoader(
                target_test_bg_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # Create validation split from target training data
            train_size = int(0.8 * len(target_train_bg_dataset))
            val_size = len(target_train_bg_dataset) - train_size
            target_train_subset, target_val_subset = torch.utils.data.random_split(
                target_train_bg_dataset, [train_size, val_size]
            )
            
            target_val_loader = DataLoader(
                target_val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            target_train_loader_subset = DataLoader(
                target_train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            
            patient_data[target_patient_id] = {
                'train_dataset': target_train_dataset,
                'test_dataset': target_test_dataset,
                'global_loader': global_loader,  # For pre-training
                'train_loader': target_train_loader_subset,  # For fine-tuning
                'val_loader': target_val_loader,  # For validation
                'test_loader': target_test_loader,  # For evaluation
                'feature_dim': target_train_dataset.data.shape[1] if hasattr(target_train_dataset, 'data') else (1 if unimodal else 4),
                'global_dataset_size': len(global_dataset),
                'target_train_size': len(target_train_dataset),
                'target_test_size': len(target_test_dataset)
            }
            
            print(f"[OK] Transfer learning setup for patient {target_patient_id}:")
            print(f"  Global dataset: {len(global_dataset)} sequences")
            print(f"  Target train: {len(target_train_dataset)} sequences")
            print(f"  Target test: {len(target_test_dataset)} sequences")
        
        self.patient_data = patient_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.unimodal = unimodal
        
        print(f"\n[OK] Transfer learning data prepared for {len(patient_data)} patients")
        return patient_data
    
    def _load_data_for_regular_training(self, patient_ids, version, sequence_length, prediction_horizon, unimodal, batch_size):
        """Load data for regular training approach (no transfer learning)."""
        # Track data loading parameters
        self.tracker.log_data_params({
            'patient_ids': patient_ids,
            'version': version,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'unimodal': unimodal,
            'batch_size': batch_size
        })
        
        print(f"Loading data for {len(patient_ids)} patients for regular training...")
        
        # Step 1: Load and preprocess all patient data
        all_patient_data = {}
        
        for patient_id in patient_ids:
            print(f"\nProcessing patient {patient_id}...")
            
            try:
                # Load raw data
                train_data = load_ohiot1dm_data(
                    data_dir=self.data_dir,
                    patient_ids=[patient_id],
                    mode='train',
                    version=version
                )
                
                test_data = load_ohiot1dm_data(
                    data_dir=self.data_dir,
                    patient_ids=[patient_id],
                    mode='test',
                    version=version
                )
                
                # Preprocess data
                processed_train = preprocess_ohiot1dm_data(
                    train_data,
                    include_feature_engineering=True,
                )
                
                processed_test = preprocess_ohiot1dm_data(
                    test_data,
                    include_feature_engineering=True,
                )
                
                all_patient_data[patient_id] = {
                    'train': processed_train[patient_id],
                    'test': processed_test[patient_id]
                }
                
                print(f"[OK] Patient {patient_id}: {len(processed_train[patient_id])} train, {len(processed_test[patient_id])} test samples")
                
            except Exception as e:
                print(f"[ERROR] Error processing patient {patient_id}: {e}")
                continue
        
        # Create datasets and dataloaders for each patient individually
        patient_data = {}
        
        for patient_id in patient_ids:
            if patient_id not in all_patient_data:
                continue
                
            print(f"\n  Preparing regular training datasets for patient {patient_id}...")
           
            train_dataset, test_dataset = prepare_personal_data(
                    patient_data=all_patient_data[patient_id],
                    sequence_length=sequence_length,
                    prediction_horizon=prediction_horizon,
                    unimodal=unimodal
                    )

            # Wrap datasets to return (input, target) pairs
            train_dataset_wrapped = BGForecastingDataset(train_dataset)
            test_dataset_wrapped = BGForecastingDataset(test_dataset)
            
            # Get feature dimension
            feature_dim = train_dataset.data.shape[1] if hasattr(train_dataset, 'data') else (1 if unimodal else 6)
            
            # Create data loaders
            # Use a portion of train data for validation
            if len(train_dataset_wrapped) == 0:
                print(f"[ERROR] No training data available for patient {patient_id}")
                continue
                
            val_size = max(1, len(train_dataset_wrapped) // 10)  # At least 1 sample for validation
            train_size = len(train_dataset_wrapped) - val_size
            
            if train_size <= 0:
                print(f"[ERROR] Insufficient training data for patient {patient_id}")
                continue
                
            train_split, val_split = torch.utils.data.random_split(train_dataset_wrapped, [train_size, val_size])
            
            train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset_wrapped, batch_size=batch_size, shuffle=False)
            
            patient_data[patient_id] = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'feature_dim': feature_dim,
                'train_size': len(train_split),
                'val_size': len(val_split),
                'test_size': len(test_dataset)
            }
            
            print(f"[OK] Regular training setup for patient {patient_id}:")
            print(f"  Train: {len(train_split)} sequences")
            print(f"  Validation: {len(val_split)} sequences")
            print(f"  Test: {len(test_dataset)} sequences")
        
        self.patient_data = patient_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.unimodal = unimodal
        
        print(f"\n[OK] Regular training data prepared for {len(patient_data)} patients")
        return patient_data
    
    def train_model(self, 
                   model_class,
                   model_name: str,
                   patient_id: int,
                   hyperparameters: dict,
                   early_stopping_patience: int = 15,
                   pretrain_epochs: int = 50,
                   finetune_epochs: int = 25):
        """
        Train a model using either transfer learning or regular training based on configuration.
        
        Args:
            model_class: Model class (RNNBGModel, LSTMBGModel, GRUBGModel)
            model_name: Name for the model
            patient_id: Patient ID to train on
            hyperparameters: Model hyperparameters
            early_stopping_patience: Patience for early stopping
            pretrain_epochs: Number of pre-training epochs
            finetune_epochs: Number of fine-tuning epochs
            
        Returns:
            Trained model and training history
        """
        if patient_id not in self.patient_data:
            raise ValueError(f"Patient {patient_id} data not loaded")
        
        data = self.patient_data[patient_id]
        
        # Create model
        model = model_class(
            model_name=f"{model_name}_patient_{patient_id}",
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            feature_dim=data['feature_dim'],
            hyperparameters=hyperparameters
        )
        
        # Check if transfer learning is enabled
        transfer_learning_enabled = self.config.get('training', {}).get('transfer_learning', {}).get('enabled', True)
        
        
        if transfer_learning_enabled and pretrain_epochs > 0:
            # Transfer learning approach
            print(f"\n. Training {model_name} with TRANSFER LEARNING on patient {patient_id}...")
            self.tracker.log_training_start(model_name, patient_id, hyperparameters)
            
            # Phase 1: Pre-training on global dataset
            print(f". Pre-training on global dataset ({len(data['global_loader'].dataset)} sequences)...")
            pretrain_history = model.fit(
                train_loader=data['global_loader'],
                validation_loader=data['val_loader'],  # Use target patient validation
                epochs=pretrain_epochs,
                learning_rate=hyperparameters.get('learning_rate', 0.001),
                early_stopping_patience=early_stopping_patience
            )
            
            # Phase 2: Fine-tuning on target patient data
            print(f". Fine-tuning on target patient data ({len(data['train_loader'].dataset)} sequences)...")
            finetune_history = model.fit(
                train_loader=data['train_loader'],
                validation_loader=data['val_loader'],
                epochs=finetune_epochs,
                learning_rate=hyperparameters.get('learning_rate', 0.001),
                early_stopping_patience=early_stopping_patience
            )
            
            # Combine training histories
            total_epochs = pretrain_history['epochs_completed'] + finetune_history['epochs_completed']
            history = {
                'epochs_completed': total_epochs,
                'best_val_loss': finetune_history['best_val_loss'],
                'pretrain_epochs': pretrain_history['epochs_completed'],
                'finetune_epochs': finetune_history['epochs_completed'],
                'pretrain_history': pretrain_history,
                'finetune_history': finetune_history
            }
            
            print(f"[OK] Transfer learning completed:")
            print(f"  Pre-training: {pretrain_history['epochs_completed']} epochs")
            print(f"  Fine-tuning: {finetune_history['epochs_completed']} epochs")
            print(f"  Final validation loss: {finetune_history['best_val_loss']:.6f}")
            
        else:
            # Regular training (no transfer learning)
            total_epochs = finetune_epochs if finetune_epochs > 0 else pretrain_epochs
            print(f"\n  Training {model_name} with REGULAR TRAINING on patient {patient_id}...")
            print(f"  Training on patient data ({len(data['train_loader'].dataset)} sequences) for {total_epochs} epochs...")
            self.tracker.log_training_start(model_name, patient_id, hyperparameters)
            
            # Single-phase training on target patient data
            history = model.fit(
                train_loader=data['train_loader'],
                validation_loader=data['val_loader'],
                epochs=total_epochs,
                learning_rate=hyperparameters.get('learning_rate', 0.001),
                early_stopping_patience=early_stopping_patience
            )
            
            print(f"[OK] Regular training completed:")
            print(f"  Training: {history['epochs_completed']} epochs")
            print(f"  Final validation loss: {history['best_val_loss']:.6f}")
        
        self.tracker.log_training_completion(model_name, patient_id, history)
        return model, history
    
    def evaluate_model(self, model, patient_id: int, save_predictions: bool = True):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            patient_id: Patient ID to evaluate on
            save_predictions: Whether to save predictions
            
        Returns:
            Evaluation results dictionary
        """
        if patient_id not in self.patient_data:
            raise ValueError(f"Patient {patient_id} data not loaded")
        
        data = self.patient_data[patient_id]
        
        print(f"Evaluating {model.model_name}...")
        
        # Get predictions
        predictions = model.predict(data['test_loader'])
        print(f"[DEBUG] Raw prediction shape: {predictions.shape}")
        
        # Extract true values AND input sequences from the same DataLoader
        # We need to iterate through the loader again to get targets and inputs
        true_values = []
        input_sequences = []
        
        # Reset the DataLoader by getting it from the BGForecastingDataset
        test_bg_dataset = data['test_loader'].dataset
        
        for i in range(len(test_bg_dataset)):
            sequence, target = test_bg_dataset[i]
            true_values.append(target.numpy())
            # Store input sequences for temporal visualization
            input_sequences.append(sequence.numpy())
        
        # Handle multi-step predictions correctly
        # For multi-step predictions, extract only the final timestep
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-step predictions: extract final horizon only (last column)
            predictions = predictions[:, -1]
            print(f"[INFO] Extracted final timestep prediction (t+{self.prediction_horizon}), shape: {predictions.shape}")
        elif predictions.ndim > 1 and predictions.shape[1] == 1:
            # Single column: flatten
            predictions = predictions.flatten()
            print(f"[INFO] Flattened single-column prediction, shape: {predictions.shape}")
        
        # Handle multi-step targets correctly
        if len(true_values) > 0 and true_values[0].ndim > 0 and len(true_values[0]) > 1:
            # Multi-step targets: extract final horizon only (last value)
            y_true = np.array([t[-1] if t.ndim > 0 else t for t in true_values])
            print(f"[INFO] Extracted final timestep target (t+{self.prediction_horizon}), shape: {y_true.shape}")
        else:
            # Single-step targets: concatenate as before
            y_true = np.concatenate([t.reshape(-1) if t.ndim > 0 else [t] for t in true_values])
            print(f"[INFO] Concatenated single-step targets, shape: {y_true.shape}")

        # Targets and predictions are standardized for training. Clinical
        # metrics must receive the original glucose units exactly once.
        normalization_dataset = data.get('test_dataset')
        if normalization_dataset is None and hasattr(test_bg_dataset, 'ohio_dataset'):
            normalization_dataset = test_bg_dataset.ohio_dataset
        if normalization_dataset is None:
            raise ValueError("Test normalization state is unavailable for inverse transformation")
        predictions = normalization_dataset.inverse_transform_target(predictions)
        y_true = normalization_dataset.inverse_transform_target(y_true)
        input_sequences_array = np.asarray(input_sequences, dtype=np.float64)
        glucose_index = normalization_dataset.feature_columns.index('glucose')
        input_sequences_array[:, :, glucose_index] = normalization_dataset.inverse_transform_feature(
            input_sequences_array[:, :, glucose_index], 'glucose'
        )

        if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(y_true)):
            raise ValueError("Prediction and target values must be finite before evaluation")
        lower, upper = GLUCOSE_PLAUSIBLE_RANGE_MG_DL
        observed_min = min(float(np.min(predictions)), float(np.min(y_true)))
        observed_max = max(float(np.max(predictions)), float(np.max(y_true)))
        if observed_min < lower or observed_max > upper:
            raise ValueError(
                f"Glucose values outside plausible range [{lower:g}, {upper:g}] mg/dL: "
                f"observed [{observed_min:.3f}, {observed_max:.3f}]"
            )
        
        print(f"[DEBUG] Final prediction shape: {predictions.shape}")
        print(f"[DEBUG] Final target shape: {y_true.shape}")
        
        # Convert input sequences to numpy array for visualization
        input_sequences_array = np.array(input_sequences_array)
        print(f"[DEBUG] Input sequences shape: {input_sequences_array.shape}")
        
        # Verify shapes match
        if predictions.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch after processing: predictions {predictions.shape} vs targets {y_true.shape}. "
                f"This indicates an issue with multi-step prediction handling."
            )
        
        # Evaluate with comprehensive metrics
        results = self.evaluator.compute_metrics(
            y_true=y_true,
            y_pred=predictions,
            metrics=['mae', 'rmse', 'mape', 'mard', 'tir', 'clarke_ega', 'parkes_ega']
        )
        
        # Add model info
        results['model_info'] = {
            'model_name': model.model_name,
            'model_type': model.__class__.__name__,
            'patient_id': patient_id,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'prediction_horizon_minutes': self.prediction_horizon * 5,
            'target_units': 'mg/dL',
            'plausible_glucose_range_mg_dl': list(GLUCOSE_PLAUSIBLE_RANGE_MG_DL),
            'feature_dim': model.feature_dim,
            'hyperparameters': model.hyperparameters
        }
        
        # Save predictions if requested
        if save_predictions:
            # Create patient subdirectory
            patient_dir = self.experiment_dir / f"patient_{patient_id}"
            patient_dir.mkdir(exist_ok=True)
            
            pred_df = pd.DataFrame({
                'true_glucose_mg_dl': y_true,
                'predicted_glucose_mg_dl': predictions,
                'absolute_error_mg_dl': np.abs(y_true - predictions)
            })
            pred_file = patient_dir / f"{model.model_name}_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            results['predictions_file'] = str(pred_file)
            
            # Generate visualizations
            try:
                # Create prediction dashboard with temporal visualization
                dashboard_file = patient_dir / f"{model.model_name}_dashboard.png"
                create_prediction_dashboard(
                    y_true=y_true,
                    y_pred=predictions,
                    title=f"{model.model_name.split('_')[0]} Model for patient {patient_id}",
                    prediction_horizon=self.prediction_horizon,
                    sequence_length=self.sequence_length,
                    input_sequences=input_sequences_array,
                    sample_interval_minutes=5,
                    save_path=dashboard_file
                )
                results['dashboard_file'] = str(dashboard_file)
                print(f"  [OK] Dashboard saved: {dashboard_file.name}")
                
                # Clarke Error Grid Analysis visualization
                clarke_file = patient_dir / f"{model.model_name}_clarke_ega.png"
                plot_clarke_analysis(
                    y_true=y_true,
                    y_pred=predictions,
                    title=f"Clarke EGA - {model.model_name}",
                    save_path=clarke_file
                )
                results['clarke_plot_file'] = str(clarke_file)
                print(f"  [OK] Clarke EGA saved: {clarke_file.name}")
                
                # Parkes Error Grid Analysis visualization
                parkes_file = patient_dir / f"{model.model_name}_parkes_ega.png"
                plot_parkes_analysis(
                    y_true=y_true,
                    y_pred=predictions,
                    diabetes_type=1,  # Type 1 diabetes
                    title=f"Parkes EGA - {model.model_name}",
                    save_path=parkes_file
                )
                results['parkes_plot_file'] = str(parkes_file)
                print(f"  [OK] Parkes EGA saved: {parkes_file.name}")
                
            except Exception as e:
                print(f"      Warning: Could not generate visualizations: {e}")
        
        # Track evaluation results
        self.tracker.log_evaluation_results(model.model_name, patient_id, results)
        
        return results
    
    def run_experiment(self, 
                      patient_ids: list = None,
                      hyperparameters: dict = None,
                      pretrain_epochs: int = None,
                      finetune_epochs: int = None,
                      **data_params):
        """
        Run complete experiment with multiple models and patients.
        
        Args:
            patient_ids: List of patient IDs to test on
            hyperparameters: Dictionary of hyperparameters for each model type
            pretrain_epochs: Number of pre-training epochs
            finetune_epochs: Number of fine-tuning epochs
            **data_params: Additional data loading parameters
        """
        # Start experiment tracking
        self.tracker.start_experiment()
        
        # Get configuration values
        training_config = self.config.get('training', {})
        data_config = self.config.get('data', {})
        
        # Use config values if not provided as arguments
        if patient_ids is None:
            patient_ids = data_config.get('patient_ids', [540, 544])
        
        if pretrain_epochs is None:
            # Check if transfer learning is enabled
            if training_config.get('transfer_learning', {}).get('enabled', True):
                pretrain_epochs = training_config.get('epochs', 50)
            else:
                pretrain_epochs = 0  # No pre-training if transfer learning disabled
        
        if finetune_epochs is None:
            # Check if transfer learning is enabled
            if training_config.get('transfer_learning', {}).get('enabled', True):
                finetune_epochs = training_config.get('transfer_learning', {}).get('fine_tune_epochs', 25)
            else:
                finetune_epochs = training_config.get('epochs', 50)  # Use full epochs for regular training
        
        if hyperparameters is None:
            hyperparameters = {
                'RNN': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'nonlinearity': 'tanh'
                },
                'LSTM': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001
                },
                'GRU': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001
                }
            }
        
        model_classes = {
            'RNN': RNNBGModel,
            'LSTM': LSTMBGModel,
            'GRU': GRUBGModel
        }
        
        print(f"\n{'='*60}")
        print("RUNNING MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"Models: {list(model_classes.keys())}")
        print(f"Patients: {patient_ids}")
        
        # Determine and display the training approach
        transfer_learning_enabled = training_config.get('transfer_learning', {}).get('enabled', True)
        if transfer_learning_enabled and pretrain_epochs > 0:
            print(f"Approach: Transfer Learning")
            print(f"Pre-train epochs: {pretrain_epochs}")
            print(f"Fine-tune epochs: {finetune_epochs}")
        else:
            total_epochs = finetune_epochs if finetune_epochs > 0 else pretrain_epochs
            print(f"Approach: Regular Training")
            print(f"Training epochs: {total_epochs}")
            print(f"Transfer learning: Disabled")
        
        # Load data
        self.load_and_prepare_data(patient_ids=patient_ids, **data_params)
        
        comparison_results = {}
        
        for patient_id in patient_ids:
            if patient_id not in self.patient_data:
                print(f"    Skipping patient {patient_id} - data not loaded")
                continue
            
            print(f"\n{'='*40}")
            print(f"PATIENT {patient_id}")
            print(f"{'='*40}")
            
            patient_results = {}
            
            for model_name, model_class in model_classes.items():
                try:
                    # Train model
                    model, history = self.train_model(
                        model_class=model_class,
                        model_name=model_name,
                        patient_id=patient_id,
                        hyperparameters=hyperparameters[model_name],
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs
                    )
                    
                    # Evaluate model
                    results = self.evaluate_model(model, patient_id)
                    
                    # Add training history
                    results['training_history'] = history
                    
                    # Save model in patient directory
                    patient_dir = self.experiment_dir / f"patient_{patient_id}"
                    patient_dir.mkdir(exist_ok=True)
                    model_file = patient_dir / f"{model.model_name}.pth"
                    model.save_model(model_file)
                    results['model_file'] = str(model_file)
                    
                    patient_results[model_name] = results
                    
                    print(f"\n{model_name} Results:")
                    print(f"  MAE: {results['mae']:.3f} mg/dL")
                    print(f"  RMSE: {results['rmse']:.3f} mg/dL")
                    print(f"  MARD: {results['mard']:.2f}%")
                    print(f"  TIR: {results['tir']['time_in_range']:.1f}%")

                    # Handle nested clarke_zones dictionary
                    clarke_zones = results.get('clarke_zones', {});
                    if clarke_zones:
                        clarke_ab = clarke_zones.get('A', 0) + clarke_zones.get('B', 0);
                        print(f"  Clarke A+B: {clarke_ab:.1f}%");
                    
                except Exception as e:
                    print(f"  Error training {model_name} for patient {patient_id}: {e}")
                    continue
            
            comparison_results[patient_id] = patient_results
        
        # Save comprehensive results
        self.experiment_results = comparison_results
        
        # Create comprehensive metrics JSON with patient data and averages
        comprehensive_metrics = self._create_comprehensive_metrics_json(comparison_results)
        
        # Save the main comparison results
        results_file = self.experiment_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_results = self._prepare_results_for_json(comparison_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save the comprehensive metrics JSON
        metrics_file = self.experiment_dir / f"comprehensive_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        
        print(f"\n[OK] Results saved to {results_file}")
        print(f"[OK] Comprehensive metrics saved to {metrics_file}")
        
        # Generate comprehensive reports
        print(f"\n{'='*60}")
        print("GENERATING REPORTS")
        print(f"{'='*60}")
        
        # Generate comprehensive reports
        self._generate_comprehensive_reports(comparison_results)
        
        # End experiment tracking
        self.tracker.end_experiment(comparison_results)
        
        return comparison_results
    
    def _generate_comprehensive_reports(self, comparison_results):
        """
        Generate comprehensive reports including CSV, JSON, LaTeX, and comparison reports.
        
        Args:
            comparison_results: Dictionary with experiment results
        """
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORTS")
        print(f"{'='*60}")
        
        # 1. Export individual metrics to CSV (ordered format)
        self._export_individual_metrics_csv(comparison_results)
        
        # 2. Generate model comparison reports
        self._generate_model_comparison_reports(comparison_results)
        
        # 3. Generate patient comparison reports
        self._generate_patient_comparison_reports(comparison_results)
        
        # 4. Generate LaTeX reports
        self._generate_latex_reports(comparison_results)
        
        # 5. Generate aggregated summary reports
        self._generate_summary_reports(comparison_results)
        
        print(f"[OK] All reports generated successfully!")
    
    def _export_individual_metrics_csv(self, comparison_results):
        """Export individual model metrics to CSV."""
        try:
            csv_file = self.experiment_dir / f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            first_export = True
            
            for patient_id, patient_results in comparison_results.items():
                for model_name, results in patient_results.items():
                    # Create a flattened metrics dictionary for this model
                    flat_metrics = {}
                    for key, value in results.items():
                        if key in ['model_info', 'training_history', 'predictions_file', 'dashboard_file', 'clarke_plot_file', 'parkes_plot_file', 'model_file']:
                            continue  # Skip non-metric fields
                        flat_metrics[key] = value
                    
                    # Add patient ID to metrics
                    flat_metrics['patient_id'] = patient_id
                    
                    # Export each model's metrics
                    export_metrics_to_csv(
                        metrics=flat_metrics,
                        filepath=str(csv_file),
                        model_name=model_name,
                        append=not first_export
                    )
                    first_export = False
            
            print(f"  ✓ Individual metrics CSV: {csv_file.name}")
        except Exception as e:
            print(f"    Warning: Could not export individual metrics CSV: {e}")
    
    def _generate_model_comparison_reports(self, comparison_results):
        """Generate model comparison reports across all patients."""
        try:
            from ..evaluation.reporting import generate_model_comparison_report, export_comparison_to_file
            
            # Aggregate metrics across all patients for each model
            model_aggregated = {}
            
            for patient_id, patient_results in comparison_results.items():
                for model_name, results in patient_results.items():
                    if model_name not in model_aggregated:
                        model_aggregated[model_name] = []
                    
                    # Extract numeric metrics only
                    metrics = {}
                    for key, value in results.items():
                        if key in ['model_info', 'training_history', 'predictions_file', 'dashboard_file', 'clarke_plot_file', 'parkes_plot_file', 'model_file']:
                            continue
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        elif isinstance(value, dict):
                            # Handle nested metrics (like clarke_zones, parkes_zones)
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    metrics[f"{key}_{subkey}"] = subvalue
                    
                    model_aggregated[model_name].append(metrics)
            
            # Calculate averages
            model_averages = {}
            for model_name, metrics_list in model_aggregated.items():
                if not metrics_list:
                    continue
                
                # Get all metric keys
                all_keys = set()
                for metrics in metrics_list:
                    all_keys.update(metrics.keys())
                
                # Calculate averages
                averages = {}
                for key in all_keys:
                    values = [m.get(key, 0) for m in metrics_list if key in m]
                    if values:
                        averages[key] = sum(values) / len(values)
                
                model_averages[model_name] = averages
            
            # Export model comparison reports
            if model_averages:
                # CSV format
                csv_file = self.experiment_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                export_comparison_to_file(model_averages, str(csv_file), 'csv')
                print(f"  ✓ Model comparison CSV: {csv_file.name}")
                
                # JSON format
                json_file = self.experiment_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                export_comparison_to_file(model_averages, str(json_file), 'json')
                print(f"  ✓ Model comparison JSON: {json_file.name}")
                
                # Markdown format
                md_file = self.experiment_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                export_comparison_to_file(model_averages, str(md_file), 'markdown')
                print(f"  ✓ Model comparison Markdown: {md_file.name}")
                
        except Exception as e:
            print(f"    Warning: Could not generate model comparison reports: {e}")
    
    def _generate_patient_comparison_reports(self, comparison_results):
        """Generate patient comparison reports across all models."""
        try:
            from ..evaluation.reporting import generate_patient_comparison_report
            
            # Prepare patient metrics structure
            patient_metrics = {}
            
            for patient_id, patient_results in comparison_results.items():
                patient_metrics[str(patient_id)] = {}
                
                for model_name, results in patient_results.items():
                    # Extract numeric metrics only
                    metrics = {}
                    for key, value in results.items():
                        if key in ['model_info', 'training_history', 'predictions_file', 'dashboard_file', 'clarke_plot_file', 'parkes_plot_file', 'model_file']:
                            continue
                        if isinstance(value, (int, float)):
                            metrics[key] = value
                        elif isinstance(value, dict):
                            # Handle nested metrics
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    metrics[f"{key}_{subkey}"] = subvalue
                    
                    patient_metrics[str(patient_id)][model_name] = metrics
            
            # Generate patient comparison report
            if patient_metrics:
                # Markdown format
                md_content = generate_patient_comparison_report(patient_metrics, 'markdown')
                md_file = self.experiment_dir / f"patient_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                
                with open(md_file, 'w') as f:
                    f.write("# Patient Comparison Report\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(md_content)
                
                print(f"  ✓ Patient comparison Markdown: {md_file.name}")
                
        except Exception as e:
            print(f"    Warning: Could not generate patient comparison reports: {e}")
    
    def _generate_latex_reports(self, comparison_results):
        """Generate comprehensive LaTeX reports."""
        try:
            from ..evaluation.reporting import generate_latex_report
            
            # Generate overall experiment report
            self._generate_overall_latex_report(comparison_results)
            
            # Generate individual patient reports
            self._generate_patient_latex_reports(comparison_results)
            
        except Exception as e:
            print(f"    Warning: Could not generate LaTeX reports: {e}")
    
    def _generate_overall_latex_report(self, comparison_results):
        """Generate overall experiment LaTeX report."""
        try:
            # Aggregate all models across all patients
            model_metrics = {}
            y_true_dict = {}
            y_pred_dict = {}
            
            for patient_id, patient_results in comparison_results.items():
                for model_name, results in patient_results.items():
                    # Create unique model name with patient
                    unique_model_name = f"{model_name}_Patient_{patient_id}"
                    
                    # Extract metrics (exclude non-metric fields)
                    metrics = {}
                    for key, value in results.items():
                        if key in ['model_info', 'training_history', 'predictions_file', 'dashboard_file', 'clarke_plot_file', 'parkes_plot_file', 'model_file']:
                            continue
                        metrics[key] = value
                    
                    model_metrics[unique_model_name] = metrics
                    
                    # Load actual predictions from CSV file if available
                    if 'predictions_file' in results:
                        try:
                            pred_df = pd.read_csv(results['predictions_file'])
                            y_true_dict[unique_model_name] = pred_df['true_values'].values
                            y_pred_dict[unique_model_name] = pred_df['predictions'].values
                        except Exception as e:
                            print(f"      Warning: Could not load predictions for {unique_model_name}: {e}")
                            continue
            
            # Generate overall report if we have data
            if y_true_dict and y_pred_dict:
                transfer_learning_enabled = self.config.get('training', {}).get('transfer_learning', {}).get('enabled', True)
                training_type = "Transfer Learning" if transfer_learning_enabled else "Regular Training"
                
                latex_file = generate_latex_report(
                    model_metrics=model_metrics,
                    y_true_dict=y_true_dict,
                    y_pred_dict=y_pred_dict,
                    output_dir=str(self.experiment_dir),
                    report_title=f"Blood Glucose Forecasting Benchmark Report - {training_type}",
                    author="BG-Forecasting Benchmark System",
                    include_visualizations=True
                )
                print(f"  ✓ Overall LaTeX report: {Path(latex_file).name}")
            else:
                print("    Warning: Skipping overall LaTeX report - no prediction data available")
                
        except Exception as e:
            print(f"    Warning: Could not generate overall LaTeX report: {e}")
    
    def _generate_patient_latex_reports(self, comparison_results):
        """Generate individual patient LaTeX reports."""
        try:
            from ..evaluation.reporting import generate_latex_report
            
            for patient_id, patient_results in comparison_results.items():
                # Prepare data for LaTeX report
                model_metrics = {}
                y_true_dict = {}
                y_pred_dict = {}
                
                for model_name, results in patient_results.items():
                    # Extract metrics (exclude non-metric fields)
                    metrics = {k: v for k, v in results.items() 
                             if k not in ['model_info', 'training_history', 'predictions_file', 'dashboard_file', 'clarke_plot_file', 'parkes_plot_file', 'model_file']}
                    model_metrics[model_name] = metrics
                    
                    # Load actual predictions from CSV file if available
                    if 'predictions_file' in results:
                        try:
                            pred_df = pd.read_csv(results['predictions_file'])
                            y_true_dict[model_name] = pred_df['true_values'].values
                            y_pred_dict[model_name] = pred_df['predictions'].values
                        except Exception as e:
                            print(f"      Warning: Could not load predictions for {model_name}: {e}")
                            continue
                
                # Only generate report if we have actual data
                if y_true_dict and y_pred_dict:
                    # Create patient-specific directory
                    patient_dir = self.experiment_dir / f"patient_{patient_id}"
                    patient_dir.mkdir(exist_ok=True)
                    
                    transfer_learning_enabled = self.config.get('training', {}).get('transfer_learning', {}).get('enabled', True)
                    training_type = "Transfer Learning" if transfer_learning_enabled else "Regular Training"
                    
                    latex_file = generate_latex_report(
                        model_metrics=model_metrics,
                        y_true_dict=y_true_dict,
                        y_pred_dict=y_pred_dict,
                        output_dir=str(patient_dir),
                        report_title=f"Blood Glucose Forecasting Results - Patient {patient_id} ({training_type})",
                        author="BG-Forecasting Benchmark System",
                        include_visualizations=True
                    )
                    print(f"  ✓ Patient {patient_id} LaTeX report: {Path(latex_file).name}")
                else:
                    print(f"    Warning: Skipping LaTeX report for patient {patient_id} - no prediction data available")
                    
        except Exception as e:
            print(f"    Warning: Could not generate patient LaTeX reports: {e}")
    
    def _generate_summary_reports(self, comparison_results):
        """Generate aggregated summary reports."""
        try:
            # Generate experiment summary
            summary_file = self.experiment_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(summary_file, 'w') as f:
                f.write("# Experiment Summary Report\n\n")
                f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Training configuration
                transfer_learning_enabled = self.config.get('training', {}).get('transfer_learning', {}).get('enabled', True)
                f.write(f"**Training Type:** {'Transfer Learning' if transfer_learning_enabled else 'Regular Training'}\n")
                f.write(f"**Patients:** {len(comparison_results)}\n")
                
                # Get models from first patient
                if comparison_results:
                    first_patient = list(comparison_results.keys())[0]
                    models = list(comparison_results[first_patient].keys())
                    f.write(f"**Models:** {', '.join(models)}\n\n")
                
                # Performance summary
                f.write("## Performance Summary\n\n")
                
                # Calculate averages per model
                model_stats = {}
                for patient_id, patient_results in comparison_results.items():
                    for model_name, results in patient_results.items():
                        if model_name not in model_stats:
                            model_stats[model_name] = {'mae': [], 'rmse': [], 'mape': []}
                        
                        if 'mae' in results:
                            model_stats[model_name]['mae'].append(results['mae'])
                        if 'rmse' in results:
                            model_stats[model_name]['rmse'].append(results['rmse'])
                        if 'mape' in results:
                            model_stats[model_name]['mape'].append(results['mape'])
                
                # Write summary table
                f.write("| Model | MAE (mg/dL) | RMSE (mg/dL) | MAPE (%) |\n")
                f.write("|-------|-------------|--------------|----------|\n")
                
                for model_name, stats in model_stats.items():
                    mae_avg = sum(stats['mae']) / len(stats['mae']) if stats['mae'] else 0
                    rmse_avg = sum(stats['rmse']) / len(stats['rmse']) if stats['rmse'] else 0
                    mape_avg = sum(stats['mape']) / len(stats['mape']) if stats['mape'] else 0
                    
                    f.write(f"| {model_name} | {mae_avg:.3f} | {rmse_avg:.3f} | {mape_avg:.3f} |\n")
                
                f.write("\n## Individual Patient Results\n\n")
                
                for patient_id, patient_results in comparison_results.items():
                    f.write(f"### Patient {patient_id}\n\n")
                    f.write("| Model | MAE | RMSE | MAPE |\n")
                    f.write("|-------|-----|------|------|\n")
                    
                    for model_name, results in patient_results.items():
                        mae = results.get('mae', 0)
                        rmse = results.get('rmse', 0)
                        mape = results.get('mape', 0)
                        f.write(f"| {model_name} | {mae:.3f} | {rmse:.3f} | {mape:.3f} |\n")
                    
                    f.write("\n")
            
            print(f"  ✓ Experiment summary: {summary_file.name}")
            
        except Exception as e:
            print(f"    Warning: Could not generate summary reports: {e}")
    
    def _create_comprehensive_metrics_json(self, comparison_results):
        """
        Create a comprehensive metrics JSON with patient data and averages.
        
        Args:
            comparison_results: Dictionary with experiment results
            
        Returns:
            Dictionary with organized metrics by patient and model averages
        """
        comprehensive_metrics = {
            "experiment_info": {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_patients": len(comparison_results),
                "models": list(next(iter(comparison_results.values())).keys()) if comparison_results else []
            },
            "patient_metrics": {},
            "model_averages": {}
        }
        
        # Organize metrics by patient
        for patient_id, patient_results in comparison_results.items():
            comprehensive_metrics["patient_metrics"][str(patient_id)] = {}
            
            for model_name, results in patient_results.items():
                # Extract key metrics for this patient/model
                patient_metrics = {
                    "mae": results.get('mae'),
                    "rmse": results.get('rmse'),
                    "mape": results.get('mape'),
                    "mard": results.get('mard'),
                    "tir": results.get('tir', {}),
                    "clarke_zones": results.get('clarke_zones', {}),
                    "parkes_zones": results.get('parkes_zones', {})
                }
                
                # Calculate derived metrics
                clarke_zones = results.get('clarke_zones', {})
                patient_metrics["clarke_a_b"] = clarke_zones.get('A', 0) + clarke_zones.get('B', 0)
                
                parkes_zones = results.get('parkes_zones', {})
                patient_metrics["parkes_a_b"] = parkes_zones.get('A', 0) + parkes_zones.get('B', 0)
                
                comprehensive_metrics["patient_metrics"][str(patient_id)][model_name] = patient_metrics
        
        # Calculate model averages across all patients
        if comparison_results:
            model_names = list(next(iter(comparison_results.values())).keys())
            
            for model_name in model_names:
                model_metrics = []
                
                for patient_id, patient_results in comparison_results.items():
                    if model_name in patient_results:
                        model_metrics.append(patient_results[model_name])
                
                if model_metrics:
                    # Calculate averages for numerical metrics
                    avg_metrics = {}
                    numerical_fields = ['mae', 'rmse', 'mape', 'mard']
                    
                    for field in numerical_fields:
                        values = [m.get(field) for m in model_metrics if m.get(field) is not None]
                        if values:
                            avg_metrics[field] = sum(values) / len(values)
                    
                    # Average TIR metrics
                    tir_fields = ['time_in_range', 'time_above_range', 'time_below_range']
                    avg_tir = {}
                    for field in tir_fields:
                        values = [m.get('tir', {}).get(field) for m in model_metrics 
                                if m.get('tir', {}).get(field) is not None]
                        if values:
                            avg_tir[field] = sum(values) / len(values)
                    avg_metrics['tir'] = avg_tir
                    
                    # Average Clarke zones
                    clarke_zones = ['A', 'B', 'C', 'D', 'E']
                    avg_clarke = {}
                    for zone in clarke_zones:
                        values = [m.get('clarke_zones', {}).get(zone) for m in model_metrics
                                if m.get('clarke_zones', {}).get(zone) is not None]
                        if values:
                            avg_clarke[zone] = sum(values) / len(values)
                    avg_metrics['clarke_zones'] = avg_clarke
                    avg_metrics['clarke_a_b'] = avg_clarke.get('A', 0) + avg_clarke.get('B', 0)
                    
                    # Average Parkes zones
                    parkes_zones = ['A', 'B', 'C', 'D', 'E']
                    avg_parkes = {}
                    for zone in parkes_zones:
                        values = [m.get('parkes_zones', {}).get(zone) for m in model_metrics
                                if m.get('parkes_zones', {}).get(zone) is not None]
                        if values:
                            avg_parkes[zone] = sum(values) / len(values)
                    avg_metrics['parkes_zones'] = avg_parkes
                    avg_metrics['parkes_a_b'] = avg_parkes.get('A', 0) + avg_parkes.get('B', 0)
                    
                    # Add patient count
                    avg_metrics['patient_count'] = len(model_metrics)
                    
                    comprehensive_metrics["model_averages"][model_name] = avg_metrics
        
        return self._prepare_results_for_json(comprehensive_metrics)

    
    def _prepare_results_for_json(self, results):
        """Convert numpy types to JSON-serializable types."""
        def convert_item(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            elif isinstance(item, np.floating):
                return float(item)
            elif isinstance(item, np.integer):
                return int(item)
            elif isinstance(item, dict):
                return {k: convert_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert_item(v) for v in item]
            else:
                return item
        
        return convert_item(results)
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.experiment_results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        # Collect all results for summary
        all_results = []
        for patient_id, patient_results in self.experiment_results.items():
            for model_name, results in patient_results.items():
                # Handle nested clarke_zones dictionary
                clarke_zones = results.get('clarke_zones', {})
                clarke_ab = clarke_zones.get('A', 0) + clarke_zones.get('B', 0) if clarke_zones else 0
                
                # Extract TIR values
                tir_values = results['tir']
                
                all_results.append({
                    'patient_id': patient_id,
                    'model': model_name,
                    'mae': results['mae'],
                    'rmse': results['rmse'],
                    'mard': results['mard'],
                    'time_in_range': tir_values['time_in_range'],
                    'time_below_range': tir_values['time_below_range'],
                    'time_above_range': tir_values['time_above_range'],
                    'clarke_ab': clarke_ab
                })
        
        if not all_results:
            print("No valid results to summarize")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(all_results)
        
        print("\nPer-Patient Results:")
        print("-" * 80)
        print(f"{'Patient':<8} {'Model':<6} {'MAE':<8} {'RMSE':<8} {'MARD':<8} {'TIR':<8} {'Clarke A+B':<12}")
        print("-" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"{row['patient_id']:<8} {row['model']:<6} {row['mae']:<8.2f} {row['rmse']:<8.2f} "
                  f"{row['mard']:<8.1f} {row['time_in_range']:<8.1f} {row['clarke_ab']:<12.1f}")
        
        # Overall averages
        print("\nOverall Model Performance (Average across patients):")
        print("-" * 60)
        model_avg = summary_df.groupby('model').agg({
            'mae': 'mean',
            'rmse': 'mean', 
            'mard': 'mean',
            'time_in_range': 'mean',
            'clarke_ab': 'mean'
        }).round(2)
        
        print(model_avg)
        
        # Best performing model
        best_model = model_avg.loc[model_avg['mae'].idxmin()]
        print(f"\n  Best Overall Model (lowest MAE): {best_model.name}")
        print(f"   MAE: {best_model['mae']:.2f} mg/dL")
        print(f"   MARD: {best_model['mard']:.1f}%")
        print(f"   TIR: {best_model['time_in_range']:.1f}%")


def run_experiment_from_config(config_path: str = None, **kwargs):
    """
    Run experiment from configuration file or parameters.
    
    Args:
        config_path: Path to YAML configuration file
        **kwargs: Direct experiment parameters
    """
    if not config_path:
        raise ValueError("config_path is required; direct parameter execution is not supported")
    if kwargs:
        raise ValueError("Configuration overrides are not supported; update the YAML file instead")

    from ..configs import load_config, validate_data_files
    from .configured import run_configured_experiment

    config = load_config(config_path)
    return run_configured_experiment(config, validate_data_files(config))


def main(argv=None):
    """Compatibility entry point; the primary interface is ``benchmark.cli``."""
    parser = argparse.ArgumentParser(description="Run one benchmark experiment")
    parser.add_argument("--config", required=True, help="path to a versioned YAML config")
    args = parser.parse_args(argv)
    run_experiment_from_config(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
