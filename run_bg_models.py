#!/usr/bin/env python3
"""
Comprehensive transfer learning training and evaluation script for blood glucose forecasting models.

This script demonstrates how to:
1. Load and preprocess OhioT1DM data
2. Create PyTorch datasets for transfer learning
3. Train different RNN models (RNN, LSTM, GRU) using transfer learning approach
4. Pre-train models on global dataset (all other patients)
5. Fine-tune models on target patient data
6. Evaluate models with comprehensive BG-specific metrics
7. Compare model performance across patients
8. Save results and trained models

Transfer Learning Approach:
- Pre-training phase: Models learn from data of all other patients
- Fine-tuning phase: Models adapt to specific target patient
- Improved performance through knowledge transfer
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import json
import argparse
from datetime import datetime
import warnings
from typing import List
warnings.filterwarnings('ignore')

# Add benchmark to path
sys.path.append('benchmark')

# Import our modules
try:
    from data.loaders import load_ohiot1dm_data
    from data.preprocessors import preprocess_ohiot1dm_data
    from data.torch_dataset import prepare_patient_datasets, prepare_multi_patient_dataset
    from models.rnn import RNNBGModel, LSTMBGModel, GRUBGModel
    from evaluation.evaluator import BGEvaluator
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

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
        # We need to find which sub-dataset this index belongs to and get the target
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

class BGModelTrainer:
    """
    Comprehensive trainer for blood glucose forecasting models.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 results_dir: str = "results",
                 models_dir: str = "saved_models"):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Path to raw OhioT1DM data
            results_dir: Directory to save results
            models_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = BGEvaluator(
            default_metrics=['mae', 'rmse', 'mard', 'tir', 'tbr', 'tar', 'clarke', 'parkes'],
        )
        
        # Store results
        self.experiment_results = {}
    
    def load_and_prepare_data(self, 
                             patient_ids: list = [540, 544, 552, 567, 584, 596],
                             version: List[str] = ["2020"],
                             sequence_length: int = 12,
                             prediction_horizon: int = 6,
                             unimodal: bool = False,
                             batch_size: int = 32):
        """
        Load and prepare data for all specified patients using transfer learning approach.
        
        Args:
            patient_ids: List of patient IDs to load
            version: Dataset version ('2018' or '2020')
            sequence_length: Input sequence length (history)
            prediction_horizon: Prediction horizon (future steps)
            unimodal: Whether to use only glucose features
            batch_size: Batch size for DataLoaders
            
        Returns:
            Dictionary with patient data and DataLoaders prepared for transfer learning
        """
        print(f"\n{'='*60}")
        print("LOADING AND PREPARING DATA")
        print(f"{'='*60}")
        print("🔄 Using TRANSFER LEARNING approach")
        
        return self._load_data_for_transfer_learning(
            patient_ids, version, sequence_length, prediction_horizon, unimodal, batch_size
        )
    
    def _load_data_for_transfer_learning(self, patient_ids, version, sequence_length, prediction_horizon, unimodal, batch_size):
        """Load data for transfer learning approach."""
        if len(patient_ids) < 2:
            raise ValueError(
                f"Transfer learning requires at least 2 patients (got {len(patient_ids)}). "
                "Need at least one source patient and one target patient. "
                "Please provide 2 or more patient IDs."
            )
        
        print(f"Loading data for {len(patient_ids)} patients for transfer learning...")
        
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
                
                print(f"✓ Patient {patient_id}: {len(processed_train[patient_id])} train, {len(processed_test[patient_id])} test samples")
                
            except Exception as e:
                print(f"✗ Error processing patient {patient_id}: {e}")
                continue
        
        # Step 2: Create transfer learning datasets for each target patient
        patient_data = {}
        
        for target_patient_id in patient_ids:
            if target_patient_id not in all_patient_data:
                continue
                
            print(f"\n🎯 Preparing transfer learning datasets for target patient {target_patient_id}...")
            
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
            
            print(f"✓ Transfer learning setup for patient {target_patient_id}:")
            print(f"  Global dataset: {len(global_dataset)} sequences")
            print(f"  Target train: {len(target_train_dataset)} sequences")
            print(f"  Target test: {len(target_test_dataset)} sequences")
        
        self.patient_data = patient_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.unimodal = unimodal
        
        print(f"\n✓ Transfer learning data prepared for {len(patient_data)} patients")
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
        Train a model using transfer learning approach.
        
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
        
        print(f"\n🔄 Training {model_name} with TRANSFER LEARNING on patient {patient_id}...")
        print(f"Model: {model}")
        
        # Step 1: Pre-train on global data (all other patients)
        print(f"📚 Pre-training on global dataset ({data['global_dataset_size']} sequences)...")
        pretrain_history = model.fit(
            train_loader=data['global_loader'],
            validation_loader=data['val_loader'],  # Use target patient validation
            epochs=pretrain_epochs,
            learning_rate=hyperparameters.get('learning_rate', 0.001),
            early_stopping_patience=early_stopping_patience
        )
        
        # Step 2: Fine-tune on target patient data
        print(f"🎯 Fine-tuning on target patient data ({data['target_train_size']} sequences)...")
        finetune_history = model.fit(
            train_loader=data['train_loader'],
            validation_loader=data['val_loader'],
            epochs=finetune_epochs,
            learning_rate=hyperparameters.get('learning_rate', 0.001) * 0.1,  # Lower learning rate for fine-tuning
            early_stopping_patience=early_stopping_patience // 2  # Less patience for fine-tuning
        )
        
        # Combine histories
        total_epochs = pretrain_history['epochs_completed'] + finetune_history['epochs_completed']
        history = {
            'epochs_completed': total_epochs,
            'best_val_loss': finetune_history['best_val_loss'],
            'pretrain_epochs': pretrain_history['epochs_completed'],
            'finetune_epochs': finetune_history['epochs_completed'],
            'pretrain_history': pretrain_history,
            'finetune_history': finetune_history
        }
        
        print(f"✓ Transfer learning completed:")
        print(f"  Pre-training: {pretrain_history['epochs_completed']} epochs")
        print(f"  Fine-tuning: {finetune_history['epochs_completed']} epochs")
        print(f"  Final validation loss: {finetune_history['best_val_loss']:.6f}")
        
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
        
        # Extract true values from the same DataLoader that was used for prediction
        # We need to iterate through the loader again to get targets
        true_values = []
        
        # Reset the DataLoader by getting it from the BGForecastingDataset
        test_bg_dataset = data['test_loader'].dataset
        
        for i in range(len(test_bg_dataset)):
            _, target = test_bg_dataset[i]
            true_values.append(target.numpy())
        
        y_true = np.concatenate([t.reshape(-1) if t.ndim > 0 else [t] for t in true_values])
        
        # Ensure predictions have the same shape
        if predictions.ndim > 1:
            predictions = predictions.reshape(-1)
        
        # Handle shape mismatch
        if predictions.shape != y_true.shape:
            if self.prediction_horizon == 1:
                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    predictions = predictions.flatten()
                if y_true.ndim == 2 and y_true.shape[1] == 1:
                    y_true = y_true.flatten()
        
        # Evaluate with comprehensive metrics
        results = self.evaluator.compute_metrics(
            y_true=y_true,
            y_pred=predictions,
            metrics=['mae', 'rmse', 'mard', 'tir', 'tbr', 'tar', 'clarke', 'parkes']
        )
        
        # Add model info
        results['model_info'] = {
            'model_name': model.model_name,
            'model_type': model.__class__.__name__,
            'patient_id': patient_id,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'feature_dim': model.feature_dim,
            'hyperparameters': model.hyperparameters
        }
        
        # Save predictions if requested
        if save_predictions:
            pred_df = pd.DataFrame({
                'true_values': y_true,
                'predictions': predictions,
                'error': np.abs(y_true - predictions)
            })
            pred_file = self.results_dir / f"{model.model_name}_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            results['predictions_file'] = str(pred_file)
        
        return results
    
    def run_model_comparison(self, 
                           patient_ids: list = [540, 544],
                           hyperparameters: dict = None,
                           pretrain_epochs: int = 50,
                           finetune_epochs: int = 25):
        """
        Run comprehensive model comparison across multiple patients using transfer learning.
        
        Args:
            patient_ids: List of patient IDs to test on
            hyperparameters: Dictionary of hyperparameters for each model type
            pretrain_epochs: Number of pre-training epochs
            finetune_epochs: Number of fine-tuning epochs
        """
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
        print(f"Approach: Transfer Learning")
        print(f"Pre-train epochs: {pretrain_epochs}")
        print(f"Fine-tune epochs: {finetune_epochs}")
        
        comparison_results = {}
        
        for patient_id in patient_ids:
            if patient_id not in self.patient_data:
                print(f"⚠️  Skipping patient {patient_id} - data not loaded")
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
                    
                    # Save model
                    model_file = self.models_dir / f"{model.model_name}.pth"
                    model.save_model(model_file)
                    results['model_file'] = str(model_file)
                    
                    patient_results[model_name] = results
                    
                    # Print results
                    print(f"\n{model_name} Results:")
                    print(f"  MAE: {results['mae']:.3f} mg/dL")
                    print(f"  RMSE: {results['rmse']:.3f} mg/dL")
                    print(f"  MARD: {results['mard']:.2f}%")
                    print(f"  TIR: {results['tir']:.1f}%")
                    print(f"  Clarke A+B: {results['clarke_zones']['A'] + results['clarke_zones']['B']:.1f}%")
                    print(f"  Parkes A+B: {results['parkes_zones']['A'] + results['parkes_zones']['B']:.1f}%")

                except Exception as e:
                    print(f"✗ Error with {model_name}: {e}")
                    continue
            
            comparison_results[patient_id] = patient_results
        
        # Save comprehensive results
        self.experiment_results = comparison_results
        results_file = self.results_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types for JSON serialization
        json_results = self._prepare_results_for_json(comparison_results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
        return comparison_results
    
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
                all_results.append({
                    'patient_id': patient_id,
                    'model': model_name,
                    'mae': results['mae'],
                    'rmse': results['rmse'],
                    'mard': results['mard'],
                    'tir': results['tir'],
                    'clarke_ab': results['clarke_zones']['A'] + results['clarke_zones']['B']
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
                  f"{row['mard']:<8.1f} {row['tir']:<8.1f} {row['clarke_ab']:<12.1f}")
        
        # Overall averages
        print("\nOverall Model Performance (Average across patients):")
        print("-" * 60)
        model_avg = summary_df.groupby('model').agg({
            'mae': 'mean',
            'rmse': 'mean', 
            'mard': 'mean',
            'tir': 'mean',
            'clarke_ab': 'mean'
        }).round(2)
        
        print(model_avg)
        
        # Best performing model
        best_model = model_avg.loc[model_avg['mae'].idxmin()]
        print(f"\n🏆 Best Overall Model (lowest MAE): {best_model.name}")
        print(f"   MAE: {best_model['mae']:.2f} mg/dL")
        print(f"   MARD: {best_model['mard']:.1f}%")
        print(f"   TIR: {best_model['tir']:.1f}%")
        
    def debug_dataset(self, patient_id: int):
        """Debug dataset to check for shape issues."""
        if patient_id not in self.patient_data:
            print(f"Patient {patient_id} not loaded")
            return
        
        data = self.patient_data[patient_id]
        train_dataset = data['train_dataset']
        
        print(f"\nDebugging dataset for patient {patient_id}...")
        
        # Check first 10 samples
        empty_targets = 0
        valid_targets = 0
        
        for i in range(min(100, len(train_dataset))):
            try:
                sequence = train_dataset[i]
                target = train_dataset.get_target(i)
                
                if target is None or len(target) == 0:
                    empty_targets += 1
                else:
                    valid_targets += 1
                    
                if i < 5:  # Print first 5 for inspection
                    print(f"Sample {i}: seq_shape={sequence.shape}, target_shape={target.shape if target is not None else 'None'}")
                    
            except Exception as e:
                print(f"Error at index {i}: {e}")
        
        print(f"Found {valid_targets} valid targets, {empty_targets} empty targets")
        return valid_targets, empty_targets


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description="Train and evaluate BG forecasting models")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--models-dir", default="saved_models", help="Models directory")
    parser.add_argument("--patients", nargs="+", type=int, default=[540, 544,552,567,584,596], help="Patient IDs to use (minimum 2 required for transfer learning)")
    parser.add_argument("--sequence-length", type=int, default=12, help="Input sequence length")
    parser.add_argument("--prediction-horizon", type=int, default=6, help="Prediction horizon")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--unimodal", action="store_true", help="Use only glucose features")
    parser.add_argument("--version", default="2020", choices=["2018", "2020"], help="Dataset version")
    parser.add_argument("--pretrain-epochs", type=int, default=50, help="Pre-training epochs")
    parser.add_argument("--finetune-epochs", type=int, default=25, help="Fine-tuning epochs")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BGModelTrainer(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir
    )
    
    try:
        # Load and prepare data
        trainer.load_and_prepare_data(
            patient_ids=args.patients,
            version=args.version,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            unimodal=args.unimodal,
            batch_size=args.batch_size
        )
        
        # Run model comparison
        trainer.run_model_comparison(
            patient_ids=args.patients,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs
        )
        
        # Print summary
        trainer.print_summary()
        
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved in: {trainer.results_dir}")
        print(f"Models saved in: {trainer.models_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
