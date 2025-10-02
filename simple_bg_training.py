#!/usr/bin/env python3
"""
Simple training script using preprocessed CSV files.

This script loads preprocessed CSV files and trains models directly
without using the complex OhioDataset pipeline.
"""

import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add benchmark to path
sys.path.append('benchmark')

# Import our modules
try:
    from models.rnn import RNNBGModel, LSTMBGModel, GRUBGModel
    from evaluation.evaluator import BGEvaluator
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_sequences(data, sequence_length=12, prediction_horizon=1):
    """
    Create input sequences and targets from time series data.
    
    Args:
        data: DataFrame with glucose and other features
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict ahead
        
    Returns:
        X: Input sequences array (n_sequences, sequence_length, n_features)
        y: Target values array (n_sequences, prediction_horizon)
    """
    # Use glucose column as primary feature
    glucose = data['glucose'].values
    
    # Create feature matrix (can add more features here)
    if 'basal' in data.columns:
        features = data[['glucose', 'basal', 'bolus', 'carbs']].fillna(0).values
    else:
        features = glucose.reshape(-1, 1)
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(features) - sequence_length - prediction_horizon + 1):
        # Input sequence
        seq = features[i:i + sequence_length]
        
        # Target (next glucose values)
        if prediction_horizon == 1:
            target = glucose[i + sequence_length]
        else:
            target = glucose[i + sequence_length:i + sequence_length + prediction_horizon]
        
        sequences.append(seq)
        targets.append(target)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, 
                            sequence_length, prediction_horizon, hyperparameters, epochs=10):
    """
    Train and evaluate a single model.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    # Split training into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    feature_dim = X_train.shape[2]
    model = model_class(
        model_name=model_name,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        feature_dim=feature_dim,
        hyperparameters=hyperparameters
    )
    
    print(f"Model: {model}")
    
    # Train model
    print("Training...")
    history = model.fit(
        train_loader=train_loader,
        validation_loader=val_loader,
        epochs=epochs,
        learning_rate=hyperparameters.get('learning_rate', 0.001),
        early_stopping_patience=max(5, epochs//5)
    )
    
    print(f"✓ Training completed in {history['epochs_completed']} epochs")
    print(f"  Final train loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"  Final val loss: {history['val_losses'][-1]:.4f}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_loader)
    
    # Handle shapes
    if prediction_horizon == 1:
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
    
    print(f"✓ Predictions shape: {predictions.shape}, True values shape: {y_test.shape}")
    
    # Evaluate
    evaluator = BGEvaluator()
    try:
        results = evaluator.compute_metrics(
            y_true=y_test,
            y_pred=predictions,
            metrics=['mae', 'rmse', 'mard', 'tir', 'tbr', 'tar', 'clarke']
        )
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  MAE: {results['mae']:.3f} mg/dL")
        print(f"  RMSE: {results['rmse']:.3f} mg/dL")
        print(f"  MARD: {results['mard']:.2f}%")
        print(f"  TIR (true): {results['tir']:.1f}%")
        print(f"  TIR (pred): {results['tir_pred']:.1f}%")
        
        if 'clarke_zones' in results:
            clarke = results['clarke_zones']
            print(f"  Clarke Error Grid:")
            print(f"    Zone A (acceptable): {clarke['A']:.1f}%")
            print(f"    Zone B (benign): {clarke['B']:.1f}%")
            print(f"    Zone A+B: {clarke['A'] + clarke['B']:.1f}%")
        
        return {
            'model': model,
            'metrics': results,
            'training_history': history,
            'predictions': predictions,
            'true_values': y_test
        }
    
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Simple BG model training with CSV data")
    parser.add_argument("--patients", nargs="+", type=int, default=[540], help="Patient IDs")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--sequence-length", type=int, default=12, help="Sequence length")
    parser.add_argument("--prediction-horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--data-dir", default="processed_data", help="Processed data directory")
    
    args = parser.parse_args()
    
    # Model configurations
    models_to_test = {
        'RNN': (RNNBGModel, {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'nonlinearity': 'tanh'
        }),
        'LSTM': (LSTMBGModel, {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        }),
        'GRU': (GRUBGModel, {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        })
    }
    
    all_results = {}
    
    # Process each patient
    for patient_id in args.patients:
        print(f"\n{'='*60}")
        print(f"PATIENT {patient_id}")
        print(f"{'='*60}")
        
        # Load data
        train_file = f"{args.data_dir}/patient_{patient_id}_train_processed.csv"
        test_file = f"{args.data_dir}/patient_{patient_id}_test_processed.csv"
        
        if not Path(train_file).exists() or not Path(test_file).exists():
            print(f"⚠️  Data files not found for patient {patient_id}")
            continue
        
        print(f"Loading data from {train_file} and {test_file}")
        train_data = pd.read_csv(train_file, index_col=0, parse_dates=True)
        test_data = pd.read_csv(test_file, index_col=0, parse_dates=True)
        
        print(f"Train data: {len(train_data)} rows, Test data: {len(test_data)} rows")
        
        # Create sequences
        X_train, y_train = create_sequences(
            train_data, 
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
        
        X_test, y_test = create_sequences(
            test_data,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
        
        patient_results = {}
        
        # Test each model
        for model_name, (model_class, hyperparams) in models_to_test.items():
            try:
                result = train_and_evaluate_model(
                    model_class=model_class,
                    model_name=f"{model_name}_patient_{patient_id}",
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    sequence_length=args.sequence_length,
                    prediction_horizon=args.prediction_horizon,
                    hyperparameters=hyperparams,
                    epochs=args.epochs
                )
                
                if result:
                    patient_results[model_name] = result
                    
                    # Save model
                    model_file = f"saved_models/{result['model'].model_name}.pth"
                    Path("saved_models").mkdir(exist_ok=True)
                    result['model'].save_model(model_file)
                    print(f"✓ Model saved to {model_file}")
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[patient_id] = patient_results
    
    # Print summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        summary_data = []
        for patient_id, patient_results in all_results.items():
            for model_name, result in patient_results.items():
                if result and 'metrics' in result:
                    metrics = result['metrics']
                    summary_data.append({
                        'Patient': patient_id,
                        'Model': model_name,
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'MARD': metrics['mard'],
                        'TIR': metrics['tir'],
                        'Clarke A+B': metrics.get('clarke_zones', {}).get('A', 0) + 
                                    metrics.get('clarke_zones', {}).get('B', 0)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\nResults Summary:")
            print(summary_df.round(2))
            
            # Find best model
            best_row = summary_df.loc[summary_df['MAE'].idxmin()]
            print(f"\n🏆 Best Model: {best_row['Model']} (Patient {best_row['Patient']})")
            print(f"   MAE: {best_row['MAE']:.2f} mg/dL")
            print(f"   MARD: {best_row['MARD']:.1f}%")
            print(f"   Clarke A+B: {best_row['Clarke A+B']:.1f}%")
        
        # Save results
        Path("results").mkdir(exist_ok=True)
        results_file = f"results/simple_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert for JSON serialization
        json_results = {}
        for patient_id, patient_results in all_results.items():
            json_results[str(patient_id)] = {}
            for model_name, result in patient_results.items():
                if result:
                    json_results[str(patient_id)][model_name] = {
                        'metrics': {k: float(v) if isinstance(v, np.number) else v 
                                  for k, v in result['metrics'].items()},
                        'final_train_loss': float(result['training_history']['train_losses'][-1])
                    }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
