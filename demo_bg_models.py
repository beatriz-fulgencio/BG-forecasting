#!/usr/bin/env python3
"""
Simple demo script to test RNN models with synthetic blood glucose data.

This script demonstrates:
1. Creating synthetic BG time series data
2. Setting up PyTorch datasets and loaders
3. Training RNN/LSTM/GRU models
4. Evaluating with BG-specific metrics
5. Comparing model performance

Run this script to verify your models work correctly before using real data.
"""

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from datetime import datetime

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


def generate_synthetic_bg_data(n_sequences=1000, sequence_length=12, prediction_horizon=6, 
                              feature_dim=1, noise_level=0.1):
    """
    Generate synthetic blood glucose time series data for testing.
    
    Args:
        n_sequences: Number of sequences to generate
        sequence_length: Length of input sequences
        prediction_horizon: Number of future steps to predict
        feature_dim: Number of features (1=glucose only, 4=glucose+insulin+carbs+basal)
        noise_level: Amount of random noise to add
    
    Returns:
        Tuple of (X, y) where X is input sequences and y is targets
    """
    print(f"Generating {n_sequences} synthetic sequences...")
    
    # Generate realistic BG patterns
    sequences = []
    targets = []
    
    for i in range(n_sequences):
        # Generate base pattern (sinusoidal with trend)
        t = np.linspace(0, 4*np.pi, sequence_length + prediction_horizon)
        
        # Base glucose pattern (70-180 mg/dL range)
        base_bg = 125 + 30 * np.sin(t) + 20 * np.sin(0.5 * t)
        
        # Add meal spikes randomly
        if np.random.random() > 0.5:
            spike_start = np.random.randint(0, len(t)//2)
            spike_height = np.random.uniform(30, 80)
            for j in range(spike_start, min(spike_start + 6, len(t))):
                decay = np.exp(-(j - spike_start) * 0.3)
                base_bg[j] += spike_height * decay
        
        # Add noise
        bg_values = base_bg + np.random.normal(0, noise_level * 10, len(base_bg))
        
        # Clip to realistic range
        bg_values = np.clip(bg_values, 50, 300)
        
        if feature_dim == 1:
            # Glucose only
            sequence = bg_values[:sequence_length].reshape(-1, 1)
        else:
            # Multi-modal: glucose, basal insulin, bolus insulin, carbs
            glucose = bg_values[:sequence_length]
            basal = np.random.uniform(0.5, 2.0, sequence_length)  # Basal insulin
            bolus = np.zeros(sequence_length)  # Bolus insulin (sparse)
            carbs = np.zeros(sequence_length)  # Carbohydrates (sparse)
            
            # Add occasional bolus and carbs
            if np.random.random() > 0.7:
                bolus_time = np.random.randint(0, sequence_length)
                bolus[bolus_time] = np.random.uniform(2, 10)
            
            if np.random.random() > 0.6:
                carb_time = np.random.randint(0, sequence_length)
                carbs[carb_time] = np.random.uniform(20, 60)
            
            sequence = np.column_stack([glucose, basal, bolus, carbs])
        
        # Target is the next prediction_horizon glucose values
        target = bg_values[sequence_length:sequence_length + prediction_horizon]
        if prediction_horizon == 1:
            target = target[0]  # Single value for single-step prediction
        
        sequences.append(sequence)
        targets.append(target)
    
    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    
    print(f"✓ Generated data shapes: X={X.shape}, y={y.shape}")
    return X, y


def create_data_loaders(X, y, train_split=0.7, val_split=0.2, batch_size=32):
    """
    Create train/validation/test data loaders.
    
    Args:
        X: Input sequences
        y: Target values
        train_split: Fraction for training
        val_split: Fraction for validation (remaining goes to test)
        batch_size: Batch size for loaders
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_samples = len(X)
    n_train = int(train_split * n_samples)
    n_val = int(val_split * n_samples)
    
    # Split data
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create tensors
    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(y[train_idx])
    X_val = torch.FloatTensor(X[val_idx])
    y_val = torch.FloatTensor(y[val_idx])
    X_test = torch.FloatTensor(X[test_idx])
    y_test = torch.FloatTensor(y[test_idx])
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Data split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


def test_model(model_class, model_name, train_loader, val_loader, test_loader, 
               sequence_length, prediction_horizon, feature_dim, hyperparameters, epochs=20):
    """
    Train and evaluate a single model.
    
    Args:
        model_class: Model class to test
        model_name: Name of the model
        train_loader, val_loader, test_loader: Data loaders
        sequence_length: Input sequence length
        prediction_horizon: Prediction horizon
        feature_dim: Number of input features
        hyperparameters: Model hyperparameters
        epochs: Number of training epochs
    
    Returns:
        Dictionary with model, predictions, and metrics
    """
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    # Create model
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
        early_stopping_patience=10
    )
    
    print(f"✓ Training completed in {history['epochs_completed']} epochs")
    print(f"  Final train loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        print(f"  Final val loss: {history['val_losses'][-1]:.6f}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(test_loader)
    
    # Extract true values
    true_values = []
    for batch_X, batch_y in test_loader:
        true_values.append(batch_y.numpy())
    y_true = np.concatenate(true_values)
    
    # Handle shape issues - for multi-step prediction, evaluate only the first step
    if prediction_horizon == 1:
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true.flatten()
    else:
        # For multi-step, evaluate only the first prediction step
        if predictions.ndim == 2:
            predictions = predictions[:, 0]  # Take first step
        if y_true.ndim == 2:
            y_true = y_true[:, 0]  # Take first step
    
    print(f"✓ Predictions shape: {predictions.shape}, True values shape: {y_true.shape}")
    
    # Evaluate with BG metrics
    evaluator = BGEvaluator()
    results = evaluator.compute_metrics(
        y_true=y_true,
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
        'predictions': predictions,
        'true_values': y_true,
        'metrics': results,
        'training_history': history
    }


def run_demo():
    """Run the complete demo."""
    print("🩺 Blood Glucose Forecasting Models Demo")
    print("=" * 60)
    
    # Configuration
    sequence_length = 12  # 1 hour of history (5-minute intervals)
    prediction_horizon = 1  # Single step prediction to avoid complexity
    feature_dim = 1  # Start with glucose-only
    n_sequences = 1000  # Fewer sequences for faster demo
    epochs = 15  # Fewer epochs for faster demo
    batch_size = 32
    
    print(f"Configuration:")
    print(f"  Sequence length: {sequence_length} (input history)")
    print(f"  Prediction horizon: {prediction_horizon} (future steps)")
    print(f"  Feature dimension: {feature_dim} ({'glucose-only' if feature_dim == 1 else 'multimodal'})")
    print(f"  Training sequences: {n_sequences}")
    print(f"  Training epochs: {epochs}")
    
    # Generate synthetic data
    X, y = generate_synthetic_bg_data(
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        feature_dim=feature_dim
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X, y, batch_size=batch_size)
    
    # Define models to test
    models_to_test = {
        'RNN': (RNNBGModel, {
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'nonlinearity': 'tanh'
        }),
        'LSTM': (LSTMBGModel, {
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        }),
        'GRU': (GRUBGModel, {
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001
        })
    }
    
    # Test all models
    results = {}
    
    for model_name, (model_class, hyperparams) in models_to_test.items():
        try:
            result = test_model(
                model_class=model_class,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                feature_dim=feature_dim,
                hyperparameters=hyperparams,
                epochs=epochs
            )
            results[model_name] = result
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        
        print(f"{'Model':<8} {'MAE':<8} {'RMSE':<8} {'MARD':<8} {'TIR':<8} {'Clarke A+B':<12}")
        print("-" * 60)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            clarke_ab = (metrics.get('clarke_zones', {}).get('A', 0) + 
                        metrics.get('clarke_zones', {}).get('B', 0))
            
            print(f"{model_name:<8} {metrics['mae']:<8.2f} {metrics['rmse']:<8.2f} "
                  f"{metrics['mard']:<8.1f} {metrics['tir']:<8.1f} {clarke_ab:<12.1f}")
        
        # Find best model (lowest MAE)
        best_model = min(results.items(), key=lambda x: x[1]['metrics']['mae'])
        print(f"\n🏆 Best Model: {best_model[0]} (MAE: {best_model[1]['metrics']['mae']:.2f} mg/dL)")
        
        # Save results
        results_dir = Path("demo_results")
        results_dir.mkdir(exist_ok=True)
        
        # Convert results for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'metrics': {k: (float(v) if isinstance(v, np.number) else v) 
                           for k, v in result['metrics'].items()},
                'training_epochs': result['training_history']['epochs_completed'],
                'final_train_loss': float(result['training_history']['train_losses'][-1]),
                'final_val_loss': float(result['training_history']['val_losses'][-1]) if result['training_history']['val_losses'] else None
            }
        
        results_file = results_dir / f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Demo results saved to {results_file}")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETED!")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Run with real OhioT1DM data using run_bg_models.py")
    print("2. Try different hyperparameters")
    print("3. Experiment with multimodal features (set feature_dim=4)")
    print("4. Test different sequence lengths and prediction horizons")


if __name__ == "__main__":
    run_demo()
