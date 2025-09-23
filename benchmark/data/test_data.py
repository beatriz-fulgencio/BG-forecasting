"""Test data pipeline for model training with PyTorch."""

import os
import sys
from pathlib import Path

from loaders import load_ohiot1dm_data
from preprocessors import preprocess_ohiot1dm_data

try:
    import torch
    from torch.utils.data import DataLoader
    from torch_dataset import (
        OhioDataset, 
        prepare_patient_datasets,
        prepare_multi_patient_dataset
    )
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    TORCH_AVAILABLE = False


def example_single_patient_training():
    """
    Example: Prepare datasets for training on a single patient.
    """
    if not TORCH_AVAILABLE:
        return
        
    print("="*60)
    print("SINGLE PATIENT PYTORCH DATASET EXAMPLE")
    print("="*60)
    
    # Configuration
    DATA_DIR = "../../data"
    PATIENT_ID = 540 # Example patient
    SEQUENCE_LENGTH = 12  # 1 hour of history at 5-min intervals
    PREDICTION_HORIZON = 6  # 30 minutes ahead
    
    print(f"Loading data for patient {PATIENT_ID}...")
    
    # Step 1: Load and preprocess data using your existing pipeline
    train_data = load_ohiot1dm_data(
        data_dir=DATA_DIR,
        patient_ids=[PATIENT_ID],
        mode='train',
        version='2020'
    )
    
    test_data = load_ohiot1dm_data(
        data_dir=DATA_DIR,
        patient_ids=[PATIENT_ID], 
        mode='test',
        version='2020'
    )
    
    # Preprocess the data
    preprocessed_train = preprocess_ohiot1dm_data(
        train_data,
        include_feature_engineering=True,
        normalize=False  # We'll let PyTorch dataset handle normalization
    )
    
    preprocessed_test = preprocess_ohiot1dm_data(
        test_data,
        include_feature_engineering=True,
        normalize=False
    )
    
    print(f"Train data shape: {preprocessed_train[PATIENT_ID].shape}")
    print(f"Test data shape: {preprocessed_test[PATIENT_ID].shape}")
    
    print(f"Train data columns: {preprocessed_train[PATIENT_ID].columns}")
    print("First few rows of train data:")
    print(preprocessed_train[PATIENT_ID].head())
    
    # Step 2: Create PyTorch datasets
    train_dataset, test_dataset = prepare_patient_datasets(
        train_df=preprocessed_train[PATIENT_ID],
        test_df=preprocessed_test[PATIENT_ID],
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        unimodal=False  # Use multimodal features
    )
    
    print(f"PyTorch train dataset: {len(train_dataset)} sequences")
    print(f"PyTorch test dataset: {len(test_dataset)} sequences")
    
    # Step 3: Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for performance
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print("\n✓ Single patient dataset ready for training!")


def example_multi_patient_training():
    """
    Example: Prepare datasets for multi-patient/transfer learning.
    """
    if not TORCH_AVAILABLE:
        return
        
    print("\n" + "="*60)
    print("MULTI-PATIENT PYTORCH DATASET EXAMPLE")
    print("="*60)
    
    # Configuration
    DATA_DIR = "../../data"
    PATIENT_IDS = [540, 544, 552, 567, 584, 596]  
    VERSION = '2020'
    TARGET_PATIENT = 540
    SEQUENCE_LENGTH = 12
    PREDICTION_HORIZON = 1  # Single-step for simplicity
    
    print("="*60)
    print(f"Loading data for patients {PATIENT_IDS}...")
    
    # Load and preprocess data for all patients
    patient_data = {}
    
    for patient_id in PATIENT_IDS:
        print(f"Processing patient {patient_id}...")
        
        # Load train and test data
        train_data = load_ohiot1dm_data(
            data_dir=DATA_DIR,
            patient_ids=[patient_id],
            mode='train',
            version= VERSION
        )
        
        test_data = load_ohiot1dm_data(
            data_dir=DATA_DIR,
            patient_ids=[patient_id],
            mode='test', 
            version=VERSION
        )
        
        # Preprocess
        preprocessed_train = preprocess_ohiot1dm_data(
            train_data,
            include_feature_engineering=False,  # Keep simple for example
            normalize=False
        )
        
        preprocessed_test = preprocess_ohiot1dm_data(
            test_data,
            include_feature_engineering=False,
            normalize=False
        )
        
        patient_data[patient_id] = {
            'train': preprocessed_train[patient_id],
            'test': preprocessed_test[patient_id]
        }
    
    # Create multi-patient dataset for transfer learning
    global_dataset, target_train, target_test = prepare_multi_patient_dataset(
        patient_data=patient_data,
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        target_patient_id=TARGET_PATIENT,
        unimodal=False
    )
    
    print(f"Global dataset (all other patients): {len(global_dataset)} sequences")
    print(f"Target patient train: {len(target_train)} sequences")
    print(f"Target patient test: {len(target_test)} sequences")
    
    # Create DataLoaders
    global_loader = DataLoader(global_dataset, batch_size=64, shuffle=True)
    target_train_loader = DataLoader(target_train, batch_size=32, shuffle=True)
    target_test_loader = DataLoader(target_test, batch_size=32, shuffle=False)
    
    print(f"Global training batches: {len(global_loader)}")
    print(f"Target training batches: {len(target_train_loader)}")
    print(f"Target test batches: {len(target_test_loader)}")
    
    print("\n✓ Multi-patient datasets ready for transfer learning!")


def main():
    """Run all examples."""
    try:
        example_single_patient_training()
        example_multi_patient_training() 
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED!")
        print("="*60)
        print("DATA PIPELINE TESTS PASSED -- ready for model training!")
        

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your data directory and files are set up correctly.")


if __name__ == "__main__":
    main()
