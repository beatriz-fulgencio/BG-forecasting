#!/usr/bin/env python3
"""
Test script for basal preprocessing functionality
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the benchmark directory to the path
sys.path.append('/Users/beatrizfcmenezes/Documents/GitHub/BG-forecasting')

def test_simple():
    """Simple test of basal preprocessing"""
    try:
        from benchmark.data.preprocessors import OhioBGDataPreprocessor
        print("✓ Import successful")
        
        # Create simple test data
        times = [datetime(2023, 1, 1, 0, 0, 0) + timedelta(minutes=5*i) for i in range(10)]
        df = pd.DataFrame({
            'index': times,
            'glucose': [120] * 10,
            'basal': [1.0, -1, -1, -1, -1, 2.0, -1, -1, -1, -1],
            'temp_basal': [-1] * 10,
            'basal_end': [-1] * 10,
        })
        df.set_index('index', inplace=True)
        print("✓ Test data created")
        
        preprocessor = OhioBGDataPreprocessor(sampling_rate=5)
        result = preprocessor.basic_preprocessing(df)
        print("✓ Preprocessing completed")
        
        if 'basal' in result.columns:
            basal_values = result['basal'].values
            print(f"✓ Basal values: {basal_values[:5]}")
            
            # Expected: 1.0 U/hr = 1.0 * (5/60) = 0.0833 per 5-min
            # Expected: 2.0 U/hr = 2.0 * (5/60) = 0.1667 per 5-min
            expected_1 = 1.0 * (5/60)
            expected_2 = 2.0 * (5/60)
            
            print(f"Expected for 1.0 U/hr: {expected_1:.4f}")
            print(f"Expected for 2.0 U/hr: {expected_2:.4f}")
            print(f"Actual values: {basal_values}")
        else:
            print("✗ No basal column found")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()
