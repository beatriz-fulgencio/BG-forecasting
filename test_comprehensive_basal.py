#!/usr/bin/env python3
"""
Comprehensive test for basal preprocessing with temp_basal events
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Add the benchmark directory to the path
sys.path.append('/Users/beatrizfcmenezes/Documents/GitHub/BG-forecasting')

def test_temp_basal():
    """Test temporary basal functionality"""
    try:
        from benchmark.data.preprocessors import OhioBGDataPreprocessor
        print("=== Testing Temporary Basal Functionality ===\n")
        
        # Create test data with temp basal event
        times = [datetime(2023, 1, 1, 0, 0, 0) + timedelta(minutes=5*i) for i in range(20)]
        df = pd.DataFrame({
            'index': times,
            'glucose': [120] * 20,
            'basal': [1.0] + [-1] * 19,  # 1.0 U/hr regular rate at start
            'temp_basal': [-1] * 20,
            'basal_end': [-1] * 20,
        })
        
        # Add temp basal event from position 5 to position 10 (25 minutes duration)
        df.loc[5, 'temp_basal'] = 2.5  # 2.5 U/hr temp rate
        df.loc[5, 'basal_end'] = times[10]  # Ends at position 10
        
        # Add another regular basal rate change at position 15
        df.loc[15, 'basal'] = 0.75  # 0.75 U/hr new rate
        
        df.set_index('index', inplace=True)
        
        print("Original data structure:")
        for i in range(20):
            basal = df.iloc[i]['basal']
            temp_basal = df.iloc[i]['temp_basal'] 
            basal_end = df.iloc[i]['basal_end']
            print(f"Row {i:2d}: basal={basal:4.1f}, temp_basal={temp_basal:4.1f}, basal_end={basal_end}")
        
        # Process the data
        preprocessor = OhioBGDataPreprocessor(sampling_rate=5)
        result = preprocessor.basic_preprocessing(df)
        
        print("\\nProcessed basal values:")
        if 'basal' in result.columns:
            basal_values = result['basal'].values
            for i, val in enumerate(basal_values):
                print(f"Row {i:2d}: {val:.4f}")
            
            print("\\nExpected behavior verification:")
            print("- Rows 0-4:  1.0 U/hr regular rate → 0.0833")
            print("- Rows 5-10: 2.5 U/hr temp rate   → 0.2083") 
            print("- Rows 11-14: 1.0 U/hr regular rate → 0.0833 (resume)")
            print("- Rows 15-19: 0.75 U/hr new rate   → 0.0625")
            
            # Verify key values
            expected_regular = 1.0 * (5/60)    # 0.0833
            expected_temp = 2.5 * (5/60)       # 0.2083  
            expected_new = 0.75 * (5/60)       # 0.0625
            
            print(f"\\nActual vs Expected:")
            print(f"Regular rate: {basal_values[2]:.4f} vs {expected_regular:.4f}")
            print(f"Temp rate:    {basal_values[7]:.4f} vs {expected_temp:.4f}")
            print(f"Resume rate:  {basal_values[12]:.4f} vs {expected_regular:.4f}")
            print(f"New rate:     {basal_values[17]:.4f} vs {expected_new:.4f}")
            
            # Check if values are correct
            tolerance = 0.0001
            checks = [
                (abs(basal_values[2] - expected_regular) < tolerance, "Regular rate"),
                (abs(basal_values[7] - expected_temp) < tolerance, "Temp rate"),
                (abs(basal_values[12] - expected_regular) < tolerance, "Resume rate"),
                (abs(basal_values[17] - expected_new) < tolerance, "New rate")
            ]
            
            print("\\nValidation Results:")
            all_passed = True
            for passed, description in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{status}: {description}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print("\\n🎉 All tests PASSED! Basal preprocessing is working correctly.")
            else:
                print("\\n❌ Some tests FAILED. Please check the implementation.")
                
        else:
            print("✗ No basal column found in processed data")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_temp_basal()
