#!/usr/bin/env python3
"""
Script to inspect the actual model files and understand their structure
"""

import h5py
import numpy as np
import os

def inspect_h5_file(file_path):
    """Inspect the structure of an H5 file."""
    print(f"\nInspecting: {file_path}")
    print("-" * 50)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("File structure:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset {name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"  Group {name}/")
            
            f.visititems(print_structure)
            
            # Check if it's a weights file or full model
            if 'model_config' in f.keys():
                print("This is a full model file")
            elif 'layer_names' in f.keys():
                print("This is a weights file")
            else:
                print("Unknown file format")
                
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    """Main inspection function."""
    print("Model File Inspector")
    print("=" * 50)
    
    model_files = [
        "models/bci2a_crdae_gtaa_best_weights.weights.h5",
        "models/best_eeg_pd_model.h5"
    ]
    
    for model_file in model_files:
        inspect_h5_file(model_file)

if __name__ == "__main__":
    main()
