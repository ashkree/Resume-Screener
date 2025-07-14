# utils/data_loader.py
# Simple, clean data loader for your resume screening experiments

import pickle
import os

def load_train_val_data(data_dir="../data/processed/"):
    """
    Load training and validation data for model experiments
    
    Args:
        data_dir: Directory containing the processed data files
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    print("📥 Loading train/val data...")
    
    # Load features
    with open(f"{data_dir}/X_train.pkl", 'rb') as f:
        X_train = pickle.load(f)
    
    with open(f"{data_dir}/X_val.pkl", 'rb') as f:
        X_val = pickle.load(f)
    
    # Load labels
    with open(f"{data_dir}/y_train.pkl", 'rb') as f:
        y_train = pickle.load(f)
    
    with open(f"{data_dir}/y_val.pkl", 'rb') as f:
        y_val = pickle.load(f)
    
    print(f"✅ Data loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   y_train: {len(y_train)} samples")
    print(f"   y_val: {len(y_val)} samples")
    
    return X_train, X_val, y_train, y_val

def load_test_data(data_dir="../data/processed/"):
    """
    Load test data for FINAL EVALUATION ONLY
    
    Args:
        data_dir: Directory containing the processed data files
        
    Returns:
        X_test, y_test
    """
    print("🚨 Loading TEST data - use only for final evaluation!")
    
    # Check if test files exist
    test_files = [f"{data_dir}/X_test.pkl", f"{data_dir}/y_test.pkl"]
    missing = [f for f in test_files if not os.path.exists(f)]
    
    if missing:
        raise FileNotFoundError(f"Test files not found: {missing}")
    
    # Load test data
    with open(f"{data_dir}/X_test.pkl", 'rb') as f:
        X_test = pickle.load(f)
    
    with open(f"{data_dir}/y_test.pkl", 'rb') as f:
        y_test = pickle.load(f)
    
    print(f"✅ Test data loaded:")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_test: {len(y_test)} samples")
    
    return X_test, y_test