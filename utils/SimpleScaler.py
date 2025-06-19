# Add this to your notebook or create a separate utils/scaling.py file

from sklearn.preprocessing import StandardScaler
import numpy as np

class SimpleScaler:
    """
    Simple scaling utility for experiments
    Use this when you want to scale features for Logistic Regression
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit_transform(self, X_train):
        """Fit scaler on training data and transform"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.fitted = True
        print(f"📏 Scaler fitted on {X_train.shape} training data")
        return X_train_scaled
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted first")
        return self.scaler.transform(X)
    
    def fit_transform_split(self, X_train, X_val):
        """Convenience method to scale train/val split"""
        X_train_scaled = self.fit_transform(X_train)
        X_val_scaled = self.transform(X_val)
        
        print(f"✅ Scaling completed:")
        print(f"   Train: {X_train_scaled.shape} (mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f})")
        print(f"   Val: {X_val_scaled.shape} (mean: {X_val_scaled.mean():.3f}, std: {X_val_scaled.std():.3f})")
        
        return X_train_scaled, X_val_scaled

# Usage examples for your notebook:
"""
def experiment_with_scaling():

    
    # Option 1: No scaling (for comparison)
    print("🧪 Experiment 1: No Scaling")
    lr_no_scale = LogisticRegressionModel()
    # Train directly with X_train, X_val
    
    # Option 2: With scaling  
    print("🧪 Experiment 2: With Scaling")
    scaler = SimpleScaler()
    X_train_scaled, X_val_scaled = scaler.fit_transform_split(X_train, X_val)
    
    lr_scaled = LogisticRegressionModel()
    # Train with X_train_scaled, X_val_scaled
    
    return lr_no_scale, lr_scaled

def quick_scaling_comparison(X_train, X_val, y_train, y_val, trainer):

    
    results = {}
    
    # Test without scaling
    print("🧪 Testing WITHOUT scaling...")
    lr_unscaled = LogisticRegressionModel()
    lr_unscaled.name = "LR_Unscaled"
    
    results['unscaled'] = trainer.train_model(
        model=lr_unscaled,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimize=False
    )
    
    # Test with scaling
    print("🧪 Testing WITH scaling...")
    scaler = SimpleScaler()
    X_train_scaled, X_val_scaled = scaler.fit_transform_split(X_train, X_val)
    
    lr_scaled = LogisticRegressionModel()
    lr_scaled.name = "LR_Scaled"
    
    results['scaled'] = trainer.train_model(
        model=lr_scaled,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        optimize=False
    )
    
    # Compare results
    print(f"\n📊 SCALING COMPARISON:")
    print(f"   Unscaled: {results['unscaled']['val_accuracy']:.4f}")
    print(f"   Scaled:   {results['scaled']['val_accuracy']:.4f}")
    print(f"   Improvement: {results['scaled']['val_accuracy'] - results['unscaled']['val_accuracy']:+.4f}")
    
    return results, scaler

# Parameter spaces for Logistic Regression experiments

# Anti-overfitting space (for high-dimensional data like TF-IDF)
lr_anti_overfitting_space = {
    'C': (0.001, 1.0),              # Strong to moderate regularization
    'penalty': ['l1', 'l2'],        # L1 for feature selection
    'solver': ['liblinear', 'saga'], # Supports both penalties
    'max_iter': [1000, 2000, 5000],
    'class_weight': ['balanced', None]
}

# High-performance space 
lr_high_performance_space = {
    'C': (0.01, 100.0),             # Wider range
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga'],
    'l1_ratio': [0.1, 0.5, 0.9],    # For elasticnet
    'max_iter': [2000, 5000],
    'class_weight': ['balanced', None]
}

# Feature selection space (emphasizes L1)
lr_feature_selection_space = {
    'C': (0.001, 10.0),
    'penalty': ['l1'],              # L1 only for sparsity
    'solver': ['liblinear', 'saga'],
    'max_iter': [2000, 5000],
    'class_weight': ['balanced']
}

print("✅ Minimal LogisticRegressionModel and scaling utilities ready!")
print("📝 Copy the parameter spaces above for your experiments")
"""