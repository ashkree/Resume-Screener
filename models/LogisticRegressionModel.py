from sklearn.linear_model import LogisticRegression
from typing import Dict, Any
import numpy as np

from .BaseModel import BaseModel

class LogisticRegressionModel(BaseModel):
    """
    Simple Logistic Regression model definition
    Parameter spaces handled externally by ModelTrainer
    Scaling handled externally for flexibility
    """
    
    def __init__(self, **kwargs):
        super().__init__("Logistic Regression", **kwargs)
        
    def _create_model(self, **params) -> LogisticRegression:
        """Create Logistic Regression model with specified parameters"""
        default_params = {
            'C': 0.1,                    # Strong regularization by default
            'penalty': 'l2',
            'solver': 'liblinear',
            'class_weight': 'balanced',   # Handle class imbalance
            'random_state': 42,
            'max_iter': 2000,
            'fit_intercept': True
        }
        
        # Merge: defaults <- config (from init) <- runtime params
        final_params = {**default_params, **self.config, **params}
        
        return LogisticRegression(**final_params)
    
    def get_coefficients(self):
        """Get model coefficients if available"""
        if not self.is_fitted:
            return None
        if hasattr(self.model, 'coef_'):
            return self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        return None
    
    def get_intercept(self):
        """Get model intercept if available"""
        if not self.is_fitted:
            return None
        if hasattr(self.model, 'intercept_'):
            return self.model.intercept_
        return None
    
    def get_detailed_coefficients(self):
        """Get detailed coefficient analysis"""
        if not self.is_fitted:
            return None
            
        coef = self.get_coefficients()
        if coef is None:
            return None
            
        return {
            'coefficients': coef,
            'intercept': self.get_intercept(),
            'positive_coef_count': np.sum(coef > 0),
            'negative_coef_count': np.sum(coef < 0),
            'coef_magnitude_mean': np.abs(coef).mean(),
            'coef_magnitude_std': np.abs(coef).std(),
            'max_positive_coef': coef.max(),
            'max_negative_coef': coef.min(),
            'sparsity': np.sum(np.abs(coef) < 0.001) / len(coef),
            'large_coefficients_count': np.sum(np.abs(coef) > 10),
            'regularization_strength': getattr(self.model, 'C', None)
        }
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Get Logistic Regression specific information"""
        info = self.get_model_info()
        
        if self.is_fitted:
            info.update({
                'regularization_C': getattr(self.model, 'C', None),
                'penalty': getattr(self.model, 'penalty', None),
                'solver': getattr(self.model, 'solver', None),
                'n_features': getattr(self.model, 'n_features_in_', None),
                'n_classes': getattr(self.model, 'classes_', None),
                'n_iter': getattr(self.model, 'n_iter_', None),
                'coefficient_analysis_available': self.get_coefficients() is not None
            })
        
        return info