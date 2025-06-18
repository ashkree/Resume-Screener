from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import time
from sklearn.base import BaseEstimator

class BaseModel(ABC):
    """
    Abstract base class for all ML models in the resume screening system
    Simple model definition - all parameter space logic handled externally
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.config = kwargs  # Only sklearn parameters
        
    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the specific model instance with given parameters"""
        pass
    
    def create_model_with_params(self, **params) -> BaseEstimator:
        """
        Create model instance with specific parameters
        Used by ModelTrainer for optimization
        """
        return self._create_model(**params)
    
    def fit(self, X, y, **fit_params):
        """
        Fit the model to training data
        
        Args:
            X: Training features
            y: Training labels
            **fit_params: Additional fitting parameters
            
        Returns:
            self: Returns fitted model
        """
        if self.model is None:
            self.model = self._create_model()
            
        print(f"🔧 Fitting {self.name}...")
        start_time = time.time()
        
        self.model.fit(X, y, **fit_params)
        
        fit_time = time.time() - start_time
        self.is_fitted = True
        
        print(f"✅ {self.name} fitted in {fit_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
            
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.name} does not support probability prediction")
            
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if not self.is_fitted:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
        else:
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get basic model information"""
        return {
            'name': self.name,
            'fitted': self.is_fitted,
            'config': self.config
        }