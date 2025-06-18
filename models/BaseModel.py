from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

class BaseModel(ABC):
    """
    Abstract base class for all ML models in the resume screening system
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.fit_time = 0.0
        self.predict_time = 0.0
        self.best_params = {}
        self.cv_scores = []
        self.config = kwargs
        
    @abstractmethod
    def _create_model(self, **params) -> BaseEstimator:
        """Create the specific model instance with given parameters"""
        pass
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        """Define parameter space for Bayesian optimization"""
        pass
    
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

        # Handle sparse matrix shape properly
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)

            
        print(f"🔧 Fitting {self.name}...")
        start_time = time.time()
        
        self.model.fit(X, y, **fit_params)
        
        self.fit_time = time.time() - start_time
        self.is_fitted = True
        
        print(f"✅ {self.name} fitted in {self.fit_time:.2f} seconds")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
            
        start_time = time.time()
        predictions = self.model.predict(X)
        self.predict_time = time.time() - start_time
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
            
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.name} does not support probability prediction")
            
        return self.model.predict_proba(X)
    
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels  
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Array of CV scores
        """
        if self.model is None:
            self.model = self._create_model()
            
        self.cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return self.cv_scores
    
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
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration and statistics"""
        return {
            'name': self.name,
            'fitted': self.is_fitted,
            'fit_time': self.fit_time,
            'predict_time': self.predict_time,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores.tolist() if len(self.cv_scores) > 0 else [],
            'cv_mean': self.cv_scores.mean() if len(self.cv_scores) > 0 else None,
            'cv_std': self.cv_scores.std() if len(self.cv_scores) > 0 else None,
            'config': self.config
        }
