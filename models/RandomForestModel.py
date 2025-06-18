from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
import numpy as np

from .BaseModel import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest model with Bayesian optimization support
    """
    
    def __init__(self, **kwargs):
        super().__init__("Random Forest", **kwargs)
        
    def _create_model(self, **params) -> RandomForestClassifier:
        """Create Random Forest model with specified parameters"""
        default_params = {
            'n_estimators': kwargs.get('n_estimators', 200),
            'max_depth':   kwargs.get('max_depth', 10),          # cap depth
            'min_samples_leaf': kwargs.get('min_samples_leaf', 5),
            'max_features':     kwargs.get('max_features', 'sqrt'),
            'bootstrap':        True,
            'oob_score':        True,                            # enable OOB
            'random_state':     kwargs.get('random_state', 42),
        }
        
        # Update with provided parameters
        default_params.update(params)
        default_params.update(self.config)
        
        return RandomForestClassifier(**default_params)
    
    def get_param_space(self) -> Dict[str, Any]:
        """Define parameter space for Bayesian optimization"""
        return {
            'n_estimators': (50, 300),           # Integer range
            'max_depth': (3, 30),                # Integer range (None handled separately)
            'min_samples_split': (2, 20),        # Integer range
            'min_samples_leaf': (1, 10),         # Integer range
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],  # Categorical + float
        }
    
    def get_detailed_importance(self):
        """Get detailed feature importance analysis"""
        if not self.is_fitted:
            return None
            
        importance = self.get_feature_importance()
        if importance is None:
            return None
            
        return {
            'importance_scores': importance,
            'mean_importance': importance.mean(),
            'std_importance': importance.std(),
            'top_features_count': np.sum(importance > importance.mean()),
            'importance_concentration': np.sum(importance[:10]) / np.sum(importance)  # Top 10 concentration
        }

