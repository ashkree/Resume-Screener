from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
import numpy as np

from .BaseModel import BaseModel

class RandomForestModel(BaseModel):
    """
    Simple Random Forest model definition
    Parameter spaces handled externally by ModelTrainer
    """
    
    def __init__(self, **kwargs):
        super().__init__("Random Forest", **kwargs)
        
    def _create_model(self, **params) -> RandomForestClassifier:
        """Create Random Forest model with specified parameters"""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
        }
        
        # Merge: defaults <- config (from init) <- runtime params
        final_params = {**default_params, **self.config, **params}
        
        return RandomForestClassifier(**final_params)
    
    def get_oob_score(self):
        """Get out-of-bag score if available"""
        if not self.is_fitted:
            return None
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None
    
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
            'importance_concentration': np.sum(importance[:10]) / np.sum(importance),
            'oob_score': self.get_oob_score()
        }
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Get Random Forest specific information"""
        info = self.get_model_info()
        
        if self.is_fitted:
            info.update({
                'n_trees': getattr(self.model, 'n_estimators', None),
                'oob_score': self.get_oob_score(),
                'feature_importance_available': self.get_feature_importance() is not None,
                'n_features': getattr(self.model, 'n_features_in_', None),
                'n_classes': getattr(self.model, 'n_classes_', None)
            })
        
        return info