from sklearn.ensemble import VotingClassifier

from typing import Dict, Any, List
import numpy as np
import time

from .BaseModel import BaseModel

class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models
    """
    
    def __init__(self, base_models: List[BaseModel], voting='soft', weights=None, **kwargs):
        super().__init__("Ensemble", **kwargs)
        self.base_models = base_models
        self.voting = voting
        self.weights = weights
        self.individual_predictions = {}
        
    def _create_model(self, **params) -> VotingClassifier:
        """Create ensemble model from base models"""
        # Ensure all base models have their sklearn models created
        estimators = []
        for base_model in self.base_models:
            if base_model.model is None:
                base_model.model = base_model._create_model()
            estimators.append((base_model.name.lower().replace(' ', '_'), base_model.model))
        
        return VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights
        )
    
    def get_param_space(self) -> Dict[str, Any]:
        """Define parameter space for ensemble (weights optimization)"""
        if len(self.base_models) == 2:
            return {
                'weight_1': (0.1, 0.9),  # Weight for first model (second is 1-weight_1)
            }
        else:
            # For more models, define individual weights
            param_space = {}
            for i in range(len(self.base_models)):
                param_space[f'weight_{i}'] = (0.1, 0.9)
            return param_space
    
    def fit(self, X, y, **fit_params):
        """Fit ensemble and store individual model predictions"""
        # First fit base models individually
        for base_model in self.base_models:
            if not base_model.is_fitted:
                base_model.fit(X, y, **fit_params)
        
        # Then fit the ensemble
        result = super().fit(X, y, **fit_params)
        
        # Store individual predictions for analysis
        for base_model in self.base_models:
            self.individual_predictions[base_model.name] = {
                'train_pred': base_model.predict(X),
                'train_proba': base_model.predict_proba(X) if hasattr(base_model.model, 'predict_proba') else None
            }
        
        return result
    
    def get_agreement_analysis(self, X):
        """Analyze agreement between base models"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        predictions = {}
        probabilities = {}
        
        for base_model in self.base_models:
            predictions[base_model.name] = base_model.predict(X)
            if hasattr(base_model.model, 'predict_proba'):
                probabilities[base_model.name] = base_model.predict_proba(X)
        
        # Calculate agreement metrics
        pred_arrays = list(predictions.values())
        agreement = np.mean([pred_arrays[0] == pred_arrays[i] for i in range(1, len(pred_arrays))], axis=0)
        
        return {
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'agreement_rate': agreement.mean(),
            'agreement_per_sample': agreement,
            'disagreement_indices': np.where(agreement < 1.0)[0]
        }
