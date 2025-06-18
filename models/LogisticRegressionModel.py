from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .BaseModel import BaseModel
from typing import Dict, Any  
import numpy as np


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model with anti-overfitting measures
    """
    
    def __init__(self, use_scaler=True, regularization_strength='auto', **kwargs):
        super().__init__("Logistic Regression", **kwargs)
        self.use_scaler = use_scaler
        self.regularization_strength = regularization_strength
        
    def _create_model(self, **params) -> Pipeline:
        """Create Logistic Regression model with strong regularization to prevent overfitting"""
        
        # Anti-overfitting defaults
        default_params = {
            'C': 0.1,  # STRONGER regularization (lower C = more regularization)
            'penalty': 'l2',  # L2 regularization by default
            'solver': 'liblinear',
            'class_weight': 'balanced',  # Handle class imbalance
            'random_state': 42,
            'max_iter': 2000,  # Increased for convergence
            'fit_intercept': True
        }
        
        # Auto-adjust regularization based on data size
        if self.regularization_strength == 'auto' and hasattr(self, '_data_info'):
            n_samples = self._data_info.get('n_samples', 1000)
            n_features = self._data_info.get('n_features', 100)
            
            # Stronger regularization for high-dimensional data
            if n_features > n_samples:
                default_params['C'] = 0.01  # Very strong regularization
            elif n_features > n_samples * 0.5:
                default_params['C'] = 0.05  # Strong regularization
            else:
                default_params['C'] = 0.1   # Moderate regularization
        
        # Update with provided parameters
        default_params.update(params)
        default_params.update(self.config)
        
        # Create the logistic regression model
        lr_model = LogisticRegression(**default_params)
        
        if self.use_scaler:
            # Create pipeline with StandardScaler
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', lr_model)
            ])
            return pipeline
        else:
            return lr_model
    
    def get_param_space(self) -> Dict[str, Any]:
        """Define parameter space focused on preventing overfitting"""
        if self.use_scaler:
            return {
                # Much stronger regularization range to prevent overfitting
                'classifier__C': (0.001, 10.0),  # Lower max C, much lower min C
                'classifier__penalty': ['l1', 'l2', 'elasticnet'],  # Include ElasticNet
                'classifier__solver': ['liblinear', 'saga'],  # saga supports elasticnet
                'classifier__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # For elasticnet
                'classifier__max_iter': [1000, 2000, 5000],
            }
        else:
            return {
                'C': (0.001, 10.0),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000, 5000],
            }
    
    def fit(self, X, y, **fit_params):
        """Override fit to store data info for auto-regularization"""
        # Store data information for auto-regularization
        if hasattr(X, 'shape'):
            self._data_info = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1]
            }
        else:
            self._data_info = {
                'n_samples': len(X),
                'n_features': len(X[0]) if len(X) > 0 else 0
            }
        
        print(f"📊 Data info: {self._data_info['n_samples']} samples, {self._data_info['n_features']} features")
        
        # Check for potential overfitting conditions
        ratio = self._data_info['n_features'] / self._data_info['n_samples']
        if ratio > 0.1:
            print(f"⚠️  High feature-to-sample ratio: {ratio:.2f} (overfitting risk)")
        if ratio > 0.5:
            print("🚨 Very high dimensionality - using strong regularization")
            
        return super().fit(X, y, **fit_params)
    
    def get_overfitting_analysis(self, X_train, y_train, X_val, y_val):
        """Analyze overfitting by comparing train vs validation performance"""
        if not self.is_fitted:
            return {"error": "Model must be fitted first"}
        
        # Training performance
        train_pred = self.predict(X_train)
        train_proba = self.predict_proba(X_train)
        train_accuracy = np.mean(train_pred == y_train)
        
        # Validation performance
        val_pred = self.predict(X_val)
        val_proba = self.predict_proba(X_val)
        val_accuracy = np.mean(val_pred == y_val)
        
        # Overfitting metrics
        accuracy_gap = train_accuracy - val_accuracy
        
        # Confidence analysis
        train_max_proba = np.max(train_proba, axis=1).mean()
        val_max_proba = np.max(val_proba, axis=1).mean()
        confidence_gap = train_max_proba - val_max_proba
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'accuracy_gap': accuracy_gap,
            'train_confidence': train_max_proba,
            'val_confidence': val_max_proba,
            'confidence_gap': confidence_gap,
            'overfitting_severity': self._classify_overfitting(accuracy_gap),
            'recommendations': self._get_overfitting_recommendations(accuracy_gap, confidence_gap)
        }
    
    def _classify_overfitting(self, accuracy_gap):
        """Classify severity of overfitting"""
        if accuracy_gap < 0.05:
            return "None/Minimal"
        elif accuracy_gap < 0.15:
            return "Moderate" 
        elif accuracy_gap < 0.3:
            return "Severe"
        else:
            return "Extreme"
    
    def _get_overfitting_recommendations(self, accuracy_gap, confidence_gap):
        """Get recommendations to reduce overfitting"""
        recommendations = []
        
        if accuracy_gap > 0.1:
            recommendations.append("Increase regularization (decrease C parameter)")
        
        if accuracy_gap > 0.2:
            recommendations.append("Try L1 regularization for feature selection")
            recommendations.append("Consider feature selection/dimensionality reduction")
        
        if accuracy_gap > 0.3:
            recommendations.append("Use ElasticNet penalty (L1 + L2)")
            recommendations.append("Collect more training data")
            recommendations.append("Use cross-validation for model selection")
        
        if confidence_gap > 0.2:
            recommendations.append("Model is overconfident - increase regularization")
        
        if not recommendations:
            recommendations.append("Overfitting is minimal - current settings are good")
        
        return recommendations

    def get_coefficients_analysis(self):
        """Enhanced coefficient analysis with overfitting indicators"""
        if not self.is_fitted:
            return None
        
        # Handle both pipeline and direct model cases
        if self.use_scaler and hasattr(self.model, 'named_steps'):
            lr_model = self.model.named_steps['classifier']
            scaler = self.model.named_steps['scaler']
            coef = lr_model.coef_[0] if len(lr_model.coef_.shape) > 1 else lr_model.coef_
            
            analysis = {
                'coefficients': coef,
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'using_scaler': True,
                'regularization_param_C': lr_model.C,
                'penalty_type': lr_model.penalty,
                'solver': lr_model.solver
            }
        else:
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            analysis = {
                'coefficients': coef,
                'using_scaler': False,
                'regularization_param_C': self.model.C,
                'penalty_type': self.model.penalty,
                'solver': self.model.solver
            }
        
        # Common analysis
        analysis.update({
            'positive_coef_count': np.sum(coef > 0),
            'negative_coef_count': np.sum(coef < 0),
            'coef_magnitude_mean': np.abs(coef).mean(),
            'coef_magnitude_std': np.abs(coef).std(),
            'max_positive_coef': coef.max(),
            'max_negative_coef': coef.min(),
            'sparsity': np.sum(np.abs(coef) < 0.001) / len(coef),
            
            # Overfitting indicators
            'large_coefficients_count': np.sum(np.abs(coef) > 10),  # Very large coefficients
            'extreme_coefficients_count': np.sum(np.abs(coef) > 50),  # Extreme coefficients
            'coefficient_variance': np.var(coef),  # High variance indicates overfitting
            'overfitting_risk_score': self._calculate_overfitting_risk(coef)
        })
        
        return analysis
    
    def _calculate_overfitting_risk(self, coef):
        """Calculate a risk score for overfitting based on coefficients"""
        # Factors that indicate overfitting:
        # 1. Very large coefficient magnitudes
        # 2. High variance in coefficients
        # 3. Many extreme coefficients
        
        large_coef_ratio = np.sum(np.abs(coef) > 10) / len(coef)
        extreme_coef_ratio = np.sum(np.abs(coef) > 50) / len(coef)
        coef_std = np.std(np.abs(coef))
        
        # Normalize and combine (0-1 scale)
        risk_score = (
            min(large_coef_ratio * 2, 1) * 0.4 +  # Large coefficients
            min(extreme_coef_ratio * 10, 1) * 0.4 +  # Extreme coefficients  
            min(coef_std / 100, 1) * 0.2  # Coefficient variance
        )
        
        return min(risk_score, 1.0)
