# ModelTrainer with Optuna Bayesian Optimization
# Much cleaner and more reliable than skopt

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not installed. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

class ModelTrainer:
    """
    Advanced model trainer with Optuna Bayesian optimization
    Much cleaner and more reliable than scikit-optimize
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 scoring: str = 'f1_weighted',
                 n_trials: int = 50,
                 random_state: int = 42,
                 verbose: bool = True,
                 study_name: Optional[str] = None):
        """
        Initialize ModelTrainer with Optuna
        
        Args:
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_trials: Number of Optuna optimization trials
            random_state: Random state for reproducibility
            verbose: Whether to print progress
            study_name: Name for Optuna study (for persistence)
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self.study_name = study_name or f"resume_screening_{int(time.time())}"
        
        # Training history
        self.training_history = {}
        self.optimization_studies = {}
        
        # CV strategy
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=random_state
        )
        
        # Optuna configuration
        optuna.logging.set_verbosity(optuna.logging.WARNING if not verbose else optuna.logging.INFO)
    
    def _create_objective_function(self, model, X, y):
        """
        Create Optuna objective function for a specific model
        
        Args:
            model: Model instance to optimize
            X: Training features
            y: Training labels
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            try:
                # Get parameter space from model
                param_space = model.get_param_space()
                
                # Sample parameters using Optuna
                params = {}
                for param_name, param_def in param_space.items():
                    
                    if isinstance(param_def, tuple) and len(param_def) == 2:
                        low, high = param_def
                        if isinstance(low, int) and isinstance(high, int):
                            # Integer parameter
                            params[param_name] = trial.suggest_int(param_name, low, high)
                        else:
                            # Float parameter
                            params[param_name] = trial.suggest_float(param_name, low, high)
                    
                    elif isinstance(param_def, list):
                        # Categorical parameter
                        params[param_name] = trial.suggest_categorical(param_name, param_def)
                    
                    else:
                        raise ValueError(f"Invalid parameter definition for {param_name}: {param_def}")
                
                # Create model with suggested parameters
                model_instance = model._create_model(**params)
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model_instance, X, y,
                    cv=self.cv_strategy,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                
                score = cv_scores.mean()
                
                # Optuna handles reporting and pruning
                trial.report(score, step=0)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score  # Optuna maximizes by default for most metrics
                
            except Exception as e:
                if self.verbose:
                    print(f"   Trial failed: {str(e)}")
                # Return a poor score for failed trials
                return 0.0 if 'accuracy' in self.scoring or 'f1' in self.scoring else float('inf')
        
        return objective
    
    def optimize_hyperparameters(self, 
                                model, 
                                X, 
                                y, 
                                direction: str = 'maximize') -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            model: Model instance to optimize
            X: Training features
            y: Training labels
            direction: 'maximize' or 'minimize' (usually 'maximize' for accuracy/f1)
            
        Returns:
            Dictionary with optimization results
        """
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Using default parameters.")
            return {'best_params': {}, 'best_score': None, 'optimization_time': 0}
        
        print(f"🔍 Optimizing {model.name} hyperparameters with Optuna...")
        print(f"   CV folds: {self.cv_folds}")
        print(f"   Optimization trials: {self.n_trials}")
        print(f"   Scoring metric: {self.scoring}")
        print(f"   Direction: {direction}")
        
        # Create Optuna study
        study_name_full = f"{self.study_name}_{model.name.lower().replace(' ', '_')}"
        
        study = optuna.create_study(
            direction=direction,
            study_name=study_name_full,
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Create objective function
        objective = self._create_objective_function(model, X, y)
        
        # Run optimization
        start_time = time.time()
        
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
            callbacks=[lambda study, trial: print(f"   Trial {trial.number}: {trial.value:.4f}")] if self.verbose else None
        )
        
        optimization_time = time.time() - start_time
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        optimization_result = {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_time': optimization_time,
            'n_trials': len(study.trials),
            'study': study,  # Store study for further analysis
            'best_trial': study.best_trial,
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        }
        
        # Store study for later analysis
        self.optimization_studies[model.name] = study
        
        print(f"✅ Optimization completed in {optimization_time:.2f} seconds")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best parameters: {best_params}")
        print(f"   Completed trials: {optimization_result['completed_trials']}")
        print(f"   Pruned trials: {optimization_result['pruned_trials']}")
        print(f"   Failed trials: {optimization_result['failed_trials']}")
        
        return optimization_result
    
    def train_model(self, 
                   model, 
                   X_train, 
                   y_train, 
                   X_val=None, 
                   y_val=None,
                   optimize_hyperparameters: bool = True,
                   **fit_params) -> Dict[str, Any]:
        """
        Train a single model with optional Optuna hyperparameter optimization
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize_hyperparameters: Whether to optimize hyperparameters
            **fit_params: Additional fitting parameters
            
        Returns:
            Dictionary with training results
        """
        print(f"🚀 Training {model.name}...")

        # Handle sparse matrix shapes properly
        if hasattr(X_train, 'shape'):
            n_train_samples = X_train.shape[0]
        else: 
            n_train_samples = len(X_train)
        
        training_result = {
            'model_name': model.name,
            'hyperparameter_optimization': optimize_hyperparameters,
            'training_samples': len(X_train),
            'validation_samples': len(X_val) if X_val is not None else 0
        }
        
        # Hyperparameter optimization with Optuna
        if optimize_hyperparameters:
            # Determine direction based on scoring metric
            direction = 'maximize' if any(metric in self.scoring for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']) else 'minimize'
            
            opt_result = self.optimize_hyperparameters(model, X_train, y_train, direction=direction)
            training_result['optimization_result'] = opt_result
            
            # Update model with best parameters
            model.best_params = opt_result['best_params']
            model.model = model._create_model(**opt_result['best_params'])
        else:
            # Use default parameters
            model.model = model._create_model()
        
        # Train the model
        start_time = time.time()
        model.fit(X_train, y_train, **fit_params)
        training_time = time.time() - start_time
        
        training_result['training_time'] = training_time
        training_result['fit_time'] = model.fit_time
        
        # Cross-validation on training data
        cv_scores = model.cross_validate(X_train, y_train, cv=self.cv_folds, scoring=self.scoring)
        training_result['cv_scores'] = cv_scores
        training_result['cv_mean'] = cv_scores.mean()
        training_result['cv_std'] = cv_scores.std()
        
        # Training predictions
        train_pred = model.predict(X_train)
        training_result['train_accuracy'] = accuracy_score(y_train, train_pred)
        
        # Validation predictions (if validation data provided)
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            training_result['val_accuracy'] = accuracy_score(y_val, val_pred)
            training_result['val_predictions'] = val_pred
            
            if hasattr(model.model, 'predict_proba'):
                val_proba = model.predict_proba(X_val)
                training_result['val_probabilities'] = val_proba
        
        # Feature importance (if available)
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            training_result['feature_importance'] = feature_importance
        
        # Store in history
        self.training_history[model.name] = training_result
        
        print(f"✅ {model.name} training completed!")
        print(f"   Training accuracy: {training_result['train_accuracy']:.4f}")
        print(f"   CV score: {training_result['cv_mean']:.4f} (±{training_result['cv_std']:.4f})")
        if 'val_accuracy' in training_result:
            print(f"   Validation accuracy: {training_result['val_accuracy']:.4f}")
        
        return training_result
    
    def get_optimization_insights(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed insights from Optuna optimization
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with optimization insights
        """
        if model_name not in self.optimization_studies:
            return {"error": f"No optimization study found for {model_name}"}
        
        study = self.optimization_studies[model_name]
        
        insights = {
            'best_trial': study.best_trial.number,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_history': [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE],
            'param_importance': {},
            'trials_dataframe': study.trials_dataframe() if len(study.trials) > 0 else None
        }
        
        # Parameter importance (if enough trials)
        if len(study.trials) >= 10:
            try:
                param_importance = optuna.importance.get_param_importances(study)
                insights['param_importance'] = param_importance
            except:
                insights['param_importance'] = {"error": "Could not calculate parameter importance"}
        
        return insights
    
    def train_ensemble(self, 
                      ensemble_model, 
                      X_train, 
                      y_train, 
                      X_val=None, 
                      y_val=None,
                      optimize_base_models: bool = True,
                      optimize_ensemble_weights: bool = False) -> Dict[str, Any]:
        """
        Train ensemble model and its base models using Optuna
        
        Args:
            ensemble_model: EnsembleModel instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize_base_models: Whether to optimize base model hyperparameters
            optimize_ensemble_weights: Whether to optimize ensemble weights
            
        Returns:
            Dictionary with ensemble training results
        """
        print(f"🎯 Training Ensemble with {len(ensemble_model.base_models)} base models...")
        
        ensemble_result = {
            'ensemble_name': ensemble_model.name,
            'base_model_count': len(ensemble_model.base_models),
            'base_model_results': {},
            'voting_method': ensemble_model.voting
        }
        
        # Train base models first
        for base_model in ensemble_model.base_models:
            if not base_model.is_fitted:
                base_result = self.train_model(
                    base_model, X_train, y_train, X_val, y_val,
                    optimize_hyperparameters=optimize_base_models
                )
                ensemble_result['base_model_results'][base_model.name] = base_result
        
        # Optimize ensemble weights if requested
        if optimize_ensemble_weights:
            print("🔧 Optimizing ensemble weights with Optuna...")
            direction = 'maximize' if any(metric in self.scoring for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']) else 'minimize'
            weight_result = self.optimize_hyperparameters(ensemble_model, X_train, y_train, direction=direction)
            ensemble_result['weight_optimization'] = weight_result
            
            # Apply optimized weights
            if len(ensemble_model.base_models) == 2:
                w1 = weight_result['best_params'].get('weight_1', 0.5)
                ensemble_model.weights = [w1, 1-w1]
            else:
                # More complex weight optimization for >2 models
                weights = []
                for i in range(len(ensemble_model.base_models)):
                    weights.append(weight_result['best_params'].get(f'weight_{i}', 1.0))
                # Normalize weights
                total_weight = sum(weights)
                ensemble_model.weights = [w/total_weight for w in weights]
        
        # Train the ensemble
        ensemble_training = self.train_model(
            ensemble_model, X_train, y_train, X_val, y_val,
            optimize_hyperparameters=False  # Already optimized base models
        )
        
        ensemble_result.update(ensemble_training)
        
        # Agreement analysis
        if X_val is not None:
            agreement_analysis = ensemble_model.get_agreement_analysis(X_val)
            ensemble_result['agreement_analysis'] = agreement_analysis
            
            print(f"   Model agreement rate: {agreement_analysis['agreement_rate']:.4f}")
            print(f"   Disagreement samples: {len(agreement_analysis['disagreement_indices'])}")
        
        return ensemble_result
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all training results
        
        Returns:
            DataFrame with training summary
        """
        summary_data = []
        
        for model_name, result in self.training_history.items():
            summary_data.append({
                'Model': model_name,
                'Train_Accuracy': result.get('train_accuracy', 0),
                'Val_Accuracy': result.get('val_accuracy', 0), 
                'CV_Mean': result.get('cv_mean', 0),
                'CV_Std': result.get('cv_std', 0),
                'Training_Time': result.get('training_time', 0),
                'Hyperparameter_Optimized': result.get('hyperparameter_optimization', False),
                'Training_Samples': result.get('training_samples', 0)
            })
        
        return pd.DataFrame(summary_data)
