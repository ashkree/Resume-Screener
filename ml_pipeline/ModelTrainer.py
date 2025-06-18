import time
import pickle
import os
from typing import Any, Dict, Optional
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

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
    Simplified model trainer focused on core optimization and CV
    """
    
    def __init__(
        self,
        cv_folds: int = 5,
        scoring: str = 'f1_weighted',
        n_trials: int = 50,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        
        # CV strategy
        self.cv_strategy = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
        
        # Simple training history
        self.training_history = {}
        
        # Configure Optuna logging
        if OPTUNA_AVAILABLE:
            optuna.logging.set_verbosity(
                optuna.logging.INFO if verbose else optuna.logging.WARNING
            )
    
    def _create_objective_function(self, model, X, y, param_space):
        """Create Optuna objective function"""
        def objective(trial):
            try:
                # Sample parameters from space
                params = {}
                for name, spec in param_space.items():
                    if isinstance(spec, tuple) and len(spec) == 2:
                        low, high = spec
                        if isinstance(low, int) and isinstance(high, int):
                            params[name] = trial.suggest_int(name, low, high)
                        else:
                            params[name] = trial.suggest_float(name, low, high)
                    elif isinstance(spec, list):
                        params[name] = trial.suggest_categorical(name, spec)
                    else:
                        raise ValueError(f"Invalid parameter spec for {name}: {spec}")
                
                # Create model instance and run CV
                model_instance = model._create_model(**params)
                cv_scores = cross_val_score(
                    model_instance, X, y,
                    cv=self.cv_strategy,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                
                return cv_scores.mean()
                
            except Exception as e:
                if self.verbose:
                    print(f"   Trial failed: {e}")
                # Return poor score for failed trials
                return 0.0 if 'accuracy' in self.scoring or 'f1' in self.scoring else float('inf')
        
        return objective
    
    def optimize_hyperparameters(
        self,
        model,
        X,
        y,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Bayesian hyperparameter optimization"""
        
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available. Skipping optimization.")
            return {'best_params': {}, 'best_score': None}
        
        if self.verbose:
            print(f"🔍 Optimizing {model.name} hyperparameters...")
        
        # Create study
        direction = 'maximize' if any(m in self.scoring for m in ['accuracy', 'f1', 'precision', 'recall']) else 'minimize'
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        # Optimize
        objective = self._create_objective_function(model, X, y, param_space)
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        optimization_time = time.time() - start_time
        
        if self.verbose:
            print(f"✅ Optimization completed in {optimization_time:.1f}s")
            print(f"   Best score: {study.best_value:.4f}")
            print(f"   Best params: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'optimization_time': optimization_time,
            'n_trials': len(study.trials)
        }
    
    def train_model(
        self,
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        param_space: Optional[Dict[str, Any]] = None,
        optimize: bool = True,
        **fit_params  # Only sklearn fit parameters
    ) -> Dict[str, Any]:
        """
        Train a model with optional hyperparameter optimization
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            param_space: Parameter space for optimization
            optimize: Whether to run hyperparameter optimization
        
        Returns:
            Training results dictionary
        """
        
        if self.verbose:
            print(f"🚀 Training {model.name}...")
        
        start_time = time.time()
        results = {
            'model_name': model.name,
            'n_samples': len(y_train)
        }
        
        # 1. Hyperparameter optimization
        if optimize and param_space:
            opt_results = self.optimize_hyperparameters(model, X_train, y_train, param_space)
            results['optimization'] = opt_results
            # Create model with best params
            model.model = model._create_model(**opt_results['best_params'])
        else:
            # Use default model
            model.model = model._create_model()
            results['optimization'] = None
        
        # 2. Train the model
        fit_start = time.time()
        model.fit(X_train, y_train, **fit_params)  # Only pass sklearn fit params
        fit_time = time.time() - fit_start
        
        # 3. Cross-validation on training data
        if self.verbose:
            print(f"🔄 Running {self.cv_folds}-fold cross-validation...")
        
        cv_scores = cross_val_score(
            model.model, X_train, y_train,
            cv=self.cv_strategy,
            scoring=self.scoring,
            n_jobs=-1
        )
        
        # 4. Predictions and metrics
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
        
        # 5. Compile results
        total_time = time.time() - start_time
        
        results.update({
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'fit_time': fit_time,
            'total_time': total_time
        })
        
        # Store in history
        self.training_history[model.name] = results
        
        if self.verbose:
            gap = train_accuracy - val_accuracy if val_accuracy else 0
            print(f"✅ {model.name} completed in {total_time:.1f}s")
            print(f"   CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   Train: {train_accuracy:.4f}")
            if val_accuracy:
                print(f"   Val: {val_accuracy:.4f} (gap: {gap:.4f})")
        
        return results
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get all training history"""
        return self.training_history.copy()
    
    def compare_models(self) -> None:
        """Print comparison of all trained models"""
        if not self.training_history:
            print("No models trained yet.")
            return
        
        print(f"\n📊 Model Comparison:")
        print(f"{'Model':<20} {'CV Score':<12} {'Val Acc':<10} {'Gap':<8}")
        print("-" * 55)
        
        for name, results in self.training_history.items():
            cv_score = f"{results['cv_mean']:.4f}±{results['cv_std']:.3f}"
            val_acc = f"{results['val_accuracy']:.4f}" if results['val_accuracy'] else "N/A"
            gap = results['train_accuracy'] - results['val_accuracy'] if results['val_accuracy'] else 0
            gap_str = f"{gap:.4f}" if results['val_accuracy'] else "N/A"
            
            print(f"{name:<20} {cv_score:<12} {val_acc:<10} {gap_str:<8}")
    
    def save_model(
        self, 
        model, 
        filepath: str, 
        include_results: bool = True
    ) -> None:
        """
        Save trained model to disk
        
        Args:
            model: Trained model instance
            filepath: Path to save the model
            include_results: Whether to include training results
        """
        if not model.is_fitted:
            raise ValueError(f"Model {model.name} is not fitted yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare save data
        save_data = {
            'model': model,
            'model_name': model.name,
            'sklearn_model': model.model,  # The actual sklearn model
            'config': model.config,
            'is_fitted': model.is_fitted
        }
        
        # Include training results if available and requested
        if include_results and model.name in self.training_history:
            save_data['training_results'] = self.training_history[model.name]
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load from disk
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore model
        model = save_data['model']
        
        # Restore training results if available
        if 'training_results' in save_data:
            self.training_history[model.name] = save_data['training_results']
        
        if self.verbose:
            print(f"📂 Model loaded from: {filepath}")
            print(f"   Model: {model.name}")
            print(f"   Fitted: {model.is_fitted}")
        
        return model
    
    def save_best_models(
        self, 
        save_dir: str = "./models/", 
        metric: str = "cv_mean"
    ) -> Dict[str, str]:
        """
        Save all trained models, optionally sorted by performance
        
        Args:
            save_dir: Directory to save models
            metric: Metric to use for "best" model identification
            
        Returns:
            Dictionary mapping model names to file paths
        """
        if not self.training_history:
            print("No models to save.")
            return {}
        
        os.makedirs(save_dir, exist_ok=True)
        saved_models = {}
        
        # Sort models by performance if requested
        if metric in ['cv_mean', 'val_accuracy', 'train_accuracy']:
            sorted_models = sorted(
                self.training_history.items(),
                key=lambda x: x[1].get(metric, 0),
                reverse=True
            )
            
            if self.verbose:
                print(f"\n💾 Saving models (sorted by {metric}):")
        else:
            sorted_models = list(self.training_history.items())
            if self.verbose:
                print(f"\n💾 Saving models:")
        
        for i, (model_name, results) in enumerate(sorted_models):
            # Generate filename
            clean_name = model_name.lower().replace(' ', '_').replace('-', '_')
            if metric in ['cv_mean', 'val_accuracy', 'train_accuracy']:
                score = results.get(metric, 0)
                filename = f"{clean_name}_{metric}_{score:.4f}.pkl"
                if i == 0:  # Best model
                    best_filename = f"best_{clean_name}.pkl"
                    # Save best model with special name
                    best_path = os.path.join(save_dir, best_filename)
                    # We need to find the actual model instance
                    # This is a limitation - we should store model references
                    print(f"   Note: To save models, pass model instances to save_model()")
            else:
                filename = f"{clean_name}.pkl"
            
            filepath = os.path.join(save_dir, filename)
            saved_models[model_name] = filepath
            
            if self.verbose:
                score_str = f"{results.get(metric, 0):.4f}" if metric in results else "N/A"
                print(f"   {model_name}: {filename} ({metric}: {score_str})")
        
        return saved_models
    
    def save_training_session(self, filepath: str) -> None:
        """
        Save entire training session (all models and results)
        
        Args:
            filepath: Path to save the session
        """
        session_data = {
            'training_history': self.training_history,
            'trainer_config': {
                'cv_folds': self.cv_folds,
                'scoring': self.scoring,
                'n_trials': self.n_trials,
                'random_state': self.random_state
            },
            'timestamp': time.time()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(session_data, f)
        
        if self.verbose:
            print(f"💾 Training session saved to: {filepath}")
    
    def load_training_session(self, filepath: str) -> None:
        """
        Load training session (restores training history)
        
        Args:
            filepath: Path to the saved session
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Session file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            session_data = pickle.load(f)
        
        # Restore training history
        self.training_history = session_data.get('training_history', {})
        
        if self.verbose:
            print(f"📂 Training session loaded from: {filepath}")
            print(f"   Models in history: {len(self.training_history)}")
            if self.training_history:
                print(f"   Models: {list(self.training_history.keys())}")
    
    def get_best_model_info(self, metric: str = "cv_mean") -> Optional[Dict[str, Any]]:
        """
        Get information about the best performing model
        
        Args:
            metric: Metric to use for determining "best"
            
        Returns:
            Dictionary with best model info, or None if no models trained
        """
        if not self.training_history:
            return None
        
        best_model = max(
            self.training_history.items(),
            key=lambda x: x[1].get(metric, 0)
        )
        
        model_name, results = best_model
        
        return {
            'model_name': model_name,
            'metric_used': metric,
            'metric_value': results.get(metric, 0),
            'full_results': results
        }