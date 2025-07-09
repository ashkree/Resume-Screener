import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, pipeline_factory, param_space=None, n_trials=20, cv_folds=5,
                 scoring='accuracy', random_state=42, writer=None):
        """
        Parameters:
        - pipeline_factory: callable that accepts a param dict and returns a pipeline
        - param_space: callable defining the Optuna hyperparameter space (trial -> dict)
        - n_trials: number of optimisation trials
        - cv_folds: number of Stratified K-Fold splits
        - scoring: metric to optimise
        - random_state: seed for reproducibility
        - writer: shared SummaryWriter instance (optional)
        """
        self.pipeline_factory = pipeline_factory
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.writer = writer

        self.best_pipeline = None
        self.best_score = None
        self.best_params = None
        self.training_history = []

    def train(self, X, y, optimise=False):
        """Train method with tqdm trial-level progress tracking"""
        self.training_history = []

        if not optimise or self.param_space is None:
            pipeline = self.pipeline_factory({})
            pipeline.fit(X, y)
            self.best_pipeline = pipeline
            self.best_score = pipeline.score(X, y)
            self.best_params = {}
            return self.best_pipeline

        # Initialize progress bar for trials only
        pbar = tqdm(total=self.n_trials, desc="Hyperparameter Optimization", 
                   unit="trial", ncols=120, disable=False, leave=True)

        # Track best score for display
        best_score_so_far = 0

        def objective(trial):
            nonlocal best_score_so_far
            
            params = self.param_space(trial)
            
            # Create pipeline ONCE for this trial
            pipeline = self.pipeline_factory(params)

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            fold_scores = []

            y_array = y.values if hasattr(y, 'values') else y

            # Cross-validation loop - REUSE the same pipeline instance
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_array)):
                if hasattr(X, 'iloc'):
                    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                
                if hasattr(y, 'iloc'):
                    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Simply refit the SAME pipeline on each fold
                pipeline.fit(X_train_fold, y_train_fold)
                score = pipeline.score(X_val_fold, y_val_fold)
                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            
            # === TRAINING ON FULL DATASET FOR COMPARISON ===
            # Reuse the SAME pipeline for full dataset training
            pipeline.fit(X, y)
            training_accuracy = pipeline.score(X, y)
            validation_accuracy = mean_score
            
            # Update best score
            if validation_accuracy > best_score_so_far:
                best_score_so_far = validation_accuracy
                status_msg = f"Train: {training_accuracy:.4f} | Val: {validation_accuracy:.4f} | Best: {best_score_so_far:.4f} ⭐"
            else:
                status_msg = f"Train: {training_accuracy:.4f} | Val: {validation_accuracy:.4f} | Best: {best_score_so_far:.4f}"
            
            # Update progress bar with current status
            pbar.set_postfix_str(status_msg)
            pbar.update(1)
            
            # === MINIMAL TRIAL LOGGING ===
            if self.writer:
                # Only log the essential optimization curves
                self.writer.add_scalar("Optimization/Training_Accuracy", training_accuracy, trial.number)
                self.writer.add_scalar("Optimization/Validation_Accuracy", validation_accuracy, trial.number)
                self.writer.add_scalar("Optimization/Best_So_Far", best_score_so_far, trial.number)
                self.writer.flush()

            trial_result = {
                "trial_number": trial.number,
                "params": params,
                "fold_scores": fold_scores,
                "mean_score": mean_score,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy
            }

            self.training_history.append(trial_result)
            return trial_result["mean_score"]

        # Run optimization
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
        finally:
            pbar.close()

        print(f"\n🎯 Optimization completed!")
        print(f"   Best score: {study.best_value:.4f}")
        print(f"   Total trials: {len(study.trials)}")
        
        # Build final model
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_pipeline = self.pipeline_factory(self.best_params)
        
        print(f"🔧 Training final model with best parameters...")
        self.best_pipeline.fit(X, y)
        print(f"✅ Training complete!")

        # === POST-OPTIMIZATION ANALYSIS ===
        if self.writer:
            self._log_optimization_summary()

        return self.best_pipeline
    
    def _log_optimization_summary(self):
        """Log summary metrics after optimization is complete"""
        if not self.training_history:
            return
            
        print("📊 Logging optimization summary...")
        
        # Extract data for analysis
        validation_scores = [t["mean_score"] for t in self.training_history]
        training_scores = [t["training_accuracy"] for t in self.training_history]
        fold_scores_all = [t["fold_scores"] for t in self.training_history]
        
        # === SUMMARY STATISTICS ===
        best_trial_idx = np.argmax(validation_scores)
        best_trial = self.training_history[best_trial_idx]
        
        # Log final summary metrics (single values)
        self.writer.add_scalar("Summary/Best_Validation_Score", max(validation_scores))
        self.writer.add_scalar("Summary/Best_Training_Score", training_scores[best_trial_idx])
        self.writer.add_scalar("Summary/Best_Trial_Number", best_trial_idx)
        self.writer.add_scalar("Summary/Total_Trials", len(self.training_history))
        
        # Overfitting analysis for best trial
        best_overfitting_gap = training_scores[best_trial_idx] - validation_scores[best_trial_idx]
        self.writer.add_scalar("Summary/Best_Trial_Overfitting_Gap", best_overfitting_gap)
        
        # Stability analysis for best trial
        best_stability = 1.0 - np.std(best_trial["fold_scores"])
        self.writer.add_scalar("Summary/Best_Trial_Stability", best_stability)
        
        # Overall optimization statistics
        score_improvement = max(validation_scores) - validation_scores[0]
        self.writer.add_scalar("Summary/Total_Score_Improvement", score_improvement)
        
        # Convergence analysis
        trials_to_best = best_trial_idx + 1
        self.writer.add_scalar("Summary/Trials_To_Best", trials_to_best)
        
        # === HYPERPARAMETER LOGGING FOR BEST TRIAL ===
        best_params = self._flatten_params(best_trial["params"])
        best_metrics = {
            "validation_accuracy": validation_scores[best_trial_idx],
            "training_accuracy": training_scores[best_trial_idx],
            "overfitting_gap": best_overfitting_gap,
            "stability": best_stability
        }
        
        # Log best hyperparameters
        self.writer.add_hparams(best_params, best_metrics)
        
        # Ensure all data is written
        self.writer.flush()
        print("✅ Optimization summary logged!")

    def _flatten_params(self, params):
        """Flatten nested parameter dictionary for hparams logging"""
        flat_params = {}
        for group, param_set in params.items():
            if isinstance(param_set, dict):
                for key, val in param_set.items():
                    # Convert unsupported types to strings
                    if isinstance(val, (tuple, list)):
                        val = str(val)
                    elif val is None:
                        val = "None"
                    elif not isinstance(val, (int, float, str, bool)):
                        val = str(val)
                    flat_params[f"{group}__{key}"] = val
            else:
                # Handle non-dict parameters
                if isinstance(param_set, (tuple, list)):
                    param_set = str(param_set)
                elif param_set is None:
                    param_set = "None"
                elif not isinstance(param_set, (int, float, str, bool)):
                    param_set = str(param_set)
                flat_params[group] = param_set
        return flat_params