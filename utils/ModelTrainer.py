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
        - cv_folds: number of Stratified K-Fold splits (used only if no validation data provided)
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
        self.using_custom_validation = False

    def train(self, X_train, y_train, X_val=None, y_val=None, optimise=False):
        """
        Train method with support for custom validation split

        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_val: Validation features (optional)
        - y_val: Validation labels (optional)
        - optimise: Whether to run hyperparameter optimization
        """
        self.training_history = []
        self.using_custom_validation = X_val is not None and y_val is not None

        if not optimise or self.param_space is None:
            pipeline = self.pipeline_factory({})
            pipeline.fit(X_train, y_train)
            self.best_pipeline = pipeline

            if self.using_custom_validation:
                self.best_score = pipeline.score(X_val, y_val)
            else:
                self.best_score = pipeline.score(X_train, y_train)

            self.best_params = {}
            return self.best_pipeline

        # Initialize progress bar for trials
        validation_method = "Custom Val Split" if self.using_custom_validation else "Cross-Validation"
        pbar = tqdm(total=self.n_trials, desc=f"Hyperparameter Optimization ({validation_method})",
                    unit="trial", ncols=120, disable=False, leave=True)

        # Track best score for display
        best_score_so_far = 0

        def objective(trial):
            nonlocal best_score_so_far

            params = self.param_space(trial)
            pipeline = self.pipeline_factory(params)

            if self.using_custom_validation:
                # Use your provided validation split
                validation_accuracy = self._train_with_custom_validation(
                    pipeline, X_train, y_train, X_val, y_val
                )
            else:
                # Use cross-validation
                validation_accuracy = self._train_with_cross_validation(
                    pipeline, X_train, y_train
                )

            # Train on full training data for training accuracy
            pipeline.fit(X_train, y_train)
            training_accuracy = pipeline.score(X_train, y_train)

            # Update best score
            if validation_accuracy > best_score_so_far:
                best_score_so_far = validation_accuracy
                status_msg = f"Train: {training_accuracy:.4f} | Val: {validation_accuracy:.4f} | Best: {best_score_so_far:.4f} ⭐"
            else:
                status_msg = f"Train: {training_accuracy:.4f} | Val: {validation_accuracy:.4f} | Best: {best_score_so_far:.4f}"

            # Update progress bar
            pbar.set_postfix_str(status_msg)
            pbar.update(1)

            # TensorBoard logging
            if self.writer:
                overfit_gap = training_accuracy - validation_accuracy

                self.writer.add_scalars(
                    "Optimization/Accuracies",
                    {
                        "Training_Accuracy": training_accuracy,
                        "Validation_Accuracy": validation_accuracy,
                        "Best_So_Far": best_score_so_far,
                        "Overfit_Gap": overfit_gap
                    },
                    trial.number
                )
                self.writer.flush()

            # Store trial results
            trial_result = {
                "trial_number": trial.number,
                "params": params,
                "training_accuracy": training_accuracy,
                "validation_accuracy": validation_accuracy,
                "validation_method": "custom_split" if self.using_custom_validation else "cross_validation"
            }

            # Add fold scores only for CV
            if not self.using_custom_validation:
                trial_result["fold_scores"] = getattr(
                    self, '_last_fold_scores', [])
                trial_result["mean_score"] = validation_accuracy

            self.training_history.append(trial_result)
            return validation_accuracy

        # Run optimization
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
        finally:
            pbar.close()

        print(f"\n🎯 Optimization completed using {validation_method}!")
        print(f"   Best score: {study.best_value:.4f}")
        print(f"   Total trials: {len(study.trials)}")

        # Build final model with best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        self.best_pipeline = self.pipeline_factory(self.best_params)

        print(f"🔧 Training final model with best parameters...")
        self.best_pipeline.fit(X_train, y_train)
        print(f"✅ Training complete!")

        # Log optimization summary
        if self.writer:
            self._log_optimization_summary(X_val, y_val)

        return self.best_pipeline

    def _train_with_custom_validation(self, pipeline, X_train, y_train, X_val, y_val):
        """Train with custom validation split"""
        pipeline.fit(X_train, y_train)
        return pipeline.score(X_val, y_val)

    def _train_with_cross_validation(self, pipeline, X_train, y_train):
        """Train with cross-validation"""
        skf = StratifiedKFold(n_splits=self.cv_folds,
                              shuffle=True, random_state=self.random_state)
        fold_scores = []

        y_array = y_train.values if hasattr(y_train, 'values') else y_train

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_array)):
            # Handle both pandas and numpy inputs
            if hasattr(X_train, 'iloc'):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]

            if hasattr(y_train, 'iloc'):
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            else:
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            pipeline.fit(X_train_fold, y_train_fold)
            score = pipeline.score(X_val_fold, y_val_fold)
            fold_scores.append(score)

        # Store for trial results
        self._last_fold_scores = fold_scores
        return np.mean(fold_scores)

    def _log_optimization_summary(self, X_val=None, y_val=None):
        """Log summary metrics after optimization is complete"""
        if not self.training_history:
            return

        print("📊 Logging optimization summary...")

        # Extract data for analysis
        validation_scores = [t["validation_accuracy"]
                             for t in self.training_history]
        training_scores = [t["training_accuracy"]
                           for t in self.training_history]

        # Summary statistics
        best_trial_idx = np.argmax(validation_scores)
        best_trial = self.training_history[best_trial_idx]

        best_validation_score = max(validation_scores)
        best_training_score = training_scores[best_trial_idx]
        best_overfitting_gap = best_training_score - best_validation_score
        score_improvement = best_validation_score - validation_scores[0]
        trials_to_best = best_trial_idx + 1

        # Calculate stability based on validation method
        if self.using_custom_validation:
            stability_info = "Single validation split used"
            cv_details = f"**Validation Method**: Custom train/validation split"
        else:
            fold_scores = best_trial.get("fold_scores", [])
            if fold_scores:
                best_stability = 1.0 - np.std(fold_scores)
                stability_info = f"{best_stability:.4f}"
                cv_details = f"""**Validation Method**: {self.cv_folds}-fold cross-validation
- **Fold Scores**: {[f'{score:.4f}' for score in fold_scores]}
- **Mean CV Score**: {best_trial.get("mean_score", "N/A"):.4f}
- **CV Standard Deviation**: {np.std(fold_scores):.4f}"""
            else:
                stability_info = "N/A"
                cv_details = f"**Validation Method**: {self.cv_folds}-fold cross-validation"

        # Test set evaluation if provided
        test_performance = ""
        if X_val is not None and y_val is not None and self.using_custom_validation:
            test_score = self.best_pipeline.score(X_val, y_val)
            test_performance = f"""
## 🧪 **Final Validation Performance**
- **Validation Score**: {test_score:.4f}
- **Training Score**: {self.best_pipeline.score(X_val.iloc[:len(X_val)//2] if hasattr(X_val, 'iloc') else X_val[:len(X_val)//2], y_val.iloc[:len(y_val)//2] if hasattr(y_val, 'iloc') else y_val[:len(y_val)//2]) if len(X_val) > 1 else "N/A"}"""

        # Create text summary
        summary_text = f"""
# 🎯 Hyperparameter Optimization Summary

## 📊 **Performance Metrics**
- **Best Validation Score**: {best_validation_score:.4f}
- **Best Training Score**: {best_training_score:.4f}
- **Overfitting Gap**: {best_overfitting_gap:.4f}
- **Model Stability**: {stability_info}

## 🔍 **Optimization Analysis**
- **Total Trials**: {len(self.training_history)}
- **Best Trial Number**: {best_trial_idx + 1}
- **Trials to Best**: {trials_to_best}
- **Score Improvement**: {score_improvement:.4f}

## ⚙️ **Best Hyperparameters**
{self._format_params_for_display(best_trial["params"])}

## 📈 **Validation Details**
{cv_details}
{test_performance}

## 🎲 **Convergence Analysis**
- **Convergence Rate**: {(trials_to_best / len(self.training_history)) * 100:.1f}% of trials needed
- **Final vs Initial**: {((best_validation_score - validation_scores[0]) / validation_scores[0] * 100):+.1f}% improvement

---
*Generated after {len(self.training_history)} optimization trials using {'custom validation split' if self.using_custom_validation else 'cross-validation'}*
"""

        # Log the text summary
        self.writer.add_text("Experiment_Summary", summary_text, global_step=0)

        # Individual scalar metrics
        self.writer.add_scalar(
            "Summary/Best_Validation_Score", best_validation_score)
        self.writer.add_scalar(
            "Summary/Best_Training_Score", best_training_score)
        self.writer.add_scalar("Summary/Best_Trial_Number", best_trial_idx)
        self.writer.add_scalar("Summary/Total_Trials",
                               len(self.training_history))
        self.writer.add_scalar(
            "Summary/Best_Trial_Overfitting_Gap", best_overfitting_gap)
        self.writer.add_scalar(
            "Summary/Total_Score_Improvement", score_improvement)
        self.writer.add_scalar("Summary/Trials_To_Best", trials_to_best)

        # Add stability only for CV
        if not self.using_custom_validation and "fold_scores" in best_trial:
            best_stability = 1.0 - np.std(best_trial["fold_scores"])
            self.writer.add_scalar(
                "Summary/Best_Trial_Stability", best_stability)

        # Hyperparameter logging
        best_params = self._flatten_params(best_trial["params"])
        best_metrics = {
            "validation_accuracy": best_validation_score,
            "training_accuracy": best_training_score,
            "overfitting_gap": best_overfitting_gap,
        }

        if not self.using_custom_validation and "fold_scores" in best_trial:
            best_metrics["stability"] = 1.0 - np.std(best_trial["fold_scores"])

        self.writer.add_hparams(best_params, best_metrics)
        self.writer.flush()
        print("✅ Optimization summary logged!")

    def _format_params_for_display(self, params):
        """Format parameters for readable display in text summary"""
        if not params:
            return "- No hyperparameters optimized"

        formatted = []
        for group, param_set in params.items():
            if isinstance(param_set, dict):
                formatted.append(f"**{group}:**")
                for key, val in param_set.items():
                    formatted.append(f"  - {key}: `{val}`")
            else:
                formatted.append(f"- **{group}**: `{param_set}`")

        return "\n".join(formatted)

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
