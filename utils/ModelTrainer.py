import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold

class ModelTrainer:
    def __init__(self, pipeline_factory, param_space=None, n_trials=20, cv_folds=5,
                 scoring='accuracy', random_state=42):
        """
        Parameters:
        - pipeline_factory: callable that accepts a param dict and returns a pipeline (e.g. sklearn Pipeline)
        - param_space: callable defining the Optuna hyperparameter space (trial -> dict)
        - n_trials: number of optimisation trials
        - cv_folds: number of Stratified K-Fold splits
        - scoring: metric to optimise
        - random_state: seed for reproducibility
        """
        self.pipeline_factory = pipeline_factory
        self.param_space = param_space
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

        self.best_pipeline = None
        self.best_score = None
        self.best_params = None
        self.training_history = []

    def train(self, X, y, optimise=False):
        """
        Train the model using standard fit or Optuna-based optimisation with cross-validation.
        Returns the best fitted pipeline.
        """
        self.training_history = []

        # Keep data in original format (don't convert to numpy arrays)
        # The pipeline should handle the conversion internally

        if not optimise or self.param_space is None:
            # Simple training with default parameters
            pipeline = self.pipeline_factory({})
            pipeline.fit(X, y)
            self.best_pipeline = pipeline
            self.best_score = pipeline.score(X, y)
            self.best_params = {}
            return self.best_pipeline

        # Define Optuna objective
        def objective(trial):
            params = self.param_space(trial)
            pipeline = self.pipeline_factory(params)

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            fold_scores = []

            # Convert y to numpy array for indexing in cross-validation
            y_array = y.values if hasattr(y, 'values') else y

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_array)):
                # Use iloc for pandas DataFrames, direct indexing for numpy arrays
                if hasattr(X, 'iloc'):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                
                if hasattr(y, 'iloc'):
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                else:
                    y_train, y_val = y[train_idx], y[val_idx]

                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_val, y_val)
                fold_scores.append(score)

            trial_result = {
                "trial_number": trial.number,
                "params": params,
                "fold_scores": fold_scores,
                "mean_score": np.mean(fold_scores)
            }

            self.training_history.append(trial_result)
            return trial_result["mean_score"]

        # Run optimisation
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        # Final best model - need to construct params properly for pipeline factory
        best_params = study.best_params
        
        # Convert flat params back to nested structure if needed
        structured_params = {}
        for key, value in best_params.items():
            if '.' in key:
                parts = key.split('.')
                if parts[0] not in structured_params:
                    structured_params[parts[0]] = {}
                structured_params[parts[0]][parts[1]] = value
            else:
                structured_params[key] = value

        self.best_params = structured_params
        self.best_score = study.best_value
        self.best_pipeline = self.pipeline_factory(self.best_params)
        self.best_pipeline.fit(X, y)

        return self.best_pipeline