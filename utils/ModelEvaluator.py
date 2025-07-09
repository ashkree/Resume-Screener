import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    brier_score_loss, precision_recall_fscore_support
)

class ModelEvaluator:
    def __init__(self, writer=None):
        """
        Parameters:
        - writer: shared SummaryWriter instance (optional)
        """
        self.writer = writer

    def evaluate_splits(self, pipeline, splits, split_names=None, class_labels=None):
        """
        Evaluate pipeline on multiple splits and log scalar metrics.
        Returns detailed results for each split.
        """
        results = []
        if split_names is None:
            split_names = [f"Split {i+1}" for i in range(len(splits))]

        for (X, y), name in zip(splits, split_names):
            # Basic predictions and metrics
            preds = pipeline.predict(X)
            result = self._compute_metrics(y, preds, name, class_labels=class_labels)
            
            # Print summary
            self._print_evaluation_summary(result, name)
            
            # Log scalar metrics to shared TensorBoard writer
            if self.writer:
                self._log_scalar_metrics(result, name)
            
            # Probability-based metrics (if available)
            if hasattr(pipeline, "predict_proba"):
                prob_metrics = self._compute_probability_metrics(pipeline, X, y, name)
                result.update(prob_metrics)
                if self.writer:
                    self._log_probability_metrics(prob_metrics, name)

            results.append(result)

        if self.writer:
            self.writer.flush()

        return results

    def _compute_metrics(self, y_true, y_pred, split_name, class_labels=None):
        """Compute basic classification metrics"""
        # Core metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Get class names for the classification report
        target_names = self._get_class_names(len(np.unique(y_true)), class_labels)
        report = classification_report(y_true, y_pred, digits=4, output_dict=True, 
                                    target_names=target_names)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        
        # Class-specific recall (diagonal of confusion matrix normalized)
        class_recalls = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        class_recalls = np.nan_to_num(class_recalls)  # Handle division by zero
        
        return {
            "split": split_name,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
            "per_class_support": support,
            "class_recalls": class_recalls,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "y_true": y_true,
            "y_pred": y_pred,
            "class_labels": target_names
        }

    def _get_class_names(self, num_classes, class_labels=None):
        """Get class names based on number of classes"""
        if class_labels:
            return class_labels
        elif num_classes == 2:
            return ["Good Fit", "No Fit"]
        elif num_classes == 3:
            return ["Good Fit", "Potential Fit", "No Fit"]
        else:
            return [f"Class {i}" for i in range(num_classes)]

    def _compute_probability_metrics(self, pipeline, X, y, split_name):
        """Compute probability-based metrics"""
        try:
            probs = pipeline.predict_proba(X)
            confidences = np.max(probs, axis=1)
            
            prob_metrics = {
                "probabilities": probs,
                "confidences": confidences,
                "mean_confidence": np.mean(confidences),
                "std_confidence": np.std(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
            }
            
            # Binary classification specific metrics
            if probs.shape[1] == 2:
                brier_score = brier_score_loss(y, probs[:, 1])
                prob_metrics["brier_score"] = brier_score
                
                # Entropy (uncertainty measure)
                epsilon = 1e-15  # Small value to avoid log(0)
                probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
                entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
                prob_metrics["mean_entropy"] = np.mean(entropy)
                prob_metrics["std_entropy"] = np.std(entropy)
            
            return prob_metrics
            
        except Exception as e:
            print(f"⚠️ Failed to compute probability metrics for {split_name}: {e}")
            return {}

    def _log_scalar_metrics(self, result, split_name):
        """Log scalar metrics using dot notation for clean TensorBoard tabs"""
        
        # === MAIN PERFORMANCE METRICS ===
        self.writer.add_scalar(f"Performance.{split_name}.Accuracy", result["accuracy"])
        self.writer.add_scalar(f"Performance.{split_name}.Macro_F1", result["macro_f1"])
        self.writer.add_scalar(f"Performance.{split_name}.Micro_F1", result["micro_f1"])
        
        # === PRECISION/RECALL BREAKDOWN ===
        self.writer.add_scalar(f"Metrics.{split_name}.Macro_Precision", result["macro_precision"])
        self.writer.add_scalar(f"Metrics.{split_name}.Macro_Recall", result["macro_recall"])
        self.writer.add_scalar(f"Metrics.{split_name}.Micro_Precision", result["micro_precision"])
        self.writer.add_scalar(f"Metrics.{split_name}.Micro_Recall", result["micro_recall"])
        
        # === PER-CLASS BREAKDOWN ===
        class_labels = result["class_labels"]
        for i, (precision, recall, f1, support) in enumerate(zip(
            result["per_class_precision"], 
            result["per_class_recall"], 
            result["per_class_f1"],
            result["per_class_support"]
        )):
            class_name = class_labels[i] if i < len(class_labels) else f"Class_{i}"
            # Clean class names (remove spaces)
            clean_class_name = class_name.replace(" ", "_")
            
            self.writer.add_scalar(f"PerClass.{split_name}.{clean_class_name}.Precision", precision)
            self.writer.add_scalar(f"PerClass.{split_name}.{clean_class_name}.Recall", recall)
            self.writer.add_scalar(f"PerClass.{split_name}.{clean_class_name}.F1", f1)
            self.writer.add_scalar(f"PerClass.{split_name}.{clean_class_name}.Support", support)

    def _log_probability_metrics(self, prob_metrics, split_name):
        """Log probability metrics using dot notation"""
        if not prob_metrics:
            return
        
        # === CONFIDENCE METRICS ===
        self.writer.add_scalar(f"Confidence.{split_name}.Mean", prob_metrics["mean_confidence"])
        self.writer.add_scalar(f"Confidence.{split_name}.Std", prob_metrics["std_confidence"])
        self.writer.add_scalar(f"Confidence.{split_name}.Min", prob_metrics["min_confidence"])
        self.writer.add_scalar(f"Confidence.{split_name}.Max", prob_metrics["max_confidence"])
        
        # === CALIBRATION METRICS ===
        if "brier_score" in prob_metrics:
            self.writer.add_scalar(f"Calibration.{split_name}.Brier_Score", prob_metrics["brier_score"])
        
        if "mean_entropy" in prob_metrics:
            self.writer.add_scalar(f"Uncertainty.{split_name}.Mean_Entropy", prob_metrics["mean_entropy"])
            self.writer.add_scalar(f"Uncertainty.{split_name}.Std_Entropy", prob_metrics["std_entropy"])

    def _print_evaluation_summary(self, result, split_name):
        """Print evaluation summary to console"""
        print(f"\n--- {split_name} Evaluation ---")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Micro F1: {result['micro_f1']:.4f}")
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        report = result['classification_report']

        # Formatting constants
        CLASS_WIDTH = 15
        METRIC_WIDTH = 10
        PRECISION = 4

        # Header
        print(f"{'':>{CLASS_WIDTH}} {'precision':>{METRIC_WIDTH}} {'recall':>{METRIC_WIDTH}} {'f1-score':>{METRIC_WIDTH}} {'support':>{METRIC_WIDTH}}")
        print()

        # Per-class metrics
        for class_name in sorted(report.keys()):
            if class_name not in ['accuracy', 'macro avg', 'micro avg', 'weighted avg']:
                metrics = report[class_name]
                print(f"{class_name:>{CLASS_WIDTH}} {metrics['precision']:>{METRIC_WIDTH}.{PRECISION}f} {metrics['recall']:>{METRIC_WIDTH}.{PRECISION}f} "
                    f"{metrics['f1-score']:>{METRIC_WIDTH}.{PRECISION}f} {int(metrics['support']):>{METRIC_WIDTH}}")

        print()

        # Overall accuracy
        if 'accuracy' in report:
            total_support = int(report['macro avg']['support'])
            print(f"{'accuracy':>{CLASS_WIDTH}} {'':>{METRIC_WIDTH}} {'':>{METRIC_WIDTH}} {report['accuracy']:>{METRIC_WIDTH}.{PRECISION}f} {total_support:>{METRIC_WIDTH}}")

        # Average metrics
        for avg_type in ['macro avg', 'micro avg', 'weighted avg']:
            if avg_type in report:
                metrics = report[avg_type]
                print(f"{avg_type:>{CLASS_WIDTH}} {metrics['precision']:>{METRIC_WIDTH}.{PRECISION}f} {metrics['recall']:>{METRIC_WIDTH}.{PRECISION}f} "
                    f"{metrics['f1-score']:>{METRIC_WIDTH}.{PRECISION}f} {int(metrics['support']):>{METRIC_WIDTH}}")
        
        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])

    def log_training_history(self, training_history, experiment_name="default"):
        """Log training history with dot notation"""
        if not training_history or not self.writer:
            return
            
        # Calculate running best for optimization tracking
        running_best_scores = []
        best_so_far = 0
        
        for trial in training_history:
            trial_num = trial["trial_number"]
            mean_score = trial["mean_score"]
            std_score = np.std(trial["fold_scores"])
            
            # Update running best
            if mean_score > best_so_far:
                best_so_far = mean_score
            running_best_scores.append(best_so_far)
            
            # === CV PERFORMANCE ===
            self.writer.add_scalar(f"CrossValidation.{experiment_name}.Mean_Accuracy", mean_score, trial_num)
            self.writer.add_scalar(f"CrossValidation.{experiment_name}.Stability", 1.0 - std_score, trial_num)
            
            # === OPTIMIZATION TRACKING ===
            self.writer.add_scalar(f"Optimization.{experiment_name}.Running_Best", best_so_far, trial_num)
            
            # === HYPERPARAMETER LOGGING ===
            flat_params = self._flatten_params(trial["params"])
            hparam_metrics = {
                "accuracy": mean_score,
                "stability": 1.0 - std_score,
            }
            
            if "training_accuracy" in trial:
                hparam_metrics["training_accuracy"] = trial["training_accuracy"]
                hparam_metrics["overfitting_gap"] = trial["training_accuracy"] - mean_score
            
            self.writer.add_hparams(flat_params, hparam_metrics)
        
        self.writer.flush()

    def _flatten_params(self, params):
        """Flatten nested parameter dictionary and convert unsupported types"""
        flat_params = {}
        for group, param_set in params.items():
            if isinstance(param_set, dict):
                for key, val in param_set.items():
                    if isinstance(val, tuple):
                        val = str(val)
                    elif val is None:
                        val = "None"
                    elif not isinstance(val, (int, float, str, bool)):
                        val = str(val)
                    flat_params[f"{group}.{key}"] = val
            else:
                if isinstance(param_set, tuple):
                    param_set = str(param_set)
                elif param_set is None:
                    param_set = "None"
                elif not isinstance(param_set, (int, float, str, bool)):
                    param_set = str(param_set)
                flat_params[group] = param_set
        return flat_params

    def log_custom_metric(self, metric_name, value, split_name=None, step=None):
        """Log a custom scalar metric"""
        if not self.writer:
            return
            
        if split_name:
            metric_name = f"{split_name}.{metric_name}"
        
        if step is not None:
            self.writer.add_scalar(metric_name, value, step)
        else:
            self.writer.add_scalar(metric_name, value)
        
        self.writer.flush()