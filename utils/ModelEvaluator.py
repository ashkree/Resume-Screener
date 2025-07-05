import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    brier_score_loss, precision_recall_fscore_support
)
from torch.utils.tensorboard import SummaryWriter

class ModelEvaluator:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

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
            result = self._compute_metrics(y, preds, name)
            
            # Print summary
            self._print_evaluation_summary(result, name)
            
            # Log scalar metrics to TensorBoard
            self._log_scalar_metrics(result, name)
            
            # Probability-based metrics (if available)
            if hasattr(pipeline, "predict_proba"):
                prob_metrics = self._compute_probability_metrics(pipeline, X, y, name)
                result.update(prob_metrics)
                self._log_probability_metrics(prob_metrics, name)

            results.append(result)

        self.writer.flush()
        return results

    def _compute_metrics(self, y_true, y_pred, split_name):
        """Compute basic classification metrics"""
        # Core metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)

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
            "y_pred": y_pred
        }

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
        """Log scalar metrics to TensorBoard"""
        # Core metrics
        self.writer.add_scalar(f"{split_name}/Accuracy", result["accuracy"])
        self.writer.add_scalar(f"{split_name}/Macro_Precision", result["macro_precision"])
        self.writer.add_scalar(f"{split_name}/Macro_Recall", result["macro_recall"])
        self.writer.add_scalar(f"{split_name}/Macro_F1", result["macro_f1"])
        self.writer.add_scalar(f"{split_name}/Micro_Precision", result["micro_precision"])
        self.writer.add_scalar(f"{split_name}/Micro_Recall", result["micro_recall"])
        self.writer.add_scalar(f"{split_name}/Micro_F1", result["micro_f1"])
        
        # Per-class metrics
        for i, (precision, recall, f1, support) in enumerate(zip(
            result["per_class_precision"], 
            result["per_class_recall"], 
            result["per_class_f1"],
            result["per_class_support"]
        )):
            self.writer.add_scalar(f"{split_name}/Class_{i}/Precision", precision)
            self.writer.add_scalar(f"{split_name}/Class_{i}/Recall", recall)
            self.writer.add_scalar(f"{split_name}/Class_{i}/F1", f1)
            self.writer.add_scalar(f"{split_name}/Class_{i}/Support", support)

    def _log_probability_metrics(self, prob_metrics, split_name):
        """Log probability-based metrics to TensorBoard"""
        if not prob_metrics:
            return
            
        # Confidence metrics
        self.writer.add_scalar(f"{split_name}/Mean_Confidence", prob_metrics["mean_confidence"])
        self.writer.add_scalar(f"{split_name}/Std_Confidence", prob_metrics["std_confidence"])
        self.writer.add_scalar(f"{split_name}/Min_Confidence", prob_metrics["min_confidence"])
        self.writer.add_scalar(f"{split_name}/Max_Confidence", prob_metrics["max_confidence"])
        
        # Binary classification metrics
        if "brier_score" in prob_metrics:
            self.writer.add_scalar(f"{split_name}/Brier_Score", prob_metrics["brier_score"])
        
        if "mean_entropy" in prob_metrics:
            self.writer.add_scalar(f"{split_name}/Mean_Entropy", prob_metrics["mean_entropy"])
            self.writer.add_scalar(f"{split_name}/Std_Entropy", prob_metrics["std_entropy"])

    def _print_evaluation_summary(self, result, split_name):
        """Print evaluation summary to console"""
        print(f"\n--- {split_name} Evaluation ---")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Micro F1: {result['micro_f1']:.4f}")
        
        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])

        print("\nClassfication Report:")
        print(classification_report(result["y_true"], result["y_pred"], digits=4))

        

    def log_training_history(self, training_history, experiment_name="default"):
        """Log training history scalar metrics"""
        if not training_history:
            return
            
        for trial in training_history:
            trial_num = trial["trial_number"]
            mean_score = trial["mean_score"]
            std_score = np.std(trial["fold_scores"])

            # Log main metrics
            self.writer.add_scalar(f"{experiment_name}/Validation/Mean_CV_Accuracy", mean_score, trial_num)
            self.writer.add_scalar(f"{experiment_name}/Validation/Std_CV_Accuracy", std_score, trial_num)
            
            # Log individual fold scores
            for fold_idx, score in enumerate(trial["fold_scores"]):
                self.writer.add_scalar(f"{experiment_name}/Validation/Fold_{fold_idx}_Accuracy", score, trial_num)

            # Log hyperparameters (flattened)
            flat_params = self._flatten_params(trial["params"])
            self.writer.add_hparams(flat_params, {
                "accuracy": mean_score,
                "std_accuracy": std_score
            })

        self.writer.flush()

    def _flatten_params(self, params):
        """Flatten nested parameter dictionary"""
        flat_params = {}
        for group, param_set in params.items():
            if isinstance(param_set, dict):
                for key, val in param_set.items():
                    flat_params[f"{group}.{key}"] = val
            else:
                flat_params[group] = param_set
        return flat_params

    def log_custom_metric(self, metric_name, value, split_name=None, step=None):
        """Log a custom scalar metric"""
        if split_name:
            metric_name = f"{split_name}/{metric_name}"
        
        if step is not None:
            self.writer.add_scalar(metric_name, value, step)
        else:
            self.writer.add_scalar(metric_name, value)
        
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()