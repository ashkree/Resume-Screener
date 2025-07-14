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

    def evaluate_test_set(self, pipeline, X_test, y_test, class_labels=None):
        """
        Evaluate pipeline on test set and log comprehensive metrics.
        Returns detailed test results.
        """
        # Basic predictions and metrics
        y_pred = pipeline.predict(X_test)

        # Compute all metrics
        results = self._compute_metrics(y_test, y_pred, class_labels)

        # Print comprehensive summary
        self._print_test_summary(results)

        # Log to TensorBoard
        if self.writer:
            self._log_test_metrics(results)

            # Log comprehensive text summary
            self._log_evaluation_summary(results)

            self.writer.flush()

        return results

    def _compute_metrics(self, y_true, y_pred, class_labels=None):
        """Compute comprehensive classification metrics"""
        # Core metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Get class names
        target_names = self._get_class_names(
            len(np.unique(y_true)), class_labels)

        # Classification report
        report = classification_report(
            y_true, y_pred,
            digits=4,
            output_dict=True,
            target_names=target_names,
            zero_division=0
        )

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

        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": report,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "per_class_f1": f1,
            "per_class_support": support,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "y_true": y_true,
            "y_pred": y_pred,
            "class_labels": target_names,
            "num_samples": len(y_true),
            "num_classes": len(np.unique(y_true))
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

    def _log_test_metrics(self, results):
        """Log test metrics to TensorBoard"""

        # === CORE PERFORMANCE METRICS ===
        self.writer.add_scalar("Test_Performance/Accuracy", results["accuracy"])
        self.writer.add_scalar("Test_Performance/Macro_F1", results["macro_f1"])
        self.writer.add_scalar("Test_Performance/Micro_F1", results["micro_f1"])
        self.writer.add_scalar("Test_Performance/Weighted_F1", results["weighted_f1"])

        # === PRECISION/RECALL METRICS ===
        self.writer.add_scalar("Test_Precision/Macro_Precision", results["macro_precision"])
        self.writer.add_scalar("Test_Precision/Micro_Precision", results["micro_precision"])
        self.writer.add_scalar("Test_Precision/weighted_Precision", results["weighted_precision"])
        
        self.writer.add_scalar("Test_Recall/Macro_Recall", results["macro_recall"])
        self.writer.add_scalar("Test_Recall/Micro_Recall", results["micro_recall"])
        self.writer.add_scalar("Test_Recall/weighted_Recall", results["weighted_recall"])

    def _print_test_summary(self, results):
        """Print comprehensive test evaluation summary"""
        print("\n" + "="*60)
        print("🎯 TEST SET EVALUATION RESULTS")
        print("="*60)

        # === OVERVIEW ===
        print(f"\n📊 OVERVIEW")
        print(f"   Test Samples: {results['num_samples']:,}")
        print(f"   Classes: {results['num_classes']}")
        print(f"   Overall Accuracy: {results['accuracy']:.4f}")

        # === MAIN METRICS ===
        print(f"\n🎯 MAIN PERFORMANCE METRICS")
        print(f"   Macro F1:     {results['macro_f1']:.4f}")
        print(f"   Micro F1:     {results['micro_f1']:.4f}")
        print(f"   Weighted F1:  {results['weighted_f1']:.4f}")

        # === PRECISION/RECALL SUMMARY ===
        print(f"\n📈 PRECISION/RECALL SUMMARY")
        print(
            f"   Macro    - P: {results['macro_precision']:.4f}  R: {results['macro_recall']:.4f}")
        print(
            f"   Micro    - P: {results['micro_precision']:.4f}  R: {results['micro_recall']:.4f}")
        print(
            f"   Weighted - P: {results['weighted_precision']:.4f}  R: {results['weighted_recall']:.4f}")

        # === DETAILED CLASSIFICATION REPORT ===
        print(f"\n📋 DETAILED CLASSIFICATION REPORT")
        self._print_classification_table(results['classification_report'])

        # === CONFUSION MATRIX ===
        print(f"\n🔢 CONFUSION MATRIX")
        print(f"   Rows: True Labels, Columns: Predicted Labels")
        self._print_confusion_matrix(
            results['confusion_matrix'], results['class_labels'])

        print("\n" + "="*60)

    def _print_classification_table(self, report):
        """Print formatted classification report table"""
        # Constants for formatting
        CLASS_WIDTH = 16
        METRIC_WIDTH = 10
        PRECISION = 4

        # Header
        print(f"   {'Class':<{CLASS_WIDTH}} {'Precision':>{METRIC_WIDTH}} {'Recall':>{METRIC_WIDTH}} {'F1-Score':>{METRIC_WIDTH}} {'Support':>{METRIC_WIDTH}}")
        print(
            f"   {'-'*CLASS_WIDTH} {'-'*METRIC_WIDTH} {'-'*METRIC_WIDTH} {'-'*METRIC_WIDTH} {'-'*METRIC_WIDTH}")

        # Per-class metrics
        for class_name in sorted(report.keys()):
            if class_name not in ['accuracy', 'macro avg', 'micro avg', 'weighted avg']:
                metrics = report[class_name]
                print(f"   {class_name:<{CLASS_WIDTH}} {metrics['precision']:>{METRIC_WIDTH}.{PRECISION}f} {metrics['recall']:>{METRIC_WIDTH}.{PRECISION}f} "
                      f"{metrics['f1-score']:>{METRIC_WIDTH}.{PRECISION}f} {int(metrics['support']):>{METRIC_WIDTH}}")

        print(
            f"   {'-'*CLASS_WIDTH} {'-'*METRIC_WIDTH} {'-'*METRIC_WIDTH} {'-'*METRIC_WIDTH} {'-'*METRIC_WIDTH}")

        # Averages
        for avg_type in ['macro avg', 'micro avg', 'weighted avg']:
            if avg_type in report:
                metrics = report[avg_type]
                print(f"   {avg_type:<{CLASS_WIDTH}} {metrics['precision']:>{METRIC_WIDTH}.{PRECISION}f} {metrics['recall']:>{METRIC_WIDTH}.{PRECISION}f} "
                      f"{metrics['f1-score']:>{METRIC_WIDTH}.{PRECISION}f} {int(metrics['support']):>{METRIC_WIDTH}}")

    def _print_confusion_matrix(self, conf_matrix, class_labels):
        """Print formatted confusion matrix"""
        print(f"   Predicted →")

        # Header with class labels
        header = "   True ↓   "
        for label in class_labels:
            header += f"{label[:8]:>8} "
        print(header)

        # Matrix rows
        for i, row in enumerate(conf_matrix):
            if i < len(class_labels):
                row_label = class_labels[i][:8]
                row_str = f"   {row_label:<8} "
                for val in row:
                    row_str += f"{val:>8} "
                print(row_str)

    def _log_evaluation_summary(self, results):
        """Log comprehensive evaluation summary as text to TensorBoard"""
        if not self.writer:
            return

        # Create comprehensive text summary
        summary_text = f"""
# 🎯 Test Set Evaluation Summary

## 📊 **Dataset Overview**
- **Total Samples**: {results['num_samples']:,}
- **Number of Classes**: {results['num_classes']}
- **Class Distribution**: {dict(zip(results['class_labels'], results['per_class_support']))}

## 🎯 **Overall Performance**
- **Accuracy**: {results['accuracy']:.4f}
- **Macro F1**: {results['macro_f1']:.4f}
- **Micro F1**: {results['micro_f1']:.4f}
- **Weighted F1**: {results['weighted_f1']:.4f}

## 📈 **Precision & Recall Breakdown**
- **Macro Precision**: {results['macro_precision']:.4f}
- **Macro Recall**: {results['macro_recall']:.4f}
- **Micro Precision**: {results['micro_precision']:.4f}
- **Micro Recall**: {results['micro_recall']:.4f}
- **Weighted Precision**: {results['weighted_precision']:.4f}
- **Weighted Recall**: {results['weighted_recall']:.4f}

## 🔍 **Per-Class Performance**
{self._format_per_class_metrics(results)}

## 🔢 **Confusion Matrix**
```
{self._format_confusion_matrix_text(results['confusion_matrix'], results['class_labels'])}
```

---
*Test evaluation completed with {results['num_samples']} samples across {results['num_classes']} classes*
"""

        self.writer.add_text("Test_Evaluation_Summary",
                             summary_text, global_step=0)

    def _format_per_class_metrics(self, results):
        """Format per-class metrics for text summary"""
        formatted = []
        for i, (precision, recall, f1, support) in enumerate(zip(
            results["per_class_precision"],
            results["per_class_recall"],
            results["per_class_f1"],
            results["per_class_support"]
        )):
            if i < len(results["class_labels"]):
                class_name = results["class_labels"][i]
                formatted.append(
                    f"- **{class_name}**: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, Support={int(support)}")
        return "\n".join(formatted)

    def _format_confusion_matrix_text(self, conf_matrix, class_labels):
        """Format confusion matrix for text display"""
        lines = []

        # Header
        header = "        "
        for label in class_labels:
            header += f"{label[:8]:>8} "
        lines.append(header)

        # Matrix rows
        for i, row in enumerate(conf_matrix):
            if i < len(class_labels):
                row_label = class_labels[i][:8]
                row_str = f"{row_label:<8} "
                for val in row:
                    row_str += f"{val:>8} "
                lines.append(row_str)

        return "\n".join(lines)
