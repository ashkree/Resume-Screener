import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from torch.utils.tensorboard import SummaryWriter

class ModelVisualiser:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def create_evaluation_plot(self, predictions, true_labels, confusion_matrix, 
                             classification_report, split_name, pipeline, X, 
                             class_labels=None):
        """Create a comprehensive evaluation plot combining all visualizations"""
        labels = class_labels if class_labels else self._get_class_names(confusion_matrix.shape[0])
        
        # Determine layout based on available data
        has_proba = hasattr(pipeline, "predict_proba")
        
        if has_proba:
            try:
                probs = pipeline.predict_proba(X)
                # 2x3 grid for comprehensive visualization
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
            except:
                # Fallback if predict_proba fails
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                has_proba = False
        else:
            # 2x2 grid for basic visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

        # Plot 1: Confusion Matrix
        self._plot_confusion_matrix(confusion_matrix, labels, split_name, axes[0])

        # Plot 2: Per-Class Metrics Bar Chart
        self._plot_metrics_bars(classification_report, labels, split_name, axes[1])

        # Plot 3: Classification Report Table
        self._plot_classification_table(classification_report, labels, split_name, axes[2])

        # Plot 4: Confidence Histogram (if available)
        if has_proba and len(axes) > 3:
            self._plot_confidence_histogram(pipeline, X, split_name, axes[3])
        else:
            self._plot_unavailable(axes[3], split_name, "Confidence")

        # Plot 5: Calibration Curve (if binary classification)
        if has_proba and len(axes) > 4:
            self._plot_calibration_curve(pipeline, X, true_labels, split_name, axes[4])
        elif len(axes) > 4:
            self._plot_unavailable(axes[4], split_name, "Calibration")

        # Plot 6: Feature Importance (if available)
        if len(axes) > 5:
            self._plot_feature_importance(pipeline, split_name, axes[5])

        # Hide any unused subplots
        for i in range(len(axes)):
            if i >= 6:
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'{split_name} - Complete Evaluation Summary', fontsize=16, y=0.98)
        
        # Log to TensorBoard
        self.writer.add_figure(f"{split_name}/Complete_Evaluation", fig)
        self.writer.flush()
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return fig

    def _plot_confusion_matrix(self, conf_matrix, labels, split_name, ax):
        """Plot confusion matrix"""
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{split_name} - Confusion Matrix')

    def _plot_metrics_bars(self, report, labels, split_name, ax):
        """Plot per-class metrics as bar chart"""
        precisions = []
        recalls = []
        f1s = []
        
        for label in labels:
            label_str = str(label)
            if label_str in report:
                precisions.append(report[label_str]['precision'])
                recalls.append(report[label_str]['recall'])
                f1s.append(report[label_str]['f1-score'])
            else:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)

        x = np.arange(len(labels))
        width = 0.25
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'{split_name} - Per-Class Metrics')
        ax.legend()

    def _plot_classification_table(self, report, labels, split_name, ax):
        """Plot classification report as table"""
        ax.axis('off')
        table_data = []
        
        for label in labels:
            label_str = str(label)
            if label_str in report:
                scores = report[label_str]
                table_data.append([
                    label, 
                    f"{scores['precision']:.3f}",
                    f"{scores['recall']:.3f}",
                    f"{scores['f1-score']:.3f}",
                    f"{int(scores['support'])}"
                ])
        
        # Add overall metrics
        if 'accuracy' in report:
            table_data.append(['Overall', '', '', f"{report['accuracy']:.3f}", ''])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Class', 'Precision', 'Recall', 'F1', 'Support'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title(f'{split_name} - Classification Report')

    def _plot_confidence_histogram(self, pipeline, X, split_name, ax):
        """Plot prediction confidence histogram"""
        try:
            probs = pipeline.predict_proba(X)
            confidences = np.max(probs, axis=1)
            ax.hist(confidences, bins=20, range=(0, 1), color='skyblue', 
                   edgecolor='black', alpha=0.7)
            ax.set_title(f'{split_name} - Prediction Confidence')
            ax.set_xlabel('Max Predicted Probability')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add mean confidence line
            mean_conf = np.mean(confidences)
            ax.axvline(mean_conf, color='red', linestyle='--', 
                      label=f'Mean: {mean_conf:.3f}')
            ax.legend()
        except:
            self._plot_unavailable(ax, split_name, "Confidence")

    def _plot_calibration_curve(self, pipeline, X, y, split_name, ax):
        """Plot calibration curve for binary classification"""
        try:
            probs = pipeline.predict_proba(X)
            if probs.shape[1] == 2:
                prob_true, prob_pred = calibration_curve(y, probs[:, 1], n_bins=10)
                ax.plot(prob_pred, prob_true, marker='o', label='Model', linewidth=2)
                ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', alpha=0.7)
                ax.set_title(f'{split_name} - Calibration Curve')
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                self._plot_unavailable(ax, split_name, "Calibration", 
                                     "Only available for\nbinary classification")
        except:
            self._plot_unavailable(ax, split_name, "Calibration")

    def _plot_feature_importance(self, pipeline, split_name, ax):
        """Plot feature importance or coefficients"""
        try:
            # Try to get feature importance from the final estimator
            final_estimator = pipeline.named_steps['clf']
            
            if hasattr(final_estimator, 'feature_importances_'):
                # For tree-based models
                importances = final_estimator.feature_importances_
                # Show top 20 features
                top_indices = np.argsort(importances)[-20:]
                ax.barh(range(len(top_indices)), importances[top_indices])
                ax.set_title(f'{split_name} - Top 20 Feature Importances')
                ax.set_xlabel('Importance')
                ax.set_yticks(range(len(top_indices)))
                ax.set_yticklabels([f'Feature {i}' for i in top_indices])
                
            elif hasattr(final_estimator, 'coef_'):
                # For linear models
                coef = final_estimator.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for multi-class
                abs_coef = np.abs(coef)
                top_indices = np.argsort(abs_coef)[-20:]
                ax.barh(range(len(top_indices)), abs_coef[top_indices])
                ax.set_title(f'{split_name} - Top 20 Feature Coefficients')
                ax.set_xlabel('|Coefficient|')
                ax.set_yticks(range(len(top_indices)))
                ax.set_yticklabels([f'Feature {i}' for i in top_indices])
                
            else:
                self._plot_unavailable(ax, split_name, "Feature Importance")
                
        except:
            self._plot_unavailable(ax, split_name, "Feature Importance")

    def _plot_unavailable(self, ax, split_name, metric_name, message=None):
        """Plot placeholder for unavailable metrics"""
        if message is None:
            message = f'{metric_name} data\nnot available'
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{split_name} - {metric_name} (N/A)')
        ax.set_xticks([])
        ax.set_yticks([])

    def log_training_visualization(self, training_history, experiment_name="default"):
        """Create visualization for training history"""
        if not training_history:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Mean CV accuracy over trials
        trial_nums = [trial["trial_number"] for trial in training_history]
        mean_scores = [trial["mean_score"] for trial in training_history]
        
        axes[0].plot(trial_nums, mean_scores, marker='o', linewidth=2)
        axes[0].set_title('Optimization Progress')
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('Mean CV Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Add best score line
        best_score = max(mean_scores)
        axes[0].axhline(best_score, color='red', linestyle='--', alpha=0.7,
                       label=f'Best: {best_score:.4f}')
        axes[0].legend()
        
        # Plot 2: Score distribution
        axes[1].hist(mean_scores, bins=min(15, len(mean_scores)//2 + 1), 
                    alpha=0.7, edgecolor='black')
        axes[1].axvline(best_score, color='red', linestyle='--', 
                       label=f'Best: {best_score:.4f}')
        axes[1].set_title('Score Distribution')
        axes[1].set_xlabel('Mean CV Accuracy')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'{experiment_name} - Training History', fontsize=14, y=1.02)
        
        # Log to TensorBoard
        self.writer.add_figure(f"{experiment_name}/Training_History", fig)
        self.writer.flush()
        
        plt.close(fig)
        return fig

    def _get_class_names(self, num_classes):
        """Get class names based on number of classes"""
        if num_classes == 2:
            # Binary classification: Good Fit vs No Fit
            return ["Good Fit", "No Fit"]
        elif num_classes == 3:
            # Multiclass: Good Fit, Potential Fit, No Fit
            return ["Good Fit", "Potential Fit", "No Fit"]
        else:
            # Fallback to numeric labels
            return [f"Class {i}" for i in range(num_classes)]

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()