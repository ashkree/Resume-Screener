import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

class ModelVisualiser:
    def __init__(self, writer=None):
        """
        Parameters:
        - writer: shared SummaryWriter instance (optional)
        """
        self.writer = writer

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
        if self.writer:
            self.writer.add_figure(f"{split_name}/Complete_Evaluation", fig)
            self.writer.flush()
        
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return fig

    def log_training_visualization(self, training_history, experiment_name="default"):
        """Create comprehensive training visualization with training/validation curves"""
        if not training_history:
            return

        # Check if we have training accuracy data (from enhanced ModelTrainer)
        has_training_acc = any("training_accuracy" in trial for trial in training_history)
        
        if has_training_acc:
            # 2x2 grid if we have training vs validation data
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
        else:
            # 1x2 grid for basic optimization plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = [axes] if len(axes.shape) == 0 else axes.flatten()

        # Extract data
        trial_nums = [trial["trial_number"] for trial in training_history]
        validation_scores = [trial["mean_score"] for trial in training_history]
        
        # Plot 1: Training vs Validation Curves (if available)
        if has_training_acc:
            training_scores = [trial["training_accuracy"] for trial in training_history]
            best_validation = [max(validation_scores[:i+1]) for i in range(len(validation_scores))]
            
            axes[0].plot(trial_nums, training_scores, 'o-', label='Training Accuracy', 
                        color='orange', linewidth=2, markersize=4)
            axes[0].plot(trial_nums, validation_scores, 'o-', label='Validation Accuracy', 
                        color='blue', linewidth=2, markersize=4)
            axes[0].plot(trial_nums, best_validation, '--', label='Best Validation', 
                        color='green', linewidth=2, alpha=0.8)
            
            axes[0].set_title('Training vs Validation Curves', fontweight='bold')
            axes[0].set_xlabel('Trial Number')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Add gap analysis
            overfitting_gaps = [t - v for t, v in zip(training_scores, validation_scores)]
            ax_twin = axes[0].twinx()
            ax_twin.fill_between(trial_nums, overfitting_gaps, alpha=0.2, color='red', 
                               label='Overfitting Gap')
            ax_twin.set_ylabel('Overfitting Gap (Train - Val)', color='red')
            ax_twin.tick_params(axis='y', labelcolor='red')
            
            # Plot 2: Optimization Progress
            plot_idx = 1
        else:
            # Plot 1: Basic optimization progress (no training data)
            plot_idx = 0
            
        # Optimization Progress Plot
        axes[plot_idx].plot(trial_nums, validation_scores, 'o-', linewidth=2, markersize=4)
        best_score = max(validation_scores)
        axes[plot_idx].axhline(best_score, color='red', linestyle='--', alpha=0.7,
                              label=f'Best: {best_score:.4f}')
        axes[plot_idx].set_title('Optimization Progress', fontweight='bold')
        axes[plot_idx].set_xlabel('Trial Number')
        axes[plot_idx].set_ylabel('Validation Accuracy')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Plot 3: Score Distribution
        plot_idx += 1
        if plot_idx < len(axes):
            axes[plot_idx].hist(validation_scores, bins=min(15, len(validation_scores)//2 + 1), 
                               alpha=0.7, edgecolor='black', color='skyblue')
            axes[plot_idx].axvline(best_score, color='red', linestyle='--', 
                                  label=f'Best: {best_score:.4f}')
            axes[plot_idx].set_title('Validation Score Distribution', fontweight='bold')
            axes[plot_idx].set_xlabel('Validation Accuracy')
            axes[plot_idx].set_ylabel('Frequency')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)

        # Plot 4: Hyperparameter Impact (if we have 2x2 grid)
        plot_idx += 1
        if plot_idx < len(axes) and has_training_acc:
            self._plot_hyperparameter_impact(training_history, axes[plot_idx])

        # Hide any unused subplots
        for i in range(plot_idx + 1, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.suptitle(f'{experiment_name} - Training Analysis', fontsize=16, y=0.98)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_figure(f"{experiment_name}/Training_Analysis", fig)
            self.writer.flush()
        
        plt.close(fig)
        return fig

    def _plot_hyperparameter_impact(self, training_history, ax):
        """Plot hyperparameter impact on performance"""
        try:
            # Extract C values and corresponding scores (example for logistic regression)
            c_values = []
            scores = []
            
            for trial in training_history:
                params = trial["params"]
                score = trial["mean_score"]
                
                # Look for regularization parameter
                if "clf__C" in params:
                    c_values.append(params["clf__C"])
                    scores.append(score)
            
            if c_values:
                # Create scatter plot
                ax.scatter(c_values, scores, alpha=0.7, s=50)
                ax.set_xscale('log')  # C is usually on log scale
                ax.set_xlabel('Regularization Parameter (C)')
                ax.set_ylabel('Validation Accuracy')
                ax.set_title('Regularization vs Performance', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(c_values) > 3:
                    z = np.polyfit(np.log10(c_values), scores, 1)
                    p = np.poly1d(z)
                    x_trend = np.logspace(np.log10(min(c_values)), np.log10(max(c_values)), 100)
                    ax.plot(x_trend, p(np.log10(x_trend)), "--", alpha=0.8, color='red')
            else:
                self._plot_unavailable(ax, "Hyperparameter", "Impact Analysis")
                
        except Exception as e:
            self._plot_unavailable(ax, "Hyperparameter", "Impact Analysis")

    def create_optimization_summary(self, training_history, experiment_name="default"):
        """Create a dedicated optimization summary plot"""
        if not training_history:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Extract data
        trial_nums = [trial["trial_number"] for trial in training_history]
        validation_scores = [trial["mean_score"] for trial in training_history]
        cv_stds = [np.std(trial["fold_scores"]) for trial in training_history]
        
        # Plot 1: Main optimization curve with confidence bands
        axes[0].plot(trial_nums, validation_scores, 'o-', linewidth=2, label='Validation Score')
        
        # Add confidence bands (mean ± std)
        upper_bound = [score + std for score, std in zip(validation_scores, cv_stds)]
        lower_bound = [score - std for score, std in zip(validation_scores, cv_stds)]
        axes[0].fill_between(trial_nums, lower_bound, upper_bound, alpha=0.2, label='CV Std Range')
        
        best_score = max(validation_scores)
        axes[0].axhline(best_score, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_score:.4f}')
        axes[0].set_title('Optimization with Uncertainty', fontweight='bold')
        axes[0].set_xlabel('Trial Number')
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Model Stability (1 - CV std)
        stability_scores = [1 - std for std in cv_stds]
        axes[1].plot(trial_nums, stability_scores, 'o-', color='green', linewidth=2)
        axes[1].set_title('Model Stability Across CV Folds', fontweight='bold')
        axes[1].set_xlabel('Trial Number')
        axes[1].set_ylabel('Stability Score (1 - CV Std)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Performance vs Stability Scatter
        axes[2].scatter(validation_scores, stability_scores, alpha=0.7, s=50)
        axes[2].set_xlabel('Validation Accuracy')
        axes[2].set_ylabel('Stability Score')
        axes[2].set_title('Performance vs Stability Trade-off', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Highlight best trials
        best_trial_idx = validation_scores.index(best_score)
        axes[2].scatter(validation_scores[best_trial_idx], stability_scores[best_trial_idx], 
                       color='red', s=100, marker='*', label='Best Trial')
        axes[2].legend()
        
        # Plot 4: Cumulative Best Score
        cumulative_best = [max(validation_scores[:i+1]) for i in range(len(validation_scores))]
        axes[3].plot(trial_nums, cumulative_best, 'o-', color='purple', linewidth=2)
        axes[3].set_title('Cumulative Best Score', fontweight='bold')
        axes[3].set_xlabel('Trial Number')
        axes[3].set_ylabel('Best Score So Far')
        axes[3].grid(True, alpha=0.3)
        
        # Add plateauing detection
        plateau_threshold = 0.001
        for i in range(1, len(cumulative_best)):
            if cumulative_best[i] - cumulative_best[i-1] < plateau_threshold:
                axes[3].axvspan(trial_nums[i-1], trial_nums[i], alpha=0.1, color='red')
        
        plt.tight_layout()
        plt.suptitle(f'{experiment_name} - Optimization Summary', fontsize=16, y=0.98)
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_figure(f"{experiment_name}/Optimization_Summary", fig)
            self.writer.flush()
        
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

    def _get_class_names(self, num_classes):
        """Get class names based on number of classes"""
        if num_classes == 2:
            return ["Good Fit", "No Fit"]
        elif num_classes == 3:
            return ["Good Fit", "Potential Fit", "No Fit"]
        else:
            return [f"Class {i}" for i in range(num_classes)]