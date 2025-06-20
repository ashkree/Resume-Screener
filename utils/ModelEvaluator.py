# ============================================================================
# MODEL EVALUATOR - Comprehensive evaluation and analysis
# ============================================================================

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis
    """
    
    def __init__(self, class_names=None, verbose=True):
        self.class_names = class_names or ['Good Fit', 'No Fit', 'Potential Fit']
        self.verbose = verbose
        self.evaluation_history = {}
    
    def calculate_comprehensive_metrics(
        self, 
        y_true, 
        y_pred, 
        y_proba=None
    ) -> Dict[str, float]:
        """Calculate all relevant metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
        }
        
        # Add per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'f1_{class_name.lower().replace(" ", "_")}'] = f1_per_class[i]
            metrics[f'precision_{class_name.lower().replace(" ", "_")}'] = precision_per_class[i]
            metrics[f'recall_{class_name.lower().replace(" ", "_")}'] = recall_per_class[i]
        
        # Add AUC if probabilities available (for multi-class)
        if y_proba is not None:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
            except:
                pass  # Skip if not applicable
        
        return metrics
    
    def detect_overfitting(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detect overfitting patterns"""
        gaps = {}
        severity = "none"
        
        for metric in ['accuracy', 'f1_weighted']:
            if metric in train_metrics and metric in val_metrics:
                gap = train_metrics[metric] - val_metrics[metric]
                gaps[f'{metric}_gap'] = gap
                
                if gap > threshold:
                    if gap > 0.2:
                        severity = "severe"
                    elif gap > 0.15:
                        severity = "high"
                    else:
                        severity = "moderate"
        
        return {
            'gaps': gaps,
            'severity': severity,
            'max_gap': max(gaps.values()) if gaps else 0
        }
    
    def evaluate_model(
        self,
        model,
        X_train, y_train,
        X_val=None, y_val=None,
        X_test=None, y_test=None,
        model_name=None
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        model_name = model_name or getattr(model, 'name', 'Unknown')
        
        if self.verbose:
            print(f"📊 Evaluating {model_name}...")
        
        results = {
            'model_name': model_name,
            'evaluation_timestamp': time.time()
        }
        
        # Training set evaluation
        train_pred = model.predict(X_train)
        train_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                train_proba = model.predict_proba(X_train)
            except:
                pass
        
        train_metrics = self.calculate_comprehensive_metrics(y_train, train_pred, train_proba)
        results['train_metrics'] = train_metrics
        
        # Validation set evaluation
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    val_proba = model.predict_proba(X_val)
                except:
                    pass
            
            val_metrics = self.calculate_comprehensive_metrics(y_val, val_pred, val_proba)
            results['val_metrics'] = val_metrics
            
            # Overfitting analysis
            overfitting = self.detect_overfitting(train_metrics, val_metrics)
            results['overfitting_analysis'] = overfitting
        
        # Test set evaluation
        if X_test is not None and y_test is not None:
            test_pred = model.predict(X_test)
            test_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    test_proba = model.predict_proba(X_test)
                except:
                    pass
            
            test_metrics = self.calculate_comprehensive_metrics(y_test, test_pred, test_proba)
            results['test_metrics'] = test_metrics
            
            # Detailed classification report
            results['classification_report'] = classification_report(
                y_test, test_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, test_pred)
        
        # Feature importance (if available) - handle both wrapped and direct models
        feature_importance = self._extract_feature_importance(model)
        if feature_importance:
            results['feature_importance'] = feature_importance
        
        # Store evaluation
        self.evaluation_history[model_name] = results
        
        if self.verbose:
            print(f"✅ Evaluation completed for {model_name}")
            print(f"   Train Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_weighted']:.4f}")
            if 'val_metrics' in results:
                val_acc = results['val_metrics']['accuracy']
                val_f1 = results['val_metrics']['f1_weighted']
                gap = results['overfitting_analysis']['max_gap']
                print(f"   Val Acc: {val_acc:.4f}, F1: {val_f1:.4f} (max gap: {gap:.4f})")
            if 'test_metrics' in results:
                test_acc = results['test_metrics']['accuracy']
                test_f1 = results['test_metrics']['f1_weighted']
                print(f"   Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        return results
    
    def _extract_feature_importance(self, model):
        """Extract feature importance from various model types"""
        # Try wrapped model first (model.model)
        actual_model = getattr(model, 'model', model)
        
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
            return {
                'type': 'tree_importance',
                'importances': importances,
                'top_10_indices': np.argsort(importances)[-10:][::-1],
                'mean_importance': importances.mean(),
                'std_importance': importances.std()
            }
        elif hasattr(actual_model, 'coef_'):
            coefs = np.abs(actual_model.coef_[0] if actual_model.coef_.ndim > 1 else actual_model.coef_)
            return {
                'type': 'linear_coef',
                'coefficients': coefs,
                'top_10_indices': np.argsort(coefs)[-10:][::-1],
                'mean_coef': coefs.mean(),
                'std_coef': coefs.std()
            }
        
        return None
    
    def compare_models(self, models_list=None) -> pd.DataFrame:
        """Compare multiple model evaluations"""
        
        data_source = models_list if models_list else self.evaluation_history
        
        if not data_source:
            print("No evaluations available.")
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, results in data_source.items():
            row = {
                'Model': name,
                'Train_Acc': results['train_metrics']['accuracy'],
                'Train_F1': results['train_metrics']['f1_weighted'],
            }
            
            if 'val_metrics' in results:
                row.update({
                    'Val_Acc': results['val_metrics']['accuracy'],
                    'Val_F1': results['val_metrics']['f1_weighted'],
                    'Overfit_Severity': results['overfitting_analysis']['severity'],
                    'Max_Gap': results['overfitting_analysis']['max_gap']
                })
            
            if 'test_metrics' in results:
                row.update({
                    'Test_Acc': results['test_metrics']['accuracy'],
                    'Test_F1': results['test_metrics']['f1_weighted'],
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by test F1 if available, otherwise validation F1
        if 'Test_F1' in df.columns and not df['Test_F1'].isna().all():
            sort_col = 'Test_F1'
        elif 'Val_F1' in df.columns and not df['Val_F1'].isna().all():
            sort_col = 'Val_F1'
        else:
            sort_col = 'Train_F1'
        
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        
        if self.verbose:
            print(f"\n🏆 MODEL EVALUATION COMPARISON (sorted by {sort_col}):")
            print("=" * 90)
            print(df.to_string(index=False, float_format='%.4f'))
        
        return df
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """Plot confusion matrix for a model"""
        if model_name not in self.evaluation_history:
            print(f"No evaluation found for {model_name}")
            return
        
        results = self.evaluation_history[model_name]
        if 'confusion_matrix' not in results:
            print(f"No confusion matrix available for {model_name}")
            return
        
        cm = results['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self, metric='f1_weighted', splits=['train', 'val', 'test']):
        """Plot metric comparison across models and splits"""
        if not self.evaluation_history:
            print("No evaluations available.")
            return
        
        data = []
        for model_name, results in self.evaluation_history.items():
            for split in splits:
                metrics_key = f'{split}_metrics'
                if metrics_key in results and metric in results[metrics_key]:
                    data.append({
                        'Model': model_name,
                        'Split': split.title(),
                        'Score': results[metrics_key][metric]
                    })
        
        if not data:
            print(f"No data available for metric: {metric}")
            return
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Score', hue='Split')
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_overfitting_analysis(self):
        """Plot overfitting gaps across all models"""
        if not self.evaluation_history:
            print("No evaluations available.")
            return
        
        models = []
        acc_gaps = []
        f1_gaps = []
        severities = []
        
        for model_name, results in self.evaluation_history.items():
            if 'overfitting_analysis' in results:
                analysis = results['overfitting_analysis']
                models.append(model_name)
                acc_gaps.append(analysis['gaps'].get('accuracy_gap', 0))
                f1_gaps.append(analysis['gaps'].get('f1_weighted_gap', 0))
                severities.append(analysis['severity'])
        
        if not models:
            print("No overfitting analysis data available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy gaps
        colors = ['green' if s == 'none' else 'yellow' if s == 'moderate' 
                 else 'orange' if s == 'high' else 'red' for s in severities]
        
        ax1.bar(models, acc_gaps, color=colors, alpha=0.7)
        ax1.set_title('Accuracy Overfitting Gaps')
        ax1.set_ylabel('Train - Val Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax1.legend()
        
        # F1 gaps
        ax2.bar(models, f1_gaps, color=colors, alpha=0.7)
        ax2.set_title('F1-Weighted Overfitting Gaps')
        ax2.set_ylabel('Train - Val F1')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_report(self, model_name):
        """Print detailed evaluation report"""
        if model_name not in self.evaluation_history:
            print(f"No evaluation found for {model_name}")
            return
        
        results = self.evaluation_history[model_name]
        
        print(f"\n📋 DETAILED EVALUATION REPORT: {model_name}")
        print("=" * 60)
        
        # Performance summary
        print(f"📊 PERFORMANCE SUMMARY:")
        for split in ['train', 'val', 'test']:
            metrics_key = f'{split}_metrics'
            if metrics_key in results:
                metrics = results[metrics_key]
                print(f"   {split.upper():<6}: Acc={metrics['accuracy']:.4f}, "
                      f"F1-W={metrics['f1_weighted']:.4f}, "
                      f"F1-M={metrics['f1_macro']:.4f}")
        
        # Overfitting analysis
        if 'overfitting_analysis' in results:
            overfit = results['overfitting_analysis']
            print(f"\n⚠️  OVERFITTING ANALYSIS:")
            print(f"   Severity: {overfit['severity']}")
            print(f"   Max Gap: {overfit['max_gap']:.4f}")
            if overfit['gaps']:
                for gap_name, gap_value in overfit['gaps'].items():
                    print(f"   {gap_name}: {gap_value:.4f}")
        
        # Per-class performance
        if 'test_metrics' in results:
            print(f"\n🎯 PER-CLASS PERFORMANCE (Test Set):")
            for class_name in self.class_names:
                clean_name = class_name.lower().replace(" ", "_")
                f1_key = f'f1_{clean_name}'
                precision_key = f'precision_{clean_name}'
                recall_key = f'recall_{clean_name}'
                
                f1 = results['test_metrics'].get(f1_key, 'N/A')
                precision = results['test_metrics'].get(precision_key, 'N/A')
                recall = results['test_metrics'].get(recall_key, 'N/A')
                
                if f1 != 'N/A':
                    print(f"   {class_name:<12}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
                else:
                    print(f"   {class_name:<12}: No metrics available")
        
        # Feature importance
        if 'feature_importance' in results:
            importance = results['feature_importance']
            print(f"\n🔍 FEATURE IMPORTANCE ({importance['type']}):")
            print(f"   Top 5 indices: {importance['top_10_indices'][:5]}")
            if 'importances' in importance:
                print(f"   Mean importance: {importance['mean_importance']:.4f}")
                print(f"   Std importance: {importance['std_importance']:.4f}")
            elif 'coefficients' in importance:
                print(f"   Mean coefficient: {importance['mean_coef']:.4f}")
                print(f"   Std coefficient: {importance['std_coef']:.4f}")
        
        # Classification report summary
        if 'classification_report' in results:
            print(f"\n📊 CLASSIFICATION REPORT SUMMARY:")
            report = results['classification_report']
            print(f"   Macro Avg: F1={report['macro avg']['f1-score']:.4f}, "
                  f"P={report['macro avg']['precision']:.4f}, "
                  f"R={report['macro avg']['recall']:.4f}")
            print(f"   Weighted Avg: F1={report['weighted avg']['f1-score']:.4f}, "
                  f"P={report['weighted avg']['precision']:.4f}, "
                  f"R={report['weighted avg']['recall']:.4f}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all evaluations"""
        if not self.evaluation_history:
            return {'message': 'No evaluations available'}
        
        # Collect metrics across all models
        metrics_data = {
            'val_accuracy': [],
            'val_f1_weighted': [],
            'test_accuracy': [],
            'test_f1_weighted': [],
            'overfitting_gaps': [],
            'overfitting_severities': []
        }
        
        for results in self.evaluation_history.values():
            if 'val_metrics' in results:
                metrics_data['val_accuracy'].append(results['val_metrics']['accuracy'])
                metrics_data['val_f1_weighted'].append(results['val_metrics']['f1_weighted'])
                
                if 'overfitting_analysis' in results:
                    metrics_data['overfitting_gaps'].append(results['overfitting_analysis']['max_gap'])
                    metrics_data['overfitting_severities'].append(results['overfitting_analysis']['severity'])
            
            if 'test_metrics' in results:
                metrics_data['test_accuracy'].append(results['test_metrics']['accuracy'])
                metrics_data['test_f1_weighted'].append(results['test_metrics']['f1_weighted'])
        
        # Calculate summary statistics
        summary = {
            'total_models_evaluated': len(self.evaluation_history),
            'models_with_validation': len(metrics_data['val_accuracy']),
            'models_with_test': len(metrics_data['test_accuracy']),
        }
        
        # Add statistics for metrics with data
        for metric, values in metrics_data.items():
            if values and metric != 'overfitting_severities':
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_best'] = np.max(values)
                summary[f'{metric}_worst'] = np.min(values)
        
        # Overfitting severity counts
        if metrics_data['overfitting_severities']:
            severity_counts = pd.Series(metrics_data['overfitting_severities']).value_counts().to_dict()
            summary['overfitting_severity_distribution'] = severity_counts
        
        return summary
    
    def export_results(self, filepath: str, format='pickle'):
        """Export evaluation results to file"""
        import pickle
        import json
        
        if format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.evaluation_history, f)
        elif format.lower() == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for model_name, results in self.evaluation_history.items():
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        json_results[key] = value.tolist()
                    elif isinstance(value, dict):
                        json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in value.items()}
                    else:
                        json_results[key] = value
                json_data[model_name] = json_results
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError("Format must be 'pickle' or 'json'")
        
        if self.verbose:
            print(f"💾 Evaluation results exported to: {filepath}")


# USAGE EXAMPLES
if __name__ == "__main__":
    print("""
    🔧 USAGE EXAMPLES:
    
    # Initialize evaluator
    evaluator = ModelEvaluator(class_names=['Good Fit', 'No Fit', 'Potential Fit'])
    
    # Evaluate a model
    results = evaluator.evaluate_model(
        model=trained_model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_name="RandomForest_Chi2"
    )
    
    # Compare models
    comparison_df = evaluator.compare_models()
    
    # Visualizations
    evaluator.plot_confusion_matrix("RandomForest_Chi2")
    evaluator.plot_metrics_comparison('f1_weighted')
    evaluator.plot_overfitting_analysis()
    
    # Detailed analysis
    evaluator.print_detailed_report("RandomForest_Chi2")
    summary = evaluator.get_evaluation_summary()
    
    # Export results
    evaluator.export_results("model_evaluations.pkl")
    """)