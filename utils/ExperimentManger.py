import datetime
import numpy as np
import joblib
import json
import os
from torch.utils.tensorboard import SummaryWriter
from .ModelEvaluator import ModelEvaluator
from .ModelTrainer import ModelTrainer
from .ModelVisualiser import ModelVisualiser

class Experiment:
    def __init__(self, 
                 name, 
                 description, 
                 pipeline_factory=None, 
                 param_space=None):
        
        self.name = name
        self.description = description
        self.pipeline_factory = pipeline_factory
        self.param_space = param_space
        self.timestamp = None
        self.results = None
        self.training_history = None
        self.status = "Untrained"

class ExperimentManager:
    def __init__(self, log_dir, class_labels):
        self.experiments = []
        self.log_dir = log_dir
        self.class_labels = class_labels
        self.writer = None  

    def run_experiment(self, experiment, splits, **kwargs):
        """
        Trains and evaluates a pipeline, stores results and training history.
        
        Parameters:
        - experiment: Experiment object
        - train_data: tuple of (X_train, y_train)
        - create_visualizations: whether to create visualization plots
        - **kwargs: additional parameters for ModelTrainer
            - n_trials: number of optimization trials (default: 20)
            - cv_folds: number of CV folds (default: 5)
            - scoring: scoring metric (default: 'accuracy')
            - random_state: random seed (default: 42)
        """
        print(f"\n=== Running Experiment: {experiment.name} ===")

        # Create centralized SummaryWriter for this experiment
        experiment_log_dir = f"{self.log_dir}/{experiment.name}"
        self.writer = SummaryWriter(log_dir=experiment_log_dir)
        

        # === Phase 0: INSTATIATE HANDLERS ===
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

        try:

            # === PHASE 1: MODEL TRAINING ===
            trainer = ModelTrainer(
                pipeline_factory = experiment.pipeline_factory,
                param_space = experiment.param_space,
                n_trials=kwargs.get("n_trials"),
                cv_folds=kwargs.get("cv_folds"),
                scoring=kwargs.get("scoring"),
                random_state=kwargs.get("random_state"),
                writer = self.writer
            )

            trained_pipeline = trainer.train(X_train, y_train, X_val, y_val, optimise=kwargs.get("optimise"))
            experiment.training_history = trainer.training_history
            # === PHASE 2: MODEL EVALUATION ===
            evaluator = ModelEvaluator(self.writer)

            results = evaluator.evaluate_test_set(trained_pipeline, X_test, y_test, self.class_labels)
            
            # === PHASE 3: MODEL EVALUATION ===

            visualiser = ModelVisualiser(self.writer)

            visualiser.create_evaluation_plot(
                X=X_test,
                predictions=results["y_pred"],
                true_labels=y_test,
                confusion_matrix=results["confusion_matrix"],
                classification_report=results["classification_report"],
                split_name="Test",
                pipeline=trained_pipeline,
                class_labels=self.class_labels
            )

            visualiser.log_training_visualization(
                experiment.training_history, 
                experiment_name=experiment.name
            )

            # === PHASE 4: FINISH EXPERIMENT ===
            self._finalize_experiment(experiment=experiment, results=results, trained_pipeline=trained_pipeline)

        except Exception as e:
            print(f"❌ Experiment '{experiment.name}' failed: {str(e)}")
            experiment.status = "Failed"
            raise
        finally:
            # Always close the writer when done with this experiment
            if self.writer:
                self.writer.close()
                self.writer = None

        print(f"\n✅ Experiment '{experiment.name}' completed and logged.\n")
        return experiment

    def _finalize_experiment(self, experiment, results, trained_pipeline):
        """Phase 4: Store final experiment results and metadata"""
        print("💾 Finalizing experiment...")
        
        # Update experiment object with all results
        experiment.results = results
        experiment.status = "Completed"
        experiment.timestamp = datetime.datetime.now().isoformat()
        
        # Add to experiment history
        self.experiments.append(experiment)
        
        # Log experiment metadata
        if self.writer:
            text = f"""
# {experiment.name}
- Status {experiment.status}
- {experiment.timestamp}
{experiment.description}
"""
            self.writer.add_text("Experiment_information", text)

            self.writer.flush()

        # Save best model along with metadata and results


        model_dir = f"../models/{experiment.name}"
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(trained_pipeline, f"{model_dir}/model.pkl")
        metadata = {
            "name": experiment.name,
            "description": experiment.description,
            "timestamp": experiment.timestamp
        }
        with open( f"{model_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4, default=str)
        
        with open( f"{model_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4, default=str)


    def list_experiments(self):
        """Display a summary of all experiments."""
        if not self.experiments:
            print("📭 No experiments run yet.")
            return

        print("\n=== Experiment History ===")
        for idx, exp in enumerate(self.experiments):
            status_emoji = "✅" if exp.status == "Completed" else "❌" if exp.status == "Failed" else "⏳"
            print(f"{idx+1}. {status_emoji} {exp.name} | Status: {exp.status} | Timestamp: {exp.timestamp}")

    def get_experiment(self, name):
        """Retrieve experiment by name."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        print(f"⚠️ No experiment found with name '{name}'")
        return None
    
    def compare_experiments(self, experiment_names=None, metric="accuracy"):
        """
        Compare experiments and print summary statistics.
        
        Parameters:
        - experiment_names: list of experiment names to compare (None for all)
        - metric: metric to compare ("accuracy", "macro_f1", etc.)
        """
        if experiment_names is None:
            experiments_to_compare = [exp for exp in self.experiments if exp.status == "Completed"]
        else:
            experiments_to_compare = [self.get_experiment(name) for name in experiment_names]
            experiments_to_compare = [exp for exp in experiments_to_compare if exp is not None and exp.status == "Completed"]
        
        if not experiments_to_compare:
            print("No completed experiments to compare.")
            return
        
        print(f"\n=== Experiment Comparison ({metric}) ===")
        print(f"{'Experiment':<30} {'Validation':<12} {'Test':<12} {'Status':<10}")
        print("-" * 65)
        
        for exp in experiments_to_compare:
            if exp.results and len(exp.results) >= 2:
                val_score = exp.results[0].get(metric, "N/A")
                test_score = exp.results[1].get(metric, "N/A")
                
                val_str = f"{val_score:.4f}" if isinstance(val_score, float) else str(val_score)
                test_str = f"{test_score:.4f}" if isinstance(test_score, float) else str(test_score)
                status_emoji = "✅" if exp.status == "Completed" else "❌"
                
                print(f"{exp.name:<30} {val_str:<12} {test_str:<12} {status_emoji} {exp.status}")
            else:
                print(f"{exp.name:<30} {'N/A':<12} {'N/A':<12} ❌ No Results")

    def get_best_experiment(self, metric="accuracy", split="test"):
        """
        Get the best performing experiment based on a specific metric.
        
        Parameters:
        - metric: metric to optimize ("accuracy", "macro_f1", etc.)
        - split: which split to use for comparison ("test", "validation")
        
        Returns:
        - best experiment object
        """
        if not self.experiments:
            print("No experiments available.")
            return None
        
        # Only consider completed experiments
        completed_experiments = [exp for exp in self.experiments if exp.status == "Completed"]
        if not completed_experiments:
            print("No completed experiments available.")
            return None
        
        valid_experiments = []
        for exp in completed_experiments:
            if exp.results and exp.status == "Completed":
                if split.lower() == "validation" and len(exp.results) >= 1:
                    score = exp.results[0].get(metric)
                elif split.lower() == "test" and len(exp.results) >= 2:
                    score = exp.results[1].get(metric)
                else:
                    continue
                    
                if score is not None:
                    valid_experiments.append((exp, score))
        
        if not valid_experiments:
            print(f"No experiments with valid {metric} scores on {split} split.")
            return None
        
        # Find best experiment (highest score)
        best_exp, best_score = max(valid_experiments, key=lambda x: x[1])
        print(f"🏆 Best experiment: {best_exp.name} with {metric} = {best_score:.4f} on {split}")
        return best_exp

    def export_experiment_summary(self, filename=None):
        """Export experiment results to CSV for external analysis"""
        if not self.experiments:
            print("No experiments to export.")
            return
        
        import pandas as pd
        
        # Collect experiment data
        exp_data = []
        for exp in self.experiments:
            row = {
                'name': exp.name,
                'status': exp.status,
                'timestamp': exp.timestamp,
                'description': exp.description
            }
            
            # Add performance metrics if available
            if exp.results:
                for i, (split_name, result) in enumerate(zip(exp.split_names or [], exp.results)):
                    row[f'{split_name}_accuracy'] = result.get('accuracy', None)
                    row[f'{split_name}_macro_f1'] = result.get('macro_f1', None)
                    row[f'{split_name}_micro_f1'] = result.get('micro_f1', None)
            
            exp_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(exp_data)
        filename = filename or f"experiment_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"📊 Experiment summary exported to: {filename}")
        return df

    def close(self):
        """Close any open writers"""
        if self.writer:
            self.writer.close()
            self.writer = None