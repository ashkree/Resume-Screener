import datetime
from .ModelEvaluator import ModelEvaluator
from .ModelTrainer import ModelTrainer
from .ModelVisualiser import ModelVisualiser

class Experiment:
    def __init__(self, 
                 name, 
                 description, 
                 pipeline_factory=None, 
                 splits=None, 
                 split_names=None, 
                 param_space=None):
        
        self.name = name
        self.description = description
        self.pipeline_factory = pipeline_factory
        self.param_space = param_space
        self.splits = splits
        self.split_names = split_names
        self.timestamp = None
        self.results = None
        self.training_history = None
        self.status = "Untrained"

class ExperimentManager:
    def __init__(self, log_dir):
        self.experiments = []
        self.log_dir = log_dir

    def run_experiment(self, experiment, train_data, create_visualizations=True):
        """
        Trains and evaluates a pipeline, stores results and training history.
        
        Parameters:
        - experiment: Experiment object
        - train_data: tuple of (X_train, y_train)
        - create_visualizations: whether to create visualization plots
        """
        print(f"\n=== Running Experiment: {experiment.name} ===")

        # Build trainer
        trainer = ModelTrainer(
            pipeline_factory=experiment.pipeline_factory,
            param_space=experiment.param_space
        )

        optimise = experiment.param_space is not None
        trained_pipeline = trainer.train(*train_data, optimise=optimise)

        # Evaluate with metrics
        evaluator = ModelEvaluator(log_dir=f"{self.log_dir}/{experiment.name}")
        results = evaluator.evaluate_splits(
            trained_pipeline, 
            experiment.splits, 
            experiment.split_names
        )

        # Create visualizations if requested
        if create_visualizations:
            visualiser = ModelVisualiser(log_dir=f"{self.log_dir}/{experiment.name}")
            
            # Create evaluation plots for each split
            for result, (X, y), split_name in zip(results, experiment.splits, experiment.split_names):
                preds = trained_pipeline.predict(X)
                visualiser.create_evaluation_plot(
                    predictions=preds,
                    true_labels=y,
                    confusion_matrix=result['confusion_matrix'],
                    classification_report=result['classification_report'],
                    split_name=split_name,
                    pipeline=trained_pipeline,
                    X=X
                )
            
            # Create training history visualization if available
            if trainer.training_history:
                visualiser.log_training_visualization(
                    trainer.training_history, 
                    experiment_name=experiment.name
                )
            
            visualiser.close()

        # Log training history scalars
        if trainer.training_history:
            evaluator.log_training_history(trainer.training_history, experiment_name=experiment.name)

        # Update experiment object
        experiment.results = results
        experiment.training_history = trainer.training_history
        experiment.status = "Completed"
        experiment.timestamp = datetime.datetime.now().isoformat()
        self.experiments.append(experiment)

        # Close evaluator
        evaluator.close()

        print(f"\n✅ Experiment '{experiment.name}' completed and logged.\n")
        return experiment

    def list_experiments(self):
        """
        Display a summary of all experiments.
        """
        if not self.experiments:
            print("📭 No experiments run yet.")
            return

        print("\n=== Experiment History ===")
        for idx, exp in enumerate(self.experiments):
            print(f"{idx+1}. {exp.name} | Status: {exp.status} | Timestamp: {exp.timestamp}")

    def get_experiment(self, name):
        """
        Retrieve experiment by name.
        """
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
            experiments_to_compare = self.experiments
        else:
            experiments_to_compare = [self.get_experiment(name) for name in experiment_names]
            experiments_to_compare = [exp for exp in experiments_to_compare if exp is not None]
        
        if not experiments_to_compare:
            print("No experiments to compare.")
            return
        
        print(f"\n=== Experiment Comparison ({metric}) ===")
        print(f"{'Experiment':<30} {'Validation':<12} {'Test':<12}")
        print("-" * 55)
        
        for exp in experiments_to_compare:
            if exp.results and len(exp.results) >= 2:
                val_score = exp.results[0].get(metric, "N/A")
                test_score = exp.results[1].get(metric, "N/A")
                
                val_str = f"{val_score:.4f}" if isinstance(val_score, float) else str(val_score)
                test_str = f"{test_score:.4f}" if isinstance(test_score, float) else str(test_score)
                
                print(f"{exp.name:<30} {val_str:<12} {test_str:<12}")
            else:
                print(f"{exp.name:<30} {'N/A':<12} {'N/A':<12}")

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
        
        valid_experiments = []
        for exp in self.experiments:
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
        print(f"Best experiment: {best_exp.name} with {metric} = {best_score:.4f} on {split}")
        return best_exp