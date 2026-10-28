import os
import sys
import torch
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from util.data_processing import read_cached_graph_dataset
from torch_geometric.loader import DataLoader as GraphDataLoader

class ModelEvaluator:
    def __init__(self, model_path, dataset_config, base_path=".", sparsity=None):
        """Initialize the model evaluator with target scaling support."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_path = base_path
        self.dataset_config = dataset_config
        self.sparsity = sparsity  # Store sparsity
        
        # Load the model and target scaling information
        self.model, self.target_scaling_info = self._load_model_with_scaling(model_path)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")
        print(f"Using sparsity level: {sparsity}")
        print(f"Using dataset type: {dataset_config['dataset_type']}")
        
        if self.target_scaling_info['use_target_scaling']:
            print(f"Target scaling enabled - Mean: {self.target_scaling_info['target_scaler_mean']:.4f}, "
                  f"Std: {self.target_scaling_info['target_scaler_std']:.4f}")
        else:
            print("Target scaling disabled")
        
        # Load dataset with sparsity and dataset type parameters
        self.data_struct = read_cached_graph_dataset(
            num_train=dataset_config['num_train'],
            num_test=dataset_config['num_test'],
            dataset_name=dataset_config['dataset_name'],
            parent_dir=base_path,
            sparsity=sparsity,  # Pass sparsity to data loading function
            dataset_type=dataset_config['dataset_type']  # Pass dataset type to data loading function
        )

    def _load_model_with_scaling(self, model_path):
        """Load a saved model and extract target-scaling info, with clean fallbacks."""
        model_dir = os.path.dirname(model_path)

        # 1) Load checkpoint safely
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"Error loading checkpoint at {model_path}: {e}")
            raise

        # 2) Initialize scaling info
        target_scaling_info = {
            'use_target_scaling': False,
            'target_scaler_mean': 0.0,
            'target_scaler_std': 1.0,
        }

        # 3) Pick up scaling info from checkpoint if present
        if isinstance(checkpoint, dict) and 'target_scaling' in checkpoint:
            target_scaling_info.update(checkpoint['target_scaling'])
            print(f"Target scaling info found in checkpoint: {target_scaling_info}")
        else:
            # Try to infer from experiment_config.yaml (do NOT crash if missing)
            experiment_config_path = os.path.join(model_dir, "experiment_config.yaml")
            if os.path.exists(experiment_config_path):
                try:
                    import yaml
                    with open(experiment_config_path, 'r') as f:
                        exp_config = yaml.safe_load(f)
                    # Prefer an explicit boolean flag if your config uses one
                    uses_scaling = False
                    if isinstance(exp_config, dict):
                        # Adjust these keys to your real schema
                        uses_scaling = (
                            exp_config['use_target_scaling']
                            or (exp_config['training'] or {})['use_target_scaling']
                        )
                    else:
                        # Last-resort heuristic
                        uses_scaling = 'use_target_scaling' in str(exp_config)

                    if uses_scaling:
                        print("Found evidence of target scaling in experiment config; computing from data...")
                        target_scaling_info = self._compute_target_scaling_from_data()
                except Exception as e:
                    print(f"Could not parse experiment config '{experiment_config_path}': {e}")

        # 4) Build / load the model
        # Case A: checkpoint is a state-dict bundle for a separately saved model definition
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_complete_path = os.path.join(model_dir, "model_complete.pth")
            if not os.path.exists(model_complete_path):
                raise FileNotFoundError(
                    f"'model_state_dict' found, but '{model_complete_path}' is missing; "
                    "cannot reconstruct model class to load weights."
                )
            try:
                model = torch.load(model_complete_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model class from model_complete.pth and applied state_dict from checkpoint.")
                return model.to(self.device), target_scaling_info
            except Exception as e:
                print(f"Failed to load model from '{model_complete_path}' or apply state_dict: {e}")
                raise

        # Case B: checkpoint is already a full torch model object (rare but possible)
        if hasattr(checkpoint, "state_dict") and callable(getattr(checkpoint, "state_dict")):
            print("Checkpoint appears to be a full model object; using it directly.")
            return checkpoint.to(self.device), target_scaling_info

        # Case C: as a last fallback, try model_complete.pth alone
        model_complete_path = os.path.join(model_dir, "model_complete.pth")
        if os.path.exists(model_complete_path):
            try:
                model = torch.load(model_complete_path, map_location=self.device)
                print("Fell back to loading full model from model_complete.pth.")
                return model.to(self.device), target_scaling_info
            except Exception as e:
                print(f"Failed to load full model from '{model_complete_path}': {e}")
                raise

        # If none of the above succeeded, bail out explicitly
        raise RuntimeError(
            f"Could not construct model from '{model_path}'. "
            "Expected one of: (a) dict with 'model_state_dict' and model_complete.pth present, "
            "(b) full model object checkpoint, or (c) model_complete.pth present."
        )

    def _compute_target_scaling_from_data(self):
        """Compute target scaling parameters from training data if not saved."""
        print("Computing target scaling parameters from training data...")
        
        # Collect all training targets
        train_targets = []
        train_dataset = self.data_struct["geometric"]["sgcn_train_data"]
        
        for data in train_dataset:
            if hasattr(data, 'y'):
                train_targets.append(data.y.item() if data.y.dim() == 0 else data.y.cpu().numpy())
        
        train_targets = np.array(train_targets)
        mean = np.mean(train_targets)
        std = np.std(train_targets)
        
        print(f"Computed target scaling - Mean: {mean:.4f}, Std: {std:.4f}")
        
        return {
            'use_target_scaling': True,
            'target_scaler_mean': mean,
            'target_scaler_std': std
        }

    def _inverse_transform_targets(self, scaled_targets):
        """Convert scaled targets back to original scale."""
        if self.target_scaling_info['use_target_scaling']:
            mean = self.target_scaling_info['target_scaler_mean']
            std = self.target_scaling_info['target_scaler_std']
            return scaled_targets * std + mean
        return scaled_targets
    
    def _transform_targets(self, original_targets):
        """Convert original targets to scaled targets."""
        if self.target_scaling_info['use_target_scaling']:
            mean = self.target_scaling_info['target_scaler_mean']
            std = self.target_scaling_info['target_scaler_std']
            return (original_targets - mean) / std
        return original_targets
    
    def _move_data_to_device(self, data):
        """Move graph data to target device."""
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.to(self.device)
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index = data.edge_index.to(self.device)
        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.to(self.device)
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos.to(self.device)
        if hasattr(data, 'batch') and data.batch is not None:
            data.batch = data.batch.to(self.device)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.to(self.device)
        if hasattr(data, 'demographics') and data.demographics is not None:
            data.demographics = data.demographics.to(self.device)
        return data
    
    def evaluate_dataset(self, dataset_type='test', batch_size=64):
        """Evaluate the model on train or test dataset with proper target scaling."""
        # Select the appropriate dataset
        if dataset_type == 'train':
            dataset = self.data_struct["geometric"]["sgcn_train_data"]
        elif dataset_type == 'test':
            dataset = self.data_struct["geometric"]["sgcn_test_data"]
        else:
            raise ValueError("dataset_type must be 'train' or 'test'")
        
        # Create data loader
        data_loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Collect predictions and true values (in original scale)
        all_predictions_orig = []
        all_true_values_orig = []
        
        print(f"Evaluating on {dataset_type} dataset...")
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data = self._move_data_to_device(data)
                
                # Get predictions (in scaled space)
                predictions_scaled = self.model(data)
                
                # Convert to numpy
                if isinstance(predictions_scaled, torch.Tensor):
                    if predictions_scaled.dim() == 0:
                        predictions_scaled = predictions_scaled.unsqueeze(0)
                    elif predictions_scaled.dim() == 2 and predictions_scaled.size(1) == 1:
                        predictions_scaled = predictions_scaled.squeeze(-1)

                predictions_scaled_np = predictions_scaled.cpu().numpy()
                targets_np = data.y.cpu().numpy()
                
                # Convert predictions back to original scale
                predictions_orig = self._inverse_transform_targets(predictions_scaled_np)
                
                # Targets are already in original scale
                targets_orig = targets_np

                # Collect results in original scale
                all_predictions_orig.append(predictions_orig)
                all_true_values_orig.append(targets_orig)

        # Concatenate all results
        predictions = np.concatenate(all_predictions_orig)
        true_values = np.concatenate(all_true_values_orig)
        
        # Calculate metrics in original scale
        r2 = r2_score(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        correlation = np.corrcoef(true_values, predictions)[0, 1]
        
        results = {
            'dataset_type': dataset_type,
            'n_samples': len(predictions),
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'predictions': predictions,
            'true_values': true_values,
            'target_scaling_used': self.target_scaling_info['use_target_scaling']
        }
        
        return results

    def plot_predictions(self, results, save_path=None, show_plot=False):
        """
        Plot predictions vs true values.
        
        Args:
            results: Results dictionary from evaluate_dataset
            save_path: Optional path to save the plot
            show_plot: Whether to show the plot (requires interactive backend)
        """
        predictions = results['predictions']
        true_values = results['true_values']
        r2 = results['r2_score']
        dataset_type = results['dataset_type']
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(true_values, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Predictions vs True Values ({dataset_type.title()} Set)\nR² = {r2:.4f}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Make plot square
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def plot_training_curves(self, experiment_dir, save_path=None, show_plot=False):
        """
        Plot training and test MSE curves from saved pickle files.
        
        Args:
            experiment_dir: Directory containing train_mse.pk and test_mse.pk files
            save_path: Optional path to save the plot
            show_plot: Whether to show the plot (requires interactive backend)
        """
        train_mse_path = os.path.join(experiment_dir, "train_mse.pk")
        test_mse_path = os.path.join(experiment_dir, "test_mse.pk")
        
        # Check if files exist
        if not os.path.exists(train_mse_path):
            print(f"Train MSE file not found: {train_mse_path}")
            return
        if not os.path.exists(test_mse_path):
            print(f"Test MSE file not found: {test_mse_path}")
            return
        
        # Load the MSE data
        with open(train_mse_path, 'rb') as f:
            train_mse = pickle.load(f)
        with open(test_mse_path, 'rb') as f:
            test_mse = pickle.load(f)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        epochs = range(1, len(train_mse) + 1)
        
        plt.plot(epochs, train_mse, 'b-', linewidth=2, label='Training MSE', alpha=0.8)
        plt.plot(epochs, test_mse, 'r-', linewidth=2, label='Test MSE', alpha=0.8)
        
        # Add markers for better visibility
        plt.plot(epochs[::max(1, len(epochs)//20)], 
                [train_mse[i] for i in range(0, len(train_mse), max(1, len(epochs)//20))], 
                'bo', markersize=4, alpha=0.7)
        plt.plot(epochs[::max(1, len(epochs)//20)], 
                [test_mse[i] for i in range(0, len(test_mse), max(1, len(epochs)//20))], 
                'ro', markersize=4, alpha=0.7)
        
        # Formatting
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=14)
        plt.title('Training and Test MSE Over Time', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        min_train_mse = min(train_mse)
        min_test_mse = min(test_mse)
        final_train_mse = train_mse[-1]
        final_test_mse = test_mse[-1]
        
        # Add text box with statistics
        stats_text = f'Final Train MSE: {final_train_mse:.6f}\n'
        stats_text += f'Final Test MSE: {final_test_mse:.6f}\n'
        stats_text += f'Best Train MSE: {min_train_mse:.6f}\n'
        stats_text += f'Best Test MSE: {min_test_mse:.6f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
        
        # Print summary
        print(f"\nTraining Curves Summary:")
        print(f"{'='*50}")
        print(f"Total epochs: {len(train_mse)}")
        print(f"Final Training MSE: {final_train_mse:.6f}")
        print(f"Final Test MSE: {final_test_mse:.6f}")
        print(f"Best Training MSE: {min_train_mse:.6f} (epoch {train_mse.index(min_train_mse) + 1})")
        print(f"Best Test MSE: {min_test_mse:.6f} (epoch {test_mse.index(min_test_mse) + 1})")
        print(f"{'='*50}")

    def print_evaluation_summary(self, results):
        """Print evaluation summary including epoch information."""
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        # Check if we have epoch information
        model_dir = os.path.dirname(results.get('model_path', ''))
        training_info_path = os.path.join(model_dir, "training_info.txt")
        
        if os.path.exists(training_info_path):
            print(f"Training Information:")
            with open(training_info_path, 'r') as f:
                for line in f:
                    print(f"  {line.strip()}")
            print()
        
        for dataset_type in ['train', 'test']:
            if f'{dataset_type}_results' in results:
                result = results[f'{dataset_type}_results']
                print(f"{dataset_type.upper()} SET RESULTS:")
                print(f"  R² Score: {result['r2_score']:.4f}")
                print(f"  RMSE: {result['rmse']:.4f}")
                print(f"  MAE: {result['mae']:.4f}")
                print(f"  MSE: {result['mse']:.4f}")
                print()
        
        if 'plots_saved_to' in results:
            print(f"Plots saved to: {results['plots_saved_to']}")
        
        print(f"{'='*80}")


def load_config_from_experiment(experiment_dir):
    """
    Load configuration from experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory (e.g., "Experiments_FC_normal_gridsearch/run_DSST_regression_lr_0.0001/sgcn")
    
    Returns:
        Configuration dictionary with sparsity, architecture, and dataset type information
    """
    # Try to load from saved experiment config first
    config_path = os.path.join(experiment_dir, "experiment_config.yaml")
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                exp_config = yaml.safe_load(f)
            dataset_config = exp_config.get('dataset_config', {})
            split_config = exp_config.get('split_config', {})
            
            # Extract sparsity from experiment_id if available
            experiment_id = exp_config.get('experiment_id', '')
            sparsity = 100  # default
            if 'sparsity_' in experiment_id:
                try:
                    sparsity_part = experiment_id.split('sparsity_')[1].split('_')[0]
                    sparsity = int(sparsity_part)
                except (IndexError, ValueError):
                    print(f"Warning: Could not extract sparsity from experiment_id: {experiment_id}")
            
            # Get architecture type from config or infer from directory structure
            architecture_type = exp_config['architecture_type']
            
            # Get dataset type from config or infer from directory structure
            dataset_type = exp_config['dataset_type']

            # If not in config, try to infer from experiment directory path
            if 'Experiments_' in experiment_dir:
                try:
                    # Extract from path like "Experiments_FC_residual_gridsearch" or "Experiments_MIND_normal_gridsearch"
                    experiments_folder = experiment_dir.split('Experiments_')[1].split('_gridsearch')[0]
                    parts = experiments_folder.split('_')
                    if len(parts) >= 2:
                        inferred_dataset_type = parts[0]  # FC or MIND
                        inferred_architecture_type = parts[1]  # normal, residual, inject
                        
                        if inferred_dataset_type in ['FC', 'MIND']:
                            dataset_type = inferred_dataset_type
                        if inferred_architecture_type in ['normal', 'residual', 'inject']:
                            architecture_type = inferred_architecture_type
                except:
                    pass  # Keep defaults
            
            return {
                'dataset_name': dataset_config.get('dataset_name', 'custom_dataset_selfloops_True_edgeft_None_norm_True'),
                'num_train': split_config['train'],
                'num_test': split_config['test'],
                'sparsity': sparsity,
                'architecture_type': architecture_type,
                'dataset_type': dataset_type
            }
        except Exception as e:
            print(f"Warning: Could not load experiment config: {e}")
            return {
                'dataset_name': 'custom_dataset_selfloops_True_edgeft_None_norm_True',
                'num_train': 1000,  # fallback values
                'num_test': 250,
                'sparsity': 100,
                'architecture_type': 'normal',
                'dataset_type': 'FC'
            }

def main():
    """Main evaluation function."""
    # Configuration - Update paths according to setup
    base_path = "/Users/baileyng/MIND_models/QuantNets"
    
    # Example: evaluate SGCN model with specific architecture and dataset type
    experiment_id = "GF_FC_regression_GATv2_normal_config_1_lr_0.0001_epochs_496_scheduler_step"  # Update this to match your experiment
    model_path = os.path.join(base_path, "Experiments_FC_normal_gridsearch", f"run_{experiment_id}", "sgcn", "model.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available experiments:")
        
        # Check different architecture and dataset type experiment directories
        for dataset_type in ['FC', 'MIND']:
            for arch_type in ['normal', 'residual', 'inject']:
                experiments_dir = os.path.join(base_path, f"Experiments_{dataset_type}_{arch_type}_gridsearch")
                if os.path.exists(experiments_dir):
                    print(f"\n{dataset_type} {arch_type.upper()} Architecture Experiments:")
                    for exp in os.listdir(experiments_dir):
                        exp_path = os.path.join(experiments_dir, exp)
                        if os.path.isdir(exp_path):
                            print(f"  - {exp}")
                            # Check for model files
                            sgcn_model = os.path.join(exp_path, "sgcn", "model.pth")
                            qgcn_model = os.path.join(exp_path, "qgcn", "model.pth")
                            if os.path.exists(sgcn_model):
                                print(f"    SGCN model: ✓")
                            if os.path.exists(qgcn_model):
                                print(f"    QGCN model: ✓")
        return
    
    # Load dataset configuration including architecture type and dataset type
    dataset_config = load_config_from_experiment(os.path.dirname(model_path))
    sparsity = dataset_config['sparsity']  # Extract sparsity from config
    architecture_type = dataset_config['architecture_type']  # Extract architecture type
    dataset_type = dataset_config['dataset_type']  # Extract dataset type
    
    try:
        # Initialize evaluator with sparsity, architecture type, and dataset type info
        evaluator = ModelEvaluator(model_path, dataset_config, base_path, sparsity=sparsity)
        
        print(f"Using architecture type: {architecture_type}")
        print(f"Using dataset type: {dataset_type}")
        
        # Evaluate on test set
        test_results = evaluator.evaluate_dataset('test', batch_size=64)
        evaluator.print_evaluation_summary(test_results)
        
        # Evaluate on training set for comparison
        train_results = evaluator.evaluate_dataset('train', batch_size=64)
        evaluator.print_evaluation_summary(train_results)
        
        # Plot results
        plot_dir = os.path.dirname(model_path)
        evaluator.plot_predictions(test_results, save_path=os.path.join(plot_dir, "test_predictions.png"))
        evaluator.plot_predictions(train_results, save_path=os.path.join(plot_dir, "train_predictions.png"))
        
        # Plot training curves from saved MSE files
        evaluator.plot_training_curves(plot_dir, save_path=os.path.join(plot_dir, "training_curves.png"))
        
        # Save detailed results
        results_path = os.path.join(plot_dir, "evaluation_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({'train': train_results, 'test': test_results}, f)
        print(f"Detailed results saved to {results_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()