import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from itertools import product

from util.reproducibility import set_deterministic_training

# Set seed for reproducibility
SEED = 42
generator = set_deterministic_training(SEED)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def load_architecture_models(architecture_type):
    """Dynamically import the appropriate architecture module."""
    architectures = {
        'normal': 'gnn.architectures_normal',
        'residual': 'gnn.architectures_residual',
        'inject': 'gnn.architectures_inject'
    }
    
    if architecture_type not in architectures:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    
    module = __import__(architectures[architecture_type], fromlist=['GraphConvNet', 'GATv2ConvNet'])
    return module.GraphConvNet, module.GATv2ConvNet


def load_qgrn_architecture_model(architecture_type):
    """Dynamically import the QGRN architecture."""
    qgrn_architectures = {
        'normal': 'qgrn.architectures_normal',
        'residual': 'qgrn.architectures_residual',
        'inject': 'qgrn.architectures_inject',
        'injectresidualdropout': 'qgrn.architectures_injectresidualdropout',
        'injectdropout': 'qgrn.architectures_injectdropout',
        'residualdropout': 'qgrn.architectures_residualdropout',
        'injectresidual': 'qgrn.architectures_injectresidual'
    }
    
    if architecture_type not in qgrn_architectures:
        raise ValueError(f"Unknown QGRN architecture type: {architecture_type}")
    
    module = __import__(qgrn_architectures[architecture_type], fromlist=['QGRN'])
    return module.QGRN


def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_model(model_type, dataset_config, architecture_type):
    """Create model based on type specification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == 'QGRN':
        QGRN = load_qgrn_architecture_model(architecture_type)
        return QGRN(
            out_dim=dataset_config['out_dim'],
            in_channels=dataset_config['in_channels'],
            hidden_channels=dataset_config['hidden_channels'],
            out_channels=dataset_config['out_channels'],
            layers_num=dataset_config['layers_num'],
            embedding_dim=dataset_config['embedding_dim'],
            demo_dim=dataset_config['demo_dim'],
            num_sub_kernels=dataset_config['num_sub_kernels'],
            use_shared_bias=dataset_config.get('use_shared_bias', True),  # not in YAML
            edge_attr_dim=dataset_config['edge_attr_dim'],
            pos_descr_dim=dataset_config['pos_descr_dim'],
            quant_net_depth=dataset_config.get('quant_net_depth', 2),  # not in YAML
            quant_net_expansion=dataset_config.get('quant_net_expansion', 2),  # not in YAML
            apply_mixture_net=dataset_config.get('apply_mixture_net', True),  # not in YAML
            mixture_net_depth=dataset_config.get('mixture_net_depth', 2),  # not in YAML
            mixture_net_expansion=dataset_config.get('mixture_net_expansion', 2),  # not in YAML
            aggr="add",
            device=device,
            include_demo=dataset_config['include_demo'],
            dropout_rate=dataset_config['dropout_rate']
        )
    
    GraphConvNet, GATv2ConvNet = load_architecture_models(architecture_type)
    
    model_classes = {
        'GraphConv': (GraphConvNet, {
            'hidden_sf': dataset_config['graph_hidden_sf'],
            'out_sf': dataset_config['graph_out_sf']
        }),
        'GATv2': (GATv2ConvNet, {
            'hidden_sf': dataset_config['gatv2_hidden_sf'],
            'out_sf': dataset_config['gatv2_out_sf'],
            'hidden_heads': dataset_config['gatv2_hidden_heads']
        })
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    ModelClass, extra_params = model_classes[model_type]
    
    # Now using consistent naming - directly from config
    return ModelClass(
        out_dim=dataset_config['out_dim'],
        in_channels=dataset_config['in_channels'],
        out_channels=dataset_config['out_channels'],
        layers_num=dataset_config['layers_num'],
        hidden_channels=dataset_config['hidden_channels'],
        embedding_dim=dataset_config['embedding_dim'],
        include_demo=dataset_config['include_demo'],
        demo_dim=dataset_config['demo_dim'],
        dropout_rate=dataset_config['dropout_rate'],
        **extra_params
    )


def generate_grid_search_configs(run_config, splits_config, datasets_config, dataset_key, model_type):
    """Generate all combinations of grid search parameters."""
    grid_params = run_config['grid_search'][dataset_key]
    
    dropout_rates = grid_params['dropout_rates']
    weight_decays = grid_params['weight_decays']
    layers_nums = grid_params['layers_nums']
    num_sub_kernels_list = grid_params['num_sub_kernels'] if model_type == 'QGRN' else [None]
    lrs = grid_params['lrs']

    base_epochs = run_config['epochs'][dataset_key][0]
    base_scheduler = run_config['schedulers'][dataset_key][0]
    base_early_stopping = run_config['early_stopping'][dataset_key][0]
    base_split = splits_config[dataset_key]
    base_dataset = datasets_config[dataset_key]
    
    configs = []
    for idx, (dropout, wd, layers, kernels, lr) in enumerate(
        product(dropout_rates, weight_decays, layers_nums, num_sub_kernels_list, lrs)
    ):
        dataset_config = base_dataset.copy()
        dataset_config.update({
            'dropout_rate': dropout,
            'weight_decay': wd,
            'layers_num': layers
        })
        if model_type == 'QGRN' and kernels is not None:
            dataset_config['num_sub_kernels'] = kernels
        
        configs.append({
            'config_idx': idx,
            'dataset_config': dataset_config,
            'split_config': base_split,
            'lr': lr,
            'epochs': base_epochs,
            'scheduler_config': base_scheduler,
            'early_stopping_config': base_early_stopping,
            'dropout_rate': dropout,
            'weight_decay': wd,
            'layers_num': layers,
            'num_sub_kernels': kernels
        })
    
    return configs


def run_single_experiment(config, model_type, dataset_key, architecture_type, 
                          dataset_type, run_evaluation=True, use_target_scaling=True, sparsity=None):
    """Run a single experiment with given configuration."""
    from experiment import ExperimentRegression
    
    config_idx = config['config_idx']
    dataset_config = config['dataset_config']
    split_config = config['split_config']
    lr = config['lr']
    epochs = config['epochs']
    scheduler_config = config['scheduler_config']
    early_stopping_config = config['early_stopping_config']
    
    print(f"\n{'='*80}")
    print(f"Experiment {config_idx + 1}: {dataset_key} ({dataset_type})")
    print(f"Model: {model_type}, Architecture: {architecture_type}, Sparsity: {sparsity}")
    print(f"Dropout: {config['dropout_rate']}, Weight Decay: {config['weight_decay']}, Layers: {config['layers_num']}")
    print(f"{'='*80}\n")
    
    # Create model
    model = create_model(model_type, dataset_config, architecture_type)
    
    # Build experiment ID
    experiment_id = (
        f"{dataset_key}_{dataset_type}_sparsity_{sparsity}_{model_type}_{architecture_type}_"
        f"config_{config_idx + 1}_lr_{lr}_drop_{config['dropout_rate']}_wd_{config['weight_decay']}_"
        f"layers_{config['layers_num']}"
    )
    
    # Prepare optimizer params
    optim_params = {
        'lr': lr,
        'weight_decay': config['weight_decay'],
        'scheduler': scheduler_config.get('scheduler'),
        'scheduler_params': scheduler_config.get('scheduler_params', {})
    }
    
    # Create and run experiment
    experiment = ExperimentRegression(
        model=model,
        optim_params=optim_params,
        base_path=current_dir,
        num_train=split_config['train'],
        num_test=split_config['test'],
        dataset_name=dataset_config['dataset_name'],
        train_batch_size=split_config['batch_size'],
        test_batch_size=split_config['batch_size'],
        experiment_id=experiment_id,
        early_stopping_config=early_stopping_config,
        use_target_scaling=use_target_scaling,
        sparsity=sparsity,
        architecture_type=architecture_type,
        dataset_type=dataset_type,
        model_type=model_type
    )
    
    results = experiment.run(
        num_epochs=epochs,
        eval_training_set=True,
        run_evaluation=run_evaluation
    )
    
    # Save experiment config
    if experiment.model_dir:
        exp_config = {
            'dataset_config': dataset_config,
            'split_config': split_config,
            'model_type': model_type,
            'architecture_type': architecture_type,
            'dataset_type': dataset_type,
            'sparsity': sparsity,
            'experiment_id': experiment_id,
            **config
        }
        with open(os.path.join(experiment.model_dir, 'experiment_config.yaml'), 'w') as f:
            yaml.dump(exp_config, f)
    
    # Extract best metrics
    test_r2_arr = results.get('test_r2', [])
    best_r2 = max(test_r2_arr) if test_r2_arr else 0.0
    best_epoch = test_r2_arr.index(best_r2) + 1 if test_r2_arr else 0
    
    results['grid_search_params'] = {
        'dropout_rate': config['dropout_rate'],
        'weight_decay': config['weight_decay'],
        'layers_num': config['layers_num'],
        'config_idx': config_idx,
        'lr': lr,
        'num_sub_kernels': config['num_sub_kernels']  # for QGRN
    }
    results['saved_model_metrics'] = {
        'epoch': best_epoch,
        'test_r2': best_r2
    }
    
    print(f"\nExperiment {config_idx + 1} complete! Best Test R²: {best_r2:.4f} at epoch {best_epoch}")
    
    return results


def run_grid_search(model_type, dataset_key, architecture_type, dataset_type, 
                    run_evaluation=True, use_target_scaling=True, sparsity=None):
    """Run grid search over all parameter combinations."""
    datasets_config = load_config(os.path.join(current_dir, 'datasets_dnanexus.yaml'))
    splits_config = load_config(os.path.join(current_dir, 'data.splits_dnanexus.yaml'))
    run_config = load_config(os.path.join(current_dir, 'run.settings_dnanexus.yaml'))
    
    configs = generate_grid_search_configs(run_config, splits_config, datasets_config, dataset_key, model_type)
    
    print(f"Running Grid Search: {len(configs)} configurations")
    print(f"Dataset: {dataset_key}, Type: {dataset_type}, Sparsity: {sparsity}")
    print(f"Model: {model_type}, Architecture: {architecture_type}\n")
    
    all_results = []
    for config in configs:
        try:
            result = run_single_experiment(
                config, model_type, dataset_key, architecture_type,
                dataset_type, run_evaluation, use_target_scaling, sparsity
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error in experiment {config['config_idx'] + 1}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*80}")
    
    for r in all_results:
        params = r['grid_search_params']
        metrics = r['saved_model_metrics']
        print(f"Config {params['config_idx']+1}: "
              f"dropout={params['dropout_rate']}, wd={params['weight_decay']}, "
              f"layers={params['layers_num']} -> R²={metrics['test_r2']:.4f}")
    
    if all_results:
        best = max(all_results, key=lambda x: x['saved_model_metrics']['test_r2'])
        print(f"\nBest: Config {best['grid_search_params']['config_idx']+1} "
              f"with R²={best['saved_model_metrics']['test_r2']:.4f}")
    
    # ========== NEW CODE: Save summary to .txt file ==========
    summary_filename = f"grid_search_summary_{dataset_key}_{dataset_type}_{model_type}_{architecture_type}_sparsity_{sparsity}.txt"
    summary_path = os.path.join(current_dir, f"Experiments_{dataset_type}_{architecture_type}_gridsearch", summary_filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRID SEARCH SUMMARY RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Write experiment info
        f.write(f"Dataset Key: {dataset_key}\n")
        f.write(f"Dataset Type: {dataset_type}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Architecture: {architecture_type}\n")
        f.write(f"Sparsity: {sparsity}\n")
        f.write(f"Total Configurations: {len(configs)}\n")
        f.write(f"Successful Runs: {len(all_results)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("INDIVIDUAL CONFIG RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        # Write header - add Kernels column for QGRN
        f.write(f"{'Config':<8} {'Dropout':<10} {'Weight Decay':<14} {'Layers':<8} {'LR':<12} {'Kernels':<10} {'Test R²':<12} {'Best Epoch':<12}\n")
        f.write("-"*96 + "\n")
        
        for r in all_results:
            params = r['grid_search_params']
            metrics = r['saved_model_metrics']
            
            f.write(f"{params['config_idx']+1:<8} "
                    f"{params['dropout_rate']:<10} "
                    f"{params['weight_decay']:<14} "
                    f"{params['layers_num']:<8} "
                    f"{params['lr']:<12.6f} "
                    f"{str(params.get('num_sub_kernels', 'N/A')):<10} "
                    f"{metrics['test_r2']:<12.6f} "
                    f"{metrics['epoch']:<12}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST CONFIGURATION\n")
        f.write("="*80 + "\n\n")
        
        if all_results:
            best = max(all_results, key=lambda x: x['saved_model_metrics']['test_r2'])
            best_params = best['grid_search_params']
            best_metrics = best['saved_model_metrics']
            
            f.write(f"Config Index: {best_params['config_idx']+1}\n")
            f.write(f"Dropout Rate: {best_params['dropout_rate']}\n")
            f.write(f"Weight Decay: {best_params['weight_decay']}\n")
            f.write(f"Number of Layers: {best_params['layers_num']}\n")
            f.write(f"Learning Rate: {best_params['lr']}\n")
            if best_params.get('num_sub_kernels') is not None:
                f.write(f"Num Sub Kernels: {best_params['num_sub_kernels']}\n")
            f.write(f"Best Test R²: {best_metrics['test_r2']:.6f}\n")
            f.write(f"Best Epoch: {best_metrics['epoch']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\nSummary saved to: {summary_path}")
    # ========== END NEW CODE ==========
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN regression experiments")
    parser.add_argument("--model", type=str, choices=['GraphConv', 'GATv2', 'QGRN'], required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset key (e.g., gf_custom_fc25)")
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--dataset-type", type=str, choices=['FC25', 'FC100', 'MIND'], required=True)
    parser.add_argument("--sparsity", type=int, required=True)
    parser.add_argument("--config", type=int, nargs='+', help="Specific config indices to run")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-target-scaling", action="store_true")
    
    args = parser.parse_args()
    
    if args.config:
        # Load configs and run specific ones
        datasets_config = load_config(os.path.join(current_dir, 'datasets_dnanexus.yaml'))
        splits_config = load_config(os.path.join(current_dir, 'data.splits_dnanexus.yaml'))
        run_config = load_config(os.path.join(current_dir, 'run.settings_dnanexus.yaml'))
        
        configs = generate_grid_search_configs(
            run_config, splits_config, datasets_config, args.dataset, args.model
        )
        
        for idx in args.config:
            if 0 <= idx < len(configs):
                run_single_experiment(
                    configs[idx], args.model, args.dataset, args.architecture,
                    args.dataset_type, not args.no_eval, not args.no_target_scaling, args.sparsity
                )
    else:
        run_grid_search(
            args.model, args.dataset, args.architecture, args.dataset_type,
            not args.no_eval, not args.no_target_scaling, args.sparsity
        )