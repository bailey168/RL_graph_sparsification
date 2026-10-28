# tailored regression experiment class
import shutil
import os
import sys
import torch
import pickle
import torch.nn.functional as F
import torch
import torch_geometric
import time
import statistics
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.metrics import r2_score
from util.schedulers import get_cosine_schedule_with_warmup

from util.data_processing import *
from util.early_stopping import EarlyStopping
from util.reproducibility import set_deterministic_training

# define a wrapper time_it decorator function
def time_it(func):
    def wrapper_function(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print(f'Function {func.__name__} took: {stop-start}s')
        return res
    return wrapper_function


class ExperimentRegression:
    # list of static variables ...
    experiment_id = 1 # static variable for creating experiments dir ...

    # -----------------------
    # Internal helpers (keys)
    # -----------------------
    def _geo(self):
        return self.data_struct["geometric"]

    def _get_qgcn_train_data(self):
        # Try qgcn key first, then gcn key - raise error if neither exists
        geo = self._geo()
        if "qgcn_train_data" in geo:
            return geo["qgcn_train_data"]
        elif "gcn_train_data" in geo:
            return geo["gcn_train_data"]
        else:
            raise KeyError("Neither 'qgcn_train_data' nor 'gcn_train_data' found in data_struct['geometric']")

    def _get_qgcn_test_data(self):
        geo = self._geo()
        if "qgcn_test_data" in geo:
            return geo["qgcn_test_data"]
        elif "gcn_test_data" in geo:
            return geo["gcn_test_data"]
        else:
            raise KeyError("Neither 'qgcn_test_data' nor 'gcn_test_data' found in data_struct['geometric']")

    def _get_sgcn_train_data(self):
        return self._geo()["sgcn_train_data"]  # should exist

    def _get_sgcn_test_data(self):
        return self._geo()["sgcn_test_data"]  # should exist

    def __init__(self, 
                sgcn_model = None, 
                qgcn_model = None, 
                optim_params = None, 
                base_path = ".", 
                num_train = None,
                num_test = None,
                dataset_name = None,
                train_batch_size = 64,
                test_batch_size = 64,
                train_shuffle_data = True,
                test_shuffle_data = False,
                profile_run = False,
                walk_clock_num_runs = 10,
                id = None,
                early_stopping_config = None,
                use_target_scaling = True,
                sparsity = None,
                architecture_type = None,
                dataset_type = None):  # Add dataset_type parameter
        
        # Store architecture type, sparsity, and dataset type as instance variables
        self.architecture_type = architecture_type
        self.sparsity = sparsity
        self.dataset_type = dataset_type
        
        # Controls whether to print runtime per model
        self.profile_run = profile_run
        self.walk_clock_num_runs = walk_clock_num_runs

        # Load the dataset from cached graph files with sparsity and dataset type
        data_struct = read_cached_graph_dataset(
            num_train=num_train, 
            num_test=num_test, 
            dataset_name=dataset_name, 
            parent_dir=base_path,
            sparsity=sparsity,  # Pass sparsity to data loading function
            dataset_type=dataset_type  # Pass dataset type to data loading function
        )

        # Save the references to the datasets
        self.data_struct = data_struct

        # Get the geometric graph data for brain connectivity
        geometric_qgcn_train_data = self._get_qgcn_train_data()
        geometric_qgcn_test_data  = self._get_qgcn_test_data()
        geometric_sgcn_train_data = self._get_sgcn_train_data()
        geometric_sgcn_test_data  = self._get_sgcn_test_data()

        print('Dataset Info:')
        print(f'Geometric qgcn dataset train: {len(geometric_qgcn_train_data)}')
        print(f'Geometric qgcn dataset test: {len(geometric_qgcn_test_data)}')
        print(f'Geometric sgcn dataset train: {len(geometric_sgcn_train_data)}')
        print(f'Geometric sgcn dataset test: {len(geometric_sgcn_test_data)}')
        
        # Print dataset statistics
        if geometric_sgcn_train_data:
            sample_graph = geometric_sgcn_train_data[0]
            print(f'Sample graph - Nodes: {sample_graph.num_nodes}, Edges: {sample_graph.num_edges}')
            print(f'Node features shape: {sample_graph.x.shape}')
            print(f'Edge features shape: {sample_graph.edge_attr.shape if hasattr(sample_graph, "edge_attr") else "None"}')
            print(f'Target (score): {sample_graph.y.item()}')

        self.generator = set_deterministic_training(42)

        # Create graph data loaders for brain connectivity
        shuffle_qgcn_geo_train_data = (len(geometric_qgcn_train_data) != 0) and train_shuffle_data
        geometric_qgcn_train_loader = GraphDataLoader(
            geometric_qgcn_train_data, 
            batch_size=train_batch_size, 
            shuffle=shuffle_qgcn_geo_train_data,
            num_workers=0,
            generator=self.generator
        )
        shuffle_qgcn_geo_test_data = (len(geometric_qgcn_test_data) != 0) and test_shuffle_data
        geometric_qgcn_test_loader = GraphDataLoader(
            geometric_qgcn_test_data,
            batch_size=test_batch_size,
            shuffle=shuffle_qgcn_geo_test_data,
            num_workers=0,
            generator=self.generator
        )
        shuffle_sgcn_geo_train_data = (len(geometric_sgcn_train_data) != 0) and train_shuffle_data
        geometric_sgcn_train_loader = GraphDataLoader(
            geometric_sgcn_train_data, 
            batch_size=train_batch_size, 
            shuffle=shuffle_sgcn_geo_train_data,
            num_workers=0,
            generator=self.generator
        )
        shuffle_sgcn_geo_test_data = (len(geometric_sgcn_test_data) != 0) and test_shuffle_data
        geometric_sgcn_test_loader = GraphDataLoader(
            geometric_sgcn_test_data,
            batch_size=test_batch_size,
            shuffle=shuffle_sgcn_geo_test_data,
            num_workers=0,
            generator=self.generator
        )

        # Save the references here
        self.sp_qgcn_train_dataloader = geometric_qgcn_train_loader
        self.sp_qgcn_test_dataloader = geometric_qgcn_test_loader
        self.sp_sgcn_train_dataloader = geometric_sgcn_train_loader
        self.sp_sgcn_test_dataloader = geometric_sgcn_test_loader

        # Define what device we are using
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.deterministic = True
        print(f'Using device: {self.device}')

        # Define the experiments folder and add the directory for this run
        self.cache_run = True
        self.specific_run_dir = None
        self.qgcn_specific_run_dir = None
        self.sgcn_specific_run_dir = None
        
        if base_path == None:
            self.cache_run = False
        else:
            local_experiment_id = ExperimentRegression.experiment_id
            if id == None:
                ExperimentRegression.experiment_id += 1
            else:
                local_experiment_id = id
            self.local_experiment_id = local_experiment_id
            # Create the folder structure for this run
            self.__create_experiment_folder_structure(base_path, local_experiment_id)

        # Save the models
        self.qgcn_model = qgcn_model
        self.sgcn_model = sgcn_model

        # Assert that at least one graph model exists
        assert any([self.qgcn_model, self.sgcn_model]), "At least one graph model (QGCN or SGCN) must be provided for brain connectivity analysis"
        self.qgcn_model_exists = qgcn_model != None
        self.sgcn_model_exists = sgcn_model != None

        # Put the models on the device
        if self.qgcn_model_exists:
            self.qgcn_model.to(self.device)
        if self.sgcn_model_exists:
            self.sgcn_model.to(self.device)

        # Define the optimizers for the different models
        self.optim_params = optim_params
        learning_rate = 0.001  # Default learning rate for brain connectivity regression
        weight_decay = 0.001   # Default weight decay
        
        if optim_params != None:
            if "lr" in optim_params.keys():
                learning_rate = optim_params["lr"]
            if "weight_decay" in optim_params.keys():
                weight_decay = optim_params["weight_decay"]
        
        self.qgcn_model_optimizer = None
        self.sgcn_model_optimizer = None
        
        if self.qgcn_model_exists:
            self.qgcn_model_optimizer = torch.optim.AdamW(self.qgcn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if self.sgcn_model_exists:
            self.sgcn_model_optimizer = torch.optim.AdamW(self.sgcn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Add learning rate schedulers
        self.qgcn_model_scheduler = None
        self.sgcn_model_scheduler = None
        
        # Configure schedulers based on optim_params
        scheduler_type = optim_params["scheduler"]
        scheduler_params = optim_params["scheduler_params"]
        
        if self.qgcn_model_exists:
            self.qgcn_model_scheduler = self._create_scheduler(self.qgcn_model_optimizer, scheduler_type, scheduler_params)
        if self.sgcn_model_exists:
            self.sgcn_model_scheduler = self._create_scheduler(self.sgcn_model_optimizer, scheduler_type, scheduler_params)

        # Add early stopping configuration with R² support
        self.early_stopping_config = early_stopping_config or {}
        self.use_early_stopping = self.early_stopping_config['enabled']
        
        if self.use_early_stopping:
            # Initialize early stopping for each model
            es_patience = self.early_stopping_config['patience']
            es_min_delta = self.early_stopping_config['min_delta']
            es_restore_weights = self.early_stopping_config['restore_best_weights']
            es_verbose = self.early_stopping_config['verbose']
            es_monitor = self.early_stopping_config['monitor']

            self.qgcn_early_stopping = EarlyStopping(
                patience=es_patience,
                min_delta=es_min_delta,
                restore_best_weights=es_restore_weights,
                verbose=es_verbose,
                monitor=es_monitor
            ) if self.qgcn_model_exists else None
            
            self.sgcn_early_stopping = EarlyStopping(
                patience=es_patience,
                min_delta=es_min_delta,
                restore_best_weights=es_restore_weights,
                verbose=es_verbose,
                monitor=es_monitor
            ) if self.sgcn_model_exists else None
        else:
            self.qgcn_early_stopping = None
            self.sgcn_early_stopping = None

        # Add target scaling configuration
        self.use_target_scaling = use_target_scaling
        self.target_scaler_mean = None
        self.target_scaler_std = None
        
        if self.use_target_scaling:
            self._fit_target_scaler()

        # Print model statistics if profiling
        if self.profile_run: 
            self.__print_models_stats()

    def _create_scheduler(self, optimizer, scheduler_type, scheduler_params):
        """Create learning rate scheduler based on configuration."""
        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params["step_size"],
                gamma=scheduler_params["gamma"]
            )
        elif scheduler_type == "multistep":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_params["milestones"],
                gamma=scheduler_params["gamma"]
            )
        elif scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_params["gamma"]
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params["T_max"],
                eta_min=scheduler_params["eta_min"]
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_params["factor"],
                patience=scheduler_params["patience"],
                min_lr=scheduler_params["min_lr"]
            )
        elif scheduler_type == "warmup_cosine":
            # Linear warmup followed by cosine decay
            num_warmup_steps = scheduler_params["num_warmup_steps"]
            num_training_steps = scheduler_params["num_training_steps"]
            num_cycles = scheduler_params["num_cycles"]
            last_epoch = scheduler_params["last_epoch"]

            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                last_epoch=last_epoch
            )
        else:
            return None

    def __print_models_stats(self):
        """Print model statistics for brain connectivity models."""
        try:
            from flops_counter.ptflops import get_model_complexity_info
        except ImportError:
            print("ptflops not available for model profiling")
            return

        if self.sgcn_model_exists:
            # Pick the largest data (by node degree) to profile all the models
            crit_lst = [data.x.numel() + data.edge_index.numel() for data in self.data_struct["geometric"]["sgcn_train_data"]]
            _, max_crit_index = max(crit_lst), crit_lst.index(max(crit_lst))
            data_sample = self.data_struct["geometric"]["sgcn_train_data"][max_crit_index].clone().detach().to(self.device)
            model = self.sgcn_model
            model.eval()
            
            try:
                macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
                flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
                
                # Profile Inference Wall Time
                wall_times = []
                for _ in range(self.walk_clock_num_runs):
                    start_time = time.time()
                    _ = model(self.data_struct["geometric"]["sgcn_train_data"][max_crit_index].clone().detach().to(self.device))
                    end_time = time.time()
                    wall_times.append(end_time - start_time)
                wall_time_mean = statistics.mean(wall_times)
                wall_time_std = statistics.stdev(wall_times)
                
                print("\n-----------------")
                print("SGCN Model Stats (Brain Connectivity):")
                print(f"Profiling data sample: {self.data_struct['geometric']['sgcn_train_data'][max_crit_index]}")
                print("-------------------------------------------------------------------------------------------")
                print(f'Number of parameters: {params} k')
                print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
                print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
                print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
                print("-------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"Error profiling SGCN model: {e}")

        if self.qgcn_model_exists:
            # Similar profiling for QGCN
            crit_lst = [data.x.numel() + data.edge_index.numel() for data in self.data_struct["geometric"]["qgcn_train_data"]]
            qgcn_train = self._get_qgcn_train_data()
            if not qgcn_train:
                print("No QGCN training data found for profiling")
                return
            _, max_crit_index = max(crit_lst), crit_lst.index(max(crit_lst))
            data_sample = qgcn_train[max_crit_index].clone().detach().to(self.device)
            model = self.qgcn_model
            model.eval()
            
            try:
                macs, params = get_model_complexity_info(model, data_sample, as_strings=False, print_per_layer_stat=False, verbose=False)
                flops, macs, params = round(2*macs / 1e3, 3), round(macs / 1e3, 3), round(params / 1e3, 3)
                
                # Profile Inference Wall Time
                wall_times = []
                for _ in range(self.walk_clock_num_runs):
                    start_time = time.time()
                    _ = model(qgcn_train[max_crit_index].clone().detach().to(self.device))
                    end_time = time.time()
                    wall_times.append(end_time - start_time)
                wall_time_mean = statistics.mean(wall_times)
                wall_time_std = statistics.stdev(wall_times)
                
                print("\n-----------------")
                print("QGCN Model Stats (Brain Connectivity):")
                print(f"Profiling data sample: {self.data_struct['geometric']['qgcn_train_data'][max_crit_index]}")
                print("-------------------------------------------------------------------------------------------")
                print(f'Number of parameters: {params} k')
                print(f'Theoretical Computational Complexity (FLOPs): {flops} kFLOPs')
                print(f'Theoretical Computational Complexity (MACs):  {macs} kMACs')
                print(f'Wall Clock  Computational Complexity (s):     {wall_time_mean} +/- {wall_time_std} s ')
                print("-------------------------------------------------------------------------------------------")
            except Exception as e:
                print(f"Error profiling QGCN model: {e}")

    def __create_experiment_folder_structure(self, base_path, experiment_id):
        """Create folder structure for experiment results."""
        if not os.path.exists(base_path):
            print("Ensure that your base path exists -> {}".format(base_path))
            sys.exit(1)
        
        # Include architecture type and dataset type in experiment folder name
        experiments_dir = os.path.join(base_path, f"Experiments_{self.dataset_type}_{self.architecture_type}_gridsearch")
        if not os.path.exists(experiments_dir):
            os.mkdir(experiments_dir)
        underscored_experiment_id = "_".join(str(experiment_id).strip().split(" "))
        specific_run_dir = os.path.join(experiments_dir, "run_" + underscored_experiment_id)
        if not os.path.exists(specific_run_dir):
            os.mkdir(specific_run_dir)
        self.specific_run_dir = specific_run_dir

        # Create the respective folders for this run
        qgcn_specific_run_dir = os.path.join(specific_run_dir, "qgcn")
        if not os.path.exists(qgcn_specific_run_dir):
            os.mkdir(qgcn_specific_run_dir)
        self.qgcn_specific_run_dir = qgcn_specific_run_dir
        
        sgcn_specific_run_dir = os.path.join(specific_run_dir, "sgcn")
        if not os.path.exists(sgcn_specific_run_dir):
            os.mkdir(sgcn_specific_run_dir)
        self.sgcn_specific_run_dir = sgcn_specific_run_dir

        # Copy architecture files based on architecture type
        try:
            gnn_source_filepath = os.path.join(base_path, "gnn", f"architectures_{self.architecture_type}.py")
            if os.path.exists(gnn_source_filepath):
                import shutil
                qgcn_destination_filepath = os.path.join(self.qgcn_specific_run_dir, f"architectures_{self.architecture_type}.py")
                if not os.path.exists(qgcn_destination_filepath):
                    shutil.copy2(gnn_source_filepath, qgcn_destination_filepath)
                sgcn_destination_filepath = os.path.join(self.sgcn_specific_run_dir, f"architectures_{self.architecture_type}.py")
                if not os.path.exists(sgcn_destination_filepath):
                    shutil.copy2(gnn_source_filepath, sgcn_destination_filepath)
        except Exception as e:
            print(f"Warning: Could not copy architecture files: {e}")

    def __cache_models(self):
        """Cache models with epoch information."""
        if self.qgcn_model_exists:
            # When early stopping is enabled, we should NOT save again if early stopping was used
            # because the best model was already saved during training
            # BUT if early stopping was enabled and DIDN'T trigger, we should save the best model
            should_save_qgcn = True
            
            if self.use_early_stopping and hasattr(self, 'qgcn_early_stopping') and self.qgcn_early_stopping:
                if self.qgcn_early_stopping.early_stop:
                    # Early stopping triggered - best model already saved, don't save again
                    should_save_qgcn = False
                    print(f"Skipping final QGCN model save - early stopping triggered, best model already saved")
                else:
                    # Early stopping was enabled but didn't trigger - save the best model we found
                    should_save_qgcn = True
                    print(f"Early stopping enabled but didn't trigger - saving best QGCN model")
            
            if should_save_qgcn:
                # For early stopping that didn't trigger, restore best weights before saving
                if self.use_early_stopping and hasattr(self, 'qgcn_early_stopping') and self.qgcn_early_stopping:
                    if hasattr(self.qgcn_early_stopping, 'best_weights') and self.qgcn_early_stopping.best_weights:
                        self.qgcn_model.load_state_dict(self.qgcn_early_stopping.best_weights)
                        print(f"Restored QGCN best weights before final save (best epoch: {getattr(self.qgcn_early_stopping, 'best_epoch', 'unknown')})")
                
                # Save complete model first (most reliable for loading)
                torch.save(self.qgcn_model, os.path.join(self.qgcn_specific_run_dir, "model_complete.pth"))
                
                # Save model with metadata including epoch info
                best_epoch = getattr(self.qgcn_early_stopping, 'best_epoch', self.final_qgcn_epoch) if self.use_early_stopping and hasattr(self, 'qgcn_early_stopping') else self.final_qgcn_epoch
                
                qgcn_save_dict = {
                    'model_state_dict': self.qgcn_model.state_dict(),
                    'optimizer_state_dict': self.qgcn_model_optimizer.state_dict(),
                    'best_epoch': best_epoch,
                    'final_epoch': getattr(self, 'final_qgcn_epoch', 0),
                    'is_best_model': True,
                    'model_config': {
                        'model_type': type(self.qgcn_model).__name__,
                        'architecture': str(self.qgcn_model)
                    },
                    # Add target scaling information
                    'target_scaling': {
                        'use_target_scaling': self.use_target_scaling,
                        'target_scaler_mean': self.target_scaler_mean,
                        'target_scaler_std': self.target_scaler_std
                    }
                }
                torch.save(qgcn_save_dict, os.path.join(self.qgcn_specific_run_dir, "model.pth"))
            
            # Save epoch info in a separate text file for easy reading
            with open(os.path.join(self.qgcn_specific_run_dir, "training_info.txt"), 'w') as f:
                f.write(f"Final training epoch: {getattr(self, 'final_qgcn_epoch', 0)}\n")
                f.write(f"Model type: {type(self.qgcn_model).__name__}\n")
                f.write(f"Early stopping: {'Yes' if self.use_early_stopping else 'No'}\n")
                if self.use_early_stopping and hasattr(self, 'qgcn_early_stopping'):
                    f.write(f"Early stopped: {'Yes' if self.qgcn_early_stopping.early_stop else 'No'}\n")
                    if hasattr(self.qgcn_early_stopping, 'best_epoch'):
                        f.write(f"Best model epoch: {getattr(self.qgcn_early_stopping, 'best_epoch', 'unknown')}\n")
                        f.write(f"Monitor metric: {self.early_stopping_config['monitor']}\n")

        if self.sgcn_model_exists:
            # Same logic for SGCN
            should_save_sgcn = True
            
            if self.use_early_stopping and hasattr(self, 'sgcn_early_stopping') and self.sgcn_early_stopping:
                if self.sgcn_early_stopping.early_stop:
                    # Early stopping triggered - best model already saved, don't save again
                    should_save_sgcn = False
                    print(f"Skipping final SGCN model save - early stopping triggered, best model already saved")
                else:
                    # Early stopping was enabled but didn't trigger - save the best model we found
                    should_save_sgcn = True
                    print(f"Early stopping enabled but didn't trigger - saving best SGCN model")
            
            if should_save_sgcn:
                # For early stopping that didn't trigger, restore best weights before saving
                if self.use_early_stopping and hasattr(self, 'sgcn_early_stopping') and self.sgcn_early_stopping:
                    if hasattr(self.sgcn_early_stopping, 'best_weights') and self.sgcn_early_stopping.best_weights:
                        self.sgcn_model.load_state_dict(self.sgcn_early_stopping.best_weights)
                        print(f"Restored SGCN best weights before final save (best epoch: {getattr(self.sgcn_early_stopping, 'best_epoch', 'unknown')})")
                
                # Save complete model first (most reliable for loading)
                torch.save(self.sgcn_model, os.path.join(self.sgcn_specific_run_dir, "model_complete.pth"))
                
                # Save model with metadata including epoch info
                best_epoch = getattr(self.sgcn_early_stopping, 'best_epoch', self.final_sgcn_epoch) if self.use_early_stopping and hasattr(self, 'sgcn_early_stopping') else self.final_sgcn_epoch
                
                sgcn_save_dict = {
                    'model_state_dict': self.sgcn_model.state_dict(),
                    'optimizer_state_dict': self.sgcn_model_optimizer.state_dict(),
                    'best_epoch': best_epoch,
                    'final_epoch': getattr(self, 'final_sgcn_epoch', 0),
                    'is_best_model': True,
                    'model_config': {
                        'model_type': type(self.sgcn_model).__name__,
                        'architecture': str(self.sgcn_model)
                    },
                    # Add target scaling information
                    'target_scaling': {
                        'use_target_scaling': self.use_target_scaling,
                        'target_scaler_mean': self.target_scaler_mean,
                        'target_scaler_std': self.target_scaler_std
                    }
                }
                torch.save(sgcn_save_dict, os.path.join(self.sgcn_specific_run_dir, "model.pth"))
            
            # Save epoch info in a separate text file for easy reading
            with open(os.path.join(self.sgcn_specific_run_dir, "training_info.txt"), 'w') as f:
                f.write(f"Final training epoch: {getattr(self, 'final_sgcn_epoch', 0)}\n")
                f.write(f"Model type: {type(self.sgcn_model).__name__}\n")
                f.write(f"Early stopping: {'Yes' if self.use_early_stopping else 'No'}\n")
                if self.use_early_stopping and hasattr(self, 'sgcn_early_stopping'):
                    f.write(f"Early stopped: {'Yes' if self.sgcn_early_stopping.early_stop else 'No'}\n")
                    if hasattr(self.sgcn_early_stopping, 'best_epoch'):
                        f.write(f"Best model epoch: {getattr(self.sgcn_early_stopping, 'best_epoch', 'unknown')}\n")
                        f.write(f"Monitor metric: {self.early_stopping_config['monitor']}\n")

    def __cache_results(self, train_qgcn_loss_array, train_sgcn_loss_array, 
                        train_qgcn_mse_array, train_sgcn_mse_array,
                        test_qgcn_mse_array, test_sgcn_mse_array,
                        train_qgcn_r2_array=None, train_sgcn_r2_array=None,
                        test_qgcn_r2_array=None, test_sgcn_r2_array=None,
                        learning_rates_qgcn=None, learning_rates_sgcn=None):
        """Save training results to disk."""
        if self.qgcn_specific_run_dir != None and self.qgcn_model_exists:
            train_qgcn_loss_filepath = os.path.join(self.qgcn_specific_run_dir, "train_loss.pk")
            train_qgcn_mse_filepath = os.path.join(self.qgcn_specific_run_dir, "train_mse.pk")
            test_qgcn_mse_filepath = os.path.join(self.qgcn_specific_run_dir, "test_mse.pk")
            with open(train_qgcn_loss_filepath, 'wb') as f:
                pickle.dump(train_qgcn_loss_array, f)
            with open(train_qgcn_mse_filepath, 'wb') as f:
                pickle.dump(train_qgcn_mse_array, f)
            with open(test_qgcn_mse_filepath, 'wb') as f:
                pickle.dump(test_qgcn_mse_array, f)
            
            # Save R² arrays if provided
            if train_qgcn_r2_array is not None:
                train_qgcn_r2_filepath = os.path.join(self.qgcn_specific_run_dir, "train_r2.pk")
                with open(train_qgcn_r2_filepath, 'wb') as f:
                    pickle.dump(train_qgcn_r2_array, f)
            if test_qgcn_r2_array is not None:
                test_qgcn_r2_filepath = os.path.join(self.qgcn_specific_run_dir, "test_r2.pk")
                with open(test_qgcn_r2_filepath, 'wb') as f:
                    pickle.dump(test_qgcn_r2_array, f)
            
            if learning_rates_qgcn is not None:
                lr_qgcn_filepath = os.path.join(self.qgcn_specific_run_dir, "learning_rates.pk")
                with open(lr_qgcn_filepath, 'wb') as f:
                    pickle.dump(learning_rates_qgcn, f)
        
        if self.sgcn_specific_run_dir != None and self.sgcn_model_exists:
            train_sgcn_loss_filepath = os.path.join(self.sgcn_specific_run_dir, "train_loss.pk")
            train_sgcn_mse_filepath = os.path.join(self.sgcn_specific_run_dir, "train_mse.pk")
            test_sgcn_mse_filepath = os.path.join(self.sgcn_specific_run_dir, "test_mse.pk")
            with open(train_sgcn_loss_filepath, 'wb') as f:
                pickle.dump(train_sgcn_loss_array, f)
            with open(train_sgcn_mse_filepath, 'wb') as f:
                pickle.dump(train_sgcn_mse_array, f)
            with open(test_sgcn_mse_filepath, 'wb') as f:
                pickle.dump(test_sgcn_mse_array, f)
            
            # Save R² arrays if provided
            if train_sgcn_r2_array is not None:
                train_sgcn_r2_filepath = os.path.join(self.sgcn_specific_run_dir, "train_r2.pk")
                with open(train_sgcn_r2_filepath, 'wb') as f:
                    pickle.dump(train_sgcn_r2_array, f)
            if test_sgcn_r2_array is not None:
                test_sgcn_r2_filepath = os.path.join(self.sgcn_specific_run_dir, "test_r2.pk")
                with open(test_sgcn_r2_filepath, 'wb') as f:
                    pickle.dump(test_sgcn_r2_array, f)
            
            if learning_rates_sgcn is not None:
                lr_sgcn_filepath = os.path.join(self.sgcn_specific_run_dir, "learning_rates.pk")
                with open(lr_sgcn_filepath, 'wb') as f:
                    pickle.dump(learning_rates_sgcn, f)

    def __move_graph_data_to_device(self, data):
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

    def _fit_target_scaler(self):
        """Fit target scaler using training data."""
        print("Fitting target scaler on training data...")
        # Collect all training targets
        train_targets = []
        
        # Use SGCN training data (or QGCN if SGCN doesn't exist)
        if self.sgcn_model_exists:
            for data in self._get_sgcn_train_data():
                train_targets.append(data.y.item() if data.y.dim() == 0 else data.y.cpu().numpy())
        elif self.qgcn_model_exists:
            for data in self._get_qgcn_train_data():
                train_targets.append(data.y.item() if data.y.dim() == 0 else data.y.cpu().numpy())
        
        train_targets = np.array(train_targets)
        
        # Calculate mean and std
        self.target_scaler_mean = np.mean(train_targets)
        self.target_scaler_std = np.std(train_targets)
        
        print(f"Target scaling - Mean: {self.target_scaler_mean:.4f}, Std: {self.target_scaler_std:.4f}")
        
        # Apply scaling to all datasets
        self._apply_target_scaling()
    
    def _apply_target_scaling(self):
        """Apply target scaling to all datasets."""
        print("Applying target scaling to datasets...")
        
        # Scale SGCN datasets
        if self.sgcn_model_exists:
            for data in self._get_sgcn_train_data():
                data.y = (data.y - self.target_scaler_mean) / self.target_scaler_std
            for data in self._get_sgcn_test_data():
                data.y = (data.y - self.target_scaler_mean) / self.target_scaler_std
        
        # Scale QGCN datasets
        if self.qgcn_model_exists:
            for data in self._get_qgcn_train_data():
                data.y = (data.y - self.target_scaler_mean) / self.target_scaler_std
            for data in self._get_qgcn_test_data():
                data.y = (data.y - self.target_scaler_mean) / self.target_scaler_std
    
    def _inverse_transform_targets(self, scaled_targets):
        """Convert scaled targets back to original scale."""
        if self.use_target_scaling:
            return scaled_targets * self.target_scaler_std + self.target_scaler_mean
        return scaled_targets
    
    def _transform_targets(self, original_targets):
        """Convert original targets to scaled targets."""
        if self.use_target_scaling:
            return (original_targets - self.target_scaler_mean) / self.target_scaler_std
        return original_targets

    def __train(self, train_qgcn=True, train_sgcn=True):
        """Train models for one epoch with selective training."""
        # For QGCN training
        qgcn_loss_all, qgcn_total_graphs = 0, 0
        if self.qgcn_model_exists and train_qgcn:
            self.qgcn_model.train()
            if self.profile_run: start_time = time.time()
            for data in self.sp_qgcn_train_dataloader:
                qgcn_data = self.__move_graph_data_to_device(data)
                qgcn_total_graphs += qgcn_data.num_graphs
                
                self.qgcn_model_optimizer.zero_grad()
                qgcn_output = self.qgcn_model(qgcn_data)
                qgcn_loss = F.mse_loss(qgcn_output, qgcn_data.y.float())
                qgcn_loss.backward()
                qgcn_loss_all += qgcn_data.num_graphs * qgcn_loss.item()
                self.qgcn_model_optimizer.step()
            if self.profile_run: 
                stop_time = time.time()
                profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
                print(f"Single epoch training... qgcn_model done{profile_stats}")

        # For SGCN training
        sgcn_loss_all, sgcn_total_graphs = 0, 0
        if self.sgcn_model_exists and train_sgcn:
            self.sgcn_model.train()
            if self.profile_run: start_time = time.time()
            for data in self.sp_sgcn_train_dataloader:
                sgcn_data = self.__move_graph_data_to_device(data)
                sgcn_total_graphs += sgcn_data.num_graphs
                
                self.sgcn_model_optimizer.zero_grad()
                sgcn_output = self.sgcn_model(sgcn_data)
                sgcn_loss = F.mse_loss(sgcn_output, sgcn_data.y.float())
                sgcn_loss.backward()
                sgcn_loss_all += sgcn_data.num_graphs * sgcn_loss.item()
                self.sgcn_model_optimizer.step()
            if self.profile_run: 
                stop_time = time.time()
                profile_stats = f"; Epoch took a total of {stop_time - start_time}s"
                print(f"Single epoch training... sgcn_model done{profile_stats}")

        # Remove the automatic model caching here
        # if self.cache_run:
        #     self.__cache_models()

        # Normalize loss by the total length of training set
        if self.qgcn_model_exists and qgcn_total_graphs > 0:
            qgcn_loss_all /= qgcn_total_graphs
        if self.sgcn_model_exists and sgcn_total_graphs > 0:
            sgcn_loss_all /= sgcn_total_graphs

        return qgcn_loss_all, sgcn_loss_all

    def __save_best_model(self, model_type, current_epoch=None):
        """Save model when it achieves best performance."""
        if model_type == 'sgcn' and self.sgcn_model_exists:
            # Save complete model first
            torch.save(self.sgcn_model, os.path.join(self.sgcn_specific_run_dir, "model_complete.pth"))
            
            # Save state dict with metadata INCLUDING target scaling info
            sgcn_save_dict = {
                'model_state_dict': self.sgcn_model.state_dict(),
                'optimizer_state_dict': self.sgcn_model_optimizer.state_dict(),
                'best_epoch': current_epoch or getattr(self, 'current_epoch', 0),
                'final_epoch': current_epoch or getattr(self, 'current_epoch', 0),
                'is_best_model': True,
                'model_config': {
                    'model_type': type(self.sgcn_model).__name__,
                    'architecture': str(self.sgcn_model)
                },
                'target_scaling': {
                    'use_target_scaling': self.use_target_scaling,
                    'target_scaler_mean': self.target_scaler_mean,
                    'target_scaler_std': self.target_scaler_std
                }
            }
            torch.save(sgcn_save_dict, os.path.join(self.sgcn_specific_run_dir, "model.pth"))
            if self.use_early_stopping:
                monitor_metric = self.early_stopping_config['monitor']
                print(f"✓ Saved best SGCN model from epoch {current_epoch} (best {monitor_metric})")
            else:
                print(f"✓ Saved SGCN model from epoch {current_epoch}")

        elif model_type == 'qgcn' and self.qgcn_model_exists:
            # Similar changes for QGCN
            torch.save(self.qgcn_model, os.path.join(self.qgcn_specific_run_dir, "model_complete.pth"))
            
            qgcn_save_dict = {
                'model_state_dict': self.qgcn_model.state_dict(),
                'optimizer_state_dict': self.qgcn_model_optimizer.state_dict(),
                'best_epoch': current_epoch or getattr(self, 'current_epoch', 0),
                'final_epoch': current_epoch or getattr(self, 'current_epoch', 0),
                'is_best_model': True,
                'model_config': {
                    'model_type': type(self.qgcn_model).__name__,
                    'architecture': str(self.qgcn_model)
                },
                'target_scaling': {
                    'use_target_scaling': self.use_target_scaling,
                    'target_scaler_mean': self.target_scaler_mean,
                    'target_scaler_std': self.target_scaler_std
                }
            }
            torch.save(qgcn_save_dict, os.path.join(self.qgcn_specific_run_dir, "model.pth"))
            if self.use_early_stopping:
                monitor_metric = self.early_stopping_config['monitor']
                print(f"✓ Saved best QGCN model from epoch {current_epoch} (best {monitor_metric})")
            else:
                print(f"✓ Saved QGCN model from epoch {current_epoch}")

    def __evaluate(self, eval_train_data=False):
        """Evaluate models on train or test data and return both MSE and R² in original scale."""
        # For QGCN evaluation
        qgcn_mse, qgcn_r2 = 0, 0
        if self.qgcn_model_exists:
            sp_qgcn_dataset_loader = self.sp_qgcn_train_dataloader if eval_train_data else self.sp_qgcn_test_dataloader
            self.qgcn_model.eval()
            
            total_mse, total_samples = 0, 0
            all_predictions, all_targets = [], []
            
            with torch.no_grad():
                for data in sp_qgcn_dataset_loader:
                    data = self.__move_graph_data_to_device(data)
                    predictions = self.qgcn_model(data)
                    
                    # Convert predictions and targets back to original scale
                    predictions_orig = self._inverse_transform_targets(predictions.cpu().numpy())
                    targets_orig = self._inverse_transform_targets(data.y.cpu().numpy())
                    
                    # Calculate MSE in original scale
                    mse = np.mean((predictions_orig - targets_orig) ** 2)
                    total_mse += mse * len(predictions_orig)
                    total_samples += len(predictions_orig)
                    
                    all_predictions.extend(predictions_orig)
                    all_targets.extend(targets_orig)
            
            qgcn_mse = total_mse / total_samples
            
            # Calculate R² score in original scale
            if len(all_predictions) > 1:
                qgcn_r2 = r2_score(all_targets, all_predictions)
            else:
                qgcn_r2 = 0

        # For SGCN evaluation
        sgcn_mse, sgcn_r2 = 0, 0
        if self.sgcn_model_exists:
            sp_sgcn_dataset_loader = self.sp_sgcn_train_dataloader if eval_train_data else self.sp_sgcn_test_dataloader
            self.sgcn_model.eval()
            
            total_mse, total_samples = 0, 0
            all_predictions, all_targets = [], []
            
            with torch.no_grad():
                for data in sp_sgcn_dataset_loader:
                    data = self.__move_graph_data_to_device(data)
                    predictions = self.sgcn_model(data)
                    
                    # Convert predictions and targets back to original scale
                    predictions_orig = self._inverse_transform_targets(predictions.cpu().numpy())
                    targets_orig = self._inverse_transform_targets(data.y.cpu().numpy())
                    
                    # Calculate MSE in original scale
                    mse = np.mean((predictions_orig - targets_orig) ** 2)
                    total_mse += mse * len(predictions_orig)
                    total_samples += len(predictions_orig)
                    
                    all_predictions.extend(predictions_orig)
                    all_targets.extend(targets_orig)
            
            sgcn_mse = total_mse / total_samples
            
            # Calculate R² score in original scale
            if len(all_predictions) > 1:
                sgcn_r2 = r2_score(all_targets, all_predictions)
            else:
                sgcn_r2 = 0

        return qgcn_mse, sgcn_mse, qgcn_r2, sgcn_r2

    @time_it
    def run(self, num_epochs=None, eval_training_set=True, run_evaluation=True):
        """Run the complete training and evaluation loop with early stopping."""
        if num_epochs == None or num_epochs <= 0:
            print("num_epochs ({}) in [ExperimentRegression.run] is invalid".format(num_epochs))
            sys.exit(1)

        # Define variables to hold the stats
        test_qgcn_mse_array, test_sgcn_mse_array = [], []
        train_qgcn_mse_array, train_sgcn_mse_array = [], []
        test_qgcn_r2_array, test_sgcn_r2_array = [], []
        train_qgcn_r2_array, train_sgcn_r2_array = [], []
        train_qgcn_loss_array, train_sgcn_loss_array = [], []
        learning_rates_qgcn, learning_rates_sgcn = [], []
        
        print("Starting Brain Connectivity Regression Training...")
        print(f"Training for up to {num_epochs} epochs")
        print(f"Models: QGCN={self.qgcn_model_exists}, SGCN={self.sgcn_model_exists}")
        print(f"Early Stopping: {'Enabled' if self.use_early_stopping else 'Disabled'}")
        
        if self.use_early_stopping:
            monitor_metric = self.early_stopping_config['monitor']
            print(f"Early Stopping Config: patience={self.early_stopping_config['patience']}, "
                  f"min_delta={self.early_stopping_config['min_delta']}, "
                  f"monitor={monitor_metric}")
        
        # Flags to track if models should continue training
        qgcn_continue_training = self.qgcn_model_exists
        sgcn_continue_training = self.sgcn_model_exists
        
        # Track final epochs for each model
        final_qgcn_epoch = 0
        final_sgcn_epoch = 0
        
        for epoch in range(1, num_epochs + 1):
            # Time epoch operations
            start_time = time.time()
            print("training... epoch {}".format(epoch))
            
            # Set current epoch for models (for early stopping tracking)
            if self.qgcn_model_exists:
                self.qgcn_model.current_epoch = epoch
            if self.sgcn_model_exists:
                self.sgcn_model.current_epoch = epoch
            
            # UPDATE FINAL EPOCHS FOR MODELS STILL TRAINING
            if qgcn_continue_training:
                final_qgcn_epoch = epoch
            if sgcn_continue_training:
                final_sgcn_epoch = epoch
            
            # Train models (only if they haven't stopped early)
            qgcn_loss, sgcn_loss = self.__train(
                train_qgcn=qgcn_continue_training, 
                train_sgcn=sgcn_continue_training
            )
            train_qgcn_loss_array.append(qgcn_loss)
            train_sgcn_loss_array.append(sgcn_loss)
            
            # Evaluate and get both MSE and R²
            train_qgcn_mse, train_sgcn_mse, train_qgcn_r2, train_sgcn_r2 = 0, 0, 0, 0
            if eval_training_set:
                train_qgcn_mse, train_sgcn_mse, train_qgcn_r2, train_sgcn_r2 = self.__evaluate(eval_train_data=True)
            train_qgcn_mse_array.append(train_qgcn_mse)
            train_sgcn_mse_array.append(train_sgcn_mse)
            train_qgcn_r2_array.append(train_qgcn_r2)
            train_sgcn_r2_array.append(train_sgcn_r2)
            
            test_qgcn_mse, test_sgcn_mse, test_qgcn_r2, test_sgcn_r2 = self.__evaluate(eval_train_data=False)
            test_qgcn_mse_array.append(test_qgcn_mse)
            test_sgcn_mse_array.append(test_sgcn_mse)
            test_qgcn_r2_array.append(test_qgcn_r2)
            test_sgcn_r2_array.append(test_sgcn_r2)
            
            # Step schedulers and record learning rates (only for models still training)
            current_lr_qgcn, current_lr_sgcn = 0, 0
            
            if self.qgcn_model_scheduler is not None and qgcn_continue_training:
                if isinstance(self.qgcn_model_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use appropriate metric for plateau scheduler
                    monitor_metric = self.early_stopping_config['monitor'] if self.use_early_stopping else 'loss'
                    plateau_metric = test_qgcn_r2 if monitor_metric == 'r2' else test_qgcn_mse
                    self.qgcn_model_scheduler.step(plateau_metric)
                else:
                    self.qgcn_model_scheduler.step()
                current_lr_qgcn = self.qgcn_model_optimizer.param_groups[0]['lr']
            
            if self.sgcn_model_scheduler is not None and sgcn_continue_training:
                if isinstance(self.sgcn_model_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use appropriate metric for plateau scheduler
                    monitor_metric = self.early_stopping_config['monitor'] if self.use_early_stopping else 'loss'
                    plateau_metric = test_sgcn_r2 if monitor_metric == 'r2' else test_sgcn_mse
                    self.sgcn_model_scheduler.step(plateau_metric)
                else:
                    self.sgcn_model_scheduler.step()
                current_lr_sgcn = self.sgcn_model_optimizer.param_groups[0]['lr']
            
            learning_rates_qgcn.append(current_lr_qgcn)
            learning_rates_sgcn.append(current_lr_sgcn)
            
            # Check early stopping conditions
            if self.use_early_stopping:
                monitor_metric = self.early_stopping_config['monitor']
                
                if qgcn_continue_training and self.qgcn_early_stopping is not None:
                    # Choose the metric to monitor
                    early_stop_metric = test_qgcn_r2 if monitor_metric == 'r2' else test_qgcn_mse
                    should_stop = self.qgcn_early_stopping(early_stop_metric, self.qgcn_model)
                    
                    # ONLY save if the model improved (counter was reset to 0)
                    if not should_stop and self.qgcn_early_stopping.counter == 0:
                        if self.cache_run:
                            self.__save_best_model('qgcn', epoch)
                    elif should_stop:
                        print(f"QGCN early stopping triggered at epoch {epoch}")
                        qgcn_continue_training = False
                
                if sgcn_continue_training and self.sgcn_early_stopping is not None:
                    # Choose the metric to monitor
                    early_stop_metric = test_sgcn_r2 if monitor_metric == 'r2' else test_sgcn_mse
                    should_stop = self.sgcn_early_stopping(early_stop_metric, self.sgcn_model)
                    
                    # ONLY save if the model improved (counter was reset to 0)
                    if not should_stop and self.sgcn_early_stopping.counter == 0:
                        if self.cache_run:
                            self.__save_best_model('sgcn', epoch)
                    elif should_stop:
                        print(f"SGCN early stopping triggered at epoch {epoch}")
                        sgcn_continue_training = False
                
                # If both models have stopped, break the training loop
                if not qgcn_continue_training and not sgcn_continue_training:
                    print(f"All models stopped early at epoch {epoch}. Ending training.")
                    break
            else:
                # Without early stopping, save models periodically or at the end
                if epoch == num_epochs:  # Save at the final epoch
                    if self.cache_run:
                        if self.qgcn_model_exists:
                            self.__save_best_model('qgcn', epoch)
                        if self.sgcn_model_exists:
                            self.__save_best_model('sgcn', epoch)
            
            stop_time = time.time()

            # Build the display string with R² scores
            epoch_str = "Epoch: {:03d}, ".format(epoch)
            loss_str = "QGCN_Loss: {:.5f}, SGCN_Loss: {:.5f}, ".format(qgcn_loss, sgcn_loss)
            train_mse_str = "QGCN_Train_MSE: {:.5f}, SGCN_Train_MSE: {:.5f}, ".format(train_qgcn_mse, train_sgcn_mse)
            test_mse_str = "QGCN_Test_MSE: {:.5f}, SGCN_Test_MSE: {:.5f}, ".format(test_qgcn_mse, test_sgcn_mse)
            train_r2_str = "QGCN_Train_R²: {:.4f}, SGCN_Train_R²: {:.4f}, ".format(train_qgcn_r2, train_sgcn_r2)
            test_r2_str = "QGCN_Test_R²: {:.4f}, SGCN_Test_R²: {:.4f}, ".format(test_qgcn_r2, test_sgcn_r2)
            lr_str = "QGCN_LR: {:.2e}, SGCN_LR: {:.2e}".format(current_lr_qgcn, current_lr_sgcn)
            
            # Add early stopping status
            if self.use_early_stopping:
                es_str = "ES_QGCN: {}/{}, ES_SGCN: {}/{}".format(
                    self.qgcn_early_stopping.counter if self.qgcn_early_stopping else 0,
                    self.qgcn_early_stopping.patience if self.qgcn_early_stopping else 0,
                    self.sgcn_early_stopping.counter if self.sgcn_early_stopping else 0,
                    self.sgcn_early_stopping.patience if self.sgcn_early_stopping else 0
                )
                print("{}".format("".join([epoch_str, loss_str, train_mse_str, test_mse_str, train_r2_str, test_r2_str, lr_str, ", ", es_str])))
            else:
                print("{}".format("".join([epoch_str, loss_str, train_mse_str, test_mse_str, train_r2_str, test_r2_str, lr_str])))
            
            print(f"Epoch took a total of {stop_time - start_time}s")

        # SAVE FINAL EPOCH INFORMATION
        self.final_qgcn_epoch = final_qgcn_epoch
        self.final_sgcn_epoch = final_sgcn_epoch

        # Cache models AFTER restoring weights (if early stopping was used) or just cache normally
        if self.cache_run:
            self.__cache_models()

        # Cache results if we need to
        if self.cache_run:
            self.__cache_results(
                train_qgcn_loss_array, train_sgcn_loss_array,
                train_qgcn_mse_array, train_sgcn_mse_array,
                test_qgcn_mse_array, test_sgcn_mse_array,
                train_qgcn_r2_array, train_sgcn_r2_array,
                test_qgcn_r2_array, test_sgcn_r2_array,
                learning_rates_qgcn, learning_rates_sgcn
            )
        
        final_epoch = len(train_qgcn_loss_array)
        print(f"Brain Connectivity Regression Training Complete! (Stopped at epoch {final_epoch})")
        
        # Run evaluation if requested
        evaluation_results = None
        if run_evaluation:
            evaluation_results = self.__run_post_training_evaluation()
        
        training_results = {
            'train_qgcn_loss': train_qgcn_loss_array,
            'train_sgcn_loss': train_sgcn_loss_array,
            'train_qgcn_mse': train_qgcn_mse_array,
            'train_sgcn_mse': train_sgcn_mse_array,
            'test_qgcn_mse': test_qgcn_mse_array,
            'test_sgcn_mse': test_sgcn_mse_array,
            'train_qgcn_r2': train_qgcn_r2_array,
            'train_sgcn_r2': train_sgcn_r2_array,
            'test_qgcn_r2': test_qgcn_r2_array,
            'test_sgcn_r2': test_sgcn_r2_array,
            'learning_rates_qgcn': learning_rates_qgcn,
            'learning_rates_sgcn': learning_rates_sgcn,
            'final_qgcn_epoch': final_qgcn_epoch,
            'final_sgcn_epoch': final_sgcn_epoch,
            'total_planned_epochs': num_epochs,
            'early_stopped': self.use_early_stopping and (
                (self.qgcn_early_stopping and self.qgcn_early_stopping.early_stop) or
                (self.sgcn_early_stopping and self.sgcn_early_stopping.early_stop)
            )
        }
        
        if evaluation_results:
            training_results['evaluation'] = evaluation_results
            
        return training_results

    def __run_post_training_evaluation(self):
        """Run detailed evaluation using the ModelEvaluator class."""
        try:
            print("\n" + "="*80)
            print("RUNNING POST-TRAINING EVALUATION")
            print("="*80)
            
            # Import ModelEvaluator here to avoid circular imports
            from evaluate_gnn import ModelEvaluator
            
            # Determine which model to evaluate (prefer SGCN, fallback to QGCN)
            model_path = None
            model_dir = None
            
            if self.sgcn_model_exists and self.sgcn_specific_run_dir:
                model_type = 'SGCN'
                model_path = os.path.join(self.sgcn_specific_run_dir, "model.pth")
                model_dir = self.sgcn_specific_run_dir
            elif self.qgcn_model_exists and self.qgcn_specific_run_dir:
                model_type = 'QGCN'
                model_path = os.path.join(self.qgcn_specific_run_dir, "model.pth")
                model_dir = self.qgcn_specific_run_dir
            else:
                print("No trained model found for evaluation")
                return None
            
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return None
            
            # Create dataset configuration from experiment data
            # Use the data that was actually loaded in this experiment
            train_data = self.data_struct["geometric"]["sgcn_train_data"] if self.sgcn_model_exists else self.data_struct["geometric"]["qgcn_train_data"]
            test_data = self.data_struct["geometric"]["sgcn_test_data"] if self.sgcn_model_exists else self.data_struct["geometric"]["qgcn_test_data"]
            
            dataset_config = {
                'dataset_name': 'custom_dataset_selfloops_True_edgeft_None_norm_True',
                'num_train': len(train_data),
                'num_test': len(test_data),
                'sparsity': self.sparsity,
                'dataset_type': self.dataset_type
            }
            
            # Get base path (parent of the experiment directory)
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))
            
            print(f"Evaluating {model_type} model: {model_path}")
            print(f"Using dataset config: {dataset_config}")
            print(f"Using dataset type: {self.dataset_type}")
            print(f"Using sparsity: {self.sparsity}")
            
            # Initialize evaluator with sparsity and dataset type
            evaluator = ModelEvaluator(model_path, dataset_config, base_path, sparsity=self.sparsity)
            
            # Evaluate on test set
            print(f"\nEvaluating {model_type} model on test set...")
            test_results = evaluator.evaluate_dataset('test', batch_size=64)
            evaluator.print_evaluation_summary(test_results)
            
            # Evaluate on training set for comparison
            print(f"\nEvaluating {model_type} model on training set...")
            train_results = evaluator.evaluate_dataset('train', batch_size=64)
            evaluator.print_evaluation_summary(train_results)
            
            # Print comprehensive comparison
            print(f"\n{'='*80}")
            print(f"COMPREHENSIVE DATASET COMPARISON")
            print(f"{'='*80}")
            print(f"TRAINING SET:")
            print(f"  True values:   mean={train_results['true_values'].mean():.4f}, std={train_results['true_values'].std():.4f}, n={len(train_results['true_values'])}")
            print(f"  Predictions:   mean={train_results['predictions'].mean():.4f}, std={train_results['predictions'].std():.4f}")
            print(f"  R² Score:      {train_results['r2_score']:.4f}")
            print(f"  RMSE:          {train_results['rmse']:.4f}")
            print()
            print(f"TEST SET:")
            print(f"  True values:   mean={test_results['true_values'].mean():.4f}, std={test_results['true_values'].std():.4f}, n={len(test_results['true_values'])}")
            print(f"  Predictions:   mean={test_results['predictions'].mean():.4f}, std={test_results['predictions'].std():.4f}")
            print(f"  R² Score:      {test_results['r2_score']:.4f}")
            print(f"  RMSE:          {test_results['rmse']:.4f}")
            print(f"{'='*80}")
            
            # Generate plots
            print(f"\nGenerating evaluation plots...")
            
            # Plot predictions
            test_plot_path = os.path.join(model_dir, "test_predictions.png")
            train_plot_path = os.path.join(model_dir, "train_predictions.png")
            
            evaluator.plot_predictions(test_results, save_path=test_plot_path)
            evaluator.plot_predictions(train_results, save_path=train_plot_path)
            
            # Plot training curves
            training_curves_path = os.path.join(model_dir, "training_curves.png")
            evaluator.plot_training_curves(model_dir, save_path=training_curves_path)
            
            # Save detailed results
            results_path = os.path.join(model_dir, "evaluation_results.pkl")
            detailed_results = {'train': train_results, 'test': test_results}
            with open(results_path, 'wb') as f:
                pickle.dump(detailed_results, f)
            print(f"Detailed evaluation results saved to {results_path}")
            
            print("\n" + "="*80)
            print("POST-TRAINING EVALUATION COMPLETE")
            print("="*80)
            
            return {
                'model_type': model_type,
                'model_path': model_path,
                'train_results': train_results,
                'test_results': test_results,
                'plots_generated': [test_plot_path, train_plot_path, training_curves_path],
                'results_saved': results_path
            }
            
        except Exception as e:
            print(f"Error during post-training evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def plot_history(data, labels):
        """Plot training history."""
        font = {'weight':'bold', 'size':8}
        matplotlib.rc('font', **font)
        matplotlib.rcParams.update({'font.size':8})

        indicators = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        leftover_count = len(data) % 2
        num_rows = len(data) // 2
        col_count = 2
        row_count = num_rows + leftover_count
        fig, ax = plt.subplots(nrows=row_count, ncols=col_count, figsize=(12, 8))
        fig.tight_layout(pad=0.8)
        plt.subplots_adjust(wspace=0.4, hspace=0.5)

        # Loop and plot the data and their labels
        for i, row_plts in enumerate(ax):
            for j, row_col_plt in enumerate(row_plts):
                data_index = i * col_count + j
                if data_index < len(data):
                    xdata = list(range(1, len(data[data_index]) + 1))
                    ydata = data[data_index]
                    data_label = labels[data_index]
                    data_indicator = indicators[data_index % len(indicators)]
                    row_col_plt.plot(xdata, ydata, color=data_indicator, label=data_label)
                    row_col_plt.set_xticks(xdata)
                    row_col_plt.legend(loc="upper right")
                    row_col_plt.set_xlabel('Epoch')
                    row_col_plt.set_ylabel(data_label)
                    row_col_plt.set_title('{} vs. No. of epochs'.format(data_label))
                else:
                    row_col_plt.set_visible(False)
        plt.show()