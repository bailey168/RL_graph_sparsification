import numpy as np
import torch
from sklearn.metrics import r2_score

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=False, monitor='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                             Default: 0
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value.
                                       Default: True
            verbose (bool): If True, prints a message for each validation metric improvement. 
                          Default: False
            monitor (str): Metric to monitor. Options: 'loss', 'r2'. Default: 'loss'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_metric, model=None):
        """
        Call this method after each epoch with the validation metric.
        
        Args:
            val_metric (float): Current validation metric (loss or r2_score)
            model (torch.nn.Module): Model to save weights from (if restore_best_weights=True)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        current_epoch = getattr(model, 'current_epoch', 0) if model else 0
        
        if self.monitor == 'r2':
            # For R², higher is better
            score = val_metric
            is_improvement = score > self.best_score + self.min_delta if self.best_score is not None else True
        else:
            # For loss, lower is better (traditional behavior)
            score = -val_metric  # We want to maximize the negative loss (minimize loss)
            is_improvement = score > self.best_score + self.min_delta if self.best_score is not None else True
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = current_epoch  # Initialize best_epoch
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                metric_name = 'R²' if self.monitor == 'r2' else 'validation loss'
                print(f'Initial {metric_name}: {val_metric:.6f}')
        elif not is_improvement:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            self.best_score = score
            self.best_epoch = current_epoch  # Update best_epoch when improvement found
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                metric_name = 'R²' if self.monitor == 'r2' else 'validation loss'
                print(f'{metric_name} improved: {val_metric:.6f}')
                
        return self.early_stop
    
    def restore_weights(self, model):
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print('Restored best model weights')