import os
import random
import numpy as np
import torch


def set_deterministic_training(seed=42):
    """Seed everything and enable deterministic ops (adapted from others/gnn_training_pipeline/util)."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed)


def calc_r2_corr(y_true, y_pred):
    """Squared Pearson correlation coefficient R^2(corr)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(np.unique(y_pred)) == 1:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)


class EarlyStopping:
    """Stops training when the monitored loss hasn't improved for `patience` epochs.

    Keeps a CPU copy of the best state_dict so it can be restored after training.
    """

    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0

    def step(self, loss, model, epoch):
        """Returns True if training should stop."""
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
