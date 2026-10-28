# metrics.py
import numpy as np

def calc_r2_corr(y_true, y_pred):
    """Calculates squared Pearson correlation coefficient R^2(corr)"""
    # Convert to numpy arrays to handle both pandas.Series and numpy arrays safely
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    
    if len(np.unique(y_pred_flat)) == 1:
        return 0.0
    return np.corrcoef(y_true_flat, y_pred_flat)[0, 1] ** 2