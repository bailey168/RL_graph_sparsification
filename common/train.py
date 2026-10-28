import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from common.utils import EarlyStopping, calc_r2_corr


def get_optimizer_params(model, weight_decay):
    """No weight decay on biases and norm layers (as in the notebook)."""
    no_decay = ['bias', 'norm']
    return [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
         "weight_decay": 0.0},
    ]


def make_scheduler(optimizer, max_epochs, warmup_epochs):
    """Linear warmup followed by cosine decay, stepped once per epoch."""
    warmup_epochs = min(warmup_epochs, max_epochs - 1)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs))
    if warmup_epochs <= 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, criterion, device):
    """Returns (mean loss on scaled target, scaled preds, scaled actuals, eids)."""
    model.eval()
    total_loss = 0.0
    preds, actuals, eids = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        total_loss += criterion(out, data.y.view(-1)).item() * data.num_graphs
        preds.append(out.cpu().view(-1).numpy())
        actuals.append(data.y.cpu().view(-1).numpy())
        eids.append(data.eid.cpu().view(-1).numpy())
    return (total_loss / len(loader.dataset), np.concatenate(preds),
            np.concatenate(actuals), np.concatenate(eids))


def evaluate(model, loader, criterion, device, y_preprocessor):
    """Metrics on the original target scale."""
    _, preds, actuals, eids = predict(model, loader, criterion, device)
    preds = y_preprocessor.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals = y_preprocessor.inverse_transform(actuals.reshape(-1, 1)).flatten()
    metrics = {
        'r2': float(r2_score(actuals, preds)),
        'r2_corr': calc_r2_corr(actuals, preds),
        'mae': float(mean_absolute_error(actuals, preds)),
        'rmse': float(np.sqrt(mean_squared_error(actuals, preds))),
    }
    return metrics, actuals, preds, eids


def fit(model, loaders, train_cfg, device, y_preprocessor):
    """Train with early stopping on val MSE, restore the best weights, and evaluate all splits."""
    start = time.time()
    model = model.to(device)
    fwd_model = torch.compile(model, dynamic=True) if train_cfg['compile'] else model

    optimizer = torch.optim.AdamW(get_optimizer_params(model, train_cfg['weight_decay']), lr=train_cfg['lr'])
    scheduler = make_scheduler(optimizer, train_cfg['max_epochs'], train_cfg['warmup_epochs'])
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(train_cfg['early_stop_patience'], train_cfg['early_stop_min_delta'])

    train_losses, val_losses, lrs = [], [], []
    with tqdm(range(1, train_cfg['max_epochs'] + 1), desc="Epochs", leave=False) as pbar:
        for epoch in pbar:
            train_loss = train_epoch(fwd_model, loaders['train'], optimizer, criterion, device)
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
            val_loss, _, _, _ = predict(fwd_model, loaders['val'], criterion, device)

            if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} (train={train_loss}, val={val_loss})")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", es=early_stopping.counter)

            if early_stopping.step(val_loss, model, epoch):
                break

    model.load_state_dict(early_stopping.best_state)

    metrics, test_out = {}, None
    for name in ('train', 'val', 'test'):
        split_metrics, actuals, preds, eids = evaluate(model, loaders[name], criterion, device, y_preprocessor)
        metrics.update({f'{name}_{k}': v for k, v in split_metrics.items()})
        if name == 'test':
            test_out = {'eid': eids, 'actual': actuals, 'predicted': preds}

    return {
        'metrics': metrics,
        'best_epoch': early_stopping.best_epoch,
        'epochs_run': len(train_losses),
        'history': {'train_loss': train_losses, 'val_loss': val_losses, 'lr': lrs},
        'test_predictions': test_out,
        'minutes': (time.time() - start) / 60.0,
    }
