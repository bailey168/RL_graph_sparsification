"""Vectorised pruning episodes on the GPU, rewarded by a frozen predictor.

One row per subject. A row starts from its full graph; each step it either STOPs (episode ends, reward 0) or removes
the next 5% of the original edges (reward e_t - e_{t+1}, the decrease in the frozen predictor's squared error on the
scaled target). A row that reaches 0 edges is done. The return of an episode is e_0 - e_stop.

Rows leave only by stopping or reaching the end, so every active row has taken the same number of steps t, and they
all remove the same number of edges next (schedule[t] - schedule[t+1]).
"""
import torch

from rl.masking import removal_schedule
from rl.study import predict_masked


class PruningEnv:
    def __init__(self, predictor, data, n_steps=20, predict_batch_size=4096):
        self.predictor = predictor
        self.data = data
        self.n_steps = n_steps
        self.schedule = removal_schedule(data.num_edges, n_steps)
        self.predict_batch_size = predict_batch_size

    def _predict(self, subjects, mask):
        return predict_masked(self.predictor, subjects, mask, self.data, self.predict_batch_size)

    def reset(self, subjects):
        """Start one episode per subject (SubjectTensors) from the full graph."""
        self.subjects = subjects
        B, E = len(subjects), self.data.num_edges
        device = subjects.y.device
        self.t = 0
        self.mask = torch.ones(B, E, dtype=torch.bool, device=device)
        self.done = torch.zeros(B, dtype=torch.bool, device=device)
        self.pred = self._predict(subjects, self.mask)
        self.err = (self.pred - subjects.y) ** 2
        self.pred0, self.err0 = self.pred.clone(), self.err.clone()
        self.stop_step = torch.full((B,), self.n_steps, dtype=torch.long, device=device)   # removal steps taken
        self.removal_step = torch.full((B, E), -1, dtype=torch.int8, device=device)       # step an edge was removed
        return self

    @property
    def n_remove(self):
        """Edges every active row removes at the next step."""
        return self.schedule[self.t] - self.schedule[self.t + 1]

    def step(self, stop, remove):
        """Apply one action per row. stop: (B,) bool; remove: (B, E) bool edges to drop (used where not stopping).

        Done rows are ignored. Returns (reward (B,), done (B,) after the step).
        """
        if self.t >= self.n_steps:
            raise RuntimeError("All episodes are finished")
        active = ~self.done
        stopping = active & stop
        prune = active & ~stop
        reward = torch.zeros_like(self.err)

        if prune.any():
            rows = prune.nonzero(as_tuple=True)[0]
            drop = remove & prune.unsqueeze(1)
            self.mask &= ~drop
            self.removal_step[drop] = self.t
            pred = self._predict(self.subjects.subset(rows), self.mask[rows])
            err = (pred - self.subjects.y[rows]) ** 2
            reward[rows] = self.err[rows] - err
            self.err[rows], self.pred[rows] = err, pred

        self.stop_step[stopping] = self.t
        self.t += 1
        self.done |= stopping
        if self.t == self.n_steps:   # 0 edges left
            self.done |= prune
        return reward, self.done.clone()

    @property
    def keep_frac(self):
        """(B,) fraction of the original edges each row currently keeps."""
        return self.mask.sum(1).float() / self.data.num_edges
