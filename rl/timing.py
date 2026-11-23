"""Per-stage timing for the RL pipeline, kept in {stage_dir}/timing.json and accumulated across resumed sessions.

    timer = StageTimer(stage_dir, device)
    with timer.section('load_data'):
        ...
    timer.add('ppo_rollout', seconds)   # time measured elsewhere (e.g. per PPO iteration)
    timer.finish()
    row.update(timer.flat())            # {'time_load_data_min': ..., 'time_total_min': ...}

timing.json holds {'sections': {name: minutes}, 'total_minutes', 'sessions', 'started_at', 'finished_at'}. It is
rewritten after every section, so an interruption loses at most the section that was running.
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

import torch

TIMING_FILE = 'timing.json'


def _now():
    return datetime.now().isoformat(timespec='seconds')


def read_timing(stage_dir):
    """The stage's timing.json as a dict, or None if it doesn't exist."""
    path = os.path.join(stage_dir, TIMING_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class StageTimer:
    def __init__(self, stage_dir, device=None):
        self.path = os.path.join(stage_dir, TIMING_FILE)
        self.cuda = device is not None and torch.device(device).type == 'cuda'
        os.makedirs(stage_dir, exist_ok=True)
        prev = read_timing(stage_dir) or {}
        self.sections = dict(prev.get('sections', {}))
        self.sessions = prev.get('sessions', 0) + 1
        self.started_at = prev.get('started_at') or _now()
        self.finished_at = None
        self.save()

    def sync(self):
        """Wait for queued GPU work, so a clock read after it measures that work."""
        if self.cuda:
            torch.cuda.synchronize()

    @property
    def total_minutes(self):
        return sum(self.sections.values())

    def add(self, name, seconds, save=True):
        self.sections[name] = self.sections.get(name, 0.0) + seconds / 60.0
        if save:
            self.save()

    @contextmanager
    def section(self, name):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.add(name, time.perf_counter() - start)

    def finish(self):
        self.finished_at = _now()
        self.save()

    def to_dict(self):
        return {'sections': self.sections, 'total_minutes': self.total_minutes, 'sessions': self.sessions,
                'started_at': self.started_at, 'finished_at': self.finished_at}

    def save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, self.path)

    def flat(self, prefix='time_'):
        """Section and total minutes as flat metrics-row fields."""
        return {**{f'{prefix}{k}_min': v for k, v in self.sections.items()}, f'{prefix}total_min': self.total_minutes,
                f'{prefix}sessions': self.sessions}


def pipeline_timing(stage_dirs):
    """Per-stage and total minutes of a run. stage_dirs: {stage name: dir}, in pipeline order.

    Returns {'<stage>_minutes': ..., 'pipeline_minutes_total': sum of the stages that have timing.json,
    'pipeline_started_at': first stage's start, 'pipeline_finished_at': last stage's finish}.
    """
    out, total, started, finished = {}, 0.0, None, None
    for stage, d in stage_dirs.items():
        t = read_timing(d)
        out[f'{stage}_minutes'] = t['total_minutes'] if t else None
        if t:
            total += t['total_minutes']
            started = started or t.get('started_at')
            finished = t.get('finished_at') or finished
    out.update(pipeline_minutes_total=total, pipeline_started_at=started, pipeline_finished_at=finished)
    return out


def timing_rows(stage_dirs, **keys):
    """Long-format rows {**keys, stage, component, minutes}: every section, a 'total' per stage, and pipeline/total."""
    rows = []
    for stage, d in stage_dirs.items():
        t = read_timing(d)
        if not t:
            continue
        rows += [{**keys, 'stage': stage, 'component': name, 'minutes': m} for name, m in t['sections'].items()]
        rows.append({**keys, 'stage': stage, 'component': 'total', 'minutes': t['total_minutes']})
    if rows:
        rows.append({**keys, 'stage': 'pipeline', 'component': 'total',
                     'minutes': pipeline_timing(stage_dirs)['pipeline_minutes_total']})
    return rows
