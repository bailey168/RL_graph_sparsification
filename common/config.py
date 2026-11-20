import copy
import os

import yaml


def deep_merge(base, override):
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path):
    """Load a YAML file, resolving an optional `base: parent.yaml` key (relative to this file) recursively."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop('base', None)
    if base is None:
        return cfg
    base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(os.path.abspath(path)), base)
    return deep_merge(_load_yaml(base_path), cfg)


def get_dotted(cfg, key):
    """cfg['a']['b']['c'] for key 'a.b.c'."""
    node = cfg
    for part in key.split('.'):
        node = node[part]
    return node


def set_dotted(cfg, key, value):
    """Set cfg['a']['b']['c'] = value for key 'a.b.c', creating missing dicts."""
    node = cfg
    parts = key.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path, overrides=None):
    """Load a YAML config and apply 'a.b.c=value' overrides (values parsed as YAML).

    A config may start with `base: other.yaml` to inherit from another config; its own keys are
    deep-merged on top.
    """
    cfg = _load_yaml(path)

    for item in overrides or []:
        key, value = item.split('=', 1)
        set_dotted(cfg, key, yaml.safe_load(value))
    return cfg


def snapshot(cfg, **extra):
    """Deep copy of the config with run-specific fields added, for saving next to results."""
    snap = copy.deepcopy(cfg)
    snap['run'] = extra
    return snap
