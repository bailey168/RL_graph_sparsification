import copy
import yaml


def load_config(path, overrides=None):
    """Load a YAML config and apply 'a.b.c=value' overrides (values parsed as YAML)."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    for item in overrides or []:
        key, value = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)

    return cfg


def snapshot(cfg, **extra):
    """Deep copy of the config with run-specific fields added, for saving next to results."""
    snap = copy.deepcopy(cfg)
    snap['run'] = extra
    return snap
