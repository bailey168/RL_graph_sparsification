import os

import numpy as np
import pytest

from common.data import parse_region_map

REGION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'region_names')


@pytest.fixture
def fc21():
    """(edge_list, X): the FC21 edge list and 8 synthetic subjects' signed edge values."""
    rng = np.random.default_rng(0)
    _, edge_list, _ = parse_region_map(os.path.join(REGION_DIR, 'FC21_regions.txt'))
    X = rng.normal(0, 0.5, size=(8, len(edge_list))).astype(np.float32)
    return edge_list, X
