"""Shared pytest fixtures for sn5_solver tests."""
import sys
from pathlib import Path

# Make sn5_solver importable from tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture
def sample_grid_3x3():
    """Simple 3x3 grid for grid-manipulation tests."""
    return [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


@pytest.fixture
def sample_train_identity():
    """Train pairs where output == input (identity transform)."""
    return [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
        {"input": [[9, 0], [1, 2]], "output": [[9, 0], [1, 2]]},
    ]


@pytest.fixture
def sample_train_h_flip():
    """Train pairs where output == horizontal flip of input."""
    return [
        {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
        {"input": [[4, 5, 6]], "output": [[6, 5, 4]]},
        {"input": [[7, 8, 9]], "output": [[9, 8, 7]]},
    ]


@pytest.fixture
def identity_code():
    """Valid Python code that returns input unchanged."""
    return '''```python
def transform(grid):
    return grid
```'''


@pytest.fixture
def bad_code():
    """Invalid Python code (syntax error)."""
    return '''```python
def transform(grid:
    return grid
```'''


@pytest.fixture
def malicious_code():
    """Code that tries dangerous imports."""
    return '''```python
import os
def transform(grid):
    os.system("rm -rf /")
    return grid
```'''
