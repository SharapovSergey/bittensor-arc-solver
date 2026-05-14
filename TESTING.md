# Testing sn5_solver

## Setup

```bash
# from project root
python3 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-asyncio pytest-cov httpx
```

## Running tests

```bash
cd sn5_solver

# Fast tests (default — unit only)
pytest

# Verbose
pytest -v

# Specific class/test
pytest tests/test_arc_prep_phase.py::TestParseGrid -v

# With coverage
pytest --cov=. --cov-report=html

# Include slow / GPU / integration
pytest -m ""  # disable default marker filter
```

## Test categories (pytest markers)

- (default, no marker) — fast unit tests, no network, no GPU
- `@pytest.mark.slow` — bench-as-regression, may take minutes
- `@pytest.mark.gpu` — TTT, vLLM (needs GPU)
- `@pytest.mark.integration` — hits real OpenRouter (needs `OPENROUTER_API_KEY`)

Default config (in `pyproject.toml`) skips slow/gpu/integration.

## Adding new tests

- Place in `tests/test_<module>.py`
- Class names `Test*`, function names `test_*`
- Use fixtures from `tests/conftest.py` (sample grids, codes)
- For new fixtures, add to `conftest.py` so other tests can reuse

## CI

`.github/workflows/test.yml` runs all fast tests on every push / PR to `main`.
Status check blocks merge if tests fail.
