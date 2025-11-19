# Claude Notes

- This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Run `uv sync` after pulling new commits.
- Prefer `uv run …` for every Python entry point (for example, `uv run python scripts/convert_bloomberg_to_parquet.py` or `uv run pytest`).
- If you want an interactive shell, source `.venv/bin/activate` after the `uv sync` step; uv creates and updates the environment automatically.

## Running Tests

Due to ROS plugin conflicts on this system, always run tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

For convenience, you can create an alias:
```bash
alias pytest-xtrend='PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest'
```

Examples:
```bash
# Run all tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest

# Run specific test file
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_smoke.py -v

# Run with coverage
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest --cov=src --cov-report=term-missing
```
- Parquets are in this folder: data/bloomberg/processed