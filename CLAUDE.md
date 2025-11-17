# Claude Notes

- This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Run `uv sync` after pulling new commits.
- Prefer `uv run …` for every Python entry point (for example, `uv run python scripts/convert_bloomberg_to_parquet.py` or `uv run pytest`).
- If you want an interactive shell, source `.venv/bin/activate` after the `uv sync` step; uv creates and updates the environment automatically.
