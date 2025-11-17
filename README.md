# X-Trend (Revised)

> **Environment note:** This repository standardizes on [uv](https://github.com/astral-sh/uv) for Python dependency management and script execution. Run `uv sync` to create/update `.venv`, then prefix commands with `uv run` (for example, `uv run python scripts/convert_bloomberg_to_parquet.py`).

## Quick Start
- Install dependencies: `uv sync --frozen`
- Activate the virtual environment (optional): `source .venv/bin/activate`
- Run the main entry point: `uv run python main.py`
- Execute helper scripts the same way, e.g., `uv run python scripts/generate_bloomberg_workbook.py --expanded`

## Project Structure Highlights
- `data/bloomberg/` – Bloomberg export workflow, symbol maps, reshaping scripts
- `xtrend/` – Core library code for the X-Trend experiments
- `dev/` – Persistent dev-doc pattern for Claude/Code agents
- `docs/` – Planning artifacts and research roadmaps

## Troubleshooting
- Re-sync dependencies after pulling new changes: `uv sync`
- Refresh the lock if dependencies change: `uv lock`
- Run tests via uv to ensure consistent environments: `uv run pytest`

See `data/bloomberg/README.md` for the Bloomberg-specific flow and `phases.md` for the overall implementation plan.
