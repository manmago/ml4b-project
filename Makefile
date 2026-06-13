# ML4B Gym Exercise Recognition — common developer commands.
#
# Usage: make <target>   (run from the project root)
# These wrap the canonical `uv run ...` commands so contributors don't have to
# remember them. See README.md for the full workflow.

.PHONY: help run train update calibrate test lint format check setup

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Install all dependencies (incl. dev tools) into .venv.
	uv sync --extra dev

run:  ## Launch the Streamlit app at http://localhost:8501.
	uv run streamlit run app/streamlit_app.py

train:  ## Train baseline + current models on the Kaggle anchor (needs data/raw/kaggle_gym_imu/).
	uv run python scripts/train_model.py

update:  ## Rebuild BOTH models (baseline Kaggle-only + current Kaggle+Testdaten) for the app comparison.
	uv run python scripts/rebuild_from_testdaten.py

calibrate:  ## Analyse the activity-gate thresholds against rest + exercise energy (read-only).
	uv run python scripts/calibrate_gate.py

test:  ## Run the unit test suite.
	uv run pytest

lint:  ## Lint the codebase with ruff.
	uv run ruff check .

format:  ## Auto-format the codebase with ruff.
	uv run ruff format .

check: lint  ## Run all quality gates (format check + lint + tests).
	uv run ruff format --check .
	uv run pytest -q
