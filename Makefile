.PHONY: install lint format type test smoke run-proposed run-ablation docs clean

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

type:
	uv run mypy src

test:
	uv run pytest -v

smoke:
	uv run pytest tests/integration/test_smoke_pipeline.py -v

run-proposed:
	uv run msdl-jci --mode proposed --epochs 40

run-ablation:
	uv run python -m msdl_jci.experiments.run_ablation

docs:
	echo "Docs are Markdown under docs/. No build step."

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov coverage.xml
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
