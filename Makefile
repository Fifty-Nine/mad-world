.PHONY: setup lint typecheck test check format

setup:
	uv sync
	uv run pre-commit install

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

testverbose:
	uv run pytest --verbose

check: lint typecheck test
	@echo "\nAll checks passed successfully!"
