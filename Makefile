.PHONY: setup lint typecheck test check format lint-sg

all-checks:
	uv run pre-commit run --all-files
	@echo "\nAll checks passed successfully!"

setup:
	uv sync
	uv run pre-commit install

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint lint-sg typecheck
lint:
	uv run ruff check .
	uv run ruff format --check .

lint-sg:
	uv run sg scan --report-style medium

typecheck:
	uv run ty check .

test:
	uv run pytest

testverbose:
	uv run pytest --verbose
