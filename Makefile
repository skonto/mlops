# ======================
# Development & Quality
# ======================
install-dev:
	uv sync --dev --extra torch --extra onnx

lint:
	uv run ruff check src/ app/

format:
	uv run black --check src/ app/

type-check:
	uv run mypy src/ app/

test:
	uv run pytest -q

validate: lint type-check test