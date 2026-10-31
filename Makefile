.PHONY: install data benchmark test lint

install:
	pip install -e ".[dev,spark,plots,tracking]"

data:
	cmapss-rul download

benchmark: data
	cmapss-rul benchmark --seeds 42 43 44 --figures docs/figures

test:
	pytest

lint:
	ruff check src tests && ruff format --check src tests
