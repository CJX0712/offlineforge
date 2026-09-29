.PHONY: install test lint benchmark demo

install:
	pip install -r requirements.txt
	pip install pytest ruff

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

benchmark:
	python -m offlineforge.cli

demo: benchmark
