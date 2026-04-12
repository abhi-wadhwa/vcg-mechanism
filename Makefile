.PHONY: install dev test lint format typecheck app clean demo docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

app:
	streamlit run src/viz/app.py

demo:
	python -m examples.demo

cli-demo:
	python -m src.cli demo

docker:
	docker build -t vcg-mechanism .
	docker run -p 8501:8501 vcg-mechanism

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf src/__pycache__ src/core/__pycache__ src/viz/__pycache__
	rm -rf tests/__pycache__
	rm -rf *.egg-info dist build
	find . -name "*.pyc" -delete
