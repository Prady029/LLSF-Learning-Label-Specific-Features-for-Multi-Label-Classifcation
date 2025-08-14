.PHONY: help clean install install-dev test test-cov lint format type-check build upload-test upload

help:
	@echo "Available commands:"
	@echo "  clean        Remove build artifacts"
	@echo "  install      Install package"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run flake8 linter"
	@echo "  format       Format code with black"
	@echo "  type-check   Run mypy type checker"
	@echo "  build        Build distribution packages"
	@echo "  upload-test  Upload to TestPyPI"
	@echo "  upload       Upload to PyPI"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=llsf --cov-report=html --cov-report=term

lint:
	flake8 llsf tests

format:
	black llsf tests

type-check:
	mypy llsf

build: clean
	python -m build

upload-test: build
	python -m twine upload --repository testpypi dist/*

upload: build
	python -m twine upload dist/*
