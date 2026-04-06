# =============================================================================
# E-Commerce Similarity Search — Makefile
# =============================================================================
# Common development commands for convenience.
#
# Usage:
#   make run          → Start the dev server
#   make test         → Run all tests
#   make docker       → Build and run with Docker Compose
#   make benchmark    → Run search latency benchmark
# =============================================================================

.PHONY: run test test-coverage docker docker-down benchmark clean lint help

# --- Default target ---
help:
	@echo "Available commands:"
	@echo "  make run            Start development server"
	@echo "  make test           Run all tests"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make test-unit      Run unit tests only (fast)"
	@echo "  make test-integration Run integration tests only"
	@echo "  make docker         Build and run with Docker Compose"
	@echo "  make docker-down    Stop Docker Compose services"
	@echo "  make benchmark      Run search latency benchmark"
	@echo "  make lint           Run linting checks"
	@echo "  make clean          Remove caches, temp files"
	@echo "  make setup          Install dependencies"

# --- Development ---
run:
	python -m app.main

setup:
	pip install -r requirements.txt

# --- Testing ---
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=app --cov-report=term-missing -v

test-unit:
	pytest tests/test_search_service.py tests/test_preprocessing_service.py tests/test_data_loader.py -v

test-integration:
	pytest tests/test_integration.py -v

# --- Docker ---
docker:
	docker-compose up --build

docker-down:
	docker-compose down

docker-build:
	docker build -t similarity-search .

# --- Benchmarking ---
benchmark:
	python scripts/benchmark.py

# --- Code Quality ---
lint:
	python -m py_compile app/main.py
	python -m py_compile app/config.py
	python -m py_compile app/services/search_service.py
	python -m py_compile app/services/embedding_service.py
	python -m py_compile app/services/preprocessing_service.py
	python -m py_compile app/services/data_loader.py
	@echo "All modules compile successfully."

# --- Cleanup ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned caches and temp files."
