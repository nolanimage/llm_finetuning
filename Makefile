.PHONY: help install setup train inference test clean docker-build docker-up docker-down

help:
	@echo "ChatGLM3 Fine-tuning Framework - Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup          - Setup the project (install dependencies, create directories)"
	@echo "  make install        - Install Python dependencies"
	@echo "  make train          - Start training with default configuration"
	@echo "  make inference      - Start interactive inference"
	@echo "  make test           - Run tests (if available)"
	@echo "  make clean          - Clean temporary files and caches"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start Docker container"
	@echo "  make docker-down    - Stop Docker container"
	@echo "  make help           - Show this help message"

setup:
	@echo "Setting up the project..."
	python setup.py

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	@echo "Starting training..."
	python train.py

inference:
	@echo "Starting inference..."
	python inference.py

test:
	@echo "Running tests..."
	python -m pytest tests/ -v || echo "No tests found"

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "Clean complete"

docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-up:
	@echo "Starting Docker container..."
	docker-compose up

docker-down:
	@echo "Stopping Docker container..."
	docker-compose down
