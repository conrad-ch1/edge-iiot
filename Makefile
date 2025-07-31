# Edge-IIoT Dataset Project Makefile
# This Makefile provides common tasks for working with the Edge-IIoT dataset

.PHONY: help clean data check-data lint test setup install dev-install format docs

# Default target
.DEFAULT_GOAL := help

# Variables
DATA_DIR := data
SCRIPTS_DIR := scripts
CSV_FILE := $(DATA_DIR)/DNN-EdgeIIoT-dataset.csv
MARKER_FILE := $(DATA_DIR)/.DNN-EdgeIIoT-dataset.ok

# Colors for help output
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m

help: ## Show this help message
	@echo "$(BOLD)Edge-IIoT Dataset Project$(RESET)"
	@echo ""
	@echo "$(BOLD)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Data management targets
data: ## Download the Edge-IIoT dataset
	@echo "Setting up data directory..."
	@mkdir -p $(DATA_DIR)
	@if [ -f "$(MARKER_FILE)" ]; then \
		echo "CSV already present."; \
	else \
		echo "Downloading CSV..."; \
		curl -L --retry 3 -o "$(CSV_FILE)" "https://edge-iiot-bucket.s3.eu-central-1.amazonaws.com/DNN-EdgeIIoT-dataset.csv?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCIQCYaAHy4cvedoLNgDhRH8NjMy%2BR80gZjMHSYzBRs0hDywIgGuDtto%2BcWvaiz2BGJDgDUapZZFRDNmHzXOR313GluMEqggMIwP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw1NzkxMTExMTQzMjQiDF9F3PIYbRxoG3%2FK9yrWAhDdfdzMeh%2FXkNyvfI%2FYIXG0dibTFSsCfyfHwoG5IKC3wB7ftzXOTl7QcoFR3QioyBaubSBes9FSQzdg1AV2i9hAXinbLetCCv4aFtlISiQENWUXSqCQYfgN%2FELWwfkX%2BHpeQwffN3%2FlsOyJIJ07rqdC4B99oyKx%2BOdkd8GQD7j%2F2FlevTG960WYrSmwCGdgX%2B2S0ofTmJPkrDMJH6fEKaE2vMVT1xWqWqAcfLknzAywrK0q6HFN2DetwsbNDa1qlnDJFjH3XP%2FxECTsHjiAHTC5meVIVxDbzXTRyaCmu18xTEfjdWQBBw8Ancv2TMCe%2FLyn2H0i8rXLYmQKj3cPB3yF0ZvoKUX%2BcByLCSfnB3XYCll558CAeVjJxDvGS92uChV53lkfG1KTNEudFWNOO0FoHeEwd7v3T1s2lIHgJcZKghR096J4KkrcQ6gZhXg5TLIvHZdiqDDS1qjEBjqPAjU8XyTroBj8loH8%2BsM3uAuyhn5909DxOykNVl5Eaw2KBR3rKbpR2G0T6GOcN7E8Dyv9QcC4%2BMM36%2B18ISnk%2FKrbAhLU6ylrE%2BECGVBeeBBKmFgWWpfhUkpMtA0eNhKxg62GnBdWqLuAMj4W%2FZvKMQr70VUWbU%2BKZrccwBYYk7WSwtT8%2BfUf1KwKhyBJHrD%2Fg2PBCCma7s7uhKUtaK%2Fi8%2FvWE%2FTHAzmFq6IaoEY1OExGLNpPvcerSajQE1n1VmI1Mt9lNqBJx5WRZBvWmnGT1eo8R5x6Pvw9UVtchxEowcQGUEmThLtYxlFpO%2BJ304U4XpNvmSYS7b72RIzeYTXMKW7EZKLwWat2R5qyE9aa9Bs%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAYNVNWKZKCNKKOCQC%2F20250730%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20250730T145348Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=119775296ec046ca812d9139924ce8d4e6cab378f646b057f5938a4341e85139"; \
		touch "$(MARKER_FILE)"; \
		echo "Done – data ready in $(DATA_DIR)"; \
	fi

check-data: ## Check if dataset is available
	@if [ -f "$(MARKER_FILE)" ] && [ -f "$(CSV_FILE)" ]; then \
		echo "✓ Dataset is available"; \
		echo "  File: $(CSV_FILE)"; \
		echo "  Size: $$(du -h $(CSV_FILE) | cut -f1)"; \
	else \
		echo "✗ Dataset not found. Run 'make data' to download it."; \
		exit 1; \
	fi

clean-data: ## Remove downloaded dataset
	@echo "Removing dataset files..."
	@rm -f $(CSV_FILE) $(MARKER_FILE)
	@echo "Dataset files removed."

# Development environment targets
setup: ## Set up the development environment
	@echo "Setting up development environment..."
	@if command -v python3 >/dev/null 2>&1; then \
		echo "Python 3 found"; \
		if [ ! -d "venv" ]; then \
			echo "Creating virtual environment..."; \
			python3 -m venv venv; \
		fi; \
		echo "Virtual environment ready."; \
		echo "To activate: source venv/bin/activate"; \
	else \
		echo "Python 3 not found. Please install Python 3."; \
	fi

install: setup ## Install project dependencies (if requirements.txt exists)
	@if [ -f "requirements.txt" ]; then \
		echo "Installing dependencies..."; \
		./venv/bin/pip install -r requirements.txt; \
	else \
		echo "No requirements.txt found. Create one if needed."; \
	fi

dev-install: setup ## Install development dependencies (if requirements-dev.txt exists)
	@if [ -f "requirements-dev.txt" ]; then \
		echo "Installing development dependencies..."; \
		./venv/bin/pip install -r requirements-dev.txt; \
	else \
		echo "No requirements-dev.txt found."; \
	fi

# Code quality targets
lint: ## Run linting (if Python files exist)
	@if find . -name "*.py" -not -path "./venv/*" | grep -q .; then \
		if command -v flake8 >/dev/null 2>&1; then \
			echo "Running flake8..."; \
			flake8 --exclude=venv .; \
		else \
			echo "flake8 not found. Install with: pip install flake8"; \
		fi; \
		if command -v pylint >/dev/null 2>&1; then \
			echo "Running pylint..."; \
			find . -name "*.py" -not -path "./venv/*" -exec pylint {} +; \
		fi; \
	else \
		echo "No Python files found to lint."; \
	fi

format: ## Format code (if Python files exist)
	@if find . -name "*.py" -not -path "./venv/*" | grep -q .; then \
		if command -v black >/dev/null 2>&1; then \
			echo "Running black formatter..."; \
			black --exclude=venv .; \
		else \
			echo "black not found. Install with: pip install black"; \
		fi; \
		if command -v isort >/dev/null 2>&1; then \
			echo "Running isort..."; \
			isort --skip=venv .; \
		else \
			echo "isort not found. Install with: pip install isort"; \
		fi; \
	else \
		echo "No Python files found to format."; \
	fi

test: ## Run tests (if test files exist)
	@if find . -name "*test*.py" -not -path "./venv/*" | grep -q .; then \
		if command -v pytest >/dev/null 2>&1; then \
			echo "Running pytest..."; \
			pytest; \
		elif command -v python3 >/dev/null 2>&1; then \
			echo "Running unittest discover..."; \
			python3 -m unittest discover; \
		fi; \
	else \
		echo "No test files found."; \
	fi

# Documentation targets
docs: ## Generate documentation (if docs directory exists)
	@if [ -d "docs" ]; then \
		if command -v sphinx-build >/dev/null 2>&1; then \
			echo "Building Sphinx documentation..."; \
			sphinx-build -b html docs docs/_build; \
		else \
			echo "sphinx-build not found. Install with: pip install sphinx"; \
		fi; \
	else \
		echo "No docs directory found."; \
	fi

# Utility targets
info: ## Show project information
	@echo "$(BOLD)Project Information$(RESET)"
	@echo "Repository: edge-iiot"
	@echo "Current branch: $$(git branch --show-current 2>/dev/null || echo 'unknown')"
	@echo "Git status:"
	@git status --porcelain 2>/dev/null || echo "Not a git repository"
	@echo ""
	@$(MAKE) check-data 2>/dev/null || true

clean: ## Clean up temporary files and caches
	@echo "Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete."

# CI/CD targets
ci: clean lint test ## Run continuous integration checks
	@echo "CI pipeline completed."

all: clean data setup install lint test ## Run all setup and verification tasks
	@echo "All tasks completed successfully!"
