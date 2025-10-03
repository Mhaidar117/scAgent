.PHONY: fmt lint test all clean install install-dev help

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"

fmt: ## Format code with black and ruff
	black scqc_agent tests
	ruff check --fix scqc_agent tests

lint: ## Run linting checks
	ruff check scqc_agent tests
	black --check scqc_agent tests
	mypy scqc_agent

test: ## Run tests
	pytest -v

test-cov: ## Run tests with coverage
	pytest -v --cov=scqc_agent --cov-report=html --cov-report=term

all: lint test ## Run linting and tests

clean: ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Development workflow commands
dev-setup: install-dev pre-commit-install ## Complete development setup
	@echo "✅ Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

check: fmt lint test ## Format, lint, and test (full check)

# Quick development commands
quick-test: ## Run tests without coverage (faster)
	pytest -x --tb=short

watch-test: ## Run tests in watch mode (requires pytest-watch)
	ptw --runner "pytest -x --tb=short"

# End-to-end testing
e2e: ## Run end-to-end smoke test
	@echo "🚀 Running end-to-end smoke test..."
	bash scripts/e2e_smoke.sh

e2e-pytest: ## Run end-to-end pytest tests
	@echo "🧪 Running end-to-end pytest tests..."
	pytest tests/test_e2e_agent.py -v -s --tb=short

e2e-full: ## Run complete end-to-end test suite
	@echo "🔧 Running complete end-to-end test suite..."
	@echo "1. Running smoke test..."
	bash scripts/e2e_smoke.sh
	@echo ""
	@echo "2. Running pytest e2e tests..."
	pytest tests/test_e2e_agent.py tests/test_kb_retriever.py tests/test_doublets_stub.py -v -s --tb=short
	@echo ""
	@echo "✅ All end-to-end tests completed!"

test-kb: ## Test knowledge base retriever functionality
	pytest tests/test_kb_retriever.py -v -s

test-synth: ## Test synthetic data generation
	@echo "🧬 Testing synthetic data generation..."
	@if [ -d "scQC" ]; then \
		echo "Using virtual environment..."; \
		source scQC/bin/activate && python -c "from scqc_agent.tests.synth import make_synth_adata; adata = make_synth_adata(); print(f'Generated {adata.n_obs} cells × {adata.n_vars} genes')"; \
	else \
		python3 -c "from scqc_agent.tests.synth import make_synth_adata; adata = make_synth_adata(); print(f'Generated {adata.n_obs} cells × {adata.n_vars} genes')"; \
	fi

