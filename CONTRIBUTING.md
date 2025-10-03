# Contributing to scQC Agent

Thank you for your interest in contributing to scQC Agent! This document provides guidelines and information for contributors.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/scqc-agent.git
   cd scqc-agent
   ```

2. **Set up the development environment**:
   ```bash
   python -m venv scQC
   source scQC/bin/activate  # On Windows: scQC\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify the setup**:
   ```bash
   make test
   scqc --help
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines below

3. **Run tests and checks**:
   ```bash
   make all  # Runs linting and tests
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of your changes"
   ```

5. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Guidelines

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (handled by Black)
- **Import organization**: Use `ruff` for import sorting
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings

### Code Quality Tools

The project uses several tools to maintain code quality:

- **ruff**: Linting and import sorting
- **black**: Code formatting
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality

Run these tools with:

```bash
make fmt    # Format code
make lint   # Run linting
make test   # Run tests
```

### Example Code Style

```python
from typing import Optional, Dict, Any
from pathlib import Path

from pydantic import BaseModel

from ..state import ToolResult


def compute_qc_metrics(
    adata_path: str,
    species: str = "human",
    config: Optional[Dict[str, Any]] = None
) -> ToolResult:
    """Compute quality control metrics for single-cell data.
    
    Args:
        adata_path: Path to AnnData file
        species: Species for mitochondrial gene detection
        config: Additional configuration parameters
        
    Returns:
        ToolResult with QC metrics and generated artifacts
        
    Raises:
        FileNotFoundError: If adata_path does not exist
        ValueError: If species is not supported
    """
    if config is None:
        config = {}
    
    # Implementation here...
    
    return ToolResult(
        message="QC metrics computed successfully",
        state_delta={"qc_computed": True},
        artifacts=[Path("qc_metrics.csv")],
        citations=["Luecken & Theis (2019) Nat Methods"]
    )
```

## Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory
- Use descriptive test names: `test_functionality_description`
- Group related tests in classes
- Use fixtures for common setup

### Test Categories

1. **Unit tests**: Test individual functions and classes
2. **Integration tests**: Test component interactions
3. **CLI tests**: Test command-line interface
4. **Phase tests**: Test complete workflow phases

### Writing Tests

```python
import pytest
from pathlib import Path
from scqc_agent.tools.qc import compute_qc_metrics


class TestQCMetrics:
    """Test QC metrics computation."""
    
    def test_compute_qc_metrics_basic(self, sample_adata: Path) -> None:
        """Test basic QC metrics computation."""
        result = compute_qc_metrics(str(sample_adata))
        
        assert "QC metrics computed" in result.message
        assert result.state_delta["qc_computed"] is True
        assert len(result.artifacts) > 0
    
    def test_compute_qc_metrics_invalid_path(self) -> None:
        """Test QC metrics with invalid file path."""
        with pytest.raises(FileNotFoundError):
            compute_qc_metrics("/nonexistent/path.h5ad")
```

### Test Data

- Use synthetic data for unit tests
- Keep test data small (< 1MB)
- Place test data in `tests/data/`
- Use fixtures to create temporary data

## Documentation

### Docstrings

Use Google-style docstrings for all public functions:

```python
def load_anndata(path: str, backed: bool = False) -> ToolResult:
    """Load AnnData file from disk.
    
    Args:
        path: Path to the AnnData file (.h5ad)
        backed: Whether to load in backed mode for large files
        
    Returns:
        ToolResult containing load status and updated state
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid AnnData format
        
    Example:
        >>> result = load_anndata("data/pbmc3k.h5ad")
        >>> print(result.message)
        Successfully loaded AnnData with 2700 cells, 32738 genes
    """
```

### README Updates

When adding new features:

1. Update the appropriate section in README.md
2. Add examples if the feature is user-facing
3. Update the roadmap/status if completing a phase milestone

## Pull Request Process

### Before Submitting

1. **Run the full test suite**: `make all`
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** (if exists)

### PR Description Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as such)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainer(s)
3. **Testing** on different environments if needed
4. **Merge** after approval

## Project Phases

The project is developed in phases. Please align contributions with the current phase:

### Phase 0: Foundation (DONE)
- Core data models
- CLI interface
- Testing infrastructure

### Phase 1: Core QC (CURRENT) 
- Scanpy integration
- QC metrics and plotting
- Filtering operations

### Phase 2+: Advanced Features
- Graph analysis
- Model integration
- Natural language processing

## Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

- **Environment details** (Python version, OS, package versions)
- **Minimal reproduction example**
- **Expected vs actual behavior**
- **Error messages and tracebacks**

### Feature Requests

Use the feature request template and include:

- **Use case description**
- **Proposed solution**
- **Alternative approaches considered**
- **Phase alignment** (which phase this belongs to)

### Questions

For questions about usage:

1. Check the documentation first
2. Search existing issues
3. Use the discussion forum for general questions
4. Open an issue for specific problems

## Release Process

### Version Numbering

We use semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Publish to PyPI (maintainers only)

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Follow the golden rule

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions and ideas
- **Email**: Security issues or private matters

## Recognition

Contributors will be:

- Listed in the AUTHORS file
- Mentioned in release notes
- Recognized in the documentation

Thank you for contributing to scQC Agent! 
