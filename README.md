# scQC Agent

An open-source agent for reproducible scRNA-seq quality control workflows via natural language.

## Mission

The goal of scAgent is to streamline quality control (QC) for novice scRNA-seq researchers and reduce the friction of reproducing processing pipelines across labs. One of the biggest challenges in replicating published experiments is adapting QC pipelines to match another study’s workflow.

scAgent addresses this by:

- Providing state JSONs and automated HTML reports that make QC deterministic, transparent, and shareable
- Allowing researchers to reproduce experiments quickly and consistently by re-using shared states and reports
- Standardizing scRNA-seq QC pipelines so results are easier to validate, compare, and build upon

With scAgent, reproducing another study’s QC becomes a matter of re-running a saved state, enabling more reliable and collaborative research. This project is open to collaboration with the scRNA-seq community to further improve reproducibility and usability.

[![CI](https://github.com/your-org/scqc-agent/workflows/CI/badge.svg)](https://github.com/your-org/scqc-agent/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Why Contribute?

- Help make scRNA-seq research more reproducible across labs
- Build tools that benefit the single-cell community
- contribute to open science infastructure

## Ways to Contribute

- Report bugs or request features via GitHub Issues
- Improve documentation for accessibility and clarity
- Add new QC tools or improve exisiting steps
- Share QC state file from your work to build a community library
- Improve the retrieval quality by expanding the knowledge base with your own SOPs
- Contribute test datasets to improve validation and halde edge-case

## Overview

scQC Agent provides a natural language interface for executing single-cell RNA-seq quality control workflows. The agent orchestrates typed Python tools corresponding to standard QC pipeline steps and returns structured artifacts (plots, CSVs, .h5ad snapshots) with citations.


## Key Features

-  **Natural Language Interface**: Describe QC workflows in plain English
-  **Structured State Management**: Persistent session state with checkpoints and artifacts
-  **Typed Tool System**: Composable tools for each QC pipeline step
-  **Reproducible Outputs**: Deterministic file paths and comprehensive artifact tracking
-  **Developer-Friendly**: Full test suite, type checking, and modern Python tooling

## Installation

### Quick Start

```bash
# Install core package
pip install scqc-agent

# Install with development dependencies
pip install "scqc-agent[dev]"

# Install with QC dependencies (Phase 1+)
pip install "scqc-agent[qc]"

# Install for current working condition
pip install -e ".[dev,ui,agent,qc,models]"
```

### Development Installation

```bash
git clone https://github.com/your-org/scqc-agent.git
cd scqc-agent

# Create virtual environment
python -m venv scQC  # or python3.11 -m venv scQC
source scQC/bin/activate  # On Windows: scQC\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Quick Start

### CLI Interface

```bash
# Initialize a new session
scqc init

# Load your data file
scqc load --path data/pbmc3k.h5ad

# Generate a workflow plan
scqc chat "compute qc metrics, filter cells with min_genes=200, and create umap" --interactive

# Check session state
scqc state show
```

### Python API

```python
import scqc_agent
from scqc_agent import SessionState, Agent

# Create a session
state = SessionState(
    adata_path="data/pbmc3k.h5ad",
    config={"species": "human", "random_seed": 42}
)

# Initialize agent
agent = Agent("state.json")

# Process natural language requests
response = agent.handle_message("compute qc metrics and generate plots")
print(response["plan"])
```

### Jupyter Notebook

See `examples/quickstart.ipynb` for a complete walkthrough of the package functionality.

### Streamlit Web Interface

scQC Agent provides an interactive web interface built with Streamlit for visual workflow management and real-time monitoring.

#### Launching the App

```bash
# Activate your virtual environment
source scQC/bin/activate  # On Windows: scQC\Scripts\activate

# Launch the Streamlit app
python -m streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`.

#### Key Features

**Session Management**
- Create new analysis sessions or switch between existing ones
- Sessions are stored in `streamlit_sessions/` directory
- Each session maintains separate workflow state, artifacts, and history
- Dropdown selector shows all sessions with metadata (cell count, last modified date)

**Data Upload Options**

The app supports two upload modes:

1. **Multi-file Upload (Kidney Workflow)**
   - Upload raw droplet matrix (.h5)
   - Upload filtered cell matrix (.h5)
   - Upload metadata file (.csv or .xlsx) - optional
   - Designed for complete kidney scRNA-seq analysis workflow

2. **Single File Upload**
   - Upload pre-processed AnnData files (.h5ad or .h5ad.gz)
   - Quick start for datasets already in AnnData format

**Plan and Execute Workflow**

The app follows a two-phase workflow pattern:

1. **Planning Phase**: Submit a natural language query (e.g., "Compute QC metrics for my mouse kidney data")
   - Agent generates an execution plan
   - Review proposed steps and parameters
   - Plan is displayed with tool names and arguments

2. **Execution Phase**: Review and approve the plan
   - Click "Execute Plan" to run the workflow
   - Monitor real-time progress and tool outputs
   - View generated artifacts immediately

**Real-time Monitoring**
- Execution history with timestamps
- Artifact catalog with inline image previews
- Workflow history showing all completed steps
- Validation results and error messages
- Tool execution summaries with citations

**Provider Selection**

Switch between language model backends directly in the sidebar:
- **Local Ollama**: Free, runs locally, no API key needed
- **OpenAI API**: Cloud-based, requires API key, faster responses

The active provider is displayed in both the sidebar and main header.

#### Example Streamlit Workflow

```bash
# 1. Start the app
python -m streamlit run streamlit_app.py

# 2. In the browser:
#    - Create new session or select existing one
#    - Upload your data files
#    - Enter query: "Compute QC metrics and generate violin plots"
#    - Review the generated plan
#    - Click "Execute Plan"
#    - View artifacts in the catalog

# 3. Continue analysis:
#    - "Filter cells with min_genes=500 and max_pct_mt=10"
#    - "Run PCA, build neighbor graph, and create UMAP"
#    - "Detect doublets using DoubletFinder"
```

#### Artifact Visualization

The app automatically displays generated artifacts:
- **Images** (PNG, JPG): Inline preview with zoom
- **Data files** (CSV, JSON): Download button
- **Checkpoints** (.h5ad): Path and metadata display

All artifacts are preserved throughout the workflow, allowing you to track analysis progression.

#### Session Persistence

Sessions are automatically saved and can be resumed later:
- State files stored in `streamlit_sessions/`
- Complete workflow history preserved
- Artifacts remain accessible across sessions
- Session metadata includes creation date, last modified, and cell counts

## Language Model Providers

scAgent supports both local Ollama models and the OpenAI Chat API. Ollama remains the default provider so existing workflows keep running without changes.

### Configuration

- `LLM_PROVIDER`: Set to `ollama` (default) or `openai` to choose the backend.
- `OPENAI_API_KEY`: Required when `LLM_PROVIDER=openai`; requests fail without it.
- `OPENAI_MODEL`: Optional override for the OpenAI chat model (defaults to `gpt-4o-mini`).
- `OPENAI_API_BASE`: Optional base URL for Azure OpenAI or proxy deployments.

Install the optional dependency with `pip install langchain-openai` if you plan to use the OpenAI provider.

### Staying on Ollama

No extra configuration is required to continue using Ollama. Ensure the local Ollama service is running and leave `LLM_PROVIDER` unset (or set it to `ollama`).

### OpenAI Usage Notes

- Set `OPENAI_API_KEY` before starting the CLI or Streamlit app (for example: `export OPENAI_API_KEY=sk-...`).
- OpenAI usage is billed per request and subject to rate limits. Monitor your account quota when running large analyses.

## Architecture

### Core Components

- **SessionState**: Manages workflow state, checkpoints, and artifacts
- **ToolResult**: Standardized output from tool executions
- **Agent**: Handles natural language messages and orchestrates tool calls
- **Tools**: Modular functions for each QC pipeline step

### Workflow Structure

```
runs/YYYYmmdd_HHMMSS/
├── state.json              # Session state
├── logs/                   # Execution logs  
├── step_00_load/           # Data loading
├── step_01_qc_compute/     # QC metrics
├── step_02_qc_plot/        # QC visualizations
├── step_03_qc_filter/      # Apply filters
└── ...
```

## Development

### Setup

```bash
make dev-setup     # Complete development environment setup
make test          # Run tests
make lint          # Run linting
make fmt           # Format code
make all           # Run linting and tests
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scqc_agent

# Run specific test file
pytest tests/test_state.py -v
```

### Code Quality

This project uses:

- **ruff**: Fast Python linting and formatting
- **black**: Code formatting
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality
- **pytest**: Testing framework with coverage

## Roadmap

### Foundation 

- [x] Core data models (SessionState, ToolResult)
- [x] CLI interface with basic commands
- [x] Agent runtime skeleton
- [x] Tool system architecture
- [x] Test suite and CI/CD
- [x] Developer tooling

### Core QC Loop 

- [x] Scanpy integration for QC metrics
- [x] QC plotting (violin plots, scatter plots)
- [x] QC filtering with thresholds
- [x] Real AnnData I/O operations

### Graph Analysis 

- [X] PCA, neighbors, UMAP computation
- [ ] Leiden clustering
- [X] Quick graph generation tools

### Advanced Methods 

- [x] scAR denoising (optional)
- [ ] scVI latent representation
- [x] Doublet detection and removal

### LangChain Integration 

- [x] Natural language processing
- [x] RAG knowledge base
- [x] Tool selection and parameter extraction

### Reporting & Polish 

- [x] HTML/PDF report generation
- [ ] Enhanced visualizations
- [ ] Performance telemetry

## Validation & Testing

scQC Agent includes comprehensive end-to-end testing to ensure the LangChain agent and all functionality work correctly.

### Quick Validation

Test that everything works with the provided smoke test:

```bash
# Run full end-to-end smoke test
make e2e

# Test synthetic data generation
make test-synth

# Test knowledge base retrieval
make test-kb
```

### Test Suite Components

#### 1. Synthetic Data Generator
- **Location**: `scqc_agent/tests/synth.py`
- **Purpose**: Creates realistic scRNA-seq data for testing without external downloads
- **Features**: Configurable cell counts, gene counts, mitochondrial genes, batch effects

```python
from scqc_agent.tests.synth import make_synth_adata

# Generate test data
adata = make_synth_adata(n_cells=600, n_genes=1500, n_batches=2)
print(f"Generated {adata.n_obs} cells × {adata.n_vars} genes")
```

#### 2. End-to-End Agent Tests
- **Location**: `tests/test_e2e_agent.py`
- **Purpose**: Test complete workflows using Python API
- **Coverage**: Data loading, QC computation, filtering, graph analysis, citations

```bash
# Run Python API tests
pytest tests/test_e2e_agent.py -v
```

#### 3. Knowledge Base Tests
- **Location**: `tests/test_kb_retriever.py`
- **Purpose**: Verify hybrid retrieval and citation functionality
- **Coverage**: BM25 + vector search, mitochondrial queries, doublet detection

```bash
# Test knowledge base functionality
pytest tests/test_kb_retriever.py -v
```

#### 4. CLI Smoke Test
- **Location**: `scripts/e2e_smoke.sh`
- **Purpose**: Test CLI interface end-to-end
- **Runtime**: < 90 seconds
- **Coverage**: All major CLI commands, artifact generation, state management

```bash
# Run CLI smoke test
bash scripts/e2e_smoke.sh

# Or via Makefile
make e2e
```

#### 5. Doublet Detection Stubs
- **Location**: `tests/test_doublets_stub.py`
- **Purpose**: Fast doublet testing for CI environments
- **Features**: Mock Scrublet implementation, performance testing

### Expected Artifacts

The test suite verifies creation of key artifacts:

- **QC Plots**: `qc_pre_violins.png`, `qc_post_violins.png`
- **Data Summaries**: `qc_summary.csv`, `qc_filters.json`
- **Graph Analysis**: `umap_pre.png`, `cluster_counts.csv`
- **Data Checkpoints**: `adata_step*.h5ad` files
- **Session State**: `.scqc_state.json`

### Performance Requirements

- **Smoke test runtime**: < 90 seconds
- **Individual query time**: < 5 seconds  
- **Agent initialization**: < 30 seconds
- **Deterministic results**: Same seed → same outputs

### Test Commands

```bash
# Full test suite
make all                    # lint + unit tests
make e2e-full              # complete e2e suite

# Individual test components  
make e2e                   # CLI smoke test
make e2e-pytest          # Python API tests
make test-kb              # knowledge base tests
make test-synth           # synthetic data test

# Development testing
make quick-test           # fast unit tests
make test-cov            # with coverage report
```

### CI Integration

GitHub Actions workflow includes:
- Matrix testing (Python 3.10, 3.11)
- Full end-to-end validation
- Artifact upload on failure
- Performance benchmarking

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `make all`
5. Validate end-to-end: `make e2e-full`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Testing Guidelines

- Add unit tests for new tools in `tests/`
- Update synthetic data if changing QC requirements
- Test both Python API and CLI interfaces
- Ensure deterministic behavior (use seeds)
- Verify artifact generation and formats
- Test error handling and edge cases

## License

This project is licensed under the [LICENSE](LICENSE) file for details.

## Citation

If you use scQC Agent in your research, please cite:

```bibtex
@software{scqc_agent,
  title={scQC Agent: Natural Language Interface for scRNA-seq Quality Control},
  author={scQC Agent Team},
  year={2025},
  url={https://github.com/your-org/scqc-agent}
}
```

## Acknowledgments

- Built on the excellent [Scanpy](https://scanpy.readthedocs.io/) ecosystem
- Inspired by QC best practices from [Luecken & Theis (2019)](https://doi.org/10.15252/msb.20188746)
- Powered by [LangChain](https://langchain.com/) for natural language processing

---

**Status**: Graph Analysis under development 9/28/25
