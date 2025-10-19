# Multi-file Kidney Data Loader Integration Guide

## Overview

This document provides complete integration instructions for the `load_kidney_data` tool in scQC Agent. The tool loads kidney scRNA-seq datasets from three files:

1. **Raw 10X HDF5** (all droplets) - for ambient RNA correction with SCAR
2. **Filtered 10X HDF5** (cells only) - for primary analysis
3. **Metadata CSV** - sample annotations

## Files Created

### 1. Tool Implementation
**Location**: `/Users/michaelhaidar/Documents/Vanderbilt/Brain_Research/scAgent/scqc_agent/tools/multiload.py`

**Key Features**:
- Loads both raw and filtered 10X HDF5 matrices
- Merges metadata from CSV into filtered.obs
- Stores original counts in filtered.layers['counts_raw']
- Creates checkpoints for both raw (SCAR) and filtered (primary) data
- Generates summary statistics and data preview plots
- Full error handling with detailed error messages
- Comprehensive docstrings with examples

**State Management Pattern**:
```python
# CRITICAL: Call checkpoint() BEFORE add_artifact()
state.checkpoint(str(filtered_checkpoint_path), "multiload_filtered")
state.add_artifact(str(summary_path), "Load Summary (JSON)")
state.add_artifact(str(summary_csv_path), "Load Summary (CSV)")
```

### 2. Pydantic Schema
**Location**: `/Users/michaelhaidar/Documents/Vanderbilt/Brain_Research/scAgent/scqc_agent/agent/schemas.py`

**Schema Class**: `LoadKidneyDataInput`

**Validation Features**:
- File existence checks for all 3 files
- File extension validation (.h5/.hdf5 for matrices, .csv for metadata)
- Sample ID column validation (non-empty)
- Absolute path conversion for reproducibility

**Schema Fields**:
```python
class LoadKidneyDataInput(BaseModel):
    raw_h5_path: str  # Required
    filtered_h5_path: str  # Required
    meta_csv_path: str  # Required
    sample_id_column: str = "sample_ID"  # Default
    metadata_merge_column: Optional[str] = None  # Optional
    make_unique: bool = True  # Default
```

### 3. Test Suite
**Location**: `/Users/michaelhaidar/Documents/Vanderbilt/Brain_Research/scAgent/tests/test_multiload.py`

**Test Coverage**:
- Basic 3-file loading workflow
- Checkpoint creation in state history
- Artifact tracking in history
- Raw data checkpoint for SCAR
- Original counts storage in layers
- Metadata merge correctness
- Error handling (missing files, invalid formats, invalid columns)
- State management (all required fields present)
- Reproducibility (deterministic loading)
- Integration with downstream tools (QC, SCAR)
- Pydantic schema validation

**Run Tests**:
```bash
pytest tests/test_multiload.py -v
pytest tests/test_multiload.py::test_load_kidney_data_basic -v
pytest tests/test_multiload.py::test_load_creates_checkpoint -v
```

## Integration Steps

### Step 1: Register Tool in Runtime

**File**: `scqc_agent/agent/runtime.py`

**Location**: In `_init_tool_registry()` method (around line 330)

**Code to Add**:
```python
def _init_tool_registry(self) -> None:
    """Initialize registry of available tools."""
    self.tools = {
        "load_data": self._load_data_tool,
        "load_kidney_data": self._load_kidney_data_tool,  # ADD THIS LINE
        "compute_qc_metrics": self._compute_qc_tool,
        "plot_qc": self._plot_qc_tool,
        "apply_qc_filters": self._apply_qc_filters_tool,
        "quick_graph": self._quick_graph_tool,
        "graph_from_rep": self._graph_from_rep_tool,
        "run_scar": self._run_scar_tool,
        "run_scvi": self._run_scvi_tool,
        "detect_doublets": self._detect_doublets_tool,
        "apply_doublet_filter": self._apply_doublet_filter_tool,
        "final_graph": self._final_graph_tool,
        "batch_diagnostics": self._batch_diagnostics_tool,
        "detect_marker_genes": self._detect_marker_genes_tool,
        "annotate_clusters": self._annotate_clusters_tool,
        "compare_clusters": self._compare_clusters_tool,
    }
```

### Step 2: Add Tool Wrapper Method

**File**: `scqc_agent/agent/runtime.py`

**Location**: After other tool wrapper methods (search for `def _load_data_tool`)

**Code to Add**:
```python
def _load_kidney_data_tool(self, params: Dict[str, Any]) -> ToolResult:
    """Wrapper for load_kidney_data tool.

    Loads kidney scRNA-seq datasets from raw 10X H5, filtered 10X H5,
    and metadata CSV files.

    Args:
        params: Tool parameters from agent plan

    Returns:
        ToolResult with loaded data checkpoints and artifacts
    """
    from ..tools.multiload import load_kidney_data
    return load_kidney_data(self.state, **params)
```

### Step 3: Update Tool Imports (Optional)

If runtime.py imports tools at the top of the file, add:

```python
from ..tools.multiload import load_kidney_data
```

However, the current codebase uses lazy imports in wrapper methods, so this is **optional**.

### Step 4: Verify Schema Registration

**File**: `scqc_agent/agent/schemas.py`

**Verify these lines exist** (already added):
```python
# In TOOL_SCHEMAS dict (around line 616):
TOOL_SCHEMAS = {
    "load_data": LoadDataInput,
    "load_kidney_data": LoadKidneyDataInput,  # ✓ Already added
    # ... other tools
}

# In TOOL_DESCRIPTIONS dict (around line 698):
TOOL_DESCRIPTIONS = {
    "load_data": "Import AnnData files (.h5ad) into the session for analysis",
    "load_kidney_data": "Load kidney scRNA-seq datasets from raw 10X H5, filtered 10X H5, and metadata CSV files",  # ✓ Already added
    # ... other tools
}
```

## Usage Examples

### Python API

```python
from scqc_agent.agent.runtime import Agent

# Initialize agent
agent = Agent(state_path=".scqc_state.json")

# Plan mode - generate execution plan
result = agent.chat(
    "Load kidney dataset from raw.h5, filtered.h5, and metadata.csv",
    mode="plan"
)

# Review plan, then execute
result = agent.chat(
    "Load kidney dataset",
    mode="execute",
    plan_path=result["plan_path"]
)

# Access loaded data paths
print(f"Filtered data: {agent.state.adata_path}")
print(f"Raw data: {agent.state.metadata['raw_adata_path']}")
```

### Direct Tool Call

```python
from scqc_agent.state import SessionState
from scqc_agent.tools.multiload import load_kidney_data

# Create state
state = SessionState(run_id="kidney_analysis")

# Load data
result = load_kidney_data(
    state,
    raw_h5_path="data/kidney_raw.h5",
    filtered_h5_path="data/kidney_filtered.h5",
    meta_csv_path="data/kidney_metadata.csv",
    sample_id_column="sample_ID"
)

# Check result
if not result.message.startswith("Error"):
    print(f"Success: {result.message}")
    print(f"Artifacts: {result.artifacts}")

    # Update state
    state.adata_path = result.state_delta["adata_path"]
    state.update_metadata(result.state_delta)
    state.save(".scqc_state.json")
```

### CLI Integration (Future)

After adding CLI commands in `cli.py`:

```bash
# Initialize session
scqc init

# Load kidney data
scqc load-kidney \
  --raw data/kidney_raw.h5 \
  --filtered data/kidney_filtered.h5 \
  --metadata data/kidney_metadata.csv \
  --sample-id-column sample_ID

# View state
scqc state

# Continue with QC
scqc qc compute --species mouse
scqc qc plot --stage pre
```

## State Updates

The tool updates session state with these fields:

```python
state_delta = {
    "adata_path": str,              # Path to filtered data checkpoint
    "raw_adata_path": str,          # Path to raw data checkpoint (for SCAR)
    "n_cells_raw": int,             # Number of raw droplets
    "n_cells_filtered": int,        # Number of filtered cells
    "n_genes": int,                 # Number of genes
    "sample_id": str,               # Sample identifier
    "metadata_columns": List[str],  # List of metadata column names
    "cells_retained_fraction": float, # Fraction of cells retained
    "data_loaded": bool,            # True
    "load_method": str              # "multiload_kidney"
}
```

## Artifacts Generated

1. **load_summary.json** - Detailed summary with file paths, counts, metadata info
2. **load_summary.csv** - Simple CSV with key metrics
3. **load_preview.png** - Data preview plot (raw vs filtered counts, genes detected)
4. **adata_raw.h5ad** - Raw data checkpoint (for SCAR)
5. **adata_filtered.h5ad** - Filtered data checkpoint (primary)

## Workflow Integration

### Typical Pipeline

```python
# 1. Load kidney data (raw + filtered + metadata)
load_kidney_data(state, ...)

# 2. Run SCAR ambient RNA removal on raw data
# Tool automatically finds raw_adata_path from state
run_scar(state, batch_key="batch", epochs=100)

# 3. Compute QC metrics on filtered data
compute_qc_metrics(state, species="mouse")

# 4. Apply QC filters
apply_qc_filters(state, min_genes=200, max_pct_mt=10)

# 5. Continue with downstream analysis
quick_graph(state, n_neighbors=15, resolution=0.5)
```

### Integration with SCAR Tool

The SCAR tool can automatically use the raw data checkpoint:

```python
# In tools/scar.py, modify to check for raw_adata_path:
def run_scar(state, ...):
    # Check if raw data is available
    if "raw_adata_path" in state.metadata:
        raw_path = state.metadata["raw_adata_path"]
        if Path(raw_path).exists():
            adata_raw = sc.read_h5ad(raw_path)
            # Use raw data for ambient RNA modeling
```

## Testing Integration

### Run All Tests
```bash
# Full test suite
make test

# Multiload tests only
pytest tests/test_multiload.py -v

# Coverage report
pytest tests/test_multiload.py --cov=scqc_agent.tools.multiload --cov-report=term-missing
```

### Key Tests to Verify
1. **test_load_creates_checkpoint** - Ensures state.checkpoint() is called
2. **test_load_adds_artifacts_to_history** - Ensures artifacts are tracked
3. **test_integration_raw_data_for_scar** - Ensures SCAR compatibility
4. **test_metadata_merge** - Ensures metadata is correctly merged

## Troubleshooting

### Issue: Artifacts not showing in `scqc summary`

**Cause**: Tool is calling `state.add_artifact()` without first calling `state.checkpoint()`

**Solution**: Verify this pattern in multiload.py:
```python
state.checkpoint(str(checkpoint_path), "multiload_filtered")  # FIRST
state.add_artifact(str(artifact), "label")  # THEN
```

### Issue: "Unknown tool: load_kidney_data"

**Cause**: Tool not registered in runtime.py

**Solution**: Add to `_init_tool_registry()`:
```python
"load_kidney_data": self._load_kidney_data_tool,
```

### Issue: File validation errors

**Cause**: Pydantic schema validates files at plan generation time

**Solution**: Ensure files exist before calling tool, or handle validation errors:
```python
try:
    schema = LoadKidneyDataInput(**params)
except ValueError as e:
    print(f"Validation error: {e}")
```

### Issue: Raw data not found by SCAR

**Cause**: raw_adata_path not stored in state.metadata

**Solution**: After load_kidney_data, verify:
```python
assert "raw_adata_path" in state.metadata
assert Path(state.metadata["raw_adata_path"]).exists()
```

## Design Decisions

### Why Store Raw Data Separately?

Raw data contains all droplets (cells + empty droplets), which is required for:
- SCAR ambient RNA correction
- Doublet detection (some methods use raw data)
- Quality control validation

Filtered data is the primary analysis dataset.

### Why Use `layers['counts_raw']`?

Storing original counts in layers preserves them through normalization and transformation steps, allowing tools to access raw counts when needed (e.g., for differential expression).

### Why Pydantic Validation?

Pydantic schemas provide:
- Runtime type checking
- Automatic file path validation
- Self-documenting API with examples
- Integration with LangChain agent planning

### Why Checkpoint Before Artifacts?

The state management pattern requires:
1. `checkpoint()` creates a history entry
2. `add_artifact()` adds to the most recent history entry

Without checkpoint, artifacts are added to `state.artifacts` dict but not `state.history`, so they won't appear in `scqc summary`.

## Future Enhancements

1. **Multi-sample support**: Handle metadata with multiple samples, map cells to samples
2. **Custom merge logic**: Allow user-specified merge strategies for metadata
3. **Data validation**: Add QC checks for raw vs filtered consistency
4. **Compression support**: Handle .h5.gz compressed files
5. **CLI commands**: Add `scqc load-kidney` command to cli.py
6. **Progress tracking**: Add progress bars for large file loading
7. **Batch loading**: Support loading multiple samples in one call

## References

- CLAUDE.md lines 250-280: State management pattern
- CLAUDE.md lines 282-356: Adding new tools guide
- scqc_agent/tools/qc.py: Example tool with checkpoint/artifact pattern
- scqc_agent/tools/scar.py: Example of using state metadata for file paths
