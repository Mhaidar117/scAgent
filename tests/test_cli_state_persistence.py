"""Test that CLI commands properly persist state history."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scqc_agent.state import SessionState

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def temp_state_with_data():
    """Create temporary state with synthetic data file."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")

    temp_dir = tempfile.mkdtemp()

    # Create synthetic data
    np.random.seed(42)
    n_obs, n_vars = 300, 500
    X = np.random.negative_binomial(2, 0.5, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Add some basic QC
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Normalize for clustering
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Save data
    adata_path = Path(temp_dir) / "test.h5ad"
    adata.write(adata_path)

    # Create state
    state = SessionState(run_id="test_cli")
    state.adata_path = str(adata_path)

    state_file = Path(temp_dir) / ".scqc_state.json"
    state.save(str(state_file))

    yield state_file, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_quick_graph_persists_history(temp_state_with_data):
    """Test that quick_graph adds a history entry that persists."""
    state_file, temp_dir = temp_state_with_data

    # Load state
    state = SessionState.load(str(state_file))

    # Verify no history initially
    assert len(state.history) == 0, "Should start with empty history"

    # Run quick_graph tool
    from scqc_agent.tools.graph import quick_graph
    result = quick_graph(state, seed=0, resolution=1.0, n_neighbors=15, n_pcs=50)

    # Verify tool ran successfully
    assert not result.message.startswith("❌"), f"Tool failed: {result.message}"

    # Check that history was updated on the state object
    assert len(state.history) > 0, "Tool should have created history entry"
    assert state.history[0]["label"] == "quick_graph"

    # Save state (simulating what CLI does)
    state.save(str(state_file))

    # Load state again (fresh load)
    state_reloaded = SessionState.load(str(state_file))

    # Verify history persisted
    assert len(state_reloaded.history) > 0, "History should persist after save/load"
    assert state_reloaded.history[0]["label"] == "quick_graph"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_detect_doublets_persists_history(temp_state_with_data):
    """Test that detect_doublets adds a history entry that persists."""
    state_file, temp_dir = temp_state_with_data

    # Load state
    state = SessionState.load(str(state_file))

    # Verify no history initially
    assert len(state.history) == 0

    # Run detect_doublets tool
    from scqc_agent.tools.doublets import detect_doublets
    result = detect_doublets(state, method="scrublet", expected_rate=0.06)

    # Verify tool ran successfully
    assert not result.message.startswith("❌"), f"Tool failed: {result.message}"

    # Check that history was updated
    assert len(state.history) > 0
    assert state.history[0]["label"] == "doublets_detected"

    # Apply state_delta (simulating what CLI does)
    if result.state_delta:
        if "adata_path" in result.state_delta:
            state.adata_path = result.state_delta["adata_path"]
        state.update_metadata(result.state_delta)

    # Save state
    state.save(str(state_file))

    # Load state again
    state_reloaded = SessionState.load(str(state_file))

    # Verify history persisted
    assert len(state_reloaded.history) > 0
    assert state_reloaded.history[0]["label"] == "doublets_detected"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_apply_doublet_filter_persists_history(temp_state_with_data):
    """Test that apply_doublet_filter adds a history entry that persists."""
    state_file, temp_dir = temp_state_with_data

    # Load state
    state = SessionState.load(str(state_file))

    # First detect doublets
    from scqc_agent.tools.doublets import detect_doublets, apply_doublet_filter
    detect_result = detect_doublets(state, method="scrublet", expected_rate=0.06)

    assert not detect_result.message.startswith("❌")
    assert len(state.history) == 1

    # Apply state_delta from detection
    if detect_result.state_delta:
        if "adata_path" in detect_result.state_delta:
            state.adata_path = detect_result.state_delta["adata_path"]
        state.update_metadata(detect_result.state_delta)

    # Save after detection
    state.save(str(state_file))

    # Now apply doublet filter
    filter_result = apply_doublet_filter(state, threshold=None)

    assert not filter_result.message.startswith("❌")
    assert len(state.history) == 2, "Should have 2 history entries"
    assert state.history[1]["label"] == "doublets_filtered"

    # Apply state_delta from filtering
    if filter_result.state_delta:
        if "adata_path" in filter_result.state_delta:
            state.adata_path = filter_result.state_delta["adata_path"]
        state.update_metadata(filter_result.state_delta)

    # Save state
    state.save(str(state_file))

    # Load state again
    state_reloaded = SessionState.load(str(state_file))

    # Verify both history entries persisted
    assert len(state_reloaded.history) == 2
    assert state_reloaded.history[0]["label"] == "doublets_detected"
    assert state_reloaded.history[1]["label"] == "doublets_filtered"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_multiple_tools_sequential_history(temp_state_with_data):
    """Test that running multiple tools sequentially builds up history."""
    state_file, temp_dir = temp_state_with_data

    # Load state
    state = SessionState.load(str(state_file))

    # Run quick_graph
    from scqc_agent.tools.graph import quick_graph
    from scqc_agent.tools.doublets import detect_doublets

    result1 = quick_graph(state, seed=0, resolution=1.0, n_neighbors=15, n_pcs=50)
    assert not result1.message.startswith("❌")
    assert len(state.history) == 1

    # Save after first tool
    state.save(str(state_file))

    # Load fresh and run second tool
    state = SessionState.load(str(state_file))
    result2 = detect_doublets(state, method="scrublet", expected_rate=0.06)
    assert not result2.message.startswith("❌")
    assert len(state.history) == 2

    # Apply state_delta
    if result2.state_delta:
        if "adata_path" in result2.state_delta:
            state.adata_path = result2.state_delta["adata_path"]
        state.update_metadata(result2.state_delta)

    # Save after second tool
    state.save(str(state_file))

    # Load fresh and verify both entries
    state_final = SessionState.load(str(state_file))
    assert len(state_final.history) == 2
    assert state_final.history[0]["label"] == "quick_graph"
    assert state_final.history[1]["label"] == "doublets_detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
