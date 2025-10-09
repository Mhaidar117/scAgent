"""Tests for state history tracking and summary display."""

import tempfile
import shutil
from pathlib import Path
import pytest

from scqc_agent.state import SessionState
from scqc_agent.agent.runtime import Agent

try:
    import scanpy as sc
    import anndata as ad
    import numpy as np
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def temp_state_file():
    """Create temporary state file."""
    temp_dir = tempfile.mkdtemp()
    state_file = Path(temp_dir) / "test_state.json"

    yield state_file

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_state_history_after_checkpoint(temp_state_file):
    """Test that checkpoint creates a history entry."""
    state = SessionState(run_id="test_checkpoint")

    # Create checkpoint
    checkpoint_path = state.checkpoint("data/test.h5ad", "initial_load")

    # Verify history entry was created
    assert len(state.history) == 1
    assert state.history[0]["label"] == "initial_load"
    assert state.history[0]["step"] == 0
    assert "timestamp" in state.history[0]
    assert "artifacts" in state.history[0]


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_state_history_with_artifacts(temp_state_file):
    """Test that artifacts are added to history entries."""
    state = SessionState(run_id="test_artifacts")

    # Create checkpoint
    state.checkpoint("data/test.h5ad", "qc_metrics")

    # Add artifacts
    state.add_artifact("plots/qc_violin.png", "QC violin plot")
    state.add_artifact("tables/qc_metrics.csv", "QC metrics table")

    # Verify artifacts in history
    assert len(state.history) == 1
    assert len(state.history[0]["artifacts"]) == 2

    artifact_paths = [a["path"] for a in state.history[0]["artifacts"]]
    assert "plots/qc_violin.png" in artifact_paths
    assert "tables/qc_metrics.csv" in artifact_paths


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_state_history_multiple_steps(temp_state_file):
    """Test that multiple tool executions create multiple history entries."""
    state = SessionState(run_id="test_multi_step")

    # Step 1: QC
    state.checkpoint("data/qc.h5ad", "qc_metrics")
    state.add_artifact("plots/qc.png", "QC plot")

    # Step 2: Filtering
    state.checkpoint("data/filtered.h5ad", "qc_filtering")
    state.add_artifact("plots/filter.png", "Filter plot")

    # Step 3: Clustering
    state.checkpoint("data/clustered.h5ad", "clustering")
    state.add_artifact("plots/umap.png", "UMAP plot")

    # Verify all steps in history
    assert len(state.history) == 3
    assert state.history[0]["label"] == "qc_metrics"
    assert state.history[1]["label"] == "qc_filtering"
    assert state.history[2]["label"] == "clustering"

    # Verify artifacts per step
    assert len(state.history[0]["artifacts"]) == 1
    assert len(state.history[1]["artifacts"]) == 1
    assert len(state.history[2]["artifacts"]) == 1


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_state_persistence_preserves_history(temp_state_file):
    """Test that saving and loading state preserves history."""
    # Create state with history
    state = SessionState(run_id="test_persistence")
    state.checkpoint("data/step1.h5ad", "step_one")
    state.add_artifact("plot1.png", "Plot 1")
    state.checkpoint("data/step2.h5ad", "step_two")
    state.add_artifact("plot2.png", "Plot 2")

    # Save state
    state.save(str(temp_state_file))

    # Load state
    loaded_state = SessionState.load(str(temp_state_file))

    # Verify history preserved
    assert len(loaded_state.history) == 2
    assert loaded_state.history[0]["label"] == "step_one"
    assert loaded_state.history[1]["label"] == "step_two"
    assert len(loaded_state.history[0]["artifacts"]) == 1
    assert len(loaded_state.history[1]["artifacts"]) == 1


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_artifact_without_checkpoint_warning(temp_state_file):
    """Test that adding artifact without checkpoint still works."""
    state = SessionState(run_id="test_no_checkpoint")

    # Add artifact without checkpoint - should not crash
    state.add_artifact("plot.png", "Orphan plot")

    # Artifact should be in artifacts dict
    assert "plot.png" in state.artifacts

    # But not in history since no checkpoint exists
    assert len(state.history) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_tools_create_proper_history_entries():
    """Test that actual tools create proper history entries."""
    from scqc_agent.tools.differential_expression import compare_clusters

    # Create synthetic data directly
    n_obs, n_vars = 300, 500
    X = np.random.negative_binomial(2, 0.5, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Add clusters
    import pandas as pd
    adata.obs['leiden'] = pd.Categorical(['0'] * 100 + ['1'] * 100 + ['2'] * 100)

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    adata_path = Path(temp_dir) / "test.h5ad"
    adata.write(adata_path)

    # Create state
    state = SessionState(run_id="test_tool_history")
    state.adata_path = str(adata_path)

    # Run tool
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="0",
        group2="1",
        method="wilcoxon"
    )

    # Verify checkpoint was created
    assert len(state.history) == 1, f"Expected 1 history entry, got {len(state.history)}"
    assert "differential_expression" in state.history[0]["label"]

    # Verify artifacts were added to history
    assert len(state.history[0]["artifacts"]) > 0, "No artifacts in history entry"

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
