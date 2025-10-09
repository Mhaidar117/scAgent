"""Tests for graph_from_rep agent integration."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from scqc_agent.state import SessionState
from scqc_agent.agent.runtime import Agent

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def synthetic_adata_with_pca():
    """Create synthetic AnnData with PCA for testing graph_from_rep."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")

    np.random.seed(42)
    n_obs, n_vars = 300, 1000

    # Create synthetic data
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Add QC metrics
    adata.var['mt'] = [i < 20 for i in range(n_vars)]
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Normalize and compute PCA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=30)

    return adata


@pytest.fixture
def temp_agent_setup():
    """Create temporary agent setup."""
    temp_dir = tempfile.mkdtemp()
    state_file = Path(temp_dir) / "test_state.json"

    yield temp_dir, state_file

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_in_tool_registry(temp_agent_setup):
    """Test that graph_from_rep is registered in agent tools."""
    temp_dir, state_file = temp_agent_setup

    agent = Agent(str(state_file))

    # Check that graph_from_rep is in the tool registry
    assert "graph_from_rep" in agent.tools
    assert callable(agent.tools["graph_from_rep"])


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_tool_execution(synthetic_adata_with_pca, temp_agent_setup):
    """Test that graph_from_rep tool can be executed via agent."""
    temp_dir, state_file = temp_agent_setup

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_pca.write(adata_path)

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Execute graph_from_rep tool directly
    result = agent.tools["graph_from_rep"]({
        "use_rep": "X_pca",
        "n_neighbors": 10,
        "resolution": 0.5,
        "seed": 42
    })

    # Verify result
    assert result.message is not None
    assert not result.message.startswith("❌"), f"Tool failed: {result.message}"
    assert result.state_delta is not None
    assert "n_clusters_pca" in result.state_delta
    assert result.state_delta["n_clusters_pca"] > 0
    assert len(result.artifacts) >= 2  # UMAP plot + cluster counts


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_plan_execution(synthetic_adata_with_pca, temp_agent_setup):
    """Test graph_from_rep in a complete plan execution."""
    temp_dir, state_file = temp_agent_setup

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_pca.write(adata_path)

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Create plan with graph_from_rep
    plan = [
        {
            "tool": "graph_from_rep",
            "description": "Generate graph from PCA representation",
            "params": {
                "use_rep": "X_pca",
                "n_neighbors": 10,
                "resolution": 0.5,
                "seed": 42
            }
        }
    ]

    # Execute plan
    results = agent._execute_plan(plan)

    # Verify execution
    assert len(results) == 1
    result = results[0]
    assert result.message is not None
    assert not result.message.startswith("❌")
    assert "n_clusters_pca" in agent.state.metadata


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_invalid_representation(synthetic_adata_with_pca, temp_agent_setup):
    """Test graph_from_rep with invalid representation."""
    temp_dir, state_file = temp_agent_setup

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_pca.write(adata_path)

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Execute with non-existent representation
    result = agent.tools["graph_from_rep"]({
        "use_rep": "X_nonexistent",
        "n_neighbors": 10,
        "resolution": 0.5,
        "seed": 42
    })

    # Should return error message
    assert "not found" in result.message or "❌" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_reproducibility(synthetic_adata_with_pca, temp_agent_setup):
    """Test that graph_from_rep produces reproducible results with same seed."""
    temp_dir, state_file = temp_agent_setup

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_pca.write(adata_path)

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Execute twice with same seed
    result1 = agent.tools["graph_from_rep"]({
        "use_rep": "X_pca",
        "seed": 42
    })

    # Reset state artifacts for second run
    agent.state.artifacts = {}
    agent.state.history = []

    result2 = agent.tools["graph_from_rep"]({
        "use_rep": "X_pca",
        "seed": 42
    })

    # Results should be identical
    assert result1.state_delta["n_clusters_pca"] == result2.state_delta["n_clusters_pca"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
