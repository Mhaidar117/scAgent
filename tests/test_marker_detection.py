"""Tests for marker gene detection functionality."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.markers import detect_marker_genes
from scqc_agent.agent.runtime import Agent

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def synthetic_adata_with_clusters():
    """Create synthetic AnnData with clear cluster structure."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")

    np.random.seed(42)
    n_obs, n_vars = 300, 500
    n_clusters = 3

    # Create 3 distinct groups with different marker genes
    cells_per_cluster = n_obs // n_clusters
    X = []

    # Cluster 0: high expression in genes 0-100
    X0 = np.random.negative_binomial(2, 0.5, size=(cells_per_cluster, n_vars)).astype(np.float32)
    X0[:, :100] += np.random.negative_binomial(10, 0.3, size=(cells_per_cluster, 100))
    X.append(X0)

    # Cluster 1: high expression in genes 100-200
    X1 = np.random.negative_binomial(2, 0.5, size=(cells_per_cluster, n_vars)).astype(np.float32)
    X1[:, 100:200] += np.random.negative_binomial(10, 0.3, size=(cells_per_cluster, 100))
    X.append(X1)

    # Cluster 2: high expression in genes 200-300
    remaining = n_obs - 2 * cells_per_cluster
    X2 = np.random.negative_binomial(2, 0.5, size=(remaining, n_vars)).astype(np.float32)
    X2[:, 200:300] += np.random.negative_binomial(10, 0.3, size=(remaining, 100))
    X.append(X2)

    X = np.vstack(X)

    # Create AnnData
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Add cluster labels
    cluster_labels = np.repeat([0, 1, 2], [cells_per_cluster, cells_per_cluster, remaining])
    adata.obs['leiden'] = pd.Categorical(cluster_labels.astype(str))

    # Add QC metrics
    adata.var['mt'] = [i < 20 for i in range(n_vars)]
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_markers")

    yield state, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_detect_marker_genes_basic(synthetic_adata_with_clusters, temp_state):
    """Test basic marker gene detection."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    # Set up state
    state.adata_path = str(adata_path)

    # Run marker detection
    result = detect_marker_genes(
        state,
        cluster_key="leiden",
        method="wilcoxon",
        n_genes=25,
        species="human"
    )

    # Check result structure
    assert result.message is not None
    assert not result.message.startswith("❌"), f"Marker detection failed: {result.message}"
    assert result.state_delta is not None
    assert len(result.artifacts) >= 1  # At least CSV

    # Check state updates
    assert "n_clusters_with_markers" in result.state_delta
    assert result.state_delta["n_clusters_with_markers"] == 3
    assert "marker_detection_method" in result.state_delta
    assert result.state_delta["marker_detection_method"] == "wilcoxon"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_marker_genes_csv_format(synthetic_adata_with_clusters, temp_state):
    """Test that marker genes CSV has correct format."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run marker detection
    result = detect_marker_genes(state, cluster_key="leiden", n_genes=10)

    # Find CSV artifact
    csv_artifacts = [art for art in result.artifacts if art.endswith(".csv")]
    assert len(csv_artifacts) >= 1

    csv_path = csv_artifacts[0]

    # Verify CSV exists and can be read
    marker_df = pd.read_csv(csv_path)

    # Check core required columns (always present)
    required_cols = ['cluster', 'rank', 'gene', 'score']
    for col in required_cols:
        assert col in marker_df.columns, f"Missing column: {col}"

    # Check data validity
    assert len(marker_df) > 0
    assert marker_df['cluster'].nunique() == 3  # 3 clusters
    assert marker_df['rank'].max() <= 10  # n_genes=10


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_marker_genes_different_methods(synthetic_adata_with_clusters, temp_state):
    """Test marker detection with different statistical methods."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    methods = ["wilcoxon", "t-test", "logreg"]

    for method in methods:
        # Reset state for each method
        state.history = []
        state.artifacts = {}

        result = detect_marker_genes(
            state,
            cluster_key="leiden",
            method=method,
            n_genes=10
        )

        assert not result.message.startswith("❌"), f"Method {method} failed: {result.message}"
        assert result.state_delta["marker_detection_method"] == method


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_marker_genes_invalid_cluster_key(synthetic_adata_with_clusters, temp_state):
    """Test marker detection with invalid cluster key."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Use non-existent cluster key
    result = detect_marker_genes(
        state,
        cluster_key="nonexistent_clusters"
    )

    # Should return error
    assert "not found" in result.message or "❌" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_marker_genes_top_markers_summary(synthetic_adata_with_clusters, temp_state):
    """Test that top markers summary is generated correctly."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run marker detection
    result = detect_marker_genes(state, cluster_key="leiden", n_genes=10)

    # Check that top_markers is in state_delta
    assert "top_markers" in result.state_delta
    top_markers = result.state_delta["top_markers"]

    # Should have entries for each cluster
    assert len(top_markers) == 3

    # Each cluster should have top genes
    for cluster_id in ["0", "1", "2"]:
        assert cluster_id in top_markers
        assert isinstance(top_markers[cluster_id], list)
        assert len(top_markers[cluster_id]) <= 5  # Top 5 per cluster


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_marker_detection_agent_integration(synthetic_adata_with_clusters, temp_state):
    """Test marker detection via agent tool registry."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    # Create state file
    state_file = Path(temp_dir) / "test_state.json"

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Check that tool is registered
    assert "detect_marker_genes" in agent.tools

    # Execute via agent
    result = agent.tools["detect_marker_genes"]({
        "cluster_key": "leiden",
        "method": "wilcoxon",
        "n_genes": 15
    })

    assert not result.message.startswith("❌")
    assert "n_clusters_with_markers" in result.state_delta


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_marker_detection_species_awareness(synthetic_adata_with_clusters, temp_state):
    """Test that marker detection respects species parameter."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Test with explicit species
    result = detect_marker_genes(
        state,
        cluster_key="leiden",
        species="mouse"
    )

    assert not result.message.startswith("❌")
    assert result.state_delta.get("marker_species") == "mouse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
