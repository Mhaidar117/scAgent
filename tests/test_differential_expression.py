"""Tests for differential expression analysis functionality."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.differential_expression import compare_clusters
from scqc_agent.agent.runtime import Agent

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def synthetic_data_with_clusters():
    """Create synthetic data with clear cluster-specific signatures."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")

    np.random.seed(42)
    n_obs, n_vars = 600, 1000

    # Create 3 clusters with different expression patterns
    cells_per_cluster = n_obs // 3

    X = []

    # Cluster 0: High expression of first 100 genes
    cluster0_expr = np.random.negative_binomial(5, 0.3, size=(cells_per_cluster, n_vars)).astype(np.float32)
    cluster0_expr[:, :100] *= 5  # Boost first 100 genes
    X.append(cluster0_expr)

    # Cluster 1: High expression of second 100 genes (100-199)
    cluster1_expr = np.random.negative_binomial(5, 0.3, size=(cells_per_cluster, n_vars)).astype(np.float32)
    cluster1_expr[:, 100:200] *= 5  # Boost genes 100-199
    X.append(cluster1_expr)

    # Cluster 2: High expression of third 100 genes (200-299)
    cluster2_expr = np.random.negative_binomial(5, 0.3, size=(n_obs - 2*cells_per_cluster, n_vars)).astype(np.float32)
    cluster2_expr[:, 200:300] *= 5  # Boost genes 200-299
    X.append(cluster2_expr)

    X = np.vstack(X)

    # Create AnnData
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Add cluster labels
    cluster_labels = np.repeat([0, 1, 2], [cells_per_cluster, cells_per_cluster, n_obs - 2*cells_per_cluster])
    adata.obs['leiden'] = pd.Categorical(cluster_labels.astype(str))

    # Add QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_de")

    yield state, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_basic(synthetic_data_with_clusters, temp_state):
    """Test basic cluster comparison."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run comparison
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="0",
        group2="1",
        method="wilcoxon",
        n_genes=50
    )

    # Check result structure
    assert result.message is not None
    assert not result.message.startswith("❌"), f"Comparison failed: {result.message}"
    assert result.state_delta is not None
    assert len(result.artifacts) >= 1

    # Check state updates
    assert "de_group1" in result.state_delta
    assert "de_group2" in result.state_delta
    assert "de_n_significant" in result.state_delta
    assert result.state_delta["de_group1"] == "0"
    assert result.state_delta["de_group2"] == "1"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_multiple_groups(synthetic_data_with_clusters, temp_state):
    """Test comparison between multiple cluster groups."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run comparison with multiple clusters per group
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1=["0", "1"],
        group2="2",
        method="wilcoxon"
    )

    assert not result.message.startswith("❌")
    assert result.state_delta["de_group1"] == "0+1"
    assert result.state_delta["de_group2"] == "2"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_vs_rest(synthetic_data_with_clusters, temp_state):
    """Test comparison of one cluster vs rest."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run comparison vs rest
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="0",
        group2="rest",
        method="wilcoxon"
    )

    assert not result.message.startswith("❌")
    assert result.state_delta["de_group1"] == "0"
    assert result.state_delta["de_group2"] == "rest"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_invalid_cluster(synthetic_data_with_clusters, temp_state):
    """Test comparison with non-existent cluster."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Try with non-existent cluster
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="99",
        group2="0"
    )

    # Should return error about empty group
    assert "empty" in result.message.lower() or "❌" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_invalid_cluster_key(synthetic_data_with_clusters, temp_state):
    """Test comparison with invalid cluster key."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Try with non-existent cluster key
    result = compare_clusters(
        state,
        cluster_key="nonexistent_clusters",
        group1="0",
        group2="1"
    )

    # Should return error
    assert "not found" in result.message or "❌" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_artifacts_generated(synthetic_data_with_clusters, temp_state):
    """Test that all expected artifacts are generated."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run comparison
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="0",
        group2="1",
        method="wilcoxon"
    )

    # Check that artifacts were created
    assert len(result.artifacts) >= 3, "Should have CSV, volcano plot, heatmap, and summary"

    # Check specific artifacts
    csv_artifacts = [art for art in result.artifacts if art.endswith(".csv")]
    assert len(csv_artifacts) >= 1, "Should have DE genes CSV"

    png_artifacts = [art for art in result.artifacts if art.endswith(".png")]
    assert len(png_artifacts) >= 2, "Should have volcano plot and heatmap"

    json_artifacts = [art for art in result.artifacts if art.endswith(".json")]
    assert len(json_artifacts) >= 1, "Should have summary JSON"

    # Check that files actually exist
    for artifact in result.artifacts:
        assert Path(artifact).exists(), f"Artifact file should exist: {artifact}"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_csv_format(synthetic_data_with_clusters, temp_state):
    """Test that DE genes CSV has correct format."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run comparison
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="0",
        group2="1",
        method="wilcoxon"
    )

    # Find CSV
    csv_artifacts = [art for art in result.artifacts if art.endswith(".csv")]
    assert len(csv_artifacts) >= 1

    # Check CSV content
    de_df = pd.read_csv(csv_artifacts[0])

    # Should have required columns
    assert "gene" in de_df.columns
    assert "score" in de_df.columns
    assert "rank" in de_df.columns
    assert "pval_adj" in de_df.columns  # Wilcoxon has pvals
    assert "logfoldchange" in de_df.columns  # Wilcoxon has logFC

    # Should have data
    assert len(de_df) > 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_different_methods(synthetic_data_with_clusters, temp_state):
    """Test that different statistical methods work."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Test each method
    for method in ["wilcoxon", "t-test", "logreg"]:
        result = compare_clusters(
            state,
            cluster_key="leiden",
            group1="0",
            group2="1",
            method=method
        )

        assert not result.message.startswith("❌"), f"{method} method failed: {result.message}"
        assert result.state_delta["de_method"] == method


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_agent_integration(synthetic_data_with_clusters, temp_state):
    """Test DE comparison via agent tool registry."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    # Create state file
    state_file = Path(temp_dir) / "test_state.json"

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Check that tool is registered
    assert "compare_clusters" in agent.tools

    # Execute via agent
    result = agent.tools["compare_clusters"]({
        "cluster_key": "leiden",
        "group1": "0",
        "group2": "1",
        "method": "wilcoxon"
    })

    assert not result.message.startswith("❌")
    assert "de_group1" in result.state_delta


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_compare_clusters_summary_json(synthetic_data_with_clusters, temp_state):
    """Test that summary JSON contains expected information."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_data_with_clusters.write(adata_path)

    state.adata_path = str(adata_path)

    # Run comparison
    result = compare_clusters(
        state,
        cluster_key="leiden",
        group1="0",
        group2="1",
        method="wilcoxon"
    )

    # Find summary JSON
    json_artifacts = [art for art in result.artifacts if art.endswith(".json")]
    assert len(json_artifacts) >= 1

    # Load summary
    import json
    with open(json_artifacts[0], 'r') as f:
        summary = json.load(f)

    # Check required fields
    assert "group1" in summary
    assert "group2" in summary
    assert "n_cells_group1" in summary
    assert "n_cells_group2" in summary
    assert "method" in summary
    assert "n_de_genes_tested" in summary
    assert "n_significant_genes" in summary

    # Check values
    assert summary["group1"] == "0"
    assert summary["group2"] == "1"
    assert summary["method"] == "wilcoxon"
    assert summary["n_cells_group1"] == 200
    assert summary["n_cells_group2"] == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
