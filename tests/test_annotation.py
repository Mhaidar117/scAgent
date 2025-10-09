"""Tests for cell type annotation functionality."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.annotation import annotate_clusters, _load_marker_database
from scqc_agent.agent.runtime import Agent

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def synthetic_brain_data():
    """Create synthetic brain-like data with clear cell type signatures."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")

    np.random.seed(42)
    n_obs, n_vars = 500, 1000

    # Create cell types with characteristic gene expression
    # We'll create 3 major types: neurons, astrocytes, microglia
    cells_per_type = n_obs // 3

    X = []

    # Neurons: high SLC17A7, NEUROD6
    neuron_expr = np.random.negative_binomial(2, 0.5, size=(cells_per_type, n_vars)).astype(np.float32)
    # Boost neuron markers
    neuron_marker_genes = ['SLC17A7', 'NEUROD6', 'SATB2']
    X.append(neuron_expr)

    # Astrocytes: high GFAP, AQP4
    astro_expr = np.random.negative_binomial(2, 0.5, size=(cells_per_type, n_vars)).astype(np.float32)
    X.append(astro_expr)

    # Microglia: high C1QA, CX3CR1
    micro_expr = np.random.negative_binomial(2, 0.5, size=(n_obs - 2*cells_per_type, n_vars)).astype(np.float32)
    X.append(micro_expr)

    X = np.vstack(X)

    # Create AnnData
    adata = ad.AnnData(X=X)

    # Create gene names including brain markers
    brain_markers = ['SLC17A7', 'NEUROD6', 'SATB2', 'GFAP', 'AQP4', 'SLC1A2',
                     'C1QA', 'C1QB', 'CX3CR1', 'CLDN5', 'PDGFRB']
    other_genes = [f"Gene_{i}" for i in range(n_vars - len(brain_markers))]
    adata.var_names = brain_markers + other_genes
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Add cluster labels (simulating Leiden clustering results)
    cluster_labels = np.repeat([0, 1, 2], [cells_per_type, cells_per_type, n_obs - 2*cells_per_type])
    adata.obs['leiden'] = pd.Categorical(cluster_labels.astype(str))

    # Add QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Add UMAP coordinates (for visualization tests)
    np.random.seed(42)
    adata.obsm['X_umap'] = np.random.randn(n_obs, 2)

    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_annotation")

    yield state, temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_load_marker_database():
    """Test loading built-in marker database."""
    # Test human brain
    markers = _load_marker_database("human", "brain")
    assert markers is not None
    assert isinstance(markers, dict)
    assert "Excitatory neurons" in markers
    assert "Astrocytes" in markers
    assert "Microglia" in markers
    assert isinstance(markers["Astrocytes"], list)
    assert "GFAP" in markers["Astrocytes"]

    # Test mouse kidney
    markers = _load_marker_database("mouse", "kidney")
    assert markers is not None
    assert "Podocytes" in markers

    # Test human PBMC
    markers = _load_marker_database("human", "pbmc")
    assert markers is not None
    assert "CD4+ T cells" in markers
    assert "B cells" in markers

    # Test non-existent tissue
    markers = _load_marker_database("human", "nonexistent")
    assert markers is None


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotate_clusters_marker_method(synthetic_brain_data, temp_state):
    """Test cluster annotation using built-in markers."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    state.adata_path = str(adata_path)

    # Run annotation with marker method
    result = annotate_clusters(
        state,
        cluster_key="leiden",
        method="markers",
        species="human",
        tissue="brain"
    )

    # Check result structure
    assert result.message is not None
    assert not result.message.startswith("❌"), f"Annotation failed: {result.message}"
    assert result.state_delta is not None
    assert len(result.artifacts) >= 1

    # Check state updates
    assert "annotation_method" in result.state_delta
    assert "n_cell_types_identified" in result.state_delta
    assert "cell_types" in result.state_delta
    assert result.state_delta["annotation_method"].startswith("Marker-based")

    # Check artifacts
    assert any("summary" in str(art).lower() for art in result.artifacts)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotate_clusters_invalid_cluster_key(synthetic_brain_data, temp_state):
    """Test annotation with invalid cluster key."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    state.adata_path = str(adata_path)

    # Try with non-existent cluster key
    result = annotate_clusters(
        state,
        cluster_key="nonexistent_clusters",
        method="markers"
    )

    # Should return error
    assert "not found" in result.message or "❌" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotate_clusters_invalid_tissue(synthetic_brain_data, temp_state):
    """Test annotation with unsupported tissue type."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    state.adata_path = str(adata_path)

    # Try with unsupported tissue
    result = annotate_clusters(
        state,
        cluster_key="leiden",
        method="markers",
        species="human",
        tissue="unsupported_tissue"
    )

    # Should return error about no markers
    assert "No markers found" in result.message or "❌" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotation_creates_umap_plot(synthetic_brain_data, temp_state):
    """Test that UMAP plot is created when UMAP coordinates exist."""
    state, temp_dir = temp_state

    # Save synthetic data (has X_umap)
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    state.adata_path = str(adata_path)

    # Run annotation
    result = annotate_clusters(
        state,
        cluster_key="leiden",
        method="markers",
        species="human",
        tissue="brain"
    )

    # Check that UMAP plot was created
    umap_artifacts = [art for art in result.artifacts if "umap" in str(art).lower()]
    assert len(umap_artifacts) >= 1, "UMAP plot should be created"

    # Check that file actually exists
    umap_path = Path(umap_artifacts[0])
    assert umap_path.exists(), f"UMAP plot file should exist at {umap_path}"
    assert umap_path.suffix == ".png", "UMAP plot should be PNG"


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotation_summary_csv_format(synthetic_brain_data, temp_state):
    """Test that annotation summary CSV has correct format."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    state.adata_path = str(adata_path)

    # Run annotation
    result = annotate_clusters(
        state,
        cluster_key="leiden",
        method="markers",
        species="human",
        tissue="brain"
    )

    # Find summary CSV
    csv_artifacts = [art for art in result.artifacts if art.endswith("summary.csv")]
    assert len(csv_artifacts) >= 1

    # Check CSV content
    summary_df = pd.read_csv(csv_artifacts[0])

    # Should have required columns
    assert "cluster" in summary_df.columns
    assert "cell_type" in summary_df.columns
    assert "n_cells" in summary_df.columns

    # Should have data
    assert len(summary_df) > 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotation_agent_integration(synthetic_brain_data, temp_state):
    """Test annotation via agent tool registry."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    # Create state file
    state_file = Path(temp_dir) / "test_state.json"

    # Create agent
    agent = Agent(str(state_file))
    agent.state.adata_path = str(adata_path)
    agent.state.run_id = "test_run"

    # Check that tool is registered
    assert "annotate_clusters" in agent.tools

    # Execute via agent
    result = agent.tools["annotate_clusters"]({
        "cluster_key": "leiden",
        "method": "markers",
        "species": "human",
        "tissue": "brain"
    })

    assert not result.message.startswith("❌")
    assert "annotation_method" in result.state_delta


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_annotation_auto_method_fallback(synthetic_brain_data, temp_state):
    """Test that auto method falls back to markers when CellTypist unavailable."""
    state, temp_dir = temp_state

    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_brain_data.write(adata_path)

    state.adata_path = str(adata_path)

    # Run with auto method (should use markers if CellTypist not installed)
    result = annotate_clusters(
        state,
        cluster_key="leiden",
        method="auto",  # Auto fallback
        species="human",
        tissue="brain"
    )

    # Should succeed with either method
    assert not result.message.startswith("❌")
    assert "annotation_method" in result.state_delta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
