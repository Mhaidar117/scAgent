"""Tests for quick graph analysis functionality."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.graph import quick_graph

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData for testing."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")
    
    # Create synthetic data with clear structure
    np.random.seed(42)
    n_obs, n_vars = 500, 2000
    
    # Create 3 distinct groups
    group_size = n_obs // 3
    X = []
    
    # Group 1: higher expression in first 500 genes
    X1 = np.random.negative_binomial(5, 0.3, size=(group_size, n_vars))
    X1[:, :500] += np.random.negative_binomial(10, 0.2, size=(group_size, 500))
    X.append(X1)
    
    # Group 2: higher expression in middle 500 genes  
    X2 = np.random.negative_binomial(5, 0.3, size=(group_size, n_vars))
    X2[:, 500:1000] += np.random.negative_binomial(10, 0.2, size=(group_size, 500))
    X.append(X2)
    
    # Group 3: higher expression in last 500 genes
    remaining = n_obs - 2 * group_size
    X3 = np.random.negative_binomial(5, 0.3, size=(remaining, n_vars))
    X3[:, 1500:2000] += np.random.negative_binomial(10, 0.2, size=(remaining, 500))
    X.append(X3)
    
    X = np.vstack(X)
    
    # Create AnnData object
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add some mitochondrial genes for QC
    mt_genes = [f"MT-{i}" for i in range(20)]
    regular_genes = [f"Gene_{i}" for i in range(20, n_vars)]
    adata.var_names = mt_genes + regular_genes
    
    # Add basic QC metrics using scanpy functions
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    # Note: scanpy adds 'pct_counts_mt' automatically when mt column exists
    
    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_graph")
    
    yield state, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_quick_graph_basic(synthetic_adata, temp_state):
    """Test basic quick graph functionality."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    # Set up state
    state.metadata["adata_path"] = str(adata_path)
    
    # Run quick graph
    result = quick_graph(state, seed=42, resolution=0.5, n_neighbors=10, n_pcs=20)
    
    # Check result structure
    assert result.message is not None
    assert not result.message.startswith("❌"), f"Graph analysis failed: {result.message}"
    assert result.state_delta is not None
    assert result.artifacts is not None
    assert len(result.citations) > 0
    
    # Check state updates
    assert "n_clusters" in result.state_delta
    assert "connectivity_rate" in result.state_delta
    assert result.state_delta["n_clusters"] > 0
    assert result.state_delta["connectivity_rate"] > 0
    
    # Check artifacts exist (paths should be created during test)
    assert len(result.artifacts) >= 2  # Should have umap and cluster counts
    
    # Check artifact types
    artifact_types = [Path(art).suffix for art in result.artifacts]
    assert ".png" in artifact_types  # UMAP plot
    assert ".csv" in artifact_types  # Cluster counts


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_quick_graph_with_processed_data(synthetic_adata, temp_state):
    """Test quick graph with pre-processed data."""
    state, temp_dir = temp_state
    
    # Pre-process the data
    sc.pp.normalize_total(synthetic_adata, target_sum=1e4)
    sc.pp.log1p(synthetic_adata)
    
    # Save processed data
    adata_path = Path(temp_dir) / "processed_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Run with different parameters
    result = quick_graph(state, seed=123, resolution=1.0, n_neighbors=15, n_pcs=30)
    
    assert not result.message.startswith("❌")
    assert result.state_delta["n_clusters"] > 0
    assert len(result.artifacts) >= 2


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_quick_graph_parameter_validation(synthetic_adata, temp_state):
    """Test parameter validation in quick graph."""
    state, temp_dir = temp_state
    
    # Create very small dataset
    small_adata = synthetic_adata[:5, :10].copy()  # Only 5 cells, 10 genes
    
    adata_path = Path(temp_dir) / "small_data.h5ad"
    small_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should handle small datasets gracefully
    result = quick_graph(state, seed=42, n_neighbors=10, n_pcs=50)
    
    # Should either work with adjusted parameters or give informative error
    if result.message.startswith("❌"):
        assert "few" in result.message.lower()  # Should mention too few cells/genes
    else:
        # If it works, parameters should be adjusted
        assert result.state_delta is not None


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_quick_graph_reproducibility(synthetic_adata, temp_state):
    """Test that results are reproducible with same seed."""
    state, temp_dir = temp_state
    
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    state.metadata["adata_path"] = str(adata_path)
    
    # Run twice with same seed
    result1 = quick_graph(state, seed=42, resolution=0.5)
    
    # Reset state for second run
    state.history = []
    state.artifacts = {}
    result2 = quick_graph(state, seed=42, resolution=0.5)
    
    # Results should be identical
    assert result1.state_delta["n_clusters"] == result2.state_delta["n_clusters"]
    assert abs(result1.state_delta["connectivity_rate"] - 
              result2.state_delta["connectivity_rate"]) < 0.01


def test_quick_graph_no_scanpy():
    """Test behavior when scanpy is not available."""
    # Temporarily mock SCANPY_AVAILABLE
    import scqc_agent.tools.graph as graph_module
    original_scanpy = graph_module.SCANPY_AVAILABLE
    graph_module.SCANPY_AVAILABLE = False
    
    try:
        state = SessionState()
        result = quick_graph(state)
        
        assert "Scanpy not available" in result.message
        assert len(result.artifacts) == 0
        assert len(result.state_delta) == 0
    finally:
        # Restore original value
        graph_module.SCANPY_AVAILABLE = original_scanpy


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_quick_graph_no_data_loaded(temp_state):
    """Test behavior when no data is loaded."""
    state, temp_dir = temp_state
    
    # Don't set adata_path
    result = quick_graph(state)
    
    assert result.message.startswith("❌")
    assert "No AnnData file loaded" in result.message or "failed" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available") 
def test_cluster_counts_format(synthetic_adata, temp_state):
    """Test that cluster counts CSV has correct format."""
    state, temp_dir = temp_state
    
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    state.metadata["adata_path"] = str(adata_path)
    
    result = quick_graph(state, seed=42)
    
    if not result.message.startswith("❌"):
        # Find cluster counts CSV
        csv_artifacts = [art for art in result.artifacts if art.endswith(".csv")]
        assert len(csv_artifacts) >= 1
        
        csv_path = csv_artifacts[0]
        
        # Verify the CSV exists and can be read
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Check required columns
        assert "cluster" in df.columns
        assert "n_cells" in df.columns  
        assert "percentage" in df.columns
        
        # Check data validity
        assert len(df) > 0
        assert df["n_cells"].sum() == len(synthetic_adata)
        assert abs(df["percentage"].sum() - 100.0) < 0.1  # Should sum to ~100%


if __name__ == "__main__":
    pytest.main([__file__])
