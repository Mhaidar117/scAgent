"""Tests for graph analysis from representations functionality."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.graph import graph_from_rep

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.fixture
def synthetic_adata_with_reps():
    """Create synthetic AnnData with multiple representations."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")
    
    # Create synthetic data
    np.random.seed(42)
    n_obs, n_vars = 200, 1000
    
    # Create count data
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create gene names with mitochondrial genes
    mt_genes = [f"MT-{i}" for i in range(50)]
    regular_genes = [f"Gene_{i}" for i in range(50, n_vars)]
    gene_names = mt_genes + regular_genes
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add batch information
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], n_obs)
    
    # Add mitochondrial gene marker
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Create fake representations
    # X_pca: PCA-like representation
    adata.obsm["X_pca"] = np.random.randn(n_obs, 50).astype(np.float32)
    
    # X_scAR: scAR-like denoised representation
    adata.obsm["X_scAR"] = np.random.randn(n_obs, 20).astype(np.float32)
    
    # X_scVI: scVI-like latent representation  
    adata.obsm["X_scVI"] = np.random.randn(n_obs, 30).astype(np.float32)
    
    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_graph_rep")
    
    yield state, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_basic(synthetic_adata_with_reps, temp_state):
    """Test basic graph analysis from representation."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_reps.write(adata_path)
    
    # Set up state
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with X_scVI representation
    result = graph_from_rep(state, use_rep="X_scVI", seed=42, resolution=0.5)
    
    # Check result structure
    assert result.message is not None
    assert not result.message.startswith("❌"), f"Graph analysis failed: {result.message}"
    assert result.state_delta is not None
    assert result.artifacts is not None
    assert len(result.citations) > 0
    
    # Check state updates
    assert "n_clusters_scvi" in result.state_delta
    assert "connectivity_rate_scvi" in result.state_delta
    assert result.state_delta["n_clusters_scvi"] > 0
    assert result.state_delta["connectivity_rate_scvi"] > 0
    
    # Check artifacts exist
    assert len(result.artifacts) >= 2  # Should have umap and cluster counts
    
    # Check artifact types
    artifact_types = [Path(art).suffix for art in result.artifacts]
    assert ".png" in artifact_types  # UMAP plot
    assert ".csv" in artifact_types  # Cluster counts


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_different_representations(synthetic_adata_with_reps, temp_state):
    """Test graph analysis with different representations."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_reps.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test different representations
    representations = ["X_pca", "X_scAR", "X_scVI"]
    
    for use_rep in representations:
        state.history = []  # Reset history for each test
        result = graph_from_rep(state, use_rep=use_rep, seed=42)
        
        rep_short = use_rep.replace("X_", "").lower()
        
        assert not result.message.startswith("❌"), f"Failed for {use_rep}: {result.message}"
        assert f"n_clusters_{rep_short}" in result.state_delta
        assert len(result.artifacts) >= 2
        
        # Check that artifacts contain the representation name
        assert any(rep_short in str(art) for art in result.artifacts)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_invalid_representation(synthetic_adata_with_reps, temp_state):
    """Test graph analysis with invalid representation."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_reps.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with non-existent representation
    result = graph_from_rep(state, use_rep="X_nonexistent")
    
    assert result.message.startswith("❌")
    assert "not found" in result.message
    assert "Available:" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_small_representation(synthetic_adata_with_reps, temp_state):
    """Test graph analysis with very small representation."""
    state, temp_dir = temp_state
    
    # Modify data to have small representation
    adata = synthetic_adata_with_reps.copy()
    adata.obsm["X_tiny"] = np.random.randn(len(adata), 1).astype(np.float32)  # Only 1 dimension
    
    adata_path = Path(temp_dir) / "test_data.h5ad"
    adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with 1D representation (should fail)
    result = graph_from_rep(state, use_rep="X_tiny")
    
    assert result.message.startswith("❌")
    assert "too few dimensions" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_reproducibility(synthetic_adata_with_reps, temp_state):
    """Test that results are reproducible with same seed."""
    state, temp_dir = temp_state
    
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_reps.write(adata_path)
    state.metadata["adata_path"] = str(adata_path)
    
    # Run twice with same seed
    result1 = graph_from_rep(state, use_rep="X_scVI", seed=42, resolution=0.5)
    
    # Reset state for second run
    state.history = []
    state.artifacts = {}
    result2 = graph_from_rep(state, use_rep="X_scVI", seed=42, resolution=0.5)
    
    # Results should be identical
    assert result1.state_delta["n_clusters_scvi"] == result2.state_delta["n_clusters_scvi"]
    assert abs(result1.state_delta["connectivity_rate_scvi"] - 
              result2.state_delta["connectivity_rate_scvi"]) < 0.01


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_parameter_validation(synthetic_adata_with_reps, temp_state):
    """Test parameter validation and adjustment."""
    state, temp_dir = temp_state
    
    # Create very small dataset
    small_adata = synthetic_adata_with_reps[:5, :].copy()  # Only 5 cells
    small_adata.obsm["X_small"] = np.random.randn(5, 10).astype(np.float32)
    
    adata_path = Path(temp_dir) / "small_data.h5ad"
    small_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should handle small datasets gracefully
    result = graph_from_rep(state, use_rep="X_small", n_neighbors=10)
    
    # Should either work with adjusted parameters or give informative error
    if result.message.startswith("❌"):
        assert "few" in result.message.lower()  # Should mention too few cells
    else:
        # If it works, should have valid results
        assert result.state_delta is not None


def test_graph_from_rep_no_scanpy():
    """Test behavior when scanpy is not available."""
    # Temporarily mock SCANPY_AVAILABLE
    import scqc_agent.tools.graph as graph_module
    original_scanpy = graph_module.SCANPY_AVAILABLE
    graph_module.SCANPY_AVAILABLE = False
    
    try:
        state = SessionState()
        result = graph_from_rep(state, use_rep="X_test")
        
        assert "Scanpy not available" in result.message
        assert len(result.artifacts) == 0
        assert len(result.state_delta) == 0
    finally:
        # Restore original value
        graph_module.SCANPY_AVAILABLE = original_scanpy


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_graph_from_rep_no_data_loaded(temp_state):
    """Test behavior when no data is loaded."""
    state, temp_dir = temp_state
    
    # Don't set adata_path
    result = graph_from_rep(state, use_rep="X_test")
    
    assert result.message.startswith("❌")
    assert "No AnnData file loaded" in result.message or "failed" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_cluster_counts_format_from_rep(synthetic_adata_with_reps, temp_state):
    """Test that cluster counts CSV has correct format."""
    state, temp_dir = temp_state
    
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata_with_reps.write(adata_path)
    state.metadata["adata_path"] = str(adata_path)
    
    result = graph_from_rep(state, use_rep="X_scVI", seed=42)
    
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
        assert df["n_cells"].sum() == len(synthetic_adata_with_reps)
        assert abs(df["percentage"].sum() - 100.0) < 0.1  # Should sum to ~100%


if __name__ == "__main__":
    pytest.main([__file__])
