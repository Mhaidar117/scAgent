"""Tests for scVI integration functionality with dependency handling."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.scvi import run_scvi

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import scvi
    import torch
    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData for testing."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")
    
    # Create synthetic data
    np.random.seed(42)
    n_obs, n_vars = 300, 500  # Smaller for faster testing
    
    # Create count data - non-negative integers for scVI
    X = np.random.negative_binomial(10, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add batch information (required for scVI)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2', 'batch3'], n_obs)
    adata.obs['SampleID'] = np.random.choice(['sample1', 'sample2'], n_obs)
    
    # Add some basic metadata
    adata.var['highly_variable'] = np.random.choice([True, False], n_vars)
    
    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_scvi")
    
    yield state, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_scvi_no_dependencies():
    """Test scVI behavior when dependencies are not available."""
    # Temporarily mock availability flags
    import scqc_agent.tools.scvi as scvi_module
    original_scanpy = scvi_module.SCANPY_AVAILABLE
    original_scvi = scvi_module.SCVI_AVAILABLE
    original_torch = scvi_module.TORCH_AVAILABLE
    
    # Test with no scanpy
    scvi_module.SCANPY_AVAILABLE = False
    scvi_module.SCVI_AVAILABLE = True
    scvi_module.TORCH_AVAILABLE = True
    
    try:
        state = SessionState()
        result = run_scvi(state)
        
        assert "Scanpy not available" in result.message
        assert len(result.artifacts) == 0
        assert len(result.state_delta) == 0
    finally:
        scvi_module.SCANPY_AVAILABLE = original_scanpy
    
    # Test with no scVI/torch
    scvi_module.SCANPY_AVAILABLE = True
    scvi_module.SCVI_AVAILABLE = False
    scvi_module.TORCH_AVAILABLE = False
    
    try:
        state = SessionState()
        result = run_scvi(state)
        
        assert "not available" in result.message
        assert "scvi-tools" in result.message or "torch" in result.message
        assert "pip install" in result.message
        assert len(result.artifacts) == 0
        assert len(result.citations) > 0  # Should still have citations
    finally:
        # Restore original values
        scvi_module.SCVI_AVAILABLE = original_scvi
        scvi_module.TORCH_AVAILABLE = original_torch


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scvi_no_data_loaded(temp_state):
    """Test scVI behavior when no data is loaded."""
    state, temp_dir = temp_state
    
    result = run_scvi(state)
    
    assert result.message.startswith("❌")
    assert "No AnnData file loaded" in result.message or "failed" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scvi_invalid_batch_key(synthetic_adata, temp_state):
    """Test scVI with invalid batch key."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with non-existent batch key
    result = run_scvi(state, batch_key="nonexistent_key")
    
    assert result.message.startswith("❌")
    assert "not found" in result.message
    assert "Available keys:" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scvi_small_dataset(synthetic_adata, temp_state):
    """Test scVI with very small dataset."""
    state, temp_dir = temp_state
    
    # Create very small dataset
    small_adata = synthetic_adata[:50, :].copy()  # Only 50 cells
    
    adata_path = Path(temp_dir) / "small_data.h5ad"
    small_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should fail due to too few cells
    result = run_scvi(state, batch_key="batch")
    
    assert result.message.startswith("❌")
    assert "Too few cells" in result.message
    assert "200" in result.message  # Minimum recommended


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scvi_negative_data(synthetic_adata, temp_state):
    """Test scVI with negative data (should fail)."""
    state, temp_dir = temp_state
    
    # Add negative values to the data
    adata = synthetic_adata.copy()
    adata.X = adata.X - 100  # Make some values negative
    
    adata_path = Path(temp_dir) / "negative_data.h5ad"
    adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should fail due to negative values
    result = run_scvi(state, batch_key="batch")
    
    assert result.message.startswith("❌")
    assert "non-negative" in result.message


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCVI_AVAILABLE), 
                    reason="scVI dependencies not available")
def test_scvi_basic_functionality(synthetic_adata, temp_state):
    """Test basic scVI functionality if dependencies are available."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Run scVI with minimal epochs for testing
    result = run_scvi(
        state, 
        batch_key="batch", 
        n_latent=10,  # Small latent space
        epochs=2,     # Very few epochs for fast testing
        random_seed=42
    )
    
    # Should complete successfully
    assert not result.message.startswith("❌"), f"scVI failed: {result.message}"
    assert result.state_delta is not None
    assert "scvi_n_latent" in result.state_delta
    assert result.state_delta["scvi_n_latent"] == 10
    assert "n_batches" in result.state_delta
    assert result.state_delta["n_batches"] > 0
    
    # Should have some artifacts
    assert len(result.artifacts) >= 1
    
    # Should have citations
    assert len(result.citations) > 0
    assert any("Lopez" in citation for citation in result.citations)


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCVI_AVAILABLE), 
                    reason="scVI dependencies not available")
def test_scvi_parameter_adjustment(synthetic_adata, temp_state):
    """Test scVI parameter adjustment for small datasets."""
    state, temp_dir = temp_state
    
    # Create dataset with fewer cells but still above minimum
    medium_adata = synthetic_adata[:250, :].copy()  # 250 cells
    
    adata_path = Path(temp_dir) / "medium_data.h5ad"
    medium_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Try with large n_latent that should be adjusted
    result = run_scvi(
        state,
        batch_key="batch",
        n_latent=300,  # Larger than n_cells
        epochs=1,
        random_seed=42
    )
    
    # Should work with adjusted parameters or give informative message
    if not result.message.startswith("❌"):
        # Parameters should be adjusted
        assert result.state_delta["scvi_n_latent"] < 300
    
    # Or should contain adjustment message
    if "adjusted" in result.message:
        assert "n_latent" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scvi_state_management(synthetic_adata, temp_state):
    """Test that scVI properly manages state updates."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Even if scVI fails, state management should work
    result = run_scvi(state, batch_key="batch", epochs=1)
    
    # State should be properly updated even on failure
    if result.message.startswith("❌"):
        # Should handle failure gracefully
        assert len(result.state_delta) == 0
        assert len(result.artifacts) == 0
    else:
        # On success, should have proper state updates
        assert "scvi_epochs" in result.state_delta or len(result.state_delta) > 0


if __name__ == "__main__":
    pytest.main([__file__])
