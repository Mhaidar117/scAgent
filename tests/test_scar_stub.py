"""Tests for scAR denoising functionality with dependency handling."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.scar import run_scar

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import scar
    SCAR_AVAILABLE = True
except ImportError:
    SCAR_AVAILABLE = False


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData for testing."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")
    
    # Create synthetic data
    np.random.seed(42)
    n_obs, n_vars = 200, 300  # Smaller for faster testing
    
    # Create count data - non-negative integers for scAR
    X = np.random.negative_binomial(8, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add batch information (required for scAR)
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], n_obs)
    adata.obs['SampleID'] = np.random.choice(['sample1', 'sample2'], n_obs)
    
    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_scar")
    
    yield state, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_scar_no_dependencies():
    """Test scAR behavior when dependencies are not available."""
    # Temporarily mock availability flags
    import scqc_agent.tools.scar as scar_module
    original_scanpy = scar_module.SCANPY_AVAILABLE
    original_scar = scar_module.SCAR_AVAILABLE
    
    # Test with no scanpy
    scar_module.SCANPY_AVAILABLE = False
    scar_module.SCAR_AVAILABLE = True
    
    try:
        state = SessionState()
        result = run_scar(state)
        
        assert "Scanpy not available" in result.message
        assert len(result.artifacts) == 0
        assert len(result.state_delta) == 0
    finally:
        scar_module.SCANPY_AVAILABLE = original_scanpy
    
    # Test with no scAR
    scar_module.SCANPY_AVAILABLE = True
    scar_module.SCAR_AVAILABLE = False
    
    try:
        state = SessionState()
        result = run_scar(state)
        
        assert "scAR not available" in result.message
        assert "optional dependency" in result.message
        assert "pip install" in result.message
        assert len(result.artifacts) == 0
        assert len(result.citations) > 0  # Should still have citations
    finally:
        # Restore original values
        scar_module.SCAR_AVAILABLE = original_scar


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scar_no_data_loaded(temp_state):
    """Test scAR behavior when no data is loaded."""
    state, temp_dir = temp_state
    
    result = run_scar(state)
    
    assert result.message.startswith("❌")
    assert "No AnnData file loaded" in result.message or "failed" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scar_invalid_batch_key(synthetic_adata, temp_state):
    """Test scAR with invalid batch key."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with non-existent batch key
    result = run_scar(state, batch_key="nonexistent_key")
    
    assert result.message.startswith("❌")
    assert "not found" in result.message
    assert "Available keys:" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scar_small_dataset(synthetic_adata, temp_state):
    """Test scAR with very small dataset."""
    state, temp_dir = temp_state
    
    # Create very small dataset
    small_adata = synthetic_adata[:50, :].copy()  # Only 50 cells
    
    adata_path = Path(temp_dir) / "small_data.h5ad"
    small_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should fail due to too few cells
    result = run_scar(state, batch_key="batch")
    
    assert result.message.startswith("❌")
    assert "Too few cells" in result.message
    assert "100" in result.message  # Minimum recommended


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scar_negative_data(synthetic_adata, temp_state):
    """Test scAR with negative data (should fail)."""
    state, temp_dir = temp_state
    
    # Add negative values to the data
    adata = synthetic_adata.copy()
    adata.X = adata.X - 50  # Make some values negative
    
    adata_path = Path(temp_dir) / "negative_data.h5ad"
    adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should fail due to negative values
    result = run_scar(state, batch_key="batch")
    
    assert result.message.startswith("❌")
    assert "non-negative" in result.message


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCAR_AVAILABLE), 
                    reason="scAR dependencies not available")
def test_scar_basic_functionality(synthetic_adata, temp_state):
    """Test basic scAR functionality if dependencies are available."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Run scAR with minimal epochs for testing
    result = run_scar(
        state, 
        batch_key="batch", 
        epochs=2,     # Very few epochs for fast testing
        replace_X=False,  # Don't replace X to preserve original
        random_seed=42
    )
    
    # Should complete successfully
    assert not result.message.startswith("❌"), f"scAR failed: {result.message}"
    assert result.state_delta is not None
    assert "scar_epochs" in result.state_delta
    assert result.state_delta["scar_epochs"] == 2
    assert "scar_batch_key" in result.state_delta
    assert result.state_delta["scar_batch_key"] == "batch"
    
    # Should have some artifacts
    assert len(result.artifacts) >= 1
    
    # Should have citations
    assert len(result.citations) > 0
    assert any("Sheng" in citation for citation in result.citations)


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCAR_AVAILABLE), 
                    reason="scAR dependencies not available")
def test_scar_replace_x_option(synthetic_adata, temp_state):
    """Test scAR replace_X option if dependencies are available."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with replace_X=True
    result = run_scar(
        state,
        batch_key="batch",
        epochs=1,
        replace_X=True,
        random_seed=42
    )
    
    if not result.message.startswith("❌"):
        assert "replace_X" in result.state_delta
        assert result.state_delta["replace_X"] == True
        assert "replaced" in result.message
    
    # Reset state for second test
    state.history = []
    state.artifacts = {}
    
    # Test with replace_X=False
    result = run_scar(
        state,
        batch_key="batch",
        epochs=1,
        replace_X=False,
        random_seed=42
    )
    
    if not result.message.startswith("❌"):
        assert "replace_X" in result.state_delta
        assert result.state_delta["replace_X"] == False
        assert "preserved" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scar_state_management(synthetic_adata, temp_state):
    """Test that scAR properly manages state updates."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Even if scAR fails, state management should work
    result = run_scar(state, batch_key="batch", epochs=1)
    
    # State should be properly updated even on failure
    if result.message.startswith("❌"):
        # Should handle failure gracefully
        assert len(result.state_delta) == 0
        assert len(result.artifacts) == 0
    else:
        # On success, should have proper state updates
        assert "scar_epochs" in result.state_delta or len(result.state_delta) > 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_scar_parameters_validation(synthetic_adata, temp_state):
    """Test scAR parameter validation."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test various parameter combinations
    test_params = [
        {"epochs": 1, "replace_X": True},
        {"epochs": 5, "replace_X": False},
        {"epochs": 10, "replace_X": True, "random_seed": 123}
    ]
    
    for params in test_params:
        state.history = []  # Reset for each test
        result = run_scar(state, batch_key="batch", **params)
        
        # Should either work or fail gracefully
        assert result.message is not None
        assert result.state_delta is not None
        assert result.artifacts is not None
        assert result.citations is not None


if __name__ == "__main__":
    pytest.main([__file__])
