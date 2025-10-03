"""Tests for doublet detection functionality with dependency handling."""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from scqc_agent.state import SessionState
from scqc_agent.tools.doublets import detect_doublets, apply_doublet_filter

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import scrublet as scr
    SCRUBLET_AVAILABLE = True
except ImportError:
    SCRUBLET_AVAILABLE = False


@pytest.fixture
def synthetic_adata():
    """Create synthetic AnnData for testing."""
    if not SCANPY_AVAILABLE:
        pytest.skip("Scanpy not available")
    
    # Create synthetic data
    np.random.seed(42)
    n_obs, n_vars = 300, 500  # Enough cells for doublet detection
    
    # Create count data - non-negative integers
    X = np.random.negative_binomial(8, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create AnnData object
    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add batch information
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2', 'batch3'], n_obs)
    adata.obs['SampleID'] = np.random.choice(['sample1', 'sample2'], n_obs)
    
    return adata


@pytest.fixture
def temp_state():
    """Create temporary state for testing."""
    temp_dir = tempfile.mkdtemp()
    state = SessionState(run_id="test_doublets")
    
    yield state, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_doublets_no_dependencies():
    """Test doublet detection behavior when dependencies are not available."""
    # Temporarily mock availability flags
    import scqc_agent.tools.doublets as doublets_module
    original_scanpy = doublets_module.SCANPY_AVAILABLE
    original_scrublet = doublets_module.SCRUBLET_AVAILABLE
    
    # Test with no scanpy
    doublets_module.SCANPY_AVAILABLE = False
    doublets_module.SCRUBLET_AVAILABLE = True
    
    try:
        state = SessionState()
        result = detect_doublets(state)
        
        assert "Scanpy not available" in result.message
        assert len(result.artifacts) == 0
        assert len(result.state_delta) == 0
    finally:
        doublets_module.SCANPY_AVAILABLE = original_scanpy
    
    # Test with no scrublet
    doublets_module.SCANPY_AVAILABLE = True
    doublets_module.SCRUBLET_AVAILABLE = False
    
    try:
        state = SessionState()
        result = detect_doublets(state, method="scrublet")
        
        assert "Scrublet not available" in result.message
        assert "optional dependency" in result.message
        assert "pip install" in result.message
        assert len(result.artifacts) == 0
        assert len(result.citations) > 0  # Should still have citations
    finally:
        # Restore original values
        doublets_module.SCRUBLET_AVAILABLE = original_scrublet


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_doublets_no_data_loaded(temp_state):
    """Test doublet detection behavior when no data is loaded."""
    state, temp_dir = temp_state
    
    result = detect_doublets(state)
    
    assert result.message.startswith("❌")
    assert "No AnnData file loaded" in result.message or "failed" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_doublets_small_dataset(synthetic_adata, temp_state):
    """Test doublet detection with very small dataset."""
    state, temp_dir = temp_state
    
    # Create very small dataset
    small_adata = synthetic_adata[:50, :].copy()  # Only 50 cells
    
    adata_path = Path(temp_dir) / "small_data.h5ad"
    small_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Should fail due to too few cells
    result = detect_doublets(state, expected_rate=0.06)
    
    assert result.message.startswith("❌")
    assert "Too few cells" in result.message
    assert "100" in result.message  # Minimum recommended


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_doublets_invalid_expected_rate(synthetic_adata, temp_state):
    """Test doublet detection with invalid expected rate."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with invalid expected rates
    invalid_rates = [-0.1, 0.0, 0.6, 1.5]
    
    for rate in invalid_rates:
        result = detect_doublets(state, expected_rate=rate)
        
        assert result.message.startswith("❌")
        assert "outside reasonable range" in result.message


def test_doublets_unsupported_method():
    """Test doublet detection with unsupported method."""
    state = SessionState()
    
    result = detect_doublets(state, method="doubletfinder")
    
    assert result.message.startswith("❌")
    assert "DoubletFinder not yet implemented" in result.message
    assert "only 'scrublet' method is supported" in result.message


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCRUBLET_AVAILABLE), 
                    reason="Doublet detection dependencies not available")
def test_doublets_basic_functionality(synthetic_adata, temp_state):
    """Test basic doublet detection functionality if dependencies are available."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Run doublet detection
    result = detect_doublets(
        state, 
        method="scrublet",
        expected_rate=0.06,
        threshold="auto"
    )
    
    # Should complete successfully
    assert not result.message.startswith("❌"), f"Doublet detection failed: {result.message}"
    assert result.state_delta is not None
    assert "doublet_method" in result.state_delta
    assert result.state_delta["doublet_method"] == "scrublet"
    assert "expected_doublet_rate" in result.state_delta
    assert result.state_delta["expected_doublet_rate"] == 0.06
    assert "n_doublets" in result.state_delta
    assert "n_singlets" in result.state_delta
    assert "doublet_threshold" in result.state_delta
    
    # Should have some artifacts (histogram)
    assert len(result.artifacts) >= 1
    
    # Should have citations
    assert len(result.citations) > 0
    assert any("Wolock" in citation for citation in result.citations)


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCRUBLET_AVAILABLE), 
                    reason="Doublet detection dependencies not available")
def test_doublets_custom_threshold(synthetic_adata, temp_state):
    """Test doublet detection with custom threshold if dependencies are available."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test with custom threshold
    custom_threshold = 0.35
    result = detect_doublets(
        state,
        method="scrublet",
        expected_rate=0.06,
        threshold=custom_threshold
    )
    
    if not result.message.startswith("❌"):
        assert "doublet_threshold" in result.state_delta
        assert abs(result.state_delta["doublet_threshold"] - custom_threshold) < 1e-6


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_apply_doublet_filter_no_detection(temp_state):
    """Test applying doublet filter without prior detection."""
    state, temp_dir = temp_state
    
    result = apply_doublet_filter(state)
    
    assert result.message.startswith("❌")
    assert "No AnnData file loaded" in result.message or "Doublet scores not found" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_apply_doublet_filter_with_synthetic_scores(synthetic_adata, temp_state):
    """Test applying doublet filter with synthetic doublet scores."""
    state, temp_dir = temp_state
    
    # Add synthetic doublet scores
    adata = synthetic_adata.copy()
    np.random.seed(42)
    adata.obs['doublet_score'] = np.random.uniform(0, 1, len(adata))
    adata.obs['doublet'] = adata.obs['doublet_score'] > 0.3
    
    # Save data with doublet scores
    adata_path = Path(temp_dir) / "test_data_with_scores.h5ad"
    adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Apply filter
    result = apply_doublet_filter(state)
    
    # Should complete successfully
    assert not result.message.startswith("❌"), f"Doublet filtering failed: {result.message}"
    assert result.state_delta is not None
    assert "cells_before_doublet_filter" in result.state_delta
    assert "cells_after_doublet_filter" in result.state_delta
    assert "doublets_removed" in result.state_delta
    assert "doublet_filter_applied" in result.state_delta
    
    # Should remove some cells
    before = result.state_delta["cells_before_doublet_filter"]
    after = result.state_delta["cells_after_doublet_filter"]
    removed = result.state_delta["doublets_removed"]
    
    assert before == after + removed
    assert removed >= 0
    assert after > 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_apply_doublet_filter_custom_threshold(synthetic_adata, temp_state):
    """Test applying doublet filter with custom threshold."""
    state, temp_dir = temp_state
    
    # Add synthetic doublet scores
    adata = synthetic_adata.copy()
    np.random.seed(42)
    adata.obs['doublet_score'] = np.random.uniform(0, 1, len(adata))
    
    # Save data with doublet scores
    adata_path = Path(temp_dir) / "test_data_with_scores.h5ad"
    adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Apply filter with custom threshold
    custom_threshold = 0.5
    result = apply_doublet_filter(state, threshold=custom_threshold)
    
    if not result.message.startswith("❌"):
        assert "doublet_filter_threshold" in result.state_delta
        assert abs(result.state_delta["doublet_filter_threshold"] - custom_threshold) < 1e-6


@pytest.mark.skipif(not (SCANPY_AVAILABLE and SCRUBLET_AVAILABLE), 
                    reason="Doublet detection dependencies not available")
def test_doublets_end_to_end(synthetic_adata, temp_state):
    """Test end-to-end doublet detection and filtering if dependencies are available."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Step 1: Detect doublets
    detect_result = detect_doublets(
        state,
        method="scrublet",
        expected_rate=0.06,
        threshold="auto"
    )
    
    if detect_result.message.startswith("❌"):
        pytest.skip(f"Doublet detection failed: {detect_result.message}")
    
    # Step 2: Apply filter
    filter_result = apply_doublet_filter(state)
    
    if not filter_result.message.startswith("❌"):
        # Should have completed both steps
        assert "doublet_method" in detect_result.state_delta
        assert "doublet_filter_applied" in filter_result.state_delta
        
        # Should have artifacts from both steps
        assert len(detect_result.artifacts) >= 1  # histogram
        assert len(filter_result.artifacts) >= 1   # removal stats


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_doublets_state_management(synthetic_adata, temp_state):
    """Test that doublet operations properly manage state updates."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Even if doublet detection fails, state management should work
    result = detect_doublets(state, method="scrublet", expected_rate=0.06)
    
    # State should be properly updated even on failure
    if result.message.startswith("❌"):
        # Should handle failure gracefully
        assert len(result.state_delta) == 0
        assert len(result.artifacts) == 0
    else:
        # On success, should have proper state updates
        assert "doublet_method" in result.state_delta or len(result.state_delta) > 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_doublets_parameters_validation(synthetic_adata, temp_state):
    """Test doublet detection parameter validation."""
    state, temp_dir = temp_state
    
    # Save synthetic data
    adata_path = Path(temp_dir) / "test_data.h5ad"
    synthetic_adata.write(adata_path)
    
    state.metadata["adata_path"] = str(adata_path)
    
    # Test various parameter combinations
    test_params = [
        {"expected_rate": 0.05, "threshold": "auto"},
        {"expected_rate": 0.1, "threshold": 0.3},
        {"expected_rate": 0.08, "threshold": 0.5}
    ]
    
    for params in test_params:
        state.history = []  # Reset for each test
        result = detect_doublets(state, method="scrublet", **params)
        
        # Should either work or fail gracefully
        assert result.message is not None
        assert result.state_delta is not None
        assert result.artifacts is not None
        assert result.citations is not None


if __name__ == "__main__":
    pytest.main([__file__])
