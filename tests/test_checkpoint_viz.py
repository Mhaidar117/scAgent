"""Tests for checkpoint visualization tool."""

import pytest
from pathlib import Path
from scqc_agent.state import SessionState
from scqc_agent.tools.checkpoint_viz import generate_checkpoint_umap

# Import synthetic data generator
try:
    from scqc_agent.tests.synth import make_synth_adata
    SYNTH_AVAILABLE = True
except ImportError:
    SYNTH_AVAILABLE = False

# Check for scanpy
try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not installed")
@pytest.mark.skipif(not SYNTH_AVAILABLE, reason="Synthetic data generator not available")
def test_generate_checkpoint_umap_basic(tmp_path):
    """Test basic checkpoint UMAP generation."""
    # Create synthetic data
    adata = make_synth_adata(n_cells=500, n_genes=1000, n_batches=2, random_seed=42)

    # Save to temp file
    data_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(data_path)

    # Initialize state
    state = SessionState(run_id="test_checkpoint")
    state.adata_path = str(data_path)

    # Generate checkpoint UMAP
    result = generate_checkpoint_umap(
        state,
        stage_label="test_stage",
        resolution=2.0,
        n_pcs=40,
        random_seed=42
    )

    # Check result
    assert not result.message.startswith("Error")
    assert len(result.artifacts) == 1
    assert "umap_test_stage.pdf" in result.artifacts[0]
    assert Path(result.artifacts[0]).exists()
    assert "Wolf" in result.citations[0]


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not installed")
@pytest.mark.skipif(not SYNTH_AVAILABLE, reason="Synthetic data generator not available")
def test_generate_checkpoint_umap_with_layer(tmp_path):
    """Test checkpoint UMAP with layer specification."""
    # Create synthetic data with layer
    adata = make_synth_adata(n_cells=500, n_genes=1000, random_seed=42)
    adata.layers["counts_denoised"] = adata.X.copy()

    # Save to temp file
    data_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(data_path)

    # Initialize state
    state = SessionState(run_id="test_checkpoint_layer")
    state.adata_path = str(data_path)

    # Generate checkpoint UMAP with layer
    result = generate_checkpoint_umap(
        state,
        stage_label="postSCAR",
        layer="counts_denoised",
        resolution=1.5,
        n_pcs=30,
        random_seed=42
    )

    # Check result
    assert not result.message.startswith("Error")
    assert "postSCAR" in result.message
    assert len(result.artifacts) == 1
    assert Path(result.artifacts[0]).exists()


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not installed")
@pytest.mark.skipif(not SYNTH_AVAILABLE, reason="Synthetic data generator not available")
def test_generate_checkpoint_umap_invalid_layer(tmp_path):
    """Test checkpoint UMAP with invalid layer specification."""
    # Create synthetic data
    adata = make_synth_adata(n_cells=500, n_genes=1000, random_seed=42)

    # Save to temp file
    data_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(data_path)

    # Initialize state
    state = SessionState(run_id="test_checkpoint_invalid")
    state.adata_path = str(data_path)

    # Try to generate with non-existent layer
    result = generate_checkpoint_umap(
        state,
        stage_label="test",
        layer="nonexistent_layer",
        random_seed=42
    )

    # Should return error
    assert result.message.startswith("Error")
    assert "Layer" in result.message
    assert len(result.artifacts) == 0


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not installed")
def test_generate_checkpoint_umap_no_data():
    """Test checkpoint UMAP without loaded data."""
    # Initialize state without data
    state = SessionState(run_id="test_no_data")

    # Try to generate UMAP
    result = generate_checkpoint_umap(
        state,
        stage_label="test",
        random_seed=42
    )

    # Should return error
    assert result.message.startswith("Error")
    assert "No data loaded" in result.message


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not installed")
@pytest.mark.skipif(not SYNTH_AVAILABLE, reason="Synthetic data generator not available")
def test_generate_checkpoint_umap_adds_artifact_only(tmp_path):
    """CRITICAL: Test that checkpoint UMAP adds artifact but not checkpoint."""
    # Create synthetic data
    adata = make_synth_adata(n_cells=500, n_genes=1000, random_seed=42)

    # Save to temp file
    data_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(data_path)

    # Initialize state
    state = SessionState(run_id="test_artifact_only")
    state.adata_path = str(data_path)

    # Record initial history length
    initial_history_len = len(state.history)

    # Generate checkpoint UMAP
    result = generate_checkpoint_umap(
        state,
        stage_label="viz_test",
        random_seed=42
    )

    # Verify NO checkpoint was created
    assert len(state.history) == initial_history_len

    # Verify artifact was added
    assert len(result.artifacts) == 1
    assert len(state.artifacts) > 0

    # Verify no state_delta
    assert result.state_delta == {}


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not installed")
@pytest.mark.skipif(not SYNTH_AVAILABLE, reason="Synthetic data generator not available")
def test_generate_checkpoint_umap_reproducibility(tmp_path):
    """Test that checkpoint UMAP is reproducible with same seed."""
    # Create synthetic data
    adata = make_synth_adata(n_cells=500, n_genes=1000, random_seed=42)

    # Save to temp file
    data_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(data_path)

    # Initialize state
    state1 = SessionState(run_id="test_repro_1")
    state1.adata_path = str(data_path)

    state2 = SessionState(run_id="test_repro_2")
    state2.adata_path = str(data_path)

    # Generate UMAPs with same seed
    result1 = generate_checkpoint_umap(state1, "stage1", random_seed=42)
    result2 = generate_checkpoint_umap(state2, "stage2", random_seed=42)

    # Both should succeed
    assert not result1.message.startswith("Error")
    assert not result2.message.startswith("Error")

    # Same number of clusters (reproducibility check)
    # Note: We can't compare exact coordinates, but clustering should be identical
    assert "clusters" in result1.message
    assert "clusters" in result2.message

    # Extract cluster counts from messages
    clusters1 = int(result1.message.split("clusters")[0].split()[-1])
    clusters2 = int(result2.message.split("clusters")[0].split()[-1])
    assert clusters1 == clusters2
