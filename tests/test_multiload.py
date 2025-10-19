"""Comprehensive tests for multi-file kidney data loader.

Tests cover:
- Valid 3-file loading workflow
- File existence validation
- File format validation
- Metadata merge correctness
- State checkpoint creation
- Artifact tracking in history
- Error handling for missing files
- Error handling for invalid formats
- Reproducibility and determinism
"""

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from scqc_agent.state import SessionState, ToolResult
from scqc_agent.tools.multiload import load_kidney_data

# Test fixtures
pytest.importorskip("scanpy")
pytest.importorskip("anndata")

import scanpy as sc
import anndata as ad


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary directory for test artifacts."""
    test_dir = tmp_path / "multiload_test"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def synthetic_10x_h5_data(temp_test_dir):
    """Create synthetic 10X HDF5 files (raw and filtered) with realistic structure.

    Returns:
        tuple: (raw_h5_path, filtered_h5_path, metadata_csv_path)
    """
    # Create synthetic raw data (all droplets including empty ones)
    n_droplets_raw = 10000
    n_genes = 2000

    # Raw data: mostly empty droplets + some cells
    np.random.seed(42)

    # Simulate counts: ~30% cells, 70% empty droplets
    counts_raw = np.zeros((n_droplets_raw, n_genes))
    n_real_cells = 3000

    # Real cells have higher counts
    for i in range(n_real_cells):
        n_genes_expressed = np.random.randint(500, 1500)
        expressed_genes = np.random.choice(n_genes, n_genes_expressed, replace=False)
        counts_raw[i, expressed_genes] = np.random.poisson(3, n_genes_expressed)

    # Empty droplets have low ambient counts
    for i in range(n_real_cells, n_droplets_raw):
        n_ambient_genes = np.random.randint(10, 50)
        ambient_genes = np.random.choice(n_genes, n_ambient_genes, replace=False)
        counts_raw[i, ambient_genes] = np.random.poisson(0.5, n_ambient_genes)

    # Create raw AnnData
    adata_raw = ad.AnnData(X=counts_raw.astype(np.float32))
    adata_raw.obs_names = [f"DROPLET_{i}" for i in range(n_droplets_raw)]
    adata_raw.var_names = [f"GENE_{i}" for i in range(n_genes)]

    # Add some MT genes for realism
    mt_genes = [f"MT-{gene}" for gene in ["ND1", "ND2", "CO1", "CO2", "ATP6"]]
    adata_raw.var_names = mt_genes + list(adata_raw.var_names[len(mt_genes):])

    # Save raw H5AD in 10X format (using .h5ad as proxy for .h5)
    raw_h5_path = temp_test_dir / "raw_feature_bc_matrix.h5"
    adata_raw.write_h5ad(raw_h5_path)

    # Create filtered data (cells only)
    adata_filtered = adata_raw[:n_real_cells, :].copy()
    adata_filtered.obs_names = [f"CELL_{i}" for i in range(n_real_cells)]

    filtered_h5_path = temp_test_dir / "filtered_feature_bc_matrix.h5"
    adata_filtered.write_h5ad(filtered_h5_path)

    # Create metadata CSV
    metadata = pd.DataFrame({
        'sample_ID': ['KidneySample_001'],
        'animal_species': ['mouse'],
        'sex': ['male'],
        'age': [8],
        'tissue_type': ['kidney'],
        'treatment': ['control'],
        'batch': ['batch1']
    })

    metadata_csv_path = temp_test_dir / "metadata.csv"
    metadata.to_csv(metadata_csv_path, index=False)

    return str(raw_h5_path), str(filtered_h5_path), str(metadata_csv_path)


@pytest.fixture
def test_state(temp_test_dir):
    """Create test session state."""
    state = SessionState(run_id="test_multiload")
    return state


# ==================== BASIC FUNCTIONALITY TESTS ====================

def test_load_kidney_data_basic(test_state, synthetic_10x_h5_data):
    """Test basic 3-file loading workflow."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv,
        sample_id_column="sample_ID"
    )

    # Verify success
    assert not result.message.startswith("Error"), f"Tool failed: {result.message}"
    assert "Successfully loaded kidney dataset" in result.message

    # Verify artifacts generated
    assert len(result.artifacts) >= 2, "Should generate at least summary JSON and CSV"

    # Verify state updates
    assert "adata_path" in result.state_delta
    assert "raw_adata_path" in result.state_delta
    assert "n_cells_raw" in result.state_delta
    assert "n_cells_filtered" in result.state_delta
    assert "n_genes" in result.state_delta

    # Verify counts are correct
    assert result.state_delta["n_cells_raw"] == 10000
    assert result.state_delta["n_cells_filtered"] == 3000
    assert result.state_delta["n_genes"] == 2000


def test_load_creates_checkpoint(test_state, synthetic_10x_h5_data):
    """CRITICAL: Test that checkpoint is created in state history."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    # Verify history is empty before load
    assert len(test_state.history) == 0

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Verify checkpoint creates history entry
    assert len(test_state.history) == 1, "Should create one history entry"
    assert test_state.history[0]["label"] == "multiload_filtered"
    assert "checkpoint_path" in test_state.history[0]

    # Verify checkpoint path points to filtered data
    checkpoint_path = test_state.history[0]["checkpoint_path"]
    assert "adata_filtered.h5ad" in checkpoint_path or "adata_step00.h5ad" in checkpoint_path


def test_load_adds_artifacts_to_history(test_state, synthetic_10x_h5_data):
    """CRITICAL: Test that artifacts appear in history entry."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Verify artifacts in history entry
    assert len(test_state.history) == 1
    history_artifacts = test_state.history[0]["artifacts"]
    assert len(history_artifacts) >= 2, "Should have at least 2 artifacts in history"

    # Check artifact structure
    for artifact in history_artifacts:
        assert "path" in artifact
        assert "label" in artifact
        assert "timestamp" in artifact

    # Verify specific artifacts exist
    artifact_paths = [a["path"] for a in history_artifacts]
    assert any("load_summary.json" in p for p in artifact_paths)
    assert any("load_summary.csv" in p for p in artifact_paths)


def test_load_stores_raw_checkpoint(test_state, synthetic_10x_h5_data):
    """Test that raw data checkpoint is created for SCAR."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Verify raw_adata_path in state delta
    assert "raw_adata_path" in result.state_delta
    raw_path = Path(result.state_delta["raw_adata_path"])

    # Verify raw checkpoint exists
    assert raw_path.exists(), f"Raw checkpoint not found: {raw_path}"

    # Verify raw checkpoint is loadable
    adata_raw = sc.read_h5ad(str(raw_path))
    assert adata_raw.n_obs == 10000, "Raw checkpoint should have all droplets"


def test_load_stores_original_counts_in_layers(test_state, synthetic_10x_h5_data):
    """Test that original counts are stored in filtered.layers['counts_raw']."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Load filtered checkpoint
    filtered_path = result.state_delta["adata_path"]
    adata_filtered = sc.read_h5ad(filtered_path)

    # Verify counts_raw layer exists
    assert "counts_raw" in adata_filtered.layers, "Should store original counts in layers"

    # Verify layer has correct shape
    assert adata_filtered.layers["counts_raw"].shape == adata_filtered.X.shape


def test_metadata_merge(test_state, synthetic_10x_h5_data):
    """Test that metadata is correctly merged into filtered.obs."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv,
        sample_id_column="sample_ID"
    )

    # Load filtered checkpoint
    filtered_path = result.state_delta["adata_path"]
    adata_filtered = sc.read_h5ad(filtered_path)

    # Verify metadata columns exist in obs
    expected_cols = ["sample_ID", "animal_species", "sex", "age", "tissue_type", "treatment", "batch"]
    for col in expected_cols:
        assert col in adata_filtered.obs.columns, f"Metadata column '{col}' not found in obs"

    # Verify values are correct
    assert adata_filtered.obs["animal_species"].iloc[0] == "mouse"
    assert adata_filtered.obs["tissue_type"].iloc[0] == "kidney"


# ==================== ERROR HANDLING TESTS ====================

def test_missing_raw_file(test_state, synthetic_10x_h5_data):
    """Test error handling for missing raw H5 file."""
    _, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path="/nonexistent/raw.h5",
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    assert "not found" in result.message.lower() or "error" in result.message.lower()
    assert len(result.artifacts) == 0


def test_missing_filtered_file(test_state, synthetic_10x_h5_data):
    """Test error handling for missing filtered H5 file."""
    raw_h5, _, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path="/nonexistent/filtered.h5",
        meta_csv_path=meta_csv
    )

    assert "not found" in result.message.lower() or "error" in result.message.lower()
    assert len(result.artifacts) == 0


def test_missing_metadata_file(test_state, synthetic_10x_h5_data):
    """Test error handling for missing metadata CSV file."""
    raw_h5, filtered_h5, _ = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path="/nonexistent/metadata.csv"
    )

    assert "not found" in result.message.lower() or "error" in result.message.lower()
    assert len(result.artifacts) == 0


def test_invalid_sample_id_column(test_state, synthetic_10x_h5_data):
    """Test error handling for invalid sample ID column name."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv,
        sample_id_column="INVALID_COLUMN"
    )

    assert "not found" in result.message.lower()
    assert "Available columns" in result.message


# ==================== STATE MANAGEMENT TESTS ====================

def test_state_delta_contains_all_required_fields(test_state, synthetic_10x_h5_data):
    """Test that state_delta contains all required fields."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    required_fields = [
        "adata_path",
        "raw_adata_path",
        "n_cells_raw",
        "n_cells_filtered",
        "n_genes",
        "sample_id",
        "metadata_columns",
        "cells_retained_fraction"
    ]

    for field in required_fields:
        assert field in result.state_delta, f"Missing required field: {field}"


def test_cells_retained_fraction_calculation(test_state, synthetic_10x_h5_data):
    """Test that cells_retained_fraction is correctly calculated."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    n_raw = result.state_delta["n_cells_raw"]
    n_filtered = result.state_delta["n_cells_filtered"]
    fraction = result.state_delta["cells_retained_fraction"]

    expected_fraction = n_filtered / n_raw
    assert abs(fraction - expected_fraction) < 1e-6, "Fraction calculation incorrect"


def test_summary_json_artifact_content(test_state, synthetic_10x_h5_data):
    """Test that summary JSON artifact contains correct structure."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Find summary JSON artifact
    json_artifacts = [a for a in result.artifacts if a.endswith("load_summary.json")]
    assert len(json_artifacts) == 1, "Should generate exactly one JSON summary"

    # Load and verify structure
    with open(json_artifacts[0], 'r') as f:
        summary = json.load(f)

    assert "timestamp" in summary
    assert "files" in summary
    assert "counts" in summary
    assert "metadata" in summary

    # Verify counts section
    assert summary["counts"]["n_raw_droplets"] == 10000
    assert summary["counts"]["n_filtered_cells"] == 3000
    assert summary["counts"]["n_genes_filtered"] == 2000


# ==================== REPRODUCIBILITY TESTS ====================

def test_reproducibility_deterministic_loading(test_state, synthetic_10x_h5_data):
    """Test that loading is deterministic (same inputs → same outputs)."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    # Load twice with same parameters
    result1 = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Create fresh state for second load
    state2 = SessionState(run_id="test_multiload_2")
    result2 = load_kidney_data(
        state2,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Verify counts are identical
    assert result1.state_delta["n_cells_raw"] == result2.state_delta["n_cells_raw"]
    assert result1.state_delta["n_cells_filtered"] == result2.state_delta["n_cells_filtered"]
    assert result1.state_delta["n_genes"] == result2.state_delta["n_genes"]


# ==================== INTEGRATION TESTS ====================

def test_integration_with_downstream_qc(test_state, synthetic_10x_h5_data):
    """Test that loaded data can be used for downstream QC analysis."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    # Load data
    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Update state with delta
    test_state.adata_path = result.state_delta["adata_path"]
    test_state.update_metadata(result.state_delta)

    # Verify state is ready for QC
    assert test_state.adata_path is not None
    assert Path(test_state.adata_path).exists()

    # Load data and verify it's valid for QC
    adata = sc.read_h5ad(test_state.adata_path)
    assert adata.n_obs > 0
    assert adata.n_vars > 0
    assert "counts_raw" in adata.layers


def test_integration_raw_data_for_scar(test_state, synthetic_10x_h5_data):
    """Test that raw data checkpoint is usable for SCAR tool."""
    raw_h5, filtered_h5, meta_csv = synthetic_10x_h5_data

    result = load_kidney_data(
        test_state,
        raw_h5_path=raw_h5,
        filtered_h5_path=filtered_h5,
        meta_csv_path=meta_csv
    )

    # Verify raw data is accessible via state
    raw_path = result.state_delta["raw_adata_path"]
    assert Path(raw_path).exists()

    # Load raw data
    adata_raw = sc.read_h5ad(raw_path)

    # Verify raw data characteristics for SCAR
    assert adata_raw.n_obs > result.state_delta["n_cells_filtered"], "Raw should have more cells than filtered"
    assert adata_raw.X.min() >= 0, "SCAR requires non-negative counts"


# ==================== PYDANTIC SCHEMA VALIDATION TESTS ====================

def test_pydantic_schema_validation():
    """Test that Pydantic schema correctly validates inputs."""
    from scqc_agent.agent.schemas import LoadKidneyDataInput

    # Valid input
    valid_input = {
        "raw_h5_path": "/tmp/test_raw.h5",
        "filtered_h5_path": "/tmp/test_filtered.h5",
        "meta_csv_path": "/tmp/test_meta.csv",
        "sample_id_column": "sample_ID"
    }

    # This will fail validation due to file not existing, but tests schema structure
    try:
        schema = LoadKidneyDataInput(**valid_input)
    except ValueError as e:
        # Expected - files don't exist
        assert "does not exist" in str(e)


def test_pydantic_schema_rejects_invalid_extensions():
    """Test that Pydantic schema rejects invalid file extensions."""
    from scqc_agent.agent.schemas import LoadKidneyDataInput
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with wrong extensions
        wrong_h5 = Path(tmpdir) / "data.txt"
        wrong_h5.touch()

        wrong_csv = Path(tmpdir) / "meta.txt"
        wrong_csv.touch()

        # Schema should reject non-.h5 files
        with pytest.raises(ValueError, match="must be a 10X H5 file"):
            LoadKidneyDataInput(
                raw_h5_path=str(wrong_h5),
                filtered_h5_path=str(wrong_h5),
                meta_csv_path=str(wrong_csv)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
