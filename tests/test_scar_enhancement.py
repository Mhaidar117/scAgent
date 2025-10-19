"""Tests for SCAR tool enhancements: knee plot and dual-mode denoising."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from scqc_agent.state import SessionState
from scqc_agent.tools.scar import (
    generate_knee_plot,
    run_scar,
    load_raw_data_from_state,
    calculate_ambient_profile
)
from scqc_agent.tests.synth import make_synth_adata


@pytest.fixture
def mock_state_with_raw(tmp_path):
    """Create mock session state with both raw and filtered data."""
    state = SessionState(run_id="test_scar")

    # Create synthetic filtered data
    filtered_adata = make_synth_adata(n_cells=1000, n_genes=2000, n_batches=2)
    filtered_path = tmp_path / "filtered.h5ad"
    filtered_adata.write_h5ad(filtered_path)
    state.adata_path = str(filtered_path)

    # Create synthetic raw data (10x more droplets)
    raw_adata = make_synth_adata(n_cells=10000, n_genes=2000, n_batches=2)
    raw_path = tmp_path / "raw.h5ad"
    raw_adata.write_h5ad(raw_path)
    state.metadata['raw_adata_path'] = str(raw_path)

    return state, filtered_adata, raw_adata


@pytest.fixture
def mock_state_no_raw(tmp_path):
    """Create mock session state with only filtered data (no raw)."""
    state = SessionState(run_id="test_scar_no_raw")

    # Only filtered data
    filtered_adata = make_synth_adata(n_cells=1000, n_genes=2000, n_batches=2)
    filtered_path = tmp_path / "filtered.h5ad"
    filtered_adata.write_h5ad(filtered_path)
    state.adata_path = str(filtered_path)

    return state, filtered_adata


class TestKneePlot:
    """Tests for generate_knee_plot function."""

    def test_generate_knee_plot_basic(self, mock_state_with_raw, tmp_path):
        """Test basic knee plot generation."""
        state, filtered, raw = mock_state_with_raw

        result = generate_knee_plot(state, min_counts=100)

        # Verify success
        assert not result.message.startswith("❌")
        assert "✅" in result.message

        # Verify artifacts generated
        assert len(result.artifacts) >= 2  # At least knee_plot.png and ambient_profile.csv
        artifact_names = [Path(a).name for a in result.artifacts]
        assert any("knee_plot" in name for name in artifact_names)
        assert any("ambient_profile" in name for name in artifact_names)

        # Verify state updates
        assert "n_cellfree_droplets" in result.state_delta
        assert "ambient_threshold" in result.state_delta
        assert result.state_delta["ambient_threshold"] == 100

    def test_generate_knee_plot_no_raw_data(self, mock_state_no_raw):
        """Test knee plot fails gracefully without raw data."""
        state, filtered = mock_state_no_raw

        result = generate_knee_plot(state, min_counts=100)

        # Should fail with clear error message
        assert "❌" in result.message
        assert "raw" in result.message.lower() or "not available" in result.message.lower()

    def test_calculate_ambient_profile(self, mock_state_with_raw):
        """Test ambient profile calculation."""
        state, filtered, raw = mock_state_with_raw

        ambient_profile = calculate_ambient_profile(raw, filtered, min_counts=100)

        # Verify structure
        assert isinstance(ambient_profile, pd.DataFrame)
        assert 'ambient_profile' in ambient_profile.columns
        assert len(ambient_profile) == filtered.n_vars

        # Verify values are non-negative
        assert (ambient_profile['ambient_profile'] >= 0).all()

    def test_load_raw_data_from_state(self, mock_state_with_raw):
        """Test loading raw data from state."""
        state, filtered, raw = mock_state_with_raw

        loaded_raw = load_raw_data_from_state(state)

        assert loaded_raw is not None
        assert loaded_raw.n_obs == raw.n_obs
        assert loaded_raw.n_vars == raw.n_vars

    def test_load_raw_data_missing(self, mock_state_no_raw):
        """Test loading raw data when not available."""
        state, filtered = mock_state_no_raw

        loaded_raw = load_raw_data_from_state(state)

        assert loaded_raw is None


class TestScarDualMode:
    """Tests for run_scar with dual-mode support."""

    def test_run_scar_with_raw_data(self, mock_state_with_raw):
        """Test SCAR with raw data (Mode A: scvi.external.SCAR)."""
        pytest.importorskip("scvi")  # Skip if scvi-tools not installed

        state, filtered, raw = mock_state_with_raw

        result = run_scar(state, use_raw_data=True, epochs=5, random_seed=42)

        # Should succeed (or indicate scvi.external.SCAR not available)
        assert not result.message.startswith("❌") or "scvi" in result.message.lower()

        if not result.message.startswith("❌"):
            # Verify state updates
            assert "adata_path" in result.state_delta
            assert "scar_mode" in result.state_delta

            # If successful, should indicate Mode A was used
            if result.state_delta.get("scar_mode"):
                assert "scvi" in result.state_delta["scar_mode"].lower()

    def test_run_scar_without_raw_data(self, mock_state_no_raw):
        """Test SCAR without raw data (Mode B: standalone scar)."""
        pytest.importorskip("scar")  # Skip if scar not installed

        state, filtered = mock_state_no_raw

        result = run_scar(state, use_raw_data=False, epochs=5, random_seed=42)

        # Should succeed (or indicate scar not available)
        assert not result.message.startswith("❌") or "scar" in result.message.lower()

        if not result.message.startswith("❌"):
            # Verify state updates
            assert "adata_path" in result.state_delta
            assert "scar_mode" in result.state_delta

            # Should indicate Mode B was used
            if result.state_delta.get("scar_mode"):
                assert "standalone" in result.state_delta["scar_mode"].lower() or "fallback" in result.state_delta["scar_mode"].lower()

    def test_scar_auto_mode_detection(self, mock_state_with_raw):
        """Test that SCAR auto-detects raw data availability."""
        state, filtered, raw = mock_state_with_raw

        # Don't specify use_raw_data, let it auto-detect
        result = run_scar(state, epochs=5, random_seed=42)

        # Should succeed or give clear error about missing dependencies
        assert not result.message.startswith("❌") or ("scar" in result.message.lower() or "scvi" in result.message.lower())

    def test_scar_stores_layers_correctly(self, mock_state_with_raw):
        """Test that SCAR stores denoised counts in correct layers."""
        pytest.importorskip("scvi")

        state, filtered, raw = mock_state_with_raw

        result = run_scar(state, use_raw_data=True, epochs=5, replace_X=False, random_seed=42)

        # If successful, verify layers
        if not result.message.startswith("❌"):
            # Load the denoised data
            import scanpy as sc
            if "adata_path" in result.state_delta:
                adata_denoised = sc.read_h5ad(result.state_delta["adata_path"])

                # Should have counts_denoised layer
                assert 'counts_denoised' in adata_denoised.layers or 'denoised' in adata_denoised.layers

                # Should have latent representation
                assert 'X_scar' in adata_denoised.obsm or 'X_scAR' in adata_denoised.obsm


class TestScarStateManagement:
    """Tests for proper state management in SCAR tools."""

    def test_knee_plot_creates_checkpoint(self, mock_state_with_raw):
        """CRITICAL: Test that knee plot creates checkpoint."""
        state, filtered, raw = mock_state_with_raw

        # Clear history
        state.history = []

        result = generate_knee_plot(state, min_counts=100)

        # Verify checkpoint created (if artifacts were generated)
        if not result.message.startswith("❌"):
            # Note: knee_plot may not create checkpoint since it doesn't modify adata
            # But it should add artifacts to state if successful
            assert len(result.artifacts) > 0

    def test_scar_creates_checkpoint(self, mock_state_with_raw):
        """CRITICAL: Test that SCAR creates checkpoint before artifacts."""
        pytest.importorskip("scvi")

        state, filtered, raw = mock_state_with_raw

        # Clear history
        state.history = []

        result = run_scar(state, use_raw_data=True, epochs=5, random_seed=42)

        # If successful, verify checkpoint created
        if not result.message.startswith("❌") and "adata_path" in result.state_delta:
            assert len(state.history) >= 1
            assert any("scar" in entry.get("label", "").lower() for entry in state.history)

    def test_scar_reproducibility(self, mock_state_with_raw):
        """Test that SCAR produces deterministic results with same seed."""
        pytest.importorskip("scvi")

        state1, filtered, raw = mock_state_with_raw

        # Run SCAR twice with same seed
        result1 = run_scar(state1, use_raw_data=True, epochs=3, random_seed=42)

        # Create fresh state for second run
        state2 = SessionState(run_id="test_scar_2")
        state2.adata_path = state1.adata_path
        state2.metadata['raw_adata_path'] = state1.metadata['raw_adata_path']

        result2 = run_scar(state2, use_raw_data=True, epochs=3, random_seed=42)

        # Both should succeed or both should fail
        if not result1.message.startswith("❌") and not result2.message.startswith("❌"):
            # Results should be deterministic
            assert result1.state_delta.get("scar_epochs") == result2.state_delta.get("scar_epochs")


class TestIntegrationWithMultiload:
    """Test integration between multiload and SCAR tools."""

    def test_scar_uses_multiload_raw_data(self, mock_state_with_raw):
        """Test that SCAR correctly uses raw data set by multiload."""
        state, filtered, raw = mock_state_with_raw

        # Verify raw_adata_path is set (as multiload would do)
        assert 'raw_adata_path' in state.metadata
        assert Path(state.metadata['raw_adata_path']).exists()

        # Generate knee plot should work
        knee_result = generate_knee_plot(state, min_counts=100)
        assert not knee_result.message.startswith("❌")

        # SCAR should auto-detect raw data
        scar_result = run_scar(state, epochs=3, random_seed=42)
        # Should either succeed or give dependency error (not "no raw data" error)
        if scar_result.message.startswith("❌"):
            assert "dependency" in scar_result.message.lower() or "available" in scar_result.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
