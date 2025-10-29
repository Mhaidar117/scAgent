"""Unit tests for DoubletFinder implementation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

from scqc_agent.state import SessionState
from scqc_agent.tools.doublets import (
    _doubletfinder_core,
    _run_pk_sweep,
    _choose_optimal_pk,
    detect_doublets,
    run_pk_sweep_only,
    curate_doublets_by_markers
)


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
class TestDoubletFinderCore:
    """Test core DoubletFinder functions."""

    @pytest.fixture
    def synthetic_adata(self):
        """Create synthetic AnnData for testing."""
        np.random.seed(42)
        n_cells = 500
        n_genes = 200

        # Generate synthetic counts
        counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
        adata = ad.AnnData(counts)
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        return adata

    def test_doubletfinder_core_basic(self, synthetic_adata):
        """Test basic DoubletFinder execution."""
        result = _doubletfinder_core(
            synthetic_adata,
            pK=0.01,
            expected_rate=0.06,
            pN=0.25,
            n_prin_comps=20,
            random_seed=42
        )

        # Check that required columns were added
        assert 'pANN' in result.obs.columns
        assert 'DF.class' in result.obs.columns
        assert 'doublet_score' in result.obs.columns
        assert 'doublet' in result.obs.columns

        # Check that doublet rate is close to expected
        doublet_rate = (result.obs['doublet'] == True).mean()
        assert 0.01 <= doublet_rate <= 0.15  # Reasonable range

    def test_doubletfinder_reproducibility(self, synthetic_adata):
        """Test that DoubletFinder is reproducible with same seed."""
        result1 = _doubletfinder_core(
            synthetic_adata.copy(),
            pK=0.01,
            expected_rate=0.06,
            random_seed=42
        )

        result2 = _doubletfinder_core(
            synthetic_adata.copy(),
            pK=0.01,
            expected_rate=0.06,
            random_seed=42
        )

        # Check scores are identical
        np.testing.assert_array_equal(
            result1.obs['pANN'].values,
            result2.obs['pANN'].values
        )

    def test_pk_sweep(self, synthetic_adata):
        """Test pK sweep functionality."""
        pK_grid = (0.005, 0.01, 0.02)
        sweep_results = _run_pk_sweep(
            synthetic_adata,
            pK_grid=pK_grid,
            expected_rate=0.06,
            pN=0.25,
            n_prin_comps=20,
            random_seed=42
        )

        # Check results structure
        assert len(sweep_results) == len(pK_grid)
        for pK_val, doublet_frac in sweep_results:
            assert pK_val in pK_grid
            assert 0.0 <= doublet_frac <= 1.0

    def test_choose_optimal_pk(self):
        """Test optimal pK selection."""
        sweep_results = [
            (0.005, 0.03),
            (0.01, 0.05),
            (0.02, 0.06),
            (0.03, 0.08)
        ]

        optimal = _choose_optimal_pk(sweep_results, expected_rate=0.06, tol=0.02)

        # Should choose 0.02 as it's closest to 0.06
        assert optimal == 0.02

    def test_choose_optimal_pk_fallback(self):
        """Test optimal pK selection fallback when none within tolerance."""
        sweep_results = [
            (0.005, 0.01),
            (0.01, 0.02),
            (0.02, 0.03)
        ]

        optimal = _choose_optimal_pk(sweep_results, expected_rate=0.10, tol=0.02)

        # Should fallback to maximum (0.02 with fraction 0.03)
        assert optimal == 0.02


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
class TestDoubletDetectionIntegration:
    """Test full doublet detection tool integration."""

    @pytest.fixture
    def setup_state(self, tmp_path):
        """Setup SessionState with synthetic data."""
        np.random.seed(42)
        n_cells = 300
        n_genes = 150

        counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
        adata = ad.AnnData(counts)
        adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

        # Save to file
        data_path = tmp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)

        # Create state
        state = SessionState(run_id="test_df")
        state.adata_path = str(data_path)

        return state, data_path

    def test_detect_doublets_doubletfinder(self, setup_state):
        """Test detect_doublets with DoubletFinder method."""
        state, _ = setup_state

        result = detect_doublets(
            state,
            method="doubletfinder",
            expected_rate=0.06,
            pK=0.01,
            run_pk_sweep=False,
            random_seed=42
        )

        # Check result structure
        assert not result.message.startswith("❌")
        assert len(result.artifacts) > 0
        assert 'doublet_method' in result.state_delta
        assert result.state_delta['doublet_method'] == 'doubletfinder'

    def test_detect_doublets_with_pk_sweep(self, setup_state):
        """Test detect_doublets with pK sweep enabled."""
        state, _ = setup_state

        result = detect_doublets(
            state,
            method="doubletfinder",
            expected_rate=0.06,
            pK="auto",
            run_pk_sweep=True,
            random_seed=42
        )

        # Check sweep artifacts were created
        assert not result.message.startswith("❌")
        assert any("pk_sweep" in str(a) for a in result.artifacts)

    def test_run_pk_sweep_only_tool(self, setup_state):
        """Test standalone pK sweep tool."""
        state, _ = setup_state

        result = run_pk_sweep_only(
            state,
            pK_grid=(0.005, 0.01, 0.02),
            expected_rate=0.06,
            random_seed=42
        )

        # Check result
        assert not result.message.startswith("❌")
        assert 'optimal_pK' in result.state_delta
        assert len(result.artifacts) > 0

    def test_curate_doublets_by_markers(self, setup_state):
        """Test manual doublet curation by markers."""
        state, data_path = setup_state

        # Add clustering to data
        adata = sc.read_h5ad(data_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=100)
        sc.pp.pca(adata, n_comps=10)
        sc.pp.neighbors(adata, n_pcs=10)
        sc.tl.leiden(adata, resolution=0.5)
        adata.write_h5ad(data_path)

        # Define marker dict
        marker_dict = {
            "Type_A": [f"Gene_{i}" for i in range(10, 20)],
            "Type_B": [f"Gene_{i}" for i in range(30, 40)]
        }

        result = curate_doublets_by_markers(
            state,
            marker_dict=marker_dict,
            cluster_key="leiden",
            avg_exp_threshold=1.0
        )

        # Check result
        assert not result.message.startswith("❌")
        assert 'n_doublet_clusters' in result.state_delta
        assert len(result.artifacts) >= 2  # marker_coexpression.csv + doublet_clusters.json


@pytest.mark.skipif(not SCANPY_AVAILABLE, reason="Scanpy not available")
def test_doubletfinder_state_checkpoint(tmp_path):
    """Test that DoubletFinder creates proper state checkpoints."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))
    adata = ad.AnnData(counts)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

    data_path = tmp_path / "test_data.h5ad"
    adata.write_h5ad(data_path)

    state = SessionState(run_id="test_checkpoint")
    state.adata_path = str(data_path)

    result = detect_doublets(
        state,
        method="doubletfinder",
        pK=0.01,
        run_pk_sweep=False,
        random_seed=42
    )

    # Check checkpoint was created
    assert len(state.history) > 0
    assert state.history[-1]['label'] == 'doublets_detected'
    assert Path(state.history[-1]['checkpoint_path']).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
