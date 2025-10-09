"""Tests for QC filtering functionality."""

import numpy as np
import tempfile
import json
from pathlib import Path

import pytest

from scqc_agent.state import SessionState, ToolResult


def create_synthetic_adata_with_qc():
    """Create synthetic AnnData with QC metrics computed."""
    try:
        import scanpy as sc
        import anndata as ad
        import pandas as pd
    except ImportError:
        pytest.skip("Scanpy not available for QC tests")
    
    # Create synthetic data (500 cells, 1000 genes)
    n_obs, n_vars = 500, 1000
    
    # Generate counts matrix
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create gene names with mitochondrial genes
    gene_names = [f"Gene_{i:04d}" for i in range(n_vars-30)]
    mt_genes = [f"mt-gene{i}" for i in range(30)]  # 30 MT genes
    gene_names.extend(mt_genes)
    
    # Create cell barcodes
    cell_names = [f"Cell_{i:04d}" for i in range(n_obs)]
    
    # Create AnnData object
    adata = ad.AnnData(X, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
    
    # Add batch information
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2'], n_obs)
    
    # Mark mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Add mitochondrial percentage
    adata.obs['pct_counts_mt'] = (
        adata[:, adata.var['mt']].X.sum(axis=1).A1 / 
        adata.obs['total_counts'] * 100
    )
    
    return adata


class TestQCFilters:
    """Test QC filtering functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_apply_qc_filters_no_scanpy(self):
        """Test QC filtering when scanpy is not available."""
        from scqc_agent.tools.qc import SCANPY_AVAILABLE, apply_qc_filters

        if SCANPY_AVAILABLE:
            pytest.skip("Scanpy is available, cannot test fallback behavior")

        state = SessionState(run_id="test_run")
        state.adata_path = "test.h5ad"
        result = apply_qc_filters(state)

        assert "Scanpy not available" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0
    
    def test_apply_qc_filters_no_qc_metrics(self):
        """Test QC filtering when QC metrics haven't been computed."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
            import anndata as ad
            import pandas as pd
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create data without QC metrics
        n_obs, n_vars = 100, 500
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        gene_names = [f"Gene_{i:04d}" for i in range(n_vars)]
        cell_names = [f"Cell_{i:04d}" for i in range(n_obs)]
        adata = ad.AnnData(X, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
        
        # Save data without QC metrics
        data_path = self.temp_path / "no_qc_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        
        # Try to apply filters
        result = apply_qc_filters(state)
        
        assert "QC metrics not found" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0
    
    def test_apply_qc_filters_threshold_method(self):
        """Test QC filtering with threshold method."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Store original counts
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        # Create state
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        state.metadata = {"batch_key": "batch"}
        
        # Apply filters with threshold method
        result = apply_qc_filters(state, min_genes=200, max_pct_mt=20.0, method="threshold")
        
        # Verify result
        assert "QC filters applied" in result.message
        assert "Retained" in result.message
        
        # Check state delta
        assert "dataset_summary" in result.state_delta
        dataset_summary = result.state_delta["dataset_summary"]
        assert dataset_summary["qc_filtered"] is True
        assert "n_cells_after_qc" in dataset_summary
        assert "n_genes_after_qc" in dataset_summary
        assert "retained_fraction_cells" in dataset_summary
        
        # Check that some filtering occurred (unless all cells pass filters)
        assert dataset_summary["n_cells_after_qc"] <= n_cells_before
        assert dataset_summary["n_genes_after_qc"] <= n_genes_before
        
        # Check filter thresholds
        thresholds = dataset_summary["filter_thresholds"]
        assert thresholds["min_genes"] == 200
        assert thresholds["max_pct_mt"] == 20.0
        
        # Check artifacts were created
        assert len(result.artifacts) >= 1  # Filter summary and snapshot
        
        # Verify JSON file exists and contains expected data
        json_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_filters.json"):
                json_file = artifact
                break
        
        assert json_file is not None
        assert Path(json_file).exists()
        
        with open(json_file, 'r') as f:
            filter_data = json.load(f)
        
        assert filter_data["method"] == "threshold"
        assert filter_data["thresholds"]["min_genes"] == 200
        assert filter_data["thresholds"]["max_pct_mt"] == 20.0
        assert "before_filtering" in filter_data
        assert "after_filtering" in filter_data
        assert "retained_fraction" in filter_data
        
        # Check citations
        assert len(result.citations) > 0
    
    def test_apply_qc_filters_mad_method(self):
        """Test QC filtering with MAD method."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        
        # Apply filters with MAD method
        result = apply_qc_filters(state, min_genes=100, max_pct_mt=50.0, method="MAD")
        
        # Verify result
        assert "QC filters applied" in result.message
        
        # Check that MAD method was used
        json_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_filters.json"):
                json_file = artifact
                break
        
        assert json_file is not None
        
        with open(json_file, 'r') as f:
            filter_data = json.load(f)
        
        assert filter_data["method"] == "MAD"
        # MAD method should potentially adjust the thresholds
        assert "thresholds" in filter_data
    
    def test_apply_qc_filters_quantile_method(self):
        """Test QC filtering with quantile method."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        
        # Apply filters with quantile method
        result = apply_qc_filters(state, min_genes=100, max_pct_mt=50.0, method="quantile")
        
        # Verify result
        assert "QC filters applied" in result.message
        
        # Check that quantile method was used
        json_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_filters.json"):
                json_file = artifact
                break
        
        assert json_file is not None
        
        with open(json_file, 'r') as f:
            filter_data = json.load(f)
        
        assert filter_data["method"] == "quantile"
        # Quantile method should potentially adjust the thresholds
        assert "thresholds" in filter_data
    
    def test_apply_qc_filters_with_batches(self):
        """Test QC filtering with batch information."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state with batch key
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        state.metadata = {"batch_key": "batch"}
        
        # Apply filters
        result = apply_qc_filters(state, min_genes=200, max_pct_mt=20.0)
        
        # Verify batch information is included
        json_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_filters.json"):
                json_file = artifact
                break
        
        assert json_file is not None
        
        with open(json_file, 'r') as f:
            filter_data = json.load(f)
        
        assert "per_batch_retained" in filter_data
        per_batch = filter_data["per_batch_retained"]
        
        # Should have information for both batches
        assert len(per_batch) == 2
        assert "batch1" in per_batch
        assert "batch2" in per_batch
        
        for batch_name, batch_data in per_batch.items():
            assert "n_cells_retained" in batch_data
            assert "fraction_of_total" in batch_data
    
    def test_apply_qc_filters_extreme_thresholds(self):
        """Test QC filtering with extreme thresholds."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Store original counts
        n_cells_before = adata.n_obs
        
        # Create state
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        
        # Test very strict thresholds (should remove most cells)
        result = apply_qc_filters(state, min_genes=9999, max_pct_mt=0.1)
        
        assert "QC filters applied" in result.message
        
        # Check that many cells were removed
        dataset_summary = result.state_delta["dataset_summary"]
        assert dataset_summary["n_cells_after_qc"] < n_cells_before
        
        # Test very lenient thresholds (should keep most/all cells)
        result2 = apply_qc_filters(state, min_genes=1, max_pct_mt=100.0)
        
        assert "QC filters applied" in result2.message
        
        # Should retain most cells with lenient thresholds
        dataset_summary2 = result2.state_delta["dataset_summary"]
        assert dataset_summary2["n_cells_after_qc"] >= dataset_summary["n_cells_after_qc"]
    
    def test_filter_statistics_accuracy(self):
        """Test that filter statistics are calculated correctly."""
        try:
            from scqc_agent.tools.qc import apply_qc_filters
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Store original counts
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        # Create state
        state = SessionState(run_id="test_run")
        state.adata_path = str(data_path)
        
        # Apply filters
        result = apply_qc_filters(state, min_genes=200, max_pct_mt=20.0)
        
        # Get filter statistics
        json_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_filters.json"):
                json_file = artifact
                break
        
        assert json_file is not None
        
        with open(json_file, 'r') as f:
            filter_data = json.load(f)
        
        # Verify statistics consistency
        before = filter_data["before_filtering"]
        after = filter_data["after_filtering"]
        removed = filter_data["removed"]
        retained = filter_data["retained_fraction"]
        
        assert before["n_cells"] == n_cells_before
        assert before["n_genes"] == n_genes_before
        
        # Check arithmetic consistency
        assert before["n_cells"] - removed["n_cells"] == after["n_cells"]
        assert before["n_genes"] - removed["n_genes"] == after["n_genes"]
        
        # Check retained fractions
        expected_cell_fraction = after["n_cells"] / before["n_cells"]
        expected_gene_fraction = after["n_genes"] / before["n_genes"]
        
        assert abs(retained["cells"] - expected_cell_fraction) < 1e-10
        assert abs(retained["genes"] - expected_gene_fraction) < 1e-10


