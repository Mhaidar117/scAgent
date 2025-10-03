"""Tests for QC metrics computation functionality."""

import numpy as np
import tempfile
from pathlib import Path

import pytest

from scqc_agent.state import SessionState, ToolResult

# Test synthetic data creation
def create_synthetic_adata():
    """Create synthetic AnnData for testing."""
    try:
        import scanpy as sc
        import anndata as ad
        import pandas as pd
    except ImportError:
        pytest.skip("Scanpy not available for QC tests")
    
    # Create synthetic data (800 cells, 1500 genes)
    n_obs, n_vars = 800, 1500
    
    # Generate counts matrix with realistic distribution
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create gene names with some mitochondrial genes
    gene_names = [f"Gene_{i:04d}" for i in range(n_vars-50)]
    mt_genes = [f"mt-{gene}" for gene in ["Atp6", "Cytb", "Nd1", "Nd2", "Nd3", "Nd4", "Nd5", "Nd6"]]
    mt_genes += [f"mt-gene{i}" for i in range(42)]  # Total 50 MT genes
    gene_names.extend(mt_genes)
    
    # Create cell barcodes
    cell_names = [f"Cell_{i:04d}" for i in range(n_obs)]
    
    # Create AnnData object
    adata = ad.AnnData(X, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
    
    # Add some batch information for testing
    adata.obs['batch'] = np.random.choice(['batch1', 'batch2', 'batch3'], n_obs)
    
    return adata


class TestQCMetrics:
    """Test QC metrics computation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compute_qc_metrics_no_scanpy(self):
        """Test QC metrics computation when scanpy is not available."""
        from scqc_agent.tools.qc import SCANPY_AVAILABLE, compute_qc_metrics
        
        if SCANPY_AVAILABLE:
            pytest.skip("Scanpy is available, cannot test fallback behavior")
        
        state = SessionState(adata_path="test.h5ad", run_dir=str(self.temp_path))
        result = compute_qc_metrics(state)
        
        assert "Scanpy not available" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0
    
    def test_compute_qc_metrics_no_data(self):
        """Test QC metrics computation with no data loaded."""
        try:
            from scqc_agent.tools.qc import compute_qc_metrics
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        state = SessionState(run_dir=str(self.temp_path))
        result = compute_qc_metrics(state)
        
        assert "No AnnData file loaded" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0
    
    def test_compute_qc_metrics_success(self):
        """Test successful QC metrics computation."""
        try:
            from scqc_agent.tools.qc import compute_qc_metrics
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save synthetic data
        adata = create_synthetic_adata()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path),
            config={"batch_key": "batch", "random_seed": 42}
        )
        
        # Compute QC metrics
        result = compute_qc_metrics(state, species="mouse")
        
        # Verify result
        assert "QC metrics computed" in result.message
        assert "800" in result.message  # number of cells
        assert "1500" in result.message  # number of genes
        
        # Check state delta
        assert "dataset_summary" in result.state_delta
        dataset_summary = result.state_delta["dataset_summary"]
        assert dataset_summary["qc_computed"] is True
        assert dataset_summary["species"] == "mouse"
        assert dataset_summary["n_mito_genes"] == 50
        
        # Check QC metrics structure
        qc_metrics = dataset_summary["qc_metrics"]
        required_metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        for metric in required_metrics:
            assert metric in qc_metrics
            assert "mean" in qc_metrics[metric]
            assert "median" in qc_metrics[metric]
            assert "std" in qc_metrics[metric]
        
        # Check artifacts were created
        assert len(result.artifacts) >= 2  # CSV and JSON summary files
        
        # Verify files exist
        for artifact in result.artifacts:
            assert Path(artifact).exists()
        
        # Check citations
        assert len(result.citations) > 0
        assert any("Luecken" in citation for citation in result.citations)
    
    def test_compute_qc_metrics_human_species(self):
        """Test QC metrics computation with human species."""
        try:
            from scqc_agent.tools.qc import compute_qc_metrics
            import scanpy as sc
            import anndata as ad
            import pandas as pd
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create synthetic data with human MT genes
        n_obs, n_vars = 500, 1000
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        # Create gene names with human mitochondrial genes
        gene_names = [f"Gene_{i:04d}" for i in range(n_vars-20)]
        mt_genes = [f"MT-{gene}" for gene in ["ATP6", "CYTB", "ND1", "ND2", "ND3"]]
        mt_genes += [f"MT-GENE{i}" for i in range(15)]  # Total 20 MT genes
        gene_names.extend(mt_genes)
        
        cell_names = [f"Cell_{i:04d}" for i in range(n_obs)]
        adata = ad.AnnData(X, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
        
        # Save data
        data_path = self.temp_path / "human_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path)
        )
        
        # Compute QC metrics with human species
        result = compute_qc_metrics(state, species="human")
        
        # Verify species detection
        assert "QC metrics computed" in result.message
        dataset_summary = result.state_delta["dataset_summary"]
        assert dataset_summary["species"] == "human"
        assert dataset_summary["n_mito_genes"] == 20
    
    def test_compute_qc_metrics_custom_prefix(self):
        """Test QC metrics computation with custom mitochondrial prefix."""
        try:
            from scqc_agent.tools.qc import compute_qc_metrics
            import scanpy as sc
            import anndata as ad
            import pandas as pd
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create synthetic data with custom prefix
        n_obs, n_vars = 300, 800
        np.random.seed(42)
        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
        
        # Create gene names with custom mitochondrial prefix
        gene_names = [f"Gene_{i:04d}" for i in range(n_vars-10)]
        mt_genes = [f"MITO-{i}" for i in range(10)]  # Custom prefix
        gene_names.extend(mt_genes)
        
        cell_names = [f"Cell_{i:04d}" for i in range(n_obs)]
        adata = ad.AnnData(X, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
        
        # Save data
        data_path = self.temp_path / "custom_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path)
        )
        
        # Compute QC metrics with custom prefix
        result = compute_qc_metrics(state, mito_prefix="MITO-")
        
        # Verify custom prefix was used
        assert "QC metrics computed" in result.message
        dataset_summary = result.state_delta["dataset_summary"]
        assert dataset_summary["species"] == "custom"
        assert dataset_summary["n_mito_genes"] == 10
    
    def test_qc_summary_files_content(self):
        """Test that QC summary files contain expected content."""
        try:
            from scqc_agent.tools.qc import compute_qc_metrics
            import scanpy as sc
            import json
            import pandas as pd
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create and save synthetic data
        adata = create_synthetic_adata()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state with batch information
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path),
            config={"batch_key": "batch"}
        )
        
        # Compute QC metrics
        result = compute_qc_metrics(state, species="mouse")
        
        # Find the JSON summary file
        json_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_summary.json"):
                json_file = artifact
                break
        
        assert json_file is not None
        
        # Check JSON content
        with open(json_file, 'r') as f:
            summary_data = json.load(f)
        
        # Verify JSON structure
        assert "timestamp" in summary_data
        assert "species" in summary_data
        assert "n_cells" in summary_data
        assert "n_genes" in summary_data
        assert "qc_metrics" in summary_data
        assert "per_batch" in summary_data  # Since we provided batch_key
        
        # Verify per-batch information
        per_batch = summary_data["per_batch"]
        assert len(per_batch) == 3  # batch1, batch2, batch3
        for batch_name, batch_data in per_batch.items():
            assert "n_cells" in batch_data
            assert "mean_genes" in batch_data
            assert "mean_counts" in batch_data
            assert "mean_pct_mt" in batch_data
        
        # Find and check CSV file
        csv_file = None
        for artifact in result.artifacts:
            if str(artifact).endswith("qc_summary.csv"):
                csv_file = artifact
                break
        
        assert csv_file is not None
        
        # Check CSV content
        csv_data = pd.read_csv(csv_file)
        assert len(csv_data) > 0
        assert "n_genes_by_counts" in csv_data.columns
        assert "total_counts" in csv_data.columns
        assert "pct_counts_mt" in csv_data.columns
