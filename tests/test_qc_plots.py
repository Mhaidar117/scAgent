"""Tests for QC plotting functionality."""

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
    
    # Create synthetic data (300 cells, 800 genes)
    n_obs, n_vars = 300, 800
    
    # Generate counts matrix
    np.random.seed(42)
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create gene names with mitochondrial genes
    gene_names = [f"Gene_{i:04d}" for i in range(n_vars-20)]
    mt_genes = [f"mt-gene{i}" for i in range(20)]  # 20 MT genes
    gene_names.extend(mt_genes)
    
    # Create cell barcodes
    cell_names = [f"Cell_{i:04d}" for i in range(n_obs)]
    
    # Create AnnData object
    adata = ad.AnnData(X, obs=pd.DataFrame(index=cell_names), var=pd.DataFrame(index=gene_names))
    
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


class TestQCPlots:
    """Test QC plotting functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plot_qc_no_scanpy(self):
        """Test QC plotting when scanpy is not available."""
        from scqc_agent.tools.plots.qc import SCANPY_AVAILABLE, plot_qc

        if SCANPY_AVAILABLE:
            pytest.skip("Scanpy is available, cannot test fallback behavior")

        state = SessionState(run_id="test_run")
        state.adata_path = "test.h5ad"
        result = plot_qc(state)

        assert "Scanpy not available" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0
    
    def test_plot_qc_no_matplotlib(self):
        """Test QC plotting when matplotlib is not available."""
        # This test is tricky since matplotlib is likely available
        # We'll skip it if we can't mock the import
        pytest.skip("Matplotlib availability testing requires import mocking")
    
    def test_plot_qc_no_qc_metrics(self):
        """Test QC plotting when QC metrics haven't been computed."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
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
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path)
        )
        
        # Try to plot
        result = plot_qc(state)
        
        assert "QC metrics not found" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0
    
    def test_plot_qc_pre_stage_success(self):
        """Test successful QC plotting for pre-filtering stage."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
            import scanpy as sc
            import matplotlib
        except ImportError:
            pytest.skip("Required packages not available for QC plot tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path),
            config={"random_seed": 42}
        )
        
        # Generate plots for pre-filtering stage
        result = plot_qc(state, stage="pre")
        
        # Verify result
        assert "Generated QC plots for pre filtering" in result.message
        assert "300" in result.message  # number of cells
        assert "800" in result.message  # number of genes
        
        # Check artifacts were created
        assert len(result.artifacts) >= 3  # violin plot, scatter plot, summary JSON
        
        # Verify files exist and are not empty
        violin_plot = None
        scatter_plot = None
        summary_json = None
        
        for artifact in result.artifacts:
            artifact_path = Path(artifact)
            assert artifact_path.exists()
            assert artifact_path.stat().st_size > 0  # File is not empty
            
            if "violins.png" in str(artifact):
                violin_plot = artifact
            elif "scatter.png" in str(artifact):
                scatter_plot = artifact
            elif "plot_summary.json" in str(artifact):
                summary_json = artifact
        
        # Verify specific plot files exist
        assert violin_plot is not None
        assert scatter_plot is not None
        assert summary_json is not None
        
        # Check JSON summary content
        with open(summary_json, 'r') as f:
            plot_summary = json.load(f)
        
        assert plot_summary["stage"] == "pre"
        assert plot_summary["n_cells"] == 300
        assert plot_summary["n_genes"] == 800
        assert "statistics" in plot_summary
        assert "correlation_counts_genes" in plot_summary
        
        # Verify statistics structure
        stats = plot_summary["statistics"]
        required_metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        for metric in required_metrics:
            assert metric in stats
            assert "mean" in stats[metric]
            assert "median" in stats[metric]
            assert "min" in stats[metric]
            assert "max" in stats[metric]
        
        # Check citations
        assert len(result.citations) > 0
    
    def test_plot_qc_post_stage_success(self):
        """Test successful QC plotting for post-filtering stage."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
            import scanpy as sc
            import matplotlib
        except ImportError:
            pytest.skip("Required packages not available for QC plot tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path),
            config={"random_seed": 42}
        )
        
        # Generate plots for post-filtering stage
        result = plot_qc(state, stage="post")
        
        # Verify result
        assert "Generated QC plots for post filtering" in result.message
        
        # Check that post-filtering directory structure is used
        post_artifacts = [a for a in result.artifacts if "step_06_apply_qc" in str(a)]
        assert len(post_artifacts) > 0
        
        # Verify post-stage files
        for artifact in result.artifacts:
            if "qc_post_" in str(artifact):
                assert "violins.png" in str(artifact) or "scatter.png" in str(artifact)
    
    def test_plot_file_formats_and_sizes(self):
        """Test that plot files are generated in correct formats and sizes."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
            import scanpy as sc
            import matplotlib
            from PIL import Image
        except ImportError:
            pytest.skip("Required packages not available for QC plot tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path)
        )
        
        # Generate plots
        result = plot_qc(state, stage="pre")
        
        # Find PNG files
        png_files = [a for a in result.artifacts if str(a).endswith('.png')]
        assert len(png_files) >= 2  # violin and scatter plots
        
        # Check that PNG files can be opened
        for png_file in png_files:
            try:
                with Image.open(png_file) as img:
                    width, height = img.size
                    assert width > 0 and height > 0
                    assert img.format == 'PNG'
            except ImportError:
                # PIL not available, just check file size
                assert Path(png_file).stat().st_size > 1000  # At least 1KB
    
    def test_plot_qc_deterministic_output(self):
        """Test that plots are deterministic with same random seed."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
            import scanpy as sc
            import matplotlib
        except ImportError:
            pytest.skip("Required packages not available for QC plot tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state with fixed random seed
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path),
            config={"random_seed": 12345}
        )
        
        # Generate plots twice
        result1 = plot_qc(state, stage="pre")
        
        # Create a new run directory for second test
        state.run_dir = str(self.temp_path / "run2")
        result2 = plot_qc(state, stage="pre")
        
        # Both should succeed
        assert "Generated QC plots" in result1.message
        assert "Generated QC plots" in result2.message
        
        # Find summary JSON files
        json1 = None
        json2 = None
        
        for artifact in result1.artifacts:
            if "plot_summary.json" in str(artifact):
                json1 = artifact
                break
        
        for artifact in result2.artifacts:
            if "plot_summary.json" in str(artifact):
                json2 = artifact
                break
        
        assert json1 is not None and json2 is not None
        
        # Load and compare summary statistics
        with open(json1, 'r') as f:
            summary1 = json.load(f)
        with open(json2, 'r') as f:
            summary2 = json.load(f)
        
        # Statistics should be identical (same data, same seed)
        assert summary1["n_cells"] == summary2["n_cells"]
        assert summary1["n_genes"] == summary2["n_genes"]
        assert abs(summary1["correlation_counts_genes"] - summary2["correlation_counts_genes"]) < 1e-10
    
    def test_plot_directory_structure(self):
        """Test that plots are saved in correct directory structure."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
            import scanpy as sc
            import matplotlib
        except ImportError:
            pytest.skip("Required packages not available for QC plot tests")
        
        # Create and save data with QC metrics
        adata = create_synthetic_adata_with_qc()
        data_path = self.temp_path / "test_data.h5ad"
        adata.write_h5ad(data_path)
        
        # Create state
        state = SessionState(
            adata_path=str(data_path),
            run_dir=str(self.temp_path / "my_run")
        )
        
        # Test pre-filtering plots
        result_pre = plot_qc(state, stage="pre")
        
        # Check pre-filtering directory structure
        for artifact in result_pre.artifacts:
            artifact_path = str(artifact)
            assert "my_run" in artifact_path
            assert "step_05_plot_qc_pre" in artifact_path
        
        # Test post-filtering plots
        result_post = plot_qc(state, stage="post")
        
        # Check post-filtering directory structure
        for artifact in result_post.artifacts:
            artifact_path = str(artifact)
            assert "my_run" in artifact_path
            assert "step_06_apply_qc" in artifact_path
    
    def test_plot_error_handling(self):
        """Test error handling in plot generation."""
        try:
            from scqc_agent.tools.plots.qc import plot_qc
            import scanpy as sc
        except ImportError:
            pytest.skip("Scanpy not available for QC tests")
        
        # Create state pointing to non-existent file
        state = SessionState(
            adata_path=str(self.temp_path / "nonexistent.h5ad"),
            run_dir=str(self.temp_path)
        )
        
        # Try to plot
        result = plot_qc(state)
        
        # Should handle error gracefully
        assert "Error" in result.message
        assert result.state_delta == {}
        assert len(result.artifacts) == 0


