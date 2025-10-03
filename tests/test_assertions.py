"""Tests for quality gates and assertions."""

import pytest
import numpy as np
from unittest.mock import Mock

from scqc_agent.quality.assertions import (
    assert_qc_fields_present,
    assert_pct_mt_range,
    assert_neighbors_nonempty,
    assert_latent_shape,
    assert_clustering_quality,
    assert_file_exists,
    QualityGateError,
)

# Skip tests that require scanpy if not available
pytest.importorskip("scanpy")
pytest.importorskip("anndata")

import scanpy as sc
import anndata as ad


@pytest.fixture
def valid_adata():
    """Create a valid AnnData object with QC metrics."""
    np.random.seed(42)
    
    # Create count matrix
    n_cells, n_genes = 100, 50
    X = np.random.poisson(5, (n_cells, n_genes)).astype(float)
    
    # Create AnnData
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    
    # Add QC metrics
    adata.obs['n_genes_by_counts'] = (X > 0).sum(axis=1)
    adata.obs['total_counts'] = X.sum(axis=1)
    adata.obs['pct_counts_mt'] = np.random.uniform(5, 15, n_cells)
    
    return adata


@pytest.fixture
def adata_with_neighbors(valid_adata):
    """Create AnnData with computed neighbors."""
    adata = valid_adata.copy()
    
    # Add PCA
    sc.tl.pca(adata, n_comps=10)
    
    # Add neighbors
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
    
    return adata


@pytest.fixture
def adata_with_clustering(adata_with_neighbors):
    """Create AnnData with clustering."""
    adata = adata_with_neighbors.copy()
    
    # Add clustering
    sc.tl.leiden(adata, resolution=0.5)
    
    return adata


class TestQCFieldsPresent:
    """Test assert_qc_fields_present function."""
    
    def test_valid_qc_fields(self, valid_adata):
        """Test with valid QC fields."""
        # Should not raise
        assert_qc_fields_present(valid_adata)
        
    def test_missing_n_genes_by_counts(self, valid_adata):
        """Test with missing n_genes_by_counts field."""
        del valid_adata.obs['n_genes_by_counts']
        
        with pytest.raises(QualityGateError, match="Missing required QC fields"):
            assert_qc_fields_present(valid_adata)
    
    def test_missing_total_counts(self, valid_adata):
        """Test with missing total_counts field."""
        del valid_adata.obs['total_counts']
        
        with pytest.raises(QualityGateError, match="Missing required QC fields"):
            assert_qc_fields_present(valid_adata)
    
    def test_missing_pct_counts_mt(self, valid_adata):
        """Test with missing pct_counts_mt field."""
        del valid_adata.obs['pct_counts_mt']
        
        with pytest.raises(QualityGateError, match="Missing required QC fields"):
            assert_qc_fields_present(valid_adata)
    
    def test_missing_multiple_fields(self, valid_adata):
        """Test with multiple missing fields."""
        del valid_adata.obs['n_genes_by_counts']
        del valid_adata.obs['pct_counts_mt']
        
        with pytest.raises(QualityGateError) as exc_info:
            assert_qc_fields_present(valid_adata)
        
        assert "n_genes_by_counts" in str(exc_info.value)
        assert "pct_counts_mt" in str(exc_info.value)


class TestPctMtRange:
    """Test assert_pct_mt_range function."""
    
    def test_valid_mt_range(self, valid_adata):
        """Test with valid mitochondrial percentages."""
        # Should not raise
        assert_pct_mt_range(valid_adata)
    
    def test_missing_pct_counts_mt(self, valid_adata):
        """Test with missing pct_counts_mt field."""
        del valid_adata.obs['pct_counts_mt']
        
        with pytest.raises(QualityGateError, match="pct_counts_mt field not found"):
            assert_pct_mt_range(valid_adata)
    
    def test_null_values(self, valid_adata):
        """Test with null values in pct_counts_mt."""
        valid_adata.obs.loc[valid_adata.obs.index[0], 'pct_counts_mt'] = np.nan
        
        with pytest.raises(QualityGateError, match="contains null values"):
            assert_pct_mt_range(valid_adata)
    
    def test_negative_values(self, valid_adata):
        """Test with negative mitochondrial percentages."""
        valid_adata.obs['pct_counts_mt'] = np.random.uniform(-5, 10, len(valid_adata))
        
        with pytest.raises(QualityGateError, match="minimum value .* is below"):
            assert_pct_mt_range(valid_adata)
    
    def test_excessive_values(self, valid_adata):
        """Test with excessive mitochondrial percentages."""
        valid_adata.obs['pct_counts_mt'] = np.random.uniform(95, 105, len(valid_adata))
        
        with pytest.raises(QualityGateError, match="maximum value .* exceeds"):
            assert_pct_mt_range(valid_adata)
    
    def test_high_mt_content_warning(self, valid_adata):
        """Test warning for high mitochondrial content."""
        # Set 20% of cells to have >90% MT content
        n_high = int(0.2 * len(valid_adata))
        valid_adata.obs['pct_counts_mt'] = np.concatenate([
            np.random.uniform(92, 98, n_high),
            np.random.uniform(5, 15, len(valid_adata) - n_high)
        ])
        
        with pytest.raises(QualityGateError, match="have >90% mitochondrial content"):
            assert_pct_mt_range(valid_adata)
    
    def test_custom_range(self, valid_adata):
        """Test with custom min/max values."""
        valid_adata.obs['pct_counts_mt'] = np.random.uniform(20, 30, len(valid_adata))
        
        # Should pass with custom range
        assert_pct_mt_range(valid_adata, min_val=15, max_val=35)
        
        # Should fail with default range
        with pytest.raises(QualityGateError):
            assert_pct_mt_range(valid_adata, min_val=0, max_val=25)


class TestNeighborsNonempty:
    """Test assert_neighbors_nonempty function."""
    
    def test_valid_neighbors(self, adata_with_neighbors):
        """Test with valid neighbors graph."""
        # Should not raise
        assert_neighbors_nonempty(adata_with_neighbors)
    
    def test_missing_neighbors_uns(self, valid_adata):
        """Test with missing neighbors in uns."""
        with pytest.raises(QualityGateError, match="Neighbors graph not found"):
            assert_neighbors_nonempty(valid_adata)
    
    def test_missing_connectivities_obsp(self, valid_adata):
        """Test with neighbors in uns but missing connectivities."""
        valid_adata.uns['neighbors'] = {'params': {}}
        
        with pytest.raises(QualityGateError, match="Connectivities matrix not found"):
            assert_neighbors_nonempty(valid_adata)
    
    def test_wrong_connectivities_shape(self, adata_with_neighbors):
        """Test with wrong connectivities matrix shape."""
        # Create wrong-sized matrix
        from scipy.sparse import csr_matrix
        wrong_matrix = csr_matrix((50, 50))  # Should be 100x100
        adata_with_neighbors.obsp['connectivities'] = wrong_matrix
        
        with pytest.raises(QualityGateError, match="shape .* doesn't match"):
            assert_neighbors_nonempty(adata_with_neighbors)
    
    def test_empty_connectivities(self, adata_with_neighbors):
        """Test with empty connectivities matrix."""
        from scipy.sparse import csr_matrix
        n_obs = len(adata_with_neighbors)
        empty_matrix = csr_matrix((n_obs, n_obs))
        adata_with_neighbors.obsp['connectivities'] = empty_matrix
        
        with pytest.raises(QualityGateError, match="is empty"):
            assert_neighbors_nonempty(adata_with_neighbors)
    
    def test_many_disconnected_cells(self, adata_with_neighbors):
        """Test with many disconnected cells."""
        from scipy.sparse import csr_matrix
        import numpy as np
        
        n_obs = len(adata_with_neighbors)
        # Create matrix where 10% of cells have no connections
        data = np.random.random(n_obs * 10)
        row = np.repeat(range(int(0.9 * n_obs)), 10)  # Only first 90% have connections
        col = np.random.randint(0, n_obs, len(data))
        
        sparse_matrix = csr_matrix((data, (row, col)), shape=(n_obs, n_obs))
        adata_with_neighbors.obsp['connectivities'] = sparse_matrix
        
        with pytest.raises(QualityGateError, match="have no neighbors"):
            assert_neighbors_nonempty(adata_with_neighbors)


class TestLatentShape:
    """Test assert_latent_shape function."""
    
    def test_valid_latent_pca(self, adata_with_neighbors):
        """Test with valid PCA latent representation."""
        assert_latent_shape(adata_with_neighbors, 'X_pca', expected_dims=10)
    
    def test_missing_latent_key(self, valid_adata):
        """Test with missing latent representation key."""
        with pytest.raises(QualityGateError, match="not found in adata.obsm"):
            assert_latent_shape(valid_adata, 'X_missing')
    
    def test_wrong_number_rows(self, adata_with_neighbors):
        """Test with wrong number of rows in latent representation."""
        # Manually set wrong-sized matrix
        wrong_latent = np.random.random((50, 10))  # Should be 100x10
        adata_with_neighbors.obsm['X_wrong'] = wrong_latent
        
        with pytest.raises(QualityGateError, match="has .* rows but adata has"):
            assert_latent_shape(adata_with_neighbors, 'X_wrong')
    
    def test_wrong_dimensions(self, adata_with_neighbors):
        """Test with wrong number of dimensions."""
        with pytest.raises(QualityGateError, match="has .* dimensions but expected"):
            assert_latent_shape(adata_with_neighbors, 'X_pca', expected_dims=20)
    
    def test_nan_values(self, adata_with_neighbors):
        """Test with NaN values in latent representation."""
        latent = adata_with_neighbors.obsm['X_pca'].copy()
        latent[0, 0] = np.nan
        adata_with_neighbors.obsm['X_nan'] = latent
        
        with pytest.raises(QualityGateError, match="contains .* NaN values"):
            assert_latent_shape(adata_with_neighbors, 'X_nan')
    
    def test_infinite_values(self, adata_with_neighbors):
        """Test with infinite values in latent representation."""
        latent = adata_with_neighbors.obsm['X_pca'].copy()
        latent[0, 0] = np.inf
        adata_with_neighbors.obsm['X_inf'] = latent
        
        with pytest.raises(QualityGateError, match="contains .* infinite values"):
            assert_latent_shape(adata_with_neighbors, 'X_inf')
    
    def test_all_zeros(self, valid_adata):
        """Test with all-zero latent representation."""
        zero_latent = np.zeros((len(valid_adata), 10))
        valid_adata.obsm['X_zeros'] = zero_latent
        
        with pytest.raises(QualityGateError, match="contains only zero values"):
            assert_latent_shape(valid_adata, 'X_zeros')
    
    def test_low_variance_dimensions(self, valid_adata):
        """Test with low variance dimensions."""
        # Create latent with some zero-variance dimensions
        n_obs = len(valid_adata)
        latent = np.random.random((n_obs, 10))
        latent[:, :3] = 0  # First 3 dimensions have no variance
        valid_adata.obsm['X_low_var'] = latent
        
        with pytest.raises(QualityGateError, match="dimensions .* with near-zero variance"):
            assert_latent_shape(valid_adata, 'X_low_var')


class TestClusteringQuality:
    """Test assert_clustering_quality function."""
    
    def test_valid_clustering(self, adata_with_clustering):
        """Test with valid clustering."""
        assert_clustering_quality(adata_with_clustering, key='leiden', min_clusters=2)
    
    def test_missing_clustering_key(self, valid_adata):
        """Test with missing clustering key."""
        with pytest.raises(QualityGateError, match="not found in adata.obs"):
            assert_clustering_quality(valid_adata, key='leiden')
    
    def test_too_few_clusters(self, adata_with_clustering):
        """Test with too few clusters."""
        # Set all cells to same cluster
        adata_with_clustering.obs['leiden'] = '0'
        
        with pytest.raises(QualityGateError, match="Only .* clusters found"):
            assert_clustering_quality(adata_with_clustering, key='leiden', min_clusters=2)
    
    def test_too_many_clusters(self, adata_with_clustering):
        """Test with too many clusters."""
        # Set each cell to its own cluster
        adata_with_clustering.obs['leiden'] = [str(i) for i in range(len(adata_with_clustering))]
        
        with pytest.raises(QualityGateError, match="clusters found, expected at most"):
            assert_clustering_quality(adata_with_clustering, key='leiden', max_clusters=10)
    
    def test_many_tiny_clusters(self, adata_with_clustering):
        """Test with many very small clusters."""
        # Create many clusters with 1-2 cells each
        n_clusters = 40
        clusters = []
        for i in range(len(adata_with_clustering)):
            clusters.append(str(i % n_clusters))
        adata_with_clustering.obs['leiden'] = clusters
        
        with pytest.raises(QualityGateError, match="clusters .* have <5 cells"):
            assert_clustering_quality(adata_with_clustering, key='leiden')


class TestFileExists:
    """Test assert_file_exists function."""
    
    def test_existing_file(self, tmp_path):
        """Test with existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Should not raise
        assert_file_exists(test_file, "Test file")
    
    def test_missing_file(self, tmp_path):
        """Test with missing file."""
        missing_file = tmp_path / "missing.txt"
        
        with pytest.raises(QualityGateError, match="not found"):
            assert_file_exists(missing_file, "Missing file")
    
    def test_directory_not_file(self, tmp_path):
        """Test with directory instead of file."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        with pytest.raises(QualityGateError, match="is not a file"):
            assert_file_exists(test_dir, "Test directory")


class TestQualityGateError:
    """Test QualityGateError exception."""
    
    def test_quality_gate_error_inheritance(self):
        """Test that QualityGateError inherits from Exception."""
        error = QualityGateError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"
    
    def test_quality_gate_error_message(self):
        """Test QualityGateError message handling."""
        message = "Custom error message"
        error = QualityGateError(message)
        assert str(error) == message


if __name__ == "__main__":
    pytest.main([__file__])
