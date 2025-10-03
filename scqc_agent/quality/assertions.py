"""Quality gates and assertions for scQC Agent workflows.

This module provides runtime assertions to validate data integrity and ensure
workflow reliability. Quality gates check invariants after each tool execution
and fail fast with clear error messages when violations are detected.
"""

import numpy as np
from typing import Any, Optional, Union
from pathlib import Path


class QualityGateError(Exception):
    """Exception raised when a quality gate assertion fails."""
    pass


def assert_qc_fields_present(adata: Any) -> None:
    """Assert that required QC fields are present in AnnData object.
    
    Checks for essential QC metrics computed by scanpy.pp.calculate_qc_metrics:
    - n_genes_by_counts: Number of genes detected per cell
    - total_counts: Total UMI counts per cell
    - pct_counts_mt: Percentage of mitochondrial gene counts
    
    Args:
        adata: AnnData object to validate
        
    Raises:
        QualityGateError: If required QC fields are missing
    """
    required_obs_fields = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
    missing_fields = []
    
    for field in required_obs_fields:
        if field not in adata.obs.columns:
            missing_fields.append(field)
    
    if missing_fields:
        raise QualityGateError(
            f"Missing required QC fields in adata.obs: {missing_fields}. "
            f"Run compute_qc_metrics first."
        )


def assert_pct_mt_range(adata: Any, min_val: float = 0.0, max_val: float = 100.0) -> None:
    """Assert that mitochondrial percentage values are within valid range.
    
    Validates that pct_counts_mt values are reasonable percentages.
    
    Args:
        adata: AnnData object to validate
        min_val: Minimum allowed percentage (default: 0.0)
        max_val: Maximum allowed percentage (default: 100.0)
        
    Raises:
        QualityGateError: If pct_counts_mt values are outside valid range
    """
    if 'pct_counts_mt' not in adata.obs.columns:
        raise QualityGateError("pct_counts_mt field not found. Run compute_qc_metrics first.")
    
    pct_mt = adata.obs['pct_counts_mt']
    
    # Check for invalid values
    if pct_mt.isnull().any():
        raise QualityGateError("pct_counts_mt contains null values")
    
    min_observed = pct_mt.min()
    max_observed = pct_mt.max()
    
    if min_observed < min_val:
        raise QualityGateError(
            f"pct_counts_mt minimum value {min_observed:.2f} is below {min_val}"
        )
    
    if max_observed > max_val:
        raise QualityGateError(
            f"pct_counts_mt maximum value {max_observed:.2f} exceeds {max_val}"
        )
    
    # Check for suspicious values (>90% often indicates technical issues)
    high_mt_count = (pct_mt > 90).sum()
    if high_mt_count > 0:
        pct_high = (high_mt_count / len(pct_mt)) * 100
        if pct_high > 10:  # More than 10% of cells have >90% MT
            raise QualityGateError(
                f"{high_mt_count} cells ({pct_high:.1f}%) have >90% mitochondrial content. "
                f"This may indicate technical issues or dying cells."
            )


def assert_neighbors_nonempty(adata: Any) -> None:
    """Assert that neighbors graph has been computed and is non-empty.
    
    Validates that scanpy.pp.neighbors has been run successfully by checking
    for the presence of connectivities matrix and distances.
    
    Args:
        adata: AnnData object to validate
        
    Raises:
        QualityGateError: If neighbors graph is missing or empty
    """
    if 'neighbors' not in adata.uns:
        raise QualityGateError(
            "Neighbors graph not found in adata.uns. Run scanpy.pp.neighbors first."
        )
    
    neighbors_dict = adata.uns['neighbors']
    
    # Check for connectivities matrix
    if 'connectivities' not in adata.obsp:
        raise QualityGateError(
            "Connectivities matrix not found in adata.obsp. "
            "Neighbors computation may have failed."
        )
    
    connectivities = adata.obsp['connectivities']
    
    # Check that matrix is non-empty and has proper dimensions
    n_obs = adata.n_obs
    if connectivities.shape != (n_obs, n_obs):
        raise QualityGateError(
            f"Connectivities matrix shape {connectivities.shape} doesn't match "
            f"number of observations {n_obs}"
        )
    
    # Check for completely empty graph
    if connectivities.nnz == 0:
        raise QualityGateError(
            "Neighbors graph is empty (no connections found). "
            "Check data quality and neighbors parameters."
        )
    
    # Check for reasonable connectivity (each cell should have some neighbors)
    connections_per_cell = np.array(connectivities.sum(axis=1)).flatten()
    disconnected_cells = (connections_per_cell == 0).sum()
    
    if disconnected_cells > 0:
        pct_disconnected = (disconnected_cells / n_obs) * 100
        if pct_disconnected > 5:  # More than 5% disconnected is suspicious
            raise QualityGateError(
                f"{disconnected_cells} cells ({pct_disconnected:.1f}%) have no neighbors. "
                f"Consider adjusting neighbors parameters."
            )


def assert_latent_shape(adata: Any, key: str, expected_dims: Optional[int] = None) -> None:
    """Assert that latent representation has expected shape and properties.
    
    Validates dimensional reduction results (PCA, scVI, etc.) for consistency.
    
    Args:
        adata: AnnData object to validate
        key: Key for the latent representation (e.g., 'X_pca', 'X_scVI')
        expected_dims: Expected number of dimensions (optional)
        
    Raises:
        QualityGateError: If latent representation is invalid
    """
    if key not in adata.obsm:
        raise QualityGateError(
            f"Latent representation '{key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    
    latent = adata.obsm[key]
    n_obs = adata.n_obs
    
    # Check shape consistency
    if latent.shape[0] != n_obs:
        raise QualityGateError(
            f"Latent representation '{key}' has {latent.shape[0]} rows "
            f"but adata has {n_obs} observations"
        )
    
    # Check for expected dimensions
    n_dims = latent.shape[1]
    if expected_dims is not None and n_dims != expected_dims:
        raise QualityGateError(
            f"Latent representation '{key}' has {n_dims} dimensions "
            f"but expected {expected_dims}"
        )
    
    # Check for invalid values
    if np.isnan(latent).any():
        nan_count = np.isnan(latent).sum()
        raise QualityGateError(
            f"Latent representation '{key}' contains {nan_count} NaN values"
        )
    
    if np.isinf(latent).any():
        inf_count = np.isinf(latent).sum()
        raise QualityGateError(
            f"Latent representation '{key}' contains {inf_count} infinite values"
        )
    
    # Check for degenerate cases (all zeros, no variance)
    if np.allclose(latent, 0):
        raise QualityGateError(
            f"Latent representation '{key}' contains only zero values"
        )
    
    # Check variance across dimensions
    dim_vars = np.var(latent, axis=0)
    zero_var_dims = (dim_vars < 1e-10).sum()
    
    if zero_var_dims > 0:
        pct_zero_var = (zero_var_dims / n_dims) * 100
        if pct_zero_var > 20:  # More than 20% of dimensions have no variance
            raise QualityGateError(
                f"Latent representation '{key}' has {zero_var_dims} dimensions "
                f"({pct_zero_var:.1f}%) with near-zero variance"
            )


def assert_clustering_quality(adata: Any, key: str = 'leiden', min_clusters: int = 2, 
                             max_clusters: Optional[int] = None) -> None:
    """Assert that clustering results are reasonable.
    
    Validates clustering outputs for basic quality criteria.
    
    Args:
        adata: AnnData object to validate
        key: Clustering key in adata.obs (default: 'leiden')
        min_clusters: Minimum expected number of clusters
        max_clusters: Maximum expected number of clusters (optional)
        
    Raises:
        QualityGateError: If clustering results are invalid
    """
    if key not in adata.obs.columns:
        raise QualityGateError(
            f"Clustering key '{key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )
    
    clusters = adata.obs[key]
    unique_clusters = clusters.nunique()
    
    # Check minimum clusters
    if unique_clusters < min_clusters:
        raise QualityGateError(
            f"Only {unique_clusters} clusters found, expected at least {min_clusters}"
        )
    
    # Check maximum clusters
    if max_clusters is not None and unique_clusters > max_clusters:
        raise QualityGateError(
            f"{unique_clusters} clusters found, expected at most {max_clusters}"
        )
    
    # Check for empty clusters (shouldn't happen but worth checking)
    cluster_counts = clusters.value_counts()
    min_cluster_size = cluster_counts.min()
    
    if min_cluster_size == 0:
        raise QualityGateError(f"Empty clusters found in '{key}'")
    
    # Check for overly small clusters (potential overclustering)
    tiny_clusters = (cluster_counts < 5).sum()
    if tiny_clusters > 0:
        pct_tiny = (tiny_clusters / unique_clusters) * 100
        if pct_tiny > 30:  # More than 30% of clusters are very small
            raise QualityGateError(
                f"{tiny_clusters} clusters ({pct_tiny:.1f}%) have <5 cells. "
                f"Consider reducing clustering resolution."
            )


def assert_file_exists(filepath: Union[str, Path], description: str = "File") -> None:
    """Assert that a file exists at the given path.
    
    Args:
        filepath: Path to the file
        description: Description of the file for error messages
        
    Raises:
        QualityGateError: If file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise QualityGateError(f"{description} not found at {filepath}")
    
    if not path.is_file():
        raise QualityGateError(f"{description} at {filepath} is not a file")
