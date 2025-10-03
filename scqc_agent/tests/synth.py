"""Synthetic dataset generator for testing scQC Agent."""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional

try:
    import scanpy as sc
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False


def make_synth_adata(
    n_cells: int = 600,
    n_genes: int = 1500,
    n_batches: int = 2,
    mito_frac: float = 0.08,
    random_seed: int = 42
) -> 'ad.AnnData':
    """Generate a small synthetic AnnData object for testing.
    
    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes to generate  
        n_batches: Number of batches (for SampleID)
        mito_frac: Fraction of genes to mark as mitochondrial
        random_seed: Random seed for reproducibility
        
    Returns:
        AnnData object with synthetic single-cell data
        
    Raises:
        ImportError: If scanpy/anndata not available
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("scanpy and anndata are required for synthetic data generation")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Generate gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Mark a subset as mitochondrial genes (mouse-style mt- prefix)
    n_mito = int(n_genes * mito_frac)
    mito_indices = np.random.choice(n_genes, n_mito, replace=False)
    for idx in mito_indices:
        gene_names[idx] = f"mt-Gene_{idx:04d}"
    
    # Generate cell barcodes
    cell_names = [f"Cell_{i:06d}" for i in range(n_cells)]
    
    # Generate overdispersed count matrix
    # Use negative binomial to simulate realistic scRNA-seq counts
    mean_counts = np.random.gamma(2, 500, n_genes)  # Mean counts per gene
    dispersion = np.random.gamma(1, 0.1, n_genes)   # Overdispersion parameter
    
    # Generate count matrix
    X = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        for j in range(n_genes):
            # Add cell-specific scaling factor
            cell_factor = np.random.gamma(1, 1)
            adjusted_mean = mean_counts[j] * cell_factor
            
            # Generate counts from negative binomial
            p = adjusted_mean / (adjusted_mean + dispersion[j])
            n = dispersion[j]
            X[i, j] = np.random.negative_binomial(n, 1-p)
    
    # Make mitochondrial genes have higher expression (realistic pattern)
    for idx in mito_indices:
        X[:, idx] *= np.random.uniform(1.5, 3.0)
    
    # Convert to sparse matrix for efficiency
    X_sparse = sparse.csr_matrix(X)
    
    # Create observation metadata
    obs_data = {
        'SampleID': [f"Batch_{i % n_batches}" for i in range(n_cells)]
    }
    obs = pd.DataFrame(obs_data, index=cell_names)
    
    # Create variable metadata
    var_data = {}
    var = pd.DataFrame(var_data, index=gene_names)
    
    # Create AnnData object
    adata = ad.AnnData(
        X=X_sparse,
        obs=obs,
        var=var
    )
    
    # Add some realistic metadata
    adata.uns['source'] = 'synthetic'
    adata.uns['creation_date'] = pd.Timestamp.now().isoformat()
    adata.uns['n_mito_genes'] = n_mito
    
    return adata


def make_synth_adata_with_issues(
    n_cells: int = 600,
    n_genes: int = 1500,
    random_seed: int = 42
) -> 'ad.AnnData':
    """Generate synthetic data with common QC issues for testing filters.
    
    Creates data with:
    - Some cells with very few genes detected
    - Some cells with very high mitochondrial content
    - Some genes expressed in very few cells
    
    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        AnnData object with intentional QC issues
    """
    # Start with normal synthetic data
    adata = make_synth_adata(n_cells, n_genes, random_seed=random_seed)
    
    np.random.seed(random_seed + 1)  # Different seed for modifications
    
    # Make some cells have very few genes (simulate low-quality cells)
    low_quality_cells = np.random.choice(n_cells, n_cells // 10, replace=False)
    for cell_idx in low_quality_cells:
        # Zero out most genes in these cells
        genes_to_zero = np.random.choice(n_genes, int(n_genes * 0.8), replace=False)
        adata.X[cell_idx, genes_to_zero] = 0
    
    # Make some cells have very high mitochondrial content
    mito_mask = [name.startswith('mt-') for name in adata.var_names]
    high_mito_cells = np.random.choice(n_cells, n_cells // 15, replace=False)
    for cell_idx in high_mito_cells:
        # Boost mitochondrial gene expression
        adata.X[cell_idx, mito_mask] *= np.random.uniform(5, 10)
    
    # Make some genes very rare (expressed in < 3 cells)
    rare_genes = np.random.choice(n_genes, n_genes // 20, replace=False)
    for gene_idx in rare_genes:
        # Zero out this gene in most cells
        cells_to_zero = np.random.choice(n_cells, n_cells - 2, replace=False)
        adata.X[cells_to_zero, gene_idx] = 0
    
    return adata


if __name__ == "__main__":
    # Quick test
    adata = make_synth_adata()
    print(f"Generated synthetic data: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Batch distribution: {adata.obs['SampleID'].value_counts().to_dict()}")
    
    # Count mitochondrial genes
    mito_genes = [name for name in adata.var_names if name.startswith('mt-')]
    print(f"Mitochondrial genes: {len(mito_genes)}")
