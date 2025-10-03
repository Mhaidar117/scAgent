---
title: "Graph Analysis and Clustering in scRNA-seq"
topic: "graph_analysis"
species: "general"
methods: ["pca", "umap", "leiden", "neighbors"]
---

# Graph Analysis and Clustering in scRNA-seq

## Overview

Graph-based analysis is fundamental to scRNA-seq workflows, enabling dimensionality reduction, visualization, and clustering. The typical workflow involves PCA → neighbor graph → UMAP → clustering.

## Principal Component Analysis (PCA)

### Purpose
- Reduce dimensionality from thousands of genes to manageable number
- Remove noise and technical variation
- Capture major sources of variation

### Key Parameters
- `n_comps`: Number of components (default: 50)
  - Increase for complex datasets
  - Check elbow plot for optimal number
  - Balance between information and noise

- `svd_solver`: Algorithm choice
  - 'arpack': memory efficient, slower
  - 'randomized': faster for large datasets
  - 'full': exact but memory intensive

### Best Practices
```python
# Standard PCA workflow
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

# Examine explained variance
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)
```

## Neighbor Graph Construction

### k-Nearest Neighbors (kNN)
- Each cell connected to k most similar cells
- Similarity based on Euclidean distance in PC space
- Foundation for clustering and UMAP

### Parameters
- `n_neighbors`: Number of neighbors (default: 15)
  - Smaller: more granular clusters
  - Larger: broader, smoother clusters
  - Scale with dataset size

- `n_pcs`: PCs to use (default: 40)
  - Use PCs capturing biological variation
  - Avoid noisy high-order components
  - Check explained variance ratios

- `metric`: Distance metric
  - 'euclidean': standard choice
  - 'cosine': for normalized data
  - 'correlation': for specific analyses

### Implementation
```python
sc.pp.neighbors(
    adata,
    n_neighbors=15,
    n_pcs=40,
    use_rep='X_pca',  # or 'X_scVI' for integrated data
    metric='euclidean',
    knn=True
)
```

## UMAP Embedding

### Uniform Manifold Approximation and Projection
- Non-linear dimensionality reduction
- Preserves local and global structure
- Better than t-SNE for scRNA-seq

### Key Parameters
- `min_dist`: Minimum distance between points
  - Smaller (0.1): tighter clusters
  - Larger (0.5): spread out visualization
  - Default: 0.5

- `spread`: Scale of embedded points
  - Controls overall scale
  - Interact with min_dist
  - Default: 1.0

- `n_components`: Output dimensions (default: 2)
  - 2D for visualization
  - Higher for downstream analysis

### Usage
```python
sc.tl.umap(adata, min_dist=0.3, spread=1.0)
sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts'])
```

## Leiden Clustering

### Algorithm
- Community detection on neighbor graph
- Optimizes modularity with resolution parameter
- More stable than Louvain clustering

### Resolution Parameter
- Controls granularity of clustering
- Lower (0.1-0.5): fewer, larger clusters
- Higher (1.0-2.0): many, smaller clusters
- Dataset-dependent optimization needed

### Best Practices
```python
# Try multiple resolutions
for res in [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')

# Visualize different resolutions
sc.pl.umap(adata, color=['leiden_0.3', 'leiden_0.5', 'leiden_0.8'])
```

## Parameter Optimization

### Neighbor Graph Tuning
1. **Visual inspection**
   - UMAP should show clear cell type separation
   - Avoid over-connection or fragmentation
   - Balance between resolution and connectivity

2. **Connectivity metrics**
   - Average neighbors per cell
   - Graph connectivity measures
   - Silhouette scores

### Clustering Resolution
1. **Biological validation**
   - Check known marker gene expression
   - Validate against expected cell types
   - Consider prior knowledge

2. **Stability analysis**
   - Bootstrap clustering
   - Parameter sensitivity analysis
   - Consensus clustering

## Quality Control Metrics

### Graph Quality
- **Connectivity**: All cells should be connected
- **Neighborhood preservation**: Similar cells should be neighbors
- **Batch mixing**: Integration quality assessment

### Clustering Quality
- **Silhouette scores**: Within vs between cluster similarity
- **Modularity**: Strength of community structure
- **Marker gene enrichment**: Biological validation

## Advanced Techniques

### Multi-resolution Clustering
```python
# Hierarchical clustering at multiple resolutions
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_coarse')
sc.tl.leiden(adata, resolution=1.5, key_added='leiden_fine')

# Compare resolutions
sc.pl.umap(adata, color=['leiden_coarse', 'leiden_fine'])
```

### Subclustering
```python
# Focus on specific cluster
cluster_adata = adata[adata.obs['leiden'] == '3'].copy()

# Re-analyze at higher resolution
sc.pp.neighbors(cluster_adata, n_neighbors=10)
sc.tl.leiden(cluster_adata, resolution=1.0)
sc.tl.umap(cluster_adata)
```

## Integration Considerations

### Batch-Corrected Analysis
```python
# Use integrated representation
sc.pp.neighbors(adata, use_rep='X_scVI')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Check batch mixing
sc.pl.umap(adata, color=['batch', 'leiden'])
```

### Cross-Modal Integration
- Combine different data types (RNA + ATAC)
- Use appropriate distance metrics
- Consider modality-specific preprocessing

## Common Issues and Solutions

### Over-clustering
**Symptoms:**
- Too many small clusters
- Splitting of known cell types
- Low biological interpretability

**Solutions:**
- Reduce resolution parameter
- Increase n_neighbors
- Check for batch effects

### Under-clustering
**Symptoms:**
- Merging of distinct cell types
- Large heterogeneous clusters
- Poor marker gene specificity

**Solutions:**
- Increase resolution parameter
- Reduce n_neighbors
- Use more PCs in neighbor graph

### Poor Visualization
**Symptoms:**
- Overlapping clusters in UMAP
- Poor separation of cell types
- Batch effects visible

**Solutions:**
- Adjust UMAP parameters (min_dist, spread)
- Improve neighbor graph quality
- Better batch correction

## Validation Strategies

### Marker Gene Analysis
```python
# Find cluster markers
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False)

# Check specific markers
sc.pl.umap(adata, color=['CD3D', 'CD19', 'LYZ'])
```

### Functional Enrichment
- Gene set enrichment analysis
- Pathway analysis per cluster
- Functional annotation validation

### Cross-Dataset Validation
- Project onto reference atlases
- Compare with published datasets
- Cell type annotation transfer

## Species-Specific Considerations

### Human
- Higher genetic diversity
- More complex cell type hierarchies
- Consider population stratification

### Mouse
- More standardized genetic background
- Well-characterized cell types
- Abundant reference data

## Computational Considerations

### Memory Management
- Use sparse matrices
- Subsample for parameter tuning
- Consider approximate algorithms

### Parallelization
- Multi-threaded UMAP
- Distributed computing for large datasets
- GPU acceleration where available

### Reproducibility
- Set random seeds
- Document parameter choices
- Version control for code and data
