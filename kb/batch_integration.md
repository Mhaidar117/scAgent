---
title: "Batch Correction and Integration with scVI"
topic: "integration"
species: "general"
methods: ["scvi", "scanvi", "batch_correction"]
---

# Batch Correction and Integration with scVI

## Overview

Batch effects are systematic technical differences between samples that can confound biological interpretation. scVI (single-cell Variational Inference) provides state-of-the-art batch correction while preserving biological signal.

## Understanding Batch Effects

### Sources of Batch Effects
1. **Technical variation**
   - Different sequencing runs
   - Library preparation protocols
   - Sample processing times
   - Equipment differences

2. **Biological confounding**
   - Sample collection timing
   - Storage conditions
   - Patient/donor differences
   - Tissue handling variations

### Identifying Batch Effects
1. **Visual inspection**
   - PCA plots colored by batch
   - UMAP embeddings showing batch separation
   - Mixing metrics (kBET, LISI)

2. **Statistical tests**
   - PERMANOVA on PC space
   - Batch effect strength metrics
   - Silhouette analysis

## scVI Model Architecture

### Variational Autoencoder Design
- **Encoder**: Maps observations to latent space
- **Decoder**: Reconstructs gene expression
- **Batch variables**: Explicitly modeled as covariates
- **Zero-inflation**: Handles dropouts in scRNA-seq data

### Key Advantages
1. **Probabilistic framework**: Uncertainty quantification
2. **Scalability**: Handles millions of cells
3. **Flexible**: Supports continuous and categorical covariates
4. **Biologically informed**: Preserves cell type structure

## Model Parameters

### Architecture Parameters
- `n_latent`: Latent space dimensions (default: 10-30)
  - Smaller for simpler datasets
  - Larger for complex, heterogeneous data
  - Balance between information and overfitting

- `n_layers`: Number of hidden layers (default: 1-2)
  - More layers for complex datasets
  - Risk of overfitting with too many

- `n_hidden`: Hidden layer size (default: 128)
  - Scale with dataset complexity
  - Computational cost consideration

### Training Parameters
- `max_epochs`: Training epochs (default: 200-400)
  - Monitor convergence
  - Early stopping recommended
  - More epochs for large datasets

- `batch_size`: Mini-batch size (default: 128)
  - Balance memory and convergence
  - Larger batches for stable training

- `learning_rate`: Optimizer learning rate (default: 1e-3)
  - Lower for fine-tuning
  - Higher for initial training

## Integration Workflow

### 1. Preprocessing
```python
# Minimal preprocessing for scVI
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_genes=200)

# Store raw counts
adata.raw = adata

# Select highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=4000, batch_key='batch')
adata = adata[:, adata.var.highly_variable]
```

### 2. Model Setup
```python
import scvi

scvi.model.SCVI.setup_anndata(
    adata,
    layer=None,  # Use X matrix
    batch_key='batch',
    continuous_covariate_keys=['total_counts', 'n_genes_by_counts'],
    categorical_covariate_keys=['sample_type']
)
```

### 3. Model Training
```python
model = scvi.model.SCVI(adata, n_latent=30, n_layers=2)

# Train with early stopping
model.train(
    max_epochs=400,
    early_stopping=True,
    early_stopping_patience=20,
    early_stopping_min_delta=0.5
)
```

### 4. Latent Representation
```python
# Get integrated latent representation
latent = model.get_latent_representation()
adata.obsm['X_scVI'] = latent

# Optional: Get normalized expression
adata.layers['scvi_normalized'] = model.get_normalized_expression()
```

## Quality Assessment

### Integration Metrics
1. **kBET (k-nearest neighbor Batch Effect Test)**
   - Measures batch mixing
   - Values closer to 1 indicate better mixing
   - Can be computed on latent space

2. **LISI (Local Inverse Simpson's Index)**
   - Integration LISI: measures mixing
   - Cell type LISI: measures preservation
   - Balance between the two metrics

3. **Silhouette scores**
   - Batch silhouette (should be low)
   - Cell type silhouette (should be high)

### Visual Assessment
```python
# UMAP on integrated data
sc.pp.neighbors(adata, use_rep='X_scVI')
sc.tl.umap(adata)

# Plot by batch and cell type
sc.pl.umap(adata, color=['batch', 'cell_type'])
```

## Best Practices

### Model Selection
1. **Latent dimensions**
   - Start with 10-30 dimensions
   - Increase for very heterogeneous data
   - Monitor reconstruction quality

2. **Covariate inclusion**
   - Include known batch variables
   - Add continuous covariates (total counts)
   - Don't over-parametrize

### Training Strategies
1. **Early stopping**
   - Monitor training/validation loss
   - Stop when validation loss plateaus
   - Prevents overfitting

2. **Learning rate scheduling**
   - Start with default (1e-3)
   - Reduce if training is unstable
   - Use learning rate schedulers

### Validation
1. **Hold-out validation**
   - Reserve samples for testing
   - Assess generalization
   - Cross-validation for small datasets

2. **Biological validation**
   - Check marker gene preservation
   - Validate known cell type relationships
   - Compare to unintegrated analysis

## Common Issues and Solutions

### Poor Integration
**Symptoms:**
- Batch clusters remain separated
- Low mixing metrics

**Solutions:**
- Increase latent dimensions
- Add more training epochs
- Include additional batch covariates
- Check for poor quality samples

### Over-integration
**Symptoms:**
- Loss of cell type structure
- Artificially merged cell types
- High cell type mixing metrics

**Solutions:**
- Reduce latent dimensions
- Shorter training
- Remove some batch covariates
- Use batch-aware clustering

### Computational Issues
**Symptoms:**
- Out of memory errors
- Very slow training

**Solutions:**
- Reduce batch size
- Use GPU acceleration
- Subsample for parameter tuning
- Consider approximate methods

## Advanced Features

### scANVI (semi-supervised)
- Incorporates cell type labels
- Better preservation of known structure
- Useful when some labels available

### Multi-batch Integration
- Handle multiple batch variables
- Hierarchical batch structure
- Complex experimental designs

### Transfer Learning
- Pre-trained models on similar data
- Fine-tuning for new datasets
- Reduced training time

## Species and Tissue Considerations

### Human vs Mouse
- Human: more genetic diversity
- Mouse: more standardized conditions
- Adjust parameters accordingly

### Tissue-specific Challenges
- Brain: high cellular diversity
- Blood: dynamic cell states
- Solid tumors: high heterogeneity

## Downstream Analysis

### Clustering
```python
# Use integrated representation
sc.pp.neighbors(adata, use_rep='X_scVI')
sc.tl.leiden(adata, resolution=0.5)
```

### Differential Expression
```python
# scVI provides DE testing
de_results = model.differential_expression(
    groupby='cell_type',
    group1='T_cells',
    group2='B_cells'
)
```

### Trajectory Analysis
- Use integrated latent space
- Preserve batch-corrected dynamics
- Validate with known trajectories
