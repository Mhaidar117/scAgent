---
title: "Doublet Detection in scRNA-seq Data"
topic: "doublets"
species: "general"
methods: ["scrublet", "doubletfinder"]
---

# Doublet Detection in scRNA-seq Data

## Overview

Doublets are droplets containing two or more cells, which can create artificial cell states that confound downstream analysis. Detection and removal of doublets is essential for accurate clustering and cell type identification.

## What are Doublets?

Doublets occur when:
1. Two cells are captured in the same droplet
2. Free-floating nuclei aggregate during processing
3. Cells undergo partial lysis and re-encapsulation

Expected doublet rates depend on cell loading concentration:
- 1,000 cells/μL: ~0.4% doublets
- 5,000 cells/μL: ~2.3% doublets  
- 10,000 cells/μL: ~7.6% doublets

## Detection Methods

### Scrublet (Recommended)

Scrublet simulates doublets by randomly combining transcriptomes from the dataset.

**Advantages:**
- No need for known markers
- Works well across cell types
- Provides confidence scores
- Fast and memory efficient

**Parameters:**
- `expected_doublet_rate`: Typically 0.06 (6%) for 10X data
- `sim_doublet_ratio`: Ratio of simulated to observed transcriptomes (default: 2.0)
- `n_neighbors`: Number of neighbors for KNN graph (default: round(0.5 * sqrt(n_cells)))

**Usage:**
```python
import scrublet as scr

scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=0.06)
doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2, 
                                                          min_cells=3, 
                                                          min_gene_variability_pctl=85, 
                                                          n_prin_comps=30)
```

### DoubletFinder

Uses artificial nearest neighbors to identify doublets.

**Advantages:**
- High sensitivity
- Can detect heterotypic doublets
- Integrates well with Seurat workflow

**Disadvantages:**
- Requires more computational resources
- Needs parameter optimization
- Originally designed for Seurat/R

## Best Practices

### Pre-processing for Doublet Detection

1. **Minimal filtering before detection**
   - Remove genes expressed in < 3 cells
   - Remove cells with < 200 genes
   - Don't apply strict QC yet

2. **Normalization**
   - Log-normalize for most methods
   - Don't scale or center for Scrublet

### Parameter Selection

1. **Expected doublet rate**
   - Use loading concentration guidelines
   - 10X Chromium: typically 0.04-0.08
   - Higher for dense samples

2. **Threshold selection**
   - Scrublet: Use automatic threshold or examine distribution
   - Manual threshold based on biology knowledge
   - Balance false positives vs false negatives

### Interpretation

1. **Doublet scores**
   - Higher scores = more likely to be doublets
   - Examine score distribution
   - Look for bimodal distribution

2. **Cluster-based validation**
   - Doublets often form separate clusters
   - High expression of markers from multiple cell types
   - Intermediate gene expression profiles

## Common Doublet Signatures

### Transcriptional Doublets
- Express markers from two distinct cell types
- Example: T cell + B cell markers
- Easy to detect with marker-based approaches

### Homotypic Doublets
- Two cells of the same type
- Similar expression profile to singlets
- Harder to detect, rely on total UMI count

### Cell Cycle Doublets
- Often in S/G2M phases
- Higher total gene expression
- May cluster separately from G1 cells

## Integration with QC Workflow

### Recommended Order:
1. Basic gene/cell filtering
2. Doublet detection
3. Doublet removal
4. Comprehensive QC filtering
5. Normalization and downstream analysis

### Rationale:
- QC filtering can remove doublets naturally (high UMI count)
- Doublet detection needs sufficient gene coverage
- Some doublets may pass QC filters

## Validation Steps

1. **Visual inspection**
   - UMAP/t-SNE colored by doublet scores
   - Look for separated doublet clusters
   - Check marker gene expression

2. **Marker analysis**
   - Doublets should express multiple cell type markers
   - Use known marker genes for validation
   - Check for co-expression of exclusive markers

3. **Clustering validation**
   - Doublets often form distinct clusters
   - These clusters should be removed
   - Verify remaining clusters are biologically meaningful

## Species-Specific Considerations

### Human
- Higher baseline expression variability
- More complex cell type landscapes
- Consider tissue-specific markers

### Mouse
- Generally cleaner separation between cell types
- Lower expression noise
- Well-characterized marker genes

## Troubleshooting

### Low doublet detection
- Check expected doublet rate parameter
- Examine cell loading concentration
- Verify sufficient gene coverage

### High false positive rate
- Lower detection threshold
- Check for batch effects
- Validate with known markers

### Computational issues
- Reduce simulation ratio for large datasets
- Use random subsampling for very large datasets
- Consider memory-efficient implementations

## Advanced Considerations

### Batch Effects
- Run doublet detection per batch
- Merge results carefully
- Account for batch-specific doublet rates

### Rare Cell Types
- Doublets may mask rare populations
- Be conservative with rare cell type removal
- Validate with independent markers

### Multi-sample Integration
- Detect doublets before integration
- Different samples may have different rates
- Consider sample-specific thresholds
