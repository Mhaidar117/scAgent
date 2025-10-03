---
title: "Standard QC Workflow SOP"
topic: "workflow"
species: "general"
workflow_type: "quality_control"
---

# Standard Quality Control Workflow

## Overview

This document outlines the full standard operating procedure for quality control in scRNA-seq analysis using the scQC Agent. When asked for a complete workflow, this is how the process should proceed: steps 1-15. Individual steps can be executed independently as requested.


## Workflow Steps

### 1. Data Loading and Initial Assessment
```bash
load_data with file_path parameter
```

**Purpose:** 
- Initialize session tracking
- Load raw count matrix
- Assess initial data dimensions

**Expected output:**
- Session state with data path
- Initial cell and gene counts
- Data format validation

### 2. Ambient RNA Removal (scAR)
``` bash
scqc scar run
```
**Purpose:**
- Writes ambient profile to adata.uns
- Writes denoised counts to layers['scar_denoised']
- snapshots step_ambient/.

### 3. QC Metrics Calculation
```bash
scqc qc compute --species human
```

**Purpose:**
- Calculate per-cell and per-gene metrics
- Identify mitochondrial and ribosomal genes
- Compute percentage metrics

**Key metrics computed:**
- Total UMI counts per cell
- Number of genes per cell
- Mitochondrial percentage
- Ribosomal percentage
- Per-gene expression statistics

### 4. QC Visualization (Pre-filtering)
```bash
scqc qc plot --stage pre
```

**Purpose:**
- Visualize QC metric distributions
- Identify outliers and thresholds
- Check for batch effects

**Generated plots:**
- Violin plots of QC metrics
- Scatter plots (genes vs counts)
- Distribution histograms
- Batch comparison plots

### 5. Threshold Determination

**Manual approach:**
- Examine QC plots
- Set biologically reasonable thresholds
- Consider tissue and species-specific norms

**Automatic approach:**
- Use MAD-based thresholds
- Apply percentile-based filters
- Leverage prior knowledge

**Common thresholds:**
- Min genes per cell: 200-500
- Max mitochondrial %: 15-25%
- Min cells per gene: 3-10

### 6. Apply QC Filters
```bash
scqc qc apply --min-genes 200 --max-pct-mt 20 --method threshold
```

**Purpose:**
- Remove low-quality cells and genes
- Apply determined thresholds
- Create filtered dataset

**Filtering criteria:**
- Cell-level filters (genes, MT%, counts)
- Gene-level filters (expression, prevalence)
- Outlier removal

### 7. QC Visualization (Post-filtering)
```bash
scqc qc plot --stage post
```

**Purpose:**
- Validate filtering results
- Compare pre- and post-filtering distributions
- Ensure adequate cell retention

**Quality checks:**
- Retained cell count (>80% typically)
- Distribution normalization
- Batch effect persistence

### 8. Double Detection and Removal
```bash
scqc doublets detect --method scrublet
scqc doublets apply --threshold <t>
```
**Purpose:**
- Records obs['doublet_score'] and obs['predicted_doublet']
- Removes predicted doubles
- Creates snapshop

### 9. QC Plots (Post-Filter #2)
```bash
scqc qc plot --stage post
```
**Purpose:**
- Ensure counts ratained are reported

### 10. Normalize -> Log -> HVG 


### 11. Quick Graph Analysis
```bash
scqc graph quick --seed 42 --resolution 0.5
```
**Purpose:**
- Initial dimensionality reduction
- Preliminary clustering
- Quality assessment visualization

**Analysis steps:**
- PCA computation
- Neighbor graph construction
- UMAP embedding
- Leiden clustering

### 12. Batch Correction (choose one)
- A. scVI: 
```bash
scqc scvi run
```
    - uses counts; outputs obsm['X_scVI'].
- B. scAR (denoising): 
```bash
scqc scar run
```
- make sure to be explicit which representation downstream will use.

### 13. Final Graph and Clustering
- pck represetation (use_rep = X_scVI or X_scar (latent))
```bash
scqc graph from-rep X_scVI --n-neighbors 15 --resolution 0.6
# OR
scqc graph final --use-rep X_scVI --n-neighbors 15 --resolution 0.6
```
**Purpose:**
- Recompute neighbors after correction on the chosen representation (don’t reuse quick graph)

### 14. Batch Diagnostics
```bash
# Through agent planning - no direct CLI yet
scqc chat "run batch diagnostics on X_scVI embedding"
```
**Purpose:**
- Assess batch integration quality using scib-metrics
- Compute kBET, iLISI, cLISI, graph connectivity
- Generate diagnostic plots and save results


### 15. Results Summary
```bash
scqc summary
```

**Purpose:**
- Document QC results
- Record filtering decisions
- Track artifacts generated

## Quality Gates

### Minimum Requirements
- At least 80% cell retention after filtering
- Clear separation in QC metric distributions
- Reasonable mitochondrial percentages
- Adequate gene coverage per cell

### Warning Signs
- Extreme cell loss (>50%)
- Bimodal distributions suggesting doublets
- High mitochondrial percentages across samples
- Very low gene counts per cell

### Failure Criteria
- <50% cell retention
- No clear QC thresholds
- Systematic quality issues
- Technical failure indicators

## Species-Specific Guidelines

### Human Samples
```bash
# Standard human parameters
scqc qc compute --species human --mito-prefix MT-
scqc qc apply --min-genes 250 --max-pct-mt 20
```

**Typical thresholds:**
- Min genes: 250-500
- Max MT%: 15-25%
- Min cells per gene: 3

### Mouse Samples
```bash
# Standard mouse parameters
scqc qc compute --species mouse --mito-prefix mt-
scqc qc apply --min-genes 200 --max-pct-mt 15
```

**Typical thresholds:**
- Min genes: 200-400
- Max MT%: 10-20%
- Min cells per gene: 3

## Troubleshooting

### High Mitochondrial Content
**Possible causes:**
- Cell stress or death
- Tissue dissociation artifacts
- Sample degradation

**Solutions:**
- Relax MT% threshold slightly
- Check sample preparation protocol
- Consider tissue-specific norms

### Low Gene Detection
**Possible causes:**
- Poor RNA capture efficiency
- Low cell viability
- Technical issues

**Solutions:**
- Lower min_genes threshold
- Check library preparation
- Validate with control samples

### Extreme Cell Loss
**Possible causes:**
- Too stringent thresholds
- Poor sample quality
- Batch effects

**Solutions:**
- Use MAD-based adaptive thresholds
- Examine per-batch QC metrics
- Consider sample-specific filtering

## Documentation Requirements

### Session Records
- Parameter choices and rationale
- Number of cells/genes before and after filtering
- QC threshold justification
- Generated artifacts and plots

### Quality Assessment
- QC metric distributions
- Filtering effectiveness
- Batch effect evaluation
- Known marker gene expression

## Next Steps

After successful QC:
1. Proceed to normalization and scaling
2. Highly variable gene selection
3. Dimensionality reduction (PCA)
4. Batch correction if needed (scVI)
5. Graph analysis and clustering
6. Cell type annotation

## Best Practices

1. **Always visualize before filtering**
2. **Document all parameter choices**
3. **Consider biological context**
4. **Validate with known markers**
5. **Preserve session state**
6. **Generate comprehensive reports**
