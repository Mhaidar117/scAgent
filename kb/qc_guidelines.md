---
title: "Quality Control Guidelines for scRNA-seq"
topic: "quality_control"
species: "general"
priority: "high"
---

# Quality Control Guidelines for scRNA-seq

## Overview

Quality control (QC) is the critical first step in scRNA-seq analysis. Poor quality cells and genes can significantly impact downstream analysis results.

## Key QC Metrics

### Per-Cell Metrics

1. **Total UMI/read count per cell**
   - Low counts indicate poor capture efficiency or cell death
   - Typical range: 1,000-100,000 UMIs per cell
   - Very high counts may indicate doublets

2. **Number of detected genes per cell**
   - Fewer genes suggest poor RNA capture
   - Typical minimum: 200-500 genes per cell
   - Species-dependent variations exist

3. **Mitochondrial gene percentage**
   - High mitochondrial percentage indicates cell stress/death
   - Human: typically < 20% for most cell types
   - Mouse: typically < 15% for most cell types
   - Tissue-specific variations exist

4. **Ribosomal gene percentage**
   - Very high ribosomal percentage may indicate stress
   - Typical range: 10-40% depending on cell type

### Per-Gene Metrics

1. **Number of cells expressing each gene**
   - Genes expressed in very few cells are often noise
   - Typical minimum: expressed in at least 3-10 cells
   - Consider total cell count when setting thresholds

2. **Total expression level per gene**
   - Very lowly expressed genes contribute little information
   - Balance between removing noise and retaining biology

## Recommended Thresholds

### Conservative Approach (Strict QC)
- Minimum genes per cell: 500
- Maximum mitochondrial percentage: 15%
- Minimum cells per gene: 10

### Permissive Approach (Lenient QC)
- Minimum genes per cell: 200
- Maximum mitochondrial percentage: 25%
- Minimum cells per gene: 3

### Adaptive Approach
Use median absolute deviation (MAD) based thresholds:
- Filter cells with total counts < median - 3*MAD
- Filter cells with total counts > median + 3*MAD
- Filter cells with gene counts < median - 3*MAD
- Filter cells with mitochondrial % > median + 3*MAD

## Species-Specific Considerations

### Human
- Mitochondrial gene prefix: "MT-"
- Typical mitochondrial percentage: 5-20%
- Ribosomal gene prefixes: "RPS", "RPL"

### Mouse
- Mitochondrial gene prefix: "mt-"
- Typical mitochondrial percentage: 3-15%
- Generally lower mitochondrial content than human

## Tissue-Specific Considerations

### Brain/Neurons
- Neurons have high metabolic activity
- Slightly higher mitochondrial thresholds acceptable
- May have lower total UMI counts

### Immune Cells
- T cells often have lower gene counts
- Consider more permissive thresholds
- High proliferation may affect QC metrics

### Cancer/Tumor Samples
- Highly variable QC metrics
- Consider sample-specific thresholds
- Dying cells are common, adjust MT thresholds

## Quality Control Workflow

1. **Calculate QC metrics**
   ```python
   sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
   ```

2. **Add mitochondrial and ribosomal gene annotations**
   ```python
   adata.var['mt'] = adata.var_names.str.startswith('MT-')  # human
   adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
   ```

3. **Calculate percentages**
   ```python
   sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], 
                              percent_top=None, log1p=False, inplace=True)
   ```

4. **Visualize QC metrics**
   ```python
   sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)
   ```

5. **Apply filters**
   ```python
   # Filter cells
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.filter_genes(adata, min_cells=3)
   
   # Filter by QC metrics
   adata = adata[adata.obs.n_genes_by_counts < 2500, :]
   adata = adata[adata.obs.pct_counts_mt < 20, :]
   ```

## Best Practices

1. **Always visualize before filtering**
   - Plot distributions of QC metrics
   - Look for outliers and batch effects
   - Consider biological expectations

2. **Document filtering decisions**
   - Record exact thresholds used
   - Note number of cells/genes removed
   - Save pre- and post-filtering plots

3. **Consider batch effects**
   - QC metrics may vary between batches
   - Apply batch-aware filtering if needed
   - Check for systematic differences

4. **Iterative approach**
   - Start with permissive filtering
   - Examine downstream results
   - Refine thresholds as needed

5. **Preserve cell type diversity**
   - Some cell types naturally have different QC profiles
   - Don't over-filter rare cell types
   - Consider cell type annotations if available

## Common Mistakes

1. **Over-filtering**
   - Removing too many cells can bias results
   - Loss of rare cell types
   - Reduced statistical power

2. **Under-filtering**
   - Low-quality cells create spurious clusters
   - Increased noise in downstream analysis
   - Poor embedding quality

3. **Ignoring batch effects**
   - Different samples may need different thresholds
   - Technical artifacts can masquerade as biology

4. **Fixed thresholds across datasets**
   - QC metrics vary by protocol, tissue, species
   - Always adapt to your specific dataset

## Validation Steps

After QC, validate results by:
1. Checking cell type marker expression
2. Examining embedding quality (UMAP/t-SNE)
3. Assessing cluster stability
4. Comparing to expected biology
