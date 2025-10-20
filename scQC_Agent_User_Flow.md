# scQC Agent User Flow Guide
## Complete Kidney scRNA-seq Analysis Workflow

This guide provides step-by-step commands for analyzing mouse kidney scRNA-seq data using scQC Agent's natural language interface. Each command can be copied and pasted directly into your terminal after activating the scQC environment.

---

## Prerequisites

### 1. Activate the scQC Environment
```bash
source /Users/michaelhaidar/Documents/Vanderbilt/Brain_Research/scAgent/scQC/bin/activate
```

### 2. Navigate to Your Working Directory
```bash
cd /Users/michaelhaidar/Documents/Vanderbilt/Brain_Research/scAgent
```

---

## Step 1: Initialize scQC Session

Start a new analysis session. This creates the `.scqc_state.json` file to track your progress.

```bash
scqc init
```

Expected output:
- Creates `runs/` directory
- Initializes state file `.scqc_state.json`
- Ready for data loading

---

## Step 2: Load Kidney Dataset

Load the mouse kidney 10X Genomics data including raw droplets, filtered cells, and metadata.

```bash
scqc chat "Load my kidney dataset from Data_files/raw_data/7_raw_feature_bc_matrix.h5 (raw droplets), Data_files/raw_data/7_filtered_feature_bc_matrix.h5 (filtered cells), and metadata from Data_files/raw_data/metadata.xlsx - Sheet1.csv" --mode execute
```

Expected results:
- Loads 1,617,288 raw droplets
- Loads 8,877 filtered cells
- 32,285 genes detected
- Metadata integrated

---

## Step 3: Compute QC Metrics

Calculate quality control metrics for the mouse kidney data.

```bash
scqc chat "Compute QC metrics for my mouse kidney data" --mode execute
```

Expected results:
- Detects 13 mitochondrial genes
- Calculates n_genes_by_counts
- Calculates pct_counts_mt (mean ~0.69%)
- Saves QC summary CSV

---

## Step 4: Visualize Pre-Filtering QC Metrics

Generate violin plots to visualize QC metrics before filtering.

```bash
scqc chat "Create violin plots of the QC metrics for my kidney data" --mode execute
```

Expected outputs:
- `qc_pre_violins.png` showing:
  - Number of genes per cell distribution
  - Total counts per cell
  - Mitochondrial percentage distribution

---

## Step 5: Generate Knee Plot for Ambient RNA Analysis

Create knee plot to identify cell-free droplets for ambient RNA profiling.

```bash
scqc chat "Generate knee plot to identify empty droplets in my kidney data" --mode execute
```

Expected results:
- Identifies ~954,112 cell-free droplets
- Creates `knee_plot.png`
- Generates `ambient_profile.csv`
- Prepares data for SCAR correction

---

## Step 6: Run Ambient RNA Correction (SCAR)

Remove ambient RNA contamination using SCAR.

```bash
scqc chat "Run SCAR ambient RNA removal on my kidney data with 10 epochs" --mode execute
```

Expected results:
- Trains SCAR model (10 epochs)
- Denoises expression matrix
- Creates latent representation (X_scAR)
- Saves checkpoint with corrected counts
- Runtime: ~65 seconds

---

## Step 7: Apply Quality Control Filters

Filter cells based on QC metrics with tissue-appropriate thresholds.

```bash
scqc chat "Filter cells with minimum 500 genes and maximum 10% mitochondrial content for my kidney data" --mode execute
```

Expected results:
- Retains 8,794 cells (99.1% pass rate)
- Filters cells with <500 genes
- Removes cells with >10% mitochondrial reads
- Saves filtered dataset

---

## Step 8: Visualize Post-Filtering QC Metrics

Generate violin plots after quality filtering.

```bash
scqc chat "Create violin plots of QC metrics after filtering" --mode execute
```

Expected outputs:
- `qc_post_violins.png` showing improved distributions
- Confirms successful filtering

---

## Step 9: Initial Clustering and UMAP (Pre-Doublet Removal)

Create initial UMAP visualization before doublet detection.

```bash
scqc chat "Run PCA, build neighbor graph, create UMAP embedding, and perform Leiden clustering with resolution 2.0 on my kidney data" --mode execute
```

Expected results:
- Performs PCA (50 components)
- Builds k-NN graph (k=15)
- Creates UMAP embedding
- Generates ~36 clusters
- Saves `umap_pre.png`

---

## Step 10: Detect Doublets with DoubletFinder

Identify potential doublets using DoubletFinder with pK optimization.

```bash
scqc chat "Detect doublets using DoubletFinder with automatic pK optimization and expected rate of 0.06 for my kidney data" --mode execute
```

Expected results:
- Performs pK parameter sweep
- Identifies ~551 doublets (6.2% rate)
- Creates `pk_sweep_plot.png`
- Generates `doublet_detection_viz.png`
- Runtime: ~45 seconds

---

## Step 11: Remove Detected Doublets

Apply the doublet filter to clean the dataset.

```bash
scqc chat "Remove the detected doublets from my kidney data" --mode execute
```

Expected results:
- Removes 551 doublet cells
- Retains 8,243 high-quality singlets
- Updates dataset

---

## Step 12: Re-cluster After Doublet Removal

Perform final clustering on the clean dataset.

```bash
scqc chat "Create a new UMAP visualization after doublet removal with higher resolution clustering for kidney data" --mode execute
```

Expected results:
- Re-computes PCA on clean data
- Generates refined clusters
- Creates updated UMAP
- Saves `umap_post_doublets.png`

---

## Step 13: Detect Marker Genes

Find differentially expressed genes for each cluster.

```bash
scqc chat "Find marker genes for each cluster in my mouse kidney data using wilcoxon test" --mode execute
```

Expected results:
- Tests ~900 significant markers
- Generates ranked gene lists per cluster
- Creates `marker_genes.csv`
- Produces `marker_heatmap.png`
- Creates `marker_dotplot.png`
- Runtime: ~30 seconds

---

## Step 14: Annotate Cell Types

Identify cell types using kidney-specific markers.

```bash
scqc chat "Annotate my mouse kidney cell clusters to identify cell types" --mode execute
```

Expected results:
- Identifies 11+ kidney cell types:
  - Proximal tubule cells
  - Collecting duct principal cells
  - Thick ascending limb
  - Distal convoluted tubule
  - Loop of Henle
  - Endothelial cells
  - Macrophages
  - T cells
  - And more...
- Creates `annotation_summary.csv`
- Generates `umap_annotated.png`
- Produces `cell_type_distribution.png`

---

## Step 15: Generate Comprehensive Summary

Get a final summary of the complete analysis.

```bash
scqc chat "Give me a comprehensive summary of my kidney data analysis including how many cells we started with, how many remain after filtering and doublet removal, and what cell types we found" --mode execute
```

Expected output:
- Starting cells: 8,877
- Post-QC filtering: 8,794
- Post-doublet removal: 8,243
- Number of clusters and cell types
- Key QC statistics
- Complete workflow summary

---

## Step 16: Save Final Report

Generate HTML report with all results.

```bash
scqc report generate --format html --output kidney_analysis_report.html
```

---

## Optional Advanced Commands

### Differential Expression Between Cell Types
```bash
scqc chat "Compare proximal tubule cells vs collecting duct cells to find differentially expressed genes" --mode execute
```

### Export Results
```bash
scqc chat "Save the final processed dataset as an h5ad file" --mode execute
```

### Check Current State
```bash
scqc state
```

### View Summary Anytime
```bash
scqc summary
```

---

## Complete Workflow Script

For automated execution, save this as `run_kidney_workflow.sh`:

```bash
#!/bin/bash
# Complete kidney scRNA-seq workflow

# Activate environment
source /Users/michaelhaidar/Documents/Vanderbilt/Brain_Research/scAgent/scQC/bin/activate

# Initialize
scqc init

# Load data
scqc chat "Load my kidney dataset from Data_files/raw_data/7_raw_feature_bc_matrix.h5 (raw droplets), Data_files/raw_data/7_filtered_feature_bc_matrix.h5 (filtered cells), and metadata from Data_files/raw_data/metadata.xlsx - Sheet1.csv" --mode execute

# QC metrics
scqc chat "Compute QC metrics for my mouse kidney data" --mode execute
scqc chat "Create violin plots of the QC metrics for my kidney data" --mode execute

# Ambient RNA
scqc chat "Generate knee plot to identify empty droplets in my kidney data" --mode execute
scqc chat "Run SCAR ambient RNA removal on my kidney data with 10 epochs" --mode execute

# Quality filtering
scqc chat "Filter cells with minimum 500 genes and maximum 10% mitochondrial content for my kidney data" --mode execute
scqc chat "Create violin plots of QC metrics after filtering" --mode execute

# Initial clustering
scqc chat "Run PCA, build neighbor graph, create UMAP embedding, and perform Leiden clustering with resolution 2.0 on my kidney data" --mode execute

# Doublet detection
scqc chat "Detect doublets using DoubletFinder with automatic pK optimization and expected rate of 0.06 for my kidney data" --mode execute
scqc chat "Remove the detected doublets from my kidney data" --mode execute

# Final clustering
scqc chat "Create a new UMAP visualization after doublet removal with higher resolution clustering for kidney data" --mode execute

# Cell type analysis
scqc chat "Find marker genes for each cluster in my mouse kidney data using wilcoxon test" --mode execute
scqc chat "Annotate my mouse kidney cell clusters to identify cell types" --mode execute

# Summary
scqc chat "Give me a comprehensive summary of my kidney data analysis" --mode execute

echo "Kidney workflow complete!"
```

---

## Expected Timeline

- **Total runtime**: ~3-5 minutes for complete workflow
- **Most time-consuming steps**:
  - SCAR training: ~65 seconds
  - DoubletFinder with pK sweep: ~45 seconds
  - Marker gene detection: ~30 seconds

---

## Troubleshooting

### If Agent Initialization Hangs
```bash
# Kill any hanging processes
pkill -f scqc
# Restart the workflow
```

### Check Logs
```bash
# View recent chat history
cat runs/*/chat_*/messages.json | tail -100
```

### Verify State
```bash
# Check current state
scqc state

# View summary
scqc summary
```

---

## Output Directory Structure

After completing the workflow, your `runs/` directory will contain:

```
runs/YYYYMMDD_HHMMSS/
├── step_00_load/
│   └── initial_data.h5ad
├── step_01_qc_compute/
│   ├── qc_summary.csv
│   └── qc_metrics.json
├── step_02_qc_plot/
│   └── qc_pre_violins.png
├── step_03_knee_plot/
│   ├── knee_plot.png
│   └── ambient_profile.csv
├── step_04_scar/
│   └── models/scar/checkpoint
├── step_05_qc_filter/
│   └── filtered_data.h5ad
├── step_06_clustering/
│   └── umap_pre.png
├── step_07_doublets/
│   ├── pk_sweep_plot.png
│   └── doublet_scores.png
├── step_08_clustering_final/
│   └── umap_post_doublets.png
├── step_09_markers/
│   ├── marker_genes.csv
│   └── marker_heatmap.png
└── step_10_annotation/
    ├── annotation_summary.csv
    ├── umap_annotated.png
    └── cell_type_distribution.png
```

---

## Notes

1. **Species Detection**: The agent automatically detects "mouse" and "kidney" from your messages
2. **Tissue-Specific Defaults**: Kidney-appropriate thresholds are automatically applied
3. **Reproducibility**: All random seeds are set for consistent results
4. **State Persistence**: Progress is saved after each step in `.scqc_state.json`
5. **Error Recovery**: If a step fails, you can restart from the last checkpoint

---

## Citation

If you use scQC Agent in your research, please cite:
- scQC Agent: [GitHub repository](https://github.com/Mhaidar117/scAgent)
- Based on the kidney analysis framework from Nelson Lab

---

**Document Version**: 1.0
**Last Updated**: 2025-01-19
**Tested With**: scQC Agent v0.1.0, Python 3.13.9