"""Quality control tools for scRNA-seq data analysis."""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Literal, Optional
from typing import List

from ..state import ToolResult, SessionState
from ..quality.assertions import (
    assert_qc_fields_present, 
    assert_pct_mt_range,
    QualityGateError
)
from .io import ensure_run_dir, save_snapshot

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


def _load_adata_from_state(state: SessionState) -> object:
    """Load AnnData object from session state."""
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required for QC operations. Install with: pip install scanpy")

    if not state.adata_path:
        raise ValueError("No AnnData file loaded. Use 'scqc load' first.")

    # Load from most recent checkpoint if available
    if state.history:
        last_entry = state.history[-1]
        checkpoint_path = last_entry.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return sc.read_h5ad(checkpoint_path)

    # Fall back to original file - sc.read_h5ad handles both .h5ad and .h5ad.gz
    return sc.read_h5ad(state.adata_path)


def _detect_species(adata: object, mito_prefix: Optional[str] = None) -> str:
    """Detect species from gene names or use provided prefix."""
    if mito_prefix:
        return "custom"
    
    # Check for human mitochondrial genes (MT- uppercase)
    human_mt = adata.var_names.str.startswith('MT-').sum()
    # Check for mouse mitochondrial genes (mt- lowercase)
    mouse_mt = adata.var_names.str.startswith('mt-').sum()
    
    if human_mt > mouse_mt:
        return "human"
    elif mouse_mt > 0:
        return "mouse"
    else:
        return "other"


def compute_qc_metrics(
    state: SessionState, 
    species: Optional[Literal["human", "mouse", "other"]] = None,
    mito_prefix: Optional[str] = None
) -> ToolResult:
    """Compute quality control metrics for single-cell data.
    
    Args:
        state: Current session state
        species: Species for mitochondrial gene detection (auto-detected if None)
        mito_prefix: Custom mitochondrial gene prefix (overrides species)
        
    Returns:
        ToolResult with QC metrics and generated artifacts
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install -e .[qc]",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
    try:
        # Load data
        adata = _load_adata_from_state(state)
        
        # Determine species: mito_prefix > user species > auto-detection
        if mito_prefix:
            # Custom prefix provided, we'll determine species from prefix
            if mito_prefix.upper() == "MT-":
                species = "human"
            elif mito_prefix.lower() == "mt-":
                species = "mouse"
            else:
                species = "other"
        elif species is not None:
            # User explicitly specified species - respect their choice
            print(f"Using user-specified species: {species}")
        else:
            # Auto-detect species from gene names
            species = _detect_species(adata, mito_prefix)
            print(f"Auto-detected species: {species}")
        
        # Set mitochondrial gene prefix
        if mito_prefix:
            mt_prefix = mito_prefix
        elif species == "human":
            mt_prefix = "MT-"
        elif species == "mouse":
            mt_prefix = "mt-"
        else:
            mt_prefix = "MT-"  # default
        
        # Mark mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith(mt_prefix)
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata, 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
        # Add mitochondrial percentage
        mt_counts = adata[:, adata.var['mt']].X.sum(axis=1)
        # Handle both sparse and dense arrays
        if hasattr(mt_counts, 'A1'):
            mt_counts = mt_counts.A1  # Sparse matrix
        else:
            mt_counts = mt_counts.flatten()  # Dense array

        adata.obs['pct_counts_mt'] = (mt_counts / adata.obs['total_counts'] * 100)
        
        # Quality gate: Verify QC fields are present and valid
        try:
            assert_qc_fields_present(adata)
            assert_pct_mt_range(adata)
        except QualityGateError as e:
            return ToolResult(
                message=f"Quality gate failed: {e}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Create step directory
        step_dir = ensure_run_dir(state, "step_04_compute_qc")
        
        # Generate QC summary
        qc_summary = {
            "timestamp": datetime.now().isoformat(),
            "species": species,
            "mito_prefix": mt_prefix,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_mito_genes": int(adata.var['mt'].sum()),
            "qc_metrics": {
                "n_genes_by_counts": {
                    "mean": float(adata.obs['n_genes_by_counts'].mean()),
                    "median": float(adata.obs['n_genes_by_counts'].median()),
                    "std": float(adata.obs['n_genes_by_counts'].std())
                },
                "total_counts": {
                    "mean": float(adata.obs['total_counts'].mean()),
                    "median": float(adata.obs['total_counts'].median()),
                    "std": float(adata.obs['total_counts'].std())
                },
                "pct_counts_mt": {
                    "mean": float(adata.obs['pct_counts_mt'].mean()),
                    "median": float(adata.obs['pct_counts_mt'].median()),
                    "std": float(adata.obs['pct_counts_mt'].std())
                }
            }
        }
        
        # Add per-batch summary if batch key is available
        if 'batch_key' in state.metadata and state.metadata['batch_key'] in adata.obs.columns:
            batch_key = state.metadata['batch_key']
            batch_summary = {}
            
            for batch in adata.obs[batch_key].unique():
                batch_data = adata[adata.obs[batch_key] == batch]
                batch_summary[str(batch)] = {
                    "n_cells": int(batch_data.n_obs),
                    "mean_genes": float(batch_data.obs['n_genes_by_counts'].mean()),
                    "mean_counts": float(batch_data.obs['total_counts'].mean()),
                    "mean_pct_mt": float(batch_data.obs['pct_counts_mt'].mean())
                }
            
            qc_summary["per_batch"] = batch_summary
        
        # Save QC summary
        summary_path = step_dir / "qc_summary.csv"
        qc_df = pd.DataFrame([qc_summary["qc_metrics"]])
        qc_df.to_csv(summary_path, index=False)
        
        # Save detailed JSON summary
        json_path = step_dir / "qc_summary.json"
        with open(json_path, 'w') as f:
            json.dump(qc_summary, f, indent=2)
        
        # Save snapshot
        run_dir_path = f"runs/{state.run_id}"
        snapshot_result = save_snapshot("step04", run_dir_path, adata)

        # Get snapshot path from artifacts (first artifact is the snapshot)
        snapshot_path_str = str(snapshot_result.artifacts[0]) if snapshot_result.artifacts else None

        # Update dataset summary
        dataset_summary = state.dataset_summary.copy()
        dataset_summary.update({
            "qc_computed": True,
            "species": species,
            "n_mito_genes": int(adata.var['mt'].sum()),
            "qc_metrics": qc_summary["qc_metrics"]
        })

        # CRITICAL: Update adata_path so subsequent tools load data with QC metrics
        state_delta = {"dataset_summary": dataset_summary}
        if snapshot_path_str:
            state_delta["adata_path"] = snapshot_path_str

        return ToolResult(
            message=(
                f"QC metrics computed for {adata.n_obs:,} cells, {adata.n_vars:,} genes. "
                f"Species: {species}, Mitochondrial genes: {adata.var['mt'].sum()}"
            ),
            state_delta=state_delta,
            artifacts=[summary_path, json_path] + snapshot_result.artifacts,
            citations=[
                "Luecken & Theis (2019) Mol Syst Biol",
                "Wolf et al. (2018) Genome Biology"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"Error computing QC metrics: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def apply_qc_filters(
    state: SessionState,
    min_genes: int = 1000,
    max_genes: Optional[int] = None,
    max_pct_mt: float = 10.0,
    method: Literal["threshold", "MAD", "quantile"] = "threshold"
) -> ToolResult:
    """Apply quality control filters to single-cell data.
    
    Args:
        state: Current session state
        min_genes: Minimum number of genes per cell
        max_genes: Maximum number of genes per cell (optional)
        max_pct_mt: Maximum mitochondrial percentage
        method: Method for determining thresholds
        
    Returns:
        ToolResult with filtering results and artifacts
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install -e .[qc]",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
    try:
        # Load data
        adata = _load_adata_from_state(state)
        
        # Check if QC metrics have been computed
        required_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        if not all(col in adata.obs.columns for col in required_cols):
            return ToolResult(
                message="QC metrics not found. Run 'scqc qc compute' first.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Store original counts
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        # Determine thresholds based on method
        if method == "threshold":
            final_min_genes = min_genes
            final_max_genes = max_genes
            final_max_pct_mt = max_pct_mt
        elif method == "MAD":
            # Use median absolute deviation
            genes_median = adata.obs['n_genes_by_counts'].median()
            genes_mad = (adata.obs['n_genes_by_counts'] - genes_median).abs().median()
            final_min_genes = max(min_genes, genes_median - 3 * genes_mad)
            final_max_genes = max_genes if max_genes else (genes_median + 3 * genes_mad)
            
            mt_median = adata.obs['pct_counts_mt'].median()
            mt_mad = (adata.obs['pct_counts_mt'] - mt_median).abs().median()
            final_max_pct_mt = min(max_pct_mt, mt_median + 3 * mt_mad)
        elif method == "quantile":
            # Use quantiles
            final_min_genes = max(min_genes, adata.obs['n_genes_by_counts'].quantile(0.05))
            final_max_genes = max_genes if max_genes else adata.obs['n_genes_by_counts'].quantile(0.95)
            final_max_pct_mt = min(max_pct_mt, adata.obs['pct_counts_mt'].quantile(0.95))
        
        # Apply filters
        sc.pp.filter_cells(adata, min_genes=int(final_min_genes))
        
        # Apply max genes filter if specified
        if final_max_genes is not None:
            adata = adata[adata.obs.n_genes_by_counts <= final_max_genes, :].copy()
        
        # Apply mitochondrial filter
        adata = adata[adata.obs.pct_counts_mt < final_max_pct_mt, :].copy()
        
        # Filter genes (present in at least 3 cells)
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Create step directory
        step_dir = ensure_run_dir(state, "step_06_apply_qc")
        
        # Calculate filtering statistics
        n_cells_after = adata.n_obs
        n_genes_after = adata.n_vars
        
        cells_removed = n_cells_before - n_cells_after
        genes_removed = n_genes_before - n_genes_after
        
        filter_summary = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "thresholds": {
                "min_genes": float(final_min_genes),
                "max_genes": float(final_max_genes) if final_max_genes is not None else None,
                "max_pct_mt": float(final_max_pct_mt)
            },
            "before_filtering": {
                "n_cells": n_cells_before,
                "n_genes": n_genes_before
            },
            "after_filtering": {
                "n_cells": n_cells_after,
                "n_genes": n_genes_after
            },
            "removed": {
                "n_cells": cells_removed,
                "n_genes": genes_removed
            },
            "retained_fraction": {
                "cells": n_cells_after / n_cells_before,
                "genes": n_genes_after / n_genes_before
            }
        }
        
        # Add per-batch statistics if available
        if 'batch_key' in state.metadata and state.metadata['batch_key'] in adata.obs.columns:
            batch_key = state.metadata['batch_key']
            batch_stats = {}
            
            for batch in adata.obs[batch_key].unique():
                batch_count = (adata.obs[batch_key] == batch).sum()
                batch_stats[str(batch)] = {
                    "n_cells_retained": int(batch_count),
                    "fraction_of_total": float(batch_count / n_cells_after)
                }
            
            filter_summary["per_batch_retained"] = batch_stats
        
        # Save filter summary
        filters_path = step_dir / "qc_filters.json"
        with open(filters_path, 'w') as f:
            json.dump(filter_summary, f, indent=2)
        
        # Save snapshot
        run_dir_path = f"runs/{state.run_id}"
        snapshot_result = save_snapshot("step06", run_dir_path, adata)
        
        # Update dataset summary
        dataset_summary = state.dataset_summary.copy()
        dataset_summary.update({
            "qc_filtered": True,
            "n_cells_after_qc": n_cells_after,
            "n_genes_after_qc": n_genes_after,
            "retained_fraction_cells": n_cells_after / n_cells_before,
            "filter_thresholds": filter_summary["thresholds"]
        })
        
        # Update adata_path to point to filtered data
        filtered_path = snapshot_result.artifacts[0]
        
        # Create filter description
        filter_desc = f"min_genes≥{final_min_genes:.0f}"
        if final_max_genes is not None:
            filter_desc += f", max_genes≤{final_max_genes:.0f}"
        filter_desc += f", pct_mt<{final_max_pct_mt:.1f}%"
        
        return ToolResult(
            message=(
                f"✅ QC filters applied ({filter_desc}). "
                f"Retained {n_cells_after:,}/{n_cells_before:,} cells "
                f"({n_cells_after/n_cells_before:.1%}) and {n_genes_after:,}/{n_genes_before:,} genes "
                f"({n_genes_after/n_genes_before:.1%})"
            ),
            state_delta={
                "dataset_summary": dataset_summary,
                "adata_path": str(filtered_path)
            },
            artifacts=[filters_path] + snapshot_result.artifacts,
            citations=[
                "Luecken & Theis (2019) Mol Syst Biol"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"Error applying QC filters: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )

def plot_qc_metrics(
    state: SessionState,
    stage: Literal["pre", "post"] = "pre",
    plot_types: List[str] = ["violin", "scatter", "histogram"]
) -> ToolResult:
    """Generate QC visualization plots.
    
    Args:
        state: Current session state
        stage: Whether to plot pre- or post-filtering data
        plot_types: Types of plots to generate
        
    Returns:
        ToolResult with generated plot artifacts
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install -e .[qc]",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
    try:
        # Load data
        adata = _load_adata_from_state(state)
        
        # Create output directory
        step_dir = ensure_run_dir(state, "qc_plots")
        
        artifacts = []
        
        # Generate plots (placeholder implementation)
        import matplotlib.pyplot as plt
        
        if "violin" in plot_types:
            # Violin plots for QC metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot n_genes_by_counts
            axes[0].violinplot([adata.obs['n_genes_by_counts']], showmeans=True)
            axes[0].set_title('Genes per Cell')
            axes[0].set_ylabel('Number of Genes')
            
            # Plot total_counts
            axes[1].violinplot([adata.obs['total_counts']], showmeans=True)
            axes[1].set_title('Total Counts per Cell')
            axes[1].set_ylabel('Total Counts')
            
            # Plot pct_counts_mt if available
            if 'pct_counts_mt' in adata.obs.columns:
                axes[2].violinplot([adata.obs['pct_counts_mt']], showmeans=True)
                axes[2].set_title('Mitochondrial Gene %')
                axes[2].set_ylabel('Percentage')
            
            plt.tight_layout()
            violin_path = step_dir / f"qc_violin_{stage}.png"
            plt.savefig(violin_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(violin_path))

        if "scatter" in plot_types:
            # Scatter plots for QC metric relationships
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Genes vs Counts
            axes[0].scatter(adata.obs['total_counts'],
                          adata.obs['n_genes_by_counts'],
                          alpha=0.5, s=1)
            axes[0].set_xlabel('Total Counts')
            axes[0].set_ylabel('Number of Genes')
            axes[0].set_title('Genes vs Total Counts')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')

            # MT% vs Counts
            if 'pct_counts_mt' in adata.obs.columns:
                axes[1].scatter(adata.obs['total_counts'],
                              adata.obs['pct_counts_mt'],
                              alpha=0.5, s=1)
                axes[1].set_xlabel('Total Counts')
                axes[1].set_ylabel('Mitochondrial %')
                axes[1].set_title('MT% vs Total Counts')
                axes[1].set_xscale('log')

            # MT% vs Genes
            if 'pct_counts_mt' in adata.obs.columns:
                axes[2].scatter(adata.obs['n_genes_by_counts'],
                              adata.obs['pct_counts_mt'],
                              alpha=0.5, s=1)
                axes[2].set_xlabel('Number of Genes')
                axes[2].set_ylabel('Mitochondrial %')
                axes[2].set_title('MT% vs Number of Genes')
                axes[2].set_xscale('log')

            plt.tight_layout()
            scatter_path = step_dir / f"qc_scatter_{stage}.png"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(scatter_path))

        if "histogram" in plot_types:
            # Histogram distributions of QC metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Distribution of genes per cell
            axes[0].hist(adata.obs['n_genes_by_counts'], bins=50, edgecolor='black')
            axes[0].set_xlabel('Number of Genes')
            axes[0].set_ylabel('Number of Cells')
            axes[0].set_title('Distribution of Genes per Cell')
            axes[0].axvline(adata.obs['n_genes_by_counts'].median(),
                          color='red', linestyle='--', label='Median')
            axes[0].legend()

            # Distribution of counts per cell
            axes[1].hist(adata.obs['total_counts'], bins=50, edgecolor='black')
            axes[1].set_xlabel('Total Counts')
            axes[1].set_ylabel('Number of Cells')
            axes[1].set_title('Distribution of Total Counts')
            axes[1].axvline(adata.obs['total_counts'].median(),
                          color='red', linestyle='--', label='Median')
            axes[1].legend()

            # Distribution of MT%
            if 'pct_counts_mt' in adata.obs.columns:
                axes[2].hist(adata.obs['pct_counts_mt'], bins=50, edgecolor='black')
                axes[2].set_xlabel('Mitochondrial %')
                axes[2].set_ylabel('Number of Cells')
                axes[2].set_title('Distribution of MT%')
                axes[2].axvline(adata.obs['pct_counts_mt'].median(),
                              color='red', linestyle='--', label='Median')
                axes[2].legend()

            plt.tight_layout()
            hist_path = step_dir / f"qc_histogram_{stage}.png"
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(hist_path))

        # Create checkpoint for QC plots
        checkpoint_path = step_dir / f"qc_plots_{stage}_checkpoint.h5ad"
        state.checkpoint(str(checkpoint_path), f"qc_plots_{stage}")

        # Register plots with state
        for artifact in artifacts:
            if "violin" in artifact:
                state.add_artifact(str(artifact), f"QC Violin Plot ({stage})")
            elif "scatter" in artifact:
                state.add_artifact(str(artifact), f"QC Scatter Plot ({stage})")
            elif "histogram" in artifact:
                state.add_artifact(str(artifact), f"QC Histogram Plot ({stage})")
            else:
                state.add_artifact(str(artifact), f"QC Plot ({stage})")
        
        return ToolResult(
            message=f"✅ Generated {len(artifacts)} QC plots for {stage}-filtering data",
            state_delta={"qc_plots_generated": True},
            artifacts=artifacts,
            citations=[]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Error generating QC plots: {e}",
            state_delta={},
            artifacts=[],
            citations=[]
        )