"""Marker gene detection tools for cluster characterization."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

from ..state import ToolResult, SessionState
from .io import ensure_run_dir, save_snapshot

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _load_adata_from_state(state: SessionState) -> object:
    """Load AnnData object from session state."""
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required. Install with: pip install scanpy")

    adata_path = state.adata_path
    if not adata_path:
        raise ValueError("No AnnData file loaded. Use 'scqc load' first.")

    # Load from most recent checkpoint if available
    if state.history:
        last_entry = state.history[-1]
        checkpoint_path = last_entry.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return sc.read_h5ad(checkpoint_path)

    # Fall back to original file - sc.read_h5ad handles both .h5ad and .h5ad.gz
    return sc.read_h5ad(adata_path)


def detect_marker_genes(
    state: SessionState,
    cluster_key: str = "leiden",
    method: Literal["t-test", "wilcoxon", "logreg"] = "wilcoxon",
    n_genes: int = 25,
    use_raw: bool = False,
    reference: str = "rest",
    species: Optional[Literal["human", "mouse"]] = None
) -> ToolResult:
    """Detect marker genes for each cluster using differential expression.

    Uses scanpy.tl.rank_genes_groups to identify genes that distinguish
    each cluster from others. This is essential for cluster annotation
    and biological interpretation.

    Args:
        state: Current session state
        cluster_key: Key in adata.obs containing cluster assignments
        method: Statistical test to use:
            - "wilcoxon": Wilcoxon rank-sum (default, non-parametric)
            - "t-test": Welch's t-test (parametric)
            - "logreg": Logistic regression (multivariate)
        n_genes: Number of top marker genes to report per cluster
        use_raw: Use raw counts (adata.raw) if available
        reference: Reference group for comparison:
            - "rest": Compare each cluster to all others (default)
            - Cluster name/ID: Compare to specific cluster
        species: Species for gene name filtering (human/mouse)

    Returns:
        ToolResult with artifacts: marker_genes.csv, marker_heatmap.png, dotplot.png
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    try:
        # Load data
        adata = _load_adata_from_state(state)

        # Check if cluster key exists
        if cluster_key not in adata.obs.columns:
            available_keys = [k for k in adata.obs.columns if k.startswith('leiden') or k.startswith('louvain')]
            return ToolResult(
                message=f"❌ Cluster key '{cluster_key}' not found in adata.obs. Available: {available_keys}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Get cluster info
        clusters = adata.obs[cluster_key]
        n_clusters = clusters.nunique()

        if n_clusters < 2:
            return ToolResult(
                message=f"❌ Only {n_clusters} cluster found. Need at least 2 for marker detection.",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_13_marker_genes"
        step_dir = ensure_run_dir(step_dir_path)

        # Detect species from data if not provided
        if species is None:
            # Check for species in metadata
            if "species" in state.metadata:
                species = state.metadata.get("species")
            else:
                # Try to infer from gene names
                gene_names = adata.var_names.str.upper()
                if any(gene_names.str.startswith('MT-')):
                    species = "human"
                elif any(gene_names.str.startswith('MT')):
                    species = "mouse"

        # Ensure data is log-normalized for marker detection
        if use_raw and adata.raw is not None:
            adata_for_markers = adata
        else:
            # Check if data is log-normalized
            if 'log1p' not in adata.uns_keys():
                # Need to normalize for marker detection
                adata_copy = adata.copy()
                sc.pp.normalize_total(adata_copy, target_sum=1e4)
                sc.pp.log1p(adata_copy)
                adata_for_markers = adata_copy
            else:
                adata_for_markers = adata

        # Run marker gene detection
        sc.tl.rank_genes_groups(
            adata_for_markers,
            groupby=cluster_key,
            use_raw=use_raw,
            method=method,
            n_genes=n_genes,
            reference=reference,
            key_added='rank_genes_groups'
        )

        # Extract results into DataFrame
        result_dict = adata_for_markers.uns['rank_genes_groups']
        clusters_list = result_dict['names'].dtype.names

        # Build comprehensive marker table
        marker_records = []
        has_logfc = 'logfoldchanges' in result_dict
        has_pvals = 'pvals' in result_dict
        has_pvals_adj = 'pvals_adj' in result_dict

        for cluster in clusters_list:
            for i in range(n_genes):
                record = {
                    'cluster': cluster,
                    'rank': i + 1,
                    'gene': result_dict['names'][cluster][i],
                    'score': float(result_dict['scores'][cluster][i])
                }

                # Add fields only if available (different methods have different fields)
                if has_logfc:
                    record['logfoldchange'] = float(result_dict['logfoldchanges'][cluster][i])

                if has_pvals:
                    record['pval'] = float(result_dict['pvals'][cluster][i])

                if has_pvals_adj:
                    record['pval_adj'] = float(result_dict['pvals_adj'][cluster][i])

                marker_records.append(record)

        marker_df = pd.DataFrame(marker_records)

        # Save marker table
        marker_csv_path = step_dir / "marker_genes.csv"
        marker_df.to_csv(marker_csv_path, index=False)

        artifacts = [str(marker_csv_path)]

        # Generate visualizations
        if PLOTTING_AVAILABLE:
            # 1. Heatmap of top marker genes
            try:
                top_genes_per_cluster = marker_df.groupby('cluster').head(10)['gene'].tolist()

                plt.figure(figsize=(12, 10))
                sc.pl.heatmap(
                    adata_for_markers,
                    var_names=top_genes_per_cluster,
                    groupby=cluster_key,
                    cmap='viridis',
                    dendrogram=True,
                    swap_axes=True,
                    show=False,
                    save=False
                )

                heatmap_path = step_dir / "marker_heatmap.png"
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(str(heatmap_path))
            except Exception as e:
                print(f"Warning: Could not generate heatmap: {e}")

            # 2. Dotplot of top markers
            try:
                top5_per_cluster = marker_df.groupby('cluster').head(5)['gene'].unique().tolist()

                plt.figure(figsize=(14, 8))
                sc.pl.dotplot(
                    adata_for_markers,
                    var_names=top5_per_cluster,
                    groupby=cluster_key,
                    dendrogram=True,
                    show=False,
                    save=False
                )

                dotplot_path = step_dir / "marker_dotplot.png"
                plt.savefig(dotplot_path, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(str(dotplot_path))
            except Exception as e:
                print(f"Warning: Could not generate dotplot: {e}")

        # Save top markers summary (JSON)
        top_markers_dict = {}
        for cluster in clusters_list:
            top5 = marker_df[marker_df['cluster'] == cluster].head(5)
            top_markers_dict[str(cluster)] = top5['gene'].tolist()

        summary_json_path = step_dir / "top_markers_summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(top_markers_dict, f, indent=2)
        artifacts.append(str(summary_json_path))

        # Update adata with marker results and save checkpoint
        adata.uns['rank_genes_groups'] = adata_for_markers.uns['rank_genes_groups']

        checkpoint_result = save_snapshot("marker_genes", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_marker_genes.h5ad")

        # Update state
        state.checkpoint(str(checkpoint_path), "marker_genes")
        for artifact in artifacts:
            label = Path(artifact).stem.replace('_', ' ').title()
            state.add_artifact(artifact, label)

        # Count significant markers (padj < 0.05) if pval_adj is available
        if has_pvals_adj:
            sig_markers = marker_df[marker_df['pval_adj'] < 0.05]
            sig_count_per_cluster = sig_markers.groupby('cluster').size().to_dict()
            total_sig = len(sig_markers)
            avg_sig_per_cluster = total_sig / n_clusters if n_clusters > 0 else 0
        else:
            sig_count_per_cluster = {}
            total_sig = 0
            avg_sig_per_cluster = 0

        state_delta = {
            "adata_path": str(checkpoint_path),  # Update to point to checkpoint with marker results
            "n_clusters_with_markers": n_clusters,
            "marker_detection_method": method,
            "n_genes_per_cluster": n_genes,
            "significant_markers_per_cluster": sig_count_per_cluster,
            "top_markers": top_markers_dict,
            "marker_detection_complete": True
        }

        if species:
            state_delta["marker_species"] = species

        # Generate summary message
        message = (
            f"✅ Marker gene detection complete!\n"
            f"📊 Detected markers for {n_clusters} clusters using {method}\n"
            f"🧬 Top {n_genes} genes per cluster identified\n"
        )

        if has_pvals_adj:
            message += (
                f"⭐ {total_sig} significant markers (padj < 0.05)\n"
                f"📈 Average {avg_sig_per_cluster:.1f} significant markers per cluster\n"
            )

        message += f"📁 Artifacts: {len(artifacts)} files saved"

        if species:
            message = f"{message}\n🔬 Species: {species}"

        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=artifacts,
            citations=[
                "Wolf et al. (2018) Genome Biology - Scanpy",
                "Soneson & Robinson (2018) - Differential expression methods"
            ]
        )

    except Exception as e:
        return ToolResult(
            message=f"❌ Marker gene detection failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )
