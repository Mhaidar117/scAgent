"""Differential expression analysis tools for cluster comparisons."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Union

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


def _create_volcano_plot(
    de_df: pd.DataFrame,
    output_path: Path,
    group1: str,
    group2: str,
    logfc_threshold: float = 1.0,
    pval_threshold: float = 0.05
) -> None:
    """Create volcano plot for DE results.

    Args:
        de_df: DataFrame with logfoldchange and pval_adj columns
        output_path: Path to save plot
        group1: Name of first group
        group2: Name of second group
        logfc_threshold: Log fold change threshold for significance
        pval_threshold: Adjusted p-value threshold
    """
    if not PLOTTING_AVAILABLE:
        return

    # Create -log10(pval)
    de_df['-log10(padj)'] = -np.log10(de_df['pval_adj'].clip(lower=1e-300))

    # Categorize genes
    de_df['category'] = 'Not significant'
    de_df.loc[
        (de_df['logfoldchange'].abs() >= logfc_threshold) &
        (de_df['pval_adj'] < pval_threshold),
        'category'
    ] = 'Significant'

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'Not significant': 'gray', 'Significant': 'red'}

    for cat, color in colors.items():
        subset = de_df[de_df['category'] == cat]
        ax.scatter(
            subset['logfoldchange'],
            subset['-log10(padj)'],
            c=color,
            alpha=0.6,
            s=10,
            label=cat
        )

    # Add threshold lines
    ax.axhline(-np.log10(pval_threshold), ls='--', c='black', alpha=0.5, label=f'padj = {pval_threshold}')
    ax.axvline(logfc_threshold, ls='--', c='black', alpha=0.5)
    ax.axvline(-logfc_threshold, ls='--', c='black', alpha=0.5, label=f'|log2FC| = {logfc_threshold}')

    ax.set_xlabel('Log2 Fold Change', fontsize=12)
    ax.set_ylabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title(f'Volcano Plot: {group1} vs {group2}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_clusters(
    state: SessionState,
    cluster_key: str = "leiden",
    group1: Union[str, List[str]] = None,
    group2: Union[str, List[str]] = None,
    method: Literal["t-test", "wilcoxon", "logreg"] = "wilcoxon",
    use_raw: bool = False,
    n_genes: int = 100,
    logfc_threshold: float = 1.0,
    pval_threshold: float = 0.05
) -> ToolResult:
    """Perform differential expression analysis between cluster groups.

    Compares gene expression between two groups of clusters to identify
    differentially expressed genes. Supports both pairwise cluster comparisons
    and comparisons between cluster groups.

    Args:
        state: Current session state
        cluster_key: Column in adata.obs containing cluster assignments
        group1: Cluster ID(s) for first group (e.g., "0" or ["0", "1"])
        group2: Cluster ID(s) for second group (e.g., "rest" or ["2", "3"])
        method: Statistical test:
            - "wilcoxon": Wilcoxon rank-sum (default, non-parametric)
            - "t-test": Welch's t-test (parametric)
            - "logreg": Logistic regression (multivariate)
        use_raw: Use raw counts if available
        n_genes: Number of top DE genes to report
        logfc_threshold: Log2 fold change threshold for significance
        pval_threshold: Adjusted p-value threshold

    Returns:
        ToolResult with DE genes table, volcano plot, and summary statistics
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

        # Check cluster key exists
        if cluster_key not in adata.obs.columns:
            available = [k for k in adata.obs.columns if 'leiden' in k or 'louvain' in k]
            return ToolResult(
                message=f"❌ Cluster key '{cluster_key}' not found. Available: {available}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Validate groups
        if group1 is None or group2 is None:
            return ToolResult(
                message="❌ Both group1 and group2 must be specified",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_15_differential_expression"
        step_dir = ensure_run_dir(step_dir_path)

        artifacts = []

        # Normalize group inputs
        if isinstance(group1, str):
            group1 = [group1]
        if isinstance(group2, str):
            group2 = [group2] if group2 != "rest" else "rest"

        # Create group labels
        group1_label = "+".join(group1) if isinstance(group1, list) else group1
        group2_label = "+".join(group2) if isinstance(group2, list) else group2

        # Subset data by groups
        if isinstance(group1, list):
            mask1 = adata.obs[cluster_key].isin(group1)
        else:
            mask1 = adata.obs[cluster_key] == group1

        if group2 == "rest":
            mask2 = ~mask1
        elif isinstance(group2, list):
            mask2 = adata.obs[cluster_key].isin(group2)
        else:
            mask2 = adata.obs[cluster_key] == group2

        # Check group sizes
        n_group1 = mask1.sum()
        n_group2 = mask2.sum()

        if n_group1 == 0 or n_group2 == 0:
            return ToolResult(
                message=f"❌ One or both groups are empty. Group1: {n_group1} cells, Group2: {n_group2} cells",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Create temporary grouping column
        adata.obs['_de_group'] = 'other'
        adata.obs.loc[mask1, '_de_group'] = 'group1'
        adata.obs.loc[mask2, '_de_group'] = 'group2'

        # Subset to only groups of interest
        adata_subset = adata[mask1 | mask2, :].copy()

        # Run differential expression
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby='_de_group',
            groups=['group1'],
            reference='group2',
            method=method,
            use_raw=use_raw,
            key_added='de_results'
        )

        # Extract results
        result = adata_subset.uns['de_results']
        group_names = result['names'].dtype.names

        # Build results table
        de_genes = []
        for i in range(min(n_genes, len(result['names']['group1']))):
            gene = result['names']['group1'][i]
            score = float(result['scores']['group1'][i])

            record = {
                'gene': gene,
                'score': score,
                'rank': i + 1
            }

            # Add method-specific fields
            if 'logfoldchanges' in result:
                record['logfoldchange'] = float(result['logfoldchanges']['group1'][i])
            if 'pvals' in result:
                record['pval'] = float(result['pvals']['group1'][i])
            if 'pvals_adj' in result:
                record['pval_adj'] = float(result['pvals_adj']['group1'][i])

            de_genes.append(record)

        de_df = pd.DataFrame(de_genes)

        # Save CSV
        csv_path = step_dir / f"de_genes_{group1_label}_vs_{group2_label}.csv"
        de_df.to_csv(csv_path, index=False)
        artifacts.append(str(csv_path))

        # Count significant genes
        if 'pval_adj' in de_df.columns and 'logfoldchange' in de_df.columns:
            n_significant = len(de_df[
                (de_df['pval_adj'] < pval_threshold) &
                (de_df['logfoldchange'].abs() >= logfc_threshold)
            ])
        else:
            n_significant = 0

        # Generate volcano plot
        if PLOTTING_AVAILABLE and 'pval_adj' in de_df.columns and 'logfoldchange' in de_df.columns:
            volcano_path = step_dir / f"volcano_{group1_label}_vs_{group2_label}.png"
            _create_volcano_plot(de_df, volcano_path, group1_label, group2_label, logfc_threshold, pval_threshold)
            artifacts.append(str(volcano_path))

        # Generate heatmap of top DE genes
        if PLOTTING_AVAILABLE:
            top_genes = de_df['gene'].head(min(50, len(de_df))).tolist()

            heatmap_path = step_dir / f"heatmap_{group1_label}_vs_{group2_label}.png"

            # Use scanpy's heatmap with save parameter
            sc.pl.heatmap(
                adata_subset,
                var_names=top_genes,
                groupby='_de_group',
                show=False,
                cmap='RdBu_r',
                dendrogram=False,
                figsize=(12, 10),
                save=f"_{group1_label}_vs_{group2_label}.png"
            )

            # Move file to correct location (scanpy saves to figures/)
            import os
            scanpy_path = Path("figures") / f"heatmap_{group1_label}_vs_{group2_label}.png"
            if scanpy_path.exists():
                import shutil
                shutil.move(str(scanpy_path), str(heatmap_path))
                # Clean up figures directory if empty
                if scanpy_path.parent.exists() and not any(scanpy_path.parent.iterdir()):
                    scanpy_path.parent.rmdir()

            if heatmap_path.exists():
                artifacts.append(str(heatmap_path))

        # Clean up temporary column
        del adata.obs['_de_group']

        # Save summary stats
        summary = {
            "group1": group1_label,
            "group2": group2_label,
            "n_cells_group1": int(n_group1),
            "n_cells_group2": int(n_group2),
            "method": method,
            "n_de_genes_tested": len(de_df),
            "n_significant_genes": int(n_significant),
            "logfc_threshold": logfc_threshold,
            "pval_threshold": pval_threshold,
            "top_upregulated": de_df[de_df.get('logfoldchange', 0) > 0].head(5)['gene'].tolist() if 'logfoldchange' in de_df.columns else [],
            "top_downregulated": de_df[de_df.get('logfoldchange', 0) < 0].head(5)['gene'].tolist() if 'logfoldchange' in de_df.columns else []
        }

        summary_path = step_dir / f"de_summary_{group1_label}_vs_{group2_label}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        artifacts.append(str(summary_path))

        # Update state
        state_delta = {
            "de_group1": group1_label,
            "de_group2": group2_label,
            "de_n_significant": int(n_significant),
            "de_method": method,
            "de_complete": True
        }

        # Create checkpoint for differential expression results
        # Use the first artifact (CSV) as the checkpoint reference
        checkpoint_path = step_dir / f"de_checkpoint_{group1_label}_vs_{group2_label}.h5ad"
        state.checkpoint(str(checkpoint_path), f"differential_expression_{group1_label}_vs_{group2_label}")

        # Add artifacts to state
        for artifact in artifacts:
            label = Path(artifact).stem.replace('_', ' ').title()
            state.add_artifact(artifact, label)

        citations = [
            "Wolf et al. (2018) Genome Biology - SCANPY",
            f"{method.capitalize()} test for differential expression"
        ]

        message = (
            f"✅ Differential expression analysis complete!\\n"
            f"📊 Comparison: {group1_label} (n={n_group1}) vs {group2_label} (n={n_group2})\\n"
            f"🧬 Method: {method}\\n"
            f"📈 Significant genes: {n_significant} / {len(de_df)}\\n"
            f"📁 Artifacts: {len(artifacts)} files saved\\n"
        )

        if 'logfoldchange' in de_df.columns and len(de_df) > 0:
            top_gene = de_df.iloc[0]
            message += f"\\nTop DE gene: {top_gene['gene']} (log2FC = {top_gene['logfoldchange']:.2f})"

        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=artifacts,
            citations=citations
        )

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return ToolResult(
            message=f"❌ Differential expression analysis failed: {str(e)}\\n\\nTraceback:\\n{tb_str}",
            state_delta={},
            artifacts=[],
            citations=[]
        )
