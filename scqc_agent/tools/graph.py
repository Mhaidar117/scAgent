"""Graph analysis tools for scRNA-seq data analysis."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..state import ToolResult, SessionState
from ..quality.assertions import (
    assert_neighbors_nonempty,
    assert_latent_shape,
    assert_clustering_quality,
    QualityGateError
)
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
        raise ImportError("Scanpy is required for graph operations. Install with: pip install scanpy")
    
    adata_path = state.adata_path
    if not adata_path:
        raise ValueError("No AnnData file loaded. Use 'scqc load' first.")
    
    # Load from most recent checkpoint if available
    if state.history:
        last_entry = state.history[-1]
        checkpoint_path = last_entry.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return sc.read_h5ad(checkpoint_path)
    
    # Fall back to original file
    return sc.read(adata_path)


def quick_graph(
    state: SessionState,
    seed: int = 0,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 50
) -> ToolResult:
    """Quick graph analysis pipeline for sanity checks.
    
    Performs PCA → neighbors → UMAP → Leiden clustering on current data.
    This provides a quick sanity check before more complex denoising/integration.
    
    Args:
        state: Current session state
        seed: Random seed for reproducibility
        resolution: Leiden clustering resolution (higher = more clusters)
        n_neighbors: Number of neighbors for kNN graph
        n_pcs: Number of principal components to use
        
    Returns:
        ToolResult with artifacts: umap_pre.png, cluster_counts.csv
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
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Ensure we have log-normalized data for PCA
        if 'log1p' not in adata.uns_keys():
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        # Check if we have enough cells and genes
        n_cells, n_genes = adata.shape
        if n_cells < 10:
            return ToolResult(
                message=f"Too few cells ({n_cells}) for meaningful graph analysis. Need at least 10.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        if n_genes < 50:
            return ToolResult(
                message=f"Too few genes ({n_genes}) for meaningful graph analysis. Need at least 50.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Adjust n_pcs if we have fewer genes
        n_pcs = min(n_pcs, n_genes - 1, n_cells - 1)
        n_neighbors = min(n_neighbors, n_cells - 1)
        
        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_07_quick_graph"
        step_dir = ensure_run_dir(step_dir_path)
        
        # Backup current .obs if it has previous clustering
        original_obs_cols = list(adata.obs.columns)
        
        # 1. PCA
        sc.tl.pca(adata, n_comps=n_pcs, random_state=seed)
        
        # 2. Neighbors graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
        
        # Quality gate: Verify neighbors graph
        try:
            assert_neighbors_nonempty(adata)
            assert_latent_shape(adata, 'X_pca', expected_dims=n_pcs)
        except QualityGateError as e:
            return ToolResult(
                message=f"Quality gate failed: {e}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Check graph connectivity
        if adata.obsp['connectivities'].nnz == 0:
            return ToolResult(
                message="Failed to build neighbors graph - no connections found. Data may be too sparse.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # 3. UMAP
        sc.tl.umap(adata, random_state=seed)
        
        # 4. Leiden clustering
        sc.tl.leiden(adata, resolution=resolution, random_state=seed)
        
        # Quality gate: Verify clustering quality
        try:
            assert_clustering_quality(adata, key='leiden', min_clusters=2)
        except QualityGateError as e:
            return ToolResult(
                message=f"Quality gate failed: {e}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Generate artifacts
        artifacts = []
        
        # Plot UMAP with clusters
        plt.figure(figsize=(10, 8))
        sc.pl.umap(adata, color='leiden', legend_loc='on data', 
                  title=f'Quick Graph (resolution={resolution})', 
                  frameon=False, save=False, show=False)
        
        umap_path = step_dir / "umap_pre.png"
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(umap_path)
        
        # Generate cluster counts
        cluster_counts = adata.obs['leiden'].value_counts().sort_index()
        cluster_df = pd.DataFrame({
            'cluster': cluster_counts.index,
            'n_cells': cluster_counts.values,
            'percentage': (cluster_counts.values / len(adata) * 100).round(2)
        })
        
        counts_path = step_dir / "cluster_counts.csv"
        cluster_df.to_csv(counts_path, index=False)
        artifacts.append(counts_path)
        
        # Save checkpoint
        checkpoint_result = save_snapshot("quick_graph", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_quick_graph.h5ad")
        
        # Update state - convert Path objects to strings
        checkpoint_entry = state.checkpoint(str(checkpoint_path), "quick_graph")
        state.add_artifact(str(umap_path), "UMAP pre-denoising visualization")
        state.add_artifact(str(counts_path), "Cluster counts and percentages")
        
        # Compute summary statistics
        n_clusters = len(cluster_counts)
        largest_cluster = cluster_counts.max()
        largest_cluster_pct = (largest_cluster / len(adata) * 100)
        
        connectivity_rate = (adata.obsp['connectivities'].nnz / 2) / len(adata)
        
        state_delta = {
            "n_clusters": n_clusters,
            "largest_cluster_size": int(largest_cluster),
            "largest_cluster_pct": round(largest_cluster_pct, 2),
            "connectivity_rate": round(connectivity_rate, 2),
            "graph_params": {
                "n_neighbors": n_neighbors,
                "n_pcs": n_pcs,
                "resolution": resolution,
                "seed": seed
            }
        }
        
        message = (
            f"✅ Quick graph analysis complete!\n"
            f"📊 Generated {n_clusters} clusters (resolution={resolution})\n"
            f"🔗 Graph connectivity: {connectivity_rate:.1f} neighbors/cell\n"
            f"📈 Largest cluster: {largest_cluster} cells ({largest_cluster_pct:.1f}%)\n"
            f"📁 Artifacts: {len(artifacts)} files saved"
        )
        
        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Wolf et al. (2018) Genome Biology",
                "McInnes et al. (2018) UMAP",
                "Traag et al. (2019) Leiden"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Graph analysis failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def recompute_neighbors(
    state: SessionState,
    use_rep: str = "X_pca",
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None
) -> ToolResult:
    """Recompute neighbors graph using a different representation.
    
    Args:
        state: Current session state
        use_rep: Representation to use ('X_pca', 'X_scvi', etc.)
        n_neighbors: Number of neighbors
        n_pcs: Number of principal components (if using X_pca)
        
    Returns:
        ToolResult with graph computation status
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
    try:
        adata = _load_adata_from_state(state)
        
        # Check if the representation exists
        if use_rep not in adata.obsm_keys() and use_rep != "X":
            available_reps = list(adata.obsm_keys()) + ["X"]
            return ToolResult(
                message=f"Representation '{use_rep}' not found. Available: {available_reps}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Recompute neighbors
        if use_rep == "X_pca":
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)
        else:
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
        
        # Check connectivity
        connectivity_rate = (adata.obsp['connectivities'].nnz / 2) / len(adata)
        
        message = f"✅ Recomputed neighbors using {use_rep} (connectivity: {connectivity_rate:.1f}/cell)"
        
        return ToolResult(
            message=message,
            state_delta={"connectivity_rate": round(connectivity_rate, 2)},
            artifacts=[],
            citations=["Wolf et al. (2018) Genome Biology"]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Failed to recompute neighbors: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def graph_from_rep(
    state: SessionState,
    use_rep: str,
    seed: int = 0,
    resolution: float = 1.0,
    n_neighbors: int = 15
) -> ToolResult:
    """Perform graph analysis from a specific representation.
    
    Builds neighbors graph, computes UMAP, and performs Leiden clustering
    using a specified representation (e.g., X_scAR, X_scVI, X_pca).
    
    Args:
        state: Current session state
        use_rep: Representation to use from adata.obsm (e.g., 'X_scVI', 'X_scAR')
        seed: Random seed for reproducibility
        resolution: Leiden clustering resolution
        n_neighbors: Number of neighbors for kNN graph
        
    Returns:
        ToolResult with artifacts: umap_<rep>.png, cluster_counts_<rep>.csv
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
        
        # Check if representation exists
        if use_rep not in adata.obsm_keys():
            available_reps = list(adata.obsm_keys())
            return ToolResult(
                message=f"Representation '{use_rep}' not found in adata.obsm. Available: {available_reps}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Get representation data
        rep_data = adata.obsm[use_rep]
        n_cells, n_dims = rep_data.shape
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Validate dimensions
        if n_dims < 2:
            return ToolResult(
                message=f"Representation '{use_rep}' has too few dimensions ({n_dims}). Need at least 2.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        if n_cells < 10:
            return ToolResult(
                message=f"Too few cells ({n_cells}) for meaningful graph analysis. Need at least 10.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Adjust parameters for dataset size
        n_neighbors = min(n_neighbors, n_cells - 1)
        
        # Determine step number based on representation
        step_map = {
            "X_scAR": "09",
            "X_scVI": "10b", 
            "X_pca": "07b"
        }
        step_num = step_map.get(use_rep, "99")
        rep_short = use_rep.replace("X_", "").lower()
        
        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_{step_num}_graph_{rep_short}"
        step_dir = ensure_run_dir(step_dir_path)
        
        # Backup current graph info
        original_obsm_keys = list(adata.obsm.keys())
        original_uns_keys = list(adata.uns.keys())
        
        # 1. Neighbors graph from representation
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=seed)
        
        # Check graph connectivity
        if adata.obsp['connectivities'].nnz == 0:
            return ToolResult(
                message=f"Failed to build neighbors graph from '{use_rep}' - no connections found.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # 2. UMAP from the representation
        sc.tl.umap(adata, random_state=seed)
        
        # 3. Leiden clustering
        sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added=f'leiden_{rep_short}')
        
        # Generate artifacts
        artifacts = []
        
        # Plot UMAP with clusters
        plt.figure(figsize=(10, 8))
        sc.pl.umap(
            adata, 
            color=f'leiden_{rep_short}', 
            legend_loc='on data',
            title=f'Graph from {use_rep} (resolution={resolution})',
            frameon=False, 
            save=False, 
            show=False
        )
        
        umap_path = step_dir / f"umap_{rep_short}.png"
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(umap_path)
        
        # Generate cluster counts
        cluster_counts = adata.obs[f'leiden_{rep_short}'].value_counts().sort_index()
        cluster_df = pd.DataFrame({
            'cluster': cluster_counts.index,
            'n_cells': cluster_counts.values,
            'percentage': (cluster_counts.values / len(adata) * 100).round(2)
        })
        
        counts_path = step_dir / f"cluster_counts_{rep_short}.csv"
        cluster_df.to_csv(counts_path, index=False)
        artifacts.append(counts_path)
        
        # Save checkpoint with updated clustering
        checkpoint_result = save_snapshot(f"graph_{rep_short}", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / f"snapshot_graph_{rep_short}.h5ad")
        
        # Update state - convert Path objects to strings
        state.checkpoint(str(checkpoint_path), f"graph_{rep_short}")
        state.add_artifact(str(umap_path), f"UMAP from {use_rep}")
        state.add_artifact(str(counts_path), f"Cluster counts from {use_rep}")
        
        # Compute summary statistics
        n_clusters = len(cluster_counts)
        largest_cluster = cluster_counts.max()
        largest_cluster_pct = (largest_cluster / len(adata) * 100)
        
        connectivity_rate = (adata.obsp['connectivities'].nnz / 2) / len(adata)
        
        state_delta = {
            f"n_clusters_{rep_short}": n_clusters,
            f"largest_cluster_size_{rep_short}": int(largest_cluster),
            f"largest_cluster_pct_{rep_short}": round(largest_cluster_pct, 2),
            f"connectivity_rate_{rep_short}": round(connectivity_rate, 2),
            f"graph_params_{rep_short}": {
                "use_rep": use_rep,
                "n_neighbors": n_neighbors,
                "n_dims": n_dims,
                "resolution": resolution,
                "seed": seed
            }
        }
        
        message = (
            f"✅ Graph analysis from {use_rep} complete!\n"
            f"📊 Generated {n_clusters} clusters (resolution={resolution})\n"
            f"🔗 Graph connectivity: {connectivity_rate:.1f} neighbors/cell\n"
            f"📈 Largest cluster: {largest_cluster} cells ({largest_cluster_pct:.1f}%)\n"
            f"📐 Representation dims: {n_dims}\n"
            f"📁 Artifacts: {len(artifacts)} files saved"
        )
        
        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Wolf et al. (2018) Genome Biology",
                "McInnes et al. (2018) UMAP", 
                "Traag et al. (2019) Leiden"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Graph analysis from {use_rep} failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def final_graph(
    state: SessionState,
    use_rep: str = "X_scVI",
    resolution: float = 1.0,
    seed: int = 0,
    n_neighbors: int = 15
) -> ToolResult:
    """Perform final graph analysis for the complete pipeline.
    
    This is the culminating graph analysis step that produces the final
    UMAP embedding and cluster assignments for the processed dataset.
    Typically run on the integrated representation (X_scVI) after doublet removal.
    
    Args:
        state: Current session state
        use_rep: Representation to use (default: 'X_scVI')
        resolution: Leiden clustering resolution
        seed: Random seed for reproducibility
        n_neighbors: Number of neighbors for kNN graph
        
    Returns:
        ToolResult with final artifacts: umap_final.png, cluster_counts_final.csv
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
        
        # Check if representation exists
        if use_rep not in adata.obsm_keys():
            available_reps = list(adata.obsm_keys())
            return ToolResult(
                message=f"❌ Representation '{use_rep}' not found in adata.obsm. Available: {available_reps}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Get representation data
        rep_data = adata.obsm[use_rep]
        n_cells, n_dims = rep_data.shape
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Validate dimensions
        if n_dims < 2:
            return ToolResult(
                message=f"❌ Representation '{use_rep}' has too few dimensions ({n_dims}). Need at least 2.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        if n_cells < 10:
            return ToolResult(
                message=f"❌ Too few cells ({n_cells}) for meaningful graph analysis. Need at least 10.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Adjust parameters for dataset size
        n_neighbors = min(n_neighbors, n_cells - 1)
        
        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_12_final_graph"
        step_dir = ensure_run_dir(step_dir_path)
        
        # Backup current graph info
        original_obsm_keys = list(adata.obsm.keys())
        original_uns_keys = list(adata.uns.keys())
        
        # 1. Neighbors graph from representation
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=seed)
        
        # Check graph connectivity
        if adata.obsp['connectivities'].nnz == 0:
            return ToolResult(
                message=f"Failed to build neighbors graph from '{use_rep}' - no connections found.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # 2. Final UMAP embedding
        sc.tl.umap(adata, random_state=seed)
        
        # 3. Final Leiden clustering
        sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added='leiden_final')
        
        # Generate artifacts
        artifacts = []
        
        # Plot final UMAP with clusters
        plt.figure(figsize=(12, 10))
        sc.pl.umap(
            adata, 
            color='leiden_final', 
            legend_loc='on data',
            title=f'Final Clustering from {use_rep} (resolution={resolution})',
            frameon=False, 
            save=False, 
            show=False
        )
        
        umap_path = step_dir / "umap_final.png"
        plt.savefig(umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(umap_path)
        
        # Generate final cluster counts
        cluster_counts = adata.obs['leiden_final'].value_counts().sort_index()
        cluster_df = pd.DataFrame({
            'cluster': cluster_counts.index,
            'n_cells': cluster_counts.values,
            'percentage': (cluster_counts.values / len(adata) * 100).round(2)
        })
        
        counts_path = step_dir / "cluster_counts_final.csv"
        cluster_df.to_csv(counts_path, index=False)
        artifacts.append(counts_path)
        
        # Create additional visualization with batch information if available
        batch_columns = []
        for col in adata.obs.columns:
            if col.lower() in ['batch', 'sample', 'sampleid', 'sample_id', 'donor', 'patient']:
                batch_columns.append(col)
        
        if batch_columns and PLOTTING_AVAILABLE:
            batch_key = batch_columns[0]
            
            # Two-panel plot: clusters and batch
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Panel 1: Clusters
            sc.pl.umap(adata, color='leiden_final', ax=ax1, legend_loc='on data',
                      title=f'Final Clusters (n={len(cluster_counts)})', frameon=False, 
                      save=False, show=False)
            
            # Panel 2: Batch
            sc.pl.umap(adata, color=batch_key, ax=ax2, legend_loc='right margin',
                      title=f'Batch Integration ({batch_key})', frameon=False,
                      save=False, show=False)
            
            plt.tight_layout()
            
            batch_plot_path = step_dir / "umap_final_with_batch.png"
            plt.savefig(batch_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(batch_plot_path)
        
        # Save final checkpoint
        checkpoint_result = save_snapshot("final_graph", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_final_graph.h5ad")
        
        # Update state - convert Path objects to strings
        state.checkpoint(str(checkpoint_path), "final_graph")
        state.add_artifact(str(umap_path), "Final UMAP visualization")
        state.add_artifact(str(counts_path), "Final cluster counts and percentages")
        if len(artifacts) > 2:  # batch plot was created
            state.add_artifact(str(artifacts[2]), "Final UMAP with batch visualization")
        
        # Compute summary statistics
        n_clusters = len(cluster_counts)
        largest_cluster = cluster_counts.max()
        largest_cluster_pct = (largest_cluster / len(adata) * 100)
        smallest_cluster = cluster_counts.min()
        
        connectivity_rate = (adata.obsp['connectivities'].nnz / 2) / len(adata)
        
        # Calculate cluster balance (entropy-based measure)
        cluster_probs = cluster_counts.values / cluster_counts.sum()
        cluster_entropy = -np.sum(cluster_probs * np.log2(cluster_probs))
        max_entropy = np.log2(n_clusters)
        cluster_balance = cluster_entropy / max_entropy if max_entropy > 0 else 0
        
        state_delta = {
            "final_n_clusters": n_clusters,
            "final_largest_cluster_size": int(largest_cluster),
            "final_largest_cluster_pct": round(largest_cluster_pct, 2),
            "final_smallest_cluster_size": int(smallest_cluster),
            "final_connectivity_rate": round(connectivity_rate, 2),
            "final_cluster_balance": round(cluster_balance, 3),
            "final_graph_params": {
                "use_rep": use_rep,
                "n_neighbors": n_neighbors,
                "n_dims": n_dims,
                "resolution": resolution,
                "seed": seed
            },
            "pipeline_complete": True
        }
        
        message = (
            f"🎉 Final graph analysis complete!\n"
            f"📊 Generated {n_clusters} final clusters (resolution={resolution})\n"
            f"🔗 Graph connectivity: {connectivity_rate:.1f} neighbors/cell\n"
            f"📈 Largest cluster: {largest_cluster} cells ({largest_cluster_pct:.1f}%)\n"
            f"📉 Smallest cluster: {smallest_cluster} cells\n"
            f"⚖️  Cluster balance: {cluster_balance:.3f} (0=unbalanced, 1=perfectly balanced)\n"
            f"📐 Representation: {use_rep} ({n_dims} dimensions)\n"
            f"🏁 Pipeline complete: End-to-end analysis finished!\n"
            f"📁 Artifacts: {len(artifacts)} files saved"
        )
        
        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Wolf et al. (2018) Genome Biology",
                "McInnes et al. (2018) UMAP", 
                "Traag et al. (2019) Leiden",
                "Lopez et al. (2018) Nature Methods - scVI"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Final graph analysis failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )
