"""Cell type annotation tools for cluster identification."""

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

try:
    import celltypist
    from celltypist import models
    CELLTYPIST_AVAILABLE = True
except ImportError:
    CELLTYPIST_AVAILABLE = False


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


def _load_marker_database(species: str, tissue: str) -> Optional[Dict[str, List[str]]]:
    """Load marker genes from built-in database.

    Args:
        species: 'human' or 'mouse'
        tissue: Tissue type (e.g., 'brain', 'kidney', 'pbmc')

    Returns:
        Dictionary mapping cell types to marker gene lists
    """
    # Map tissue aliases
    tissue_map = {
        'immune': 'pbmc',
        'blood': 'pbmc',
        'peripheral_blood': 'pbmc'
    }
    tissue = tissue_map.get(tissue.lower(), tissue.lower())

    # Construct path to marker database
    kb_dir = Path(__file__).parent.parent.parent / "kb" / "cell_type_markers"
    marker_file = kb_dir / f"{species}_{tissue}.json"

    if not marker_file.exists():
        return None

    try:
        with open(marker_file, 'r') as f:
            data = json.load(f)

        # Extract just the markers (genes) for each cell type
        markers = {}
        for cell_type, info in data.get('markers', {}).items():
            markers[cell_type] = info.get('genes', [])

        return markers
    except Exception as e:
        print(f"Warning: Could not load marker database: {e}")
        return None


def _compute_marker_scores(adata: object, markers: Dict[str, List[str]]) -> pd.DataFrame:
    """Compute marker scores for each cluster using expression correlation.

    Args:
        adata: AnnData object with cluster assignments
        markers: Dictionary mapping cell types to marker gene lists

    Returns:
        DataFrame with clusters x cell types scores
    """
    # Get cluster labels
    cluster_col = None
    for col in ['leiden', 'leiden_final', 'louvain']:
        if col in adata.obs.columns:
            cluster_col = col
            break

    if cluster_col is None:
        raise ValueError("No clustering found. Run clustering first.")

    # Get unique clusters - convert from Categorical to regular array
    clusters = adata.obs[cluster_col].astype(str).unique()

    # Compute mean expression per cluster
    cluster_means = {}
    for cluster in clusters:
        cluster_mask = adata.obs[cluster_col] == cluster
        cluster_data = adata[cluster_mask, :]

        # Use log-normalized data if available
        if hasattr(adata, 'layers') and 'log1p' in adata.layers:
            expr = cluster_data.layers['log1p']
        elif 'log1p' in adata.uns_keys():
            expr = cluster_data.X
        else:
            # Need to normalize
            expr = cluster_data.X

        # Handle sparse matrices
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()

        cluster_means[cluster] = np.array(expr).mean(axis=0).flatten()

    # Score each cluster against each cell type
    scores = {}
    for cell_type, marker_genes in markers.items():
        # Find marker genes in dataset
        available_markers = [g for g in marker_genes if g in adata.var_names]

        if len(available_markers) == 0:
            continue

        marker_indices = [list(adata.var_names).index(g) for g in available_markers]

        # Compute score for each cluster
        cell_type_scores = []
        for cluster in clusters:
            mean_expr = cluster_means[cluster]
            marker_expr = mean_expr[marker_indices]

            # Score is mean expression of marker genes
            score = np.mean(marker_expr)
            cell_type_scores.append(score)

        scores[cell_type] = cell_type_scores

    # Create DataFrame
    score_df = pd.DataFrame(scores, index=clusters)
    return score_df


def annotate_clusters(
    state: SessionState,
    cluster_key: str = "leiden",
    method: Literal["celltypist", "markers", "auto"] = "auto",
    species: Literal["human", "mouse"] = "human",
    tissue: Optional[str] = None,
    celltypist_model: Optional[str] = None,
    majority_voting: bool = True,
    custom_markers_path: Optional[str] = None
) -> ToolResult:
    """Annotate clusters with cell type labels.

    Hybrid approach:
    - If method='celltypist' and CellTypist available: Use pre-trained models
    - If method='markers' or CellTypist unavailable: Use built-in marker database
    - If method='auto': Try CellTypist first, fall back to markers

    Args:
        state: Current session state
        cluster_key: Column in adata.obs with cluster assignments
        method: Annotation method ('celltypist', 'markers', 'auto')
        species: Species for marker selection
        tissue: Tissue type for marker selection (auto-detected if None)
        celltypist_model: Specific CellTypist model to use
        majority_voting: Use majority voting for CellTypist predictions
        custom_markers_path: Path to custom marker JSON file

    Returns:
        ToolResult with annotations, confidence scores, and visualizations
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

        # Check if clustering exists
        if cluster_key not in adata.obs.columns:
            available = [k for k in adata.obs.columns if 'leiden' in k or 'louvain' in k]
            return ToolResult(
                message=f"❌ Cluster key '{cluster_key}' not found. Available: {available}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        n_clusters = adata.obs[cluster_key].nunique()

        # Auto-detect tissue if not provided
        if tissue is None:
            # Try to infer from metadata
            tissue = state.metadata.get("tissue", "pbmc")  # Default to PBMC

        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_14_annotation"
        step_dir = ensure_run_dir(step_dir_path)

        artifacts = []
        citations = []
        annotation_method_used = None

        # Choose annotation method
        use_celltypist = False
        if method == "celltypist" or method == "auto":
            if CELLTYPIST_AVAILABLE:
                use_celltypist = True
            elif method == "celltypist":
                return ToolResult(
                    message="❌ CellTypist not installed. Install with: pip install celltypist\n"
                           "Or use method='markers' to use built-in marker database.",
                    state_delta={},
                    artifacts=[],
                    citations=[]
                )

        # Method 1: CellTypist
        if use_celltypist:
            try:
                # Auto-select model if not specified
                if celltypist_model is None:
                    model_map = {
                        ('human', 'immune'): 'Immune_All_Low.pkl',
                        ('human', 'pbmc'): 'Immune_All_Low.pkl',
                        ('human', 'blood'): 'Immune_All_Low.pkl',
                        ('mouse', 'brain'): 'Developing_Mouse_Brain.pkl',
                    }
                    celltypist_model = model_map.get((species, tissue), 'Immune_All_Low.pkl')

                # Download model if needed
                models.download_models(model=celltypist_model, force_update=False)

                # Load model
                model = models.Model.load(model=celltypist_model)

                # Annotate
                predictions = celltypist.annotate(
                    adata,
                    model=model,
                    majority_voting=majority_voting
                )

                # Extract predictions
                if majority_voting:
                    adata.obs['cell_type'] = predictions.predicted_labels.majority_voting
                    adata.obs['cell_type_confidence'] = predictions.predicted_labels.conf_score
                else:
                    adata.obs['cell_type'] = predictions.predicted_labels.predicted_labels

                annotation_method_used = f"CellTypist ({celltypist_model})"
                citations.append("Domínguez Conde et al. (2022) Science - CellTypist")

            except Exception as e:
                if method == "celltypist":
                    return ToolResult(
                        message=f"❌ CellTypist annotation failed: {e}",
                        state_delta={},
                        artifacts=[],
                        citations=[]
                    )
                else:
                    # Fall back to markers
                    use_celltypist = False
                    print(f"CellTypist failed, falling back to marker-based annotation: {e}")

        # Method 2: Built-in markers
        if not use_celltypist:
            # Load markers
            if custom_markers_path:
                with open(custom_markers_path, 'r') as f:
                    marker_data = json.load(f)
                markers = marker_data.get('markers', marker_data)
            else:
                markers = _load_marker_database(species, tissue)

            if markers is None or len(markers) == 0:
                return ToolResult(
                    message=f"❌ No markers found for {species} {tissue}. "
                           f"Available: human/mouse brain, kidney, pbmc",
                    state_delta={},
                    artifacts=[],
                    citations=[]
                )

            # Compute scores
            score_df = _compute_marker_scores(adata, markers)

            # Assign cell types based on highest score
            cell_type_assignments = {}
            confidence_scores = {}

            for cluster in score_df.index:
                scores = score_df.loc[cluster]
                best_match = scores.idxmax()
                best_score = scores.max()
                second_best = scores.nlargest(2).iloc[-1] if len(scores) > 1 else 0

                cell_type_assignments[cluster] = best_match
                # Confidence: gap between best and second-best
                confidence_scores[cluster] = best_score - second_best

            # Add to adata
            adata.obs['cell_type'] = adata.obs[cluster_key].map(cell_type_assignments)
            adata.obs['cell_type_confidence'] = adata.obs[cluster_key].map(confidence_scores).astype(float)

            # Save score matrix
            score_csv_path = step_dir / "cell_type_scores.csv"
            score_df.to_csv(score_csv_path)
            artifacts.append(str(score_csv_path))

            annotation_method_used = f"Marker-based ({species} {tissue})"
            citations.extend([
                "Built-in marker database (see kb/cell_type_markers/README.md)",
                "Scanpy marker scoring"
            ])

        # Generate summary CSV
        annotation_summary = adata.obs.groupby([cluster_key, 'cell_type']).size().reset_index()
        annotation_summary.columns = ['cluster', 'cell_type', 'n_cells']

        summary_csv_path = step_dir / "annotation_summary.csv"
        annotation_summary.to_csv(summary_csv_path, index=False)
        artifacts.append(str(summary_csv_path))

        # Generate visualizations
        if PLOTTING_AVAILABLE:
            # 1. Annotated UMAP
            if 'X_umap' in adata.obsm_keys():
                plt.figure(figsize=(12, 10))
                sc.pl.umap(
                    adata,
                    color='cell_type',
                    legend_loc='on data',
                    title=f'Cell Type Annotation ({annotation_method_used})',
                    frameon=False,
                    save=False,
                    show=False
                )

                umap_path = step_dir / "umap_annotated.png"
                plt.savefig(umap_path, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(str(umap_path))

            # 2. Cell type proportions
            fig, ax = plt.subplots(figsize=(10, 6))
            cell_type_counts = adata.obs['cell_type'].value_counts()
            cell_type_counts.plot(kind='barh', ax=ax)
            ax.set_xlabel('Number of cells')
            ax.set_ylabel('Cell type')
            ax.set_title('Cell Type Distribution')
            plt.tight_layout()

            barplot_path = step_dir / "cell_type_distribution.png"
            plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(barplot_path))

        # Save checkpoint with annotations
        checkpoint_result = save_snapshot("annotation", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_annotation.h5ad")

        # Update state
        state.checkpoint(str(checkpoint_path), "annotation")
        for artifact in artifacts:
            label = Path(artifact).stem.replace('_', ' ').title()
            state.add_artifact(artifact, label)

        # Build cell type summary
        cell_types = adata.obs['cell_type'].unique().tolist()
        n_cell_types = len(cell_types)

        # Average confidence
        avg_confidence = adata.obs['cell_type_confidence'].mean() if 'cell_type_confidence' in adata.obs.columns else 0

        state_delta = {
            "adata_path": str(checkpoint_path),  # Update to point to checkpoint with annotations
            "annotation_method": annotation_method_used,
            "n_cell_types_identified": n_cell_types,
            "cell_types": cell_types,
            "avg_annotation_confidence": round(float(avg_confidence), 3),
            "annotation_complete": True,
            "annotation_species": species,
            "annotation_tissue": tissue
        }

        message = (
            f"✅ Cell type annotation complete!\n"
            f"🔬 Method: {annotation_method_used}\n"
            f"🧬 Identified {n_cell_types} cell types across {n_clusters} clusters\n"
            f"📊 Average confidence: {avg_confidence:.3f}\n"
            f"📁 Artifacts: {len(artifacts)} files saved\n"
            f"\nCell types: {', '.join(cell_types[:5])}"
        )

        if len(cell_types) > 5:
            message += f" ... and {len(cell_types) - 5} more"

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
            message=f"❌ Cell type annotation failed: {str(e)}\n\nTraceback:\n{tb_str}",
            state_delta={},
            artifacts=[],
            citations=[]
        )
