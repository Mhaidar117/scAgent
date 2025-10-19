"""Checkpoint visualization tool for pipeline stage inspection."""

from pathlib import Path
from typing import Optional

from ..state import SessionState, ToolResult
from .io import ensure_run_dir, read_h5ad_with_gz_support

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


def generate_checkpoint_umap(
    state: SessionState,
    stage_label: str,
    layer: Optional[str] = None,
    resolution: float = 2.0,
    n_pcs: int = 40,
    random_seed: int = 42
) -> ToolResult:
    """Generate UMAP visualization at a pipeline checkpoint stage.

    Creates a temporary copy of the data, runs full preprocessing pipeline
    (normalize → log → HVG → scale → PCA → neighbors → leiden → UMAP),
    and saves the UMAP plot. Does not modify the original data or create
    a state checkpoint.

    Args:
        state: Current session state
        stage_label: Stage name (e.g., 'postSCAR', 'postDoublets')
        layer: Layer to use as X (e.g., 'counts_denoised'). If None, uses current X
        resolution: Leiden clustering resolution (0.1-5.0)
        n_pcs: Number of principal components (10-100)
        random_seed: Random seed for reproducibility

    Returns:
        ToolResult with UMAP plot artifact

    Example:
        >>> result = generate_checkpoint_umap(
        ...     state,
        ...     stage_label="postSCAR",
        ...     layer="counts_denoised",
        ...     resolution=2.0
        ... )
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Error: Scanpy is required for checkpoint visualization",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    if not state.adata_path:
        return ToolResult(
            message="Error: No data loaded. Use load_data first.",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Load data
    try:
        adata = read_h5ad_with_gz_support(state.adata_path)
    except Exception as e:
        return ToolResult(
            message=f"Error loading data: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Create output directory
    step_dir = ensure_run_dir(state, f"step_checkpoint_{stage_label}")

    # Copy data to avoid modifying original
    tmp = adata.copy()

    # Use specified layer if provided and exists
    if layer is not None:
        if layer in tmp.layers:
            tmp.X = tmp.layers[layer].copy()
        else:
            return ToolResult(
                message=f"Error: Layer '{layer}' not found in AnnData. Available: {list(tmp.layers.keys())}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

    # Run full preprocessing pipeline on copy
    try:
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
        sc.pp.scale(tmp, max_value=10)
        sc.tl.pca(tmp, n_comps=n_pcs, random_state=random_seed)
        sc.pp.neighbors(tmp, n_pcs=n_pcs, random_state=random_seed)
        sc.tl.leiden(tmp, resolution=resolution, random_state=random_seed)
        sc.tl.umap(tmp, random_state=random_seed)
    except Exception as e:
        return ToolResult(
            message=f"Error during preprocessing: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Save UMAP plot
    try:
        # Configure Scanpy plotting
        sc.settings.figdir = str(step_dir)
        sc.settings.file_format_figs = "pdf"

        # Generate UMAP plot
        # Scanpy saves to: {figdir}/umap{save}.{format}
        # So save=f"_{stage_label}" creates: umap_{stage_label}.pdf
        sc.pl.umap(
            tmp,
            color=["leiden"],
            legend_loc="right margin",
            show=False,
            save=f"_{stage_label}"
        )

        # Scanpy saves to figdir/umap_{stage_label}.pdf
        plot_path = step_dir / f"umap_{stage_label}.pdf"

        # Ensure it was created
        if not plot_path.exists():
            return ToolResult(
                message=f"Error: Plot file not created at {plot_path}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

    except Exception as e:
        return ToolResult(
            message=f"Error generating UMAP plot: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Add artifact (no checkpoint - this is just visualization)
    state.add_artifact(str(plot_path), f"UMAP {stage_label}")

    n_cells, n_genes = tmp.shape
    n_clusters = tmp.obs['leiden'].nunique()

    message = (
        f"Generated checkpoint UMAP for {stage_label} "
        f"({n_cells:,} cells, {n_genes:,} genes, {n_clusters} clusters). "
        f"Plot saved to {plot_path.name}"
    )

    return ToolResult(
        message=message,
        state_delta={},
        artifacts=[str(plot_path)],
        citations=["Wolf et al. (2018) Genome Biology"]
    )
