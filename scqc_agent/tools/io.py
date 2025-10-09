"""I/O tools for loading and saving AnnData files."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..state import ToolResult

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


def load_anndata(path: str) -> ToolResult:
    """Load AnnData file with real Scanpy support."""
    file_path = Path(path)

    if not file_path.exists():
        return ToolResult(
            message=f"Error: File not found at {path}",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    if not SCANPY_AVAILABLE:
        # Fallback to Phase 0 behavior if scanpy not installed
        if not path.endswith(('.h5ad', '.h5', '.h5ad.gz', '.h5.gz')):
            return ToolResult(
                message=f"Warning: File {path} may not be a valid AnnData format (.h5ad expected). Install scanpy for full support.",
                state_delta={"adata_path": path},
                artifacts=[],
                citations=[]
            )

        return ToolResult(
            message=f"Successfully validated AnnData file at {path}. Install scanpy for Phase 1 processing.",
            state_delta={"adata_path": path},
            artifacts=[],
            citations=[]
        )

    # Phase 1: Real AnnData loading - sc.read_h5ad handles both .h5ad and .h5ad.gz
    try:
        adata = sc.read_h5ad(path)
        n_obs, n_vars = adata.shape

        # Basic dataset summary
        dataset_summary = {
            "n_cells": n_obs,
            "n_genes": n_vars,
            "file_path": str(file_path.absolute()),
            "loaded_timestamp": datetime.now().isoformat()
        }

        return ToolResult(
            message=f"Successfully loaded AnnData with {n_obs:,} cells and {n_vars:,} genes from {path}",
            state_delta={
                "adata_path": path,
                "dataset_summary": dataset_summary
            },
            artifacts=[],
            citations=["Wolf et al. (2018) Genome Biology"]
        )

    except Exception as e:
        return ToolResult(
            message=f"Error loading AnnData file: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def ensure_run_dir(run_dir_or_state, step_name: str = None) -> Path:
    """Ensure run directory exists and return Path object.
    
    Args:
        run_dir_or_state: Either full path string OR SessionState object
        step_name: Step name for directory (when run_dir_or_state is SessionState)
    """
    if step_name is not None:
        # Two-argument pattern: ensure_run_dir(state, "step_name")
        if hasattr(run_dir_or_state, 'run_id'):
            # SessionState object
            run_path = Path(f"runs/{run_dir_or_state.run_id}/{step_name}")
        else:
            # run_id string
            run_path = Path(f"runs/{run_dir_or_state}/{step_name}")
    else:
        # Single-argument pattern: ensure_run_dir("full/path")
        run_path = Path(run_dir_or_state)
    
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def save_snapshot(label: str, run_dir: str = None, adata: Optional[object] = None) -> ToolResult:
    """Save AnnData snapshot with real .h5ad support in Phase 1."""
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"runs/{timestamp}"
    
    # Create run directory
    run_path = ensure_run_dir(run_dir)
    snapshot_path = run_path / f"snapshot_{label}.h5ad"
    
    if not SCANPY_AVAILABLE or adata is None:
        # Fallback to Phase 0 behavior
        with open(snapshot_path, 'w') as f:
            f.write(f"# Placeholder snapshot: {label}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write("# Phase 1 will replace this with actual AnnData .h5ad file\n")
        
        return ToolResult(
            message=f"Created placeholder snapshot: {snapshot_path}",
            state_delta={"run_dir": run_dir},
            artifacts=[snapshot_path],
            citations=[]
        )
    
    # Phase 1: Real .h5ad saving
    try:
        adata.write_h5ad(snapshot_path)
        n_obs, n_vars = adata.shape
        
        return ToolResult(
            message=f"Saved AnnData snapshot ({n_obs:,} cells, {n_vars:,} genes) to {snapshot_path}",
            state_delta={"run_dir": run_dir},
            artifacts=[snapshot_path],
            citations=[]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"Error saving snapshot: {str(e)}",
            state_delta={"run_dir": run_dir},
            artifacts=[],
            citations=[]
        )
