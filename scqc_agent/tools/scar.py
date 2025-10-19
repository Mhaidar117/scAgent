"""scAR denoising tools for ambient RNA removal in scRNA-seq data.

This module provides two modes of SCAR ambient RNA correction:
1. With raw data (uses scvi.external.SCAR API + ambient profile estimation)
2. Without raw data (uses standalone scar package)

The dual-mode approach enables optimal denoising when raw droplet data is available
while maintaining backward compatibility for filtered-only datasets.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from ..state import ToolResult, SessionState
from .io import ensure_run_dir, save_snapshot, read_h5ad_with_gz_support

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import scar
    SCAR_AVAILABLE = True
except ImportError:
    SCAR_AVAILABLE = False

try:
    from scvi.external import SCAR as SCVI_SCAR
    SCVI_SCAR_AVAILABLE = True
except ImportError:
    SCVI_SCAR_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _load_adata_from_state(state: SessionState) -> object:
    """Load AnnData object from session state.

    Args:
        state: Current session state

    Returns:
        AnnData object

    Raises:
        ImportError: If scanpy is not available
        ValueError: If no data is loaded in state
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required for scAR operations. Install with: pip install scanpy")

    adata_path = state.adata_path
    if not adata_path:
        raise ValueError("No AnnData file loaded. Use 'scqc load' first.")

    # Load from most recent checkpoint if available
    if state.history:
        last_entry = state.history[-1]
        checkpoint_path = last_entry.get("checkpoint_path")
        if checkpoint_path and Path(checkpoint_path).exists():
            return read_h5ad_with_gz_support(checkpoint_path)

    # Fall back to original file
    return read_h5ad_with_gz_support(adata_path)


def load_raw_data_from_state(state: SessionState) -> Optional[object]:
    """Load raw (unfiltered) AnnData object from session state.

    This function retrieves the raw data path stored by the multiload tool
    and loads the raw droplet data for ambient RNA profiling.

    Args:
        state: Current session state

    Returns:
        AnnData object with raw droplet data, or None if not available

    Examples:
        >>> raw_adata = load_raw_data_from_state(state)
        >>> if raw_adata is not None:
        ...     print(f"Loaded {raw_adata.n_obs} raw droplets")
    """
    if not SCANPY_AVAILABLE:
        return None

    raw_path = state.metadata.get('raw_adata_path')
    if not raw_path:
        return None

    raw_path = Path(raw_path)
    if not raw_path.exists():
        print(f"Warning: Raw data path in state does not exist: {raw_path}")
        return None

    try:
        return read_h5ad_with_gz_support(str(raw_path))
    except Exception as e:
        print(f"Warning: Could not load raw data from {raw_path}: {e}")
        return None


def calculate_ambient_profile(
    raw_adata: object,
    filtered_adata: object,
    min_counts: int = 100
) -> pd.DataFrame:
    """Calculate ambient RNA profile from raw droplet data.

    This function classifies droplets and computes the average expression
    profile of cell-free droplets to estimate ambient RNA contamination.

    Args:
        raw_adata: Raw (unfiltered) AnnData with all droplets
        filtered_adata: Filtered AnnData with cells only
        min_counts: Threshold for cell-free droplets (default: 100)

    Returns:
        DataFrame with ambient profile (mean expression in cell-free droplets)

    Raises:
        ValueError: If raw data is empty or incompatible with filtered data

    Examples:
        >>> ambient_profile = calculate_ambient_profile(raw, filtered, min_counts=100)
        >>> print(f"Ambient genes detected: {(ambient_profile['ambient_profile'] > 0).sum()}")
    """
    # Calculate total counts per droplet
    if hasattr(raw_adata.X, 'sum'):
        # Sparse or dense matrix
        import scipy.sparse as sp
        if sp.issparse(raw_adata.X):
            total_counts = np.ravel(raw_adata.X.sum(axis=1))
        else:
            total_counts = raw_adata.X.sum(axis=1)
    else:
        raise ValueError("raw_adata.X does not support sum operation")

    # Classify droplets
    all_droplets = pd.DataFrame({
        'total_counts': total_counts,
        'droplet_type': 'other droplets'
    }, index=raw_adata.obs_names)

    # Cell-free droplets: below threshold
    all_droplets.loc[all_droplets['total_counts'] < min_counts, 'droplet_type'] = 'cell-free droplets'

    # Cells: in filtered dataset
    all_droplets.loc[all_droplets.index.isin(filtered_adata.obs_names), 'droplet_type'] = 'cells'

    # Extract cell-free droplets
    cellfree_barcodes = all_droplets[all_droplets['droplet_type'] == 'cell-free droplets'].index
    cellfree_adata = raw_adata[cellfree_barcodes, :].copy()

    # Subset to genes in filtered data
    cellfree_adata = cellfree_adata[:, filtered_adata.var_names]

    # Calculate ambient profile: normalized mean expression
    import scipy.sparse as sp
    if sp.issparse(cellfree_adata.X):
        total_ambient_counts = cellfree_adata.X.sum()
        ambient_counts_per_gene = np.array(cellfree_adata.X.sum(axis=0)).flatten()
    else:
        total_ambient_counts = cellfree_adata.X.sum()
        ambient_counts_per_gene = cellfree_adata.X.sum(axis=0)

    ambient_profile = pd.DataFrame({
        'ambient_profile': ambient_counts_per_gene / total_ambient_counts if total_ambient_counts > 0 else ambient_counts_per_gene
    }, index=filtered_adata.var_names)

    return ambient_profile


def generate_knee_plot(
    state: SessionState,
    min_counts: int = 100,
    output_dir: Optional[str] = None
) -> ToolResult:
    """Generate knee plot visualization for droplet distribution.

    This function creates a rank-count plot showing the distribution of total UMI counts
    across all droplets, distinguishing between cells, cell-free droplets, and other droplets.
    It also calculates the ambient RNA profile from cell-free droplets.

    Args:
        state: Current session state (must have raw_adata_path in metadata)
        min_counts: Threshold for classifying cell-free droplets (default: 100)
        output_dir: Optional output directory (defaults to step_08_scar_knee)

    Returns:
        ToolResult with knee plot, ambient profile, and state updates

    Raises:
        ValueError: If raw data is not available in state
        ImportError: If plotting libraries are not available

    State Updates:
        - ambient_profile_path: Path to ambient profile CSV
        - n_cellfree_droplets: Count of cell-free droplets
        - ambient_threshold: min_counts threshold used
        - knee_plot_generated: True

    Examples:
        >>> result = generate_knee_plot(state, min_counts=100)
        >>> print(result.message)
        ✅ Knee plot generated successfully!
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="❌ Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    if not PLOTTING_AVAILABLE:
        return ToolResult(
            message="❌ Plotting libraries not available. Install with: pip install matplotlib seaborn",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Load raw and filtered data
    raw_adata = load_raw_data_from_state(state)
    if raw_adata is None:
        return ToolResult(
            message=(
                "❌ Raw data not available in state.\n"
                "Raw data must be loaded using the multiload tool (load_kidney_data).\n"
                "The raw_adata_path must be set in state.metadata."
            ),
            state_delta={},
            artifacts=[],
            citations=[]
        )

    filtered_adata = _load_adata_from_state(state)

    # Create output directory
    if output_dir is None:
        output_dir = f"runs/{state.run_id}/step_08_scar_knee"
    step_dir = ensure_run_dir(output_dir)

    artifacts = []

    # Calculate total counts per droplet
    import scipy.sparse as sp
    if sp.issparse(raw_adata.X):
        total_counts = np.ravel(raw_adata.X.sum(axis=1))
    else:
        total_counts = raw_adata.X.sum(axis=1)

    # Classify droplets
    all_droplets = pd.DataFrame({
        'total_counts': total_counts,
        'droplet_type': 'other droplets'
    }, index=raw_adata.obs_names)

    all_droplets.loc[all_droplets['total_counts'] < min_counts, 'droplet_type'] = 'cell-free droplets'
    all_droplets.loc[all_droplets.index.isin(filtered_adata.obs_names), 'droplet_type'] = 'cells'

    # Add rank
    all_droplets = all_droplets.sort_values(by='total_counts', ascending=False).reset_index()
    all_droplets['rank'] = range(1, len(all_droplets) + 1)
    all_droplets = all_droplets[all_droplets['total_counts'] > 0]  # Filter zero counts

    # Count droplets by type
    droplet_counts = all_droplets['droplet_type'].value_counts().to_dict()
    n_cellfree = droplet_counts.get('cell-free droplets', 0)
    n_cells = droplet_counts.get('cells', 0)
    n_other = droplet_counts.get('other droplets', 0)

    # Generate knee plot
    plt.figure(figsize=(8, 6), dpi=150)
    ax = sns.lineplot(
        data=all_droplets,
        x='rank',
        y='total_counts',
        hue='droplet_type',
        hue_order=['other droplets', 'cell-free droplets', 'cells'],
        palette=sns.color_palette()[-3:],
        markers=False,
        lw=2
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sorted droplets (rank)')
    ax.set_ylabel('Total UMI counts')
    ax.set_title(f'Knee Plot (threshold={min_counts} UMI)')
    ax.legend(loc='lower left', ncol=1, title=None, frameon=False)

    # Add statistics text
    stats_text = (
        f"Cells: {n_cells:,}\n"
        f"Cell-free: {n_cellfree:,}\n"
        f"Other: {n_other:,}\n"
        f"Total: {len(all_droplets):,}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()

    knee_plot_path = step_dir / "knee_plot.png"
    plt.savefig(knee_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append(str(knee_plot_path))

    # Calculate ambient profile
    ambient_profile = calculate_ambient_profile(raw_adata, filtered_adata, min_counts)

    # Save ambient profile
    ambient_profile_path = step_dir / "ambient_profile.csv"
    ambient_profile.to_csv(ambient_profile_path)
    artifacts.append(str(ambient_profile_path))

    # Save droplet classification
    droplet_summary_path = step_dir / "droplet_classification.csv"
    all_droplets.to_csv(droplet_summary_path, index=False)
    artifacts.append(str(droplet_summary_path))

    # CRITICAL: Create checkpoint BEFORE adding artifacts
    checkpoint_path = step_dir / "checkpoint_knee_plot.h5ad"
    filtered_adata.write_h5ad(checkpoint_path)
    state.checkpoint(str(checkpoint_path), "knee_plot_generated")

    # Add artifacts to checkpoint's history entry
    state.add_artifact(str(knee_plot_path), "Knee Plot")
    state.add_artifact(str(ambient_profile_path), "Ambient Profile CSV")
    state.add_artifact(str(droplet_summary_path), "Droplet Classification CSV")

    # Build state delta
    state_delta = {
        "adata_path": str(checkpoint_path),
        "ambient_profile_path": str(ambient_profile_path),
        "n_cellfree_droplets": int(n_cellfree),
        "n_cells_in_filtered": int(n_cells),
        "n_other_droplets": int(n_other),
        "ambient_threshold": min_counts,
        "knee_plot_generated": True
    }

    message = (
        f"✅ Knee plot generated successfully!\n"
        f"📊 Droplet classification:\n"
        f"   • Cells (in filtered): {n_cells:,}\n"
        f"   • Cell-free droplets (< {min_counts} UMI): {n_cellfree:,}\n"
        f"   • Other droplets: {n_other:,}\n"
        f"   • Total droplets: {len(all_droplets):,}\n"
        f"🧬 Ambient profile calculated from {n_cellfree:,} cell-free droplets\n"
        f"📁 Artifacts: {len(artifacts)} files saved"
    )

    return ToolResult(
        message=message,
        state_delta=state_delta,
        artifacts=artifacts,
        citations=[
            "Lun et al. (2019) Genome Biology - EmptyDrops method",
            "Sheng et al. (2022) Nature Biotechnology - SCAR"
        ]
    )


def run_scar(
    state: SessionState,
    batch_key: str = "SampleID",
    epochs: int = 100,
    replace_X: bool = True,
    random_seed: int = 42,
    use_raw_data: bool = True,
    prob: float = 0.995,
    min_ambient_counts: int = 100
) -> ToolResult:
    """Run scAR denoising for ambient RNA removal.

    This function supports two modes:
    1. With raw data (uses scvi.external.SCAR API + ambient profile estimation)
    2. Without raw data (uses standalone scar package)

    Args:
        state: Current session state
        batch_key: Column in adata.obs for batch information
        epochs: Number of training epochs
        replace_X: Whether to replace X with denoised counts
        random_seed: Random seed for reproducibility
        use_raw_data: Whether to use raw data if available (default: True)
        prob: Probability threshold for ambient profile (scvi.external.SCAR mode, default: 0.995)
        min_ambient_counts: Threshold for cell-free droplets (default: 100)

    Returns:
        ToolResult with training artifacts and denoised data

    State Updates:
        - adata_path: Path to denoised data checkpoint
        - scar_mode: "scvi_scar" or "standalone_scar"
        - scar_epochs: Number of epochs used
        - denoised_total_counts: Total counts after denoising
        - scar_used_raw_data: Whether raw data was used

    Examples:
        >>> # With raw data (optimal)
        >>> result = run_scar(state, use_raw_data=True, prob=0.995)

        >>> # Without raw data (fallback)
        >>> result = run_scar(state, use_raw_data=False)
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="❌ Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    try:
        # Load filtered data
        adata = _load_adata_from_state(state)

        # Detect which API to use
        raw_adata = load_raw_data_from_state(state) if use_raw_data else None
        use_scvi_scar = (raw_adata is not None and SCVI_SCAR_AVAILABLE)

        if use_raw_data and raw_adata is None:
            print("⚠ Raw data requested but not available. Falling back to standalone SCAR.")

        if use_scvi_scar:
            # Mode A: Use scvi.external.SCAR with ambient profile
            return _run_scar_with_scvi(
                state=state,
                adata=adata,
                raw_adata=raw_adata,
                batch_key=batch_key,
                epochs=epochs,
                replace_X=replace_X,
                random_seed=random_seed,
                prob=prob,
                min_ambient_counts=min_ambient_counts
            )
        else:
            # Mode B: Use standalone scar package
            if not SCAR_AVAILABLE:
                return ToolResult(
                    message=(
                        "❌ scAR not available. This is an optional dependency.\n"
                        "To install: pip install scAR\n"
                        "Or install with models extras: pip install .[models]\n"
                        "Note: scAR may require specific PyTorch versions and GPU support."
                    ),
                    state_delta={},
                    artifacts=[],
                    citations=["Sheng et al. (2022) Nature Biotechnology"]
                )

            return _run_scar_standalone(
                state=state,
                adata=adata,
                batch_key=batch_key,
                epochs=epochs,
                replace_X=replace_X,
                random_seed=random_seed
            )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return ToolResult(
            message=f"❌ scAR denoising failed: {str(e)}\n\nDetails:\n{error_details}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def _run_scar_with_scvi(
    state: SessionState,
    adata: object,
    raw_adata: object,
    batch_key: str,
    epochs: int,
    replace_X: bool,
    random_seed: int,
    prob: float,
    min_ambient_counts: int
) -> ToolResult:
    """Run SCAR using scvi.external.SCAR API with raw data.

    This is the optimal mode when raw droplet data is available.
    """
    print("=" * 60)
    print("SCAR Mode: scvi.external.SCAR (with raw data)")
    print("=" * 60)

    n_cells, n_genes = adata.shape

    # Create step directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_dir_path = f"runs/{state.run_id}/step_08_scar"
    model_dir_path = f"models/scar/{timestamp}"

    step_dir = ensure_run_dir(step_dir_path)
    model_dir = ensure_run_dir(model_dir_path)

    artifacts = []

    # Set random seed
    np.random.seed(random_seed)

    # Prepare filtered data copy
    filtered_for_scar = adata.copy()

    # Ensure counts layer exists
    if 'counts' not in filtered_for_scar.layers:
        if 'counts_raw' in filtered_for_scar.layers:
            filtered_for_scar.layers['counts'] = filtered_for_scar.layers['counts_raw'].copy()
        else:
            filtered_for_scar.layers['counts'] = filtered_for_scar.X.copy()

    print(f"[1/6] Setting up SCAR for filtered data...")
    SCVI_SCAR.setup_anndata(filtered_for_scar)

    print(f"[2/6] Estimating ambient profile from raw data (prob={prob})...")
    SCVI_SCAR.get_ambient_profile(adata=filtered_for_scar, raw_adata=raw_adata, prob=prob)
    print(f"      ✓ Ambient profile saved to varm['ambient_profile']")

    print(f"[3/6] Training SCAR model (max_epochs={epochs})...")
    scar_model = SCVI_SCAR(filtered_for_scar, ambient_profile="ambient_profile")
    scar_model.train(max_epochs=epochs)
    print(f"      ✓ Model trained")

    print(f"[4/6] Generating denoised counts...")
    denoised_counts = scar_model.get_denoised_counts()

    print(f"[5/6] Extracting latent representation...")
    latent_rep = scar_model.get_latent_representation()

    # Store results in adata
    adata.layers['counts_denoised'] = denoised_counts
    adata.obsm['X_scar'] = latent_rep

    # Optionally replace X
    if replace_X:
        if 'counts_raw' not in adata.layers:
            adata.layers['counts_raw'] = adata.X.copy()
        adata.X = denoised_counts.copy()
        print(f"[6/6] Replaced X with denoised counts (original preserved in layers['counts_raw'])")
    else:
        print(f"[6/6] Original X preserved")

    # Save model checkpoint
    model_checkpoint_path = model_dir / "model_checkpoint"
    scar_model.save(str(model_checkpoint_path), overwrite=True)
    artifacts.append(str(model_checkpoint_path))

    # Compute statistics
    denoised_total = denoised_counts.sum()

    # Save denoised data checkpoint
    checkpoint_path = step_dir / "adata_scar_denoised.h5ad"
    adata.write_h5ad(checkpoint_path)

    # CRITICAL: Create checkpoint BEFORE adding artifacts
    state.checkpoint(str(checkpoint_path), "scar_denoised")

    # Add artifacts to checkpoint's history entry
    state.add_artifact(str(model_checkpoint_path), "scAR model checkpoint")

    state_delta = {
        "adata_path": str(checkpoint_path),
        "scar_mode": "scvi_scar",
        "scar_epochs": epochs,
        "scar_batch_key": batch_key,
        "scar_prob": prob,
        "denoised_total_counts": float(denoised_total),
        "scar_model_path": str(model_checkpoint_path),
        "replace_X": replace_X,
        "scar_used_raw_data": True
    }

    message = (
        f"✅ scAR denoising complete (scvi.external.SCAR mode)!\n"
        f"🧬 Processed {n_cells} cells with {epochs} epochs\n"
        f"📊 Used raw data for ambient profile estimation (prob={prob})\n"
        f"🧹 Denoised counts stored in layers['counts_denoised']\n"
        f"📐 Latent representation stored in obsm['X_scar']\n"
        f"{'🔄 Original X replaced with denoised counts' if replace_X else '📋 Original X preserved'}\n"
        f"📁 Artifacts: {len(artifacts)} files saved"
    )

    return ToolResult(
        message=message,
        state_delta=state_delta,
        artifacts=[str(p) for p in artifacts],
        citations=[
            "Sheng et al. (2022) Nature Biotechnology",
            "Lopez et al. (2018) Nature Methods - scVI"
        ]
    )


def _run_scar_standalone(
    state: SessionState,
    adata: object,
    batch_key: str,
    epochs: int,
    replace_X: bool,
    random_seed: int
) -> ToolResult:
    """Run SCAR using standalone scar package (no raw data).

    This is the fallback mode when raw data is not available.
    """
    print("=" * 60)
    print("SCAR Mode: standalone scar package (no raw data)")
    print("=" * 60)

    # Validate batch key
    if batch_key not in adata.obs.columns:
        available_keys = list(adata.obs.columns)
        return ToolResult(
            message=f"❌ Batch key '{batch_key}' not found in adata.obs. Available keys: {available_keys}",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Check data requirements
    n_cells, n_genes = adata.shape
    if n_cells < 100:
        return ToolResult(
            message=f"❌ Too few cells ({n_cells}) for scAR denoising. Minimum recommended: 100 cells.",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Create step directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_dir_path = f"runs/{state.run_id}/step_08_scar"
    model_dir_path = f"models/scar/{timestamp}"

    step_dir = ensure_run_dir(step_dir_path)
    model_dir = ensure_run_dir(model_dir_path)

    # Set random seed
    np.random.seed(random_seed)

    artifacts = []

    # Prepare data for scAR
    if adata.X.min() < 0:
        return ToolResult(
            message="❌ scAR requires non-negative count data. Please provide raw count matrix.",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Set up scAR model
    scar.setup_anndata(adata, batch_key=batch_key)

    # Create and train model
    model = scar.model.SCAR(adata)

    # Train with tracking
    training_history = model.train(
        max_epochs=epochs,
        batch_size=min(512, n_cells // 4),
        early_stopping=True,
        check_val_every_n_epoch=5
    )

    # Get denoised counts
    denoised_X = model.get_feature_output()

    # Store denoised representation
    adata.layers['counts_denoised'] = denoised_X
    adata.obsm["X_scAR"] = denoised_X

    # Optionally replace X with denoised counts
    if replace_X:
        if 'counts_raw' not in adata.layers:
            adata.layers['counts_raw'] = adata.X.copy()
        adata.X = denoised_X.copy()

    # Save training history
    if training_history is not None:
        history_df = pd.DataFrame(training_history)
        history_path = step_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        artifacts.append(str(history_path))

        # Plot training loss if available and plotting is available
        if PLOTTING_AVAILABLE and 'train_loss' in history_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(history_df['train_loss'], label='Training Loss')
            if 'val_loss' in history_df.columns:
                plt.plot(history_df['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('scAR Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)

            loss_plot_path = step_dir / "training_loss.png"
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(loss_plot_path))

    # Save model checkpoint
    model_checkpoint_path = model_dir / "model_checkpoint.pt"
    model.save(str(model_checkpoint_path), overwrite=True)
    artifacts.append(str(model_checkpoint_path))

    # Save denoised data checkpoint
    checkpoint_path = step_dir / "adata_scar_denoised.h5ad"
    adata.write_h5ad(checkpoint_path)

    # CRITICAL: Create checkpoint BEFORE adding artifacts
    state.checkpoint(str(checkpoint_path), "scar_denoised")

    # Add artifacts to checkpoint's history entry
    for artifact in artifacts:
        artifact_str = str(artifact)
        if "training_history" in artifact_str:
            state.add_artifact(artifact_str, "scAR training history")
        elif "training_loss" in artifact_str:
            state.add_artifact(artifact_str, "scAR training loss plot")
        elif "model_checkpoint" in artifact_str:
            state.add_artifact(artifact_str, "scAR model checkpoint")

    # Compute denoising statistics
    denoised_total = denoised_X.sum()

    state_delta = {
        "adata_path": str(checkpoint_path),
        "scar_mode": "standalone_scar",
        "scar_epochs": epochs,
        "scar_batch_key": batch_key,
        "denoised_total_counts": float(denoised_total),
        "scar_model_path": str(model_checkpoint_path),
        "replace_X": replace_X,
        "scar_used_raw_data": False
    }

    message = (
        f"✅ scAR denoising complete (standalone mode)!\n"
        f"🧬 Processed {n_cells} cells with {epochs} epochs\n"
        f"📊 Batch key: {batch_key}\n"
        f"⚠ Note: No raw data used (less optimal than scvi.external.SCAR with raw data)\n"
        f"🧹 Denoised representation stored in layers['counts_denoised'] and obsm['X_scAR']\n"
        f"{'🔄 Original X replaced with denoised counts' if replace_X else '📋 Original X preserved'}\n"
        f"📁 Artifacts: {len(artifacts)} files saved"
    )

    return ToolResult(
        message=message,
        state_delta=state_delta,
        artifacts=[str(p) for p in artifacts],
        citations=[
            "Sheng et al. (2022) Nature Biotechnology",
            "Lopez et al. (2018) Nature Methods"
        ]
    )


def get_scar_denoised_data(state: SessionState) -> Optional[object]:
    """Get the most recent scAR-denoised AnnData object.

    Args:
        state: Current session state

    Returns:
        AnnData object with scAR denoising applied, or None if not available
    """
    if not SCANPY_AVAILABLE:
        return None

    try:
        # Look for most recent scAR checkpoint
        for entry in reversed(state.history):
            if entry.get("label") == "scar_denoised":
                checkpoint_path = entry.get("checkpoint_path")
                if checkpoint_path and Path(checkpoint_path).exists():
                    return read_h5ad_with_gz_support(checkpoint_path)

        # Fallback to current data
        adata = _load_adata_from_state(state)
        if "X_scAR" in adata.obsm:
            return adata

        return None

    except Exception:
        return None
