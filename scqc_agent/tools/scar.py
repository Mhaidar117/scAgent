"""scAR denoising tools for ambient RNA removal in scRNA-seq data."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..state import ToolResult, SessionState
from .io import ensure_run_dir, save_snapshot

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _load_adata_from_state(state: SessionState) -> object:
    """Load AnnData object from session state."""
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
            return sc.read_h5ad(checkpoint_path)
    
    # Fall back to original file
    return sc.read_h5ad(adata_path)


def run_scar(
    state: SessionState,
    batch_key: str = "SampleID",
    epochs: int = 100,
    replace_X: bool = True,
    random_seed: int = 42
) -> ToolResult:
    """Run scAR denoising for ambient RNA removal.
    
    Args:
        state: Current session state
        batch_key: Column in adata.obs for batch information
        epochs: Number of training epochs
        replace_X: Whether to replace X with denoised counts
        random_seed: Random seed for reproducibility
        
    Returns:
        ToolResult with training artifacts and denoised data
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
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
            citations=[
                "Sheng et al. (2022) scAR: Natural Biotechnology"
            ]
        )
    
    try:
        # Load data
        adata = _load_adata_from_state(state)
        
        # Validate batch key
        if batch_key not in adata.obs.columns:
            available_keys = list(adata.obs.columns)
            return ToolResult(
                message=f"Batch key '{batch_key}' not found in adata.obs. Available keys: {available_keys}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Check data requirements
        n_cells, n_genes = adata.shape
        if n_cells < 100:
            return ToolResult(
                message=f"Too few cells ({n_cells}) for scAR denoising. Minimum recommended: 100 cells.",
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
        # scAR typically expects raw counts
        if adata.X.min() < 0:
            return ToolResult(
                message="scAR requires non-negative count data. Please provide raw count matrix.",
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
        adata.obsm["X_scAR"] = denoised_X
        
        # Optionally replace X with denoised counts
        if replace_X:
            adata.X = denoised_X.copy()
        
        # Save training history
        if training_history is not None:
            history_df = pd.DataFrame(training_history)
            history_path = step_dir / "training_history.csv"
            history_df.to_csv(history_path, index=False)
            artifacts.append(history_path)
            
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
                artifacts.append(loss_plot_path)
        
        # Save model checkpoint
        model_checkpoint_path = model_dir / "model_checkpoint.pt"
        model.save(str(model_checkpoint_path), overwrite=True)
        artifacts.append(model_checkpoint_path)
        
        # Save denoised data checkpoint
        checkpoint_result = save_snapshot("scar_denoised", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_scar.h5ad")
        
        # Update state
        state.checkpoint(checkpoint_path, "scar_denoised")
        for artifact in artifacts:
            if "training_history" in str(artifact):
                state.add_artifact(str(artifact), "scAR training history")
            elif "training_loss" in str(artifact):
                state.add_artifact(str(artifact), "scAR training loss plot")
            elif "model_checkpoint" in str(artifact):
                state.add_artifact(str(artifact), "scAR model checkpoint")
        
        # Compute denoising statistics
        original_total = adata.X.sum() if not replace_X else denoised_X.sum()
        denoised_total = denoised_X.sum()
        noise_removed = original_total - denoised_total if not replace_X else 0
        noise_percentage = (noise_removed / original_total * 100) if original_total > 0 and not replace_X else 0
        
        state_delta = {
            "scar_epochs": epochs,
            "scar_batch_key": batch_key,
            "denoised_total_counts": float(denoised_total),
            "noise_removed_pct": round(noise_percentage, 2) if not replace_X else "N/A (X replaced)",
            "scar_model_path": str(model_checkpoint_path),
            "replace_X": replace_X
        }
        
        message = (
            f"✅ scAR denoising complete!\n"
            f"🧬 Processed {n_cells} cells with {epochs} epochs\n"
            f"📊 Batch key: {batch_key}\n"
            f"🧹 Denoised representation stored in obsm['X_scAR']\n"
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
        
    except Exception as e:
        return ToolResult(
            message=f"❌ scAR denoising failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
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
                    return sc.read_h5ad(checkpoint_path)
        
        # Fallback to current data
        adata = _load_adata_from_state(state)
        if "X_scAR" in adata.obsm:
            return adata
        
        return None
        
    except Exception:
        return None
