"""scVI integration tools for batch correction and latent representation learning."""

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
    import scvi
    SCVI_AVAILABLE = True
except ImportError:
    SCVI_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _load_adata_from_state(state: SessionState) -> object:
    """Load AnnData object from session state."""
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required for scVI operations. Install with: pip install scanpy")
    
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


def run_scvi(
    state: SessionState,
    batch_key: str = "SampleID",
    n_latent: int = 30,
    epochs: int = 200,
    random_seed: int = 42
) -> ToolResult:
    """Run scVI for batch correction and latent representation learning.
    
    Args:
        state: Current session state
        batch_key: Column in adata.obs for batch information
        n_latent: Number of latent dimensions
        epochs: Number of training epochs
        random_seed: Random seed for reproducibility
        
    Returns:
        ToolResult with training artifacts and latent representation
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
    if not SCVI_AVAILABLE or not TORCH_AVAILABLE:
        missing = []
        if not SCVI_AVAILABLE:
            missing.append("scvi-tools")
        if not TORCH_AVAILABLE:
            missing.append("torch")
        
        return ToolResult(
            message=(
                f"❌ Required packages not available: {', '.join(missing)}\n"
                "This is an optional dependency for advanced integration.\n"
                "To install: pip install scvi-tools torch\n"
                "Or install with models extras: pip install .[models]\n"
                "Note: May require CUDA for GPU acceleration."
            ),
            state_delta={},
            artifacts=[],
            citations=[
                "Lopez et al. (2018) Nature Methods",
                "Gayoso et al. (2022) Nature Biotechnology"
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
        if n_cells < 200:
            return ToolResult(
                message=f"Too few cells ({n_cells}) for scVI training. Minimum recommended: 200 cells.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        if n_latent >= n_cells:
            n_latent = min(30, n_cells // 2)
            message_suffix = f" (adjusted n_latent to {n_latent} due to small dataset)"
        else:
            message_suffix = ""
        
        # Create step directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_dir_path = f"runs/{state.run_id}/step_10_scvi"
        model_dir_path = f"models/scvi/{timestamp}"
        
        step_dir = ensure_run_dir(step_dir_path)
        model_dir = ensure_run_dir(model_dir_path)
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        scvi.settings.seed = random_seed
        
        artifacts = []
        
        # Prepare data for scVI
        # scVI expects raw counts
        if adata.X.min() < 0:
            return ToolResult(
                message="scVI requires non-negative count data. Please provide raw count matrix.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Setup AnnData for scVI
        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=batch_key,
            layer=None  # Use X (raw counts)
        )
        
        # Create scVI model
        model = scvi.model.SCVI(
            adata,
            n_latent=n_latent,
            n_layers=2,
            n_hidden=128,
            dropout_rate=0.1,
            dispersion="gene",
            gene_likelihood="zinb"
        )
        
        # Train model with tracking
        model.train(
            max_epochs=epochs,
            batch_size=min(512, n_cells // 4),
            early_stopping=True,
            check_val_every_n_epoch=10,
            plan_kwargs={"lr": 1e-3}
        )
        
        # Get training history
        training_history = model.history
        
        # Extract latent representation
        latent_repr = model.get_latent_representation()
        adata.obsm["X_scVI"] = latent_repr
        
        # Save training history
        if training_history is not None and len(training_history) > 0:
            # Convert history to DataFrame
            history_data = []
            for epoch_data in training_history:
                history_data.append(epoch_data)
            
            history_df = pd.DataFrame(history_data)
            history_path = step_dir / "training_history.csv"
            history_df.to_csv(history_path, index=False)
            artifacts.append(history_path)
            
            # Plot training metrics if available
            if PLOTTING_AVAILABLE and len(history_df) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle('scVI Training Metrics', fontsize=16)
                
                # Plot loss curves
                if 'elbo_train' in history_df.columns:
                    axes[0, 0].plot(history_df['elbo_train'], label='Training ELBO')
                    if 'elbo_validation' in history_df.columns:
                        axes[0, 0].plot(history_df['elbo_validation'], label='Validation ELBO')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('ELBO')
                    axes[0, 0].set_title('Evidence Lower Bound')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                
                # Plot reconstruction loss
                if 'reconstruction_loss_train' in history_df.columns:
                    axes[0, 1].plot(history_df['reconstruction_loss_train'], label='Training')
                    if 'reconstruction_loss_validation' in history_df.columns:
                        axes[0, 1].plot(history_df['reconstruction_loss_validation'], label='Validation')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Reconstruction Loss')
                    axes[0, 1].set_title('Reconstruction Loss')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Plot KL divergence
                if 'kl_local_train' in history_df.columns:
                    axes[1, 0].plot(history_df['kl_local_train'], label='Training')
                    if 'kl_local_validation' in history_df.columns:
                        axes[1, 0].plot(history_df['kl_local_validation'], label='Validation')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('KL Divergence')
                    axes[1, 0].set_title('KL Divergence (Local)')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Plot latent representation summary
                axes[1, 1].hist(latent_repr.flatten(), bins=50, alpha=0.7)
                axes[1, 1].set_xlabel('Latent Values')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title(f'Latent Distribution (n_latent={n_latent})')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                metrics_plot_path = step_dir / "training_metrics.png"
                plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(metrics_plot_path)
        
        # Save model checkpoint
        model_checkpoint_path = model_dir / "model_checkpoint"
        model.save(str(model_checkpoint_path), overwrite=True)
        artifacts.append(model_checkpoint_path)
        
        # Save integrated data checkpoint
        checkpoint_result = save_snapshot("scvi_integrated", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_scvi.h5ad")
        
        # Update state
        state.checkpoint(checkpoint_path, "scvi_integrated")
        for artifact in artifacts:
            if "training_history" in str(artifact):
                state.add_artifact(str(artifact), "scVI training history")
            elif "training_metrics" in str(artifact):
                state.add_artifact(str(artifact), "scVI training metrics plot")
            elif "model_checkpoint" in str(artifact):
                state.add_artifact(str(artifact), "scVI model checkpoint")
        
        # Compute integration statistics
        n_batches = adata.obs[batch_key].nunique()
        latent_variance = np.var(latent_repr, axis=0).mean()
        
        state_delta = {
            "scvi_epochs": epochs,
            "scvi_n_latent": n_latent,
            "scvi_batch_key": batch_key,
            "n_batches": n_batches,
            "latent_variance": round(float(latent_variance), 4),
            "scvi_model_path": str(model_checkpoint_path)
        }
        
        message = (
            f"✅ scVI integration complete!{message_suffix}\n"
            f"🧬 Processed {n_cells} cells across {n_batches} batches\n"
            f"📊 Latent dimensions: {n_latent}\n"
            f"🔄 Trained for {epochs} epochs\n"
            f"📈 Latent representation stored in obsm['X_scVI']\n"
            f"📁 Artifacts: {len(artifacts)} files saved"
        )
        
        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Lopez et al. (2018) Nature Methods",
                "Gayoso et al. (2022) Nature Biotechnology",
                "Xu et al. (2021) Genome Biology"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ scVI integration failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def get_scvi_integrated_data(state: SessionState) -> Optional[object]:
    """Get the most recent scVI-integrated AnnData object.
    
    Args:
        state: Current session state
        
    Returns:
        AnnData object with scVI integration applied, or None if not available
    """
    if not SCANPY_AVAILABLE:
        return None
    
    try:
        # Look for most recent scVI checkpoint
        for entry in reversed(state.history):
            if entry.get("label") == "scvi_integrated":
                checkpoint_path = entry.get("checkpoint_path")
                if checkpoint_path and Path(checkpoint_path).exists():
                    return sc.read_h5ad(checkpoint_path)
        
        # Fallback to current data
        adata = _load_adata_from_state(state)
        if "X_scVI" in adata.obsm:
            return adata
        
        return None
        
    except Exception:
        return None
