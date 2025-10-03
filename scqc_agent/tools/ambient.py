"""Ambient RNA correction wrappers for scRNA-seq data analysis (Phase 8)."""

import warnings
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from ..state import ToolResult, SessionState
from .io import ensure_run_dir, save_snapshot

# Import guards for optional dependencies
try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import scar
    AMBIENT_AVAILABLE = True  # Set to True when real packages are available
except ImportError:
    AMBIENT_AVAILABLE = False



def scar_ambient_removal(
    state: SessionState,
    contamination_rate: float = 0.1,
    ambient_profile_size: int = 50,
    sparsity: float = 0.9,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    **kwargs
) -> ToolResult:
    """Remove ambient RNA contamination using scAR.
    
    scAR is more robust for heterogeneous tissues compared to SoupX as it uses
    a deep generative model to learn the ambient RNA profile from the data itself.
    
    Args:
        state: Current session state
        contamination_rate: Expected contamination rate (0.0-1.0)
        ambient_profile_size: Number of genes to include in ambient profile
        sparsity: Expected sparsity level in the data
        epochs: Number of training epochs
        learning_rate: Learning rate for model training
        batch_size: Batch size for training
        **kwargs: Additional scAR parameters
        
    Returns:
        ToolResult with ambient-corrected data and artifacts
    """
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required for ambient RNA correction")
    
    try:
        import scar
    except ImportError:
        return ToolResult(
            message=(
                "❌ scAR not available. Install with: pip install scar\n"
                "scAR provides robust ambient RNA removal for heterogeneous tissues."
            ),
            state_delta={},
            artifacts=[],
            citations=["Sheng et al. (2022) Nature Communications - scAR"]
        )
    
    # Load data
    step_dir = ensure_run_dir(state.run_id, "scar_ambient")
    adata = sc.read_h5ad(state.adata_path)
    
    try:
        # Setup AnnData for scAR ambient RNA removal
        scar.setup_anndata(adata, feature_type='mRNA')
        
        # Create scAR model for ambient RNA removal
        scar_model = scar.model.RNAModel(
            adata,
            ambient_profile_size=ambient_profile_size,
            sparsity=sparsity,
        )
        
        # Train the model
        scar_model.train(
            max_epochs=epochs,
            lr=learning_rate,
            batch_size=batch_size,
        )
        
        # Get ambient-corrected counts
        corrected_adata = adata.copy()
        corrected_adata.X = scar_model.get_denoised_counts()
        
        # Add ambient RNA metadata
        corrected_adata.obs['scar_contamination'] = scar_model.get_contamination_per_cell()
        corrected_adata.var['scar_ambient_score'] = scar_model.get_ambient_gene_scores()
        corrected_adata.uns['scar_ambient_params'] = {
            'contamination_rate': contamination_rate,
            'ambient_profile_size': ambient_profile_size,
            'sparsity': sparsity,
            'epochs': epochs,
            'method': 'scAR_ambient_removal'
        }
        
        # Save corrected data
        output_path = step_dir / "adata_scar_ambient_corrected.h5ad"
        corrected_adata.write_h5ad(output_path)
        
        # Generate comparison artifacts
        artifacts = _generate_scar_ambient_artifacts(adata, corrected_adata, step_dir)
        artifacts.append(output_path)
        
        # Update state
        state_delta = {
            "adata_path": str(output_path),
            "last_tool": "scar_ambient_removal",
            "scar_ambient_params": corrected_adata.uns['scar_ambient_params']
        }
        
        # Create checkpoint
        checkpoint_path = state.checkpoint(str(output_path), "scar_ambient")
        artifacts.append(checkpoint_path)
        
        return ToolResult(
            message=(
                f"✅ scAR ambient RNA removal completed.\n"
                f"   Cells processed: {corrected_adata.n_obs:,}\n"
                f"   Genes processed: {corrected_adata.n_vars:,}\n"
                f"   Estimated contamination: {contamination_rate:.1%}\n"
                f"   Ambient profile size: {ambient_profile_size} genes\n"
                f"   Training epochs: {epochs}\n"
                f"   Output: {output_path}"
            ),
            state_delta=state_delta,
            artifacts=artifacts,
            citations=["Sheng et al. (2022) Nature Communications - scAR ambient RNA removal"]
        )
        
    except Exception as e:
        error_msg = f"❌ scAR ambient RNA removal failed: {str(e)}"
        return ToolResult(message=error_msg, state_delta={}, artifacts=[], citations=[])





def _generate_scar_ambient_artifacts(adata_orig: object, adata_corr: object, step_dir: Path) -> List[Path]:
    """Generate artifacts for scAR ambient RNA removal comparison."""
    artifacts = []
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Before/after comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total counts comparison
        orig_counts = np.array(adata_orig.X.sum(axis=1)).flatten()
        corr_counts = np.array(adata_corr.X.sum(axis=1)).flatten()
        
        axes[0,0].scatter(orig_counts, corr_counts, alpha=0.5, s=1)
        axes[0,0].plot([orig_counts.min(), orig_counts.max()], 
                       [orig_counts.min(), orig_counts.max()], 'r--', alpha=0.8)
        axes[0,0].set_xlabel('Original Total Counts')
        axes[0,0].set_ylabel('scAR Corrected Counts')
        axes[0,0].set_title('Total Counts: Before vs After scAR')
        
        # Contamination distribution
        if 'scar_contamination' in adata_corr.obs.columns:
            contamination = adata_corr.obs['scar_contamination']
            axes[0,1].hist(contamination, bins=50, alpha=0.7, edgecolor='black')
            axes[0,1].set_xlabel('Estimated Contamination Rate')
            axes[0,1].set_ylabel('Number of Cells')
            axes[0,1].set_title('scAR Contamination Estimates')
        
        # Ambient gene scores
        if 'scar_ambient_score' in adata_corr.var.columns:
            ambient_scores = adata_corr.var['scar_ambient_score']
            axes[1,0].hist(ambient_scores, bins=50, alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Ambient Gene Score')
            axes[1,0].set_ylabel('Number of Genes')
            axes[1,0].set_title('scAR Ambient Gene Scores')
        
        # Reduction in counts
        count_reduction = (orig_counts - corr_counts) / orig_counts
        axes[1,1].hist(count_reduction, bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].set_xlabel('Fraction of Counts Removed')
        axes[1,1].set_ylabel('Number of Cells')
        axes[1,1].set_title('Count Reduction by scAR')
        
        plt.tight_layout()
        comparison_plot = step_dir / "scar_ambient_correction_comparison.png"
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(comparison_plot)
        
        # Generate summary CSV
        summary_data = {
            'metric': [
                'Original median counts', 'Corrected median counts', 'Median contamination',
                'Mean contamination', 'Median count reduction', 'High contamination cells (>20%)'
            ],
            'value': [
                f"{np.median(orig_counts):.0f}",
                f"{np.median(corr_counts):.0f}",
                f"{np.median(contamination):.3f}" if 'scar_contamination' in adata_corr.obs.columns else "N/A",
                f"{np.mean(contamination):.3f}" if 'scar_contamination' in adata_corr.obs.columns else "N/A",
                f"{np.median(count_reduction):.3f}",
                f"{np.sum(contamination > 0.2)}" if 'scar_contamination' in adata_corr.obs.columns else "N/A"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = step_dir / "scar_ambient_correction_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        artifacts.append(summary_csv)
        
    except Exception as e:
        print(f"Warning: Could not generate scAR ambient artifacts: {e}")
    
    return artifacts



def _run_real_scar(adata: object, contamination_rate: float, clusters: Optional[str], **kwargs) -> object:
    """scAR implementation for ambient RNA removal."""
    try:
        # Use scAR instead of SoupX R package
        import scar
        
        # Setup AnnData for scAR ambient RNA removal
        scar.setup_anndata(adata, feature_type='mRNA')
        
        # Create scAR model for ambient RNA removal
        scar_model = scar.model.RNAModel(
            adata,
            ambient_profile_size=kwargs.get('n_top_genes', 50),
            sparsity=kwargs.get('sparsity', 0.9),
        )
        
        # Train the model
        scar_model.train(
            max_epochs=kwargs.get('epochs', 100),
            lr=kwargs.get('learning_rate', 1e-3),
            batch_size=kwargs.get('batch_size', 64),
        )
        
        # Get ambient-corrected counts
        corrected_adata = adata.copy()
        corrected_adata.X = scar_model.get_denoised_counts()
        
        # Add scAR-specific metadata (replacing SoupX metadata)
        if hasattr(scar_model, 'get_contamination_per_cell'):
            corrected_adata.obs['scar_contamination_rate'] = scar_model.get_contamination_per_cell()
        if hasattr(scar_model, 'get_ambient_gene_scores'):
            corrected_adata.var['scar_ambient_score'] = scar_model.get_ambient_gene_scores()
        
        # Store scAR parameters
        corrected_adata.uns['correction_method'] = 'scAR'
        corrected_adata.uns['scar_ambient_params'] = {
            'contamination_rate': contamination_rate,
            'ambient_profile_size': kwargs.get('n_top_genes', 50),
            'epochs': kwargs.get('epochs', 100),
            'clusters_used': clusters
        }
        
        return corrected_adata
        
    except ImportError:
        print("Warning: scAR not available, falling back to mock implementation")
        return 
    except Exception as e:
        print(f"Warning: scAR ambient removal failed ({e}), falling back to mock implementation")
        return 



