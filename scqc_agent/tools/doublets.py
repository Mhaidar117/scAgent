"""Doublet detection tools for scRNA-seq data analysis."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal

from ..state import ToolResult, SessionState
from .io import ensure_run_dir, save_snapshot

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

try:
    import scrublet as scr
    SCRUBLET_AVAILABLE = True
except ImportError:
    SCRUBLET_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _load_adata_from_state(state: SessionState) -> object:
    """Load AnnData object from session state."""
    if not SCANPY_AVAILABLE:
        raise ImportError("Scanpy is required for doublet operations. Install with: pip install scanpy")
    
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


def detect_doublets(
    state: SessionState,
    method: Literal["scrublet", "doubletfinder"] = "scrublet",
    expected_rate: float = 0.06,
    threshold: Union[float, Literal["auto"]] = "auto"
) -> ToolResult:
    """Detect doublets in scRNA-seq data.
    
    Args:
        state: Current session state
        method: Doublet detection method ('scrublet' or 'doubletfinder')
        expected_rate: Expected doublet rate (typically 0.06 for 10X data)
        threshold: Doublet score threshold ('auto' for automatic detection or float)
        
    Returns:
        ToolResult with doublet scores and predictions
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )
    
    if method == "scrublet" and not SCRUBLET_AVAILABLE:
        return ToolResult(
            message=(
                "❌ Scrublet not available. This is an optional dependency.\n"
                "To install: pip install scrublet\n"
                "Or install with qc extras: pip install .[qc]\n"
                "Scrublet is the recommended doublet detection method."
            ),
            state_delta={},
            artifacts=[],
            citations=[
                "Wolock et al. (2019) Cell Systems - Scrublet"
            ]
        )
    
    if method == "doubletfinder":
        return ToolResult(
            message=(
                "❌ DoubletFinder not yet implemented.\n"
                "Currently only 'scrublet' method is supported.\n"
                "DoubletFinder would require R integration."
            ),
            state_delta={},
            artifacts=[],
            citations=[
                "McGinnis et al. (2019) Cell Systems - DoubletFinder"
            ]
        )
    
    try:
        # Load data
        adata = _load_adata_from_state(state)
        
        # Check data requirements
        n_cells, n_genes = adata.shape
        if n_cells < 100:
            return ToolResult(
                message=f"Too few cells ({n_cells}) for reliable doublet detection. Minimum recommended: 100 cells.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Validate expected rate
        if not 0.01 <= expected_rate <= 0.5:
            return ToolResult(
                message=f"Expected doublet rate ({expected_rate}) outside reasonable range [0.01, 0.5]",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_11_doublets_detect"
        step_dir = ensure_run_dir(step_dir_path)
        
        artifacts = []
        
        # Run scrublet doublet detection
        if method == "scrublet":
            # Scrublet works best on raw counts
            if hasattr(adata.X, 'toarray'):
                counts_matrix = adata.X.toarray()
            else:
                counts_matrix = adata.X
            
            # Initialize Scrublet
            scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_rate)
            
            # Run doublet detection
            doublet_scores, predicted_doublets = scrub.scrub_doublets(
                min_counts=2, 
                min_cells=3, 
                min_gene_variability_pctl=85,
                n_prin_comps=30
            )
            
            # Call threshold detection explicitly if it hasn't been called
            if not hasattr(scrub, 'threshold_') or scrub.threshold_ is None:
                try:
                    scrub.call_doublets(threshold=None)  # This should set the threshold
                except Exception:
                    pass  # If this fails, we'll use fallback below
            
            # Determine threshold
            if threshold == "auto" or threshold is None:
                # Use scrublet's automatic threshold detection
                detected_threshold = None
                
                # Try multiple ways to get the threshold from scrublet
                # Check various possible threshold attributes
                threshold_attrs = ['threshold_', 'threshold', 'doublet_threshold_', 'call_doublets_threshold_']
                for attr in threshold_attrs:
                    if hasattr(scrub, attr):
                        threshold_val = getattr(scrub, attr)
                        if threshold_val is not None and np.isfinite(threshold_val):
                            detected_threshold = threshold_val
                            break
                    
                # Robust fallback threshold detection
                if detected_threshold is None or not np.isfinite(detected_threshold):
                    # Multiple fallback strategies
                    if doublet_scores is not None and len(doublet_scores) > 0:
                        # Strategy 1: Use percentile based on expected rate
                        percentile_threshold = np.percentile(doublet_scores, 100 * (1 - expected_rate))
                        
                        # Strategy 2: Use a conservative high percentile if no doublets predicted
                        if predicted_doublets.sum() == 0:
                            # When no doublets detected, use a higher percentile as a reasonable threshold
                            conservative_threshold = np.percentile(doublet_scores, 95)
                            # Choose the higher of the two for safety
                            detected_threshold = max(percentile_threshold, conservative_threshold)
                        else:
                            detected_threshold = percentile_threshold
                    else:
                        # Last resort: use a default value
                        detected_threshold = 0.5
                
                # Re-apply threshold to be consistent
                predicted_doublets = doublet_scores > detected_threshold
            else:
                detected_threshold = float(threshold)
                predicted_doublets = doublet_scores > detected_threshold
            
            # Store results in adata
            adata.obs['doublet_score'] = doublet_scores
            adata.obs['doublet'] = predicted_doublets
            
            # Generate doublet score histogram
            if PLOTTING_AVAILABLE:
                plt.figure(figsize=(10, 6))
                
                # Histogram of doublet scores
                plt.hist(doublet_scores, bins=50, alpha=0.7, density=True, label='All cells')
                plt.hist(doublet_scores[predicted_doublets], bins=50, alpha=0.7, 
                        density=True, label='Predicted doublets', color='red')
                
                # Add threshold line
                plt.axvline(detected_threshold, color='red', linestyle='--', 
                          label=f'Threshold: {detected_threshold:.3f}')
                
                plt.xlabel('Doublet Score')
                plt.ylabel('Density')
                plt.title(f'Doublet Score Distribution\n'
                         f'Expected rate: {expected_rate:.1%}, '
                         f'Detected: {predicted_doublets.sum()}/{len(predicted_doublets)} '
                         f'({predicted_doublets.sum()/len(predicted_doublets):.1%})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                hist_path = step_dir / "doublet_score_hist.png"
                plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(hist_path)
        
        # Save snapshot with doublet annotations
        checkpoint_result = save_snapshot("doublets_detected", run_dir=step_dir_path, adata=adata)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_doublets_detected.h5ad")
        
        # Update state with the checkpoint data path (this will be stored as data_path in history)
        state.checkpoint(checkpoint_path, "doublets_detected")
        for artifact in artifacts:
            if "doublet_score_hist" in str(artifact):
                state.add_artifact(str(artifact), "Doublet score histogram")
        
        # Compute statistics
        n_doublets = predicted_doublets.sum()
        doublet_rate = n_doublets / len(predicted_doublets)
        mean_score = doublet_scores.mean()
        median_score = np.median(doublet_scores)
        
        # Final safety check for detected_threshold
        if detected_threshold is None or not np.isfinite(detected_threshold):
            detected_threshold = np.percentile(doublet_scores, 100 * (1 - expected_rate))
            
        # Special handling for zero doublets case
        zero_doublets_detected = (n_doublets == 0)
        
        # Determine expected rate explanation
        if expected_rate == 0.06:
            rate_explanation = "Standard 10X rate (6%)"
        elif expected_rate == 0.08:
            rate_explanation = "High-throughput rate (8%)"
        elif expected_rate == 0.10:
            rate_explanation = "High-density loading (10%)"
        else:
            rate_explanation = f"Custom rate ({expected_rate:.1%})"
        
        try:
            threshold_float = float(detected_threshold)
        except Exception:
            # Force a reasonable value
            threshold_float = 0.5
        
        state_delta = {
            "adata_path": str(checkpoint_path),  # Update to point to checkpoint with doublet annotations
            "doublet_method": method,
            "expected_doublet_rate": expected_rate,
            "detected_doublet_rate": round(doublet_rate, 4),
            "doublet_rate_explanation": rate_explanation,
            "n_doublets": int(n_doublets),
            "n_singlets": int(len(predicted_doublets) - n_doublets),
            "doublet_threshold": round(threshold_float, 4),
            "mean_doublet_score": round(float(mean_score), 4),
            "median_doublet_score": round(float(median_score), 4)
        }
        
        if zero_doublets_detected:
            message = (
                f"✅ Doublet detection complete - No doublets detected!\n"
                f"🔬 Method: {method}\n"
                f"📊 Expected rate: {expected_rate:.1%} ({rate_explanation}), Detected: {doublet_rate:.1%}\n"
                f"🎯 Threshold: {threshold_float:.3f}\n"
                f"👥 All {len(predicted_doublets)} cells classified as singlets\n"
                f"📈 Mean score: {mean_score:.3f}, Median: {median_score:.3f}\n"
                f"📁 Artifacts: {len(artifacts)} files saved\n"
                f"ℹ️  This is normal for high-quality data or when doublets are already filtered"
            )
        else:
            message = (
                f"✅ Doublet detection complete!\n"
                f"🔬 Method: {method}\n"
                f"📊 Expected rate: {expected_rate:.1%} ({rate_explanation}), Detected: {doublet_rate:.1%}\n"
                f"🎯 Threshold: {threshold_float:.3f}\n"
                f"👥 Cells: {n_doublets} doublets, {len(predicted_doublets) - n_doublets} singlets\n"
                f"📈 Mean score: {mean_score:.3f}, Median: {median_score:.3f}\n"
                f"📁 Artifacts: {len(artifacts)} files saved"
            )
        
        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Wolock et al. (2019) Cell Systems",
                "McGinnis et al. (2019) Cell Systems"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Doublet detection failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def apply_doublet_filter(
    state: SessionState,
    threshold: Optional[float] = None
) -> ToolResult:
    """Apply doublet filter to remove detected doublets.
    
    Args:
        state: Current session state
        threshold: Custom threshold for doublet filtering (uses detected threshold if None)
        
    Returns:
        ToolResult with filtered data and removal statistics
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
        
        # Check if doublet detection has been run
        if 'doublet_score' not in adata.obs.columns:
            return ToolResult(
                message="Doublet scores not found. Run 'scqc doublets detect' first.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        # Use custom threshold or existing predictions
        if threshold is not None:
            # Apply custom threshold
            doublet_mask = adata.obs['doublet_score'] > threshold
            adata.obs['doublet'] = doublet_mask
        else:
            # Use existing predictions
            if 'doublet' not in adata.obs.columns:
                return ToolResult(
                    message="Doublet predictions not found. Either run detection first or provide a threshold.",
                    state_delta={},
                    artifacts=[],
                    citations=[]
                )
            doublet_mask = adata.obs['doublet']
            # Get threshold from state or estimate
            threshold = state.metadata.get('doublet_threshold')
            if threshold is None or not np.isfinite(threshold):
                threshold = np.percentile(adata.obs['doublet_score'], 95)
        
        # Create step directory
        step_dir_path = f"runs/{state.run_id}/step_11_doublets_apply"
        step_dir = ensure_run_dir(step_dir_path)
        
        artifacts = []
        
        # Count cells by batch before filtering
        batch_columns = []
        for col in adata.obs.columns:
            if col.lower() in ['batch', 'sample', 'sampleid', 'sample_id', 'donor', 'patient']:
                batch_columns.append(col)
        
        # Use the first batch column found, or create a dummy one
        batch_key = batch_columns[0] if batch_columns else None
        
        if batch_key:
            # Count removals by batch
            removal_stats = []
            for batch_val in adata.obs[batch_key].unique():
                batch_mask = adata.obs[batch_key] == batch_val
                batch_total = batch_mask.sum()
                batch_doublets = (batch_mask & doublet_mask).sum()
                batch_kept = batch_total - batch_doublets
                
                removal_stats.append({
                    'batch': batch_val,
                    'total_cells': batch_total,
                    'doublets_removed': batch_doublets,
                    'cells_kept': batch_kept,
                    'doublet_rate': batch_doublets / batch_total if batch_total > 0 else 0
                })
            
            removal_df = pd.DataFrame(removal_stats)
            removal_path = step_dir / "barplot_removed_by_batch.csv"
            removal_df.to_csv(removal_path, index=False)
            artifacts.append(removal_path)
            
            # Create barplot if plotting available
            if PLOTTING_AVAILABLE:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Barplot of cells kept vs removed
                x_pos = range(len(removal_df))
                ax1.bar(x_pos, removal_df['cells_kept'], label='Kept', alpha=0.8)
                ax1.bar(x_pos, removal_df['doublets_removed'], 
                       bottom=removal_df['cells_kept'], label='Removed (doublets)', alpha=0.8)
                ax1.set_xlabel('Batch')
                ax1.set_ylabel('Number of Cells')
                ax1.set_title('Cells Kept vs Removed by Batch')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(removal_df['batch'], rotation=45)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Doublet rate by batch
                ax2.bar(x_pos, removal_df['doublet_rate'] * 100, alpha=0.8, color='red')
                ax2.set_xlabel('Batch')
                ax2.set_ylabel('Doublet Rate (%)')
                ax2.set_title('Doublet Rate by Batch')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(removal_df['batch'], rotation=45)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                barplot_path = step_dir / "barplot_removed_by_batch.png"
                plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(barplot_path)
        
        # Record original counts
        original_n_cells = adata.shape[0]
        n_doublets = doublet_mask.sum()
        n_kept = original_n_cells - n_doublets
        
        # Filter out doublets
        adata_filtered = adata[~doublet_mask].copy()
        
        # Save filtered data checkpoint
        checkpoint_result = save_snapshot("doublets_filtered", run_dir=step_dir_path, adata=adata_filtered)
        checkpoint_path = checkpoint_result.artifacts[0] if checkpoint_result.artifacts else str(step_dir / "snapshot_doublets_filtered.h5ad")
        
        # Update state
        state.checkpoint(checkpoint_path, "doublets_filtered")
        for artifact in artifacts:
            if "barplot_removed_by_batch.csv" in str(artifact):
                state.add_artifact(str(artifact), "Doublet removal statistics by batch")
            elif "barplot_removed_by_batch.png" in str(artifact):
                state.add_artifact(str(artifact), "Doublet removal barplot by batch")
        
        # Compute final statistics
        final_doublet_rate = n_doublets / original_n_cells
        
        state_delta = {
            "adata_path": str(checkpoint_path),  # Update to point to filtered checkpoint
            "cells_before_doublet_filter": original_n_cells,
            "cells_after_doublet_filter": n_kept,
            "doublets_removed": int(n_doublets),
            "doublet_filter_threshold": round(float(threshold) if threshold is not None else 0.0, 4),
            "final_doublet_rate": round(final_doublet_rate, 4),
            "doublet_filter_applied": True
        }
        
        threshold_str = f"{threshold:.3f}" if threshold is not None else "auto"
        message = (
            f"✅ Doublet filter applied!\n"
            f"🎯 Threshold: {threshold_str}\n"
            f"📊 Removed: {n_doublets} doublets ({final_doublet_rate:.1%})\n"
            f"✅ Kept: {n_kept} singlets ({(1-final_doublet_rate):.1%})\n"
            f"📉 {original_n_cells} → {n_kept} cells\n"
            f"📁 Artifacts: {len(artifacts)} files saved"
        )
        
        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Wolock et al. (2019) Cell Systems",
                "McGinnis et al. (2019) Cell Systems"
            ]
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Doublet filtering failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def get_doublet_filtered_data(state: SessionState) -> Optional[object]:
    """Get the most recent doublet-filtered AnnData object.
    
    Args:
        state: Current session state
        
    Returns:
        AnnData object with doublet filtering applied, or None if not available
    """
    if not SCANPY_AVAILABLE:
        return None
    
    try:
        # Look for most recent doublet filtering checkpoint
        for entry in reversed(state.history):
            if entry.get("label") == "doublets_filtered":
                checkpoint_path = entry.get("checkpoint_path")
                if checkpoint_path and Path(checkpoint_path).exists():
                    return sc.read_h5ad(checkpoint_path)
        
        # Fallback to doublet detection checkpoint
        for entry in reversed(state.history):
            if entry.get("label") == "doublets_detected":
                checkpoint_path = entry.get("checkpoint_path")
                if checkpoint_path and Path(checkpoint_path).exists():
                    adata = sc.read_h5ad(checkpoint_path)
                    # Check if doublets are marked but not yet filtered
                    if 'doublet' in adata.obs.columns:
                        return adata
        
        # Final fallback to current data
        return _load_adata_from_state(state)
        
    except Exception:
        return None
