"""Doublet detection tools for scRNA-seq data analysis."""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal, List, Tuple
import scipy.sparse as sp

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


# ============================================================================
# DoubletFinder Core Implementation
# ============================================================================

def _doubletfinder_core(
    adata: object,
    pK: float,
    expected_rate: float,
    pN: float = 0.25,
    n_prin_comps: int = 30,
    random_seed: int = 0
) -> object:
    """
    Pure Python DoubletFinder implementation.

    Creates artificial doublets, performs PCA on combined real+artificial cells,
    calculates pANN (proportion of artificial nearest neighbors) for each cell,
    and classifies doublets based on expected rate threshold.

    Args:
        adata: AnnData object with raw counts
        pK: Neighborhood parameter (proportion of cells to use for kNN)
        expected_rate: Expected doublet rate (e.g., 0.06 for 6%)
        pN: Proportion of artificial doublets to generate (default 0.25)
        n_prin_comps: Number of PCA components (default 30)
        random_seed: Random seed for reproducibility

    Returns:
        AnnData with obs['pANN'] and obs['DF.class'] added
    """
    np.random.seed(random_seed)

    # Get counts matrix
    if 'counts' in adata.layers:
        counts = adata.layers['counts'].copy()
    else:
        counts = adata.X.copy()

    # Convert to dense if sparse
    if sp.issparse(counts):
        counts = counts.toarray()

    n_cells = counts.shape[0]

    # Step 1: Generate artificial doublets
    n_doublets = int(n_cells * pN * 2)  # Typically 2x pN proportion
    doublet_ids = np.random.choice(n_cells, size=(n_doublets, 2), replace=True)
    artificial_doublets = counts[doublet_ids[:, 0]] + counts[doublet_ids[:, 1]]

    # Step 2: Combine real and artificial cells
    combined = np.vstack([counts, artificial_doublets])

    # Step 3: Normalize, log, scale, and PCA
    combined_adata = ad.AnnData(combined)
    sc.pp.normalize_total(combined_adata, target_sum=1e4)
    sc.pp.log1p(combined_adata)
    sc.pp.scale(combined_adata)
    sc.tl.pca(combined_adata, n_comps=n_prin_comps, random_state=random_seed)

    pca_coords = combined_adata.obsm['X_pca']

    # Step 4: Calculate pANN for each real cell
    from sklearn.neighbors import NearestNeighbors

    real_coords = pca_coords[:n_cells]
    all_coords = pca_coords

    k = max(2, int(pK * n_cells))  # Ensure k >= 2

    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(all_coords)
    distances, indices = nn.kneighbors(real_coords)

    # Calculate pANN (proportion of artificial nearest neighbors)
    pANN = np.zeros(n_cells)
    for i in range(n_cells):
        neighbors = indices[i]
        pANN[i] = np.sum(neighbors >= n_cells) / k

    # Step 5: Classify doublets based on expected rate
    n_doublets_expected = int(expected_rate * n_cells)
    if n_doublets_expected > 0:
        threshold_idx = np.argsort(pANN)[-n_doublets_expected]
        threshold = pANN[threshold_idx]
    else:
        threshold = 1.0  # No doublets expected

    doublet_class = np.where(pANN >= threshold, 'Doublet', 'Singlet')

    # Add results to original adata
    adata.obs['pANN'] = pANN
    adata.obs['DF.class'] = doublet_class
    adata.obs['doublet_score'] = pANN  # Alias for consistency
    adata.obs['doublet'] = (doublet_class == 'Doublet')

    return adata


def _run_pk_sweep(
    adata: object,
    pK_grid: Tuple[float, ...],
    expected_rate: float,
    pN: float,
    n_prin_comps: int,
    random_seed: int
) -> List[Tuple[float, float]]:
    """
    Run DoubletFinder across multiple pK values.

    Args:
        adata: AnnData object
        pK_grid: Tuple of pK values to test
        expected_rate: Expected doublet rate
        pN: Proportion of artificial doublets
        n_prin_comps: Number of PCA components
        random_seed: Random seed

    Returns:
        List of (pK, doublet_fraction) tuples
    """
    sweep_results = []

    for pK in pK_grid:
        tmp = adata.copy()
        tmp = _doubletfinder_core(tmp, pK, expected_rate, pN, n_prin_comps, random_seed)
        frac = (tmp.obs['DF.class'] == 'Doublet').mean()
        sweep_results.append((float(pK), float(frac)))

    return sweep_results


def _choose_optimal_pk(
    sweep_results: List[Tuple[float, float]],
    expected_rate: float,
    tol: float = 0.02
) -> float:
    """
    Select optimal pK from sweep results.

    Chooses pK whose doublet_fraction is closest to expected_rate within tolerance.
    Falls back to maximum fraction if none within tolerance.

    Args:
        sweep_results: List of (pK, doublet_fraction) tuples
        expected_rate: Expected doublet rate
        tol: Tolerance for matching expected rate

    Returns:
        Optimal pK value
    """
    if not sweep_results:
        raise ValueError("Empty pK sweep results.")

    # Find pK closest to expected rate
    pk, frac = min(sweep_results, key=lambda x: abs(x[1] - expected_rate))

    if abs(frac - expected_rate) <= tol:
        return pk

    # Fallback: maximum doublet fraction (elbow approach)
    fallback_pk = max(sweep_results, key=lambda x: x[1])[0]
    return fallback_pk


def detect_doublets(
    state: SessionState,
    method: Literal["scrublet", "doubletfinder"] = "scrublet",
    expected_rate: float = 0.06,
    threshold: Union[float, Literal["auto"]] = "auto",
    pK: Union[float, Literal["auto"]] = "auto",
    pN: float = 0.25,
    n_prin_comps: int = 30,
    run_pk_sweep: bool = True,
    random_seed: int = 0
) -> ToolResult:
    """Detect doublets in scRNA-seq data.

    Args:
        state: Current session state
        method: Doublet detection method ('scrublet' or 'doubletfinder')
        expected_rate: Expected doublet rate (typically 0.06 for 10X data)
        threshold: Doublet score threshold ('auto' for automatic detection or float)
        pK: DoubletFinder neighborhood parameter ('auto' to optimize via sweep, or float)
        pN: DoubletFinder proportion of artificial doublets (default 0.25)
        n_prin_comps: Number of PCA components for DoubletFinder (default 30)
        run_pk_sweep: Whether to run pK optimization sweep (default True)
        random_seed: Random seed for reproducibility (default 0)

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
        # DoubletFinder implementation
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            return ToolResult(
                message=(
                    "❌ scikit-learn not available. Install with: pip install scikit-learn\n"
                    "Required for DoubletFinder nearest neighbor calculation."
                ),
                state_delta={},
                artifacts=[],
                citations=[]
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

        # Create unique step-specific directory to preserve all doublet detection results
        step_num = len(state.history)
        step_dir_path = f"runs/{state.run_id}/step_{step_num:02d}_doublets_detect"
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

        elif method == "doubletfinder":
            # DoubletFinder detection
            sweep_results = None
            optimal_pK = pK

            # Run pK sweep if requested and pK is auto
            if pK == "auto" and run_pk_sweep:
                pK_grid = (0.005, 0.01, 0.02, 0.03, 0.05)
                sweep_results = _run_pk_sweep(
                    adata, pK_grid, expected_rate, pN, n_prin_comps, random_seed
                )
                optimal_pK = _choose_optimal_pk(sweep_results, expected_rate, tol=0.02)

                # Save sweep results
                sweep_df = pd.DataFrame(sweep_results, columns=['pK', 'doublet_fraction'])
                sweep_csv = step_dir / "pk_sweep_results.csv"
                sweep_df.to_csv(sweep_csv, index=False)
                artifacts.append(sweep_csv)

                # Plot sweep results
                if PLOTTING_AVAILABLE:
                    plt.figure(figsize=(8, 6))
                    plt.plot(sweep_df['pK'], sweep_df['doublet_fraction'], 'o-', linewidth=2, markersize=8)
                    plt.axhline(expected_rate, color='red', linestyle='--', label=f'Expected rate: {expected_rate:.2%}')
                    plt.axvline(optimal_pK, color='green', linestyle='--', label=f'Optimal pK: {optimal_pK}')
                    plt.xlabel('pK')
                    plt.ylabel('Doublet Fraction')
                    plt.title('DoubletFinder pK Sweep')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    sweep_plot = step_dir / "pk_sweep_plot.png"
                    plt.savefig(sweep_plot, dpi=300, bbox_inches='tight')
                    plt.close()
                    artifacts.append(sweep_plot)
            elif pK == "auto":
                # No sweep, use default
                optimal_pK = 0.01

            # Run DoubletFinder with optimal pK
            adata = _doubletfinder_core(
                adata, float(optimal_pK), expected_rate, pN, n_prin_comps, random_seed
            )

            doublet_scores = adata.obs['pANN'].values
            predicted_doublets = adata.obs['doublet'].values
            detected_threshold = float(optimal_pK)

            # Generate visualization
            if PLOTTING_AVAILABLE:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # pANN histogram
                ax1.hist(doublet_scores, bins=50, alpha=0.7, density=True, label='All cells')
                ax1.hist(doublet_scores[predicted_doublets], bins=50, alpha=0.7,
                        density=True, label='Predicted doublets', color='red')
                ax1.set_xlabel('pANN Score')
                ax1.set_ylabel('Density')
                ax1.set_title(f'DoubletFinder pANN Distribution\n'
                            f'pK={optimal_pK:.3f}, Expected: {expected_rate:.1%}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # UMAP colored by doublet class (if UMAP exists)
                if 'X_umap' in adata.obsm:
                    umap_coords = adata.obsm['X_umap']
                    ax2.scatter(umap_coords[~predicted_doublets, 0],
                               umap_coords[~predicted_doublets, 1],
                               c='blue', s=1, alpha=0.5, label='Singlet')
                    ax2.scatter(umap_coords[predicted_doublets, 0],
                               umap_coords[predicted_doublets, 1],
                               c='red', s=1, alpha=0.5, label='Doublet')
                    ax2.set_xlabel('UMAP 1')
                    ax2.set_ylabel('UMAP 2')
                    ax2.set_title('Doublet Predictions (UMAP)')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'UMAP not computed', ha='center', va='center')
                    ax2.set_title('UMAP unavailable')

                plt.tight_layout()
                doublet_viz = step_dir / "doublet_detection_viz.png"
                plt.savefig(doublet_viz, dpi=300, bbox_inches='tight')
                plt.close()
                artifacts.append(doublet_viz)

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

        # Create unique step-specific directory to preserve all doublet filtering results
        step_num = len(state.history)
        step_dir_path = f"runs/{state.run_id}/step_{step_num:02d}_doublets_apply"
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
            "cells_before_doublet_filter": int(original_n_cells),  # Convert NumPy int64 to Python int
            "cells_after_doublet_filter": int(n_kept),  # Convert NumPy int64 to Python int
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


# ============================================================================
# Additional DoubletFinder Tools
# ============================================================================

def run_pk_sweep_only(
    state: SessionState,
    pK_grid: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05),
    expected_rate: float = 0.06,
    pN: float = 0.25,
    n_prin_comps: int = 30,
    random_seed: int = 0
) -> ToolResult:
    """
    Run DoubletFinder pK sweep optimization without applying detection.

    Returns sweep results showing doublet fraction for each pK value,
    helping users select optimal pK parameter.

    Args:
        state: Current session state
        pK_grid: Tuple of pK values to test
        expected_rate: Expected doublet rate
        pN: Proportion of artificial doublets
        n_prin_comps: Number of PCA components
        random_seed: Random seed for reproducibility

    Returns:
        ToolResult with sweep results as artifacts
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    # Check sklearn availability (used in _run_pk_sweep)
    try:
        from sklearn.neighbors import NearestNeighbors as _  # noqa: F401
    except ImportError:
        return ToolResult(
            message="scikit-learn not available. Install with: pip install scikit-learn",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    try:
        adata = _load_adata_from_state(state)

        # Create output directory
        step_dir_path = f"runs/{state.run_id}/step_11_pk_sweep"
        step_dir = ensure_run_dir(step_dir_path)

        artifacts = []

        # Run pK sweep
        sweep_results = _run_pk_sweep(
            adata, pK_grid, expected_rate, pN, n_prin_comps, random_seed
        )

        # Choose optimal pK
        optimal_pK = _choose_optimal_pk(sweep_results, expected_rate, tol=0.02)

        # Save sweep results
        sweep_df = pd.DataFrame(sweep_results, columns=['pK', 'doublet_fraction'])
        sweep_csv = step_dir / "pk_sweep_results.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        artifacts.append(sweep_csv)

        # Plot sweep results
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(10, 6))
            plt.plot(sweep_df['pK'], sweep_df['doublet_fraction'], 'o-',
                    linewidth=2, markersize=10, label='Observed')
            plt.axhline(expected_rate, color='red', linestyle='--',
                       linewidth=2, label=f'Expected rate: {expected_rate:.2%}')
            plt.axvline(optimal_pK, color='green', linestyle='--',
                       linewidth=2, label=f'Optimal pK: {optimal_pK}')
            plt.xlabel('pK', fontsize=12)
            plt.ylabel('Doublet Fraction', fontsize=12)
            plt.title('DoubletFinder pK Parameter Sweep', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            sweep_plot = step_dir / "pk_sweep_plot.png"
            plt.savefig(sweep_plot, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(sweep_plot)

        # Create checkpoint (no data modification, just tracking)
        state.checkpoint(state.adata_path, "pk_sweep_complete")
        for artifact in artifacts:
            if "pk_sweep_results.csv" in str(artifact):
                state.add_artifact(str(artifact), "pK sweep results")
            elif "pk_sweep_plot.png" in str(artifact):
                state.add_artifact(str(artifact), "pK sweep plot")

        state_delta = {
            "optimal_pK": float(optimal_pK),
            "pk_sweep_n_tests": len(pK_grid),
            "pk_sweep_expected_rate": expected_rate
        }

        message = (
            f"✅ pK sweep optimization complete!\n"
            f"🔍 Tested {len(pK_grid)} pK values: {pK_grid}\n"
            f"✨ Optimal pK: {optimal_pK}\n"
            f"📊 Expected doublet rate: {expected_rate:.1%}\n"
            f"📁 Artifacts: {len(artifacts)} files saved\n"
            f"💡 Use optimal_pK={optimal_pK} in detect_doublets(method='doubletfinder', pK={optimal_pK})"
        )

        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=["McGinnis et al. (2019) Cell Systems - DoubletFinder"]
        )

    except Exception as e:
        return ToolResult(
            message=f"❌ pK sweep failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )


def curate_doublets_by_markers(
    state: SessionState,
    marker_dict: Dict[str, List[str]],
    cluster_key: str = "leiden",
    avg_exp_threshold: float = 2.0
) -> ToolResult:
    """
    Manual doublet curation by identifying clusters with incompatible marker expression.

    Identifies clusters expressing markers from multiple incompatible cell types
    (e.g., epithelial + endothelial markers) and marks them as potential doublets.

    Args:
        state: Current session state
        marker_dict: Dictionary mapping cell type names to marker gene lists
        cluster_key: Column in adata.obs containing cluster assignments
        avg_exp_threshold: Expression threshold for considering a marker expressed

    Returns:
        ToolResult with list of doublet clusters and marker expression matrix
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

        # Validate cluster key exists
        if cluster_key not in adata.obs.columns:
            return ToolResult(
                message=f"Cluster key '{cluster_key}' not found in data. Run clustering first.",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Create output directory
        step_dir_path = f"runs/{state.run_id}/step_11_doublet_curation"
        step_dir = ensure_run_dir(step_dir_path)

        artifacts = []

        # Collect all markers
        all_markers = []
        for cell_type, markers in marker_dict.items():
            all_markers.extend(markers)
        all_markers = list(set(all_markers))

        # Filter to markers present in data
        present_markers = [m for m in all_markers if m in adata.var_names]

        if not present_markers:
            return ToolResult(
                message=f"No markers from marker_dict found in data. Check gene names.",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Calculate mean expression per cluster
        cluster_expression = pd.DataFrame()
        for cluster in adata.obs[cluster_key].unique():
            mask = adata.obs[cluster_key] == cluster
            cluster_data = adata[mask, present_markers]

            # Use raw or normalized data
            if 'counts' in cluster_data.layers:
                expr_data = cluster_data.layers['counts']
            else:
                expr_data = cluster_data.X

            if sp.issparse(expr_data):
                mean_expr = np.array(expr_data.mean(axis=0)).flatten()
            else:
                mean_expr = expr_data.mean(axis=0)

            cluster_expression[cluster] = mean_expr

        cluster_expression.index = present_markers

        # Identify doublet clusters
        doublet_clusters = []

        # Define incompatible marker pairs (example for kidney)
        incompatible_pairs = [
            ('Proximal_Tubule', 'Endothelial'),
            ('Proximal_Tubule', 'Fibroblast'),
            ('Proximal_Tubule', 'TAL'),
            ('Endothelial', 'Immune'),
            ('Epithelial', 'Immune'),
        ]

        for cluster in cluster_expression.columns:
            expr = cluster_expression[cluster]

            # Check which cell types are expressed
            cell_types_expressed = []
            for cell_type, markers in marker_dict.items():
                markers_present = [m for m in markers if m in present_markers]
                if markers_present and any(expr[m] > avg_exp_threshold for m in markers_present):
                    cell_types_expressed.append(cell_type)

            # Check for incompatible pairs
            for type1, type2 in incompatible_pairs:
                if type1 in cell_types_expressed and type2 in cell_types_expressed:
                    doublet_clusters.append(str(cluster))
                    break

        doublet_clusters = list(set(doublet_clusters))

        # Mark doublet clusters in adata
        adata.obs['manual_doublet_cluster'] = adata.obs[cluster_key].isin(doublet_clusters)

        # Save expression matrix
        expr_csv = step_dir / "marker_coexpression.csv"
        cluster_expression.to_csv(expr_csv)
        artifacts.append(expr_csv)

        # Save doublet cluster IDs
        doublet_json = step_dir / "doublet_clusters.json"
        with open(doublet_json, 'w') as f:
            json.dump({
                'doublet_clusters': doublet_clusters,
                'n_doublet_clusters': len(doublet_clusters),
                'avg_exp_threshold': avg_exp_threshold,
                'marker_dict_keys': list(marker_dict.keys())
            }, f, indent=2)
        artifacts.append(doublet_json)

        # Save updated adata
        checkpoint_path = step_dir / "adata_manual_curation.h5ad"
        adata.write_h5ad(checkpoint_path)

        # Create checkpoint
        state.checkpoint(str(checkpoint_path), "doublet_manual_curation")
        for artifact in artifacts:
            if "marker_coexpression" in str(artifact):
                state.add_artifact(str(artifact), "Marker co-expression matrix")
            elif "doublet_clusters.json" in str(artifact):
                state.add_artifact(str(artifact), "Doublet cluster IDs")

        n_cells_in_doublet_clusters = adata.obs['manual_doublet_cluster'].sum()

        state_delta = {
            "adata_path": str(checkpoint_path),
            "n_doublet_clusters": len(doublet_clusters),
            "doublet_clusters": doublet_clusters,
            "n_cells_manual_doublet": int(n_cells_in_doublet_clusters)
        }

        if doublet_clusters:
            message = (
                f"✅ Manual doublet curation complete!\n"
                f"🔍 Identified {len(doublet_clusters)} potential doublet clusters: {doublet_clusters}\n"
                f"👥 Cells affected: {n_cells_in_doublet_clusters}\n"
                f"🧬 Markers checked: {len(present_markers)}/{len(all_markers)}\n"
                f"📊 Expression threshold: {avg_exp_threshold}\n"
                f"📁 Artifacts: {len(artifacts)} files saved\n"
                f"💡 Use apply_doublet_filter() to remove these clusters"
            )
        else:
            message = (
                f"✅ Manual doublet curation complete - No doublet clusters found!\n"
                f"🧬 Markers checked: {len(present_markers)}/{len(all_markers)}\n"
                f"📊 Expression threshold: {avg_exp_threshold}\n"
                f"✨ All clusters appear to be homogeneous\n"
                f"📁 Artifacts: {len(artifacts)} files saved"
            )

        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=["McGinnis et al. (2019) Cell Systems - DoubletFinder"]
        )

    except Exception as e:
        return ToolResult(
            message=f"❌ Manual doublet curation failed: {str(e)}",
            state_delta={},
            artifacts=[],
            citations=[]
        )
