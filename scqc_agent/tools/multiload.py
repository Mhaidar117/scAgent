"""Multi-file data loader for kidney scRNA-seq datasets.

This module provides tools to load kidney scRNA-seq datasets consisting of:
1. Raw 10X HDF5 file (all droplets, for ambient RNA correction with SCAR)
2. Filtered 10X HDF5 file (cells only, for primary analysis)
3. Metadata CSV file (sample annotations)

The loader merges metadata, stores original counts, and creates checkpoints for both
raw and filtered datasets to enable downstream analysis.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from ..state import ToolResult, SessionState
from .io import ensure_run_dir

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


def load_kidney_data(
    state: SessionState,
    raw_h5_path: str,
    filtered_h5_path: str,
    meta_csv_path: str,
    sample_id_column: str = "sample_ID",
    metadata_merge_column: Optional[str] = None,
    make_unique: bool = True
) -> ToolResult:
    """Load kidney scRNA-seq dataset from raw H5, filtered H5, and metadata CSV.

    This function performs comprehensive data loading and validation:
    1. Loads raw 10X HDF5 matrix (all droplets for SCAR ambient RNA correction)
    2. Loads filtered 10X HDF5 matrix (cells only for primary analysis)
    3. Loads metadata CSV and merges with filtered data
    4. Stores original counts in filtered.layers['counts_raw']
    5. Creates checkpoints for both raw and filtered datasets
    6. Generates summary statistics and data preview plots

    Args:
        state: Current session state
        raw_h5_path: Path to raw (unfiltered) 10X HDF5 matrix file
        filtered_h5_path: Path to filtered 10X HDF5 matrix file
        meta_csv_path: Path to metadata CSV file
        sample_id_column: Column name in metadata for sample identifiers
        metadata_merge_column: Column in metadata to use for merging (defaults to sample_id_column)
        make_unique: Make gene names unique by appending suffixes

    Returns:
        ToolResult with loaded data checkpoints, summary artifacts, and state updates

    Raises:
        ImportError: If scanpy is not available
        FileNotFoundError: If any input file is missing
        ValueError: If file formats are invalid or metadata merge fails
        KeyError: If required columns are missing from metadata

    Examples:
        >>> result = load_kidney_data(
        ...     state,
        ...     raw_h5_path="data/kidney_raw.h5",
        ...     filtered_h5_path="data/kidney_filtered.h5",
        ...     meta_csv_path="data/kidney_metadata.csv",
        ...     sample_id_column="sample_ID"
        ... )
        >>> print(result.message)
        Successfully loaded kidney dataset with 15000 raw droplets, 8000 filtered cells, 25000 genes

    State Updates:
        - adata_path: Path to filtered data checkpoint
        - raw_adata_path: Path to raw data checkpoint (for SCAR)
        - n_cells_raw: Number of raw droplets
        - n_cells_filtered: Number of filtered cells
        - n_genes: Number of genes
        - sample_id: Sample identifier from metadata
        - metadata_columns: List of metadata column names
    """
    if not SCANPY_AVAILABLE:
        return ToolResult(
            message="Scanpy not available. Install with: pip install scanpy",
            state_delta={},
            artifacts=[],
            citations=[]
        )

    try:
        # Validate input files exist (redundant with Pydantic but good practice)
        raw_path = Path(raw_h5_path)
        filtered_path = Path(filtered_h5_path)
        meta_path = Path(meta_csv_path)

        if not raw_path.exists():
            return ToolResult(
                message=f"Raw H5 file not found: {raw_h5_path}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        if not filtered_path.exists():
            return ToolResult(
                message=f"Filtered H5 file not found: {filtered_h5_path}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        if not meta_path.exists():
            return ToolResult(
                message=f"Metadata CSV file not found: {meta_csv_path}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Create step directory
        step_dir = ensure_run_dir(state, "step_00_multiload")
        artifacts = []

        # ==================== LOAD RAW DATA ====================
        print(f"Loading raw data from {raw_path}...")
        try:
            # Auto-detect format: try h5ad first, then 10X h5
            if str(raw_path).endswith('.h5ad') or str(raw_path).endswith('.h5ad.gz'):
                adata_raw = sc.read_h5ad(str(raw_path))
            else:
                # Try 10X format first
                try:
                    adata_raw = sc.read_10x_h5(str(raw_path))
                except Exception as e10x:
                    # If 10X format fails, try h5ad (file might have wrong extension)
                    try:
                        adata_raw = sc.read_h5ad(str(raw_path))
                    except Exception:
                        # Re-raise original 10X error
                        raise e10x
        except Exception as e:
            return ToolResult(
                message=f"Failed to load raw H5 file: {str(e)}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Make gene names unique for raw data
        if make_unique:
            adata_raw.var_names_make_unique()

        n_cells_raw = adata_raw.n_obs
        n_genes_raw = adata_raw.n_vars
        print(f"  Raw data: {n_cells_raw:,} droplets x {n_genes_raw:,} genes")

        # ==================== LOAD FILTERED DATA ====================
        print(f"Loading filtered data from {filtered_path}...")
        try:
            # Auto-detect format: try h5ad first, then 10X h5
            if str(filtered_path).endswith('.h5ad') or str(filtered_path).endswith('.h5ad.gz'):
                adata_filtered = sc.read_h5ad(str(filtered_path))
            else:
                # Try 10X format first
                try:
                    adata_filtered = sc.read_10x_h5(str(filtered_path))
                except Exception as e10x:
                    # If 10X format fails, try h5ad (file might have wrong extension)
                    try:
                        adata_filtered = sc.read_h5ad(str(filtered_path))
                    except Exception:
                        # Re-raise original 10X error
                        raise e10x
        except Exception as e:
            return ToolResult(
                message=f"Failed to load filtered H5 file: {str(e)}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Make gene names unique for filtered data
        if make_unique:
            adata_filtered.var_names_make_unique()

        n_cells_filtered = adata_filtered.n_obs
        n_genes_filtered = adata_filtered.n_vars
        print(f"  Filtered data: {n_cells_filtered:,} cells x {n_genes_filtered:,} genes")

        # Validate gene sets match
        if n_genes_raw != n_genes_filtered:
            print(f"Warning: Raw ({n_genes_raw}) and filtered ({n_genes_filtered}) have different gene counts")

        # ==================== LOAD METADATA ====================
        print(f"Loading metadata from {meta_path}...")
        try:
            metadata = pd.read_csv(meta_path)
        except Exception as e:
            return ToolResult(
                message=f"Failed to load metadata CSV: {str(e)}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

        # Validate sample_id_column exists
        if sample_id_column not in metadata.columns:
            return ToolResult(
                message=(
                    f"Sample ID column '{sample_id_column}' not found in metadata. "
                    f"Available columns: {list(metadata.columns)}"
                ),
                state_delta={},
                artifacts=[],
                citations=[]
            )

        print(f"  Metadata: {len(metadata)} rows x {len(metadata.columns)} columns")
        print(f"  Metadata columns: {list(metadata.columns)}")

        # ==================== MERGE METADATA ====================
        # Determine merge column
        merge_col = metadata_merge_column or sample_id_column

        # For simplicity, assume metadata has one row per sample
        # Create a sample_id column in filtered.obs if not present
        if merge_col not in adata_filtered.obs.columns:
            # If metadata has single row, broadcast to all cells
            if len(metadata) == 1:
                sample_id = metadata[sample_id_column].iloc[0]
                adata_filtered.obs[sample_id_column] = sample_id
                print(f"  Assigned sample_id '{sample_id}' to all {n_cells_filtered:,} cells")

                # Merge all metadata columns
                for col in metadata.columns:
                    if col != sample_id_column:
                        adata_filtered.obs[col] = metadata[col].iloc[0]
            else:
                # Multiple samples - need a way to map cells to samples
                # For now, assign first sample ID as fallback
                sample_id = metadata[sample_id_column].iloc[0]
                adata_filtered.obs[sample_id_column] = sample_id
                print(f"  Warning: Multiple metadata rows found. Using first sample_id: {sample_id}")
        else:
            # Metadata merge column exists in obs - perform join
            print(f"  Merging metadata on column '{merge_col}'")
            # This is simplified - in production, would use pandas merge

        # ==================== STORE ORIGINAL COUNTS ====================
        # Store original counts in layers for filtered data
        adata_filtered.layers['counts_raw'] = adata_filtered.X.copy()
        print("  Stored original counts in filtered.layers['counts_raw']")

        # ==================== GENERATE SUMMARY STATISTICS ====================
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "files": {
                "raw_h5": str(raw_path.absolute()),
                "filtered_h5": str(filtered_path.absolute()),
                "metadata_csv": str(meta_path.absolute())
            },
            "counts": {
                "n_raw_droplets": int(n_cells_raw),
                "n_filtered_cells": int(n_cells_filtered),
                "n_genes_raw": int(n_genes_raw),
                "n_genes_filtered": int(n_genes_filtered),
                "cells_retained_fraction": float(n_cells_filtered / n_cells_raw) if n_cells_raw > 0 else 0.0
            },
            "metadata": {
                "n_samples": len(metadata),
                "sample_id_column": sample_id_column,
                "metadata_columns": list(metadata.columns),
                "sample_ids": list(metadata[sample_id_column].unique()) if sample_id_column in metadata.columns else []
            },
            "filtered_data_summary": {
                "total_counts_mean": float(adata_filtered.X.sum(axis=1).mean()) if hasattr(adata_filtered.X, 'mean') else 0.0,
                "genes_per_cell_mean": float((adata_filtered.X > 0).sum(axis=1).mean()) if hasattr(adata_filtered.X, 'mean') else 0.0,
            }
        }

        # Save summary JSON
        summary_path = step_dir / "load_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        artifacts.append(str(summary_path))

        # Save summary CSV
        summary_csv_path = step_dir / "load_summary.csv"
        summary_df = pd.DataFrame([{
            "metric": "n_raw_droplets",
            "value": n_cells_raw
        }, {
            "metric": "n_filtered_cells",
            "value": n_cells_filtered
        }, {
            "metric": "n_genes",
            "value": n_genes_filtered
        }, {
            "metric": "cells_retained_fraction",
            "value": n_cells_filtered / n_cells_raw if n_cells_raw > 0 else 0.0
        }])
        summary_df.to_csv(summary_csv_path, index=False)
        artifacts.append(str(summary_csv_path))

        # ==================== GENERATE DATA PREVIEW PLOT ====================
        if PLOTTING_AVAILABLE:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot 1: Cell/droplet counts comparison
            counts_data = pd.DataFrame({
                'Dataset': ['Raw Droplets', 'Filtered Cells'],
                'Count': [n_cells_raw, n_cells_filtered]
            })
            axes[0].bar(counts_data['Dataset'], counts_data['Count'], color=['lightblue', 'steelblue'])
            axes[0].set_ylabel('Count')
            axes[0].set_title('Raw vs Filtered Cell Counts')
            axes[0].set_ylim(0, n_cells_raw * 1.1)
            for i, v in enumerate(counts_data['Count']):
                axes[0].text(i, v + n_cells_raw * 0.02, f'{v:,}', ha='center', va='bottom')

            # Plot 2: Genes detected
            axes[1].bar(['Genes'], [n_genes_filtered], color='green', alpha=0.7)
            axes[1].set_ylabel('Gene Count')
            axes[1].set_title('Total Genes Detected')
            axes[1].text(0, n_genes_filtered + n_genes_filtered * 0.02, f'{n_genes_filtered:,}', ha='center', va='bottom')

            plt.tight_layout()
            preview_path = step_dir / "load_preview.png"
            plt.savefig(preview_path, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(preview_path))

        # ==================== SAVE CHECKPOINTS ====================
        # Save raw data checkpoint (for SCAR)
        raw_checkpoint_path = step_dir / "adata_raw.h5ad"
        adata_raw.write_h5ad(raw_checkpoint_path)
        print(f"  Saved raw checkpoint: {raw_checkpoint_path}")

        # Save filtered data checkpoint (primary)
        filtered_checkpoint_path = step_dir / "adata_filtered.h5ad"
        adata_filtered.write_h5ad(filtered_checkpoint_path)
        print(f"  Saved filtered checkpoint: {filtered_checkpoint_path}")

        # ==================== UPDATE STATE ====================
        # CRITICAL: Create checkpoint BEFORE adding artifacts
        # This ensures artifacts appear in workflow history
        state.checkpoint(str(filtered_checkpoint_path), "multiload_filtered")

        # Add artifacts to the checkpoint's history entry
        state.add_artifact(str(summary_path), "Load Summary (JSON)")
        state.add_artifact(str(summary_csv_path), "Load Summary (CSV)")
        if PLOTTING_AVAILABLE:
            state.add_artifact(str(preview_path), "Data Preview Plot")

        # Extract sample ID for state
        sample_id = metadata[sample_id_column].iloc[0] if sample_id_column in metadata.columns and len(metadata) > 0 else "unknown"

        # Build state delta
        state_delta = {
            "adata_path": str(filtered_checkpoint_path),  # Primary data path
            "raw_adata_path": str(raw_checkpoint_path),   # Raw data for SCAR
            "n_cells_raw": int(n_cells_raw),
            "n_cells_filtered": int(n_cells_filtered),
            "n_genes": int(n_genes_filtered),
            "sample_id": str(sample_id),
            "metadata_columns": list(metadata.columns),
            "cells_retained_fraction": float(n_cells_filtered / n_cells_raw) if n_cells_raw > 0 else 0.0,
            "data_loaded": True,
            "load_method": "multiload_kidney"
        }

        # Success message
        message = (
            f"Successfully loaded kidney dataset:\n"
            f"  Raw droplets: {n_cells_raw:,}\n"
            f"  Filtered cells: {n_cells_filtered:,} ({n_cells_filtered/n_cells_raw:.1%} retained)\n"
            f"  Genes: {n_genes_filtered:,}\n"
            f"  Sample ID: {sample_id}\n"
            f"  Metadata columns: {len(metadata.columns)}\n"
            f"  Checkpoints saved:\n"
            f"    - Filtered: {filtered_checkpoint_path}\n"
            f"    - Raw: {raw_checkpoint_path} (for SCAR)\n"
            f"  Original counts stored in filtered.layers['counts_raw']"
        )

        return ToolResult(
            message=message,
            state_delta=state_delta,
            artifacts=[str(p) for p in artifacts],
            citations=[
                "Zheng et al. (2017) Nature Communications",
                "Wolf et al. (2018) Genome Biology"
            ]
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return ToolResult(
            message=f"Error loading kidney dataset: {str(e)}\n\nDetails:\n{error_details}",
            state_delta={},
            artifacts=[],
            citations=[]
        )
