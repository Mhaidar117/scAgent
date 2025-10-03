"""Tissue-aware QC priors for scRNA-seq data analysis (Phase 8)."""

from typing import Dict, Any, Optional, Literal
import warnings

# Import guard for optional dependencies
try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False


# Tissue-specific QC threshold databases
TISSUE_QC_PRIORS = {
    "pbmc": {
        "description": "Peripheral Blood Mononuclear Cells",
        "min_genes": {"default": 200, "strict": 500, "lenient": 100},
        "max_genes": {"default": 7000, "strict": 5000, "lenient": 10000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 20.0, "strict": 15.0, "lenient": 25.0},
        "max_pct_ribo": {"default": 30.0, "strict": 25.0, "lenient": 40.0},
        "doublet_rate": {"default": 0.08, "strict": 0.06, "lenient": 0.12},
        "sources": ["10x_pbmc", "literature"],
        "notes": "Standard immune cell thresholds based on 10x datasets"
    },
    "brain": {
        "description": "Brain tissue (cortex, hippocampus, etc.)",
        "min_genes": {"default": 500, "strict": 800, "lenient": 300},
        "max_genes": {"default": 8000, "strict": 6000, "lenient": 12000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 15.0, "strict": 10.0, "lenient": 20.0},
        "max_pct_ribo": {"default": 25.0, "strict": 20.0, "lenient": 35.0},
        "doublet_rate": {"default": 0.06, "strict": 0.04, "lenient": 0.10},
        "sources": ["allen_brain", "literature"],
        "notes": "Conservative thresholds for neural tissues with lower MT tolerance"
    },
    "liver": {
        "description": "Hepatic tissue",
        "min_genes": {"default": 300, "strict": 600, "lenient": 200},
        "max_genes": {"default": 9000, "strict": 7000, "lenient": 12000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 25.0, "strict": 20.0, "lenient": 30.0},
        "max_pct_ribo": {"default": 35.0, "strict": 30.0, "lenient": 45.0},
        "doublet_rate": {"default": 0.10, "strict": 0.08, "lenient": 0.15},
        "sources": ["literature"],
        "notes": "Higher metabolic activity allows for increased MT gene expression"
    },
    "heart": {
        "description": "Cardiac tissue",
        "min_genes": {"default": 400, "strict": 700, "lenient": 250},
        "max_genes": {"default": 8500, "strict": 6500, "lenient": 11000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 30.0, "strict": 25.0, "lenient": 40.0},
        "max_pct_ribo": {"default": 40.0, "strict": 35.0, "lenient": 50.0},
        "doublet_rate": {"default": 0.08, "strict": 0.06, "lenient": 0.12},
        "sources": ["literature"],
        "notes": "High energy demands result in elevated mitochondrial gene expression"
    },
    "kidney": {
        "description": "Renal tissue",
        "min_genes": {"default": 350, "strict": 600, "lenient": 200},
        "max_genes": {"default": 8000, "strict": 6000, "lenient": 10000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 22.0, "strict": 18.0, "lenient": 28.0},
        "max_pct_ribo": {"default": 30.0, "strict": 25.0, "lenient": 40.0},
        "doublet_rate": {"default": 0.07, "strict": 0.05, "lenient": 0.10},
        "sources": ["literature"],
        "notes": "Moderate metabolic activity with standard thresholds"
    },
    "lung": {
        "description": "Pulmonary tissue",
        "min_genes": {"default": 300, "strict": 500, "lenient": 200},
        "max_genes": {"default": 7500, "strict": 5500, "lenient": 10000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 18.0, "strict": 15.0, "lenient": 25.0},
        "max_pct_ribo": {"default": 28.0, "strict": 23.0, "lenient": 35.0},
        "doublet_rate": {"default": 0.07, "strict": 0.05, "lenient": 0.10},
        "sources": ["literature"],
        "notes": "Respiratory tissue with moderate QC stringency"
    },
    "intestine": {
        "description": "Intestinal tissue",
        "min_genes": {"default": 250, "strict": 500, "lenient": 150},
        "max_genes": {"default": 7000, "strict": 5000, "lenient": 9000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 20.0, "strict": 16.0, "lenient": 26.0},
        "max_pct_ribo": {"default": 32.0, "strict": 27.0, "lenient": 40.0},
        "doublet_rate": {"default": 0.09, "strict": 0.07, "lenient": 0.12},
        "sources": ["literature"],
        "notes": "High turnover tissue with relatively lenient thresholds"
    },
    "skin": {
        "description": "Dermal and epidermal tissue",
        "min_genes": {"default": 300, "strict": 500, "lenient": 200},
        "max_genes": {"default": 7000, "strict": 5500, "lenient": 9000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 18.0, "strict": 14.0, "lenient": 24.0},
        "max_pct_ribo": {"default": 30.0, "strict": 25.0, "lenient": 38.0},
        "doublet_rate": {"default": 0.08, "strict": 0.06, "lenient": 0.11},
        "sources": ["literature"],
        "notes": "Epithelial tissues with standard QC requirements"
    },
    "embryonic": {
        "description": "Embryonic or developmental tissue",
        "min_genes": {"default": 200, "strict": 400, "lenient": 100},
        "max_genes": {"default": 6000, "strict": 4500, "lenient": 8000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 15.0, "strict": 10.0, "lenient": 20.0},
        "max_pct_ribo": {"default": 40.0, "strict": 35.0, "lenient": 50.0},
        "doublet_rate": {"default": 0.05, "strict": 0.03, "lenient": 0.08},
        "sources": ["literature"],
        "notes": "Lenient thresholds for developing tissues with high ribosomal activity"
    },
    "tumor": {
        "description": "Tumor or cancer tissue",
        "min_genes": {"default": 200, "strict": 400, "lenient": 100},
        "max_genes": {"default": 9000, "strict": 7000, "lenient": 12000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 35.0, "strict": 30.0, "lenient": 45.0},
        "max_pct_ribo": {"default": 40.0, "strict": 35.0, "lenient": 55.0},
        "doublet_rate": {"default": 0.12, "strict": 0.08, "lenient": 0.18},
        "sources": ["literature"],
        "notes": "Lenient thresholds due to high metabolic activity and aneuploidy"
    },
    "default": {
        "description": "Generic tissue (conservative defaults)",
        "min_genes": {"default": 200, "strict": 500, "lenient": 100},
        "max_genes": {"default": 7000, "strict": 5000, "lenient": 10000},
        "min_cells": {"default": 3, "strict": 5, "lenient": 1},
        "max_pct_mt": {"default": 20.0, "strict": 15.0, "lenient": 25.0},
        "max_pct_ribo": {"default": 30.0, "strict": 25.0, "lenient": 40.0},
        "doublet_rate": {"default": 0.08, "strict": 0.06, "lenient": 0.12},
        "sources": ["consensus"],
        "notes": "Conservative defaults suitable for most tissue types"
    }
}


def suggest_thresholds(
    adata: Optional[object] = None,
    tissue: str = "default", 
    stringency: Literal["lenient", "default", "strict"] = "default",
    species: Literal["human", "mouse", "other"] = "human",
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Suggest tissue-aware QC thresholds based on prior knowledge.
    
    Args:
        adata: AnnData object (optional, for data-driven adjustments)
        tissue: Tissue type for threshold selection
        stringency: QC stringency level
        species: Species for species-specific adjustments
        custom_overrides: Custom threshold overrides
        
    Returns:
        Dictionary of suggested QC thresholds
        
    Example:
        >>> thresholds = suggest_thresholds(tissue="brain", stringency="strict")
        >>> print(thresholds["max_pct_mt"])  # 10.0
    """
    # Normalize tissue name
    tissue_key = tissue.lower().strip()
    if tissue_key not in TISSUE_QC_PRIORS:
        available_tissues = list(TISSUE_QC_PRIORS.keys())
        warnings.warn(
            f"Unknown tissue '{tissue}'. Available tissues: {available_tissues}. "
            f"Using 'default' tissue priors.",
            UserWarning
        )
        tissue_key = "default"
    
    # Get base thresholds for tissue and stringency
    tissue_priors = TISSUE_QC_PRIORS[tissue_key]
    thresholds = {}
    
    for metric, values in tissue_priors.items():
        if isinstance(values, dict) and stringency in values:
            thresholds[metric] = values[stringency]
        elif not isinstance(values, dict):
            thresholds[metric] = values
    
    # Species-specific adjustments
    if species == "mouse":
        # Mouse cells are generally smaller, adjust accordingly
        thresholds["min_genes"] = int(thresholds["min_genes"] * 0.8)
        thresholds["max_genes"] = int(thresholds["max_genes"] * 0.9)
        # Mouse mitochondrial genes have different naming (mt- vs MT-)
        if "notes" not in thresholds:
            thresholds["notes"] = ""
        thresholds["notes"] += " Mouse-specific adjustments applied."
    
    # Data-driven adjustments (if AnnData is provided)
    if adata is not None and ANNDATA_AVAILABLE:
        try:
            data_adjustments = _compute_data_driven_adjustments(adata, thresholds)
            thresholds.update(data_adjustments)
        except Exception as e:
            warnings.warn(f"Could not compute data-driven adjustments: {e}", UserWarning)
    
    # Apply custom overrides
    if custom_overrides:
        thresholds.update(custom_overrides)
    
    # Add metadata
    thresholds["tissue"] = tissue_key
    thresholds["stringency"] = stringency
    thresholds["species"] = species
    
    return thresholds


def _compute_data_driven_adjustments(adata: object, base_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """Compute data-driven adjustments to base thresholds."""
    adjustments = {}
    
    try:
        # Compute gene count statistics
        if hasattr(adata, 'X'):
            n_genes_per_cell = (adata.X > 0).sum(axis=1)
            if hasattr(n_genes_per_cell, 'A1'):  # Handle sparse matrices
                n_genes_per_cell = n_genes_per_cell.A1
            
            # Adjust min_genes based on data distribution (e.g., 5th percentile)
            genes_5th_percentile = int(np.percentile(n_genes_per_cell, 5))
            if genes_5th_percentile > base_thresholds.get("min_genes", 200):
                adjustments["min_genes_suggested"] = genes_5th_percentile
                adjustments["min_genes_note"] = f"Data-driven: {genes_5th_percentile} (5th percentile)"
            
            # Adjust max_genes based on data distribution (e.g., 95th percentile)
            genes_95th_percentile = int(np.percentile(n_genes_per_cell, 95))
            if genes_95th_percentile < base_thresholds.get("max_genes", 7000):
                adjustments["max_genes_suggested"] = genes_95th_percentile
                adjustments["max_genes_note"] = f"Data-driven: {genes_95th_percentile} (95th percentile)"
        
        # Adjust MT percentage if MT genes are present
        if hasattr(adata, 'var'):
            mt_genes = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
            if mt_genes.sum() > 0:
                # Compute MT percentage distribution
                mt_counts = adata[:, mt_genes].X.sum(axis=1)
                total_counts = adata.X.sum(axis=1)
                if hasattr(mt_counts, 'A1'):  # Handle sparse matrices
                    mt_counts = mt_counts.A1
                    total_counts = total_counts.A1
                
                pct_mt = (mt_counts / total_counts) * 100
                mt_95th_percentile = float(np.percentile(pct_mt, 95))
                
                if mt_95th_percentile < base_thresholds.get("max_pct_mt", 20.0):
                    adjustments["max_pct_mt_suggested"] = mt_95th_percentile
                    adjustments["max_pct_mt_note"] = f"Data-driven: {mt_95th_percentile:.1f}% (95th percentile)"
    
    except Exception as e:
        # Fail silently for data-driven adjustments
        adjustments["adjustment_error"] = str(e)
    
    return adjustments


def get_available_tissues() -> Dict[str, str]:
    """Get dictionary of available tissue types and their descriptions.
    
    Returns:
        Dictionary mapping tissue names to descriptions
    """
    return {
        tissue: priors["description"] 
        for tissue, priors in TISSUE_QC_PRIORS.items()
    }


def get_tissue_info(tissue: str) -> Dict[str, Any]:
    """Get detailed information about a specific tissue's QC priors.
    
    Args:
        tissue: Tissue type name
        
    Returns:
        Dictionary with tissue information and thresholds
        
    Raises:
        ValueError: If tissue is not recognized
    """
    tissue_key = tissue.lower().strip()
    if tissue_key not in TISSUE_QC_PRIORS:
        available_tissues = list(TISSUE_QC_PRIORS.keys())
        raise ValueError(f"Unknown tissue '{tissue}'. Available: {available_tissues}")
    
    return TISSUE_QC_PRIORS[tissue_key].copy()


def compare_tissue_thresholds(tissues: list, stringency: str = "default") -> "pd.DataFrame":
    """Compare QC thresholds across multiple tissues.
    
    Args:
        tissues: List of tissue names to compare
        stringency: Stringency level for comparison
        
    Returns:
        DataFrame with tissue comparisons
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for threshold comparison")
    
    comparison_data = []
    
    for tissue in tissues:
        try:
            thresholds = suggest_thresholds(tissue=tissue, stringency=stringency)
            row = {
                "tissue": tissue,
                "description": thresholds.get("description", ""),
                "min_genes": thresholds.get("min_genes", "N/A"),
                "max_genes": thresholds.get("max_genes", "N/A"),
                "max_pct_mt": thresholds.get("max_pct_mt", "N/A"),
                "max_pct_ribo": thresholds.get("max_pct_ribo", "N/A"),
                "doublet_rate": thresholds.get("doublet_rate", "N/A"),
            }
            comparison_data.append(row)
        except Exception as e:
            print(f"Warning: Could not get thresholds for {tissue}: {e}")
    
    return pd.DataFrame(comparison_data)
