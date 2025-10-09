"""Pydantic schemas for tool input validation and output serialization."""

from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path


class ToolResult(BaseModel):
    """Result from executing a tool in the scQC workflow."""
    
    message: str = Field(..., description="Human-readable description of what happened")
    state_delta: Dict[str, Any] = Field(default_factory=dict, description="Changes to apply to session state")
    artifacts: List[str] = Field(default_factory=list, description="List of file paths generated")
    citations: List[str] = Field(default_factory=list, description="List of relevant citations")


# QC Tool Schemas
class ComputeQCMetricsInput(BaseModel):
    """Input schema for compute_qc_metrics tool."""
    
    species: Optional[Literal["human", "mouse", "other"]] = Field(
        default=None,
        description="Species for mitochondrial gene detection (auto-detected if not specified)"
    )
    mito_prefix: Optional[str] = Field(
        default=None,
        description="Custom mitochondrial gene prefix (overrides species detection)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "species": "mouse",
                "mito_prefix": None
            }
        }


class PlotQCMetricsInput(BaseModel):
    """Input schema for plot_qc_metrics tool."""
    
    stage: Literal["pre", "post"] = Field(
        default="pre",
        description="Whether to plot pre- or post-filtering QC metrics"
    )
    plot_types: Literal["violin", "scatter", "histogram"] = Field(
        default="violin",
        description="Type of plot to generate"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "stage": "pre",
                "plot_types": "violin"
            }
        }


class ApplyQCFiltersInput(BaseModel):
    """Input schema for apply_qc_filters tool."""
    
    min_genes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of genes per cell"
    )
    max_genes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum number of genes per cell"
    )
    min_counts: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum total UMI counts per cell"
    )
    max_counts: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum total UMI counts per cell"
    )
    max_pct_mt: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Maximum mitochondrial percentage"
    )
    min_cells: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum number of cells expressing each gene"
    )
    method: Literal["threshold", "MAD", "quantile"] = Field(
        default="threshold",
        description="Filtering method to use"
    )
    
    @validator('max_genes')
    def max_genes_greater_than_min(cls, v, values):
        if v is not None and 'min_genes' in values and values['min_genes'] is not None:
            if v <= values['min_genes']:
                raise ValueError('max_genes must be greater than min_genes')
        return v
    
    @validator('max_counts')
    def max_counts_greater_than_min(cls, v, values):
        if v is not None and 'min_counts' in values and values['min_counts'] is not None:
            if v <= values['min_counts']:
                raise ValueError('max_counts must be greater than min_counts')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "min_genes": 200,
                "max_pct_mt": 20.0,
                "min_cells": 3,
                "method": "threshold"
            }
        }


# Graph Analysis Tool Schemas
class QuickGraphInput(BaseModel):
    """Input schema for quick_graph tool."""
    
    n_neighbors: int = Field(
        default=15,
        ge=2,
        le=100,
        description="Number of neighbors for kNN graph"
    )
    n_pcs: int = Field(
        default=40,
        ge=2,
        le=100,
        description="Number of principal components to use"
    )
    resolution: float = Field(
        default=0.5,
        ge=0.1,
        le=3.0,
        description="Leiden clustering resolution"
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "n_neighbors": 15,
                "n_pcs": 40,
                "resolution": 0.5,
                "seed": 42
            }
        }


class GraphFromRepInput(BaseModel):
    """Input schema for graph_from_rep tool."""
    
    use_rep: str = Field(
        default="X_pca",
        description="Representation to use (e.g., X_scVI, X_scAR, X_pca)"
    )
    n_neighbors: int = Field(
        default=15,
        ge=2,
        le=100,
        description="Number of neighbors for kNN graph"
    )
    resolution: float = Field(
        default=0.5,
        ge=0.1,
        le=3.0,
        description="Leiden clustering resolution"
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "use_rep": "X_scVI",
                "n_neighbors": 15,
                "resolution": 1.0,
                "seed": 42
            }
        }


class FinalGraphInput(BaseModel):
    """Input schema for final_graph tool."""
    
    use_rep: str = Field(
        default="X_scVI",
        description="Representation to use for final analysis"
    )
    n_neighbors: int = Field(
        default=15,
        ge=2,
        le=100,
        description="Number of neighbors for kNN graph"
    )
    resolution: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Leiden clustering resolution"
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "use_rep": "X_scVI",
                "n_neighbors": 15,
                "resolution": 1.0,
                "seed": 42
            }
        }


# scAR Tool Schemas
class RunScarInput(BaseModel):
    """Input schema for run_scar tool."""
    
    batch_key: str = Field(
        default="SampleID",
        description="Column in adata.obs for batch information"
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of training epochs"
    )
    replace_X: bool = Field(
        default=True,
        description="Whether to replace X with denoised counts"
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "batch_key": "batch",
                "epochs": 100,
                "replace_X": True,
                "random_seed": 42
            }
        }


# scVI Tool Schemas
class RunScviInput(BaseModel):
    """Input schema for run_scvi tool."""
    
    batch_key: str = Field(
        default="SampleID",
        description="Column in adata.obs for batch information"
    )
    n_latent: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of latent dimensions"
    )
    epochs: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Number of training epochs"
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "batch_key": "batch",
                "n_latent": 30,
                "epochs": 200,
                "random_seed": 42
            }
        }


# Doublet Detection Tool Schemas
class DetectDoubletsInput(BaseModel):
    """Input schema for detect_doublets tool."""
    
    method: Literal["scrublet", "doubletfinder"] = Field(
        default="scrublet",
        description="Doublet detection method"
    )
    expected_rate: float = Field(
        default=0.06,
        ge=0.01,
        le=0.5,
        description="Expected doublet rate"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom doublet score threshold (auto if not provided)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "method": "scrublet",
                "expected_rate": 0.06,
                "threshold": None
            }
        }


class ApplyDoubletFilterInput(BaseModel):
    """Input schema for apply_doublet_filter tool."""
    
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom threshold for filtering (uses detected threshold if not provided)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "threshold": None
            }
        }


# Data Loading Tool Schemas
class LoadDataInput(BaseModel):
    """Input schema for load_data tool."""
    
    file_path: str = Field(
        ...,
        description="Path to the AnnData file (.h5ad or .h5ad.gz)"
    )
    backup: bool = Field(
        default=True,
        description="Whether to create a backup of the original file"
    )
    
    @validator('file_path')
    def file_path_must_exist(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f'File does not exist: {v}')
        if not any(path.name.lower().endswith(ext) for ext in ['.h5ad', '.h5', '.h5ad.gz', '.h5.gz']):
            raise ValueError(f'File must be an AnnData file (.h5ad): {v}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "file_path": "Data_files/pbmc3k.h5ad",
                "backup": True
            }
        }


# Marker Gene Detection Tool Schemas
class DetectMarkerGenesInput(BaseModel):
    """Input schema for detect_marker_genes tool."""

    cluster_key: str = Field(
        default="leiden",
        description="Column in adata.obs containing cluster assignments"
    )
    method: Literal["t-test", "wilcoxon", "logreg"] = Field(
        default="wilcoxon",
        description="Statistical test for differential expression"
    )
    n_genes: int = Field(
        default=25,
        ge=5,
        le=200,
        description="Number of top marker genes to report per cluster"
    )
    use_raw: bool = Field(
        default=False,
        description="Use raw counts if available"
    )
    reference: str = Field(
        default="rest",
        description="Reference group for comparison"
    )
    species: Optional[Literal["human", "mouse"]] = Field(
        default=None,
        description="Species for gene filtering (auto-detected if not specified)"
    )

    class Config:
        schema_extra = {
            "example": {
                "cluster_key": "leiden",
                "method": "wilcoxon",
                "n_genes": 25,
                "use_raw": False,
                "reference": "rest",
                "species": "human"
            }
        }


# Cluster Annotation Tool Schemas
class AnnotateClustersInput(BaseModel):
    """Input schema for annotate_clusters tool."""

    cluster_key: str = Field(
        default="leiden",
        description="Column in adata.obs with cluster assignments"
    )
    method: Literal["celltypist", "markers", "auto"] = Field(
        default="auto",
        description="Annotation method: celltypist, markers, or auto"
    )
    species: Literal["human", "mouse"] = Field(
        default="human",
        description="Species for marker selection"
    )
    tissue: Optional[str] = Field(
        default=None,
        description="Tissue type (brain, kidney, pbmc, etc.)"
    )
    celltypist_model: Optional[str] = Field(
        default=None,
        description="Specific CellTypist model to use"
    )
    majority_voting: bool = Field(
        default=True,
        description="Use majority voting for CellTypist predictions"
    )
    custom_markers_path: Optional[str] = Field(
        default=None,
        description="Path to custom marker JSON file"
    )

    class Config:
        schema_extra = {
            "example": {
                "cluster_key": "leiden",
                "method": "auto",
                "species": "human",
                "tissue": "brain",
                "majority_voting": True
            }
        }


# Differential Expression Tool Schemas
class CompareClustersInput(BaseModel):
    """Input schema for compare_clusters tool."""

    cluster_key: str = Field(
        default="leiden",
        description="Column in adata.obs with cluster assignments"
    )
    group1: Union[str, List[str]] = Field(
        ...,
        description="Cluster ID(s) for first group (e.g., '0' or ['0', '1'])"
    )
    group2: Union[str, List[str]] = Field(
        ...,
        description="Cluster ID(s) for second group (e.g., 'rest' or ['2', '3'])"
    )
    method: Literal["t-test", "wilcoxon", "logreg"] = Field(
        default="wilcoxon",
        description="Statistical test for differential expression"
    )
    use_raw: bool = Field(
        default=False,
        description="Use raw counts if available"
    )
    n_genes: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of top DE genes to report"
    )
    logfc_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Log2 fold change threshold for significance"
    )
    pval_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Adjusted p-value threshold"
    )

    class Config:
        schema_extra = {
            "example": {
                "cluster_key": "leiden",
                "group1": ["0", "1"],
                "group2": ["2", "3"],
                "method": "wilcoxon",
                "n_genes": 100,
                "logfc_threshold": 1.0,
                "pval_threshold": 0.05
            }
        }


# Tool registry mapping tool names to their input schemas
TOOL_SCHEMAS = {
    "load_data": LoadDataInput,
    "compute_qc_metrics": ComputeQCMetricsInput,
    "plot_qc": PlotQCMetricsInput,
    "apply_qc_filters": ApplyQCFiltersInput,
    "quick_graph": QuickGraphInput,
    "graph_from_rep": GraphFromRepInput,
    "final_graph": FinalGraphInput,
    "run_scar": RunScarInput,
    "run_scvi": RunScviInput,
    "detect_doublets": DetectDoubletsInput,
    "apply_doublet_filter": ApplyDoubletFilterInput,
    "detect_marker_genes": DetectMarkerGenesInput,
    "annotate_clusters": AnnotateClustersInput,
    "compare_clusters": CompareClustersInput,
}


def validate_tool_input(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tool input parameters using Pydantic schemas.
    
    Args:
        tool_name: Name of the tool
        params: Input parameters to validate
        
    Returns:
        Validated parameters
        
    Raises:
        ValueError: If tool is unknown or parameters are invalid
    """
    if tool_name not in TOOL_SCHEMAS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    schema_class = TOOL_SCHEMAS[tool_name]
    
    try:
        validated = schema_class(**params)
        return validated.dict()
    except Exception as e:
        raise ValueError(f"Invalid parameters for {tool_name}: {e}")


def get_tool_schema(tool_name: str) -> Optional[type]:
    """Get the Pydantic schema for a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Pydantic schema class or None if not found
    """
    return TOOL_SCHEMAS.get(tool_name)


def get_tool_schema_json(tool_name: str) -> Dict[str, Any]:
    """Get the JSON schema for a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        JSON schema dictionary
    """
    schema_class = TOOL_SCHEMAS.get(tool_name)
    if schema_class is None:
        return {}
    
    return schema_class.schema()


def list_available_tools() -> List[str]:
    """List all available tools with schemas.
    
    Returns:
        List of tool names
    """
    return list(TOOL_SCHEMAS.keys())


# Tool description metadata
TOOL_DESCRIPTIONS = {
    "load_data": "Import AnnData files (.h5ad) into the session for analysis",
    "compute_qc_metrics": "Calculate quality control metrics including mitochondrial percentages and gene counts",
    "plot_qc": "Generate visualizations of QC metrics to assess data quality",
    "apply_qc_filters": "Apply quality control filters to remove low-quality cells and genes",
    "quick_graph": "Perform PCA → neighbors → UMAP → Leiden clustering for quick analysis",
    "graph_from_rep": "Generate graph analysis from a specific representation (e.g., X_scVI)",
    "final_graph": "Final graph analysis step with optimized parameters",
    "run_scar": "Apply scAR (single-cell Ambient Remover) for denoising ambient RNA",
    "run_scvi": "Train scVI model for batch correction and latent representation learning",
    "detect_doublets": "Identify doublets (multi-cell droplets) using Scrublet or DoubletFinder",
    "apply_doublet_filter": "Remove detected doublets from the dataset",
    "detect_marker_genes": "Detect marker genes for each cluster using differential expression analysis",
    "annotate_clusters": "Annotate clusters with cell type labels using CellTypist or built-in markers",
    "compare_clusters": "Perform differential expression analysis between cluster groups with volcano plots",
}


def get_tool_description(tool_name: str) -> str:
    """Get description for a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool description
    """
    return TOOL_DESCRIPTIONS.get(tool_name, "No description available")
