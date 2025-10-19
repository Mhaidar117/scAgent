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
    use_raw_data: bool = Field(
        default=True,
        description="Whether to use raw data if available (enables scvi.external.SCAR mode)"
    )
    prob: float = Field(
        default=0.995,
        ge=0.9,
        le=0.999,
        description="Probability threshold for ambient profile estimation (scvi.external.SCAR mode)"
    )
    min_ambient_counts: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Threshold for cell-free droplets (used in ambient profile calculation)"
    )

    class Config:
        schema_extra = {
            "example": {
                "batch_key": "batch",
                "epochs": 100,
                "replace_X": True,
                "random_seed": 42,
                "use_raw_data": True,
                "prob": 0.995,
                "min_ambient_counts": 100
            }
        }


class GenerateKneePlotInput(BaseModel):
    """Input schema for generate_knee_plot tool."""

    min_counts: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Threshold for classifying cell-free droplets"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional output directory (defaults to step_08_scar_knee)"
    )

    class Config:
        schema_extra = {
            "example": {
                "min_counts": 100,
                "output_dir": None
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
    threshold: Union[float, Literal["auto"]] = Field(
        default="auto",
        description="Custom doublet score threshold (auto for automatic detection)"
    )
    pK: Union[float, Literal["auto"]] = Field(
        default="auto",
        description="DoubletFinder neighborhood parameter (auto to optimize via sweep)"
    )
    pN: float = Field(
        default=0.25,
        ge=0.1,
        le=0.5,
        description="DoubletFinder proportion of artificial doublets"
    )
    n_prin_comps: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Number of PCA components for DoubletFinder"
    )
    run_pk_sweep: bool = Field(
        default=True,
        description="Whether to run pK optimization sweep (DoubletFinder only)"
    )
    random_seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility"
    )

    class Config:
        schema_extra = {
            "example": {
                "method": "doubletfinder",
                "expected_rate": 0.06,
                "threshold": "auto",
                "pK": "auto",
                "pN": 0.25,
                "n_prin_comps": 30,
                "run_pk_sweep": True,
                "random_seed": 0
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


class RunPkSweepInput(BaseModel):
    """Input schema for run_pk_sweep_only tool."""

    pK_grid: Tuple[float, ...] = Field(
        default=(0.005, 0.01, 0.02, 0.03, 0.05),
        description="Tuple of pK values to test"
    )
    expected_rate: float = Field(
        default=0.06,
        ge=0.01,
        le=0.5,
        description="Expected doublet rate"
    )
    pN: float = Field(
        default=0.25,
        ge=0.1,
        le=0.5,
        description="Proportion of artificial doublets"
    )
    n_prin_comps: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Number of PCA components"
    )
    random_seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility"
    )

    class Config:
        schema_extra = {
            "example": {
                "pK_grid": (0.005, 0.01, 0.02, 0.03, 0.05),
                "expected_rate": 0.06,
                "pN": 0.25,
                "n_prin_comps": 30,
                "random_seed": 0
            }
        }


class CurateDoubletsByMarkersInput(BaseModel):
    """Input schema for curate_doublets_by_markers tool."""

    marker_dict: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping cell type names to marker gene lists"
    )
    cluster_key: str = Field(
        default="leiden",
        description="Column in adata.obs containing cluster assignments"
    )
    avg_exp_threshold: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Expression threshold for considering a marker expressed"
    )

    class Config:
        schema_extra = {
            "example": {
                "marker_dict": {
                    "Proximal_Tubule": ["Lrp2", "Slc5a12"],
                    "Endothelial": ["Flt1", "Emcn"],
                    "Immune": ["Ptprc", "Cd68"]
                },
                "cluster_key": "leiden",
                "avg_exp_threshold": 2.0
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


# Checkpoint Visualization Tool Schemas
class GenerateCheckpointUmapInput(BaseModel):
    """Input schema for generate_checkpoint_umap tool."""

    stage_label: str = Field(
        ...,
        description="Stage name (e.g., 'postSCAR', 'postDoublets')"
    )
    layer: Optional[str] = Field(
        default=None,
        description="Layer to use as X (e.g., 'counts_denoised'). If None, uses current X"
    )
    resolution: float = Field(
        default=2.0,
        ge=0.1,
        le=5.0,
        description="Leiden clustering resolution"
    )
    n_pcs: int = Field(
        default=40,
        ge=10,
        le=100,
        description="Number of principal components"
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )

    class Config:
        schema_extra = {
            "example": {
                "stage_label": "postSCAR",
                "layer": "counts_denoised",
                "resolution": 2.0,
                "n_pcs": 40,
                "random_seed": 42
            }
        }


# Multi-file Loader Tool Schemas
class LoadKidneyDataInput(BaseModel):
    """Input schema for load_kidney_data tool.

    Loads kidney scRNA-seq datasets consisting of raw 10X HDF5, filtered 10X HDF5,
    and metadata CSV files. Validates file existence and formats.
    """

    raw_h5_path: str = Field(
        ...,
        description="Path to raw (unfiltered) 10X HDF5 matrix file containing all droplets"
    )
    filtered_h5_path: str = Field(
        ...,
        description="Path to filtered 10X HDF5 matrix file containing cells only"
    )
    meta_csv_path: str = Field(
        ...,
        description="Path to metadata CSV file with sample annotations (species, sex, age, tissue_type, etc.)"
    )
    sample_id_column: str = Field(
        default="sample_ID",
        description="Column name in metadata CSV containing sample identifiers"
    )
    metadata_merge_column: Optional[str] = Field(
        default=None,
        description="Column in metadata to merge with AnnData.obs (defaults to sample_id_column)"
    )
    make_unique: bool = Field(
        default=True,
        description="Make gene names unique by appending suffixes to duplicates"
    )

    @validator('raw_h5_path')
    def raw_h5_must_exist(cls, v):
        """Validate raw H5 file exists and has correct extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Raw H5 file does not exist: {v}')
        if not path.suffix.lower() in ['.h5', '.hdf5']:
            raise ValueError(f'Raw file must be a 10X H5 file (.h5 or .hdf5): {v}')
        return str(path.absolute())

    @validator('filtered_h5_path')
    def filtered_h5_must_exist(cls, v):
        """Validate filtered H5 file exists and has correct extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Filtered H5 file does not exist: {v}')
        if not path.suffix.lower() in ['.h5', '.hdf5']:
            raise ValueError(f'Filtered file must be a 10X H5 file (.h5 or .hdf5): {v}')
        return str(path.absolute())

    @validator('meta_csv_path')
    def meta_csv_must_exist(cls, v):
        """Validate metadata CSV file exists and has correct extension."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f'Metadata CSV file does not exist: {v}')
        if not path.suffix.lower() == '.csv':
            raise ValueError(f'Metadata file must be a CSV file (.csv): {v}')
        return str(path.absolute())

    @validator('sample_id_column')
    def sample_id_column_not_empty(cls, v):
        """Validate sample ID column name is not empty."""
        if not v or not v.strip():
            raise ValueError('sample_id_column cannot be empty')
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "raw_h5_path": "data/kidney_raw.h5",
                "filtered_h5_path": "data/kidney_filtered.h5",
                "meta_csv_path": "data/kidney_metadata.csv",
                "sample_id_column": "sample_ID",
                "metadata_merge_column": None,
                "make_unique": True
            }
        }


# Tool registry mapping tool names to their input schemas
TOOL_SCHEMAS = {
    "load_data": LoadDataInput,
    "load_kidney_data": LoadKidneyDataInput,
    "compute_qc_metrics": ComputeQCMetricsInput,
    "plot_qc": PlotQCMetricsInput,
    "apply_qc_filters": ApplyQCFiltersInput,
    "quick_graph": QuickGraphInput,
    "graph_from_rep": GraphFromRepInput,
    "final_graph": FinalGraphInput,
    "run_scar": RunScarInput,
    "generate_knee_plot": GenerateKneePlotInput,
    "run_scvi": RunScviInput,
    "detect_doublets": DetectDoubletsInput,
    "apply_doublet_filter": ApplyDoubletFilterInput,
    "run_pk_sweep_only": RunPkSweepInput,
    "curate_doublets_by_markers": CurateDoubletsByMarkersInput,
    "detect_marker_genes": DetectMarkerGenesInput,
    "annotate_clusters": AnnotateClustersInput,
    "compare_clusters": CompareClustersInput,
    "generate_checkpoint_umap": GenerateCheckpointUmapInput,
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
    "load_kidney_data": "Load kidney scRNA-seq datasets from raw 10X H5, filtered 10X H5, and metadata CSV files",
    "compute_qc_metrics": "Calculate quality control metrics including mitochondrial percentages and gene counts",
    "plot_qc": "Generate visualizations of QC metrics to assess data quality",
    "apply_qc_filters": "Apply quality control filters to remove low-quality cells and genes",
    "quick_graph": "Perform PCA → neighbors → UMAP → Leiden clustering for quick analysis",
    "graph_from_rep": "Generate graph analysis from a specific representation (e.g., X_scVI)",
    "final_graph": "Final graph analysis step with optimized parameters",
    "run_scar": "Apply scAR (single-cell Ambient Remover) for denoising ambient RNA with dual-mode support (scvi.external.SCAR or standalone)",
    "generate_knee_plot": "Generate knee plot visualization showing droplet distribution and calculate ambient RNA profile",
    "run_scvi": "Train scVI model for batch correction and latent representation learning",
    "detect_doublets": "Identify doublets (multi-cell droplets) using Scrublet or DoubletFinder with pK optimization",
    "apply_doublet_filter": "Remove detected doublets from the dataset",
    "run_pk_sweep_only": "Run DoubletFinder pK parameter optimization without applying detection",
    "curate_doublets_by_markers": "Manually identify doublet clusters based on incompatible marker co-expression",
    "detect_marker_genes": "Detect marker genes for each cluster using differential expression analysis",
    "annotate_clusters": "Annotate clusters with cell type labels using CellTypist or built-in markers",
    "compare_clusters": "Perform differential expression analysis between cluster groups with volcano plots",
    "generate_checkpoint_umap": "Generate UMAP visualization at pipeline checkpoint without modifying data",
}


def get_tool_description(tool_name: str) -> str:
    """Get description for a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool description
    """
    return TOOL_DESCRIPTIONS.get(tool_name, "No description available")
