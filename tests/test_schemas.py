"""Tests for Pydantic tool schemas."""

import pytest
from pydantic import ValidationError

from scqc_agent.agent.schemas import (
    validate_tool_input,
    get_tool_schema,
    get_tool_schema_json,
    list_available_tools,
    get_tool_description,
    ComputeQCMetricsInput,
    ApplyQCFiltersInput,
    QuickGraphInput,
    RunScviInput,
    DetectDoubletsInput,
    TOOL_SCHEMAS
)


class TestToolSchemas:
    """Test suite for tool schemas."""
    
    def test_compute_qc_metrics_schema(self):
        """Test ComputeQCMetricsInput schema."""
        # Valid input
        valid_input = {"species": "human", "mito_prefix": None}
        schema = ComputeQCMetricsInput(**valid_input)
        assert schema.species == "human"
        assert schema.mito_prefix is None
        
        # Test defaults
        default_schema = ComputeQCMetricsInput()
        assert default_schema.species == "human"
        assert default_schema.mito_prefix is None
        
        # Invalid species
        with pytest.raises(ValidationError):
            ComputeQCMetricsInput(species="invalid")
    
    def test_apply_qc_filters_schema(self):
        """Test ApplyQCFiltersInput schema."""
        # Valid input
        valid_input = {
            "min_genes": 200,
            "max_genes": 5000,
            "min_counts": 1000,
            "max_counts": 50000,
            "max_pct_mt": 20.0,
            "min_cells": 3,
            "method": "threshold"
        }
        schema = ApplyQCFiltersInput(**valid_input)
        assert schema.min_genes == 200
        assert schema.max_pct_mt == 20.0
        
        # Test validation - max_genes <= min_genes
        with pytest.raises(ValidationError):
            ApplyQCFiltersInput(min_genes=500, max_genes=400)
        
        # Test validation - negative values
        with pytest.raises(ValidationError):
            ApplyQCFiltersInput(min_genes=-10)
        
        # Test validation - MT percentage > 100
        with pytest.raises(ValidationError):
            ApplyQCFiltersInput(max_pct_mt=150.0)
        
        # Test validation - invalid method
        with pytest.raises(ValidationError):
            ApplyQCFiltersInput(method="invalid_method")
    
    def test_quick_graph_schema(self):
        """Test QuickGraphInput schema."""
        # Valid input
        valid_input = {
            "n_neighbors": 15,
            "n_pcs": 40,
            "resolution": 0.5,
            "seed": 42
        }
        schema = QuickGraphInput(**valid_input)
        assert schema.n_neighbors == 15
        assert schema.resolution == 0.5
        
        # Test bounds
        with pytest.raises(ValidationError):
            QuickGraphInput(n_neighbors=1)  # Too low
        
        with pytest.raises(ValidationError):
            QuickGraphInput(n_neighbors=200)  # Too high
        
        with pytest.raises(ValidationError):
            QuickGraphInput(resolution=0.01)  # Too low
        
        with pytest.raises(ValidationError):
            QuickGraphInput(resolution=5.0)  # Too high
    
    def test_run_scvi_schema(self):
        """Test RunScviInput schema."""
        # Valid input
        valid_input = {
            "batch_key": "batch",
            "n_latent": 30,
            "epochs": 200,
            "random_seed": 42
        }
        schema = RunScviInput(**valid_input)
        assert schema.batch_key == "batch"
        assert schema.n_latent == 30
        
        # Test bounds
        with pytest.raises(ValidationError):
            RunScviInput(n_latent=2)  # Too low
        
        with pytest.raises(ValidationError):
            RunScviInput(epochs=10)  # Too low
        
        with pytest.raises(ValidationError):
            RunScviInput(epochs=2000)  # Too high
    
    def test_detect_doublets_schema(self):
        """Test DetectDoubletsInput schema."""
        # Valid input
        valid_input = {
            "method": "scrublet",
            "expected_rate": 0.06,
            "threshold": 0.35
        }
        schema = DetectDoubletsInput(**valid_input)
        assert schema.method == "scrublet"
        assert schema.expected_rate == 0.06
        
        # Test bounds
        with pytest.raises(ValidationError):
            DetectDoubletsInput(expected_rate=0.001)  # Too low
        
        with pytest.raises(ValidationError):
            DetectDoubletsInput(expected_rate=0.8)  # Too high
        
        with pytest.raises(ValidationError):
            DetectDoubletsInput(threshold=1.5)  # Too high
        
        # Test invalid method
        with pytest.raises(ValidationError):
            DetectDoubletsInput(method="invalid")


class TestSchemaValidation:
    """Test suite for schema validation functions."""
    
    def test_validate_tool_input_success(self):
        """Test successful tool input validation."""
        params = {"species": "human", "mito_prefix": None}
        validated = validate_tool_input("compute_qc_metrics", params)
        
        assert validated["species"] == "human"
        assert validated["mito_prefix"] is None
    
    def test_validate_tool_input_unknown_tool(self):
        """Test validation with unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            validate_tool_input("unknown_tool", {})
    
    def test_validate_tool_input_invalid_params(self):
        """Test validation with invalid parameters."""
        params = {"species": "invalid_species"}
        
        with pytest.raises(ValueError, match="Invalid parameters"):
            validate_tool_input("compute_qc_metrics", params)
    
    def test_validate_tool_input_missing_required(self):
        """Test validation with missing required parameters."""
        # load_data requires file_path
        params = {"backup": True}
        
        with pytest.raises(ValueError, match="Invalid parameters"):
            validate_tool_input("load_data", params)
    
    def test_get_tool_schema(self):
        """Test getting tool schema."""
        schema = get_tool_schema("compute_qc_metrics")
        assert schema == ComputeQCMetricsInput
        
        schema = get_tool_schema("unknown_tool")
        assert schema is None
    
    def test_get_tool_schema_json(self):
        """Test getting JSON schema."""
        json_schema = get_tool_schema_json("compute_qc_metrics")
        
        assert "properties" in json_schema
        assert "species" in json_schema["properties"]
        assert "mito_prefix" in json_schema["properties"]
        
        # Unknown tool returns empty dict
        empty_schema = get_tool_schema_json("unknown_tool")
        assert empty_schema == {}
    
    def test_list_available_tools(self):
        """Test listing available tools."""
        tools = list_available_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "compute_qc_metrics" in tools
        assert "quick_graph" in tools
        assert "run_scvi" in tools
    
    def test_get_tool_description(self):
        """Test getting tool descriptions."""
        desc = get_tool_description("compute_qc_metrics")
        assert "quality control" in desc.lower()
        
        desc = get_tool_description("unknown_tool")
        assert desc == "No description available"


class TestComplexValidation:
    """Test suite for complex validation scenarios."""
    
    def test_qc_filters_validation_interactions(self):
        """Test QC filters with interacting validation rules."""
        # Valid case
        params = {
            "min_genes": 200,
            "max_genes": 5000,
            "min_counts": 1000,
            "max_counts": 50000
        }
        validated = validate_tool_input("apply_qc_filters", params)
        assert validated["min_genes"] == 200
        
        # Invalid: max < min
        with pytest.raises(ValueError):
            validate_tool_input("apply_qc_filters", {
                "min_genes": 500,
                "max_genes": 400
            })
        
        with pytest.raises(ValueError):
            validate_tool_input("apply_qc_filters", {
                "min_counts": 5000,
                "max_counts": 1000
            })
    
    def test_parameter_coercion(self):
        """Test parameter type coercion."""
        # String numbers should be coerced
        params = {
            "n_neighbors": "15",
            "resolution": "0.5"
        }
        validated = validate_tool_input("quick_graph", params)
        assert validated["n_neighbors"] == 15
        assert validated["resolution"] == 0.5
    
    def test_optional_parameters(self):
        """Test handling of optional parameters."""
        # Only required parameters
        params = {}
        validated = validate_tool_input("compute_qc_metrics", params)
        assert "species" in validated  # Has default
        
        # Mix of provided and default
        params = {"species": "mouse"}
        validated = validate_tool_input("compute_qc_metrics", params)
        assert validated["species"] == "mouse"
        assert validated["mito_prefix"] is None  # Default
    
    def test_schema_completeness(self):
        """Test that all tools have schemas."""
        expected_tools = [
            "load_data", "compute_qc_metrics", "plot_qc", "apply_qc_filters",
            "quick_graph", "graph_from_rep", "final_graph",
            "run_scar", "run_scvi", 
            "detect_doublets", "apply_doublet_filter"
        ]
        
        available_tools = list_available_tools()
        
        for tool in expected_tools:
            assert tool in available_tools, f"Missing schema for tool: {tool}"
            assert tool in TOOL_SCHEMAS, f"Tool not in TOOL_SCHEMAS: {tool}"


if __name__ == "__main__":
    pytest.main([__file__])
