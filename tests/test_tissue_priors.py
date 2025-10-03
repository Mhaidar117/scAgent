"""Tests for tissue-aware QC priors module (Phase 8)."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from scqc_agent.qc.priors import (
    suggest_thresholds,
    get_available_tissues,
    get_tissue_info,
    compare_tissue_thresholds,
    TISSUE_QC_PRIORS
)


class TestTissueAwarePriors:
    """Test suite for tissue-aware QC priors."""
    
    def test_suggest_thresholds_default(self):
        """Test default threshold suggestions."""
        thresholds = suggest_thresholds()
        
        assert isinstance(thresholds, dict)
        assert "min_genes" in thresholds
        assert "max_genes" in thresholds
        assert "max_pct_mt" in thresholds
        assert "doublet_rate" in thresholds
        assert "tissue" in thresholds
        assert "stringency" in thresholds
        assert "species" in thresholds
        
        # Check default values
        assert thresholds["tissue"] == "default"
        assert thresholds["stringency"] == "default"
        assert thresholds["species"] == "human"
    
    def test_suggest_thresholds_brain_tissue(self):
        """Test brain tissue-specific thresholds."""
        thresholds = suggest_thresholds(tissue="brain", stringency="strict")
        
        assert thresholds["tissue"] == "brain"
        assert thresholds["stringency"] == "strict"
        assert thresholds["min_genes"] == 800  # Strict brain threshold
        assert thresholds["max_pct_mt"] == 10.0  # Strict brain MT threshold
        assert "brain" in thresholds["description"].lower()
    
    def test_suggest_thresholds_pbmc_tissue(self):
        """Test PBMC tissue-specific thresholds."""
        thresholds = suggest_thresholds(tissue="pbmc", stringency="lenient")
        
        assert thresholds["tissue"] == "pbmc"
        assert thresholds["stringency"] == "lenient"
        assert thresholds["min_genes"] == 100  # Lenient PBMC threshold
        assert thresholds["max_pct_mt"] == 25.0  # Lenient PBMC MT threshold
    
    def test_suggest_thresholds_mouse_species(self):
        """Test mouse species adjustments."""
        human_thresholds = suggest_thresholds(tissue="brain", species="human")
        mouse_thresholds = suggest_thresholds(tissue="brain", species="mouse")
        
        # Mouse cells should have lower gene count thresholds
        assert mouse_thresholds["min_genes"] < human_thresholds["min_genes"]
        assert mouse_thresholds["max_genes"] < human_thresholds["max_genes"]
        assert mouse_thresholds["species"] == "mouse"
        assert "mouse" in mouse_thresholds.get("notes", "").lower()
    
    def test_suggest_thresholds_unknown_tissue(self):
        """Test handling of unknown tissue types."""
        with pytest.warns(UserWarning, match="Unknown tissue.*using.*default"):
            thresholds = suggest_thresholds(tissue="unknown_tissue")
        
        assert thresholds["tissue"] == "default"
    
    def test_suggest_thresholds_custom_overrides(self):
        """Test custom parameter overrides."""
        custom_overrides = {
            "min_genes": 1000,
            "max_pct_mt": 5.0,
            "custom_param": "test_value"
        }
        
        thresholds = suggest_thresholds(
            tissue="brain",
            custom_overrides=custom_overrides
        )
        
        assert thresholds["min_genes"] == 1000
        assert thresholds["max_pct_mt"] == 5.0
        assert thresholds["custom_param"] == "test_value"
    
    @patch('scqc_agent.qc.priors.ANNDATA_AVAILABLE', True)
    def test_suggest_thresholds_with_adata(self):
        """Test data-driven adjustments with AnnData object."""
        # Mock AnnData object
        mock_adata = Mock()
        mock_adata.X = Mock()
        mock_adata.var = Mock()
        mock_adata.var_names = Mock()
        
        # Mock gene count distribution
        mock_gene_counts = np.array([100, 200, 300, 400, 500] * 100)  # 500 cells
        mock_adata.X.sum.return_value = Mock()
        mock_adata.X.sum.return_value.A1 = mock_gene_counts
        
        # Mock sparse matrix behavior
        mock_counts_per_cell = np.random.poisson(1000, (500, 1000))  # 500 cells, 1000 genes
        mock_adata.X.__gt__ = Mock()
        mock_adata.X.__gt__.return_value.sum.return_value.A1 = (mock_counts_per_cell > 0).sum(axis=1)
        
        # Mock MT genes
        mock_adata.var_names.str.startswith.return_value.sum.return_value = 10
        
        thresholds = suggest_thresholds(adata=mock_adata, tissue="brain")
        
        # Should still return valid thresholds even with mock data
        assert isinstance(thresholds, dict)
        assert "min_genes" in thresholds
        assert "tissue" in thresholds
    
    def test_get_available_tissues(self):
        """Test getting available tissue types."""
        tissues = get_available_tissues()
        
        assert isinstance(tissues, dict)
        assert "brain" in tissues
        assert "pbmc" in tissues
        assert "liver" in tissues
        assert "default" in tissues
        
        # Check descriptions
        assert "brain" in tissues["brain"].lower()
        assert "blood" in tissues["pbmc"].lower()
    
    def test_get_tissue_info_valid(self):
        """Test getting detailed tissue information."""
        brain_info = get_tissue_info("brain")
        
        assert isinstance(brain_info, dict)
        assert "description" in brain_info
        assert "min_genes" in brain_info
        assert "sources" in brain_info
        assert "notes" in brain_info
        
        # Check it's a copy (not reference to original)
        brain_info["test_modification"] = "test"
        assert "test_modification" not in TISSUE_QC_PRIORS["brain"]
    
    def test_get_tissue_info_invalid(self):
        """Test error handling for invalid tissue."""
        with pytest.raises(ValueError, match="Unknown tissue.*Available"):
            get_tissue_info("invalid_tissue")
    
    @patch('scqc_agent.qc.priors.pd')
    def test_compare_tissue_thresholds(self, mock_pd):
        """Test tissue threshold comparison."""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_pd.DataFrame.return_value = mock_df
        
        tissues = ["brain", "pbmc", "liver"]
        result = compare_tissue_thresholds(tissues, stringency="default")
        
        # Should call pandas DataFrame constructor
        mock_pd.DataFrame.assert_called_once()
        
        # Check the data structure passed to DataFrame
        call_args = mock_pd.DataFrame.call_args[0][0]
        assert len(call_args) == len(tissues)
        
        # Check first row has expected structure
        first_row = call_args[0]
        assert "tissue" in first_row
        assert "min_genes" in first_row
        assert "max_pct_mt" in first_row
    
    def test_compare_tissue_thresholds_no_pandas(self):
        """Test comparison without pandas."""
        with patch('scqc_agent.qc.priors.pd', side_effect=ImportError):
            with pytest.raises(ImportError, match="pandas is required"):
                compare_tissue_thresholds(["brain", "pbmc"])
    
    def test_all_tissues_have_required_fields(self):
        """Test that all defined tissues have required threshold fields."""
        required_fields = ["min_genes", "max_genes", "max_pct_mt", "doublet_rate"]
        
        for tissue, priors in TISSUE_QC_PRIORS.items():
            for field in required_fields:
                assert field in priors, f"Tissue {tissue} missing field {field}"
                
                # Check that each stringency level is defined
                if isinstance(priors[field], dict):
                    for stringency in ["lenient", "default", "strict"]:
                        assert stringency in priors[field], \
                            f"Tissue {tissue}, field {field} missing stringency {stringency}"
    
    def test_threshold_value_ranges(self):
        """Test that threshold values are in reasonable ranges."""
        for tissue, priors in TISSUE_QC_PRIORS.items():
            if tissue == "description":  # Skip metadata fields
                continue
            
            # Test min_genes
            if "min_genes" in priors and isinstance(priors["min_genes"], dict):
                for stringency, value in priors["min_genes"].items():
                    if isinstance(value, (int, float)):
                        assert 0 <= value <= 2000, f"min_genes out of range for {tissue}:{stringency}"
            
            # Test max_pct_mt
            if "max_pct_mt" in priors and isinstance(priors["max_pct_mt"], dict):
                for stringency, value in priors["max_pct_mt"].items():
                    if isinstance(value, (int, float)):
                        assert 0 <= value <= 100, f"max_pct_mt out of range for {tissue}:{stringency}"
            
            # Test doublet_rate
            if "doublet_rate" in priors and isinstance(priors["doublet_rate"], dict):
                for stringency, value in priors["doublet_rate"].items():
                    if isinstance(value, (int, float)):
                        assert 0 <= value <= 1, f"doublet_rate out of range for {tissue}:{stringency}"
    
    def test_stringency_ordering(self):
        """Test that stringency levels are properly ordered (strict < default < lenient)."""
        for tissue, priors in TISSUE_QC_PRIORS.items():
            if tissue == "description":
                continue
            
            # Check min_genes ordering (strict > default > lenient)
            if "min_genes" in priors and isinstance(priors["min_genes"], dict):
                min_genes = priors["min_genes"]
                if all(key in min_genes for key in ["strict", "default", "lenient"]):
                    assert min_genes["strict"] >= min_genes["default"] >= min_genes["lenient"], \
                        f"min_genes ordering incorrect for {tissue}"
            
            # Check max_pct_mt ordering (strict < default < lenient)
            if "max_pct_mt" in priors and isinstance(priors["max_pct_mt"], dict):
                max_pct_mt = priors["max_pct_mt"]
                if all(key in max_pct_mt for key in ["strict", "default", "lenient"]):
                    assert max_pct_mt["strict"] <= max_pct_mt["default"] <= max_pct_mt["lenient"], \
                        f"max_pct_mt ordering incorrect for {tissue}"


class TestTissuePriorsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_tissue_name(self):
        """Test handling of empty tissue name."""
        with pytest.warns(UserWarning):
            thresholds = suggest_thresholds(tissue="")
        assert thresholds["tissue"] == "default"
    
    def test_case_insensitive_tissue(self):
        """Test case-insensitive tissue matching."""
        thresholds1 = suggest_thresholds(tissue="BRAIN")
        thresholds2 = suggest_thresholds(tissue="brain")
        thresholds3 = suggest_thresholds(tissue="Brain")
        
        # All should resolve to the same tissue
        assert thresholds1["tissue"] == "brain"
        assert thresholds2["tissue"] == "brain"
        assert thresholds3["tissue"] == "brain"
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in tissue names."""
        thresholds = suggest_thresholds(tissue="  brain  ")
        assert thresholds["tissue"] == "brain"
    
    def test_invalid_stringency(self):
        """Test handling of invalid stringency values."""
        # Should use default stringency for invalid values
        thresholds = suggest_thresholds(tissue="brain", stringency="invalid")
        assert "min_genes" in thresholds  # Should still work
    
    def test_invalid_species(self):
        """Test handling of invalid species values."""
        # Should still work but not apply species-specific adjustments
        thresholds = suggest_thresholds(species="invalid_species")
        assert thresholds["species"] == "invalid_species"


class TestTissuePriorIntegration:
    """Test integration with other modules."""
    
    def test_mock_adata_integration(self):
        """Test integration with mock AnnData-like objects."""
        # Create a more realistic mock AnnData
        mock_adata = Mock()
        
        # Mock X matrix (sparse-like)
        mock_x = Mock()
        mock_x.__gt__.return_value.sum.return_value.A1 = np.random.poisson(500, 1000)  # Gene counts
        mock_adata.X = mock_x
        
        # Mock var (gene metadata)
        mock_var = Mock()
        mock_var_names = Mock()
        mock_var_names.str.startswith.return_value.sum.return_value = 10  # 10 MT genes
        mock_var.var_names = mock_var_names
        mock_adata.var = mock_var
        mock_adata.var_names = mock_var_names
        
        # Should not crash with mock data
        with patch('scqc_agent.qc.priors.ANNDATA_AVAILABLE', True):
            thresholds = suggest_thresholds(adata=mock_adata, tissue="brain")
            assert isinstance(thresholds, dict)
