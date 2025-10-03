"""Tests for ambient RNA correction tools (Phase 8)."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scqc_agent.state import SessionState, ToolResult
from scqc_agent.tools.ambient import (
    soupx,
    decontx,
    compare_ambient_methods,
    _mock_soupx_correction,
    _mock_decontx_correction
)


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = SessionState(run_id="test_ambient")
        state.adata_path = str(Path(temp_dir) / "test_data.h5ad")
        
        # Create a dummy file
        Path(state.adata_path).touch()
        
        yield state


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    mock_adata = Mock()
    
    # Mock basic properties
    mock_adata.n_obs = 1000
    mock_adata.n_vars = 2000
    
    # Mock X matrix (sparse matrix behavior)
    mock_x = Mock()
    mock_x.sum.return_value = Mock()
    mock_x.sum.return_value.A1 = np.random.poisson(1000, 1000)  # Total UMI per cell
    mock_x.toarray.return_value = np.random.poisson(1, (1000, 2000))
    mock_x.__getitem__ = Mock(return_value=mock_x)
    mock_adata.X = mock_x
    
    # Mock var (gene names)
    mock_adata.var_names = [f"Gene_{i}" for i in range(2000)]
    
    # Mock obs (cell metadata)
    mock_adata.obs = {"cell_id": [f"Cell_{i}" for i in range(1000)]}
    
    # Mock copy method
    mock_adata.copy.return_value = mock_adata
    
    # Mock write method
    mock_adata.write_h5ad = Mock()
    
    return mock_adata


class TestSoupXCorrection:
    """Test suite for SoupX ambient RNA correction."""
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.ambient.sc.read_h5ad')
    @patch('scqc_agent.tools.ambient.ensure_run_dir')
    def test_soupx_basic_functionality(self, mock_ensure_dir, mock_read_h5ad, mock_session_state):
        """Test basic SoupX functionality."""
        # Setup mocks
        mock_step_dir = Path("/mock/step/dir")
        mock_ensure_dir.return_value = mock_step_dir
        
        mock_adata = Mock()
        mock_adata.copy.return_value = mock_adata
        mock_adata.X = np.random.poisson(1, (100, 200))
        mock_read_h5ad.return_value = mock_adata
        
        # Mock _mock_soupx_correction
        with patch('scqc_agent.tools.ambient._mock_soupx_correction') as mock_correction:
            mock_corrected = Mock()
            mock_corrected.obs = {"soupx_contamination_rate": np.random.uniform(0, 0.1, 100)}
            mock_correction.return_value = mock_corrected
            
            # Mock _generate_soupx_artifacts
            with patch('scqc_agent.tools.ambient._generate_soupx_artifacts') as mock_artifacts:
                mock_artifacts.return_value = [Path("test_artifact.png")]
                
                # Run SoupX
                result = soupx(mock_session_state, contamination_rate=0.1)
        
        # Verify results
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ SoupX ambient RNA correction completed")
        assert "contamination_rate" in result.state_delta.get("soupx_params", {})
        assert len(result.artifacts) > 0
        assert len(result.citations) > 0
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', False)
    def test_soupx_no_scanpy(self, mock_session_state):
        """Test SoupX error handling when scanpy is not available."""
        with pytest.raises(ImportError, match="Scanpy is required"):
            soupx(mock_session_state)
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.ambient.sc.read_h5ad')
    def test_soupx_error_handling(self, mock_read_h5ad, mock_session_state):
        """Test SoupX error handling."""
        # Make read_h5ad raise an exception
        mock_read_h5ad.side_effect = Exception("File not found")
        
        result = soupx(mock_session_state)
        
        assert isinstance(result, ToolResult)
        assert result.message.startswith("❌ SoupX correction failed")
        assert result.state_delta == {}
        assert result.artifacts == []
    
    def test_soupx_parameter_validation(self, mock_session_state):
        """Test SoupX parameter validation."""
        with patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True):
            with patch('scqc_agent.tools.ambient.sc.read_h5ad'):
                with patch('scqc_agent.tools.ambient._mock_soupx_correction'):
                    with patch('scqc_agent.tools.ambient._generate_soupx_artifacts'):
                        # Test with various parameter combinations
                        result = soupx(
                            mock_session_state,
                            contamination_rate=0.15,
                            clusters="leiden",
                            n_top_genes=50,
                            use_raw_counts=False
                        )
                        
                        assert isinstance(result, ToolResult)
                        params = result.state_delta.get("soupx_params", {})
                        assert params["contamination_rate"] == 0.15
                        assert params["n_top_genes"] == 50


class TestDecontXCorrection:
    """Test suite for DecontX ambient RNA correction."""
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.ambient.sc.read_h5ad')
    @patch('scqc_agent.tools.ambient.ensure_run_dir')
    def test_decontx_basic_functionality(self, mock_ensure_dir, mock_read_h5ad, mock_session_state):
        """Test basic DecontX functionality."""
        # Setup mocks
        mock_step_dir = Path("/mock/step/dir")
        mock_ensure_dir.return_value = mock_step_dir
        
        mock_adata = Mock()
        mock_adata.copy.return_value = mock_adata
        mock_read_h5ad.return_value = mock_adata
        
        # Mock _mock_decontx_correction
        with patch('scqc_agent.tools.ambient._mock_decontx_correction') as mock_correction:
            mock_corrected = Mock()
            mock_corrected.obs = {"decontx_contamination": np.random.uniform(0, 0.2, 100)}
            mock_correction.return_value = mock_corrected
            
            # Mock _generate_decontx_artifacts
            with patch('scqc_agent.tools.ambient._generate_decontx_artifacts') as mock_artifacts:
                mock_artifacts.return_value = [Path("test_artifact.png")]
                
                # Run DecontX
                result = decontx(mock_session_state, max_contamination=0.3)
        
        # Verify results
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ DecontX ambient RNA correction completed")
        assert "max_contamination" in result.state_delta.get("decontx_params", {})
        assert len(result.artifacts) > 0
        assert len(result.citations) > 0
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.ambient.sc.read_h5ad')
    def test_decontx_parameter_validation(self, mock_read_h5ad, mock_session_state):
        """Test DecontX parameter validation."""
        mock_adata = Mock()
        mock_read_h5ad.return_value = mock_adata
        
        with patch('scqc_agent.tools.ambient._mock_decontx_correction'):
            with patch('scqc_agent.tools.ambient._generate_decontx_artifacts'):
                # Test with custom parameters
                result = decontx(
                    mock_session_state,
                    max_contamination=0.4,
                    delta=15.0,
                    estimateDelta=False,
                    convergence_threshold=0.0001,
                    max_iterations=1000
                )
                
                assert isinstance(result, ToolResult)
                params = result.state_delta.get("decontx_params", {})
                assert params["max_contamination"] == 0.4
                assert params["delta"] == 15.0
                assert params["estimateDelta"] == False


class TestAmbientMethodComparison:
    """Test suite for ambient RNA method comparison."""
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.ambient.sc.read_h5ad')
    @patch('scqc_agent.tools.ambient.ensure_run_dir')
    def test_compare_ambient_methods(self, mock_ensure_dir, mock_read_h5ad, mock_session_state):
        """Test ambient RNA method comparison."""
        # Setup mocks
        mock_step_dir = Path("/mock/step/dir")
        mock_ensure_dir.return_value = mock_step_dir
        
        mock_adata = Mock()
        mock_read_h5ad.return_value = mock_adata
        
        # Mock individual method results
        mock_soupx_result = ToolResult(
            message="SoupX completed",
            state_delta={"adata_path": "/path/to/soupx.h5ad"},
            artifacts=[],
            citations=[]
        )
        
        mock_decontx_result = ToolResult(
            message="DecontX completed",
            state_delta={"adata_path": "/path/to/decontx.h5ad"},
            artifacts=[],
            citations=[]
        )
        
        with patch('scqc_agent.tools.ambient.soupx', return_value=mock_soupx_result):
            with patch('scqc_agent.tools.ambient.decontx', return_value=mock_decontx_result):
                with patch('scqc_agent.tools.ambient._generate_method_comparison_artifacts') as mock_artifacts:
                    mock_artifacts.return_value = [Path("comparison.png")]
                    
                    result = compare_ambient_methods(
                        mock_session_state,
                        methods=["soupx", "decontx"]
                    )
        
        # Verify results
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ Ambient RNA correction comparison completed")
        assert "soupx" in result.message
        assert "decontx" in result.message
        assert len(result.artifacts) > 0


class TestMockImplementations:
    """Test suite for mock implementation functions."""
    
    def test_mock_soupx_correction(self, mock_adata):
        """Test mock SoupX correction implementation."""
        corrected = _mock_soupx_correction(
            mock_adata,
            contamination_rate=0.1,
            clusters=None,
            genes_to_use=None,
            n_top_genes=100
        )
        
        # Should be a copy
        assert corrected is not mock_adata
        
        # Check for SoupX-specific additions
        if hasattr(corrected, 'obs'):
            assert "soupx_contamination_rate" in corrected.obs
        
        if hasattr(corrected, 'var'):
            assert "soup_gene" in corrected.var
    
    def test_mock_decontx_correction(self, mock_adata):
        """Test mock DecontX correction implementation."""
        corrected = _mock_decontx_correction(
            mock_adata,
            max_contamination=0.3,
            delta=10.0,
            convergence_threshold=0.001
        )
        
        # Should be a copy
        assert corrected is not mock_adata
        
        # Check for DecontX-specific additions
        if hasattr(corrected, 'obs'):
            assert "decontx_contamination" in corrected.obs
            assert "decontx_corrected" in corrected.obs
        
        if hasattr(corrected, 'uns'):
            assert "decontx_delta" in corrected.uns


class TestArtifactGeneration:
    """Test suite for artifact generation functions."""
    
    @patch('scqc_agent.tools.ambient.plt')
    @patch('scqc_agent.tools.ambient.sns')
    def test_generate_soupx_artifacts(self, mock_sns, mock_plt, mock_adata):
        """Test SoupX artifact generation."""
        from scqc_agent.tools.ambient import _generate_soupx_artifacts
        
        # Create corrected mock with SoupX annotations
        mock_corrected = Mock()
        mock_corrected.obs = {
            "soupx_contamination_rate": np.random.uniform(0, 0.2, 100)
        }
        mock_corrected.var = {
            "soup_gene": np.random.choice([True, False], 200)
        }
        mock_corrected.var_names = [f"Gene_{i}" for i in range(200)]
        
        # Mock X matrices for comparison
        mock_adata.X.sum.return_value = np.random.poisson(1000, 100)
        mock_corrected.X = Mock()
        mock_corrected.X.sum.return_value = np.random.poisson(900, 100)  # Slightly lower after correction
        
        with tempfile.TemporaryDirectory() as temp_dir:
            step_dir = Path(temp_dir)
            artifacts = _generate_soupx_artifacts(mock_adata, mock_corrected, step_dir)
            
            # Should generate some artifacts (even if matplotlib is mocked)
            assert isinstance(artifacts, list)
    
    @patch('scqc_agent.tools.ambient.plt')
    @patch('scqc_agent.tools.ambient.sns')
    def test_generate_decontx_artifacts(self, mock_sns, mock_plt, mock_adata):
        """Test DecontX artifact generation."""
        from scqc_agent.tools.ambient import _generate_decontx_artifacts
        
        # Create corrected mock with DecontX annotations
        mock_corrected = Mock()
        mock_corrected.obs = {
            "decontx_contamination": np.random.uniform(0, 0.3, 100)
        }
        
        # Mock X matrices
        mock_adata.X.mean.return_value = np.random.poisson(10, 200)
        mock_corrected.X = Mock()
        mock_corrected.X.mean.return_value = np.random.poisson(9, 200)  # Slightly lower
        mock_corrected.var_names = [f"Gene_{i}" for i in range(200)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            step_dir = Path(temp_dir)
            artifacts = _generate_decontx_artifacts(mock_adata, mock_corrected, step_dir)
            
            # Should generate some artifacts
            assert isinstance(artifacts, list)


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_soupx_invalid_state(self):
        """Test SoupX with invalid session state."""
        invalid_state = SessionState(run_id="test")
        invalid_state.adata_path = None
        
        with patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True):
            result = soupx(invalid_state)
            
            assert isinstance(result, ToolResult)
            assert result.message.startswith("❌")
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.ambient.sc.read_h5ad')
    def test_contamination_rate_bounds(self, mock_read_h5ad, mock_session_state):
        """Test contamination rate boundary conditions."""
        mock_adata = Mock()
        mock_read_h5ad.return_value = mock_adata
        
        with patch('scqc_agent.tools.ambient._mock_soupx_correction'):
            with patch('scqc_agent.tools.ambient._generate_soupx_artifacts'):
                # Test edge case contamination rates
                result1 = soupx(mock_session_state, contamination_rate=0.0)
                result2 = soupx(mock_session_state, contamination_rate=1.0)
                
                assert isinstance(result1, ToolResult)
                assert isinstance(result2, ToolResult)
    
    def test_artifact_generation_no_matplotlib(self):
        """Test artifact generation when matplotlib is not available."""
        from scqc_agent.tools.ambient import _generate_soupx_artifacts
        
        mock_adata = Mock()
        mock_corrected = Mock()
        
        with patch('scqc_agent.tools.ambient.plt', side_effect=ImportError):
            with tempfile.TemporaryDirectory() as temp_dir:
                step_dir = Path(temp_dir)
                artifacts = _generate_soupx_artifacts(mock_adata, mock_corrected, step_dir)
                
                # Should return empty list but not crash
                assert isinstance(artifacts, list)


class TestIntegrationScenarios:
    """Test integration scenarios and realistic workflows."""
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    def test_full_soupx_workflow(self, mock_session_state):
        """Test complete SoupX workflow with realistic parameters."""
        with patch('scqc_agent.tools.ambient.sc.read_h5ad') as mock_read:
            # Create realistic mock AnnData
            mock_adata = Mock()
            mock_adata.n_obs = 5000
            mock_adata.n_vars = 20000
            mock_adata.copy.return_value = mock_adata
            mock_read.return_value = mock_adata
            
            with patch('scqc_agent.tools.ambient._mock_soupx_correction'):
                with patch('scqc_agent.tools.ambient._generate_soupx_artifacts'):
                    # Run with realistic PBMC parameters
                    result = soupx(
                        mock_session_state,
                        contamination_rate=0.08,  # Typical for 10x
                        clusters="leiden",
                        n_top_genes=100,
                        use_raw_counts=True
                    )
                    
                    assert isinstance(result, ToolResult)
                    assert result.message.startswith("✅")
                    assert "soupx_params" in result.state_delta
    
    @patch('scqc_agent.tools.ambient.SCANPY_AVAILABLE', True)
    def test_method_comparison_workflow(self, mock_session_state):
        """Test realistic method comparison workflow."""
        with patch('scqc_agent.tools.ambient.sc.read_h5ad'):
            with patch('scqc_agent.tools.ambient.soupx') as mock_soupx:
                with patch('scqc_agent.tools.ambient.decontx') as mock_decontx:
                    # Mock method results
                    mock_soupx.return_value = ToolResult(
                        message="SoupX completed",
                        state_delta={"adata_path": "/tmp/soupx.h5ad"},
                        artifacts=[Path("soupx_plot.png")],
                        citations=["SoupX paper"]
                    )
                    
                    mock_decontx.return_value = ToolResult(
                        message="DecontX completed", 
                        state_delta={"adata_path": "/tmp/decontx.h5ad"},
                        artifacts=[Path("decontx_plot.png")],
                        citations=["DecontX paper"]
                    )
                    
                    with patch('scqc_agent.tools.ambient._generate_method_comparison_artifacts'):
                        result = compare_ambient_methods(
                            mock_session_state,
                            methods=["soupx", "decontx"],
                            soupx={"contamination_rate": 0.1},
                            decontx={"max_contamination": 0.3}
                        )
                        
                        assert isinstance(result, ToolResult)
                        assert "comparison" in result.message.lower()
                        assert len(result.citations) > 0
