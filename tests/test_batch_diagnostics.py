"""Tests for batch diagnostics tools (Phase 8)."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scqc_agent.state import SessionState, ToolResult
from scqc_agent.tools.batch_diag import (
    kbet_analysis,
    lisi_analysis,
    batch_diagnostics_summary,
    _mock_kbet_analysis,
    _mock_lisi_analysis,
    _assess_batch_integration
)


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = SessionState(run_id="test_batch_diag")
        state.adata_path = str(Path(temp_dir) / "test_data.h5ad")
        
        # Create a dummy file
        Path(state.adata_path).touch()
        
        yield state


@pytest.fixture
def mock_adata_with_batches():
    """Create a mock AnnData object with batch information."""
    mock_adata = Mock()
    
    # Basic properties
    mock_adata.n_obs = 1000
    mock_adata.n_vars = 2000
    
    # Mock obs with batch information
    batch_labels = np.random.choice(['Batch1', 'Batch2', 'Batch3'], 1000)
    cell_type_labels = np.random.choice(['TypeA', 'TypeB', 'TypeC'], 1000)
    
    mock_obs = Mock()
    mock_obs.columns = ['batch', 'cell_type']
    mock_obs.__getitem__ = Mock(side_effect=lambda x: batch_labels if x == 'batch' else cell_type_labels)
    mock_obs.__contains__ = Mock(side_effect=lambda x: x in ['batch', 'cell_type'])
    mock_adata.obs = mock_obs
    
    # Mock obsm with embeddings
    pca_embedding = np.random.randn(1000, 50)
    mock_obsm = Mock()
    mock_obsm.keys.return_value = ['X_pca', 'X_umap']
    mock_obsm.__getitem__ = Mock(return_value=pca_embedding)
    mock_obsm.__contains__ = Mock(side_effect=lambda x: x in ['X_pca', 'X_umap'])
    mock_adata.obsm = mock_obsm
    
    # Mock unique method on batch column
    batch_mock = Mock()
    batch_mock.nunique.return_value = 3
    batch_mock.unique.return_value = ['Batch1', 'Batch2', 'Batch3']
    mock_adata.obs['batch'] = batch_mock
    
    cell_type_mock = Mock()
    cell_type_mock.nunique.return_value = 3
    cell_type_mock.unique.return_value = ['TypeA', 'TypeB', 'TypeC']
    mock_adata.obs['cell_type'] = cell_type_mock
    
    # Mock copy method
    mock_adata.copy.return_value = mock_adata
    
    return mock_adata


class TestKBETAnalysis:
    """Test suite for kBET batch diagnostics."""
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.sc.read_h5ad')
    @patch('scqc_agent.tools.batch_diag.ensure_run_dir')
    def test_kbet_basic_functionality(self, mock_ensure_dir, mock_read_h5ad, mock_session_state, mock_adata_with_batches):
        """Test basic kBET functionality."""
        # Setup mocks
        mock_step_dir = Path("/mock/step/dir")
        mock_ensure_dir.return_value = mock_step_dir
        mock_read_h5ad.return_value = mock_adata_with_batches
        
        # Mock kBET analysis
        with patch('scqc_agent.tools.batch_diag._mock_kbet_analysis') as mock_kbet:
            mock_kbet_results = {
                "acceptance_rate": 0.75,
                "mean_kbet_score": 0.65,
                "n_batches": 3,
                "interpretation": "Good"
            }
            mock_kbet.return_value = mock_kbet_results
            
            # Mock artifact generation
            with patch('scqc_agent.tools.batch_diag._generate_kbet_artifacts') as mock_artifacts:
                mock_artifacts.return_value = [Path("kbet_analysis.png")]
                
                # Run kBET analysis
                result = kbet_analysis(
                    mock_session_state,
                    batch_key="batch",
                    embedding_key="X_pca",
                    k=10
                )
        
        # Verify results
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ kBET analysis completed")
        assert "batch" in result.message
        assert "0.75" in result.message  # Acceptance rate
        assert "kbet_params" in result.state_delta
        assert "kbet_results" in result.state_delta
        assert len(result.artifacts) > 0
        assert len(result.citations) > 0
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', False)
    def test_kbet_no_scanpy(self, mock_session_state):
        """Test kBET error handling when scanpy is not available."""
        with pytest.raises(ImportError, match="Scanpy is required"):
            kbet_analysis(mock_session_state, batch_key="batch")
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.sc.read_h5ad')
    def test_kbet_invalid_batch_key(self, mock_read_h5ad, mock_session_state):
        """Test kBET with invalid batch key."""
        mock_adata = Mock()
        mock_adata.obs.columns = ['other_column']
        mock_read_h5ad.return_value = mock_adata
        
        result = kbet_analysis(mock_session_state, batch_key="nonexistent_batch")
        
        assert isinstance(result, ToolResult)
        assert result.message.startswith("❌ kBET analysis failed")
        assert "not found" in result.message
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.sc.read_h5ad')
    def test_kbet_invalid_embedding_key(self, mock_read_h5ad, mock_session_state, mock_adata_with_batches):
        """Test kBET with invalid embedding key."""
        mock_adata_with_batches.obsm.keys.return_value = ['X_umap']  # Remove X_pca
        mock_adata_with_batches.obsm.__contains__ = Mock(return_value=False)
        mock_read_h5ad.return_value = mock_adata_with_batches
        
        result = kbet_analysis(mock_session_state, batch_key="batch", embedding_key="X_nonexistent")
        
        assert isinstance(result, ToolResult)
        assert result.message.startswith("❌ kBET analysis failed")
    
    def test_kbet_parameter_validation(self, mock_session_state, mock_adata_with_batches):
        """Test kBET parameter validation."""
        with patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True):
            with patch('scqc_agent.tools.batch_diag.sc.read_h5ad', return_value=mock_adata_with_batches):
                with patch('scqc_agent.tools.batch_diag._mock_kbet_analysis'):
                    with patch('scqc_agent.tools.batch_diag._generate_kbet_artifacts'):
                        # Test with custom parameters
                        result = kbet_analysis(
                            mock_session_state,
                            batch_key="batch",
                            embedding_key="X_pca",
                            k=15,
                            alpha=0.01,
                            n_repeat=200,
                            subsample=500
                        )
                        
                        assert isinstance(result, ToolResult)
                        params = result.state_delta.get("kbet_params", {})
                        assert params["k"] == 15
                        assert params["alpha"] == 0.01
                        assert params["n_repeat"] == 200
                        assert params["subsample"] == 500


class TestLISIAnalysis:
    """Test suite for LISI batch diagnostics."""
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.sc.read_h5ad')
    @patch('scqc_agent.tools.batch_diag.ensure_run_dir')
    def test_lisi_basic_functionality(self, mock_ensure_dir, mock_read_h5ad, mock_session_state, mock_adata_with_batches):
        """Test basic LISI functionality."""
        # Setup mocks
        mock_step_dir = Path("/mock/step/dir")
        mock_ensure_dir.return_value = mock_step_dir
        mock_read_h5ad.return_value = mock_adata_with_batches
        
        # Mock LISI analysis
        with patch('scqc_agent.tools.batch_diag._mock_lisi_analysis') as mock_lisi:
            mock_lisi_results = {
                "batch_lisi_median": 2.1,
                "batch_lisi_mean": 2.05,
                "label_lisi_median": 1.3,
                "n_batches": 3,
                "n_labels": 3
            }
            mock_lisi.return_value = mock_lisi_results
            
            # Mock artifact generation
            with patch('scqc_agent.tools.batch_diag._generate_lisi_artifacts') as mock_artifacts:
                mock_artifacts.return_value = [Path("lisi_analysis.png")]
                
                # Run LISI analysis
                result = lisi_analysis(
                    mock_session_state,
                    batch_key="batch",
                    label_key="cell_type",
                    embedding_key="X_pca"
                )
        
        # Verify results
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ LISI analysis completed")
        assert "2.1" in result.message  # Batch LISI median
        assert "1.3" in result.message  # Label LISI median
        assert "lisi_params" in result.state_delta
        assert "lisi_results" in result.state_delta
        assert len(result.artifacts) > 0
        assert len(result.citations) > 0
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.sc.read_h5ad')
    def test_lisi_without_label_key(self, mock_read_h5ad, mock_session_state, mock_adata_with_batches):
        """Test LISI analysis without label key."""
        mock_read_h5ad.return_value = mock_adata_with_batches
        
        with patch('scqc_agent.tools.batch_diag._mock_lisi_analysis') as mock_lisi:
            mock_lisi_results = {
                "batch_lisi_median": 2.3,
                "batch_lisi_mean": 2.25,
                "n_batches": 3
            }
            mock_lisi.return_value = mock_lisi_results
            
            with patch('scqc_agent.tools.batch_diag._generate_lisi_artifacts'):
                result = lisi_analysis(
                    mock_session_state,
                    batch_key="batch",
                    label_key=None,
                    embedding_key="X_pca"
                )
        
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ LISI analysis completed")
        assert "Label LISI" not in result.message  # Should not mention label LISI
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.sc.read_h5ad')
    def test_lisi_invalid_label_key(self, mock_read_h5ad, mock_session_state, mock_adata_with_batches):
        """Test LISI with invalid label key."""
        mock_adata_with_batches.obs.__contains__ = Mock(side_effect=lambda x: x == 'batch')
        mock_read_h5ad.return_value = mock_adata_with_batches
        
        result = lisi_analysis(
            mock_session_state,
            batch_key="batch",
            label_key="nonexistent_labels"
        )
        
        assert isinstance(result, ToolResult)
        assert result.message.startswith("❌ LISI analysis failed")


class TestBatchDiagnosticsSummary:
    """Test suite for comprehensive batch diagnostics."""
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    @patch('scqc_agent.tools.batch_diag.ensure_run_dir')
    def test_batch_diagnostics_summary(self, mock_ensure_dir, mock_session_state):
        """Test comprehensive batch diagnostics summary."""
        mock_step_dir = Path("/mock/step/dir")
        mock_ensure_dir.return_value = mock_step_dir
        
        # Mock individual analysis results
        mock_kbet_result = ToolResult(
            message="kBET completed",
            state_delta={
                "kbet_results": {
                    "acceptance_rate": 0.8,
                    "interpretation": "Good"
                }
            },
            artifacts=[Path("kbet.png")],
            citations=["kBET paper"]
        )
        
        mock_lisi_result = ToolResult(
            message="LISI completed",
            state_delta={
                "lisi_results": {
                    "batch_lisi_median": 2.5,
                    "n_batches": 3
                }
            },
            artifacts=[Path("lisi.png")],
            citations=["LISI paper"]
        )
        
        with patch('scqc_agent.tools.batch_diag.kbet_analysis', return_value=mock_kbet_result):
            with patch('scqc_agent.tools.batch_diag.lisi_analysis', return_value=mock_lisi_result):
                with patch('scqc_agent.tools.batch_diag._generate_summary_artifacts') as mock_summary:
                    mock_summary.return_value = [Path("summary.png")]
                    
                    result = batch_diagnostics_summary(
                        mock_session_state,
                        batch_key="batch",
                        embedding_key="X_pca",
                        methods=["kbet", "lisi"]
                    )
        
        # Verify results
        assert isinstance(result, ToolResult)
        assert result.message.startswith("✅ Batch diagnostics summary completed")
        assert "kbet" in result.message.lower()
        assert "lisi" in result.message.lower()
        assert "batch_diagnostics" in result.state_delta
        assert "batch_assessment" in result.state_delta
        assert len(result.artifacts) > 0
        assert len(result.citations) > 0


class TestMockImplementations:
    """Test suite for mock implementation functions."""
    
    def test_mock_kbet_analysis(self, mock_adata_with_batches):
        """Test mock kBET analysis implementation."""
        results = _mock_kbet_analysis(
            mock_adata_with_batches,
            batch_key="batch",
            embedding_key="X_pca",
            k=10,
            alpha=0.05,
            n_repeat=100
        )
        
        assert isinstance(results, dict)
        assert "acceptance_rate" in results
        assert "mean_kbet_score" in results
        assert "n_batches" in results
        assert "batch_stats" in results
        assert "interpretation" in results
        
        # Check value ranges
        assert 0 <= results["acceptance_rate"] <= 1
        assert results["n_batches"] == 3  # Based on mock data
        assert results["interpretation"] in ["Good", "Moderate", "Poor"]
    
    def test_mock_lisi_analysis(self, mock_adata_with_batches):
        """Test mock LISI analysis implementation."""
        results = _mock_lisi_analysis(
            mock_adata_with_batches,
            batch_key="batch",
            label_key="cell_type",
            embedding_key="X_pca",
            perplexity=30
        )
        
        assert isinstance(results, dict)
        assert "batch_lisi_scores" in results
        assert "batch_lisi_median" in results
        assert "batch_lisi_mean" in results
        assert "n_batches" in results
        
        # With label key
        assert "label_lisi_scores" in results
        assert "label_lisi_median" in results
        assert "n_labels" in results
        
        # Check value ranges
        assert 1 <= results["batch_lisi_median"] <= results["n_batches"]
        assert 1 <= results["label_lisi_median"] <= results["n_labels"]
    
    def test_mock_lisi_analysis_no_labels(self, mock_adata_with_batches):
        """Test mock LISI analysis without label key."""
        results = _mock_lisi_analysis(
            mock_adata_with_batches,
            batch_key="batch",
            label_key=None,
            embedding_key="X_pca",
            perplexity=30
        )
        
        assert isinstance(results, dict)
        assert "batch_lisi_scores" in results
        assert "label_lisi_scores" not in results  # Should not be present without label key


class TestAssessmentFunctions:
    """Test suite for assessment and interpretation functions."""
    
    def test_assess_batch_integration_excellent(self):
        """Test assessment with excellent integration metrics."""
        results = {
            "kbet": {"acceptance_rate": 0.85, "interpretation": "Good"},
            "lisi": {"batch_lisi_median": 2.8, "n_batches": 3}
        }
        
        assessment = _assess_batch_integration(results)
        
        assert assessment["quality"] == "Excellent"
        assert "successful" in assessment["recommendation"].lower()
        assert len(assessment["individual_assessments"]) == 2
    
    def test_assess_batch_integration_poor(self):
        """Test assessment with poor integration metrics."""
        results = {
            "kbet": {"acceptance_rate": 0.15, "interpretation": "Poor"},
            "lisi": {"batch_lisi_median": 1.2, "n_batches": 3}
        }
        
        assessment = _assess_batch_integration(results)
        
        assert assessment["quality"] == "Poor"
        assert "additional correction" in assessment["recommendation"].lower()
    
    def test_assess_batch_integration_mixed(self):
        """Test assessment with mixed integration metrics."""
        results = {
            "kbet": {"acceptance_rate": 0.75, "interpretation": "Good"},
            "lisi": {"batch_lisi_median": 1.1, "n_batches": 3}  # Poor LISI
        }
        
        assessment = _assess_batch_integration(results)
        
        assert assessment["quality"] == "Good"
        assert "improvement" in assessment["recommendation"].lower()
    
    def test_assess_batch_integration_no_results(self):
        """Test assessment with no diagnostic results."""
        results = {}
        
        assessment = _assess_batch_integration(results)
        
        assert assessment["quality"] == "Unknown"
        assert "no diagnostic" in assessment["recommendation"].lower()


class TestArtifactGeneration:
    """Test suite for artifact generation functions."""
    
    @patch('scqc_agent.tools.batch_diag.plt')
    @patch('scqc_agent.tools.batch_diag.sns')
    def test_generate_kbet_artifacts(self, mock_sns, mock_plt, mock_adata_with_batches):
        """Test kBET artifact generation."""
        from scqc_agent.tools.batch_diag import _generate_kbet_artifacts
        
        kbet_results = {
            "acceptance_rate": 0.75,
            "batch_stats": {
                "Batch1": {"acceptance_rate": 0.8, "n_cells": 300},
                "Batch2": {"acceptance_rate": 0.7, "n_cells": 400},
                "Batch3": {"acceptance_rate": 0.75, "n_cells": 300}
            },
            "interpretation": "Good"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            step_dir = Path(temp_dir)
            artifacts = _generate_kbet_artifacts(
                mock_adata_with_batches, 
                kbet_results, 
                step_dir, 
                "batch"
            )
            
            assert isinstance(artifacts, list)
            # Should generate some artifacts even with mocked plotting
    
    @patch('scqc_agent.tools.batch_diag.plt')
    @patch('scqc_agent.tools.batch_diag.sns')
    def test_generate_lisi_artifacts(self, mock_sns, mock_plt, mock_adata_with_batches):
        """Test LISI artifact generation."""
        from scqc_agent.tools.batch_diag import _generate_lisi_artifacts
        
        lisi_results = {
            "batch_lisi_scores": np.random.uniform(1, 3, 1000).tolist(),
            "batch_lisi_median": 2.1,
            "label_lisi_scores": np.random.uniform(1, 3, 1000).tolist(),
            "label_lisi_median": 1.3,
            "n_batches": 3,
            "n_labels": 3
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            step_dir = Path(temp_dir)
            artifacts = _generate_lisi_artifacts(
                mock_adata_with_batches,
                lisi_results,
                step_dir,
                "batch",
                "cell_type"
            )
            
            assert isinstance(artifacts, list)


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_subsample_larger_than_dataset(self, mock_session_state, mock_adata_with_batches):
        """Test subsampling when subsample size exceeds dataset size."""
        mock_adata_with_batches.n_obs = 100  # Small dataset
        
        with patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True):
            with patch('scqc_agent.tools.batch_diag.sc.read_h5ad', return_value=mock_adata_with_batches):
                with patch('scqc_agent.tools.batch_diag._mock_kbet_analysis'):
                    with patch('scqc_agent.tools.batch_diag._generate_kbet_artifacts'):
                        # Request subsample larger than dataset
                        result = kbet_analysis(
                            mock_session_state,
                            batch_key="batch",
                            subsample=500  # Larger than n_obs=100
                        )
                        
                        assert isinstance(result, ToolResult)
                        # Should use full dataset, not crash
    
    def test_single_batch_scenario(self):
        """Test diagnostics with only one batch (should handle gracefully)."""
        mock_adata = Mock()
        mock_adata.n_obs = 1000
        mock_obs = Mock()
        mock_obs.__contains__ = Mock(return_value=True)
        mock_obs.__getitem__ = Mock(return_value=np.array(['SingleBatch'] * 1000))
        mock_adata.obs = mock_obs
        
        # Single batch should still return valid results
        kbet_results = _mock_kbet_analysis(mock_adata, "batch", "X_pca", 10, 0.05, 100)
        assert isinstance(kbet_results, dict)
        assert "n_batches" in kbet_results
    
    def test_artifact_generation_no_matplotlib(self):
        """Test artifact generation when matplotlib is not available."""
        from scqc_agent.tools.batch_diag import _generate_kbet_artifacts
        
        mock_adata = Mock()
        kbet_results = {"acceptance_rate": 0.75}
        
        with patch('scqc_agent.tools.batch_diag.plt', side_effect=ImportError):
            with tempfile.TemporaryDirectory() as temp_dir:
                step_dir = Path(temp_dir)
                artifacts = _generate_kbet_artifacts(mock_adata, kbet_results, step_dir, "batch")
                
                # Should return empty list but not crash
                assert isinstance(artifacts, list)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @patch('scqc_agent.tools.batch_diag.SCANPY_AVAILABLE', True)
    def test_realistic_pbmc_integration_scenario(self, mock_session_state):
        """Test realistic PBMC batch integration scenario."""
        # Create realistic PBMC-like data
        mock_adata = Mock()
        mock_adata.n_obs = 8000  # Typical PBMC dataset size
        mock_adata.n_vars = 3000
        
        # Simulate 4 batches (e.g., 4 donors)
        batch_labels = np.random.choice(['Donor1', 'Donor2', 'Donor3', 'Donor4'], 8000)
        cell_types = np.random.choice(['CD4_T', 'CD8_T', 'B', 'NK', 'Mono'], 8000)
        
        mock_obs = Mock()
        mock_obs.__contains__ = Mock(side_effect=lambda x: x in ['batch', 'cell_type'])
        mock_obs.__getitem__ = Mock(side_effect=lambda x: batch_labels if x == 'batch' else cell_types)
        mock_adata.obs = mock_obs
        
        # Mock batch unique methods
        batch_mock = Mock()
        batch_mock.nunique.return_value = 4
        mock_adata.obs['batch'] = batch_mock
        
        with patch('scqc_agent.tools.batch_diag.sc.read_h5ad', return_value=mock_adata):
            with patch('scqc_agent.tools.batch_diag._mock_kbet_analysis') as mock_kbet:
                with patch('scqc_agent.tools.batch_diag._mock_lisi_analysis') as mock_lisi:
                    with patch('scqc_agent.tools.batch_diag._generate_summary_artifacts'):
                        # Simulate good integration
                        mock_kbet.return_value = {"acceptance_rate": 0.78, "interpretation": "Good"}
                        mock_lisi.return_value = {"batch_lisi_median": 3.2, "n_batches": 4}
                        
                        result = batch_diagnostics_summary(
                            mock_session_state,
                            batch_key="batch",
                            label_key="cell_type",
                            embedding_key="X_pca",
                            methods=["kbet", "lisi"]
                        )
                        
                        assert isinstance(result, ToolResult)
                        assert result.message.startswith("✅")
                        assessment = result.state_delta["batch_assessment"]
                        assert assessment["quality"] in ["Good", "Excellent"]
