"""End-to-end integration tests for Phase 8 features."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scqc_agent.state import SessionState
from scqc_agent.agent.runtime import Agent


@pytest.fixture
def mock_session_state():
    """Create a mock session state for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state = SessionState(run_id="test_phase8_integration")
        state.adata_path = str(Path(temp_dir) / "test_data.h5ad")
        
        # Create a dummy file
        Path(state.adata_path).touch()
        
        # Add some metadata
        state.metadata = {
            "adata_path": state.adata_path,
            "n_obs": 1000,
            "n_vars": 2000,
            "species": "human"
        }
        
        yield state


@pytest.fixture
def mock_agent(mock_session_state):
    """Create a mock agent for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = Path(temp_dir) / "state.json"
        mock_session_state.save(str(state_file))
        
        agent = Agent(str(state_file))
        agent.state = mock_session_state  # Override with our mock state
        
        yield agent


class TestTissueAwareIntegration:
    """Test tissue-aware priors integration with agent."""
    
    def test_tissue_detection_from_message(self, mock_agent):
        """Test tissue detection from natural language messages."""
        # Test brain tissue detection
        brain_tissue = mock_agent._detect_tissue_from_message("analyze brain cortex data")
        assert brain_tissue == "brain"
        
        # Test PBMC detection
        pbmc_tissue = mock_agent._detect_tissue_from_message("process pbmc immune cells")
        assert pbmc_tissue == "pbmc"
        
        # Test liver detection
        liver_tissue = mock_agent._detect_tissue_from_message("hepatocyte liver analysis")
        assert liver_tissue == "liver"
        
        # Test no detection
        no_tissue = mock_agent._detect_tissue_from_message("generic analysis workflow")
        assert no_tissue is None
    
    @patch('scqc_agent.agent.runtime.TISSUE_PRIORS_AVAILABLE', True)
    def test_tissue_context_generation(self, mock_agent):
        """Test tissue context generation for planning."""
        with patch('scqc_agent.agent.runtime.suggest_thresholds') as mock_suggest:
            mock_suggest.return_value = {
                "min_genes": 500,
                "max_genes": 8000,
                "max_pct_mt": 15.0,
                "doublet_rate": 0.06,
                "notes": "Conservative thresholds for neural tissues"
            }
            
            context = mock_agent._get_tissue_context("analyze brain cortex neurons")
            
            assert "brain" in context.lower()
            assert "500" in context  # min_genes
            assert "15.0" in context  # max_pct_mt
            assert "0.06" in context or "6%" in context  # doublet_rate
    
    @patch('scqc_agent.agent.runtime.TISSUE_PRIORS_AVAILABLE', True)
    def test_plan_enhancement_with_tissue_priors(self, mock_agent):
        """Test plan enhancement with tissue-specific parameters."""
        # Mock basic plan
        basic_plan = [
            {
                "tool": "apply_qc_filters",
                "description": "Apply QC filters",
                "params": {}
            },
            {
                "tool": "detect_doublets",
                "description": "Detect doublets",
                "params": {}
            }
        ]
        
        with patch('scqc_agent.agent.runtime.suggest_thresholds') as mock_suggest:
            mock_suggest.return_value = {
                "min_genes": 800,
                "max_genes": 6000,
                "max_pct_mt": 10.0,
                "doublet_rate": 0.04
            }
            
            enhanced_plan = mock_agent._enhance_plan_with_tissue_priors(
                basic_plan, "brain cortex analysis"
            )
            
            # Check QC filter enhancement
            qc_step = enhanced_plan[0]
            assert qc_step["params"]["min_genes"] == 800
            assert qc_step["params"]["max_genes"] == 6000
            assert qc_step["params"]["max_pct_mt"] == 10.0
            assert "brain" in qc_step["description"]
            
            # Check doublet detection enhancement
            doublet_step = enhanced_plan[1]
            assert doublet_step["params"]["expected_doublet_rate"] == 0.04
            assert "brain" in doublet_step["description"]
    
    def test_tissue_threshold_tool_wrapper(self, mock_agent):
        """Test tissue threshold suggestion tool wrapper."""
        with patch('scqc_agent.agent.runtime.TISSUE_PRIORS_AVAILABLE', True):
            with patch('scqc_agent.agent.runtime.suggest_thresholds') as mock_suggest:
                mock_suggest.return_value = {
                    "min_genes": 500,
                    "max_genes": 7000,
                    "max_pct_mt": 20.0,
                    "doublet_rate": 0.08,
                    "tissue": "pbmc",
                    "stringency": "default"
                }
                
                result = mock_agent._suggest_tissue_thresholds_tool({
                    "tissue": "pbmc",
                    "stringency": "default",
                    "species": "human"
                })
                
                assert result.message.startswith("✅")
                assert "pbmc" in result.message
                assert "500" in result.message  # min_genes
                assert "tissue_thresholds" in result.state_delta


class TestAmbientRNAIntegration:
    """Test ambient RNA correction integration with agent."""
    
    def test_soupx_tool_wrapper(self, mock_agent):
        """Test SoupX tool wrapper through agent."""
        with patch('scqc_agent.tools.ambient.soupx') as mock_soupx:
            from scqc_agent.state import ToolResult
            mock_soupx.return_value = ToolResult(
                message="✅ SoupX completed",
                state_delta={"adata_path": "/new/path.h5ad"},
                artifacts=[Path("soupx_plot.png")],
                citations=["SoupX paper"]
            )
            
            result = mock_agent._soupx_tool({"contamination_rate": 0.1})
            
            assert result.message.startswith("✅")
            assert "soupx_plot.png" in str(result.artifacts[0])
    
    def test_decontx_tool_wrapper(self, mock_agent):
        """Test DecontX tool wrapper through agent."""
        with patch('scqc_agent.tools.ambient.decontx') as mock_decontx:
            from scqc_agent.state import ToolResult
            mock_decontx.return_value = ToolResult(
                message="✅ DecontX completed",
                state_delta={"adata_path": "/new/path.h5ad"},
                artifacts=[Path("decontx_plot.png")],
                citations=["DecontX paper"]
            )
            
            result = mock_agent._decontx_tool({"max_contamination": 0.3})
            
            assert result.message.startswith("✅")
            assert "decontx_plot.png" in str(result.artifacts[0])
    
    def test_ambient_methods_comparison_wrapper(self, mock_agent):
        """Test ambient methods comparison tool wrapper."""
        with patch('scqc_agent.tools.ambient.compare_ambient_methods') as mock_compare:
            from scqc_agent.state import ToolResult
            mock_compare.return_value = ToolResult(
                message="✅ Ambient RNA comparison completed",
                state_delta={},
                artifacts=[Path("comparison_plot.png")],
                citations=["SoupX paper", "DecontX paper"]
            )
            
            result = mock_agent._compare_ambient_methods_tool({
                "methods": ["soupx", "decontx"]
            })
            
            assert result.message.startswith("✅")
            assert "comparison" in result.message.lower()
            assert len(result.citations) == 2


class TestBatchDiagnosticsIntegration:
    """Test batch diagnostics integration with agent."""
    
    def test_kbet_tool_wrapper(self, mock_agent):
        """Test kBET tool wrapper through agent."""
        with patch('scqc_agent.tools.batch_diag.kbet_analysis') as mock_kbet:
            from scqc_agent.state import ToolResult
            mock_kbet.return_value = ToolResult(
                message="✅ kBET analysis completed",
                state_delta={
                    "kbet_results": {"acceptance_rate": 0.75}
                },
                artifacts=[Path("kbet_analysis.png")],
                citations=["kBET paper"]
            )
            
            result = mock_agent._kbet_analysis_tool({
                "batch_key": "batch",
                "embedding_key": "X_pca"
            })
            
            assert result.message.startswith("✅")
            assert "kbet" in result.message.lower()
    
    def test_lisi_tool_wrapper(self, mock_agent):
        """Test LISI tool wrapper through agent."""
        with patch('scqc_agent.tools.batch_diag.lisi_analysis') as mock_lisi:
            from scqc_agent.state import ToolResult
            mock_lisi.return_value = ToolResult(
                message="✅ LISI analysis completed",
                state_delta={
                    "lisi_results": {"batch_lisi_median": 2.1}
                },
                artifacts=[Path("lisi_analysis.png")],
                citations=["LISI paper"]
            )
            
            result = mock_agent._lisi_analysis_tool({
                "batch_key": "batch",
                "label_key": "cell_type",
                "embedding_key": "X_pca"
            })
            
            assert result.message.startswith("✅")
            assert "lisi" in result.message.lower()
    
    def test_batch_diagnostics_summary_wrapper(self, mock_agent):
        """Test batch diagnostics summary tool wrapper."""
        with patch('scqc_agent.tools.batch_diag.batch_diagnostics_summary') as mock_summary:
            from scqc_agent.state import ToolResult
            mock_summary.return_value = ToolResult(
                message="✅ Batch diagnostics summary completed",
                state_delta={
                    "batch_assessment": {"quality": "Good"}
                },
                artifacts=[Path("batch_summary.png")],
                citations=["kBET paper", "LISI paper"]
            )
            
            result = mock_agent._batch_diagnostics_summary_tool({
                "batch_key": "batch",
                "methods": ["kbet", "lisi"]
            })
            
            assert result.message.startswith("✅")
            assert "summary" in result.message.lower()


class TestFullWorkflowIntegration:
    """Test complete workflows using Phase 8 features."""
    
    def test_tissue_aware_qc_workflow(self, mock_agent):
        """Test tissue-aware QC workflow through agent chat."""
        # Mock fallback methods since we're testing without LangChain
        mock_agent.plan_chain = None
        
        with patch.object(mock_agent, '_fallback_generate_plan') as mock_plan:
            mock_plan.return_value = [
                {
                    "tool": "apply_qc_filters",
                    "description": "Apply QC filters with brain-specific thresholds",
                    "params": {"min_genes": 800, "max_pct_mt": 10.0}
                }
            ]
            
            with patch.object(mock_agent, '_apply_qc_filters_tool') as mock_qc_tool:
                from scqc_agent.state import ToolResult
                mock_qc_tool.return_value = ToolResult(
                    message="✅ QC filters applied with brain-specific thresholds",
                    state_delta={},
                    artifacts=[Path("qc_results.png")],
                    citations=[]
                )
                
                result = mock_agent.chat("apply brain-specific QC filters to cortex data")
                
                assert result["status"] == "completed"
                assert "brain" in result["summary"].lower() or "cortex" in result["summary"].lower()
    
    @patch('scqc_agent.agent.runtime.UX_AVAILABLE', True)
    def test_ambient_rna_workflow_with_ux(self, mock_agent):
        """Test ambient RNA correction workflow with UX enhancements."""
        mock_agent.plan_chain = None
        
        with patch.object(mock_agent, '_fallback_generate_plan') as mock_plan:
            mock_plan.return_value = [
                {
                    "tool": "soupx",
                    "description": "Remove ambient RNA with SoupX",
                    "params": {"contamination_rate": 0.1}
                }
            ]
            
            with patch.object(mock_agent, '_soupx_tool') as mock_soupx:
                from scqc_agent.state import ToolResult
                mock_soupx.return_value = ToolResult(
                    message="✅ SoupX ambient RNA correction completed",
                    state_delta={"adata_path": "/corrected/data.h5ad"},
                    artifacts=[Path("soupx_results.png")],
                    citations=["Young & Behjati (2020)"]
                )
                
                # Mock UX manager
                with patch('scqc_agent.agent.runtime.get_ux_manager') as mock_ux:
                    mock_ux_instance = Mock()
                    mock_ux.return_value = mock_ux_instance
                    
                    result = mock_agent.chat("remove ambient RNA contamination using SoupX")
                    
                    assert result["status"] == "completed"
                    assert "ambient" in result["summary"].lower() or "soupx" in result["summary"].lower()
    
    def test_batch_diagnostics_workflow(self, mock_agent):
        """Test batch diagnostics workflow through agent."""
        mock_agent.plan_chain = None
        
        with patch.object(mock_agent, '_fallback_generate_plan') as mock_plan:
            mock_plan.return_value = [
                {
                    "tool": "batch_diagnostics_summary",
                    "description": "Comprehensive batch diagnostics",
                    "params": {"batch_key": "donor", "methods": ["kbet", "lisi"]}
                }
            ]
            
            with patch.object(mock_agent, '_batch_diagnostics_summary_tool') as mock_batch_tool:
                from scqc_agent.state import ToolResult
                mock_batch_tool.return_value = ToolResult(
                    message="✅ Batch diagnostics completed: Good integration",
                    state_delta={"batch_assessment": {"quality": "Good"}},
                    artifacts=[Path("batch_diagnostics.png")],
                    citations=["Büttner et al. (2019)", "Korsunsky et al. (2019)"]
                )
                
                result = mock_agent.chat("analyze batch integration quality")
                
                assert result["status"] == "completed"
                assert "batch" in result["summary"].lower()
    
    def test_multi_tool_phase8_workflow(self, mock_agent):
        """Test workflow using multiple Phase 8 tools."""
        mock_agent.plan_chain = None
        
        with patch.object(mock_agent, '_fallback_generate_plan') as mock_plan:
            mock_plan.return_value = [
                {
                    "tool": "suggest_tissue_thresholds",
                    "description": "Get tissue-specific thresholds",
                    "params": {"tissue": "pbmc", "stringency": "strict"}
                },
                {
                    "tool": "soupx",
                    "description": "Remove ambient RNA",
                    "params": {"contamination_rate": 0.08}
                },
                {
                    "tool": "kbet_analysis", 
                    "description": "Analyze batch mixing",
                    "params": {"batch_key": "batch"}
                }
            ]
            
            # Mock all tool responses
            with patch.object(mock_agent, '_suggest_tissue_thresholds_tool') as mock_thresh:
                with patch.object(mock_agent, '_soupx_tool') as mock_soupx:
                    with patch.object(mock_agent, '_kbet_analysis_tool') as mock_kbet:
                        from scqc_agent.state import ToolResult
                        
                        mock_thresh.return_value = ToolResult(
                            message="✅ PBMC thresholds suggested",
                            state_delta={"tissue_thresholds": {"min_genes": 500}},
                            artifacts=[],
                            citations=[]
                        )
                        
                        mock_soupx.return_value = ToolResult(
                            message="✅ SoupX completed",
                            state_delta={"adata_path": "/corrected.h5ad"},
                            artifacts=[Path("soupx.png")],
                            citations=["SoupX paper"]
                        )
                        
                        mock_kbet.return_value = ToolResult(
                            message="✅ kBET analysis completed",
                            state_delta={"kbet_results": {"acceptance_rate": 0.8}},
                            artifacts=[Path("kbet.png")],
                            citations=["kBET paper"]
                        )
                        
                        result = mock_agent.chat(
                            "optimize PBMC data with tissue-specific QC, ambient RNA removal, and batch diagnostics"
                        )
                        
                        assert result["status"] == "completed"
                        assert len(result["tool_results"]) == 3
                        assert len(result["artifacts"]) >= 2  # soupx.png + kbet.png
                        assert len(result["citations"]) >= 2


class TestErrorHandlingIntegration:
    """Test error handling in integrated workflows."""
    
    def test_tool_not_available_graceful_handling(self, mock_agent):
        """Test graceful handling when Phase 8 tools are not available."""
        # Test when tissue priors are not available
        with patch('scqc_agent.agent.runtime.TISSUE_PRIORS_AVAILABLE', False):
            result = mock_agent._suggest_tissue_thresholds_tool({"tissue": "brain"})
            assert result.message.startswith("❌")
            assert "not available" in result.message
    
    def test_invalid_tool_parameters(self, mock_agent):
        """Test handling of invalid tool parameters."""
        with patch('scqc_agent.tools.ambient.soupx') as mock_soupx:
            # Make soupx raise an exception due to invalid parameters
            mock_soupx.side_effect = ValueError("Invalid contamination rate")
            
            result = mock_agent._soupx_tool({"contamination_rate": -0.1})  # Invalid value
            
            # Should return an error result, not crash
            assert isinstance(result, type(mock_agent._soupx_tool({})))
    
    def test_plan_enhancement_with_missing_tissue_module(self, mock_agent):
        """Test plan enhancement when tissue module is missing."""
        with patch('scqc_agent.agent.runtime.TISSUE_PRIORS_AVAILABLE', False):
            basic_plan = [{"tool": "apply_qc_filters", "params": {}}]
            
            enhanced_plan = mock_agent._enhance_plan_with_tissue_priors(
                basic_plan, "brain analysis"
            )
            
            # Should return original plan unchanged
            assert enhanced_plan == basic_plan


class TestPhase8ToolRegistry:
    """Test Phase 8 tool registration and availability."""
    
    def test_phase8_tools_registered(self, mock_agent):
        """Test that Phase 8 tools are properly registered."""
        # Check tissue-aware tools
        assert "suggest_tissue_thresholds" in mock_agent.tools
        
        # Check ambient RNA tools
        assert "soupx" in mock_agent.tools
        assert "decontx" in mock_agent.tools
        assert "compare_ambient_methods" in mock_agent.tools
        
        # Check batch diagnostics tools
        assert "kbet_analysis" in mock_agent.tools
        assert "lisi_analysis" in mock_agent.tools
        assert "batch_diagnostics_summary" in mock_agent.tools
    
    def test_tool_wrapper_callable(self, mock_agent):
        """Test that all Phase 8 tool wrappers are callable."""
        phase8_tools = [
            "suggest_tissue_thresholds",
            "soupx", "decontx", "compare_ambient_methods",
            "kbet_analysis", "lisi_analysis", "batch_diagnostics_summary"
        ]
        
        for tool_name in phase8_tools:
            assert tool_name in mock_agent.tools
            assert callable(mock_agent.tools[tool_name])
    
    def test_tool_execution_with_empty_params(self, mock_agent):
        """Test tool execution with empty parameters."""
        # Most tools should handle empty params gracefully
        # (though they may fail validation later)
        
        # Test tissue thresholds (should work with defaults)
        with patch('scqc_agent.agent.runtime.TISSUE_PRIORS_AVAILABLE', True):
            with patch('scqc_agent.agent.runtime.suggest_thresholds'):
                result = mock_agent._suggest_tissue_thresholds_tool({})
                assert hasattr(result, 'message')  # Should return a ToolResult


class TestRealisticScenarios:
    """Test realistic use case scenarios combining Phase 8 features."""
    
    def test_pbmc_donor_batch_correction_scenario(self, mock_agent):
        """Test realistic PBMC multi-donor batch correction scenario."""
        mock_agent.plan_chain = None
        
        # Simulate a user request for PBMC batch correction workflow
        with patch.object(mock_agent, '_fallback_generate_plan') as mock_plan:
            # Create a realistic plan for PBMC batch correction
            mock_plan.return_value = [
                {
                    "tool": "suggest_tissue_thresholds",
                    "description": "Get PBMC-specific QC thresholds",
                    "params": {"tissue": "pbmc", "species": "human"}
                },
                {
                    "tool": "soupx",
                    "description": "Remove ambient RNA contamination",
                    "params": {"contamination_rate": 0.08}  # Typical for 10x
                },
                {
                    "tool": "batch_diagnostics_summary",
                    "description": "Assess batch integration quality",
                    "params": {"batch_key": "donor", "methods": ["kbet", "lisi"]}
                }
            ]
            
            # Mock the tools
            with patch.object(mock_agent, '_suggest_tissue_thresholds_tool') as mock_thresh:
                with patch.object(mock_agent, '_soupx_tool') as mock_soupx:
                    with patch.object(mock_agent, '_batch_diagnostics_summary_tool') as mock_batch:
                        from scqc_agent.state import ToolResult
                        
                        # Mock realistic responses
                        mock_thresh.return_value = ToolResult(
                            message="✅ PBMC-specific thresholds: min_genes=200, max_pct_mt=20%",
                            state_delta={"tissue_thresholds": {"tissue": "pbmc"}},
                            artifacts=[],
                            citations=[]
                        )
                        
                        mock_soupx.return_value = ToolResult(
                            message="✅ SoupX: 8% contamination removed from 10,000 cells",
                            state_delta={"adata_path": "/corrected.h5ad"},
                            artifacts=[Path("contamination_plot.png")],
                            citations=["Young & Behjati (2020)"]
                        )
                        
                        mock_batch.return_value = ToolResult(
                            message="✅ Batch diagnostics: Good integration across 4 donors",
                            state_delta={"batch_assessment": {"quality": "Good"}},
                            artifacts=[Path("batch_integration.png")],
                            citations=["Büttner et al. (2019)", "Korsunsky et al. (2019)"]
                        )
                        
                        # Execute the workflow
                        result = mock_agent.chat(
                            "Process PBMC data from 4 donors with tissue-specific QC, "
                            "ambient RNA removal, and batch integration assessment"
                        )
                        
                        # Verify comprehensive workflow execution
                        assert result["status"] == "completed"
                        assert len(result["tool_results"]) == 3
                        assert "pbmc" in result["summary"].lower()
                        assert len(result["citations"]) >= 3
                        assert len(result["artifacts"]) >= 2
    
    def test_brain_tissue_analysis_scenario(self, mock_agent):
        """Test brain tissue analysis with strict QC scenario."""
        mock_agent.plan_chain = None
        
        with patch.object(mock_agent, '_detect_tissue_from_message', return_value="brain"):
            with patch.object(mock_agent, '_fallback_generate_plan') as mock_plan:
                # Brain tissue should get strict thresholds
                mock_plan.return_value = [
                    {
                        "tool": "apply_qc_filters",
                        "description": "Apply QC filters (using brain tissue priors)",
                        "params": {
                            "min_genes": 800,      # Strict brain threshold
                            "max_pct_mt": 10.0,    # Strict brain MT threshold
                        }
                    }
                ]
                
                with patch.object(mock_agent, '_apply_qc_filters_tool') as mock_qc:
                    from scqc_agent.state import ToolResult
                    mock_qc.return_value = ToolResult(
                        message="✅ Brain-specific QC: 8,543 cells retained (strict thresholds)",
                        state_delta={},
                        artifacts=[Path("brain_qc_plots.png")],
                        citations=[]
                    )
                    
                    result = mock_agent.chat("apply strict QC to brain cortex neurons")
                    
                    assert result["status"] == "completed"
                    # Should use tissue-aware thresholds
                    qc_tool_result = result["tool_results"][0]
                    assert "brain" in qc_tool_result["message"].lower()
                    assert "strict" in qc_tool_result["message"].lower()
