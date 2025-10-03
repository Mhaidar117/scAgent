"""Tests for the agent chat functionality."""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from scqc_agent.state import SessionState, ToolResult
from scqc_agent.agent.runtime import Agent


class TestAgentChat:
    """Test suite for agent chat functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / "test_state.json"
        self.kb_path = Path(self.temp_dir) / "test_kb"
        
        # Create mock knowledge base
        self.kb_path.mkdir(exist_ok=True)
        
        # Create a test KB document
        test_doc = self.kb_path / "test_qc.md"
        test_doc.write_text("""
# Test QC Guidelines

Quality control is important for scRNA-seq analysis.

## Parameters
- Min genes: 200-500
- Max MT%: 15-25%
""")
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        assert agent.state_path == str(self.state_file)
        assert agent.kb_path == str(self.kb_path)
        assert hasattr(agent, 'tools')
        assert len(agent.tools) > 0
    
    def test_agent_initialization_without_kb(self):
        """Test agent initialization without knowledge base."""
        agent = Agent(str(self.state_file))
        
        assert agent.state_path == str(self.state_file)
        assert agent.retriever is None  # No KB provided
    
    def test_fallback_intent_classification(self):
        """Test fallback intent classification when LangChain is not available."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Test various intents
        assert agent._fallback_classify_intent("load my data") == "load_data"
        assert agent._fallback_classify_intent("compute quality control") == "compute_qc"
        assert agent._fallback_classify_intent("plot QC metrics") == "plot_qc"
        assert agent._fallback_classify_intent("filter low quality cells") == "apply_filters"
        assert agent._fallback_classify_intent("run scAR denoising") == "run_scar"
        assert agent._fallback_classify_intent("apply scVI integration") == "run_scvi"
        assert agent._fallback_classify_intent("detect doublets") == "detect_doublets"
        assert agent._fallback_classify_intent("create UMAP embedding") == "graph_analysis"
        assert agent._fallback_classify_intent("what can you do?") == "other"
    
    def test_fallback_plan_generation(self):
        """Test fallback plan generation."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Test QC intent
        plan = agent._fallback_generate_plan("compute QC", "compute_qc")
        assert len(plan) == 1
        assert plan[0]["tool"] == "compute_qc_metrics"
        assert plan[0]["params"]["species"] == "human"
        
        # Test filtering intent
        plan = agent._fallback_generate_plan("apply filters", "apply_filters")
        assert len(plan) == 1
        assert plan[0]["tool"] == "apply_qc_filters"
        
        # Test graph analysis intent
        plan = agent._fallback_generate_plan("create UMAP", "graph_analysis")
        assert len(plan) == 1
        assert plan[0]["tool"] == "quick_graph"
    
    def test_execute_plan_success(self):
        """Test successful plan execution."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Mock tool that returns success
        def mock_tool(params):
            return ToolResult(
                message="✅ Test tool executed successfully",
                state_delta={"test": True},
                artifacts=["test_artifact.txt"],
                citations=["test citation"]
            )
        
        agent.tools["test_tool"] = mock_tool
        
        plan = [{
            "tool": "test_tool",
            "description": "Test tool execution",
            "params": {"test_param": "value"}
        }]
        
        results = agent._execute_plan(plan)
        
        assert len(results) == 1
        assert results[0].message == "✅ Test tool executed successfully"
        assert results[0].state_delta == {"test": True}
        assert results[0].artifacts == ["test_artifact.txt"]
        assert results[0].citations == ["test citation"]
    
    def test_execute_plan_error(self):
        """Test plan execution with errors."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Mock tool that raises an error
        def mock_tool(params):
            raise ValueError("Test error")
        
        agent.tools["error_tool"] = mock_tool
        
        plan = [{
            "tool": "error_tool",
            "description": "Tool that errors",
            "params": {}
        }]
        
        results = agent._execute_plan(plan)
        
        assert len(results) == 1
        assert results[0].message.startswith("❌ Error executing error_tool:")
        assert "Test error" in results[0].message
    
    def test_execute_plan_unknown_tool(self):
        """Test plan execution with unknown tool."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        plan = [{
            "tool": "unknown_tool",
            "description": "Non-existent tool",
            "params": {}
        }]
        
        results = agent._execute_plan(plan)
        
        assert len(results) == 1
        assert results[0].message == "❌ Unknown tool: unknown_tool"
    
    def test_validate_execution(self):
        """Test execution validation."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        tool_results = [
            ToolResult(message="✅ Success 1", state_delta={}, artifacts=["file1.txt"], citations=[]),
            ToolResult(message="❌ Error 1", state_delta={}, artifacts=[], citations=[]),
            ToolResult(message="✅ Success 2", state_delta={}, artifacts=["file2.txt"], citations=[])
        ]
        
        validation = agent._validate_execution(tool_results)
        
        assert validation["success_count"] == 2
        assert validation["error_count"] == 1
        assert validation["total_artifacts"] == 2
        assert validation["has_errors"] is True
    
    def test_fallback_summarize(self):
        """Test fallback result summarization."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        plan = [{"tool": "test_tool", "description": "Test"}]
        tool_results = [
            ToolResult(message="✅ Success", state_delta={}, artifacts=["test.txt"], citations=[])
        ]
        
        summary = agent._fallback_summarize("test message", plan, tool_results)
        
        assert "1/1 steps successfully" in summary
        assert "1 artifacts" in summary
    
    def test_chat_with_mocked_chains(self):
        """Test full chat workflow with mocked LangChain components."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Mock the chains to return None (fallback mode)
        agent.intent_chain = None
        agent.plan_chain = None
        agent.summarize_chain = None
        
        # Mock a simple tool
        def mock_qc_tool(params):
            return ToolResult(
                message="✅ QC metrics computed",
                state_delta={"qc_computed": True},
                artifacts=["qc_plots.png"],
                citations=["QC guidelines"]
            )
        
        agent.tools["compute_qc_metrics"] = mock_qc_tool
        
        # Test chat
        result = agent.chat("compute QC metrics")
        
        assert result["status"] == "completed"
        assert result["intent"] == "compute_qc"
        assert len(result["plan"]) == 1
        assert result["plan"][0]["tool"] == "compute_qc_metrics"
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["message"] == "✅ QC metrics computed"
        assert "qc_plots.png" in result["artifacts"]
        assert "QC guidelines" in result["citations"]
    
    def test_chat_run_directory_creation(self):
        """Test that chat creates run directories."""
        # Create initial state
        state = SessionState()
        state.save(str(self.state_file))
        
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Mock tool
        agent.tools["test_tool"] = lambda params: ToolResult(
            message="✅ Test", state_delta={}, artifacts=[], citations=[]
        )
        
        result = agent.chat("test message")
        
        # Check that chat run directory was created
        chat_run_dir = Path(result["chat_run_dir"])
        assert chat_run_dir.exists()
        assert "chat_" in chat_run_dir.name
        
        # Check that artifacts were saved
        assert (chat_run_dir / "messages.json").exists()
        assert (chat_run_dir / "plan.json").exists()
    
    def test_chat_error_handling(self):
        """Test chat error handling."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Mock tool that raises an error
        def error_tool(params):
            raise RuntimeError("Critical error")
        
        agent.tools["test_tool"] = error_tool
        
        # Force fallback to use the error tool
        agent._fallback_generate_plan = lambda msg, intent: [{
            "tool": "test_tool", "description": "Error tool", "params": {}
        }]
        
        result = agent.chat("test error")
        
        # Should handle the error gracefully and include error in tool results
        assert result["status"] == "completed"  # Chat completes even with tool errors
        assert len(result["tool_results"]) == 1
        assert "❌" in result["tool_results"][0]["message"]
    
    def test_legacy_handle_message_interface(self):
        """Test backward compatibility with handle_message interface."""
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        # Mock tool
        agent.tools["compute_qc_metrics"] = lambda params: ToolResult(
            message="✅ QC computed", state_delta={}, artifacts=[], citations=[]
        )
        
        result = agent.handle_message("compute QC metrics")
        
        # Should return legacy format
        assert "message" in result
        assert "plan" in result
        assert "tool_results" in result
        assert "status" in result
        assert isinstance(result["plan"], list)
        assert all(isinstance(step, str) for step in result["plan"])
    
    def test_state_summary_generation(self):
        """Test state summary generation."""
        # Create state with some data
        state = SessionState()
        state.update_metadata({"adata_path": "/path/to/data.h5ad"})
        state.checkpoint("/path/to/data.h5ad", "initial_load")
        state.save(str(self.state_file))
        
        agent = Agent(str(self.state_file), str(self.kb_path))
        
        summary = agent._get_state_summary()
        
        assert state.run_id in summary
        assert "Steps completed: 1" in summary
        assert "Data loaded: /path/to/data.h5ad" in summary


@pytest.fixture
def mock_retriever():
    """Mock retriever for testing."""
    mock = Mock()
    mock.retrieve.return_value = [
        Mock(page_content="Test QC content", metadata={"source": "test.md"})
    ]
    return mock


def test_chat_with_retriever_integration(mock_retriever):
    """Test chat with retriever integration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        state_file = Path(temp_dir) / "test_state.json"
        
        agent = Agent(str(state_file))
        agent.retriever = mock_retriever
        
        # Mock tool
        agent.tools["compute_qc_metrics"] = lambda params: ToolResult(
            message="✅ QC computed", state_delta={}, artifacts=[], citations=[]
        )
        
        result = agent.chat("compute QC metrics")
        
        # Verify retriever was called
        mock_retriever.retrieve.assert_called_once()
        assert result["status"] == "completed"


if __name__ == "__main__":
    pytest.main([__file__])
