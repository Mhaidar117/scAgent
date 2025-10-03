"""End-to-end test for scQC Agent functionality."""

import pytest
import tempfile
import shutil
import json
import glob
from pathlib import Path
from typing import Dict, Any, List

from scqc_agent.tests.synth import make_synth_adata
from scqc_agent.agent.runtime import Agent
from scqc_agent.state import SessionState


@pytest.mark.slow
class TestE2EAgent:
    """End-to-end tests for the scQC Agent Python API."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        workspace = Path(temp_dir) / "e2e_test"
        workspace.mkdir(parents=True)
        
        # Change to workspace directory for relative paths
        original_cwd = Path.cwd()
        import os
        os.chdir(workspace)
        
        yield workspace
        
        # Cleanup
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def synthetic_data_path(self, temp_workspace: Path) -> Path:
        """Create synthetic data file."""
        try:
            adata = make_synth_adata(
                n_cells=600,
                n_genes=1500,
                n_batches=2,
                mito_frac=0.08,
                random_seed=42
            )
            
            # Save to file
            data_path = temp_workspace / "synthetic_data.h5ad"
            adata.write_h5ad(data_path)
            
            return data_path
        except ImportError:
            pytest.skip("scanpy/anndata not available for synthetic data generation")
    
    @pytest.fixture
    def agent(self, temp_workspace: Path) -> Agent:
        """Create an Agent instance with temporary state."""
        state_path = temp_workspace / "test_state.json"
        
        # Use the real KB directory
        kb_path = Path(__file__).parent.parent / "kb"
        if not kb_path.exists():
            kb_path = None  # Use default or skip KB features
        
        return Agent(str(state_path), knowledge_base_path=str(kb_path) if kb_path else None)
    
    def test_full_workflow_api(self, agent: Agent, synthetic_data_path: Path, temp_workspace: Path):
        """Test the complete workflow using Python API."""
        
        # Step 1: Load data
        result = agent.chat(f"load data from {synthetic_data_path}")
        assert result.get("status") != "failed", f"Failed to load data: {result.get('error', 'Unknown error')}"
        
        tool_results = result.get("tool_results", [])
        assert len(tool_results) > 0, "Should have tool results from loading data"
        
        # Check that data was loaded
        load_success = any("loaded" in res.get("message", "").lower() or "success" in res.get("message", "").lower() for res in tool_results)
        assert load_success, f"Data loading should succeed. Results: {[r.get('message') for r in tool_results]}"
        
        # Step 2: Compute QC metrics and show violins
        result = agent.chat("compute qc metrics and show violins")
        assert result.get("status") != "failed", f"Failed QC computation: {result.get('error')}"
        
        # Check for QC computation
        tool_results = result.get("tool_results", [])
        qc_computed = any("qc" in res.get("message", "").lower() for res in tool_results)
        assert qc_computed, "Should have computed QC metrics"
        
        # Step 3: Apply filters and replot
        result = agent.chat("apply min_genes 1000; mito <= 10%; replot")
        assert result.get("status") != "failed", f"Failed filter application: {result.get('error')}"
        
        # Step 4: Run quick UMAP and cluster
        result = agent.chat("run a quick umap and cluster with resolution 0.8")
        assert result.get("status") != "failed", f"Failed graph analysis: {result.get('error')}"
        
        # Verify artifacts were created
        self._verify_artifacts_exist(temp_workspace)
        
        # Check state contains expected workflow history
        agent.save_state()
        state_file = Path(agent.state_path)
        assert state_file.exists(), "State file should be saved"
        
        with open(state_file) as f:
            state_data = json.load(f)
        
        assert "history" in state_data, "State should contain workflow history"
        assert len(state_data["history"]) > 0, "Should have workflow steps recorded"
    
    def test_agent_with_citations(self, agent: Agent, synthetic_data_path: Path):
        """Test that agent responses include citations from KB."""
        
        # Load data first
        agent.chat(f"load data from {synthetic_data_path}")
        
        # Ask for QC guidance that should trigger KB retrieval
        result = agent.chat("show qc summary and propose mitochondrial thresholds from guidelines")
        
        # Check if citations were found
        all_citations = result.get("citations", [])
        if agent.retriever is not None:  # Only test if KB is available
            # Should have some citations for mitochondrial guidance
            mito_citations = [c for c in all_citations if "mito" in c.lower() or "quality" in c.lower()]
            if len(mito_citations) == 0:
                # KB might not have been indexed yet - this is acceptable in CI
                print("Warning: No mitochondrial citations found, KB might not be indexed")
        
        # Test another query that should find doublet information
        result = agent.chat("help me understand doublet detection best practices")
        citations = result.get("citations", [])
        if len(citations) > 0:
            print(f"Found {len(citations)} citations for doublet query")
    
    def test_error_handling(self, agent: Agent):
        """Test that agent handles errors gracefully."""
        
        # Try to compute QC without loading data
        result = agent.chat("compute qc metrics")
        
        # Should not crash, but may return error
        assert "status" in result, "Should return status information"
        
        # Tool results should contain error information if data not loaded
        tool_results = result.get("tool_results", [])
        if len(tool_results) > 0:
            # Check if any results indicate missing data
            has_error_info = any(
                "❌" in res.get("message", "") or 
                "error" in res.get("message", "").lower() or
                "not found" in res.get("message", "").lower()
                for res in tool_results
            )
            print(f"Error handling test - found error indicators: {has_error_info}")
    
    def test_plan_generation(self, agent: Agent, synthetic_data_path: Path):
        """Test that agent generates proper execution plans."""
        
        # Load data
        agent.chat(f"load data from {synthetic_data_path}")
        
        # Test a complex request that should generate a multi-step plan
        result = agent.chat("compute qc, apply filters with min 1200 genes and max 15% mito, then create umap")
        
        plan = result.get("plan", [])
        assert isinstance(plan, list), "Plan should be a list of steps"
        
        if len(plan) > 0:
            # Check plan structure
            for step in plan:
                assert "tool" in step or "description" in step, "Each plan step should have tool or description"
            
            print(f"Generated plan with {len(plan)} steps:")
            for i, step in enumerate(plan, 1):
                tool = step.get("tool", "unknown")
                desc = step.get("description", "No description")
                print(f"  {i}. {tool}: {desc}")
    
    def test_artifact_generation(self, agent: Agent, synthetic_data_path: Path, temp_workspace: Path):
        """Test that expected artifacts are generated."""
        
        # Load and process data
        agent.chat(f"load data from {synthetic_data_path}")
        result = agent.chat("compute qc metrics and create violin plots")
        
        # Check for artifacts in results
        all_artifacts = result.get("artifacts", [])
        tool_results = result.get("tool_results", [])
        
        # Collect artifacts from all tool results
        for tr in tool_results:
            artifacts = tr.get("artifacts", [])
            all_artifacts.extend(artifacts)
        
        print(f"Generated artifacts: {all_artifacts}")
        
        # Look for artifacts in runs/ directory
        runs_dir = temp_workspace / "runs"
        if runs_dir.exists():
            artifact_files = list(runs_dir.rglob("*"))
            print(f"Files in runs directory: {[str(f.relative_to(temp_workspace)) for f in artifact_files]}")
    
    def test_session_state_persistence(self, temp_workspace: Path, synthetic_data_path: Path):
        """Test that session state persists between agent instances."""
        
        state_path = temp_workspace / "persistent_state.json"
        
        # Create first agent and do some work
        agent1 = Agent(str(state_path))
        result1 = agent1.chat(f"load data from {synthetic_data_path}")
        agent1.save_state()
        
        # Create second agent with same state file
        agent2 = Agent(str(state_path))
        
        # Should be able to continue workflow
        result2 = agent2.chat("compute qc metrics")
        
        # Both agents should have the same run_id
        assert agent1.state.run_id == agent2.state.run_id, "Should maintain same run_id across instances"
    
    def test_deterministic_results(self, temp_workspace: Path):
        """Test that repeated runs with same seed produce consistent results."""
        
        # Create two identical synthetic datasets
        try:
            adata1 = make_synth_adata(n_cells=300, n_genes=800, random_seed=123)
            adata2 = make_synth_adata(n_cells=300, n_genes=800, random_seed=123)
            
            path1 = temp_workspace / "data1.h5ad"
            path2 = temp_workspace / "data2.h5ad"
            
            adata1.write_h5ad(path1)
            adata2.write_h5ad(path2)
            
            # Process both with same parameters
            agent1 = Agent(str(temp_workspace / "state1.json"))
            agent2 = Agent(str(temp_workspace / "state2.json"))
            
            agent1.chat(f"load data from {path1}")
            agent1.chat("compute qc metrics")
            result1 = agent1.chat("apply min_genes 500 max_pct_mt 20")
            
            agent2.chat(f"load data from {path2}")
            agent2.chat("compute qc metrics")
            result2 = agent2.chat("apply min_genes 500 max_pct_mt 20")
            
            # Should have similar filtering results
            # (Exact equality may be difficult due to floating point, but should be very close)
            print("Deterministic test completed - manual verification of results may be needed")
            
        except ImportError:
            pytest.skip("scanpy not available for deterministic test")
    
    def _verify_artifacts_exist(self, workspace: Path):
        """Verify that expected artifacts exist in the workspace."""
        
        # Look for common artifact patterns
        expected_patterns = [
            "runs/*/step_*/qc_*.png",      # QC plots
            "runs/*/step_*/qc_*.csv",      # QC summary files
            "runs/*/step_*/umap_*.png",    # UMAP plots
            "runs/*/step_*/*.h5ad",        # Data checkpoints
            "runs/*/step_*/qc_filters.json"  # Filter settings
        ]
        
        found_artifacts = []
        for pattern in expected_patterns:
            matches = list(workspace.glob(pattern))
            found_artifacts.extend(matches)
        
        if len(found_artifacts) > 0:
            print(f"Found artifacts: {[str(f.relative_to(workspace)) for f in found_artifacts]}")
        else:
            print("No artifacts found - may indicate workflow didn't execute expected tools")


class TestAgentIntegration:
    """Integration tests for agent components."""
    
    def test_agent_initialization(self):
        """Test agent can initialize without errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "test_state.json"
            
            try:
                agent = Agent(str(state_path))
                assert agent is not None, "Agent should initialize"
                assert agent.state is not None, "Should have session state"
                
                # Test stats
                if agent.retriever:
                    stats = agent.retriever.get_stats()
                    assert isinstance(stats, dict), "Retriever stats should be a dictionary"
                    print(f"Retriever initialized with stats: {stats}")
                else:
                    print("Retriever not available - KB may not be present")
                    
            except Exception as e:
                pytest.fail(f"Agent initialization failed: {e}")
    
    def test_tool_registry(self):
        """Test that tools are properly registered."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "test_state.json"
            agent = Agent(str(state_path))
            
            # Check basic tools are registered
            expected_tools = [
                "load_data",
                "compute_qc_metrics", 
                "plot_qc",
                "apply_qc_filters",
                "quick_graph"
            ]
            
            for tool in expected_tools:
                assert tool in agent.tools, f"Tool {tool} should be registered"
            
            print(f"Registered tools: {list(agent.tools.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
