"""Tests for evaluation runner."""

import json
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from scqc_agent.eval.runner import (
    EvalRunner,
    EvalResult,
    EvalSummary,
    run_evaluation,
)
from scqc_agent.state import SessionState

# Skip tests that require scanpy if not available
pytest.importorskip("scanpy")
pytest.importorskip("anndata")

import scanpy as sc
import anndata as ad
import numpy as np


@pytest.fixture
def simple_prompts_config():
    """Create a simple prompts configuration for testing."""
    return {
        'prompts': [
            {
                'id': 'test_basic',
                'description': 'Basic test prompt',
                'prompt': 'Load test data and compute QC metrics',
                'acceptance_criteria': {
                    'adata_loaded': True,
                    'qc_metrics_computed': True,
                },
                'tags': ['basic', 'qc']
            },
            {
                'id': 'test_optional',
                'description': 'Optional test prompt',
                'prompt': 'Run advanced analysis',
                'acceptance_criteria': {
                    'advanced_analysis': True,
                },
                'tags': ['advanced'],
                'optional': True
            },
            {
                'id': 'test_failing',
                'description': 'Test that should fail',
                'prompt': 'Do something impossible',
                'acceptance_criteria': {
                    'impossible_thing': True,
                },
                'tags': ['test']
            }
        ],
        'test_config': {
            'default_dataset': {
                'n_cells': 100,
                'n_genes': 50,
                'species': 'mouse',
                'has_batch': False
            },
            'small_dataset': {
                'n_cells': 50,
                'n_genes': 25,
                'species': 'mouse',
                'has_batch': False
            },
            'timeouts': {
                'basic_tests': 30,
                'scvi_tests': 300
            },
            'pass_thresholds': {
                'overall_pass_rate': 0.95,
                'core_functionality_pass_rate': 1.0,
                'optional_pass_rate': 0.8
            }
        }
    }


@pytest.fixture
def temp_prompts_file(simple_prompts_config, tmp_path):
    """Create a temporary prompts file."""
    prompts_file = tmp_path / "test_prompts.yaml"
    with open(prompts_file, 'w') as f:
        yaml.dump(simple_prompts_config, f)
    return prompts_file


class TestEvalResult:
    """Test EvalResult dataclass."""
    
    def test_eval_result_creation(self):
        """Test creating EvalResult."""
        result = EvalResult(
            prompt_id="test_id",
            description="Test description",
            prompt="Test prompt",
            passed=True,
            execution_time=1.5
        )
        
        assert result.prompt_id == "test_id"
        assert result.description == "Test description"
        assert result.prompt == "Test prompt"
        assert result.passed is True
        assert result.execution_time == 1.5
        assert result.error_message is None
        assert result.artifacts_found == []
    
    def test_eval_result_to_dict(self):
        """Test converting EvalResult to dictionary."""
        result = EvalResult(
            prompt_id="test_id",
            description="Test description",
            prompt="Test prompt",
            passed=False,
            execution_time=2.0,
            error_message="Test error",
            artifacts_found=["file1.png", "file2.csv"],
            artifacts_missing=["file3.json"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['prompt_id'] == "test_id"
        assert result_dict['passed'] is False
        assert result_dict['error_message'] == "Test error"
        assert result_dict['artifacts_found'] == ["file1.png", "file2.csv"]
        assert result_dict['artifacts_missing'] == ["file3.json"]


class TestEvalSummary:
    """Test EvalSummary dataclass."""
    
    def test_eval_summary_creation(self):
        """Test creating EvalSummary."""
        summary = EvalSummary(
            total_prompts=10,
            passed_prompts=8,
            failed_prompts=1,
            skipped_prompts=1,
            pass_rate=0.8,
            core_pass_rate=1.0,
            optional_pass_rate=0.5,
            total_execution_time=60.0,
            timestamp="2024-01-01T00:00:00"
        )
        
        assert summary.total_prompts == 10
        assert summary.passed_prompts == 8
        assert summary.failed_prompts == 1
        assert summary.skipped_prompts == 1
        assert summary.pass_rate == 0.8
        assert summary.core_pass_rate == 1.0
        assert summary.optional_pass_rate == 0.5
    
    def test_eval_summary_to_dict(self):
        """Test converting EvalSummary to dictionary."""
        summary = EvalSummary(
            total_prompts=5,
            passed_prompts=4,
            failed_prompts=1,
            skipped_prompts=0,
            pass_rate=0.8,
            core_pass_rate=0.8,
            optional_pass_rate=1.0,
            total_execution_time=30.0,
            timestamp="2024-01-01T00:00:00"
        )
        
        summary_dict = summary.to_dict()
        
        assert summary_dict['total_prompts'] == 5
        assert summary_dict['pass_rate'] == 0.8
        assert summary_dict['results'] == []


class TestEvalRunner:
    """Test EvalRunner class."""
    
    def test_eval_runner_init(self):
        """Test EvalRunner initialization."""
        runner = EvalRunner(verbose=False)
        
        assert runner.agent is None
        assert runner.verbose is False
        assert runner.test_data_cache == {}
    
    def test_create_test_data(self):
        """Test creating synthetic test data."""
        runner = EvalRunner(verbose=False)
        
        config = {
            'n_cells': 100,
            'n_genes': 50,
            'species': 'mouse',
            'has_batch': False
        }
        
        data_path = runner.create_test_data(config)
        
        assert data_path.exists()
        assert data_path.suffix == '.h5ad'
        
        # Load and verify data
        adata = sc.read_h5ad(data_path)
        assert adata.n_obs == 100
        assert adata.n_vars == 50
        assert 'sample_id' in adata.obs.columns
        
        # Clean up
        data_path.unlink()
    
    def test_create_test_data_with_batch(self):
        """Test creating test data with batch information."""
        runner = EvalRunner(verbose=False)
        
        config = {
            'n_cells': 100,
            'n_genes': 50,
            'species': 'human',
            'has_batch': True,
            'n_batches': 3
        }
        
        data_path = runner.create_test_data(config)
        
        assert data_path.exists()
        
        # Load and verify batch data
        adata = sc.read_h5ad(data_path)
        assert 'batch' in adata.obs.columns
        assert adata.obs['batch'].nunique() <= 3
        
        # Check gene names for human
        assert any(gene.startswith('MT-') for gene in adata.var_names)
        
        # Clean up
        data_path.unlink()
    
    def test_create_test_data_caching(self):
        """Test that test data creation uses caching."""
        runner = EvalRunner(verbose=False)
        
        config = {
            'n_cells': 50,
            'n_genes': 25,
            'species': 'mouse',
            'has_batch': False
        }
        
        # Create data twice
        path1 = runner.create_test_data(config)
        path2 = runner.create_test_data(config)
        
        # Should return same path due to caching
        assert path1 == path2
        assert len(runner.test_data_cache) == 1
        
        # Clean up
        path1.unlink()
    
    def test_setup_test_session(self):
        """Test setting up test session."""
        runner = EvalRunner(verbose=False)
        
        test_config = {
            'n_cells': 50,
            'n_genes': 25,
            'species': 'mouse',
            'has_batch': False
        }
        
        state, data_path = runner.setup_test_session(test_config)
        
        assert isinstance(state, SessionState)
        assert 'adata_path' in state.metadata
        assert state.config['species'] == 'mouse'
        assert data_path.exists()
        
        # Clean up
        data_path.unlink()
    
    def test_setup_test_session_with_batch(self):
        """Test setting up test session with batch."""
        runner = EvalRunner(verbose=False)
        
        test_config = {
            'n_cells': 50,
            'n_genes': 25,
            'species': 'human',
            'has_batch': True,
            'batch': True
        }
        
        state, data_path = runner.setup_test_session(test_config)
        
        assert 'batch_key' in state.config
        assert state.config['batch_key'] == 'batch'
        assert state.config['species'] == 'human'
        
        # Clean up
        data_path.unlink()
    
    def test_check_acceptance_criteria_adata_loaded(self):
        """Test checking adata_loaded criterion."""
        runner = EvalRunner(verbose=False)
        
        # Create test data
        data_path = runner.create_test_data({'n_cells': 50, 'n_genes': 25})
        state = SessionState()
        
        criteria = {'adata_loaded': True}
        passed, details = runner.check_acceptance_criteria(
            criteria, state, {}, data_path
        )
        
        assert passed is True
        assert details['adata_loaded']['actual'] is True
        
        # Clean up
        data_path.unlink()
    
    def test_check_acceptance_criteria_qc_metrics(self):
        """Test checking QC metrics criterion."""
        runner = EvalRunner(verbose=False)
        
        # Create test data with QC metrics
        data_path = runner.create_test_data({'n_cells': 50, 'n_genes': 25})
        adata = sc.read_h5ad(data_path)
        
        # Add QC metrics
        adata.obs['n_genes_by_counts'] = np.random.randint(10, 25, 50)
        adata.obs['total_counts'] = np.random.randint(100, 1000, 50)
        adata.obs['pct_counts_mt'] = np.random.uniform(5, 15, 50)
        
        adata.write_h5ad(data_path)
        
        state = SessionState()
        criteria = {'qc_metrics_computed': True}
        passed, details = runner.check_acceptance_criteria(
            criteria, state, {}, data_path
        )
        
        assert passed is True
        assert details['qc_metrics_computed']['actual'] is True
        
        # Clean up
        data_path.unlink()
    
    def test_check_acceptance_criteria_missing_qc(self):
        """Test checking QC metrics when missing."""
        runner = EvalRunner(verbose=False)
        
        # Create test data without QC metrics
        data_path = runner.create_test_data({'n_cells': 50, 'n_genes': 25})
        
        state = SessionState()
        criteria = {'qc_metrics_computed': True}
        passed, details = runner.check_acceptance_criteria(
            criteria, state, {}, data_path
        )
        
        assert passed is False
        assert details['qc_metrics_computed']['actual'] is False
        
        # Clean up
        data_path.unlink()
    
    @patch('scqc_agent.eval.runner.Agent')
    def test_run_single_prompt_success(self, mock_agent_class):
        """Test running a single prompt successfully."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.chat.return_value = {'status': 'success', 'message': 'Test completed'}
        mock_agent_class.return_value = mock_agent
        
        runner = EvalRunner(verbose=False)
        
        prompt_config = {
            'id': 'test_prompt',
            'description': 'Test prompt',
            'prompt': 'Test command',
            'acceptance_criteria': {'adata_loaded': True}
        }
        
        test_config = {'n_cells': 50, 'n_genes': 25}
        
        result = runner.run_single_prompt(prompt_config, test_config, timeout=60)
        
        assert result.prompt_id == 'test_prompt'
        assert result.passed is True
        assert result.execution_time > 0
        assert result.error_message is None
    
    @patch('scqc_agent.eval.runner.Agent')
    def test_run_single_prompt_failure(self, mock_agent_class):
        """Test running a single prompt that fails."""
        # Mock agent to raise exception
        mock_agent = Mock()
        mock_agent.chat.side_effect = Exception("Test error")
        mock_agent_class.return_value = mock_agent
        
        runner = EvalRunner(verbose=False)
        
        prompt_config = {
            'id': 'test_prompt',
            'description': 'Test prompt',
            'prompt': 'Test command',
            'acceptance_criteria': {'adata_loaded': True}
        }
        
        test_config = {'n_cells': 50, 'n_genes': 25}
        
        result = runner.run_single_prompt(prompt_config, test_config, timeout=60)
        
        assert result.prompt_id == 'test_prompt'
        assert result.passed is False
        assert "Test error" in result.error_message
    
    @patch('scqc_agent.eval.runner.Agent')
    def test_run_single_prompt_optional_skip(self, mock_agent_class):
        """Test skipping optional prompt on import error."""
        # Mock agent to raise import error
        mock_agent = Mock()
        mock_agent.chat.side_effect = ImportError("scvi not available")
        mock_agent_class.return_value = mock_agent
        
        runner = EvalRunner(verbose=False)
        
        prompt_config = {
            'id': 'test_optional',
            'description': 'Optional test',
            'prompt': 'Run scVI',
            'acceptance_criteria': {'scvi_trained': True},
            'optional': True
        }
        
        test_config = {'n_cells': 50, 'n_genes': 25}
        
        result = runner.run_single_prompt(prompt_config, test_config, timeout=60)
        
        assert result.prompt_id == 'test_optional'
        assert result.passed is None  # Indicates skipped
        assert "Skipped (optional)" in result.error_message


class TestRunEvaluation:
    """Test run_evaluation convenience function."""
    
    def test_run_evaluation_basic(self, temp_prompts_file):
        """Test basic evaluation run."""
        with patch('scqc_agent.eval.runner.Agent') as mock_agent_class:
            # Mock successful agent
            mock_agent = Mock()
            mock_agent.chat.return_value = {'status': 'success'}
            mock_agent_class.return_value = mock_agent
            
            # Mock SessionState to avoid file operations
            with patch('scqc_agent.eval.runner.SessionState') as mock_state_class:
                mock_state = Mock()
                mock_state.history = []
                mock_state.artifacts = {}
                mock_state.config = {'species': 'mouse'}
                mock_state_class.return_value = mock_state
                
                summary = run_evaluation(
                    prompts_file=str(temp_prompts_file),
                    include_optional=False,
                    verbose=False
                )
                
                assert isinstance(summary, EvalSummary)
                assert summary.total_prompts >= 1  # At least one non-optional prompt
    
    def test_run_evaluation_with_tags(self, temp_prompts_file):
        """Test evaluation with tag filtering."""
        with patch('scqc_agent.eval.runner.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.chat.return_value = {'status': 'success'}
            mock_agent_class.return_value = mock_agent
            
            with patch('scqc_agent.eval.runner.SessionState') as mock_state_class:
                mock_state = Mock()
                mock_state.history = []
                mock_state.artifacts = {}
                mock_state.config = {'species': 'mouse'}
                mock_state_class.return_value = mock_state
                
                summary = run_evaluation(
                    prompts_file=str(temp_prompts_file),
                    tags_filter=['basic'],
                    verbose=False
                )
                
                # Should only run prompts with 'basic' tag
                assert summary.total_prompts == 1  # Only test_basic has 'basic' tag
    
    def test_run_evaluation_save_output(self, temp_prompts_file, tmp_path):
        """Test saving evaluation output."""
        output_file = tmp_path / "eval_results.json"
        
        with patch('scqc_agent.eval.runner.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent.chat.return_value = {'status': 'success'}
            mock_agent_class.return_value = mock_agent
            
            with patch('scqc_agent.eval.runner.SessionState') as mock_state_class:
                mock_state = Mock()
                mock_state.history = []
                mock_state.artifacts = {}
                mock_state.config = {'species': 'mouse'}
                mock_state_class.return_value = mock_state
                
                summary = run_evaluation(
                    prompts_file=str(temp_prompts_file),
                    output_file=str(output_file),
                    include_optional=False,
                    verbose=False
                )
                
                # Check that output file was created
                assert output_file.exists()
                
                # Verify content
                with open(output_file) as f:
                    saved_data = json.load(f)
                
                assert saved_data['total_prompts'] == summary.total_prompts
                assert saved_data['pass_rate'] == summary.pass_rate


if __name__ == "__main__":
    pytest.main([__file__])
