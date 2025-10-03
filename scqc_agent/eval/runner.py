"""Evaluation runner for scQC Agent golden prompts.

This module provides functionality to run evaluation tests against a set of golden
prompts and generate comprehensive pass/fail reports with detailed metrics.
"""

import json
import time
import yaml
import traceback
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

from ..state import SessionState, ToolResult
from ..agent.runtime import Agent


@dataclass
class EvalResult:
    """Result from running a single evaluation prompt."""
    
    prompt_id: str
    description: str
    prompt: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    artifacts_found: List[str] = field(default_factory=list)
    artifacts_missing: List[str] = field(default_factory=list)
    acceptance_details: Dict[str, Any] = field(default_factory=dict)
    agent_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass 
class EvalSummary:
    """Summary of evaluation run results."""
    
    total_prompts: int
    passed_prompts: int
    failed_prompts: int
    skipped_prompts: int
    pass_rate: float
    core_pass_rate: float
    optional_pass_rate: float
    total_execution_time: float
    timestamp: str
    results: List[EvalResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EvalRunner:
    """Runner for executing evaluation prompts and checking acceptance criteria."""
    
    def __init__(self, agent: Optional[Agent] = None, verbose: bool = True):
        """Initialize the evaluation runner.
        
        Args:
            agent: Agent instance to use for evaluation (creates new one if None)
            verbose: Whether to print progress messages
        """
        self.agent = agent
        self.verbose = verbose
        self.test_data_cache: Dict[str, Any] = {}
        
    def create_test_data(self, config: Dict[str, Any]) -> Path:
        """Create synthetic test data for evaluation.
        
        Args:
            config: Test dataset configuration
            
        Returns:
            Path to the created test data file
        """
        if not SCANPY_AVAILABLE:
            raise ImportError("Scanpy is required for test data generation")
        
        # Generate synthetic data
        n_cells = config.get('n_cells', 1000)
        n_genes = config.get('n_genes', 500)
        species = config.get('species', 'mouse')
        has_batch = config.get('has_batch', False)
        n_batches = config.get('n_batches', 2)
        
        # Create cache key
        cache_key = f"{n_cells}_{n_genes}_{species}_{has_batch}_{n_batches}"
        
        if cache_key in self.test_data_cache:
            return self.test_data_cache[cache_key]
        
        if self.verbose:
            print(f"Generating test data: {n_cells} cells x {n_genes} genes ({species})")
        
        # Generate count matrix with realistic structure
        np.random.seed(42)  # Fixed seed for reproducibility
        
        # Log-normal distribution for realistic expression
        base_expression = np.random.lognormal(mean=0.5, sigma=1.5, size=(n_cells, n_genes))
        
        # Add some highly expressed genes (ensure we don't exceed n_genes)
        n_high_expr = min(50, n_genes)
        base_expression[:, :n_high_expr] *= np.random.uniform(5, 15, size=(n_cells, n_high_expr))
        
        # Add some lowly expressed genes (zeros)
        zero_mask = np.random.random((n_cells, n_genes)) < 0.7  # 70% sparsity
        base_expression[zero_mask] = 0
        
        # Convert to integer counts
        counts = np.random.poisson(base_expression).astype(np.float32)
        
        # Create gene names based on species
        if species == 'human':
            gene_names = [f"ENSG{i:08d}" for i in range(n_genes)]
            # Add some mitochondrial genes
            mt_genes = [f"MT-{gene}" for gene in ['ATP6', 'ATP8', 'COX1', 'COX2', 'COX3', 'CYTB']]
            n_mt = min(len(mt_genes), n_genes)
            gene_names[:n_mt] = mt_genes[:n_mt]
        else:  # mouse
            gene_names = [f"ENSMUSG{i:08d}" for i in range(n_genes)]
            # Add some mitochondrial genes
            mt_genes = [f"mt-{gene}" for gene in ['Atp6', 'Atp8', 'Cox1', 'Cox2', 'Cox3', 'Cytb']]
            n_mt = min(len(mt_genes), n_genes)
            gene_names[:n_mt] = mt_genes[:n_mt]
        
        # Create cell barcodes
        cell_names = [f"CELL_{i:06d}" for i in range(n_cells)]
        
        # Create AnnData object
        adata = ad.AnnData(X=counts)
        adata.var_names = gene_names
        adata.obs_names = cell_names
        
        # Add batch information if requested
        if has_batch:
            batch_assignments = np.random.choice(
                [f"batch_{i}" for i in range(n_batches)], 
                size=n_cells
            )
            adata.obs['batch'] = batch_assignments
        
        # Add some realistic metadata
        adata.obs['sample_id'] = [f"sample_{i // 100}" for i in range(n_cells)]
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        adata.write_h5ad(temp_path)
        
        # Cache the path
        self.test_data_cache[cache_key] = temp_path
        
        return temp_path
        
    def setup_test_session(self, test_config: Dict[str, Any]) -> Tuple[SessionState, Path]:
        """Set up a test session with synthetic data.
        
        Args:
            test_config: Test configuration dictionary
            
        Returns:
            Tuple of (session_state, data_path)
        """
        # Create test data
        data_path = self.create_test_data(test_config)
        
        # Create session state
        state = SessionState()
        state.metadata['adata_path'] = str(data_path)
        
        # Add test-specific configuration
        if 'batch' in test_config and test_config.get('has_batch', False):
            state.config = {'batch_key': 'batch', 'species': test_config.get('species', 'mouse')}
        else:
            state.config = {'species': test_config.get('species', 'mouse')}
            
        return state, data_path
        
    def check_acceptance_criteria(
        self, 
        criteria: Dict[str, Any], 
        state: SessionState,
        agent_response: Dict[str, Any],
        data_path: Path
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if acceptance criteria are met.
        
        Args:
            criteria: Acceptance criteria dictionary
            state: Session state after execution
            agent_response: Response from agent
            data_path: Path to test data
            
        Returns:
            Tuple of (passed, details_dict)
        """
        details = {}
        passed = True
        
        try:
            # Load current data if available
            adata = None
            if SCANPY_AVAILABLE and data_path.exists():
                try:
                    adata = sc.read_h5ad(data_path)
                except Exception:
                    # Try to load from most recent checkpoint
                    if state.history:
                        last_checkpoint = state.history[-1].get('checkpoint_path')
                        if last_checkpoint and Path(last_checkpoint).exists():
                            adata = sc.read_h5ad(last_checkpoint)
                            
            # Check each criterion
            for criterion, expected in criteria.items():
                if criterion == 'adata_loaded':
                    actual = adata is not None
                    details[criterion] = {'expected': expected, 'actual': actual}
                    if actual != expected:
                        passed = False
                        
                elif criterion == 'qc_metrics_computed':
                    if adata is not None:
                        required_fields = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
                        actual = all(field in adata.obs.columns for field in required_fields)
                    else:
                        actual = False
                    details[criterion] = {'expected': expected, 'actual': actual}
                    if actual != expected:
                        passed = False
                        
                elif criterion == 'filters_applied':
                    # Check if filtering was performed by looking at history
                    actual = any('filter' in entry.get('label', '').lower() for entry in state.history)
                    details[criterion] = {'expected': expected, 'actual': actual}
                    if actual != expected:
                        passed = False
                        
                elif criterion in ['retained_fraction_min', 'retained_fraction_max']:
                    if len(state.history) >= 2:
                        # Compare cell counts before and after filtering
                        initial_count = 3000  # Default test data size
                        final_count = adata.n_obs if adata else initial_count
                        retained_fraction = final_count / initial_count
                        
                        if 'min' in criterion:
                            actual = retained_fraction >= expected
                        else:
                            actual = retained_fraction <= expected
                            
                        details[criterion] = {
                            'expected': expected, 
                            'actual': retained_fraction,
                            'passed': actual
                        }
                        if not actual:
                            passed = False
                    else:
                        details[criterion] = {'expected': expected, 'actual': 'no_filtering_detected'}
                        passed = False
                        
                elif criterion == 'species_detected':
                    # Check if species was correctly identified
                    actual = state.config.get('species', 'unknown')
                    details[criterion] = {'expected': expected, 'actual': actual}
                    if actual != expected:
                        passed = False
                        
                elif criterion == 'neighbors_computed':
                    if adata is not None:
                        actual = 'neighbors' in adata.uns and 'connectivities' in adata.obsp
                    else:
                        actual = False
                    details[criterion] = {'expected': expected, 'actual': actual}
                    if actual != expected:
                        passed = False
                        
                elif criterion == 'clustering_computed':
                    if adata is not None:
                        actual = any('leiden' in col for col in adata.obs.columns)
                    else:
                        actual = False
                    details[criterion] = {'expected': expected, 'actual': actual}
                    if actual != expected:
                        passed = False
                        
                elif criterion in ['n_clusters_min', 'n_clusters_max']:
                    if adata is not None:
                        leiden_cols = [col for col in adata.obs.columns if 'leiden' in col]
                        if leiden_cols:
                            n_clusters = adata.obs[leiden_cols[0]].nunique()
                            if 'min' in criterion:
                                actual = n_clusters >= expected
                            else:
                                actual = n_clusters <= expected
                            details[criterion] = {
                                'expected': expected, 
                                'actual': n_clusters,
                                'passed': actual
                            }
                        else:
                            actual = False
                            details[criterion] = {'expected': expected, 'actual': 'no_clustering'}
                    else:
                        actual = False
                        details[criterion] = {'expected': expected, 'actual': 'no_data'}
                    
                    if not actual:
                        passed = False
                        
                elif criterion == 'artifacts_generated':
                    # Check for expected artifacts in session
                    missing_artifacts = []
                    found_artifacts = []
                    
                    for expected_artifact in expected:
                        found = any(expected_artifact in path for path in state.artifacts.keys())
                        if found:
                            found_artifacts.append(expected_artifact)
                        else:
                            missing_artifacts.append(expected_artifact)
                    
                    details[criterion] = {
                        'expected': expected,
                        'found': found_artifacts,
                        'missing': missing_artifacts,
                        'passed': len(missing_artifacts) == 0
                    }
                    
                    if missing_artifacts:
                        passed = False
                        
                # Add more criteria checks as needed...
                        
        except Exception as e:
            details['error'] = str(e)
            passed = False
            
        return passed, details
        
    def run_single_prompt(
        self, 
        prompt_config: Dict[str, Any], 
        test_config: Dict[str, Any],
        timeout: int = 60
    ) -> EvalResult:
        """Run a single evaluation prompt.
        
        Args:
            prompt_config: Prompt configuration dictionary
            test_config: Test configuration dictionary  
            timeout: Timeout in seconds
            
        Returns:
            EvalResult with execution results
        """
        prompt_id = prompt_config['id']
        description = prompt_config['description']
        prompt = prompt_config['prompt']
        criteria = prompt_config.get('acceptance_criteria', {})
        is_optional = prompt_config.get('optional', False)
        
        if self.verbose:
            print(f"Running prompt: {prompt_id}")
            
        start_time = time.time()
        
        try:
            # Set up test session
            state, data_path = self.setup_test_session(test_config)
            
            # Create agent if needed
            if self.agent is None:
                # Create a minimal agent for testing
                self.agent = Agent(state_path="temp_eval_state.json")
                self.agent.state = state
            else:
                self.agent.state = state
            
            # Execute the prompt
            try:
                response = self.agent.chat(prompt)
                execution_time = time.time() - start_time
                
                # Check acceptance criteria
                passed, details = self.check_acceptance_criteria(
                    criteria, state, response, data_path
                )
                
                # Check for artifacts
                artifacts_found = list(state.artifacts.keys())
                expected_artifacts = criteria.get('artifacts_generated', [])
                artifacts_missing = [
                    art for art in expected_artifacts 
                    if not any(art in found for found in artifacts_found)
                ]
                
                return EvalResult(
                    prompt_id=prompt_id,
                    description=description,
                    prompt=prompt,
                    passed=passed,
                    execution_time=execution_time,
                    artifacts_found=artifacts_found,
                    artifacts_missing=artifacts_missing,
                    acceptance_details=details,
                    agent_response=response
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # For optional tests, treat import errors as skipped
                if is_optional and "import" in str(e).lower():
                    return EvalResult(
                        prompt_id=prompt_id,
                        description=description,
                        prompt=prompt,
                        passed=None,  # Indicates skipped
                        execution_time=execution_time,
                        error_message=f"Skipped (optional): {str(e)}"
                    )
                
                return EvalResult(
                    prompt_id=prompt_id,
                    description=description,
                    prompt=prompt,
                    passed=False,
                    execution_time=execution_time,
                    error_message=str(e)
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return EvalResult(
                prompt_id=prompt_id,
                description=description, 
                prompt=prompt,
                passed=False,
                execution_time=execution_time,
                error_message=f"Setup failed: {str(e)}"
            )
            
    def run_evaluation(
        self, 
        prompts_file: Path,
        output_file: Optional[Path] = None,
        tags_filter: Optional[List[str]] = None,
        include_optional: bool = True
    ) -> EvalSummary:
        """Run full evaluation suite.
        
        Args:
            prompts_file: Path to prompts YAML file
            output_file: Path to save results JSON (optional)
            tags_filter: Only run prompts with these tags (optional)
            include_optional: Whether to include optional tests
            
        Returns:
            EvalSummary with overall results
        """
        if self.verbose:
            print(f"Loading prompts from {prompts_file}")
            
        # Load prompts configuration
        with open(prompts_file, 'r') as f:
            config = yaml.safe_load(f)
            
        prompts = config['prompts']
        test_config = config.get('test_config', {})
        timeouts = test_config.get('timeouts', {})
        
        # Filter prompts
        filtered_prompts = []
        for prompt_config in prompts:
            # Skip optional tests if requested
            if not include_optional and prompt_config.get('optional', False):
                continue
                
            # Filter by tags if specified
            if tags_filter:
                prompt_tags = prompt_config.get('tags', [])
                if not any(tag in prompt_tags for tag in tags_filter):
                    continue
                    
            filtered_prompts.append(prompt_config)
            
        if self.verbose:
            print(f"Running {len(filtered_prompts)} prompts")
            
        # Run all prompts
        results = []
        total_time = 0
        
        for prompt_config in filtered_prompts:
            # Determine timeout
            prompt_tags = prompt_config.get('tags', [])
            timeout = timeouts.get('basic_tests', 60)
            if 'scvi' in prompt_tags:
                timeout = timeouts.get('scvi_tests', 300)
            elif 'large_dataset' in prompt_tags:
                timeout = timeouts.get('large_dataset_tests', 600)
                
            # Use appropriate test dataset
            dataset_config = test_config.get('default_dataset', {})
            if 'small_dataset' in prompt_tags:
                dataset_config = test_config.get('small_dataset', dataset_config)
            elif 'batch' in prompt_tags:
                dataset_config = test_config.get('batch_dataset', dataset_config)
                
            result = self.run_single_prompt(prompt_config, dataset_config, timeout)
            results.append(result)
            total_time += result.execution_time
            
            if self.verbose:
                status = "PASS" if result.passed else ("SKIP" if result.passed is None else "FAIL")
                print(f"  {result.prompt_id}: {status} ({result.execution_time:.1f}s)")
                
        # Calculate summary statistics
        passed = [r for r in results if r.passed is True]
        failed = [r for r in results if r.passed is False]
        skipped = [r for r in results if r.passed is None]
        
        # Calculate pass rates
        total_testable = len(passed) + len(failed)
        overall_pass_rate = len(passed) / total_testable if total_testable > 0 else 0
        
        # Core vs optional pass rates
        core_results = [r for r in results if not any(p['id'] == r.prompt_id and p.get('optional', False) for p in prompts)]
        core_passed = [r for r in core_results if r.passed is True]
        core_testable = len(core_passed) + len([r for r in core_results if r.passed is False])
        core_pass_rate = len(core_passed) / core_testable if core_testable > 0 else 0
        
        optional_results = [r for r in results if any(p['id'] == r.prompt_id and p.get('optional', False) for p in prompts)]
        optional_passed = [r for r in optional_results if r.passed is True]
        optional_testable = len(optional_passed) + len([r for r in optional_results if r.passed is False])
        optional_pass_rate = len(optional_passed) / optional_testable if optional_testable > 0 else 1.0
        
        summary = EvalSummary(
            total_prompts=len(results),
            passed_prompts=len(passed),
            failed_prompts=len(failed),
            skipped_prompts=len(skipped),
            pass_rate=overall_pass_rate,
            core_pass_rate=core_pass_rate,
            optional_pass_rate=optional_pass_rate,
            total_execution_time=total_time,
            timestamp=datetime.now().isoformat(),
            results=results
        )
        
        # Save results if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2, default=str)
                
        if self.verbose:
            print(f"\nEvaluation Summary:")
            print(f"  Total prompts: {summary.total_prompts}")
            print(f"  Passed: {summary.passed_prompts}")
            print(f"  Failed: {summary.failed_prompts}") 
            print(f"  Skipped: {summary.skipped_prompts}")
            print(f"  Overall pass rate: {summary.pass_rate:.1%}")
            print(f"  Core pass rate: {summary.core_pass_rate:.1%}")
            print(f"  Optional pass rate: {summary.optional_pass_rate:.1%}")
            print(f"  Total time: {summary.total_execution_time:.1f}s")
            
        return summary


def run_evaluation(
    prompts_file: str = "eval/prompts.yaml",
    output_file: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    include_optional: bool = True,
    verbose: bool = True
) -> EvalSummary:
    """Convenience function to run evaluation.
    
    Args:
        prompts_file: Path to prompts YAML file
        output_file: Path to save results JSON (optional)
        tags_filter: Only run prompts with these tags (optional)
        include_optional: Whether to include optional tests
        verbose: Whether to print progress
        
    Returns:
        EvalSummary with results
    """
    runner = EvalRunner(verbose=verbose)
    
    prompts_path = Path(prompts_file)
    output_path = Path(output_file) if output_file else None
    
    return runner.run_evaluation(
        prompts_path, 
        output_path, 
        tags_filter, 
        include_optional
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run scQC Agent evaluation")
    parser.add_argument("--prompts", default="eval/prompts.yaml", 
                       help="Path to prompts YAML file")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--tags", nargs="+", help="Filter by tags")
    parser.add_argument("--no-optional", action="store_true", 
                       help="Skip optional tests")
    parser.add_argument("--quiet", action="store_true", 
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    summary = run_evaluation(
        prompts_file=args.prompts,
        output_file=args.output,
        tags_filter=args.tags,
        include_optional=not args.no_optional,
        verbose=not args.quiet
    )
    
    # Exit with error code if pass rate is too low
    if summary.pass_rate < 0.9:
        print(f"FAILURE: Pass rate {summary.pass_rate:.1%} below 90% threshold")
        exit(1)
    else:
        print(f"SUCCESS: Pass rate {summary.pass_rate:.1%}")
        exit(0)
