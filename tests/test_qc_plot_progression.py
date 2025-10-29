"""Test that QC plots preserve progression across multiple calls."""

import pytest
from pathlib import Path
from scqc_agent.state import SessionState
from scqc_agent.tools.qc import plot_qc_metrics, compute_qc_metrics
import tempfile
import scanpy as sc
import numpy as np


def test_qc_plots_preserve_progression():
    """Test that plotting QC metrics multiple times creates unique directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test session
        state = SessionState(run_id="test_progression")

        # Change to temp directory for test artifacts
        import os
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            # Create synthetic data
            n_obs = 500
            n_vars = 200
            adata = sc.AnnData(
                X=np.random.negative_binomial(5, 0.3, (n_obs, n_vars)),
                var={"gene_names": [f"Gene_{i}" for i in range(n_vars)]}
            )

            # Add MT genes for QC
            mt_genes = [f"mt-{i}" for i in range(10)]
            adata.var_names = mt_genes + [f"Gene_{i}" for i in range(10, n_vars)]

            # Save synthetic data
            synth_path = Path(tmpdir) / "synthetic_data.h5ad"
            adata.write_h5ad(synth_path)
            state.adata_path = str(synth_path)

            # Compute QC metrics
            compute_result = compute_qc_metrics(state, species="mouse")
            if compute_result.state_delta:
                state.adata_path = compute_result.state_delta.get("adata_path", state.adata_path)
                state.update_metadata(compute_result.state_delta)

            # Plot QC metrics - first time (pre-filtering)
            plot_result_1 = plot_qc_metrics(state, stage="pre")
            assert not plot_result_1.message.startswith("❌"), f"First plot failed: {plot_result_1.message}"
            assert len(plot_result_1.artifacts) > 0, "First plot should create artifacts"

            # Extract directory from first plot artifacts
            first_artifact = Path(plot_result_1.artifacts[0])
            first_dir = first_artifact.parent

            # Verify directory name includes step number and stage
            assert "step_00" in str(first_dir), f"First directory should include step_00: {first_dir}"
            assert "qc_plots_pre" in str(first_dir), f"First directory should include qc_plots_pre: {first_dir}"

            # Plot QC metrics - second time (simulate after filtering, still "pre" stage)
            plot_result_2 = plot_qc_metrics(state, stage="pre")
            assert not plot_result_2.message.startswith("❌"), f"Second plot failed: {plot_result_2.message}"
            assert len(plot_result_2.artifacts) > 0, "Second plot should create artifacts"

            # Extract directory from second plot artifacts
            second_artifact = Path(plot_result_2.artifacts[0])
            second_dir = second_artifact.parent

            # Verify second directory is different from first
            assert first_dir != second_dir, "Second plot should create a different directory"
            assert "step_01" in str(second_dir), f"Second directory should include step_01: {second_dir}"

            # Plot QC metrics - third time (post-filtering stage)
            plot_result_3 = plot_qc_metrics(state, stage="post")
            assert not plot_result_3.message.startswith("❌"), f"Third plot failed: {plot_result_3.message}"
            assert len(plot_result_3.artifacts) > 0, "Third plot should create artifacts"

            # Extract directory from third plot artifacts
            third_artifact = Path(plot_result_3.artifacts[0])
            third_dir = third_artifact.parent

            # Verify third directory is unique
            assert third_dir != first_dir, "Third plot should differ from first"
            assert third_dir != second_dir, "Third plot should differ from second"
            assert "step_02" in str(third_dir), f"Third directory should include step_02: {third_dir}"
            assert "qc_plots_post" in str(third_dir), f"Third directory should include qc_plots_post: {third_dir}"

            # Verify all three directories exist
            assert first_dir.exists(), f"First plot directory should exist: {first_dir}"
            assert second_dir.exists(), f"Second plot directory should exist: {second_dir}"
            assert third_dir.exists(), f"Third plot directory should exist: {third_dir}"

            # Verify all artifacts are preserved in state
            assert len(state.history) == 3, f"Should have 3 history entries, got {len(state.history)}"

            # Verify each history entry has artifacts
            for i, entry in enumerate(state.history):
                assert len(entry["artifacts"]) > 0, f"History entry {i} should have artifacts"

            # Verify state.artifacts dictionary has all unique paths
            all_artifact_paths = list(state.artifacts.keys())
            assert len(all_artifact_paths) >= 3, f"Should have at least 3 artifact paths, got {len(all_artifact_paths)}"

            # Verify no duplicate paths (all unique)
            unique_paths = set(all_artifact_paths)
            assert len(unique_paths) == len(all_artifact_paths), "All artifact paths should be unique"

            print(f"✅ Test passed: All three plot directories are unique and preserved")
            print(f"   First:  {first_dir}")
            print(f"   Second: {second_dir}")
            print(f"   Third:  {third_dir}")

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Run the test directly
    import sys

    try:
        test_qc_plots_preserve_progression()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
