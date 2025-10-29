"""Simple test to verify visualization tools use unique step directories."""

import tempfile
import os
from pathlib import Path
from scqc_agent.state import SessionState
from scqc_agent.tools.qc import plot_qc_metrics, compute_qc_metrics
import scanpy as sc
import numpy as np


def test_qc_and_artifact_preservation():
    """Test that QC plots and other visualizations use unique directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test session
        state = SessionState(run_id="test_preservation")

        # Change to temp directory
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

            print("\n" + "=" * 70)
            print("Testing QC Plot Progression with Dynamic Step Numbering")
            print("=" * 70)

            # Compute QC metrics
            print("\n1. Computing QC metrics...")
            compute_result = compute_qc_metrics(state, species="mouse")
            assert not compute_result.message.startswith("❌"), f"QC compute failed: {compute_result.message}"
            if compute_result.state_delta:
                state.adata_path = compute_result.state_delta.get("adata_path", state.adata_path)
                state.update_metadata(compute_result.state_delta)
            print(f"   ✅ QC metrics computed, history entries: {len(state.history)}")

            # Plot QC metrics - first time (pre-filtering)
            print("\n2. Creating first QC plot (pre-filtering)...")
            qc_plot_1 = plot_qc_metrics(state, stage="pre", plot_types=["violin"])
            assert not qc_plot_1.message.startswith("❌"), f"First QC plot failed: {qc_plot_1.message}"
            assert len(qc_plot_1.artifacts) > 0, "First plot should create artifacts"

            qc_dir_1 = Path(qc_plot_1.artifacts[0]).parent
            step_name_1 = qc_dir_1.name
            print(f"   ✅ Created: {step_name_1}")
            print(f"      Full path: {qc_dir_1}")
            print(f"      Artifacts: {len(qc_plot_1.artifacts)} files")
            print(f"      History entries: {len(state.history)}")

            # Plot QC metrics - second time (simulating after filtering)
            print("\n3. Creating second QC plot (post-filtering)...")
            qc_plot_2 = plot_qc_metrics(state, stage="post", plot_types=["violin"])
            assert not qc_plot_2.message.startswith("❌"), f"Second QC plot failed: {qc_plot_2.message}"
            assert len(qc_plot_2.artifacts) > 0, "Second plot should create artifacts"

            qc_dir_2 = Path(qc_plot_2.artifacts[0]).parent
            step_name_2 = qc_dir_2.name
            print(f"   ✅ Created: {step_name_2}")
            print(f"      Full path: {qc_dir_2}")
            print(f"      Artifacts: {len(qc_plot_2.artifacts)} files")
            print(f"      History entries: {len(state.history)}")

            # Plot QC metrics - third time (another iteration)
            print("\n4. Creating third QC plot (another iteration)...")
            qc_plot_3 = plot_qc_metrics(state, stage="pre", plot_types=["violin"])
            assert not qc_plot_3.message.startswith("❌"), f"Third QC plot failed: {qc_plot_3.message}"
            assert len(qc_plot_3.artifacts) > 0, "Third plot should create artifacts"

            qc_dir_3 = Path(qc_plot_3.artifacts[0]).parent
            step_name_3 = qc_dir_3.name
            print(f"   ✅ Created: {step_name_3}")
            print(f"      Full path: {qc_dir_3}")
            print(f"      Artifacts: {len(qc_plot_3.artifacts)} files")
            print(f"      History entries: {len(state.history)}")

            print("\n" + "-" * 70)
            print("Verification")
            print("-" * 70)

            # Verify all directories are different
            all_dirs = [qc_dir_1, qc_dir_2, qc_dir_3]
            all_names = [qc_dir_1.name, qc_dir_2.name, qc_dir_3.name]

            print(f"\n✓ Checking directory uniqueness...")
            print(f"  Directory 1: {step_name_1}")
            print(f"  Directory 2: {step_name_2}")
            print(f"  Directory 3: {step_name_3}")

            assert len(set(all_dirs)) == 3, f"All directories should be unique, got: {all_names}"
            print(f"  ✅ All 3 directories are unique")

            # Verify all directories exist
            print(f"\n✓ Checking directory existence...")
            for i, dir_path in enumerate(all_dirs, 1):
                assert dir_path.exists(), f"Directory {i} should exist: {dir_path}"
                print(f"  ✅ Directory {i} exists: {dir_path.name}")

            # Verify files exist in each directory
            print(f"\n✓ Checking artifact files...")
            for i, dir_path in enumerate(all_dirs, 1):
                files = list(dir_path.glob("*.png"))
                assert len(files) > 0, f"Directory {i} should contain plot files"
                print(f"  ✅ Directory {i} contains {len(files)} plot file(s)")

            # Verify state history
            print(f"\n✓ Checking state history...")
            print(f"  Total history entries: {len(state.history)}")
            print(f"  Total artifacts in state: {len(state.artifacts)}")

            # Check that history entries have unique step numbers
            step_numbers = []
            for entry in state.history:
                if "checkpoint_path" in entry:
                    checkpoint_path = Path(entry["checkpoint_path"])
                    parent_dir = checkpoint_path.parent.name
                    # Extract step number from directory name like "step_02_qc_plots_pre"
                    if parent_dir.startswith("step_"):
                        step_num = parent_dir.split("_")[1]
                        step_numbers.append(step_num)

            print(f"  Step numbers in history: {step_numbers}")
            assert len(set(step_numbers)) == len(step_numbers), "Step numbers should be unique"
            print(f"  ✅ All step numbers are unique")

            # Verify artifacts are accumulating
            assert len(state.artifacts) >= 3, f"Should have at least 3 artifacts, got {len(state.artifacts)}"
            print(f"  ✅ Artifacts are accumulating ({len(state.artifacts)} total)")

            print("\n" + "=" * 70)
            print("✅ ALL TESTS PASSED!")
            print("=" * 70)
            print("\nSummary:")
            print(f"  • Successfully created 3 QC plot directories")
            print(f"  • All directories use unique step numbers")
            print(f"  • All plots preserved throughout workflow")
            print(f"  • State history properly tracks all steps")
            print(f"  • Total artifacts preserved: {len(state.artifacts)}")
            print("\n✨ Visualization progression fix verified!")

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    import sys

    try:
        test_qc_and_artifact_preservation()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
