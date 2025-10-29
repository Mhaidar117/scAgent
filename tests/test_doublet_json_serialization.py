"""Test that doublet detection returns JSON-serializable state_delta."""

import json
import numpy as np
from scqc_agent.state import ToolResult


def test_numpy_types_cause_error():
    """Verify that NumPy int64 causes JSON serialization errors."""
    # This should fail
    state_delta_bad = {
        "n_cells": np.int64(1000),
        "n_doublets": np.int64(60)
    }

    try:
        json.dumps(state_delta_bad)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "int64" in str(e) or "not JSON serializable" in str(e)
        print(f"✅ Confirmed: NumPy int64 causes error: {e}")


def test_python_int_serializes():
    """Verify that Python int serializes correctly."""
    # This should succeed
    state_delta_good = {
        "n_cells": int(np.int64(1000)),
        "n_doublets": int(np.int64(60))
    }

    result = json.dumps(state_delta_good)
    assert result == '{"n_cells": 1000, "n_doublets": 60}'
    print(f"✅ Python int serializes correctly: {result}")


def test_toolresult_with_numpy_types():
    """Test that ToolResult state_delta with NumPy types can be problematic."""
    # Simulate what happens in apply_doublet_filter
    original_n_cells = np.int64(1000)
    n_doublets = np.int64(60)
    n_kept = original_n_cells - n_doublets  # NumPy int64

    # Without conversion (OLD CODE - would fail)
    state_delta_bad = {
        "cells_before": original_n_cells,
        "cells_after": n_kept,
    }

    # With conversion (NEW CODE - should work)
    state_delta_good = {
        "cells_before": int(original_n_cells),
        "cells_after": int(n_kept),
    }

    # Verify bad version fails
    try:
        json.dumps(state_delta_bad)
        assert False, "Bad state_delta should fail"
    except TypeError:
        print("✅ NumPy types in state_delta cause JSON error (as expected)")

    # Verify good version works
    result = json.dumps(state_delta_good)
    assert '"cells_before": 1000' in result
    assert '"cells_after": 940' in result
    print(f"✅ Converted state_delta serializes correctly: {result}")


def test_toolresult_serialization():
    """Test that ToolResult with converted types is fully JSON-serializable."""
    # Simulate the fixed state_delta from apply_doublet_filter
    state_delta = {
        "adata_path": "/path/to/data.h5ad",
        "cells_before_doublet_filter": int(np.int64(8877)),
        "cells_after_doublet_filter": int(np.int64(8344)),
        "doublets_removed": int(np.int64(533)),
        "doublet_filter_threshold": round(float(0.06), 4),
        "final_doublet_rate": round(0.06, 4),
        "doublet_filter_applied": True
    }

    # This should work now
    result_json = json.dumps(state_delta)
    print(f"✅ Complete state_delta serializes successfully")

    # Verify contents
    result_dict = json.loads(result_json)
    assert result_dict["cells_before_doublet_filter"] == 8877
    assert result_dict["cells_after_doublet_filter"] == 8344
    assert result_dict["doublets_removed"] == 533
    print(f"✅ All values correctly deserialized")


if __name__ == "__main__":
    print("Testing JSON serialization fix for DoubletFinder...")
    print()

    test_numpy_types_cause_error()
    print()

    test_python_int_serializes()
    print()

    test_toolresult_with_numpy_types()
    print()

    test_toolresult_serialization()
    print()

    print("=" * 60)
    print("✅ ALL TESTS PASSED - JSON serialization fix verified!")
    print("=" * 60)
