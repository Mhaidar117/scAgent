"""Test stored plan execution functionality."""

import json
import tempfile
from pathlib import Path
import pytest

from scqc_agent.agent.runtime import Agent
from scqc_agent.state import SessionState


def test_execute_from_stored_plan(tmp_path):
    """Test that agent can execute from a stored plan.json file."""
    # Create a temporary state file
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create a stored plan in new envelope format
    from datetime import datetime
    import hashlib

    plan = [
        {
            "tool": "compute_qc_metrics",
            "description": "Compute QC metrics for human data",
            "params": {"species": "human"}
        }
    ]

    plan_json = json.dumps(plan, sort_keys=True)
    plan_envelope = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "message": "compute QC metrics",
        "intent": "compute_qc",
        "checksum": hashlib.sha256(plan_json.encode()).hexdigest(),
        "plan": plan
    }

    plan_file = tmp_path / "stored_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan_envelope, f, indent=2)

    # Initialize agent
    agent = Agent(str(state_file))

    # Execute from stored plan (will fail due to no data loaded, but tests the path)
    result = agent.chat(
        message="compute QC metrics",
        mode="execute",
        plan_path=str(plan_file)
    )

    # Should not error on plan loading
    assert "status" in result
    # Will fail on execution due to no data, but that's expected
    assert result.get("mode") == "execution"


def test_execute_with_invalid_plan_path(tmp_path):
    """Test that agent handles missing plan file gracefully."""
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    agent = Agent(str(state_file))

    # Try to execute with non-existent plan file
    result = agent.chat(
        message="compute QC metrics",
        mode="execute",
        plan_path="/nonexistent/plan.json"
    )

    assert result.get("status") == "failed"
    assert "not found" in result.get("error", "").lower()


def test_execute_with_malformed_plan(tmp_path):
    """Test that agent handles malformed plan.json gracefully."""
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create a malformed plan (not a list)
    plan_file = tmp_path / "malformed_plan.json"
    with open(plan_file, 'w') as f:
        json.dump({"tool": "bad_format"}, f)

    agent = Agent(str(state_file))

    result = agent.chat(
        message="compute QC metrics",
        mode="execute",
        plan_path=str(plan_file)
    )

    assert result.get("status") == "failed"
    assert "invalid plan format" in result.get("error", "").lower()


def test_execute_without_plan_path_still_works(tmp_path):
    """Test that execute mode still works without plan_path (regenerates plan)."""
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    agent = Agent(str(state_file))

    # Execute without plan_path should regenerate plan
    result = agent.chat(
        message="compute QC metrics",
        mode="execute",
        plan_path=None
    )

    # Should work (though will fail on no data loaded)
    assert "mode" in result
    assert result.get("mode") == "execution"


def test_plan_mode_ignores_plan_path(tmp_path):
    """Test that plan mode ignores plan_path parameter."""
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    plan_file = tmp_path / "stored_plan.json"
    plan = [{"tool": "compute_qc_metrics", "description": "Test", "params": {}}]
    with open(plan_file, 'w') as f:
        json.dump(plan, f)

    agent = Agent(str(state_file))

    # Plan mode should not use plan_path
    result = agent.chat(
        message="compute QC metrics",
        mode="plan",
        plan_path=str(plan_file)  # Should be ignored
    )

    assert result.get("mode") == "planning"
    assert "plan" in result
    # Should generate new plan, not use stored one


def test_planning_exposes_plan_path(tmp_path):
    """Test that planning phase returns explicit plan_path in result."""
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    agent = Agent(str(state_file))

    # Generate a plan
    result = agent.chat(
        message="compute QC metrics for mouse data",
        mode="plan"
    )

    # Verify plan_path is exposed
    assert "plan_path" in result
    assert result["plan_path"] is not None
    assert result["plan_path"].endswith("plan.json")

    # Verify the file actually exists
    plan_path = Path(result["plan_path"])
    assert plan_path.exists()

    # Verify it contains the plan envelope (new format)
    with open(plan_path, 'r') as f:
        saved_envelope = json.load(f)

    assert "plan" in saved_envelope
    assert "version" in saved_envelope
    assert "checksum" in saved_envelope
    assert "created_at" in saved_envelope
    assert saved_envelope["plan"] == result["plan"]

    # Verify next_steps mentions the plan_path
    assert "next_steps" in result
    assert result["plan_path"] in result["next_steps"]


def test_plan_checksum_validation(tmp_path):
    """Test that modified plans are detected via checksum."""
    import hashlib
    from datetime import datetime

    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create a plan with valid checksum
    plan = [{"tool": "compute_qc_metrics", "description": "Test", "params": {}}]
    plan_json = json.dumps(plan, sort_keys=True)
    valid_checksum = hashlib.sha256(plan_json.encode()).hexdigest()

    plan_envelope = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "message": "test",
        "intent": "compute_qc",
        "checksum": valid_checksum,
        "plan": plan
    }

    plan_file = tmp_path / "valid_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan_envelope, f, indent=2)

    # Now manually modify the plan without updating checksum
    plan_envelope["plan"].append({"tool": "plot_qc", "description": "Added step", "params": {}})

    modified_plan_file = tmp_path / "modified_plan.json"
    with open(modified_plan_file, 'w') as f:
        json.dump(plan_envelope, f, indent=2)

    agent = Agent(str(state_file))

    # Valid plan should load successfully
    result_valid = agent.chat("test", mode="execute", plan_path=str(plan_file))
    assert result_valid.get("mode") == "execution"

    # Modified plan should fail checksum validation
    result_modified = agent.chat("test", mode="execute", plan_path=str(modified_plan_file))
    assert result_modified.get("status") == "failed"
    assert "checksum" in result_modified.get("error", "").lower()
    assert "validation_details" in result_modified


def test_legacy_plan_format_support(tmp_path):
    """Test that legacy plans (plain list) are still supported."""
    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create a legacy format plan (plain list, no envelope)
    legacy_plan = [
        {"tool": "compute_qc_metrics", "description": "Legacy test", "params": {}}
    ]

    plan_file = tmp_path / "legacy_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(legacy_plan, f, indent=2)

    agent = Agent(str(state_file))

    # Should still work with legacy format
    result = agent.chat("test", mode="execute", plan_path=str(plan_file))
    assert result.get("mode") == "execution"
    # Execution will fail due to no data, but plan loading should succeed


def test_plan_validation_unknown_tool(tmp_path):
    """Test that plans with unknown tools are rejected."""
    import hashlib
    from datetime import datetime

    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create plan with unknown tool
    plan = [{"tool": "nonexistent_tool", "description": "Invalid", "params": {}}]
    plan_json = json.dumps(plan, sort_keys=True)

    plan_envelope = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "message": "test",
        "intent": "test",
        "checksum": hashlib.sha256(plan_json.encode()).hexdigest(),
        "plan": plan
    }

    plan_file = tmp_path / "invalid_tool_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan_envelope, f, indent=2)

    agent = Agent(str(state_file))

    result = agent.chat("test", mode="execute", plan_path=str(plan_file))

    assert result.get("status") == "failed"
    assert "unknown tool" in result.get("error", "").lower()
    assert "validation_details" in result
    assert "tool_registry" in result["validation_details"]["check"]


def test_plan_validation_empty_plan(tmp_path):
    """Test that empty plans are rejected."""
    import hashlib
    from datetime import datetime

    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create empty plan
    plan = []
    plan_json = json.dumps(plan, sort_keys=True)

    plan_envelope = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "message": "test",
        "intent": "test",
        "checksum": hashlib.sha256(plan_json.encode()).hexdigest(),
        "plan": plan
    }

    plan_file = tmp_path / "empty_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan_envelope, f, indent=2)

    agent = Agent(str(state_file))

    result = agent.chat("test", mode="execute", plan_path=str(plan_file))

    assert result.get("status") == "failed"
    assert "empty" in result.get("error", "").lower()


def test_plan_validation_missing_tool_field(tmp_path):
    """Test that steps without 'tool' field are rejected."""
    import hashlib
    from datetime import datetime

    state_file = tmp_path / "test_state.json"
    state = SessionState(run_id="test_run")
    state.save(str(state_file))

    # Create plan with step missing 'tool' field
    plan = [{"description": "No tool specified", "params": {}}]
    plan_json = json.dumps(plan, sort_keys=True)

    plan_envelope = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "message": "test",
        "intent": "test",
        "checksum": hashlib.sha256(plan_json.encode()).hexdigest(),
        "plan": plan
    }

    plan_file = tmp_path / "missing_tool_plan.json"
    with open(plan_file, 'w') as f:
        json.dump(plan_envelope, f, indent=2)

    agent = Agent(str(state_file))

    result = agent.chat("test", mode="execute", plan_path=str(plan_file))

    assert result.get("status") == "failed"
    assert "missing required 'tool' field" in result.get("error", "").lower()
