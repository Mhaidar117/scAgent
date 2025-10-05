# Plan Validation Design Document

## Overview

This document describes the comprehensive plan validation system implemented for stored plan execution in scQC Agent. The system ensures plan integrity, detects corruption, and prevents execution of invalid or stale plans.

## Motivation

Stored plans enable reproducibility and batch processing, but introduce risks:
- **Manual modification** without validation could break workflows
- **Corrupted files** could cause execution failures
- **Stale plans** might reference deprecated tools
- **Malformed JSON** could crash the agent

The validation system addresses these risks with multi-layered checks.

## Plan Format Evolution

### Legacy Format (Pre-v1.0)
```json
[
  {"tool": "compute_qc_metrics", "description": "...", "params": {...}},
  {"tool": "plot_qc", "description": "...", "params": {...}}
]
```

**Limitations:**
- No metadata (creation time, intent, original message)
- No integrity checking (manual edits undetected)
- No version tracking

### Envelope Format (v1.0)
```json
{
  "version": "1.0",
  "created_at": "2025-10-05T15:40:20.123456",
  "message": "compute QC for mouse data",
  "intent": "compute_qc",
  "checksum": "abc123def456...",  // SHA256 of plan array
  "plan": [
    {"tool": "compute_qc_metrics", "description": "...", "params": {...}},
    {"tool": "plot_qc", "description": "...", "params": {...}}
  ]
}
```

**Benefits:**
- **Metadata preservation**: Original message and intent stored
- **Integrity verification**: SHA256 checksum detects tampering
- **Timestamp tracking**: Age warnings for stale plans
- **Version tracking**: Future format migrations supported

## Validation Pipeline

The `_validate_and_load_plan()` method implements 7 validation checks:

### Check 1: File Existence
```python
if not plan_file.exists():
    return {"error": "Plan file not found: {path}"}
```

**Rationale:** Fail fast if path is invalid or file deleted.

### Check 2: JSON Parsing
```python
try:
    plan_data = json.load(f)
except json.JSONDecodeError as e:
    return {"error": "Failed to parse plan file: {e}"}
```

**Rationale:** Detect corrupted JSON before processing.

### Check 3: Format Detection
```python
if isinstance(plan_data, dict) and "plan" in plan_data:
    # Envelope format (v1.0)
elif isinstance(plan_data, list):
    # Legacy format
else:
    return {"error": "Invalid plan format"}
```

**Rationale:** Support both legacy and modern formats for backward compatibility.

### Check 4: Checksum Verification (v1.0 only)
```python
plan_json = json.dumps(plan, sort_keys=True)
computed = hashlib.sha256(plan_json.encode()).hexdigest()

if computed != metadata["checksum"]:
    return {"error": "Plan integrity check failed: checksum mismatch"}
```

**Rationale:** Detect manual modifications that could break execution.

**How it works:**
1. Plan array is serialized with sorted keys (deterministic)
2. SHA256 hash computed on serialized JSON
3. Stored checksum compared to computed checksum
4. Mismatch indicates plan was modified without updating checksum

**Legitimate modification workflow:**
```python
import json, hashlib

# Load plan
with open("plan.json", 'r') as f:
    envelope = json.load(f)

# Modify plan
envelope["plan"].append({"tool": "new_step", ...})

# Recompute checksum
plan_json = json.dumps(envelope["plan"], sort_keys=True)
envelope["checksum"] = hashlib.sha256(plan_json.encode()).hexdigest()

# Save
with open("plan.json", 'w') as f:
    json.dump(envelope, f, indent=2)
```

### Check 5: Plan Age Warning
```python
created = datetime.fromisoformat(metadata["created_at"])
age = datetime.now() - created

if age > timedelta(days=30):
    metadata["age_warning"] = f"Plan is {age.days} days old"
```

**Rationale:** Warn about stale plans that might reference deprecated tools.
**Note:** This is a warning, not a failure.

### Check 6: Plan Structure
```python
if not isinstance(plan, list):
    return {"error": "Plan must be a list"}

if len(plan) == 0:
    return {"error": "Plan is empty"}
```

**Rationale:** Ensure plan is a non-empty list of steps.

### Check 7: Step Validation
```python
for i, step in enumerate(plan):
    # Must be dictionary
    if not isinstance(step, dict):
        return {"error": f"Step {i} is not a dictionary"}

    # Must have 'tool' field
    if "tool" not in step:
        return {"error": f"Step {i} missing required 'tool' field"}

    # Tool must exist in registry
    if tool_name not in self.tools:
        return {"error": f"Step {i}: Unknown tool '{tool_name}'"}
```

**Rationale:** Prevent execution of malformed steps or unknown tools.

## Error Reporting

All validation failures return detailed error information:

```python
{
    "error": "Human-readable error message",
    "status": "failed",
    "mode": "execution",
    "validation_details": {
        "check": "checksum",  # Which check failed
        "expected": "abc123...",
        "computed": "def456...",
        "warning": "Plan may have been manually modified"
    }
}
```

This enables debugging and provides actionable feedback to users.

## Implementation Details

### Location
- **Method:** `_validate_and_load_plan()` in `scqc_agent/agent/runtime.py:686-831`
- **Called from:** `_execution_phase()` when `plan_path` is provided
- **Tests:** `tests/test_stored_plan_execution.py` (11 tests, 100% pass)

### Backward Compatibility
The system maintains full backward compatibility:
- Legacy plans (plain list) continue to work
- No checksum validation for legacy plans
- Format auto-detection is transparent to users

### Performance
- Validation adds ~5-10ms overhead per plan load
- Checksum computation is O(n) in plan size
- Negligible impact for typical plans (<100 steps)

## Security Considerations

### Checksum Algorithm
- **SHA256** chosen for:
  - Cryptographic strength (prevents collision attacks)
  - Wide availability (Python stdlib)
  - Deterministic output (reproducible checksums)

### Threat Model
The validation system protects against:
- ✅ Accidental file corruption
- ✅ Unintended manual modifications
- ✅ JSON parsing errors
- ✅ Tool registry drift

**Not protected against:**
- ❌ Intentional malicious modification (attacker can recompute checksum)
- ❌ File system tampering (no digital signature)

For production environments requiring stronger security, consider:
- GPG/PGP signing of plan files
- Storing plans in immutable storage (e.g., git with signed commits)
- Access control on plan directories

## Testing Strategy

Comprehensive test coverage ensures validation reliability:

| Test | Purpose |
|------|---------|
| `test_execute_from_stored_plan` | Basic envelope format execution |
| `test_plan_checksum_validation` | Checksum mismatch detection |
| `test_legacy_plan_format_support` | Backward compatibility |
| `test_plan_validation_unknown_tool` | Tool registry check |
| `test_plan_validation_empty_plan` | Empty plan rejection |
| `test_plan_validation_missing_tool_field` | Required field validation |

## Future Enhancements

Potential improvements for future versions:

1. **Digital Signatures** - GPG/PGP signing for tamper-proof plans
2. **Plan Versioning** - Migration system for format changes
3. **Conditional Execution** - Skip-if-exists logic for idempotent workflows
4. **Plan Diffs** - Show changes between plan versions
5. **Plan Templates** - Parameterized plans for batch processing
6. **Rollback Support** - Undo plan execution to previous checkpoint

## Examples

See `examples/stored_plan_demo.py` for comprehensive demonstrations:
- Basic plan storage and execution
- Safe plan modification with checksum update
- Validation failure scenarios
- Legacy format compatibility

## References

- **Implementation:** `scqc_agent/agent/runtime.py:686-831`
- **Tests:** `tests/test_stored_plan_execution.py`
- **Documentation:** `CLAUDE.md` (Plan Validation section)
- **Demo:** `examples/stored_plan_demo.py`
