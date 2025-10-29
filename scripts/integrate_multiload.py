#!/usr/bin/env python3
"""Script to integrate load_kidney_data tool into scQC Agent runtime.

This script adds the necessary registrations to runtime.py to complete
the integration of the multi-file kidney data loader.

Usage:
    python scripts/integrate_multiload.py

What it does:
    1. Adds "load_kidney_data" to tool registry in _init_tool_registry()
    2. Adds _load_kidney_data_tool() wrapper method
    3. Creates backup of original runtime.py
    4. Validates the integration

Requirements:
    - scqc_agent must be installed
    - runtime.py must be at expected location
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime


def find_runtime_file():
    """Find runtime.py file location."""
    # Try to find from installed package
    try:
        import scqc_agent
        package_path = Path(scqc_agent.__file__).parent
        runtime_path = package_path / "agent" / "runtime.py"
        if runtime_path.exists():
            return runtime_path
    except ImportError:
        pass

    # Try relative path from script location
    script_dir = Path(__file__).parent.parent
    runtime_path = script_dir / "scqc_agent" / "agent" / "runtime.py"
    if runtime_path.exists():
        return runtime_path

    return None


def create_backup(runtime_path):
    """Create timestamped backup of runtime.py."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = runtime_path.parent / f"runtime.py.backup_{timestamp}"
    shutil.copy2(runtime_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def check_already_integrated(content):
    """Check if tool is already integrated."""
    if '"load_kidney_data"' in content and '_load_kidney_data_tool' in content:
        return True
    return False


def add_to_tool_registry(content):
    """Add load_kidney_data to _init_tool_registry method."""
    # Find the tool registry initialization
    registry_start = content.find('self.tools = {')
    if registry_start == -1:
        print("ERROR: Could not find 'self.tools = {' in runtime.py")
        return None

    # Find the line with "load_data"
    load_data_line = content.find('"load_data": self._load_data_tool,', registry_start)
    if load_data_line == -1:
        print("ERROR: Could not find load_data tool registration")
        return None

    # Find the end of that line
    line_end = content.find('\n', load_data_line)

    # Insert new line after load_data
    new_line = '\n        "load_kidney_data": self._load_kidney_data_tool,'

    modified_content = (
        content[:line_end] +
        new_line +
        content[line_end:]
    )

    print("✓ Added tool to registry")
    return modified_content


def add_wrapper_method(content):
    """Add _load_kidney_data_tool wrapper method."""
    # Find a good place to insert - after _load_data_tool
    load_data_tool_start = content.find('def _load_data_tool(self')
    if load_data_tool_start == -1:
        print("WARNING: Could not find _load_data_tool method, inserting at end of class")
        # Find the end of the Agent class
        insert_pos = content.rfind('\n\n', 0, len(content) - 100)
    else:
        # Find the end of _load_data_tool method
        # Look for the next method definition
        next_method = content.find('\n    def ', load_data_tool_start + 10)
        if next_method == -1:
            # Last method in class
            insert_pos = content.rfind('\n\n', 0, len(content) - 100)
        else:
            # Insert before next method
            insert_pos = next_method

    # Wrapper method code
    wrapper_code = '''

    def _load_kidney_data_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Wrapper for load_kidney_data tool.

        Loads kidney scRNA-seq datasets from raw 10X H5, filtered 10X H5,
        and metadata CSV files.

        Args:
            params: Tool parameters from agent plan

        Returns:
            ToolResult with loaded data checkpoints and artifacts
        """
        from ..tools.multiload import load_kidney_data
        return load_kidney_data(self.state, **params)
'''

    modified_content = (
        content[:insert_pos] +
        wrapper_code +
        content[insert_pos:]
    )

    print("✓ Added wrapper method")
    return modified_content


def validate_integration(content):
    """Validate that integration was successful."""
    errors = []

    # Check tool registry
    if '"load_kidney_data": self._load_kidney_data_tool' not in content:
        errors.append("Tool not found in registry")

    # Check wrapper method
    if 'def _load_kidney_data_tool(self' not in content:
        errors.append("Wrapper method not found")

    # Check import
    if 'from ..tools.multiload import load_kidney_data' not in content:
        errors.append("Import statement not found")

    return errors


def main():
    """Run integration script."""
    print("\n" + "=" * 60)
    print("Multi-file Kidney Data Loader - Integration Script")
    print("=" * 60)

    # Find runtime.py
    print("\n1. Locating runtime.py...")
    runtime_path = find_runtime_file()
    if not runtime_path:
        print("ERROR: Could not find runtime.py")
        print("Please ensure scqc_agent is installed or run from project root")
        sys.exit(1)
    print(f"✓ Found: {runtime_path}")

    # Read current content
    print("\n2. Reading current runtime.py...")
    with open(runtime_path, 'r') as f:
        original_content = f.read()
    print(f"✓ Read {len(original_content)} characters")

    # Check if already integrated
    print("\n3. Checking if already integrated...")
    if check_already_integrated(original_content):
        print("⚠ Tool appears to already be integrated!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Create backup
    print("\n4. Creating backup...")
    backup_path = create_backup(runtime_path)

    # Add to tool registry
    print("\n5. Adding to tool registry...")
    modified_content = add_to_tool_registry(original_content)
    if modified_content is None:
        print("ERROR: Failed to add to tool registry")
        sys.exit(1)

    # Add wrapper method
    print("\n6. Adding wrapper method...")
    modified_content = add_wrapper_method(modified_content)
    if modified_content is None:
        print("ERROR: Failed to add wrapper method")
        sys.exit(1)

    # Validate
    print("\n7. Validating integration...")
    errors = validate_integration(modified_content)
    if errors:
        print("ERROR: Validation failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nRestoring from backup...")
        shutil.copy2(backup_path, runtime_path)
        sys.exit(1)
    print("✓ Validation passed")

    # Write modified content
    print("\n8. Writing modified runtime.py...")
    with open(runtime_path, 'w') as f:
        f.write(modified_content)
    print(f"✓ Written {len(modified_content)} characters")

    # Final validation
    print("\n9. Final validation...")
    try:
        # Try to import runtime to check syntax
        import scqc_agent.agent.runtime
        print("✓ Syntax check passed")
    except Exception as e:
        print(f"ERROR: Syntax error in modified runtime.py: {e}")
        print("Restoring from backup...")
        shutil.copy2(backup_path, runtime_path)
        sys.exit(1)

    # Success
    print("\n" + "=" * 60)
    print("Integration Complete!")
    print("=" * 60)
    print(f"\nModified file: {runtime_path}")
    print(f"Backup saved: {backup_path}")
    print("\nNext steps:")
    print("  1. Run tests: pytest tests/test_multiload.py -v")
    print("  2. Try example: python examples/multiload_quickstart.py")
    print("  3. Test with agent:")
    print("     from scqc_agent.agent.runtime import Agent")
    print("     agent = Agent()")
    print("     agent.chat('Load kidney data from ...')")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
