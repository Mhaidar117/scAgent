#!/usr/bin/env python3
"""
End-to-end test of complete kidney pipeline with agent integration.

Tests the full workflow:
1. Load kidney data (raw H5 + filtered H5 + metadata CSV)
2. Compute QC metrics
3. Generate QC violin plots
4. Generate knee plot (ambient RNA detection)
5. Run SCAR denoising
6. Detect doublets with DoubletFinder
7. Apply doublet filters
8. Quick graph (PCA + UMAP + clustering)
9. Detect marker genes
10. Annotate clusters with kidney cell types
11. Generate composition plots
"""

import os
import sys
from pathlib import Path
from scqc_agent.agent.runtime import Agent

# Test configuration
STATE_FILE = ".test_full_kidney_pipeline.json"
DATA_DIR = Path("Data_files/raw_data")

# File paths
RAW_H5 = DATA_DIR / "7_raw_feature_bc_matrix.h5"
FILTERED_H5 = DATA_DIR / "7_filtered_feature_bc_matrix.h5"
METADATA_CSV = DATA_DIR / "metadata.xlsx - Sheet1.csv"

# Verify files exist
for f in [RAW_H5, FILTERED_H5, METADATA_CSV]:
    if not f.exists():
        print(f"❌ Missing required file: {f}")
        sys.exit(1)

print("="*80)
print("KIDNEY PIPELINE END-TO-END TEST")
print("="*80)

# Initialize agent
print("\n1. Initializing agent...")
agent = Agent(STATE_FILE)
print(f"✓ Agent initialized with {len(agent.tools)} tools")

# Define workflow steps
steps = [
    {
        "name": "Load kidney data",
        "message": f"Load kidney dataset from {RAW_H5}, {FILTERED_H5}, and {METADATA_CSV}"
    },
    {
        "name": "Compute QC metrics",
        "message": "Compute QC metrics for mouse kidney data"
    },
    {
        "name": "Generate QC violin plots",
        "message": "Generate QC violin plots showing pre-filtering distributions"
    },
    {
        "name": "Generate knee plot",
        "message": "Generate knee plot to identify ambient RNA threshold"
    },
    {
        "name": "Run SCAR denoising",
        "message": "Run SCAR ambient RNA removal using raw data"
    },
    {
        "name": "Detect doublets",
        "message": "Detect doublets using DoubletFinder with automatic pK optimization"
    },
    {
        "name": "Apply doublet filter",
        "message": "Remove detected doublets from the dataset"
    },
    {
        "name": "Quick graph analysis",
        "message": "Perform PCA, UMAP, and Leiden clustering"
    },
    {
        "name": "Detect marker genes",
        "message": "Find marker genes for each cluster"
    },
    {
        "name": "Annotate clusters",
        "message": "Annotate clusters with kidney cell types using built-in markers"
    },
]

# Run workflow
print(f"\n2. Running {len(steps)}-step kidney pipeline workflow...\n")

results = []
for i, step in enumerate(steps, 1):
    print(f"\n{'='*80}")
    print(f"STEP {i}/{len(steps)}: {step['name']}")
    print(f"{'='*80}")
    print(f"Message: {step['message']}\n")

    try:
        # Generate and execute plan
        result = agent.chat(step["message"], mode="execute")

        # Check for errors
        if "error" in str(result).lower() or "❌" in str(result.get("message", "")):
            print(f"⚠️  Step completed with errors:")
            print(result.get("message", result))
            results.append({"step": step["name"], "status": "error", "result": result})
        else:
            print(f"✅ Step completed successfully")
            results.append({"step": step["name"], "status": "success", "result": result})

    except Exception as e:
        print(f"❌ Step failed with exception: {e}")
        results.append({"step": step["name"], "status": "failed", "error": str(e)})

# Summary
print("\n" + "="*80)
print("WORKFLOW SUMMARY")
print("="*80)

successes = sum(1 for r in results if r["status"] == "success")
errors = sum(1 for r in results if r["status"] == "error")
failures = sum(1 for r in results if r["status"] == "failed")

print(f"\n✅ Successful: {successes}/{len(steps)}")
print(f"⚠️  Errors: {errors}/{len(steps)}")
print(f"❌ Failures: {failures}/{len(steps)}\n")

for i, result in enumerate(results, 1):
    status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "error" else "❌"
    print(f"{i}. {status_icon} {result['step']}: {result['status']}")

# Cleanup
print("\n" + "="*80)
if os.path.exists(STATE_FILE):
    print(f"State file saved: {STATE_FILE}")
    print("To inspect: cat {STATE_FILE} | python -m json.tool")
else:
    print("Note: State file not created")

print("\nTest complete!")
print("="*80)
