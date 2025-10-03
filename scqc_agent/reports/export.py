"""Export comprehensive reports for scQC Agent workflows."""

import json
import shutil
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime
import base64

from ..state import SessionState, ToolResult


def export_report(
    state: SessionState,
    format: Literal["html", "pdf"] = "html",
    output_path: Optional[str] = None,
    include_telemetry: bool = True
) -> ToolResult:
    """Export a comprehensive report consolidating workflow results.
    
    Args:
        state: Session state containing workflow history and artifacts
        format: Export format ("html" or "pdf")
        output_path: Custom output path (auto-generated if None)
        include_telemetry: Whether to include telemetry data in report
        
    Returns:
        ToolResult with report path and summary
    """
    try:
        # Create report directory
        run_dir = Path(f"runs/{state.run_id}")
        report_dir = run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = report_dir / f"scqc_report_{timestamp}.{format}"
        else:
            output_path = Path(output_path)
        
        # Gather report data
        report_data = _gather_report_data(state, include_telemetry)
        
        # Generate report based on format
        if format == "html":
            _generate_html_report(report_data, output_path, report_dir)
        elif format == "pdf":
            # For now, generate HTML first then mention PDF conversion
            html_path = output_path.with_suffix('.html')
            _generate_html_report(report_data, html_path, report_dir)
            
            # Note: PDF generation would require additional dependencies like weasyprint
            message = (f"📊 HTML report generated: {html_path}\n"
                      f"💡 PDF conversion requires additional setup (weasyprint/puppeteer)")
            
            return ToolResult(
                message=message,
                artifacts=[str(html_path)],
                state_delta={"last_report_path": str(html_path), "report_format": "html"}
            )
        
        # Calculate report statistics
        stats = _calculate_report_stats(report_data)
        
        message = (f"📊 {format.upper()} report generated successfully!\n"
                  f"📁 Path: {output_path}\n"
                  f"📈 Sections: {stats['n_sections']}\n"
                  f"📄 Artifacts: {stats['n_artifacts']}\n"
                  f"📊 Steps: {stats['n_steps']}")
        
        return ToolResult(
            message=message,
            artifacts=[str(output_path)],
            state_delta={
                "last_report_path": str(output_path),
                "report_format": format,
                "report_stats": stats
            }
        )
        
    except Exception as e:
        return ToolResult(
            message=f"❌ Failed to generate {format} report: {str(e)}",
            state_delta={"report_error": str(e)}
        )


def _gather_report_data(state: SessionState, include_telemetry: bool) -> Dict[str, Any]:
    """Gather all data needed for the report."""
    
    # Load telemetry if available and requested
    telemetry_data = None
    if include_telemetry:
        telemetry_path = Path(f"runs/{state.run_id}/telemetry.json")
        if telemetry_path.exists():
            try:
                with open(telemetry_path, 'r') as f:
                    telemetry_data = json.load(f)
            except Exception:
                telemetry_data = None
    
    # Load QC metrics from artifacts
    qc_data = _load_qc_metrics(state)
    
    # Organize artifacts by type
    artifacts_by_type = _categorize_artifacts(state.artifacts)
    
    # Extract key metrics from metadata and history
    key_metrics = _extract_key_metrics(state)
    
    return {
        "state": state,
        "telemetry": telemetry_data,
        "qc_data": qc_data,
        "artifacts_by_type": artifacts_by_type,
        "key_metrics": key_metrics,
        "generated_at": datetime.now().isoformat(),
        "report_version": "1.0"
    }


def _load_qc_metrics(state: SessionState) -> Optional[Dict[str, Any]]:
    """Load QC metrics from artifacts."""
    run_dir = Path(f"runs/{state.run_id}")
    
    # Look for QC summary JSON files
    qc_files = []
    qc_files.extend(run_dir.glob("**/qc_summary.json"))
    qc_files.extend(run_dir.glob("**/*qc*.json"))
    
    qc_data = {}
    
    for qc_file in qc_files:
        try:
            with open(qc_file, 'r') as f:
                data = json.load(f)
                
            # Determine the type of QC data
            if "qc_metrics" in data:
                # This is a compute_qc_metrics output
                qc_data["compute_qc"] = {
                    "file_path": str(qc_file),
                    "timestamp": data.get("timestamp"),
                    "species": data.get("species"),
                    "mito_prefix": data.get("mito_prefix"),
                    "n_cells": data.get("n_cells"),
                    "n_genes": data.get("n_genes"),
                    "n_mito_genes": data.get("n_mito_genes"),
                    "metrics": data["qc_metrics"],
                    "per_batch": data.get("per_batch", {})
                }
            elif "filters_applied" in data:
                # This is a QC filtering output
                qc_data["qc_filters"] = {
                    "file_path": str(qc_file),
                    "filters": data
                }
            else:
                # Generic QC data
                qc_data[qc_file.stem] = {
                    "file_path": str(qc_file),
                    "data": data
                }
                
        except Exception as e:
            # Skip files that can't be parsed
            continue
    
    return qc_data if qc_data else None


def _categorize_artifacts(artifacts: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """Categorize artifacts by type for better report organization."""
    categories = {
        "plots": [],
        "data": [],
        "models": [],
        "metrics": [],
        "other": []
    }
    
    for path, label in artifacts.items():
        path_lower = path.lower()
        artifact_info = {"path": path, "label": label}
        
        if any(ext in path_lower for ext in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]):
            categories["plots"].append(artifact_info)
        elif any(ext in path_lower for ext in [".h5ad", ".csv", ".tsv", ".xlsx"]):
            categories["data"].append(artifact_info)
        elif any(ext in path_lower for ext in [".pkl", ".pt", ".h5", "model"]):
            categories["models"].append(artifact_info)
        elif "metric" in path_lower or "summary" in path_lower:
            categories["metrics"].append(artifact_info)
        else:
            categories["other"].append(artifact_info)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def _extract_key_metrics(state: SessionState) -> Dict[str, Any]:
    """Extract key metrics from state metadata and history."""
    metrics = {}
    
    # Extract from metadata
    if state.metadata:
        # Common QC metrics
        for key in ["n_cells_initial", "n_genes_initial", "cells_after_qc", "genes_after_qc",
                   "final_n_clusters", "doublets_removed", "n_batches"]:
            if key in state.metadata:
                metrics[key] = state.metadata[key]
        
        # Doublet detection metrics
        doublet_keys = [
            "doublet_method", "expected_doublet_rate", "detected_doublet_rate", 
            "n_doublets", "n_singlets", "doublet_threshold", "mean_doublet_score", 
            "median_doublet_score", "cells_before_doublet_filter", "cells_after_doublet_filter",
            "final_doublet_rate", "doublet_filter_applied", "doublet_filter_threshold"
        ]
        for key in doublet_keys:
            if key in state.metadata:
                metrics[key] = state.metadata[key]
    
    # Extract from history - look for state_delta in each step
    for entry in state.history:
        if "state_delta" in entry:
            delta = entry["state_delta"]
            for key, value in delta.items():
                if key not in metrics:  # Don't overwrite earlier values
                    metrics[key] = value
    
    # Calculate derived metrics
    if "n_cells_initial" in metrics and "cells_after_qc" in metrics:
        initial = metrics["n_cells_initial"]
        final = metrics["cells_after_qc"]
        if initial > 0:
            metrics["cell_retention_rate"] = final / initial
    
    # Add doublet rate interpretation if available
    if "expected_doublet_rate" in metrics:
        expected_rate = metrics["expected_doublet_rate"]
        if expected_rate == 0.06:
            metrics["doublet_rate_explanation"] = "Standard 10X rate (6%)"
        elif expected_rate == 0.08:
            metrics["doublet_rate_explanation"] = "High-throughput rate (8%)"
        else:
            metrics["doublet_rate_explanation"] = f"Custom rate ({expected_rate:.1%})"
    
    return metrics


def _generate_html_report(report_data: Dict[str, Any], output_path: Path, report_dir: Path) -> None:
    """Generate HTML report with embedded styling and plots."""
    
    # Copy and encode plot images
    plot_data = {}
    for plot_info in report_data["artifacts_by_type"].get("plots", []):
        plot_path = Path(plot_info["path"])
        if plot_path.exists():
            # Copy plot to report directory
            plot_copy = report_dir / plot_path.name
            shutil.copy2(plot_path, plot_copy)
            
            # Also encode for embedding if small enough
            if plot_path.stat().st_size < 1024 * 1024:  # < 1MB
                try:
                    with open(plot_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode()
                    plot_data[plot_info["label"]] = {
                        "path": str(plot_copy.relative_to(report_dir)),
                        "encoded": encoded,
                        "type": "image/png" if plot_path.suffix.lower() == ".png" else "image/jpeg"
                    }
                except Exception:
                    plot_data[plot_info["label"]] = {"path": str(plot_copy.relative_to(report_dir))}
    
    # Generate HTML content
    html_content = _generate_html_template(report_data, plot_data)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def _generate_html_template(report_data: Dict[str, Any], plot_data: Dict[str, Any]) -> str:
    """Generate the complete HTML report template."""
    
    state = report_data["state"]
    key_metrics = report_data["key_metrics"]
    telemetry = report_data.get("telemetry")
    qc_data = report_data.get("qc_data")
    
    # Generate workflow timeline
    timeline_html = _generate_timeline_html(state.history)
    
    # Generate metrics table
    metrics_html = _generate_metrics_table_html(key_metrics)
    
    # Generate QC metrics section
    qc_html = ""
    if qc_data:
        qc_html = _generate_qc_section_html(qc_data)
    
    # Generate plots section
    plots_html = _generate_plots_section_html(plot_data)
    
    # Generate artifacts section
    artifacts_html = _generate_artifacts_section_html(report_data["artifacts_by_type"])
    
    # Generate telemetry section if available
    telemetry_html = ""
    if telemetry:
        telemetry_html = _generate_telemetry_section_html(telemetry)
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>scQC Agent Report - {state.run_id}</title>
    <style>
        {_get_report_css()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🧬 scQC Agent Report</h1>
            <div class="header-info">
                <span class="run-id">Run ID: {state.run_id}</span>
                <span class="generated-at">Generated: {datetime.fromisoformat(report_data['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </header>
        
        <nav class="nav">
            <a href="#overview">Overview</a>
            <a href="#metrics">Key Metrics</a>
            {('<a href="#qc">QC Analysis</a>' if qc_data else '')}
            <a href="#timeline">Workflow Timeline</a>
            <a href="#plots">Visualizations</a>
            <a href="#artifacts">Artifacts</a>
            {('<a href="#telemetry">Performance</a>' if telemetry else '')}
        </nav>
        
        <main>
            <section id="overview" class="section">
                <h2>📊 Workflow Overview</h2>
                <div class="overview-grid">
                    <div class="overview-card">
                        <h3>Session Info</h3>
                        <p><strong>Run ID:</strong> {state.run_id}</p>
                        <p><strong>Created:</strong> {datetime.fromisoformat(state.created_at).strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Duration:</strong> {_calculate_duration(state.created_at, state.updated_at)}</p>
                    </div>
                    <div class="overview-card">
                        <h3>Pipeline Progress</h3>
                        <p><strong>Steps:</strong> {len(state.history)}</p>
                        <p><strong>Artifacts:</strong> {len(state.artifacts)}</p>
                        <p><strong>Status:</strong> {'Complete' if key_metrics.get('pipeline_complete') else 'In Progress'}</p>
                    </div>
                </div>
            </section>
            
            <section id="metrics" class="section">
                <h2>📈 Key Metrics</h2>
                {metrics_html}
            </section>
            
            {qc_html}
            
            <section id="timeline" class="section">
                <h2>🕐 Workflow Timeline</h2>
                {timeline_html}
            </section>
            
            <section id="plots" class="section">
                <h2>📊 Visualizations</h2>
                {plots_html}
            </section>
            
            <section id="artifacts" class="section">
                <h2>📁 Generated Artifacts</h2>
                {artifacts_html}
            </section>
            
            {telemetry_html}
        </main>
        
        <footer class="footer">
            <p>Generated by scQC Agent v{report_data['report_version']} | <a href="https://github.com/scqc-agent">GitHub</a></p>
        </footer>
    </div>
</body>
</html>
"""


def _get_report_css() -> str:
    """Return CSS styles for the HTML report."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header-info {
            display: flex;
            justify-content: center;
            gap: 2rem;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .nav {
            background: #2c3e50;
            padding: 1rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
        }
        
        .nav a {
            color: white;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        
        .nav a:hover {
            background-color: rgba(255,255,255,0.2);
        }
        
        main {
            padding: 2rem;
        }
        
        .section {
            margin-bottom: 3rem;
        }
        
        .section h2 {
            font-size: 1.8rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
            color: #2c3e50;
        }
        
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .overview-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .overview-card h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .metrics-table th,
        .metrics-table td {
            padding: 0.8rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .metrics-table th {
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }
        
        .metrics-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .metrics-table tr.section-header {
            background-color: #e8f4fd !important;
        }
        
        .metrics-table tr.section-header td {
            font-weight: 600;
            color: #2c3e50;
            border-top: 2px solid #3498db;
            border-bottom: 1px solid #3498db;
        }
        
        .timeline {
            position: relative;
            padding-left: 2rem;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #3498db;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 2rem;
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -2.5rem;
            top: 1.5rem;
            width: 12px;
            height: 12px;
            background: #3498db;
            border-radius: 50%;
            border: 3px solid white;
        }
        
        .timeline-item h4 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        
        .timeline-item .timestamp {
            font-size: 0.85rem;
            color: #7f8c8d;
            margin-bottom: 0.5rem;
        }
        
        .plots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-top: 1rem;
        }
        
        .plot-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .plot-container h4 {
            margin-bottom: 1rem;
            color: #2c3e50;
        }
        
        .artifacts-category {
            margin-bottom: 2rem;
        }
        
        .artifacts-category h3 {
            color: #3498db;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }
        
        .artifacts-list {
            list-style: none;
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
        }
        
        .artifacts-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .artifacts-list li:last-child {
            border-bottom: none;
        }
        
        .artifacts-list a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }
        
        .artifacts-list a:hover {
            text-decoration: underline;
        }
        
        .qc-overview {
            margin-bottom: 2rem;
        }
        
        .qc-metrics,
        .batch-metrics {
            margin-bottom: 2rem;
        }
        
        .qc-metrics h3,
        .batch-metrics h3 {
            color: #3498db;
            margin-bottom: 1rem;
            font-size: 1.3rem;
        }
        
        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        
        .telemetry-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
        }
        
        .telemetry-card h4 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .footer {
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 2rem;
        }
        
        .footer a {
            color: #3498db;
            text-decoration: none;
        }
        
        .footer a:hover {
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            .header-info {
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .nav {
                flex-direction: column;
                align-items: center;
            }
            
            .plots-grid {
                grid-template-columns: 1fr;
            }
            
            main {
                padding: 1rem;
            }
        }
    """


def _generate_timeline_html(history: List[Dict[str, Any]]) -> str:
    """Generate HTML for workflow timeline."""
    if not history:
        return "<p>No workflow steps recorded yet.</p>"
    
    timeline_items = []
    for entry in history:
        step = entry.get("step", 0)
        label = entry.get("label", "Unknown step")
        timestamp = entry.get("timestamp", "")
        artifacts = entry.get("artifacts", [])
        
        artifacts_html = ""
        if artifacts:
            artifacts_list = [f"<li>📄 {a.get('label', 'Artifact')}</li>" for a in artifacts]
            artifacts_html = f"<ul>{''.join(artifacts_list)}</ul>"
        
        formatted_time = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp
        
        timeline_items.append(f"""
            <div class="timeline-item">
                <h4>Step {step}: {label}</h4>
                <div class="timestamp">{formatted_time}</div>
                {artifacts_html}
            </div>
        """)
    
    return f'<div class="timeline">{"".join(timeline_items)}</div>'


def _generate_metrics_table_html(metrics: Dict[str, Any]) -> str:
    """Generate HTML table for key metrics."""
    if not metrics:
        return "<p>No metrics available.</p>"
    
    # Organize metrics into sections for better display
    sections = {
        "Cell Counts": ["n_cells_initial", "cells_after_qc", "cells_before_doublet_filter", 
                       "cells_after_doublet_filter", "cell_retention_rate"],
        "Doublet Detection": ["doublet_method", "expected_doublet_rate", "detected_doublet_rate", 
                             "doublet_rate_explanation", "doublet_threshold", "n_doublets", 
                             "n_singlets", "mean_doublet_score", "median_doublet_score", 
                             "doublets_removed", "final_doublet_rate", "doublet_filter_applied"],
        "Gene Metrics": ["n_genes_initial", "genes_after_qc"],
        "Analysis Results": ["final_n_clusters", "n_batches"],
    }
    
    # Custom display names for better readability
    display_names = {
        "n_cells_initial": "Initial Cell Count",
        "cells_after_qc": "Cells After QC",
        "cells_before_doublet_filter": "Cells Before Doublet Filter", 
        "cells_after_doublet_filter": "Cells After Doublet Filter",
        "cell_retention_rate": "Cell Retention Rate",
        "doublet_method": "Doublet Detection Method",
        "expected_doublet_rate": "Expected Doublet Rate",
        "detected_doublet_rate": "Detected Doublet Rate",
        "doublet_rate_explanation": "Expected Rate Basis",
        "doublet_threshold": "Doublet Score Threshold",
        "n_doublets": "Doublets Detected",
        "n_singlets": "Singlets Detected", 
        "mean_doublet_score": "Mean Doublet Score",
        "median_doublet_score": "Median Doublet Score",
        "doublets_removed": "Doublets Removed",
        "final_doublet_rate": "Final Doublet Rate",
        "doublet_filter_applied": "Doublet Filter Applied",
        "n_genes_initial": "Initial Gene Count",
        "genes_after_qc": "Genes After QC",
        "final_n_clusters": "Final Clusters",
        "n_batches": "Number of Batches"
    }
    
    all_rows = []
    
    for section_name, section_keys in sections.items():
        section_metrics = {k: v for k, v in metrics.items() if k in section_keys}
        if not section_metrics:
            continue
            
        # Add section header
        all_rows.append(f'<tr class="section-header"><td colspan="2"><strong>{section_name}</strong></td></tr>')
        
        for key, value in section_metrics.items():
            # Use custom display name or format the key
            display_key = display_names.get(key, key.replace("_", " ").title())
            
            # Format the value with special handling for specific metrics
            if key in ["expected_doublet_rate", "detected_doublet_rate", "final_doublet_rate", "cell_retention_rate"]:
                display_value = f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
            elif key in ["doublet_threshold", "mean_doublet_score", "median_doublet_score"]:
                display_value = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
            elif key == "doublet_method":
                display_value = value.title() if isinstance(value, str) else str(value)
            elif isinstance(value, float):
                if 0 < value < 1:
                    display_value = f"{value:.1%}"
                else:
                    display_value = f"{value:.3f}"
            elif isinstance(value, bool):
                display_value = "✅ Yes" if value else "❌ No"
            elif isinstance(value, (int, float)) and value > 1000:
                display_value = f"{value:,}"  # Add comma separators for large numbers
            else:
                display_value = str(value)
            
            all_rows.append(f"<tr><td>{display_key}</td><td>{display_value}</td></tr>")
    
    # Add any remaining metrics that weren't categorized
    remaining_metrics = {k: v for k, v in metrics.items() 
                        if not any(k in section_keys for section_keys in sections.values())}
    if remaining_metrics:
        all_rows.append('<tr class="section-header"><td colspan="2"><strong>Other Metrics</strong></td></tr>')
        for key, value in remaining_metrics.items():
            display_key = key.replace("_", " ").title()
            if isinstance(value, float):
                if 0 < value < 1:
                    display_value = f"{value:.1%}"
                else:
                    display_value = f"{value:.3f}"
            elif isinstance(value, bool):
                display_value = "✅ Yes" if value else "❌ No"
            else:
                display_value = str(value)
            all_rows.append(f"<tr><td>{display_key}</td><td>{display_value}</td></tr>")
    
    return f"""
        <table class="metrics-table">
            <thead>
                <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
                {"".join(all_rows)}
            </tbody>
        </table>
    """


def _generate_plots_section_html(plot_data: Dict[str, Any]) -> str:
    """Generate HTML for plots section."""
    if not plot_data:
        return "<p>No plots generated yet.</p>"
    
    plot_items = []
    for label, info in plot_data.items():
        if "encoded" in info:
            # Embed the plot
            img_src = f"data:{info['type']};base64,{info['encoded']}"
        else:
            # Link to the plot file
            img_src = info["path"]
        
        plot_items.append(f"""
            <div class="plot-container">
                <h4>{label}</h4>
                <img src="{img_src}" alt="{label}">
            </div>
        """)
    
    return f'<div class="plots-grid">{"".join(plot_items)}</div>'


def _generate_artifacts_section_html(artifacts_by_type: Dict[str, List[Dict[str, str]]]) -> str:
    """Generate HTML for artifacts section."""
    if not artifacts_by_type:
        return "<p>No artifacts generated yet.</p>"
    
    sections = []
    for category, artifacts in artifacts_by_type.items():
        category_title = category.replace("_", " ").title()
        artifact_items = []
        
        for artifact in artifacts:
            path = artifact["path"]
            label = artifact["label"]
            artifact_items.append(f'<li><a href="{path}" target="_blank">{label}</a><span>{Path(path).name}</span></li>')
        
        sections.append(f"""
            <div class="artifacts-category">
                <h3>📁 {category_title}</h3>
                <ul class="artifacts-list">
                    {"".join(artifact_items)}
                </ul>
            </div>
        """)
    
    return "".join(sections)


def _generate_telemetry_section_html(telemetry: Dict[str, Any]) -> str:
    """Generate HTML for telemetry section."""
    cards = []
    
    # Timing information
    if "steps" in telemetry:
        total_time = sum(step.get("duration", 0) for step in telemetry["steps"])
        cards.append(f"""
            <div class="telemetry-card">
                <h4>⏱️ Timing</h4>
                <p><strong>Total Runtime:</strong> {total_time:.2f}s</p>
                <p><strong>Steps:</strong> {len(telemetry["steps"])}</p>
                <p><strong>Avg per Step:</strong> {total_time / len(telemetry["steps"]):.2f}s</p>
            </div>
        """)
    
    # Memory information
    if "peak_memory_mb" in telemetry:
        cards.append(f"""
            <div class="telemetry-card">
                <h4>💾 Memory</h4>
                <p><strong>Peak Usage:</strong> {telemetry['peak_memory_mb']:.1f} MB</p>
                <p><strong>Start Usage:</strong> {telemetry.get('start_memory_mb', 0):.1f} MB</p>
            </div>
        """)
    
    # System information
    if "system_info" in telemetry:
        sys_info = telemetry["system_info"]
        cards.append(f"""
            <div class="telemetry-card">
                <h4>🖥️ System</h4>
                <p><strong>Python:</strong> {sys_info.get('python_version', 'Unknown')}</p>
                <p><strong>Platform:</strong> {sys_info.get('platform', 'Unknown')}</p>
                <p><strong>CPU Count:</strong> {sys_info.get('cpu_count', 'Unknown')}</p>
            </div>
        """)
    
    # Package versions
    if "package_versions" in telemetry:
        versions = telemetry["package_versions"]
        version_list = [f"<li>{pkg}: {ver}</li>" for pkg, ver in versions.items()]
        cards.append(f"""
            <div class="telemetry-card">
                <h4>📦 Packages</h4>
                <ul style="margin: 0; padding-left: 1.2rem;">
                    {"".join(version_list[:5])}
                    {('<li>... and more</li>' if len(version_list) > 5 else '')}
                </ul>
            </div>
        """)
    
    if not cards:
        return ""
    
    return f"""
        <section id="telemetry" class="section">
            <h2>⚡ Performance Telemetry</h2>
            <div class="telemetry-grid">
                {"".join(cards)}
            </div>
        </section>
    """


def _calculate_duration(start_time: str, end_time: str) -> str:
    """Calculate human-readable duration between two timestamps."""
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        delta = end - start
        
        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}m {delta.seconds % 60}s"
        else:
            return f"{delta.seconds}s"
    except:
        return "Unknown"


def _generate_qc_section_html(qc_data: Dict[str, Any]) -> str:
    """Generate HTML for the QC analysis section."""
    if not qc_data:
        return ""
    
    # Start building the QC section
    section_html = """
            <section id="qc" class="section">
                <h2>🔬 Quality Control Analysis</h2>
    """
    
    # Add compute QC results if available
    if "compute_qc" in qc_data:
        compute_qc = qc_data["compute_qc"]
        section_html += _generate_compute_qc_html(compute_qc)
    
    # Add QC filters if available
    if "qc_filters" in qc_data:
        filters = qc_data["qc_filters"]
        section_html += _generate_qc_filters_html(filters)
    
    section_html += """
            </section>
    """
    
    return section_html


def _generate_compute_qc_html(compute_qc: Dict[str, Any]) -> str:
    """Generate HTML for compute QC metrics."""
    metrics = compute_qc.get("metrics", {})
    
    html = f"""
                <div class="qc-overview">
                    <h3>🧬 Dataset Overview</h3>
                    <div class="overview-grid">
                        <div class="overview-card">
                            <h4>Dataset Info</h4>
                            <p><strong>Species:</strong> {compute_qc.get('species', 'Unknown')}</p>
                            <p><strong>Cells:</strong> {compute_qc.get('n_cells', 'N/A'):,}</p>
                            <p><strong>Genes:</strong> {compute_qc.get('n_genes', 'N/A'):,}</p>
                            <p><strong>Mitochondrial Genes:</strong> {compute_qc.get('n_mito_genes', 'N/A')}</p>
                        </div>
                        <div class="overview-card">
                            <h4>QC Timestamp</h4>
                            <p><strong>Computed:</strong> {_format_timestamp(compute_qc.get('timestamp'))}</p>
                            <p><strong>Mito Prefix:</strong> {compute_qc.get('mito_prefix', 'N/A')}</p>
                        </div>
                    </div>
                </div>
    """
    
    if metrics:
        html += """
                <div class="qc-metrics">
                    <h3>📊 QC Metrics</h3>
                    <table class="metrics-table">
                        <thead>
                            <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std Dev</th></tr>
                        </thead>
                        <tbody>
        """
        
        # Add gene counts metrics
        if "n_genes_by_counts" in metrics:
            gene_metrics = metrics["n_genes_by_counts"]
            html += f"""
                            <tr>
                                <td>🧬 Genes per Cell</td>
                                <td>{gene_metrics.get('mean', 0):.1f}</td>
                                <td>{gene_metrics.get('median', 0):.1f}</td>
                                <td>{gene_metrics.get('std', 0):.1f}</td>
                            </tr>
            """
        
        # Add total counts metrics
        if "total_counts" in metrics:
            count_metrics = metrics["total_counts"]
            html += f"""
                            <tr>
                                <td>📊 Total Counts per Cell</td>
                                <td>{count_metrics.get('mean', 0):.1f}</td>
                                <td>{count_metrics.get('median', 0):.1f}</td>
                                <td>{count_metrics.get('std', 0):.1f}</td>
                            </tr>
            """
        
        # Add mitochondrial percentage metrics
        if "pct_counts_mt" in metrics:
            mt_metrics = metrics["pct_counts_mt"]
            html += f"""
                            <tr>
                                <td>🔋 Mitochondrial %</td>
                                <td>{mt_metrics.get('mean', 0):.2f}%</td>
                                <td>{mt_metrics.get('median', 0):.2f}%</td>
                                <td>{mt_metrics.get('std', 0):.2f}%</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
        """
    
    # Add per-batch analysis if available
    per_batch = compute_qc.get("per_batch", {})
    if per_batch:
        html += """
                <div class="batch-metrics">
                    <h3>🔢 Per-Batch Analysis</h3>
                    <table class="metrics-table">
                        <thead>
                            <tr><th>Batch</th><th>Cells</th><th>Mean Genes</th><th>Mean Counts</th><th>Mean MT%</th></tr>
                        </thead>
                        <tbody>
        """
        
        for batch_id, batch_data in per_batch.items():
            html += f"""
                            <tr>
                                <td><strong>{batch_id}</strong></td>
                                <td>{batch_data.get('n_cells', 0):,}</td>
                                <td>{batch_data.get('mean_genes', 0):.1f}</td>
                                <td>{batch_data.get('mean_counts', 0):.1f}</td>
                                <td>{batch_data.get('mean_pct_mt', 0):.2f}%</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
        """
    
    return html


def _generate_qc_filters_html(filters: Dict[str, Any]) -> str:
    """Generate HTML for QC filters applied."""
    # This would be implemented when QC filtering data structure is available
    return ""


def _format_timestamp(timestamp: Optional[str]) -> str:
    """Format ISO timestamp for display."""
    if not timestamp:
        return "Unknown"
    
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp


def _calculate_report_stats(report_data: Dict[str, Any]) -> Dict[str, int]:
    """Calculate statistics about the generated report."""
    stats = {
        "n_sections": 4,  # Base sections: overview, metrics, timeline, artifacts
        "n_artifacts": len(report_data["state"].artifacts),
        "n_steps": len(report_data["state"].history),
        "n_plots": len(report_data["artifacts_by_type"].get("plots", [])),
    }
    
    # Add QC section if present
    if report_data.get("qc_data"):
        stats["n_sections"] += 1
        
    # Add telemetry section if present
    if report_data.get("telemetry"):
        stats["n_sections"] += 1
    
    return stats
