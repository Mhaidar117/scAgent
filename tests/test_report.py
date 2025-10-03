"""Tests for report generation functionality."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from scqc_agent.state import SessionState
from scqc_agent.reports.export import export_report


@pytest.fixture
def sample_state():
    """Create a sample session state for testing."""
    state = SessionState(run_id="test_report_session")
    
    # Add sample workflow history
    state.history = [
        {
            "step": 0,
            "label": "initial_load",
            "data_path": "test_data.h5ad",
            "checkpoint_path": "runs/test/step_00.h5ad",
            "timestamp": datetime.now().isoformat(),
            "artifacts": []
        },
        {
            "step": 1,
            "label": "qc_compute",
            "timestamp": datetime.now().isoformat(),
            "artifacts": [
                {"path": "test_qc_summary.csv", "label": "QC Summary"}
            ]
        },
        {
            "step": 2,
            "label": "qc_plot",
            "timestamp": datetime.now().isoformat(),
            "artifacts": [
                {"path": "test_qc_plot.png", "label": "QC Plot"}
            ]
        }
    ]
    
    # Add sample artifacts
    state.artifacts = {
        "test_qc_summary.csv": "QC Summary",
        "test_qc_plot.png": "QC Plot",
        "test_data_filtered.h5ad": "Filtered Data"
    }
    
    # Add sample metadata
    state.metadata = {
        "n_cells_initial": 1000,
        "n_genes_initial": 20000,
        "cells_after_qc": 950,
        "genes_after_qc": 18000,
        "species": "human"
    }
    
    return state


@pytest.fixture
def sample_telemetry_data():
    """Create sample telemetry data."""
    return {
        "run_id": "test_report_session",
        "collection_timestamp": datetime.now().isoformat(),
        "total_runtime": 45.67,
        "steps": [
            {
                "step_id": "qc_compute_0",
                "name": "qc_compute",
                "start_time": 1000.0,
                "end_time": 1015.5,
                "duration": 15.5,
                "metadata": {"species": "human"}
            },
            {
                "step_id": "qc_plot_1",
                "name": "qc_plot",
                "start_time": 1020.0,
                "end_time": 1025.2,
                "duration": 5.2,
                "metadata": {"stage": "pre"}
            }
        ],
        "peak_memory_mb": 256.7,
        "start_memory_mb": 128.3,
        "system_info": {
            "python_version": "3.9.0",
            "platform": "Linux-5.4.0",
            "cpu_count": 4
        },
        "package_versions": {
            "pandas": "1.3.0",
            "numpy": "1.21.0"
        }
    }


class TestReportExport:
    """Test report export functionality."""
    
    def test_html_report_generation(self, sample_state, tmp_path):
        """Test basic HTML report generation."""
        output_path = tmp_path / "test_report.html"
        
        result = export_report(
            state=sample_state,
            format="html",
            output_path=str(output_path),
            include_telemetry=False
        )
        
        # Check result
        assert not result.message.startswith("❌")
        assert len(result.artifacts) == 1
        assert str(output_path) in result.artifacts[0]
        
        # Check file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should be substantial
        
        # Check basic HTML structure
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "<!DOCTYPE html>" in content
        assert "scQC Agent Report" in content
        assert sample_state.run_id in content
        assert "test_report_session" in content
    
    def test_html_report_with_telemetry(self, sample_state, sample_telemetry_data, tmp_path):
        """Test HTML report generation with telemetry data."""
        # Create telemetry file
        telemetry_path = tmp_path / "runs" / sample_state.run_id / "telemetry.json"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(telemetry_path, 'w') as f:
            json.dump(sample_telemetry_data, f)
        
        output_path = tmp_path / "test_report_with_telemetry.html"
        
        # Change to tmp directory to find telemetry file
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            
            result = export_report(
                state=sample_state,
                format="html",
                output_path=str(output_path),
                include_telemetry=True
            )
        finally:
            os.chdir(original_cwd)
        
        # Check result
        assert not result.message.startswith("❌")
        assert output_path.exists()
        
        # Check telemetry is included
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "Performance Telemetry" in content or "telemetry" in content.lower()
        assert "45.67" in content or "45.7" in content  # Runtime
        assert "256.7" in content or "256" in content   # Memory
    
    def test_report_without_artifacts(self, tmp_path):
        """Test report generation with minimal state (no artifacts)."""
        state = SessionState(run_id="minimal_test")
        state.history = []
        state.artifacts = {}
        state.metadata = {}
        
        output_path = tmp_path / "minimal_report.html"
        
        result = export_report(
            state=state,
            format="html",
            output_path=str(output_path),
            include_telemetry=False
        )
        
        # Should still generate a report
        assert not result.message.startswith("❌")
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "minimal_test" in content
        assert "No workflow steps recorded yet" in content or "steps" in content.lower()
    
    def test_auto_generated_output_path(self, sample_state):
        """Test auto-generation of output path."""
        result = export_report(
            state=sample_state,
            format="html",
            output_path=None,  # Auto-generate
            include_telemetry=False
        )
        
        # Should generate a path
        assert not result.message.startswith("❌")
        assert len(result.artifacts) == 1
        
        generated_path = Path(result.artifacts[0])
        assert generated_path.name.startswith("scqc_report_")
        assert generated_path.suffix == ".html"
    
    def test_pdf_format_handling(self, sample_state, tmp_path):
        """Test PDF format handling (should fall back to HTML for now)."""
        output_path = tmp_path / "test_report.pdf"
        
        result = export_report(
            state=sample_state,
            format="pdf",
            output_path=str(output_path),
            include_telemetry=False
        )
        
        # Should mention PDF conversion requirements
        assert "PDF conversion requires" in result.message
        
        # Should generate HTML instead
        html_path = output_path.with_suffix('.html')
        assert html_path.exists()
    
    def test_report_statistics(self, sample_state, tmp_path):
        """Test report statistics calculation."""
        output_path = tmp_path / "stats_test.html"
        
        result = export_report(
            state=sample_state,
            format="html",
            output_path=str(output_path),
            include_telemetry=False
        )
        
        # Check statistics in state_delta
        assert "report_stats" in result.state_delta
        stats = result.state_delta["report_stats"]
        
        assert "n_sections" in stats
        assert "n_artifacts" in stats
        assert "n_steps" in stats
        assert "n_plots" in stats
        
        assert stats["n_artifacts"] == len(sample_state.artifacts)
        assert stats["n_steps"] == len(sample_state.history)
    
    def test_artifact_categorization(self, sample_state, tmp_path):
        """Test that artifacts are properly categorized in reports."""
        # Add more diverse artifacts
        sample_state.artifacts.update({
            "model.pkl": "Trained Model",
            "results.csv": "Analysis Results",
            "figure.svg": "Vector Plot",
            "report.pdf": "Analysis Report"
        })
        
        output_path = tmp_path / "categorized_test.html"
        
        result = export_report(
            state=sample_state,
            format="html",
            output_path=str(output_path),
            include_telemetry=False
        )
        
        assert not result.message.startswith("❌")
        
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have different artifact categories
        categories = ["Plots", "Data", "Models", "Reports"]
        found_categories = sum(1 for cat in categories if cat in content)
        assert found_categories >= 2  # Should find at least 2 categories
    
    def test_error_handling(self, sample_state):
        """Test error handling in report generation."""
        # Try to write to a directory that doesn't exist and can't be created
        invalid_path = "/invalid/path/that/does/not/exist/report.html"
        
        result = export_report(
            state=sample_state,
            format="html",
            output_path=invalid_path,
            include_telemetry=False
        )
        
        # Should handle the error gracefully
        assert result.message.startswith("❌")
        assert "report_error" in result.state_delta


class TestReportHelpers:
    """Test helper functions in the report module."""
    
    def test_metrics_extraction(self, sample_state):
        """Test key metrics extraction from state."""
        from scqc_agent.reports.export import _extract_key_metrics
        
        metrics = _extract_key_metrics(sample_state)
        
        # Should extract from metadata
        assert "n_cells_initial" in metrics
        assert "species" in metrics
        assert metrics["n_cells_initial"] == 1000
        assert metrics["species"] == "human"
    
    def test_artifact_categorization_helper(self, sample_state):
        """Test artifact categorization helper."""
        from scqc_agent.reports.export import _categorize_artifacts
        
        artifacts = {
            "plot.png": "Test Plot",
            "data.csv": "Test Data",
            "model.pkl": "Test Model",
            "unknown.xyz": "Unknown File"
        }
        
        categories = _categorize_artifacts(artifacts)
        
        assert "plots" in categories
        assert "data" in categories
        assert "models" in categories
        assert "other" in categories
        
        assert len(categories["plots"]) == 1
        assert len(categories["data"]) == 1
        assert len(categories["models"]) == 1
        assert len(categories["other"]) == 1
    
    def test_duration_calculation(self):
        """Test duration calculation helper."""
        from scqc_agent.reports.export import _calculate_duration
        
        start = "2025-09-25T10:00:00"
        end = "2025-09-25T10:05:30"
        
        duration = _calculate_duration(start, end)
        assert "5m 30s" in duration
        
        # Test error handling
        duration = _calculate_duration("invalid", "also-invalid")
        assert duration == "Unknown"


@pytest.mark.integration
class TestReportIntegration:
    """Integration tests for report functionality."""
    
    def test_full_report_workflow(self, sample_state, sample_telemetry_data, tmp_path):
        """Test complete report generation workflow."""
        # Setup telemetry
        telemetry_path = tmp_path / "runs" / sample_state.run_id / "telemetry.json"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(telemetry_path, 'w') as f:
            json.dump(sample_telemetry_data, f)
        
        # Create some fake plot files
        plot_dir = tmp_path / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        fake_plot = plot_dir / "test_plot.png"
        fake_plot.write_bytes(b"fake PNG data")
        
        # Update state to reference the plot
        sample_state.artifacts["test_plot.png"] = "Test Plot"
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            
            # Generate report
            result = export_report(
                state=sample_state,
                format="html",
                include_telemetry=True
            )
            
            # Verify comprehensive report
            assert not result.message.startswith("❌")
            report_path = Path(result.artifacts[0])
            assert report_path.exists()
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for all major sections
            expected_content = [
                "Workflow Overview",
                "Key Metrics",
                "Workflow Timeline",
                "Generated Artifacts",
                sample_state.run_id
            ]
            
            for expected in expected_content:
                assert expected in content, f"Missing: {expected}"
            
        finally:
            os.chdir(original_cwd)
