"""Tests for telemetry functionality."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scqc_agent.utils.telemetry import (
    TelemetryCollector,
    initialize_telemetry,
    finalize_telemetry,
    get_global_collector,
    record_step_timing,
    get_system_info,
    get_package_versions,
    add_step_metadata,
    record_memory_checkpoint
)


class TestTelemetryCollector:
    """Test TelemetryCollector class."""
    
    def test_initialization(self):
        """Test TelemetryCollector initialization."""
        collector = TelemetryCollector("test_run")
        
        assert collector.run_id == "test_run"
        assert collector.steps == []
        assert collector.start_time > 0
        assert collector.peak_memory_mb >= 0
        assert collector.start_memory_mb >= 0
    
    def test_step_timing(self):
        """Test step timing functionality."""
        collector = TelemetryCollector("test_run")
        
        # Start a step
        step_id = collector.start_step("test_step", {"param": "value"})
        assert step_id == "test_step_0"
        assert len(collector.steps) == 1
        
        step = collector.steps[0]
        assert step["name"] == "test_step"
        assert step["metadata"]["param"] == "value"
        assert "start_time" in step
        assert "start_timestamp" in step
        assert "end_time" not in step
        
        # End the step
        time.sleep(0.01)  # Small delay to ensure duration > 0
        collector.end_step(step_id, {"result": "success"})
        
        step = collector.steps[0]
        assert "end_time" in step
        assert "end_timestamp" in step
        assert "duration" in step
        assert step["duration"] > 0
        assert step["result_metadata"]["result"] == "success"
    
    def test_multiple_steps(self):
        """Test handling multiple steps."""
        collector = TelemetryCollector("test_run")
        
        # Start multiple steps
        step1_id = collector.start_step("step1")
        step2_id = collector.start_step("step2")
        
        assert len(collector.steps) == 2
        assert step1_id == "step1_0"
        assert step2_id == "step2_1"
        
        # End steps in different order
        collector.end_step(step2_id)
        collector.end_step(step1_id)
        
        # Both should be completed
        assert "duration" in collector.steps[0]
        assert "duration" in collector.steps[1]
    
    @patch('scqc_agent.utils.telemetry.HAS_PSUTIL', True)
    @patch('scqc_agent.utils.telemetry.psutil')
    def test_memory_tracking(self, mock_psutil):
        """Test memory usage tracking."""
        # Mock psutil process
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process
        
        collector = TelemetryCollector("test_run")
        
        # Should have initialized memory values
        assert collector.start_memory_mb == 100.0
        assert collector.peak_memory_mb == 100.0
        
        # Simulate memory increase
        mock_memory_info.rss = 150 * 1024 * 1024  # 150 MB
        collector._update_memory_usage()
        
        assert collector.peak_memory_mb == 150.0
    
    def test_telemetry_data_export(self):
        """Test getting complete telemetry data."""
        collector = TelemetryCollector("test_run")
        
        # Add some steps
        step_id = collector.start_step("test_step")
        collector.end_step(step_id)
        
        data = collector.get_telemetry_data()
        
        # Check structure
        assert data["run_id"] == "test_run"
        assert "collection_timestamp" in data
        assert "total_runtime" in data
        assert "steps" in data
        assert "peak_memory_mb" in data
        assert "start_memory_mb" in data
        assert "system_info" in data
        assert "package_versions" in data
        assert data["telemetry_version"] == "1.0"
        
        # Check step data
        assert len(data["steps"]) == 1
        assert data["steps"][0]["name"] == "test_step"
    
    def test_save_telemetry(self, tmp_path):
        """Test saving telemetry to file."""
        collector = TelemetryCollector("test_run")
        
        # Add a step
        step_id = collector.start_step("test_step")
        collector.end_step(step_id)
        
        # Save to custom path
        output_path = tmp_path / "custom_telemetry.json"
        saved_path = collector.save_telemetry(str(output_path))
        
        assert saved_path == str(output_path)
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert data["run_id"] == "test_run"
        assert len(data["steps"]) == 1
    
    def test_auto_path_generation(self, tmp_path):
        """Test automatic path generation for telemetry."""
        collector = TelemetryCollector("test_run")
        
        # Change to tmp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            
            saved_path = collector.save_telemetry()
            expected_path = tmp_path / "runs" / "test_run" / "telemetry.json"
            
            assert saved_path == str(expected_path)
            assert expected_path.exists()
            
        finally:
            os.chdir(original_cwd)


class TestGlobalTelemetry:
    """Test global telemetry management."""
    
    def setup_method(self):
        """Reset global state before each test."""
        # Clear any existing global collector
        import scqc_agent.utils.telemetry as telemetry_module
        with telemetry_module._collector_lock:
            telemetry_module._global_collector = None
    
    def test_initialize_and_get_collector(self):
        """Test initializing and getting global collector."""
        # Initially no collector
        assert get_global_collector() is None
        
        # Initialize
        collector = initialize_telemetry("global_test")
        assert collector.run_id == "global_test"
        
        # Should be able to get it
        retrieved = get_global_collector()
        assert retrieved is collector
    
    def test_finalize_telemetry(self, tmp_path):
        """Test finalizing telemetry."""
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            
            # Initialize and add a step
            collector = initialize_telemetry("finalize_test")
            step_id = collector.start_step("test_step")
            collector.end_step(step_id)
            
            # Finalize
            saved_path = finalize_telemetry()
            
            expected_path = tmp_path / "runs" / "finalize_test" / "telemetry.json"
            assert saved_path == str(expected_path)
            assert expected_path.exists()
            
            # Global collector should be cleared
            assert get_global_collector() is None
            
        finally:
            os.chdir(original_cwd)
    
    def test_record_step_timing_context_manager(self):
        """Test the record_step_timing context manager."""
        initialize_telemetry("context_test")
        
        # Use context manager
        with record_step_timing("context_step", {"test": "value"}) as collector:
            assert collector is not None
            time.sleep(0.01)  # Small delay
        
        # Check that step was recorded
        global_collector = get_global_collector()
        assert len(global_collector.steps) == 1
        step = global_collector.steps[0]
        assert step["name"] == "context_step"
        assert step["metadata"]["test"] == "value"
        assert "duration" in step
        assert step["duration"] > 0
    
    def test_record_step_timing_no_collector(self):
        """Test context manager when no global collector exists."""
        # No collector initialized
        assert get_global_collector() is None
        
        # Should not raise error
        with record_step_timing("no_collector_step") as collector:
            assert collector is None
    
    def test_add_step_metadata(self):
        """Test adding metadata to running steps."""
        collector = initialize_telemetry("metadata_test")
        
        # Start a step
        step_id = collector.start_step("metadata_step")
        
        # Add metadata
        add_step_metadata("metadata_step", {"added": "metadata"})
        
        # Check metadata was added
        step = collector.steps[0]
        assert step["metadata"]["added"] == "metadata"
        
        # End step
        collector.end_step(step_id)
    
    @patch('scqc_agent.utils.telemetry.HAS_PSUTIL', True)
    @patch('scqc_agent.utils.telemetry.psutil')
    def test_record_memory_checkpoint(self, mock_psutil):
        """Test recording memory checkpoints."""
        # Mock psutil
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 200 * 1024 * 1024  # 200 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process
        
        collector = initialize_telemetry("memory_test")
        
        # Start a step
        step_id = collector.start_step("memory_step")
        
        # Record memory checkpoint
        record_memory_checkpoint("checkpoint1")
        
        # Check checkpoint was recorded
        step = collector.steps[0]
        assert "memory_checkpoints" in step["metadata"]
        checkpoints = step["metadata"]["memory_checkpoints"]
        assert len(checkpoints) == 1
        assert checkpoints[0]["label"] == "checkpoint1"
        assert checkpoints[0]["memory_mb"] == 200.0
        
        collector.end_step(step_id)


class TestSystemInfo:
    """Test system information collection."""
    
    def test_get_system_info(self):
        """Test system information collection."""
        info = get_system_info()
        
        # Should have basic fields
        required_fields = ["python_version", "platform", "architecture", "processor", "hostname"]
        for field in required_fields:
            assert field in info
            assert info[field] is not None
        
        # Python version should be valid format
        assert "." in info["python_version"]
    
    @patch('scqc_agent.utils.telemetry.HAS_PSUTIL', True)
    @patch('scqc_agent.utils.telemetry.psutil')
    def test_get_system_info_with_psutil(self, mock_psutil):
        """Test system info with psutil available."""
        # Mock psutil functions
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_psutil.disk_usage.return_value.total = 500 * 1024 * 1024 * 1024  # 500 GB
        
        info = get_system_info()
        
        # Should have psutil fields
        assert "cpu_count" in info
        assert "memory_total_gb" in info
        assert "disk_usage_gb" in info
        
        assert info["cpu_count"] == 8
        assert info["memory_total_gb"] == 16.0
        assert info["disk_usage_gb"] == 500.0
    
    def test_get_package_versions(self):
        """Test package version collection."""
        versions = get_package_versions()
        
        # Should be a dictionary
        assert isinstance(versions, dict)
        
        # Should have some common packages (at least the ones we use in tests)
        # Note: Actual packages depend on test environment
        expected_packages = ["pytest"]  # pytest should be available in test environment
        
        for package in expected_packages:
            if package in versions:
                assert isinstance(versions[package], str)
                assert len(versions[package]) > 0


class TestTelemetryThreadSafety:
    """Test thread safety of telemetry operations."""
    
    def test_concurrent_steps(self):
        """Test concurrent step recording."""
        collector = TelemetryCollector("concurrent_test")
        
        def record_steps(step_prefix, count):
            for i in range(count):
                step_id = collector.start_step(f"{step_prefix}_{i}")
                time.sleep(0.001)  # Small delay
                collector.end_step(step_id)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=record_steps, args=(f"thread{i}", 5))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have recorded all steps
        assert len(collector.steps) == 15  # 3 threads * 5 steps each
        
        # All steps should be completed
        for step in collector.steps:
            assert "duration" in step
    
    def test_global_collector_thread_safety(self):
        """Test thread-safe access to global collector."""
        def init_and_use():
            collector = initialize_telemetry(f"thread_test_{threading.current_thread().ident}")
            step_id = collector.start_step("thread_step")
            time.sleep(0.001)
            collector.end_step(step_id)
            return collector.run_id
        
        # Multiple threads trying to initialize
        threads = []
        results = []
        
        def thread_worker():
            result = init_and_use()
            results.append(result)
        
        for _ in range(3):
            thread = threading.Thread(target=thread_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        # Note: Due to global state, only the last one will remain
        assert len(results) == 3


@pytest.mark.integration
class TestTelemetryIntegration:
    """Integration tests for telemetry functionality."""
    
    def test_full_telemetry_workflow(self, tmp_path):
        """Test complete telemetry workflow."""
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            
            # Initialize telemetry
            collector = initialize_telemetry("integration_test")
            
            # Record multiple steps with different patterns
            with record_step_timing("step1", {"param": "value1"}):
                time.sleep(0.01)
                record_memory_checkpoint("mid_step1")
            
            step2_id = collector.start_step("step2", {"param": "value2"})
            add_step_metadata("step2", {"additional": "data"})
            collector.end_step(step2_id, {"result": "success"})
            
            # Finalize and save
            saved_path = finalize_telemetry()
            
            # Verify saved file
            assert Path(saved_path).exists()
            
            with open(saved_path, 'r') as f:
                data = json.load(f)
            
            # Check data integrity
            assert data["run_id"] == "integration_test"
            assert len(data["steps"]) == 2
            
            # Check step1
            step1 = data["steps"][0]
            assert step1["name"] == "step1"
            assert step1["metadata"]["param"] == "value1"
            assert "duration" in step1
            
            # Check step2
            step2 = data["steps"][1]
            assert step2["name"] == "step2"
            assert step2["metadata"]["param"] == "value2"
            assert step2["metadata"]["additional"] == "data"
            assert step2["result_metadata"]["result"] == "success"
            
        finally:
            os.chdir(original_cwd)
