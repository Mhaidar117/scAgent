"""Telemetry collection for scQC Agent workflows."""

import json
import time
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
import threading
import importlib.util

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TelemetryCollector:
    """Collects and manages telemetry data for scQC workflows."""
    
    def __init__(self, run_id: str):
        """Initialize telemetry collector for a specific run.
        
        Args:
            run_id: Unique identifier for the workflow run
        """
        self.run_id = run_id
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.peak_memory_mb = 0.0
        self.start_memory_mb = 0.0
        self._lock = threading.Lock()
        
        # Initialize memory tracking
        if HAS_PSUTIL:
            process = psutil.Process()
            self.start_memory_mb = process.memory_info().rss / 1024 / 1024
            self.peak_memory_mb = self.start_memory_mb
    
    def start_step(self, step_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start timing a workflow step.
        
        Args:
            step_name: Name of the step being timed
            metadata: Optional metadata to record with the step
            
        Returns:
            Step ID for use with end_step()
        """
        step_id = f"{step_name}_{len(self.steps)}"
        
        step_data = {
            "step_id": step_id,
            "name": step_name,
            "start_time": time.time(),
            "start_timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.steps.append(step_data)
            
        # Update memory tracking
        self._update_memory_usage()
        
        return step_id
    
    def end_step(self, step_id: str, result_metadata: Optional[Dict[str, Any]] = None) -> None:
        """End timing for a workflow step.
        
        Args:
            step_id: Step ID returned from start_step()
            result_metadata: Optional metadata about the step results
        """
        end_time = time.time()
        end_timestamp = datetime.now().isoformat()
        
        with self._lock:
            # Find the step by ID
            for step in self.steps:
                if step.get("step_id") == step_id:
                    step["end_time"] = end_time
                    step["end_timestamp"] = end_timestamp
                    step["duration"] = end_time - step["start_time"]
                    if result_metadata:
                        step["result_metadata"] = result_metadata
                    break
        
        # Update memory tracking
        self._update_memory_usage()
    
    def _update_memory_usage(self) -> None:
        """Update peak memory usage if psutil is available."""
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def get_telemetry_data(self) -> Dict[str, Any]:
        """Get complete telemetry data for the current run.
        
        Returns:
            Dictionary containing all collected telemetry data
        """
        current_time = time.time()
        
        return {
            "run_id": self.run_id,
            "collection_timestamp": datetime.now().isoformat(),
            "total_runtime": current_time - self.start_time,
            "steps": list(self.steps),  # Copy to avoid race conditions
            "peak_memory_mb": self.peak_memory_mb,
            "start_memory_mb": self.start_memory_mb,
            "system_info": get_system_info(),
            "package_versions": get_package_versions(),
            "telemetry_version": "1.0"
        }
    
    def save_telemetry(self, output_path: Optional[str] = None) -> str:
        """Save telemetry data to a JSON file.
        
        Args:
            output_path: Custom output path (auto-generated if None)
            
        Returns:
            Path where telemetry was saved
        """
        if output_path is None:
            output_path = f"runs/{self.run_id}/telemetry.json"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        telemetry_data = self.get_telemetry_data()
        
        with open(output_path, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
        
        return output_path


# Global collector instance
_global_collector: Optional[TelemetryCollector] = None
_collector_lock = threading.Lock()


def get_global_collector() -> Optional[TelemetryCollector]:
    """Get the global telemetry collector instance."""
    with _collector_lock:
        return _global_collector


def initialize_telemetry(run_id: str) -> TelemetryCollector:
    """Initialize global telemetry collection for a run.
    
    Args:
        run_id: Unique identifier for the workflow run
        
    Returns:
        TelemetryCollector instance
    """
    global _global_collector
    
    with _collector_lock:
        _global_collector = TelemetryCollector(run_id)
        return _global_collector


def finalize_telemetry() -> Optional[str]:
    """Finalize and save telemetry data.
    
    Returns:
        Path where telemetry was saved, or None if no collector
    """
    global _global_collector
    
    with _collector_lock:
        if _global_collector:
            path = _global_collector.save_telemetry()
            _global_collector = None
            return path
        return None


@contextmanager
def record_step_timing(step_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for recording step timing.
    
    Args:
        step_name: Name of the step being timed
        metadata: Optional metadata to record with the step
        
    Usage:
        with record_step_timing("qc_compute", {"species": "human"}):
            # ... perform QC computation ...
            pass
    """
    collector = get_global_collector()
    
    if collector:
        step_id = collector.start_step(step_name, metadata)
        try:
            yield collector
        finally:
            collector.end_step(step_id)
    else:
        # No collector available, just yield None
        yield None


def get_system_info() -> Dict[str, Any]:
    """Get system information for telemetry.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": platform.node(),
    }
    
    if HAS_PSUTIL:
        info.update({
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "disk_usage_gb": psutil.disk_usage('.').total / 1024 / 1024 / 1024,
        })
    
    return info


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages for telemetry.
    
    Returns:
        Dictionary mapping package names to versions
    """
    packages_to_check = [
        "scanpy", "pandas", "numpy", "matplotlib", "seaborn",
        "scikit-learn", "scipy", "anndata", "scvi-tools", "torch",
        "scAR", "scrublet", "typer", "rich", "pydantic", "langchain"
    ]
    
    versions = {}
    
    for package in packages_to_check:
        try:
            if importlib.util.find_spec(package):
                module = importlib.import_module(package)
                if hasattr(module, "__version__"):
                    versions[package] = module.__version__
                elif hasattr(module, "version"):
                    versions[package] = module.version
                else:
                    versions[package] = "unknown"
        except (ImportError, AttributeError):
            continue
    
    return versions


def add_step_metadata(step_name: str, metadata: Dict[str, Any]) -> None:
    """Add metadata to the currently running step.
    
    Args:
        step_name: Name of the step to add metadata to
        metadata: Metadata to add
    """
    collector = get_global_collector()
    if collector:
        with collector._lock:
            # Find the most recent step with this name
            for step in reversed(collector.steps):
                if step.get("name") == step_name and "end_time" not in step:
                    step["metadata"].update(metadata)
                    break


def record_memory_checkpoint(label: str) -> None:
    """Record current memory usage with a label.
    
    Args:
        label: Label for this memory checkpoint
    """
    if not HAS_PSUTIL:
        return
    
    collector = get_global_collector()
    if collector:
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            
            # Add to the current step's metadata if there is one
            with collector._lock:
                if collector.steps:
                    current_step = collector.steps[-1]
                    if "memory_checkpoints" not in current_step["metadata"]:
                        current_step["metadata"]["memory_checkpoints"] = []
                    
                    current_step["metadata"]["memory_checkpoints"].append({
                        "label": label,
                        "memory_mb": current_memory,
                        "timestamp": datetime.now().isoformat()
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
