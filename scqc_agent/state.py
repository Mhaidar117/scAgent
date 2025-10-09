"""State management for scQC Agent sessions."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


def _make_json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable types to JSON-serializable equivalents.

    Args:
        obj: Any Python object

    Returns:
        JSON-serializable version of the object
    """
    # Convert numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    # Convert Path objects
    elif isinstance(obj, Path):
        return str(obj)
    # Recursively handle dicts
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    # Recursively handle lists
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    # Return as-is for standard JSON types
    return obj


@dataclass
class ToolResult:
    """Result from executing a tool in the scQC workflow.
    
    Attributes:
        message: Human-readable description of what happened
        state_delta: Changes to apply to the session state
        artifacts: List of file paths or Path objects generated
        citations: List of relevant scientific citations
    """
    message: str
    state_delta: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Union[str, Path]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message": self.message,
            "state_delta": self.state_delta,
            "artifacts": [str(p) for p in self.artifacts],
            "citations": self.citations
        }


class SessionState:
    """Manages persistent state for scQC workflow sessions.
    
    Tracks workflow history, checkpoints, artifacts, and current data state.
    """
    
    def __init__(self, run_id: Optional[str] = None):
        """Initialize a new session state.
        
        Args:
            run_id: Unique identifier for this session/run
        """
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history: List[Dict[str, Any]] = []
        self.artifacts: Dict[str, str] = {}  # path -> label mapping
        self.metadata: Dict[str, Any] = {}
        self.adata_path: Optional[str] = None
        self.dataset_summary: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        
    def checkpoint(self, data_path: str, label: str) -> str:
        """Create a checkpoint in the workflow history.

        Args:
            data_path: Path to the data file being checkpointed
            label: Human-readable label for this checkpoint

        Returns:
            Path to the checkpoint file
        """
        step_num = len(self.history)
        checkpoint_dir = f"runs/{self.run_id}/step_{step_num:02d}_{label}"
        checkpoint_path = f"{checkpoint_dir}/adata_step{step_num:02d}.h5ad"

        # Add to history - ensure path is string for JSON serialization
        history_entry = {
            "step": step_num,
            "label": label,
            "data_path": str(data_path),  # Convert Path to string
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "artifacts": []
        }
        self.history.append(history_entry)
        self.updated_at = datetime.now().isoformat()

        return checkpoint_path
    
    def add_artifact(self, path: str, label: str) -> None:
        """Add an artifact to the current session.

        Args:
            path: Path to the artifact file
            label: Human-readable label for the artifact
        """
        # Convert Path to string for JSON serialization
        path_str = str(path)
        self.artifacts[path_str] = label

        # Add to the most recent history entry if available
        if self.history:
            artifact_entry = {
                "path": path_str,
                "label": label,
                "timestamp": datetime.now().isoformat()
            }
            self.history[-1]["artifacts"].append(artifact_entry)

        self.updated_at = datetime.now().isoformat()
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update session metadata.

        Args:
            updates: Dictionary of metadata updates to apply
        """
        # Convert all values to JSON-serializable types
        serializable_updates = _make_json_serializable(updates)

        # Special handling for adata_path to normalize extensions
        if "adata_path" in serializable_updates:
            path_str = str(serializable_updates["adata_path"])
            # Fix double periods in extensions (e.g., .h5ad..gz -> .h5ad.gz)
            if ".." in path_str:
                path_str = path_str.replace("..", ".")
                serializable_updates["adata_path"] = path_str

        self.metadata.update(serializable_updates)
        self.updated_at = datetime.now().isoformat()
    
    def save(self, filepath: str) -> None:
        """Save session state to a JSON file.
        
        Args:
            filepath: Path where to save the state file
        """
        state_data = {
            "run_id": self.run_id,
            "history": self.history,
            "adata_path": self.adata_path,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
            "dataset_summary": self.dataset_summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then move to target (atomic operation)
        temp_filepath = f"{filepath}.tmp"
        try:
            with open(temp_filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Only replace the original if write was successful
            import os
            os.replace(temp_filepath, filepath)
        except Exception as e:
            # Clean up temp file if it exists
            if Path(temp_filepath).exists():
                Path(temp_filepath).unlink()
            raise e
    
    @classmethod
    def load(cls, filepath: str) -> 'SessionState':
        """Load session state from a JSON file.
        
        Args:
            filepath: Path to the state file to load
            
        Returns:
            SessionState instance with loaded data
        """
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Create new instance and populate
        state = cls(run_id=state_data.get("run_id"))
        state.history = state_data.get("history", [])
        state.artifacts = state_data.get("artifacts", {})
        state.metadata = state_data.get("metadata", {})
        state.adata_path = state_data.get("adata_path")
        state.dataset_summary = state_data.get("dataset_summary", {})
        state.created_at = state_data.get("created_at", datetime.now().isoformat())
        state.updated_at = state_data.get("updated_at", state.created_at)
        
        return state
    
    def __str__(self) -> str:
        """String representation of the session state."""
        return (
            f"SessionState(run_id='{self.run_id}', "
            f"history_entries={len(self.history)}, "
            f"artifacts={len(self.artifacts)})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
