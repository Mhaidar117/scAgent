"""scQC Agent - A runtime agent for scRNA-seq QC workflows via natural language."""

__version__ = "0.1.0"

from .state import SessionState, ToolResult

__all__ = ["SessionState", "ToolResult", "__version__"]

