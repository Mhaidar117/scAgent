"""Graph analysis utilities for scRNA-seq data (DEPRECATED - moved to tools/)."""

# This file is deprecated - functionality moved to scqc_agent/tools/graph.py
from .tools.graph import quick_graph, recompute_neighbors

__all__ = ["quick_graph", "recompute_neighbors"]


