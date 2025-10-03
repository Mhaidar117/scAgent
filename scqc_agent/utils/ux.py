"""UX utilities for enhanced user experience (Phase 8)."""

import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Iterator, Callable
from pathlib import Path

try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Confirm, Prompt
    from rich.live import Live
    from rich.layout import Layout
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..state import SessionState, ToolResult


class ProgressTracker:
    """Enhanced progress tracking with rich display."""
    
    def __init__(self, console: Optional[object] = None):
        """Initialize progress tracker.
        
        Args:
            console: Rich console instance (optional)
        """
        if RICH_AVAILABLE and console is None:
            self.console = Console()
        else:
            self.console = console
        
        self.progress: Optional[object] = None
        self.task_id: Optional[object] = None
        self._active = False
    
    @contextmanager
    def track_operation(self, description: str, total: Optional[int] = None) -> Iterator[Callable[[int, str], None]]:
        """Context manager for tracking operation progress.
        
        Args:
            description: Description of the operation
            total: Total number of steps (None for indeterminate)
            
        Yields:
            Update function: update(steps_completed, status_message)
        """
        if not RICH_AVAILABLE or not self.console:
            # Fallback for non-rich environments
            print(f"Starting: {description}")
            yield self._fallback_update
            print(f"Completed: {description}")
            return
        
        try:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn() if total else "",
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%") if total else "",
                TimeElapsedColumn(),
                TimeRemainingColumn() if total else "",
                console=self.console
            )
            
            self._active = True
            
            with self.progress:
                self.task_id = self.progress.add_task(
                    description, 
                    total=total if total else 100
                )
                
                def update_progress(steps_completed: int = 1, status: str = "") -> None:
                    """Update progress tracker."""
                    if self.progress and self.task_id is not None:
                        if total:
                            self.progress.update(self.task_id, advance=steps_completed)
                        else:
                            # For indeterminate progress, just update the description
                            self.progress.update(
                                self.task_id, 
                                description=f"{description}: {status}" if status else description
                            )
                
                yield update_progress
                
        finally:
            self._active = False
            self.progress = None
            self.task_id = None
    
    def _fallback_update(self, steps: int = 1, status: str = "") -> None:
        """Fallback update for non-rich environments."""
        if status:
            print(f"  {status}")


class ErrorHandler:
    """Enhanced error handling with user-friendly messages."""
    
    def __init__(self, console: Optional[object] = None):
        """Initialize error handler.
        
        Args:
            console: Rich console instance (optional)
        """
        if RICH_AVAILABLE and console is None:
            self.console = Console()
        else:
            self.console = console
    
    def handle_error(self, error: Exception, context: str = "", suggestions: Optional[List[str]] = None) -> None:
        """Display user-friendly error message.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            suggestions: List of suggested solutions
        """
        if not RICH_AVAILABLE or not self.console:
            self._fallback_error_display(error, context, suggestions)
            return
        
        # Create error panel
        error_msg = str(error)
        
        # Enhance error message based on common patterns
        enhanced_msg, enhanced_suggestions = self._enhance_error_message(error, error_msg)
        
        if suggestions:
            enhanced_suggestions.extend(suggestions)
        
        # Build panel content
        content = f"[red]Error:[/red] {enhanced_msg}"
        if context:
            content += f"\n[yellow]Context:[/yellow] {context}"
        
        if enhanced_suggestions:
            content += "\n\n[blue]Suggestions:[/blue]"
            for i, suggestion in enumerate(enhanced_suggestions, 1):
                content += f"\n  {i}. {suggestion}"
        
        self.console.print(Panel(
            content,
            title="❌ Error Occurred",
            border_style="red",
            expand=False
        ))
    
    def _enhance_error_message(self, error: Exception, error_msg: str) -> tuple[str, List[str]]:
        """Enhance error message with user-friendly explanations."""
        suggestions = []
        
        # ImportError patterns
        if isinstance(error, ImportError):
            if "scanpy" in error_msg.lower():
                error_msg = "Scanpy is not installed or not available"
                suggestions.extend([
                    "Install scanpy: pip install 'scqc-agent[qc]'",
                    "Ensure you're in the correct virtual environment"
                ])
            elif "torch" in error_msg.lower():
                error_msg = "PyTorch is not installed (required for scVI)"
                suggestions.extend([
                    "Install PyTorch: pip install 'scqc-agent[models]'",
                    "Check PyTorch installation instructions for your system"
                ])
            elif "langchain" in error_msg.lower():
                error_msg = "LangChain is not installed (required for agent features)"
                suggestions.extend([
                    "Install LangChain: pip install 'scqc-agent[agent]'",
                    "Use basic commands without agent features"
                ])
        
        # FileNotFoundError patterns
        elif isinstance(error, FileNotFoundError):
            if ".h5ad" in error_msg:
                error_msg = "AnnData file not found"
                suggestions.extend([
                    "Check if the file path is correct",
                    "Use 'scqc load --path <file>' to load data first",
                    "Ensure the file has .h5ad extension"
                ])
            elif "state" in error_msg.lower():
                error_msg = "Session state file not found"
                suggestions.extend([
                    "Initialize a session: scqc init",
                    "Check if you're in the correct directory",
                    "Specify state path: --state-path <path>"
                ])
        
        # ValueError patterns
        elif isinstance(error, ValueError):
            if "batch" in error_msg.lower():
                error_msg = "Invalid batch configuration"
                suggestions.extend([
                    "Check batch key exists in adata.obs",
                    "Ensure batch key contains valid batch identifiers",
                    "Use 'scqc state show' to see available metadata"
                ])
            elif "species" in error_msg.lower():
                error_msg = "Invalid species specification"
                suggestions.extend([
                    "Use 'human', 'mouse', or 'other' for species",
                    "Check species detection in QC metrics"
                ])
        
        # Memory/Resource errors
        elif isinstance(error, MemoryError):
            error_msg = "Insufficient memory for operation"
            suggestions.extend([
                "Try subsetting the data first",
                "Use lower resolution parameters",
                "Consider running on a machine with more RAM"
            ])
        
        return error_msg, suggestions
    
    def _fallback_error_display(self, error: Exception, context: str, suggestions: Optional[List[str]]) -> None:
        """Fallback error display for non-rich environments."""
        print(f"ERROR: {error}")
        if context:
            print(f"CONTEXT: {context}")
        if suggestions:
            print("SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion}")


class DryRunPlanner:
    """Dry-run planning mode for preview without execution."""
    
    def __init__(self, console: Optional[object] = None):
        """Initialize dry-run planner.
        
        Args:
            console: Rich console instance (optional)
        """
        if RICH_AVAILABLE and console is None:
            self.console = Console()
        else:
            self.console = console
    
    def preview_plan(self, plan: List[Dict[str, Any]], state: SessionState) -> bool:
        """Preview execution plan without running it.
        
        Args:
            plan: List of planned steps
            state: Current session state
            
        Returns:
            True if user approves plan, False otherwise
        """
        if not RICH_AVAILABLE or not self.console:
            return self._fallback_preview(plan, state)
        
        # Create plan preview
        self.console.print(Panel(
            "The following steps will be executed:",
            title="🔍 Execution Plan Preview",
            border_style="blue"
        ))
        
        # Display plan steps
        plan_table = Table(show_header=True, header_style="bold magenta")
        plan_table.add_column("Step", style="cyan", no_wrap=True)
        plan_table.add_column("Tool", style="green")
        plan_table.add_column("Description", style="white")
        plan_table.add_column("Parameters", style="yellow")
        
        for i, step in enumerate(plan, 1):
            tool = step.get("tool", "unknown")
            description = step.get("description", "No description")
            params = step.get("params", {})
            
            # Format parameters for display
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "None"
            if len(param_str) > 50:
                param_str = param_str[:47] + "..."
            
            plan_table.add_row(str(i), tool, description, param_str)
        
        self.console.print(plan_table)
        
        # Show current state context
        self._show_state_context(state)
        
        # Show estimated artifacts
        self._show_estimated_artifacts(plan)
        
        # Ask for confirmation
        if RICH_AVAILABLE:
            from rich.prompt import Confirm
            return Confirm.ask("\n[bold]Proceed with execution?[/bold]", default=True)
        else:
            response = input("\nProceed with execution? [Y/n]: ").strip().lower()
            return response in ['', 'y', 'yes']
    
    def _show_state_context(self, state: SessionState) -> None:
        """Show current state context."""
        if not RICH_AVAILABLE or not self.console:
            return
        
        context_info = []
        
        if hasattr(state, 'metadata') and state.metadata:
            if 'adata_path' in state.metadata:
                context_info.append(f"📊 Data: {Path(state.metadata['adata_path']).name}")
            if 'n_obs' in state.metadata:
                context_info.append(f"🔢 Cells: {state.metadata['n_obs']:,}")
            if 'n_vars' in state.metadata:
                context_info.append(f"🧬 Genes: {state.metadata['n_vars']:,}")
        
        context_info.append(f"📝 Steps completed: {len(state.history)}")
        context_info.append(f"📁 Artifacts: {len(state.artifacts)}")
        
        if context_info:
            self.console.print(Panel(
                "\n".join(context_info),
                title="📋 Current State",
                border_style="cyan"
            ))
    
    def _show_estimated_artifacts(self, plan: List[Dict[str, Any]]) -> None:
        """Show estimated artifacts that will be generated."""
        if not RICH_AVAILABLE or not self.console:
            return
        
        # Estimate artifacts based on tools
        estimated_artifacts = []
        
        for step in plan:
            tool = step.get("tool", "")
            
            if tool == "compute_qc_metrics":
                estimated_artifacts.extend(["qc_metrics.csv", "adata with QC metrics"])
            elif tool == "plot_qc":
                estimated_artifacts.extend(["qc_violin_plots.png", "qc_scatter_plots.png"])
            elif tool == "apply_qc_filters":
                estimated_artifacts.extend(["filter_summary.csv", "filtered_adata.h5ad"])
            elif tool == "quick_graph":
                estimated_artifacts.extend(["pca_plot.png", "umap_plot.png", "cluster_counts.csv"])
            elif tool == "run_scvi":
                estimated_artifacts.extend(["scvi_model/", "training_metrics.csv", "latent_embedding.h5ad"])
            elif tool == "detect_doublets":
                estimated_artifacts.extend(["doublet_scores.csv", "doublet_histogram.png"])
            elif "soupx" in tool or "decontx" in tool:
                estimated_artifacts.extend(["corrected_adata.h5ad", "contamination_plots.png"])
            elif "kbet" in tool or "lisi" in tool:
                estimated_artifacts.extend(["batch_diagnostics.csv", "diagnostic_plots.png"])
        
        if estimated_artifacts:
            self.console.print(Panel(
                "\n".join([f"• {artifact}" for artifact in estimated_artifacts]),
                title="📁 Estimated Artifacts",
                border_style="green"
            ))
    
    def _fallback_preview(self, plan: List[Dict[str, Any]], state: SessionState) -> bool:
        """Fallback preview for non-rich environments."""
        print("\nEXECUTION PLAN PREVIEW:")
        print("=" * 50)
        
        for i, step in enumerate(plan, 1):
            tool = step.get("tool", "unknown")
            description = step.get("description", "No description")
            params = step.get("params", {})
            
            print(f"{i}. {tool}: {description}")
            if params:
                print(f"   Parameters: {params}")
        
        print(f"\nCurrent state: {len(state.history)} steps completed")
        
        response = input("\nProceed with execution? [Y/n]: ").strip().lower()
        return response in ['', 'y', 'yes']


class InputValidator:
    """Enhanced input validation with user-friendly prompts."""
    
    @staticmethod
    def validate_file_path(path: str, extensions: Optional[List[str]] = None) -> bool:
        """Validate file path and extension.
        
        Args:
            path: File path to validate
            extensions: List of allowed extensions (e.g., ['.h5ad', '.csv'])
            
        Returns:
            True if valid, False otherwise
        """
        file_path = Path(path)
        
        if not file_path.exists():
            return False
        
        if extensions:
            return file_path.suffix.lower() in [ext.lower() for ext in extensions]
        
        return True
    
    @staticmethod
    def validate_species(species: str) -> bool:
        """Validate species parameter."""
        valid_species = ["human", "mouse", "other"]
        return species.lower() in valid_species
    
    @staticmethod
    def validate_tissue(tissue: str) -> bool:
        """Validate tissue parameter for priors."""
        # Import here to avoid circular imports
        try:
            from ..qc.priors import get_available_tissues
            available_tissues = get_available_tissues()
            return tissue.lower() in [t.lower() for t in available_tissues.keys()]
        except ImportError:
            return True  # Allow any tissue if priors module not available
    
    @staticmethod
    def suggest_corrections(invalid_value: str, valid_options: List[str]) -> Optional[str]:
        """Suggest closest valid option for invalid input.
        
        Args:
            invalid_value: The invalid input
            valid_options: List of valid options
            
        Returns:
            Suggested correction or None
        """
        # Simple string distance matching
        def string_distance(s1: str, s2: str) -> int:
            """Compute Levenshtein distance."""
            if len(s1) < len(s2):
                return string_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Find closest match
        min_distance = float('inf')
        closest_match = None
        
        for option in valid_options:
            distance = string_distance(invalid_value.lower(), option.lower())
            if distance < min_distance:
                min_distance = distance
                closest_match = option
        
        # Only suggest if reasonably close
        if min_distance <= len(invalid_value) // 2:
            return closest_match
        
        return None


class UXManager:
    """Centralized UX management for enhanced user experience."""
    
    def __init__(self, console: Optional[object] = None):
        """Initialize UX manager.
        
        Args:
            console: Rich console instance (optional)
        """
        if RICH_AVAILABLE and console is None:
            self.console = Console()
        else:
            self.console = console
        
        self.progress_tracker = ProgressTracker(self.console)
        self.error_handler = ErrorHandler(self.console)
        self.dry_run_planner = DryRunPlanner(self.console)
        self.input_validator = InputValidator()
    
    def execute_with_progress(
        self, 
        operation: Callable, 
        description: str, 
        *args, 
        total_steps: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Execute operation with progress tracking.
        
        Args:
            operation: Function to execute
            description: Description for progress display
            *args: Arguments for operation
            total_steps: Total number of steps for progress bar
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
        """
        with self.progress_tracker.track_operation(description, total_steps) as update:
            try:
                # Add progress callback to kwargs if supported
                if 'progress_callback' in operation.__code__.co_varnames:
                    kwargs['progress_callback'] = update
                
                result = operation(*args, **kwargs)
                update(1, "Completed")
                return result
                
            except Exception as e:
                self.error_handler.handle_error(e, context=description)
                raise
    
    def safe_execute(
        self, 
        operation: Callable, 
        context: str = "",
        suggestions: Optional[List[str]] = None,
        *args, 
        **kwargs
    ) -> Any:
        """Execute operation with error handling.
        
        Args:
            operation: Function to execute
            context: Context description for errors
            suggestions: Custom error suggestions
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation or None if failed
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            self.error_handler.handle_error(e, context, suggestions)
            return None
    
    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Get user confirmation for an action.
        
        Args:
            message: Confirmation message
            default: Default response
            
        Returns:
            True if confirmed, False otherwise
        """
        if RICH_AVAILABLE and self.console:
            from rich.prompt import Confirm
            return Confirm.ask(message, default=default)
        else:
            default_str = "Y/n" if default else "y/N"
            response = input(f"{message} [{default_str}]: ").strip().lower()
            if not response:
                return default
            return response in ['y', 'yes']
    
    def display_summary(self, title: str, data: Dict[str, Any], style: str = "blue") -> None:
        """Display a formatted summary panel.
        
        Args:
            title: Panel title
            data: Data to display
            style: Panel border style
        """
        if not RICH_AVAILABLE or not self.console:
            print(f"\n{title.upper()}")
            print("=" * len(title))
            for key, value in data.items():
                print(f"{key}: {value}")
            return
        
        content_lines = []
        for key, value in data.items():
            content_lines.append(f"[bold]{key}:[/bold] {value}")
        
        self.console.print(Panel(
            "\n".join(content_lines),
            title=title,
            border_style=style
        ))


# Global UX manager instance
_ux_manager: Optional[UXManager] = None


def get_ux_manager() -> UXManager:
    """Get global UX manager instance."""
    global _ux_manager
    if _ux_manager is None:
        _ux_manager = UXManager()
    return _ux_manager
