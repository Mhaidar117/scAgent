"""Command-line interface for scQC Agent."""

import os
import json
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .state import SessionState
from .agent.runtime import Agent
from .reports.export import export_report
from .utils.telemetry import initialize_telemetry, finalize_telemetry, get_global_collector
from .eval.runner import run_evaluation

# Initialize Typer app and console
app = typer.Typer(
    name="scqc",
    help="scQC Agent - Natural language interface for scRNA-seq QC workflows",
    add_completion=False
)
console = Console()

# QC subcommand group
qc_app = typer.Typer(name="qc", help="Quality control operations")
app.add_typer(qc_app, name="qc")

# Graph subcommand group
graph_app = typer.Typer(name="graph", help="Graph analysis operations")
app.add_typer(graph_app, name="graph")

# scAR subcommand group
scar_app = typer.Typer(name="scar", help="scAR denoising operations")
app.add_typer(scar_app, name="scar")

# scVI subcommand group
scvi_app = typer.Typer(name="scvi", help="scVI integration operations")
app.add_typer(scvi_app, name="scvi")

# Doublets subcommand group
doublets_app = typer.Typer(name="doublets", help="Doublet detection and filtering operations")
app.add_typer(doublets_app, name="doublets")

# Report subcommand group
report_app = typer.Typer(name="report", help="Report generation and export")
app.add_typer(report_app, name="report")

# Evaluation subcommand group
eval_app = typer.Typer(name="eval", help="Evaluation and testing operations")
app.add_typer(eval_app, name="eval")

# Default state file location
DEFAULT_STATE_PATH = ".scqc_state.json"


def get_state_path(state_path: Optional[str] = None) -> str:
    """Get the state file path, using default if not provided."""
    return state_path or DEFAULT_STATE_PATH


@app.command()
def init(
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Custom run ID"),
) -> None:
    """Initialize a new scQC session."""
    state_file = get_state_path(state_path)
    
    if Path(state_file).exists():
        console.print(f"[yellow]Warning: State file already exists at {state_file}[/yellow]")
        if not typer.confirm("Overwrite existing state?"):
            console.print("[red]Initialization cancelled[/red]")
            raise typer.Exit(1)
    
    # Create new session state
    state = SessionState(run_id=run_id)
    state.save(state_file)
    
    console.print(Panel(
        f"✅ Initialized new scQC session\n\n"
        f"📁 State file: {state_file}\n"
        f"🆔 Run ID: {state.run_id}\n"
        f"📅 Created: {state.created_at}",
        title="Session Initialized",
        border_style="green"
    ))


@app.command("state")
def state_show(
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Show current session state."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        # Create summary table
        table = Table(title="Session State")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Run ID", state.run_id)
        table.add_row("Created", state.created_at)
        table.add_row("Updated", state.updated_at)
        table.add_row("History Entries", str(len(state.history)))
        table.add_row("Artifacts", str(len(state.artifacts)))
        
        console.print(table)
        
        # Show recent history if any
        if state.history:
            console.print("\n📜 Recent History:")
            for i, entry in enumerate(state.history[-3:]):  # Show last 3 entries
                step_num = entry.get("step", i)
                label = entry.get("label", "Unknown")
                timestamp = entry.get("timestamp", "Unknown")
                console.print(f"  {step_num:2d}. {label} ({timestamp})")
                
                # Show artifacts for this step
                artifacts = entry.get("artifacts", [])
                for artifact in artifacts:
                    artifact_label = artifact.get("label", "Unknown")
                    artifact_path = artifact.get("path", "Unknown")
                    console.print(f"      📄 {artifact_label}: {artifact_path}")
        
    except Exception as e:
        console.print(f"[red]Error loading state: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def load(
    path: str = typer.Argument(..., help="Path to AnnData file (.h5ad)"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Load an AnnData file into the session."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    # Check if file exists
    if not Path(path).exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        # Create checkpoint for loading data
        checkpoint_path = state.checkpoint(path, "initial_load")
        state.update_metadata({"adata_path": str(Path(path).resolve())})
        state.adata_path = str(Path(path).resolve())
        state.save(state_file)
        
        console.print(Panel(
            f"✅ Loaded AnnData file\n\n"
            f"📂 File: {path}\n"
            f"🔄 Checkpoint: {checkpoint_path}\n"
            f"📊 Ready for QC operations",
            title="Data Loaded",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def plan(
    text: str = typer.Argument(..., help="Natural language description of workflow"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Generate a workflow plan from natural language."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        # Initialize agent with state file
        agent = Agent(state_file)
        result = agent.handle_message(text)
        
        console.print(Panel(
            f"📝 Input: {text}\n\n"
            f"📋 Generated Plan:",
            title="Workflow Plan",
            border_style="blue"
        ))
        
        # Display plan steps
        for i, step in enumerate(result["plan"], 1):
            console.print(f"  {i}. {step}")
        
        console.print(f"\n📊 Status: {result['status']}")
        
        # Show tool results if any
        if result.get("tool_results"):
            console.print("\n🔧 Tool Results:")
            for tool_result in result["tool_results"]:
                console.print(f"  • {tool_result.get('message', 'No message')}")
        
    except Exception as e:
        console.print(f"[red]Error generating plan: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def summary(
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed summary with telemetry"),
) -> None:
    """Show enhanced session summary with workflow progress and metrics."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        # Enhanced summary with rich tables
        console.print(Panel(
            f"🆔 **Run ID:** {state.run_id}\n"
            f"📅 **Created:** {datetime.fromisoformat(state.created_at).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🕐 **Last Updated:** {datetime.fromisoformat(state.updated_at).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"📜 **Workflow Steps:** {len(state.history)}\n"
            f"📁 **Generated Artifacts:** {len(state.artifacts)}",
            title="📊 Session Overview",
            border_style="blue"
        ))
        
        # Workflow progress table
        if state.history:
            console.print("\n🔄 **Workflow Progress**")
            progress_table = Table()
            progress_table.add_column("Step", style="cyan", width=4)
            progress_table.add_column("Stage", style="white", min_width=20)
            progress_table.add_column("Time", style="dim", width=16)
            progress_table.add_column("Artifacts", style="green", width=10)
            
            for entry in state.history[-10:]:  # Show last 10 steps
                step = entry.get("step", "?")
                label = entry.get("label", "Unknown")
                timestamp = entry.get("timestamp", "")
                artifacts = len(entry.get("artifacts", []))
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%m-%d %H:%M')
                except:
                    time_str = "Unknown"
                
                progress_table.add_row(
                    str(step), 
                    label[:30] + "..." if len(label) > 30 else label,
                    time_str,
                    str(artifacts)
                )
            
            console.print(progress_table)
        
        # Key metrics table
        if state.metadata:
            console.print("\n📈 **Key Metrics**")
            metrics_table = Table()
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="white")
            
            # Display important metrics
            metric_keys = [
                ("n_cells_initial", "Initial Cells"),
                ("n_genes_initial", "Initial Genes"),
                ("cells_after_qc", "Cells After QC"),
                ("genes_after_qc", "Genes After QC"),
                ("final_n_clusters", "Final Clusters"),
                ("doublets_removed", "Doublets Removed"),
                ("n_batches", "Batches")
            ]
            
            for key, display_name in metric_keys:
                if key in state.metadata:
                    value = state.metadata[key]
                    if isinstance(value, float):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)
                    metrics_table.add_row(display_name, value_str)
            
            if metrics_table.rows:
                console.print(metrics_table)
        
        # Artifacts by category
        if state.artifacts:
            console.print("\n📁 **Generated Artifacts**")
            
            # Categorize artifacts
            artifact_categories = {
                "Plots": [],
                "Data": [],
                "Models": [],
                "Reports": [],
                "Other": []
            }
            
            for path, label in state.artifacts.items():
                path_lower = path.lower()
                if any(ext in path_lower for ext in [".png", ".jpg", ".jpeg", ".svg"]):
                    artifact_categories["Plots"].append((label, Path(path).name))
                elif any(ext in path_lower for ext in [".h5ad", ".csv", ".tsv"]):
                    artifact_categories["Data"].append((label, Path(path).name))
                elif any(ext in path_lower for ext in [".pkl", ".pt", ".h5", "model"]):
                    artifact_categories["Models"].append((label, Path(path).name))
                elif any(ext in path_lower for ext in [".html", ".pdf", "report"]):
                    artifact_categories["Reports"].append((label, Path(path).name))
                else:
                    artifact_categories["Other"].append((label, Path(path).name))
            
            for category, items in artifact_categories.items():
                if items:
                    console.print(f"\n   **{category}** ({len(items)} files)")
                    for label, filename in items[:5]:  # Show first 5
                        console.print(f"     • {label}: [dim]{filename}[/dim]")
                    if len(items) > 5:
                        console.print(f"     ... and {len(items) - 5} more")
        
        # Telemetry information if detailed and available
        if detailed:
            telemetry_path = Path(f"runs/{state.run_id}/telemetry.json")
            if telemetry_path.exists():
                try:
                    with open(telemetry_path, 'r') as f:
                        telemetry = json.load(f)
                    
                    console.print("\n⚡ **Performance Summary**")
                    perf_table = Table()
                    perf_table.add_column("Metric", style="cyan")
                    perf_table.add_column("Value", style="white")
                    
                    if "total_runtime" in telemetry:
                        runtime = telemetry["total_runtime"]
                        if runtime >= 3600:
                            runtime_str = f"{runtime // 3600:.0f}h {(runtime % 3600) // 60:.0f}m"
                        elif runtime >= 60:
                            runtime_str = f"{runtime // 60:.0f}m {runtime % 60:.0f}s"
                        else:
                            runtime_str = f"{runtime:.1f}s"
                        perf_table.add_row("Total Runtime", runtime_str)
                    
                    if "peak_memory_mb" in telemetry:
                        memory = telemetry["peak_memory_mb"]
                        if memory >= 1024:
                            memory_str = f"{memory / 1024:.1f} GB"
                        else:
                            memory_str = f"{memory:.1f} MB"
                        perf_table.add_row("Peak Memory", memory_str)
                    
                    if "steps" in telemetry:
                        avg_time = sum(s.get("duration", 0) for s in telemetry["steps"]) / len(telemetry["steps"])
                        perf_table.add_row("Avg Step Time", f"{avg_time:.2f}s")
                    
                    console.print(perf_table)
                    
                except Exception:
                    pass
        
        # Show next steps suggestions
        console.print(Panel(
            "💡 **Next Steps**\n\n"
            "• Generate report: `scqc report export --format html`\n"
            "• Continue workflow: `scqc chat \"[describe next step]\"`\n"
            "• View detailed state: `scqc state show`\n"
            f"• View artifacts: explore `runs/{state.run_id}/`",
            title="Suggestions",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error loading summary: {e}[/red]")
        raise typer.Exit(1)


# QC subcommands
@qc_app.command("compute")
def qc_compute(
    species: Literal["human", "mouse", "other"] = typer.Option("human", "--species", help="Species for mitochondrial gene detection"),
    mito_prefix: Optional[str] = typer.Option(None, "--mito-prefix", help="Custom mitochondrial gene prefix"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Compute QC metrics for the loaded dataset."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"🧬 Species: {species}\n"
        f"🧪 Computing QC metrics...\n"
        f"⚠️  Phase 1 placeholder - real scanpy integration coming soon",
        title="QC Computation",
        border_style="yellow"
    ))
    
    # This is a Phase 0/1 placeholder
    # In Phase 1+, this would call the actual QC tools
    console.print("✅ QC metrics computation complete (placeholder)")


@qc_app.command("plot")
def qc_plot(
    stage: Literal["pre", "post"] = typer.Option("pre", "--stage", help="Plot pre- or post-filtering QC"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Generate QC plots."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"📊 Stage: {stage}-filtering\n"
        f"🎨 Generating QC plots...\n"
        f"⚠️  Phase 1 placeholder - real plotting coming soon",
        title="QC Plotting",
        border_style="yellow"
    ))
    
    # This is a Phase 0/1 placeholder
    console.print("✅ QC plots generated (placeholder)")


@qc_app.command("apply")
def qc_apply(
    min_genes: Optional[int] = typer.Option(None, "--min-genes", help="Minimum genes per cell"),
    max_pct_mt: Optional[float] = typer.Option(None, "--max-pct-mt", help="Maximum mitochondrial percentage"),
    method: Literal["threshold", "MAD", "quantile"] = typer.Option("threshold", "--method", help="Filtering method"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Apply QC filters to the dataset."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"🔧 Method: {method}\n"
        f"📊 Min genes: {min_genes or 'auto'}\n"
        f"🧪 Max MT%: {max_pct_mt or 'auto'}\n"
        f"⚠️  Phase 1 placeholder - real filtering coming soon",
        title="QC Filtering",
        border_style="yellow"
    ))
    
    # This is a Phase 0/1 placeholder
    console.print("✅ QC filters applied (placeholder)")


@qc_app.command("help")
def qc_help() -> None:
    """Show detailed help for QC commands."""
    console.print(Panel(
        "🧪 **QC Commands**\n\n"
        "• `scqc qc compute` - Compute QC metrics\n"
        "• `scqc qc plot` - Generate QC visualizations\n"
        "• `scqc qc apply` - Apply QC filters\n\n"
        "📚 **Workflow Example**\n\n"
        "```bash\n"
        "scqc init\n"
        "scqc load data/pbmc3k.h5ad\n"
        "scqc qc compute --species human\n"
        "scqc qc plot --stage pre\n"
        "scqc qc apply --min-genes 200 --max-pct-mt 20\n"
        "scqc qc plot --stage post\n"
        "```",
        title="QC Help",
        border_style="blue"
    ))


# Graph subcommands
@graph_app.command("quick")
def graph_quick(
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility"),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden clustering resolution"),
    n_neighbors: int = typer.Option(15, "--n-neighbors", help="Number of neighbors for kNN graph"),
    n_pcs: int = typer.Option(50, "--n-pcs", help="Number of principal components"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Quick graph analysis: PCA → neighbors → UMAP → Leiden clustering."""
    from .tools.graph import quick_graph
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        console.print(Panel(
            f"🔗 Building neighbors graph...\n"
            f"📊 Parameters:\n"
            f"  • Seed: {seed}\n"
            f"  • Resolution: {resolution}\n"
            f"  • Neighbors: {n_neighbors}\n"
            f"  • PCs: {n_pcs}",
            title="Quick Graph Analysis",
            border_style="blue"
        ))
        
        with console.status("[bold blue]Running PCA → neighbors → UMAP → Leiden..."):
            result = quick_graph(state, seed=seed, resolution=resolution, 
                               n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            console.print(f"\n📈 Results:")
            if "n_clusters" in result.state_delta:
                console.print(f"  • Clusters: {result.state_delta['n_clusters']}")
            if "connectivity_rate" in result.state_delta:
                console.print(f"  • Connectivity: {result.state_delta['connectivity_rate']:.1f} neighbors/cell")
            if "largest_cluster_pct" in result.state_delta:
                console.print(f"  • Largest cluster: {result.state_delta['largest_cluster_pct']:.1f}%")
        
        console.print(f"\n📁 Artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during graph analysis: {e}[/red]")
        raise typer.Exit(1)


@graph_app.command("from-rep")
def graph_from_rep(
    use_rep: str = typer.Argument(..., help="Representation to use (e.g., X_scVI, X_scAR, X_pca)"),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility"),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden clustering resolution"),
    n_neighbors: int = typer.Option(15, "--n-neighbors", help="Number of neighbors for kNN graph"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Graph analysis from a specific representation (e.g., X_scVI, X_scAR)."""
    from .tools.graph import graph_from_rep
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        console.print(Panel(
            f"🔗 Building graph from representation...\n"
            f"📊 Parameters:\n"
            f"  • Representation: {use_rep}\n"
            f"  • Seed: {seed}\n"
            f"  • Resolution: {resolution}\n"
            f"  • Neighbors: {n_neighbors}",
            title=f"Graph from {use_rep}",
            border_style="blue"
        ))
        
        with console.status(f"[bold blue]Running neighbors → UMAP → Leiden from {use_rep}..."):
            result = graph_from_rep(state, use_rep=use_rep, seed=seed, 
                                  resolution=resolution, n_neighbors=n_neighbors)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            rep_short = use_rep.replace("X_", "").lower()
            console.print(f"\n📈 Results:")
            if f"n_clusters_{rep_short}" in result.state_delta:
                console.print(f"  • Clusters: {result.state_delta[f'n_clusters_{rep_short}']}")
            if f"connectivity_rate_{rep_short}" in result.state_delta:
                console.print(f"  • Connectivity: {result.state_delta[f'connectivity_rate_{rep_short}']:.1f} neighbors/cell")
            if f"largest_cluster_pct_{rep_short}" in result.state_delta:
                console.print(f"  • Largest cluster: {result.state_delta[f'largest_cluster_pct_{rep_short}']:.1f}%")
        
        console.print(f"\n📁 Artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during graph analysis: {e}[/red]")
        raise typer.Exit(1)


@graph_app.command("final")
def graph_final(
    use_rep: str = typer.Option("X_scVI", "--use-rep", help="Representation to use for final analysis"),
    resolution: float = typer.Option(1.0, "--resolution", help="Leiden clustering resolution"),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility"),
    n_neighbors: int = typer.Option(15, "--n-neighbors", help="Number of neighbors for kNN graph"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Final graph analysis: culminating step producing final UMAP and clusters."""
    from .tools.graph import final_graph
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        console.print(Panel(
            f"🏁 Final graph analysis...\n"
            f"📊 Parameters:\n"
            f"  • Representation: {use_rep}\n"
            f"  • Resolution: {resolution}\n"
            f"  • Seed: {seed}\n"
            f"  • Neighbors: {n_neighbors}\n"
            f"🎯 This is the culminating analysis step!",
            title="Final Graph Analysis",
            border_style="green"
        ))
        
        with console.status("[bold green]Running final neighbors → UMAP → Leiden..."):
            result = final_graph(state, use_rep=use_rep, resolution=resolution, 
                               seed=seed, n_neighbors=n_neighbors)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            console.print(f"\n🎉 Final Results:")
            if "final_n_clusters" in result.state_delta:
                console.print(f"  • Final clusters: {result.state_delta['final_n_clusters']}")
            if "final_connectivity_rate" in result.state_delta:
                console.print(f"  • Connectivity: {result.state_delta['final_connectivity_rate']:.1f} neighbors/cell")
            if "final_largest_cluster_pct" in result.state_delta:
                console.print(f"  • Largest cluster: {result.state_delta['final_largest_cluster_pct']:.1f}%")
            if "final_cluster_balance" in result.state_delta:
                console.print(f"  • Cluster balance: {result.state_delta['final_cluster_balance']:.3f}")
            if result.state_delta.get("pipeline_complete"):
                console.print("  🏆 End-to-end pipeline COMPLETE!")
        
        console.print(f"\n📁 Final artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during final graph analysis: {e}[/red]")
        raise typer.Exit(1)


@graph_app.command("help")
def graph_help() -> None:
    """Show detailed help for graph commands."""
    console.print(Panel(
        "🔗 **Graph Commands**\n\n"
        "• `scqc graph quick` - Quick PCA→neighbors→UMAP→Leiden analysis\n"
        "• `scqc graph from-rep` - Graph analysis from specific representation\n"
        "• `scqc graph final` - Final graph analysis (culminating step)\n\n"
        "📚 **Workflow Example**\n\n"
        "```bash\n"
        "# Quick analysis on raw data\n"
        "scqc graph quick --seed 42 --resolution 0.5\n"
        "\n"
        "# Analysis from scVI latent space\n"
        "scqc graph from-rep X_scVI --resolution 1.0\n"
        "\n"
        "# Final analysis (after doublet removal)\n"
        "scqc graph final --use-rep X_scVI --resolution 1.0\n"
        "```\n\n"
        "🎯 **Purpose**\n"
        "Graph analysis provides clustering and visualization at different\n"
        "stages of the analysis pipeline. 'final' is the culminating step.",
        title="Graph Help",
        border_style="blue"
    ))


# scAR subcommands
@scar_app.command("run")
def scar_run(
    batch_key: str = typer.Option("SampleID", "--batch-key", help="Column in adata.obs for batch information"),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    replace_x: bool = typer.Option(True, "--replace-x/--no-replace-x", help="Replace X with denoised counts"),
    random_seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Run scAR denoising for ambient RNA removal."""
    from .tools.scar import run_scar
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        console.print(Panel(
            f"🧹 Running scAR denoising...\n"
            f"📊 Parameters:\n"
            f"  • Batch key: {batch_key}\n"
            f"  • Epochs: {epochs}\n"
            f"  • Replace X: {replace_x}\n"
            f"  • Seed: {random_seed}",
            title="scAR Denoising",
            border_style="green"
        ))
        
        with console.status("[bold green]Training scAR model for ambient RNA removal..."):
            result = run_scar(state, batch_key=batch_key, epochs=epochs, 
                             replace_X=replace_x, random_seed=random_seed)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            console.print(f"\n📈 Results:")
            if "scar_epochs" in result.state_delta:
                console.print(f"  • Training epochs: {result.state_delta['scar_epochs']}")
            if "denoised_total_counts" in result.state_delta:
                console.print(f"  • Denoised total counts: {result.state_delta['denoised_total_counts']:.0f}")
            if "noise_removed_pct" in result.state_delta:
                console.print(f"  • Noise removed: {result.state_delta['noise_removed_pct']}%")
        
        console.print(f"\n📁 Artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during scAR denoising: {e}[/red]")
        raise typer.Exit(1)


@scar_app.command("help")
def scar_help() -> None:
    """Show detailed help for scAR commands."""
    console.print(Panel(
        "🧹 **scAR Commands**\n\n"
        "• `scqc scar run` - Run scAR denoising for ambient RNA removal\n\n"
        "📚 **Workflow Example**\n\n"
        "```bash\n"
        "# Basic scAR denoising\n"
        "scqc scar run --batch-key batch --epochs 100\n"
        "\n"
        "# Preserve original X matrix\n"
        "scqc scar run --no-replace-x --epochs 150\n"
        "```\n\n"
        "🎯 **Purpose**\n"
        "scAR removes ambient RNA contamination commonly found in\n"
        "droplet-based scRNA-seq data, improving downstream analysis.",
        title="scAR Help",
        border_style="green"
    ))


# scVI subcommands
@scvi_app.command("run")
def scvi_run(
    batch_key: str = typer.Option("SampleID", "--batch-key", help="Column in adata.obs for batch information"),
    n_latent: int = typer.Option(30, "--n-latent", help="Number of latent dimensions"),
    epochs: int = typer.Option(200, "--epochs", help="Number of training epochs"),
    random_seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Run scVI for batch correction and latent representation learning."""
    from .tools.scvi import run_scvi
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        console.print(Panel(
            f"🔬 Running scVI integration...\n"
            f"📊 Parameters:\n"
            f"  • Batch key: {batch_key}\n"
            f"  • Latent dims: {n_latent}\n"
            f"  • Epochs: {epochs}\n"
            f"  • Seed: {random_seed}",
            title="scVI Integration",
            border_style="purple"
        ))
        
        with console.status("[bold purple]Training scVI model for batch correction..."):
            result = run_scvi(state, batch_key=batch_key, n_latent=n_latent, 
                             epochs=epochs, random_seed=random_seed)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            console.print(f"\n📈 Results:")
            if "scvi_n_latent" in result.state_delta:
                console.print(f"  • Latent dimensions: {result.state_delta['scvi_n_latent']}")
            if "n_batches" in result.state_delta:
                console.print(f"  • Batches integrated: {result.state_delta['n_batches']}")
            if "latent_variance" in result.state_delta:
                console.print(f"  • Avg latent variance: {result.state_delta['latent_variance']:.4f}")
        
        console.print(f"\n📁 Artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during scVI integration: {e}[/red]")
        raise typer.Exit(1)


@scvi_app.command("help")
def scvi_help() -> None:
    """Show detailed help for scVI commands."""
    console.print(Panel(
        "🔬 **scVI Commands**\n\n"
        "• `scqc scvi run` - Run scVI for batch correction and integration\n\n"
        "📚 **Workflow Example**\n\n"
        "```bash\n"
        "# Basic scVI integration\n"
        "scqc scvi run --batch-key batch --n-latent 20 --epochs 200\n"
        "\n"
        "# High-dimensional latent space\n"
        "scqc scvi run --n-latent 50 --epochs 300\n"
        "```\n\n"
        "🎯 **Purpose**\n"
        "scVI provides batch correction and learns a low-dimensional\n"
        "latent representation suitable for downstream analysis.",
        title="scVI Help",
        border_style="purple"
    ))


# Doublets subcommands
@doublets_app.command("detect")
def doublets_detect(
    method: Literal["scrublet", "doubletfinder"] = typer.Option("scrublet", "--method", help="Doublet detection method"),
    expected_rate: float = typer.Option(0.06, "--expected-rate", help="Expected doublet rate (0.01-0.5)"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Custom doublet score threshold (auto if not provided)"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Detect doublets in scRNA-seq data."""
    from .tools.doublets import detect_doublets
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        # Prepare threshold parameter
        threshold_param = threshold if threshold is not None else "auto"
        
        console.print(Panel(
            f"🔬 Detecting doublets...\n"
            f"📊 Parameters:\n"
            f"  • Method: {method}\n"
            f"  • Expected rate: {expected_rate:.1%}\n"
            f"  • Threshold: {threshold_param}\n"
            f"🎯 Identifying multi-cell droplets",
            title="Doublet Detection",
            border_style="orange"
        ))
        
        with console.status("[bold orange]Running doublet detection..."):
            result = detect_doublets(state, method=method, expected_rate=expected_rate, 
                                   threshold=threshold_param)
        
        # Apply state_delta to the state before saving
        if result.state_delta:
            if "adata_path" in result.state_delta:
                state.adata_path = result.state_delta["adata_path"]
            state.update_metadata(result.state_delta)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            console.print(f"\n📈 Detection Results:")
            if "detected_doublet_rate" in result.state_delta:
                console.print(f"  • Detected rate: {result.state_delta['detected_doublet_rate']:.1%}")
            if "n_doublets" in result.state_delta:
                console.print(f"  • Doublets: {result.state_delta['n_doublets']}")
            if "n_singlets" in result.state_delta:
                console.print(f"  • Singlets: {result.state_delta['n_singlets']}")
            if "doublet_threshold" in result.state_delta:
                console.print(f"  • Threshold: {result.state_delta['doublet_threshold']:.3f}")
        
        console.print(f"\n📁 Artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during doublet detection: {e}[/red]")
        raise typer.Exit(1)


@doublets_app.command("apply")
def doublets_apply(
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Custom threshold for filtering (uses detected threshold if not provided)"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Apply doublet filter to remove detected doublets."""
    from .tools.doublets import apply_doublet_filter
    
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        threshold_display = f"{threshold:.3f}" if threshold is not None else "auto (from detection)"
        
        console.print(Panel(
            f"🧹 Applying doublet filter...\n"
            f"📊 Parameters:\n"
            f"  • Threshold: {threshold_display}\n"
            f"⚠️  This will remove cells marked as doublets",
            title="Doublet Filtering",
            border_style="red"
        ))
        
        with console.status("[bold red]Filtering doublets..."):
            result = apply_doublet_filter(state, threshold=threshold)
        
        # Apply state_delta to the state before saving
        if result.state_delta:
            if "adata_path" in result.state_delta:
                state.adata_path = result.state_delta["adata_path"]
            state.update_metadata(result.state_delta)
        
        # Save updated state
        state.save(state_file)
        
        if result.state_delta:
            console.print(f"\n📉 Filtering Results:")
            if "cells_before_doublet_filter" in result.state_delta:
                console.print(f"  • Cells before: {result.state_delta['cells_before_doublet_filter']}")
            if "cells_after_doublet_filter" in result.state_delta:
                console.print(f"  • Cells after: {result.state_delta['cells_after_doublet_filter']}")
            if "doublets_removed" in result.state_delta:
                console.print(f"  • Doublets removed: {result.state_delta['doublets_removed']}")
            if "final_doublet_rate" in result.state_delta:
                console.print(f"  • Removal rate: {result.state_delta['final_doublet_rate']:.1%}")
        
        console.print(f"\n📁 Artifacts generated:")
        for artifact in result.artifacts:
            console.print(f"  • {artifact}")
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during doublet filtering: {e}[/red]")
        raise typer.Exit(1)


@doublets_app.command("help")
def doublets_help() -> None:
    """Show detailed help for doublet commands."""
    console.print(Panel(
        "🔬 **Doublet Commands**\n\n"
        "• `scqc doublets detect` - Detect doublets using Scrublet or DoubletFinder\n"
        "• `scqc doublets apply` - Apply doublet filter to remove detected doublets\n\n"
        "📚 **Workflow Example**\n\n"
        "```bash\n"
        "# Basic doublet detection\n"
        "scqc doublets detect --expected-rate 0.06\n"
        "\n"
        "# Custom threshold detection\n"
        "scqc doublets detect --method scrublet --threshold 0.35\n"
        "\n"
        "# Apply filter with detected threshold\n"
        "scqc doublets apply\n"
        "\n"
        "# Apply filter with custom threshold\n"
        "scqc doublets apply --threshold 0.35\n"
        "```\n\n"
        "🎯 **Purpose**\n"
        "Doublet detection identifies and removes multi-cell droplets that can\n"
        "confound downstream analysis. Critical for droplet-based protocols.",
        title="Doublets Help",
        border_style="orange"
    ))


@app.command()
def chat(
    message: str = typer.Argument(..., help="Natural language request for the scQC workflow"),
    mode: str = typer.Option("plan", "--mode", "-m", help="Mode: 'plan' (default) or 'execute'"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode with approval"),
    plan_path: Optional[str] = typer.Option(None, "--plan-path", "-p", help="Path to stored plan.json (for execute mode)"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
    kb_path: Optional[str] = typer.Option(None, "--kb-path", "-k", help="Path to knowledge base directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed execution information"),
) -> None:
    """Natural language interface for scQC workflows with planning and execution phases.

    The agent operates in two modes:
    1. PLAN mode (default): Generate and show execution plan without running tools
    2. EXECUTE mode: Execute the plan and run the actual tools
    3. INTERACTIVE mode: Plan first, ask for approval, then execute

    Examples:
        # Generate plan only
        scqc chat "compute QC metrics for human data" --mode plan

        # Execute directly (careful - this runs tools immediately)
        scqc chat "compute QC metrics for human data" --mode execute

        # Execute from stored plan
        scqc chat "compute QC metrics for human data" --mode execute --plan-path runs/*/chat_*/plan.json

        # Interactive mode (recommended)
        scqc chat "compute QC metrics for human data" --interactive
    """
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        # Initialize agent
        with console.status("[bold blue]Initializing agent..."):
            agent = Agent(state_file, knowledge_base_path=kb_path)
        
        if interactive:
            # Interactive mode: Plan → Ask for approval → Execute
            console.print(Panel(
                f"🤖 **Interactive Planning Mode**\n\n"
                f"Request: {message}\n\n"
                f"Generating execution plan...",
                title="scQC Agent - Planning",
                border_style="blue"
            ))
            
            # Step 1: Generate plan
            with console.status("[bold blue]Generating plan..."):
                plan_result = agent.chat(message, mode="plan")
            
            if plan_result.get("status") == "failed":
                console.print(f"[red]❌ Planning failed: {plan_result.get('error', 'Unknown error')}[/red]")
                raise typer.Exit(1)
            
            # Step 2: Show plan to user
            console.print(f"\n🎯 **Detected Intent**: {plan_result.get('intent', 'unknown')}")
            console.print(f"\n📋 **Proposed Execution Plan**:")
            
            plan = plan_result.get("plan", [])
            if not plan:
                console.print("[yellow]No plan generated[/yellow]")
                raise typer.Exit(1)
            
            for i, step in enumerate(plan, 1):
                tool = step.get("tool", "unknown")
                description = step.get("description", "No description")
                params = step.get("params", {})
                
                console.print(f"  {i}. [cyan]{tool}[/cyan]: {description}")
                if params and verbose:
                    console.print(f"     Parameters: {params}")
            
            # Step 3: Ask for approval
            console.print(f"\n[yellow]⚠️  This plan will execute {len(plan)} step(s) and may modify your data.[/yellow]")
            approve = typer.confirm("\nDo you want to proceed with this execution plan?")
            
            if not approve:
                console.print(Panel(
                    "Plan cancelled by user.\n\n"
                    "You can:\n"
                    "• Modify your request and try again\n"
                    "• Use --mode plan to just see plans without approval prompts\n"
                    "• Review the plan and run with --mode execute if you're confident",
                    title="Cancelled",
                    border_style="yellow"
                ))
                return
            
            # Step 4: Execute approved plan
            console.print(Panel(
                "🚀 Executing approved plan...\n\n"
                "This may take a few minutes depending on data size.",
                title="Execution Started",
                border_style="green"
            ))
            
            with console.status("[bold green]Executing plan..."):
                # Get the explicit plan path from the planning result
                stored_plan_path = plan_result.get("plan_path")
                execution_result = agent.chat(message, mode="execute", plan_path=stored_plan_path)

            result = execution_result

        else:
            # Non-interactive mode
            if mode == "plan":
                console.print(Panel(
                    f"🤖 **Planning Mode**\n\n"
                    f"Request: {message}\n\n"
                    f"Generating execution plan (no tools will be executed)...",
                    title="scQC Agent - Planning Only",
                    border_style="blue"
                ))
            else:
                # Check if plan_path is provided for execute mode
                if mode == "execute" and plan_path:
                    console.print(Panel(
                        f"🤖 **Stored Plan Execution Mode**\n\n"
                        f"Request: {message}\n\n"
                        f"📋 Plan: {plan_path}\n\n"
                        f"⚠️  Tools will be executed from stored plan!",
                        title="scQC Agent - Execute Stored Plan",
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        f"🤖 **Direct Execution Mode**\n\n"
                        f"Request: {message}\n\n"
                        f"⚠️  Tools will be executed immediately!",
                        title="scQC Agent - Direct Execution",
                        border_style="red"
                    ))

            with console.status(f"[bold blue]Processing in {mode} mode..."):
                result = agent.chat(message, mode=mode, plan_path=plan_path)
        
        # Display results based on mode
        if result.get("status") == "failed":
            console.print(f"[red]❌ Error: {result.get('error', 'Unknown error')}[/red]")
            raise typer.Exit(1)
        
        if result.get("mode") == "planning":
            # Show planning results
            console.print(f"\n🎯 **Detected Intent**: {result.get('intent', 'unknown')}")
            
            plan = result.get("plan", [])
            if plan:
                console.print(f"\n📋 **Generated Plan**:")
                for i, step in enumerate(plan, 1):
                    tool = step.get("tool", "unknown")
                    description = step.get("description", "No description")
                    console.print(f"  {i}. [cyan]{tool}[/cyan]: {description}")
                
                # Show plan file path
                plan_path = result.get("plan_path")
                if plan_path:
                    console.print(f"\n💾 **Plan saved to**: [dim]{plan_path}[/dim]")

                console.print(f"\n💡 **Next Steps**:")
                if plan_path:
                    console.print(f"• To execute stored plan: [green]scqc chat \"{message}\" --mode execute --plan-path {plan_path}[/green]")
                console.print(f"• To execute (regenerate plan): [green]scqc chat \"{message}\" --mode execute[/green]")
                console.print(f"• For interactive: [blue]scqc chat \"{message}\" --interactive[/blue]")
                console.print(f"• To modify: Adjust your request and plan again")
            
        else:
            # Show execution results
            console.print(f"\n🔧 **Execution Results**:")
            
            tool_results = result.get("tool_results", [])
            for i, tool_result in enumerate(tool_results, 1):
                message_text = tool_result.get("message", "No message")
                if message_text.startswith("❌"):
                    console.print(f"  {i}. [red]{message_text}[/red]")
                else:
                    console.print(f"  {i}. [green]{message_text}[/green]")
                
                # Show artifacts if any
                artifacts = tool_result.get("artifacts", [])
                if artifacts:
                    for artifact in artifacts:
                        console.print(f"     📄 {artifact}")
            
            # Show summary
            summary = result.get("summary", "")
            if summary:
                console.print(f"\n📝 **Summary**:")
                console.print(summary)
            
            # Show artifacts
            all_artifacts = result.get("artifacts", [])
            if all_artifacts:
                console.print(f"\n📁 **Generated Artifacts**:")
                for artifact in all_artifacts:
                    console.print(f"  • {artifact}")
            
            # Show citations (always show, not just in verbose mode)
            citations = result.get("citations", [])
            if citations:
                console.print(f"\n📚 **Knowledge Sources**:")
                for citation in citations:
                    console.print(f"  • {citation}")
        
        # Show chat run directory
        chat_run_dir = result.get("chat_run_dir")
        if chat_run_dir:
            console.print(f"\n💾 **Session artifacts saved to**: {chat_run_dir}")
        
        # CRITICAL FIX: Save state after execution
        if result.get("mode") == "execution":
            agent.save_state()
            console.print(f"\n💾 State saved to: {state_file}")
        
        # Success message
        if result.get("mode") == "planning":
            console.print(Panel(
                "📋 Plan generated successfully!\n\n"
                "Review the plan above and choose your next step.",
                title="Planning Complete",
                border_style="blue"
            ))
        else:
            console.print(Panel(
                "🎉 Execution completed successfully!\n\n"
                "Use 'scqc summary' for a comprehensive overview",
                title="Execution Complete",
                border_style="green"
            ))
        
    except Exception as e:
        console.print(f"[red]Error processing chat request: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)

# Report subcommands
@report_app.command("export")
def report_export(
    format: Literal["html", "pdf"] = typer.Option("html", "--format", "-f", help="Export format"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Custom output path"),
    include_telemetry: bool = typer.Option(True, "--telemetry/--no-telemetry", help="Include telemetry data"),
    state_path: Optional[str] = typer.Option(None, "--state-path", "-s", help="Path to state file"),
) -> None:
    """Export a comprehensive workflow report."""
    state_file = get_state_path(state_path)
    
    if not Path(state_file).exists():
        console.print(f"[red]No state file found at {state_file}[/red]")
        console.print("💡 Run 'scqc init' to create a new session")
        raise typer.Exit(1)
    
    try:
        state = SessionState.load(state_file)
        
        console.print(Panel(
            f"📊 Generating {format.upper()} report...\n"
            f"📁 Session: {state.run_id}\n"
            f"📈 Steps: {len(state.history)}\n"
            f"📄 Artifacts: {len(state.artifacts)}\n"
            f"📋 Telemetry: {'✅ Included' if include_telemetry else '❌ Excluded'}",
            title="Report Export",
            border_style="blue"
        ))
        
        with console.status(f"[bold blue]Building {format.upper()} report..."):
            result = export_report(
                state=state,
                format=format,
                output_path=output_path,
                include_telemetry=include_telemetry
            )
        
        if result.message.startswith("❌"):
            console.print(f"[red]{result.message}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[green]{result.message}[/green]")
        
        if result.artifacts:
            console.print(f"\n📁 Generated files:")
            for artifact in result.artifacts:
                console.print(f"  • {artifact}")
        
        # Show quick stats if available
        if result.state_delta and "report_stats" in result.state_delta:
            stats = result.state_delta["report_stats"]
            console.print(f"\n📊 Report contains:")
            console.print(f"  • {stats.get('n_sections', 0)} sections")
            console.print(f"  • {stats.get('n_plots', 0)} visualizations")
            console.print(f"  • {stats.get('n_steps', 0)} workflow steps")
        
        console.print(Panel(
            "🎉 Report generated successfully!\n\n"
            "💡 Open the HTML file in your browser to view the full report.\n"
            "📤 Share the report file with collaborators or include in publications.",
            title="Success",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise typer.Exit(1)


@report_app.command("help")
def report_help() -> None:
    """Show detailed help for report commands."""
    console.print(Panel(
        "📊 **Report Commands**\n\n"
        "• `scqc report export` - Generate comprehensive workflow report\n\n"
        "📚 **Export Formats**\n\n"
        "• **HTML**: Interactive report with embedded plots and styling\n"
        "• **PDF**: Static report (requires additional setup)\n\n"
        "📋 **Report Contents**\n\n"
        "• Workflow timeline and progress\n"
        "• Key metrics and statistics\n"
        "• Generated plots and visualizations\n"
        "• Artifact catalog with links\n"
        "• Performance telemetry (optional)\n"
        "• System and package information\n\n"
        "💡 **Usage Examples**\n\n"
        "```bash\n"
        "# Basic HTML report\n"
        "scqc report export\n"
        "\n"
        "# Custom output location\n"
        "scqc report export --output my_analysis_report.html\n"
        "\n"
        "# Report without telemetry\n"
        "scqc report export --no-telemetry\n"
        "\n"
        "# PDF format (requires additional dependencies)\n"
        "scqc report export --format pdf\n"
        "```",
        title="Report Help",
        border_style="blue"
    ))


# ==================== EVALUATION COMMANDS ====================

@eval_app.command("run")
def eval_run(
    prompts_file: str = typer.Option(
        "eval/prompts.yaml",
        "--prompts", "-p",
        help="Path to prompts YAML file"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o", 
        help="Path to save results JSON"
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags", "-t",
        help="Comma-separated list of tags to filter prompts"
    ),
    no_optional: bool = typer.Option(
        False,
        "--no-optional",
        help="Skip optional tests that require extra dependencies"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Reduce output verbosity"
    ),
) -> None:
    """Run evaluation suite against golden prompts."""
    try:
        console.print(Panel(
            "🧪 **scQC Agent Evaluation Suite**\n\n"
            "Running automated tests against golden prompts to validate functionality.",
            title="Evaluation",
            border_style="blue"
        ))
        
        # Parse tags
        tags_filter = None
        if tags:
            tags_filter = [tag.strip() for tag in tags.split(",")]
            console.print(f"🏷️  Filtering by tags: {', '.join(tags_filter)}")
        
        if no_optional:
            console.print("⏭️  Skipping optional tests")
            
        # Run evaluation
        summary = run_evaluation(
            prompts_file=prompts_file,
            output_file=output_file,
            tags_filter=tags_filter,
            include_optional=not no_optional,
            verbose=not quiet
        )
        
        # Display summary in a nice table
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Prompts", str(summary.total_prompts))
        table.add_row("Passed", f"[green]{summary.passed_prompts}[/green]")
        table.add_row("Failed", f"[red]{summary.failed_prompts}[/red]")
        table.add_row("Skipped", f"[yellow]{summary.skipped_prompts}[/yellow]")
        table.add_row("Overall Pass Rate", f"{summary.pass_rate:.1%}")
        table.add_row("Core Pass Rate", f"{summary.core_pass_rate:.1%}")
        table.add_row("Optional Pass Rate", f"{summary.optional_pass_rate:.1%}")
        table.add_row("Total Time", f"{summary.total_execution_time:.1f}s")
        
        console.print(table)
        
        # Show failed tests if any
        if summary.failed_prompts > 0:
            console.print("\n❌ **Failed Tests:**")
            for result in summary.results:
                if result.passed is False:
                    console.print(f"  • {result.prompt_id}: {result.error_message or 'Acceptance criteria not met'}")
        
        # Determine exit status
        if summary.pass_rate >= 0.95:
            console.print(Panel(
                f"✅ **SUCCESS**: Pass rate {summary.pass_rate:.1%} meets 95% threshold",
                title="Evaluation Complete",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"❌ **FAILURE**: Pass rate {summary.pass_rate:.1%} below 95% threshold",
                title="Evaluation Complete", 
                border_style="red"
            ))
            raise typer.Exit(1)
            
        if output_file:
            console.print(f"\n📄 Detailed results saved to: {output_file}")
            
    except Exception as e:
        console.print(f"[red]Error running evaluation: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command("help")
def eval_help() -> None:
    """Show detailed help for evaluation commands."""
    console.print(Panel(
        "🧪 **Evaluation Commands**\n\n"
        "• `scqc eval run` - Run evaluation suite against golden prompts\n\n"
        "📋 **Evaluation Features**\n\n"
        "• **Golden Prompts**: Curated test cases covering key functionality\n"
        "• **Acceptance Criteria**: Automated checking of expected outcomes\n"
        "• **Quality Gates**: Runtime assertions for data integrity\n"
        "• **Synthetic Data**: Generated test datasets for reproducible testing\n"
        "• **Pass Rate Reporting**: Detailed metrics and failure analysis\n\n"
        "🏷️  **Tag Filtering**\n\n"
        "Common tags: qc, graph, scvi, doublets, basic, advanced, optional\n\n"
        "💡 **Usage Examples**\n\n"
        "```bash\n"
        "# Run all tests\n"
        "scqc eval run\n"
        "\n"
        "# Run only QC tests\n"
        "scqc eval run --tags qc\n"
        "\n"
        "# Skip optional tests (e.g., scVI)\n"
        "scqc eval run --no-optional\n"
        "\n"
        "# Save detailed results\n"
        "scqc eval run --output eval_results.json\n"
        "\n"
        "# Quiet mode for CI\n"
        "scqc eval run --quiet\n"
        "```\n\n"
        "🎯 **Pass Criteria**\n\n"
        "• Overall pass rate: ≥95%\n"
        "• Core functionality: 100% (QC, graph basics)\n"
        "• Optional features: ≥80% (scVI, advanced tools)",
        title="Evaluation Help",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
