# streamlit_app.py
"""Streamlit interface for the scAgent plan/execute workflow with multi-file support."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from scqc_agent.agent.runtime import Agent

# Persisted resources
SESSIONS_DIR = Path("streamlit_sessions")
UPLOAD_DIR = Path("runs/streamlit_uploads")

# File type helpers for artifact rendering
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}

LLM_PROVIDER_OPTIONS = {
    "Local Ollama": "ollama",
    "OpenAI API": "openai",
}


def get_default_llm_provider() -> str:
    """Return the default provider using environment configuration."""
    configured = os.getenv("LLM_PROVIDER", "ollama")
    normalized = configured.strip().lower() if configured else "ollama"
    valid_values = set(LLM_PROVIDER_OPTIONS.values())
    return normalized if normalized in valid_values else "ollama"


def get_provider_label(provider_key: str) -> str:
    """Return the display label for the given provider key."""
    for label, key in LLM_PROVIDER_OPTIONS.items():
        if key == provider_key:
            return label
    return "Local Ollama"


def get_available_sessions() -> List[Dict[str, Any]]:
    """Get list of available session files."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
                sessions.append({
                    "filename": session_file.name,
                    "path": str(session_file),
                    "run_id": data.get("run_id", "unknown"),
                    "created_at": data.get("created_at", "unknown"),
                    "updated_at": data.get("updated_at", "unknown"),
                    "n_cells": data.get("dataset_summary", {}).get("n_cells", 0),
                })
        except Exception as e:
            st.warning(f"Could not read session {session_file.name}: {e}")

    # Sort by updated_at (most recent first)
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions


def create_new_session() -> str:
    """Create a new session file and return its path."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique session ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_file = SESSIONS_DIR / f"session_{timestamp}.json"

    return str(session_file)


def get_current_session_path() -> str:
    """Get the current session path from Streamlit session state."""
    if "current_session_path" not in st.session_state:
        # Check if there are existing sessions
        sessions = get_available_sessions()
        if sessions:
            # Use most recent session by default
            st.session_state.current_session_path = sessions[0]["path"]
        else:
            # Create a new session
            st.session_state.current_session_path = create_new_session()

    return st.session_state.current_session_path


def initialize_agent() -> Agent:
    """Create or retrieve the agent instance for this Streamlit session."""
    # Check if we need to reload agent due to session change
    current_session = get_current_session_path()
    provider = st.session_state.get("llm_provider", get_default_llm_provider())

    if (
        "agent" not in st.session_state
        or st.session_state.get("agent_session_path") != current_session
        or st.session_state.get("agent_provider") != provider
    ):
        try:
            Path(current_session).parent.mkdir(parents=True, exist_ok=True)
            st.session_state.agent = Agent(state_path=current_session, provider=provider)
            st.session_state.agent_session_path = current_session
            st.session_state.agent_provider = provider
        except Exception as e:
            st.session_state.agent = None
            st.session_state.agent_provider = None
            st.error(f"Failed to initialize agent: {e}")
            st.exception(e)
            st.stop()
    return st.session_state.agent


def initialize_app_state() -> None:
    """Ensure Streamlit session state has required structures."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_plan", None)
    st.session_state.setdefault("last_execution", None)
    st.session_state.setdefault("execution_history", [])
    st.session_state.setdefault("uploaded_files", {})
    st.session_state.setdefault("kidney_files_loaded", False)
    st.session_state.setdefault("processed_load_messages", [])
    st.session_state.setdefault("llm_provider", get_default_llm_provider())
    st.session_state.setdefault("agent_provider", None)


def render_sidebar(agent: Agent) -> None:
    """Render sidebar with session metadata and dataset summaries."""
    st.sidebar.header("🤖 LLM Provider")
    provider_labels = list(LLM_PROVIDER_OPTIONS.keys())
    current_provider = st.session_state.get("llm_provider", get_default_llm_provider())
    current_label = get_provider_label(current_provider)
    selected_label = st.sidebar.selectbox(
        "Choose the language model backend:",
        options=provider_labels,
        index=provider_labels.index(current_label),
        help="Switch between the local Ollama model and the OpenAI API",
    )
    selected_provider = LLM_PROVIDER_OPTIONS[selected_label]
    st.sidebar.caption(f"Active backend: {selected_label}")
    if selected_provider != current_provider:
        st.session_state.llm_provider = selected_provider
        st.session_state.pop("agent", None)
        st.session_state.pop("agent_session_path", None)
        st.session_state.pop("agent_provider", None)
        st.sidebar.success(f"Switching to {selected_label}...")
        st.rerun()
    if selected_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OPENAI_API_KEY is not set. OpenAI requests will fail until configured.")

    st.sidebar.divider()
    st.sidebar.header("🗂️ Session Management")

    # Display current session
    current_session = get_current_session_path()
    session_name = Path(current_session).stem
    st.sidebar.write(f"**Current Session**: `{session_name}`")
    st.sidebar.write(f"**Run ID**: `{agent.state.run_id}`")

    # New session button
    if st.sidebar.button("🆕 New Session", help="Start a fresh analysis session"):
        new_session_path = create_new_session()
        st.session_state.current_session_path = new_session_path
        # Clear agent to force reload
        if "agent" in st.session_state:
            del st.session_state.agent
        # Clear chat history and other session data
        st.session_state.messages = []
        st.session_state.pending_plan = None
        st.session_state.last_execution = None
        st.session_state.kidney_files_loaded = False
        st.success(f"Created new session: {Path(new_session_path).stem}")
        st.rerun()

    # Session selector
    sessions = get_available_sessions()
    if sessions:
        st.sidebar.divider()
        st.sidebar.subheader("📋 Load Session")

        # Create options for selectbox
        session_options = {}
        for sess in sessions:
            label = f"{sess['run_id']}"
            if sess['n_cells'] > 0:
                label += f" ({sess['n_cells']} cells)"
            label += f" - {sess['updated_at'][:10]}"
            session_options[label] = sess['path']

        selected_label = st.sidebar.selectbox(
            "Select a session to load:",
            options=list(session_options.keys()),
            index=list(session_options.values()).index(current_session) if current_session in session_options.values() else 0,
            help="Switch between existing analysis sessions"
        )

        selected_path = session_options[selected_label]

        # Check if selection changed
        if selected_path != current_session:
            st.session_state.current_session_path = selected_path
            # Clear agent to force reload
            if "agent" in st.session_state:
                del st.session_state.agent
            # Clear chat history for new session
            st.session_state.messages = []
            st.session_state.pending_plan = None
            st.session_state.last_execution = None
            st.session_state.kidney_files_loaded = False
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("📊 Dataset Info")

    dataset_summary = agent.state.dataset_summary or {}
    if dataset_summary:
        st.sidebar.subheader("Dataset Summary")
        st.sidebar.json(dataset_summary)

    metadata = agent.state.metadata or {}
    if metadata:
        st.sidebar.subheader("Metadata")
        st.sidebar.json(metadata)

    if agent.state.artifacts:
        st.sidebar.subheader("Artifacts")
        for path_str, label in agent.state.artifacts.items():
            st.sidebar.write(f"- {label}: `{path_str}`")


def save_uploaded_file(uploaded_file, file_type: str) -> Optional[Path]:
    """Persist an uploaded file and return the saved path."""
    if uploaded_file is None:
        return None

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        base = Path(uploaded_file.name)
        timestamp = int(time.time())
        destination = UPLOAD_DIR / f"{file_type}_{base.stem}_{timestamp}{base.suffix}"

        destination.write_bytes(uploaded_file.getbuffer())
        return destination
    except Exception as e:
        st.error(f"Failed to save {file_type} file: {e}")
        return None


def handle_kidney_data_upload(raw_path: Path, filtered_path: Path, metadata_path: Optional[Path] = None) -> str:
    """Generate the load command for kidney data files."""
    if metadata_path:
        return (f"Load my kidney dataset from {raw_path} (raw droplets), "
                f"{filtered_path} (filtered cells), and metadata from {metadata_path}")
    else:
        return (f"Load my kidney dataset from {raw_path} (raw droplets) and "
                f"{filtered_path} (filtered cells)")


def format_plan_message(plan_result: Dict[str, Any]) -> str:
    """Create a markdown summary for the planning response."""
    if not plan_result:
        return "Agent did not return a plan."

    if plan_result.get("error"):
        return f"⚠️ Plan failed: {plan_result['error']}"

    lines: List[str] = []
    intent = plan_result.get("intent", "unknown")
    lines.append(f"**Intent:** `{intent}`")

    plan_steps = plan_result.get("plan") or []
    if plan_steps:
        lines.append("")
        lines.append("**Proposed Plan:**")
        for idx, step in enumerate(plan_steps, start=1):
            tool = step.get("tool", "tool")
            description = step.get("description") or ""
            lines.append(f"{idx}. **{tool}** – {description}".strip())
            params = step.get("params")
            if params:
                pretty = json.dumps(params, indent=2)
                lines.append(f"```\n{pretty}\n```")

    plan_path = plan_result.get("plan_path")
    if plan_path:
        lines.append("")
        lines.append(f"_Plan stored at_: `{plan_path}`")

    return "\n".join(lines)


def handle_user_message(message: str, agent: Agent) -> None:
    """Run the planning phase for a user prompt and record transcript."""
    try:
        st.session_state.messages.append({"role": "user", "content": message})

        with st.spinner("Planning..."):
            plan_result = agent.chat(message, mode="plan")

        agent.save_state()

        st.session_state.pending_plan = {
            "message": message,
            "result": plan_result,
        }
        st.session_state.messages.append({"role": "assistant", "content": format_plan_message(plan_result)})
    except Exception as e:
        error_msg = f"❌ Failed to generate plan: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.error(error_msg)


def execute_pending_plan(agent: Agent) -> None:
    """Execute the stored plan if available and update transcript."""
    pending = st.session_state.pending_plan
    if not pending:
        return

    plan_result = pending.get("result") or {}
    plan_path = plan_result.get("plan_path")
    message = pending.get("message", "")

    try:
        with st.spinner("Executing plan..."):
            execution_result = agent.chat(message, mode="execute", plan_path=plan_path)

        agent.save_state()

        st.session_state.last_execution = execution_result
        st.session_state.execution_history.append(execution_result)

        if execution_result.get("error"):
            summary_text = f"❌ Execution failed: {execution_result['error']}"
        else:
            summary_lines = ["**Execution Complete**"]
            if execution_result.get("summary"):
                summary_lines.append(execution_result["summary"])

            tool_results = execution_result.get("tool_results") or []
            if tool_results:
                summary_lines.append("")
                summary_lines.append("**Tool Results:**")
                for tool in tool_results:
                    message_text = tool.get("message", "")
                    if message_text:
                        summary_lines.append(f"- {message_text}")

            validation = execution_result.get("validation")
            if validation:
                pretty_validation = json.dumps(validation, indent=2)
                summary_lines.append("")
                summary_lines.append("**Validation:**")
                summary_lines.append(f"```\n{pretty_validation}\n```")

            citations = execution_result.get("citations") or []
            if citations:
                summary_lines.append("")
                summary_lines.append("**Citations:**")
                for cite in citations:
                    summary_lines.append(f"- {cite}")

            summary_text = "\n".join(summary_lines)

        st.session_state.messages.append({"role": "assistant", "content": summary_text})
        st.session_state.pending_plan = None

    except Exception as e:
        error_msg = f"❌ Execution error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.error(error_msg)
        st.exception(e)


def render_plan_details() -> None:
    """Display plan details and provide execute controls."""
    pending = st.session_state.pending_plan
    if not pending:
        return

    plan_result = pending.get("result") or {}

    with st.expander("Pending Plan", expanded=True):
        st.write(f"Intent: `{plan_result.get('intent', 'unknown')}`")

        plan_steps = plan_result.get("plan") or []
        if plan_steps:
            for idx, step in enumerate(plan_steps, start=1):
                st.markdown(f"**Step {idx}: {step.get('tool', 'tool')}**")
                if step.get("description"):
                    st.write(step["description"])
                if step.get("params"):
                    st.json(step["params"])

        plan_path = plan_result.get("plan_path")
        if plan_path:
            st.caption(f"Plan path: `{plan_path}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Execute plan", type="primary", key="execute_btn"):
                execute_pending_plan(st.session_state.agent)
        with col2:
            if st.button("❌ Cancel plan", key="cancel_btn"):
                st.session_state.pending_plan = None
                st.rerun()


def render_execution_details() -> None:
    """Display information from the latest execution."""
    execution_result = st.session_state.last_execution
    if not execution_result:
        return

    with st.expander("Last Execution Summary", expanded=False):
        st.json(
            {k: v for k, v in execution_result.items() if k not in {"artifacts", "tool_results"}}
        )

        tool_results = execution_result.get("tool_results") or []
        if tool_results:
            st.subheader("Tool Outputs")
            for idx, result in enumerate(tool_results, start=1):
                st.markdown(f"**Result {idx}**")
                st.write(result.get("message", ""))
                if result.get("artifacts"):
                    st.write("Artifacts:")
                    for artifact in result["artifacts"]:
                        st.write(f"- `{artifact}`")


def render_artifacts(agent: Agent) -> None:
    """Surface artifacts from the latest execution and overall session."""
    execution_result = st.session_state.last_execution or {}
    latest_artifacts = execution_result.get("artifacts") or []

    if latest_artifacts:
        st.subheader("Latest Artifacts")
        for path_str in latest_artifacts:
            render_artifact(Path(path_str))

    if agent.state.artifacts:
        st.subheader("Artifact Catalog")
        for path_str, label in agent.state.artifacts.items():
            st.markdown(f"**{label}**")
            render_artifact(Path(path_str))


def render_artifact(path: Path) -> None:
    """Display or offer download for a specific artifact path."""
    if not path.exists():
        st.info(f"Artifact not found: `{path}`")
        return

    try:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            st.image(str(path), caption=path.name, use_container_width=True)
        else:
            with open(path, "rb") as handle:
                data = handle.read()
                st.download_button(
                    label=f"📥 Download {path.name}",
                    data=data,
                    file_name=path.name,
                    mime="application/octet-stream",
                    key=f"download-{path}-{time.time()}",
                )
    except Exception as e:
        st.error(f"Failed to render artifact {path.name}: {e}")


def render_history(agent: Agent) -> None:
    """Show workflow history with produced assets."""
    history = agent.state.history
    if not history:
        return

    st.subheader("Workflow History")
    for entry in history:
        step_label = entry.get("label", "step")
        timestamp = entry.get("timestamp", "")
        st.markdown(f"**{step_label}** — {timestamp}")
        if entry.get("artifacts"):
            for artifact in entry["artifacts"]:
                st.write(f"- {artifact.get('label', '')}: `{artifact.get('path', '')}`")


def render_kidney_file_upload(agent: Agent) -> None:
    """Render multi-file upload interface for kidney workflow."""
    st.subheader("🧬 Kidney Data Upload")

    st.info(
        "Upload the three files required for kidney scRNA-seq analysis:\n"
        "1. **Raw feature matrix** (.h5 file with raw droplets)\n"
        "2. **Filtered feature matrix** (.h5 file with filtered cells)\n"
        "3. **Metadata** (.csv or .xlsx file) - Optional"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        raw_file = st.file_uploader(
            "Raw droplets (.h5)",
            type=["h5", "h5ad"],
            key="raw_upload",
            help="Upload the raw feature-barcode matrix file"
        )

    with col2:
        filtered_file = st.file_uploader(
            "Filtered cells (.h5)",
            type=["h5", "h5ad"],
            key="filtered_upload",
            help="Upload the filtered feature-barcode matrix file"
        )

    with col3:
        metadata_file = st.file_uploader(
            "Metadata (optional)",
            type=["csv", "xlsx"],
            key="metadata_upload",
            help="Upload the metadata file (optional)"
        )

    # Check if files are ready to load
    if raw_file and filtered_file:
        if not st.session_state.kidney_files_loaded:
            if st.button("📊 Load Kidney Dataset", type="primary", key="load_kidney"):
                with st.spinner("Saving files..."):
                    raw_path = save_uploaded_file(raw_file, "raw")
                    filtered_path = save_uploaded_file(filtered_file, "filtered")
                    metadata_path = save_uploaded_file(metadata_file, "metadata") if metadata_file else None

                    if raw_path and filtered_path:
                        load_message = handle_kidney_data_upload(raw_path, filtered_path, metadata_path)
                        st.success(f"Files saved! Ready to load dataset.")

                        # Generate plan for loading
                        handle_user_message(load_message, agent)
                        st.session_state.kidney_files_loaded = True
                        st.rerun()
                    else:
                        st.error("Failed to save uploaded files")
        else:
            st.success("✅ Kidney dataset files uploaded and ready")
            if st.button("🔄 Upload Different Files", key="reset_files"):
                st.session_state.kidney_files_loaded = False
                st.session_state.uploaded_files = {}
                st.rerun()


def render_single_file_upload(agent: Agent) -> None:
    """Render single file upload interface."""
    st.subheader("📁 Single File Upload")

    uploaded_file = st.file_uploader(
        "Upload .h5ad file",
        type=["h5ad", "h5ad.gz"],
        key="single_upload",
        help="Upload a pre-processed AnnData file"
    )

    if uploaded_file:
        saved_path = save_uploaded_file(uploaded_file, "single")
        if saved_path:
            st.success(f"File saved to `{saved_path}`")
            if st.button("📊 Load Dataset", type="primary", key="load_single"):
                load_message = f"load {saved_path}"
                handle_user_message(load_message, agent)
                st.rerun()


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="scQC Agent Streamlit UI",
        page_icon="🧬",
        layout="wide"
    )

    try:
        initialize_app_state()
        agent = initialize_agent()

        render_sidebar(agent)

        st.title("🧬 scQC Agent - Kidney scRNA-seq Analysis")
        active_provider_label = get_provider_label(st.session_state.get("llm_provider", get_default_llm_provider()))
        st.caption(f"Active language model provider: {active_provider_label}")
        st.write(
            "Upload kidney scRNA-seq data, review the agent's analysis plan, "
            "and execute quality control workflows with real-time monitoring."
        )

        # File upload section
        tab1, tab2 = st.tabs(["🧬 Kidney Workflow (3 files)", "📁 Single File"])

        with tab1:
            render_kidney_file_upload(agent)

        with tab2:
            render_single_file_upload(agent)

        st.divider()

        # Chat interface
        chat_placeholder = st.container()
        with chat_placeholder:
            for msg in st.session_state.messages:
                role = msg.get("role", "assistant")
                with st.chat_message(role):
                    st.markdown(msg.get("content", ""))

            user_input = st.chat_input("Ask scQC Agent about your kidney data...")
            if user_input:
                handle_user_message(user_input, agent)
                st.rerun()

        # Plan and execution details
        render_plan_details()
        render_execution_details()

        # Artifacts and history
        col1, col2 = st.columns(2)
        with col1:
            render_artifacts(agent)
        with col2:
            render_history(agent)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
