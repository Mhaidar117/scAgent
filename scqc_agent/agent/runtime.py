"""LangChain-based agent runtime for scQC Agent (Phase 5)."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Import guard for optional dependencies
try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_community.llms import Ollama
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from ..state import SessionState, ToolResult
from .rag.retriever import HybridRetriever
from .schemas import validate_tool_input, get_tool_description, TOOL_SCHEMAS

# Import Phase 8 modules with guards
try:
    from ..qc.priors import suggest_thresholds, get_available_tissues
    TISSUE_PRIORS_AVAILABLE = True
except ImportError:
    TISSUE_PRIORS_AVAILABLE = False

try:
    from ..utils.ux import get_ux_manager
    UX_AVAILABLE = True
except ImportError:
    UX_AVAILABLE = False


class Agent:
    """LangChain-powered agent for handling scQC workflow messages."""
    
    def __init__(self, state_path: str, knowledge_base_path: Optional[str] = None):
        """Initialize agent with state file path and optional knowledge base.
        
        Args:
            state_path: Path to session state file
            knowledge_base_path: Path to knowledge base directory
        """
        self.state_path = state_path
        self.state: SessionState = SessionState()
        
        # Try to load existing state
        if Path(state_path).exists():
            self.load_state()
        
        # Initialize knowledge base and retriever
        self.kb_path = knowledge_base_path or str(Path(__file__).parent.parent.parent / "kb")
        self.retriever: Optional[HybridRetriever] = None
        self._init_retriever()
        
        # Initialize LangChain components
        self._init_chains()
        
        # Tool registry for binding existing tools
        self._init_tool_registry()
        
        # Add Phase 8 tools
        self._add_phase8_tools()
    
    def _init_retriever(self) -> None:
        """Initialize the hybrid retriever if KB exists."""
        try:
            if Path(self.kb_path).exists():
                self.retriever = HybridRetriever(self.kb_path)
        except Exception as e:
            print(f"Warning: Could not initialize retriever: {e}")
    
    def _init_chains(self) -> None:
        """Initialize LangChain LCEL chains for agent workflow."""
        if not LANGCHAIN_AVAILABLE:
            print("Warning: LangChain not available. Using fallback mode.")
            # Initialize chain attributes to None for fallback mode
            self.intent_chain = None
            self.plan_chain = None
            self.tool_exec_chain = None
            self.validation_chain = None
            self.summarize_chain = None
            self.llm = None
            return
        
        # Use local Ollama model (assuming it's available)
        try:
            self.llm = ChatOllama(model="llama3.1", temperature=0.1)
        except:
            # Fallback to a simple mock LLM for testing
            self.llm = None
        
        # Load prompts
        prompts_dir = Path(__file__).parent / "prompts"
        
        # Intent Classification Chain
        self.intent_chain = self._create_intent_chain(prompts_dir)
        
        # Plan Generation Chain  
        self.plan_chain = self._create_plan_chain(prompts_dir)
        
        # Tool Execution Chain
        self.tool_exec_chain = self._create_tool_exec_chain()
        
        # Validation Chain
        self.validation_chain = self._create_validation_chain()
        
        # Summarize Chain
        self.summarize_chain = self._create_summarize_chain(prompts_dir)
    
    def _create_intent_chain(self, prompts_dir: Path) -> Optional[Any]:
        """Create intent classification chain."""
        if not self.llm:
            return None
        
        intent_template = """
        Classify the user's intent from their message. Choose from:
        - load_data: Load or import data files
        - compute_qc: Compute quality control metrics
        - plot_qc: Generate QC visualizations
        - apply_filters: Apply QC filters to remove low-quality cells/genes
        - run_scar: Denoise ambient RNA with scAR
        - run_scvi: Batch correction and integration with scVI
        - detect_doublets: Detect multi-cell droplets
        - graph_analysis: Generate embeddings and clustering
        - other: General request or unclear intent
        
        User message: {message}
        Current state: {state_summary}
        
        Intent (one word only):
        """
        
        prompt = PromptTemplate(
            template=intent_template,
            input_variables=["message", "state_summary"]
        )
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_plan_chain(self, prompts_dir: Path) -> Optional[Any]:
        """Create plan generation chain."""
        if not self.llm:
            return None
        
        # Try to load the Jinja template
        plan_template_path = prompts_dir / "plan.j2"
        if plan_template_path.exists() and JINJA2_AVAILABLE:
            try:
                with open(plan_template_path, 'r') as f:
                    template_content = f.read()
                
                # Create a Jinja template and render it as a LangChain template
                jinja_template = Template(template_content)
                
                # Create a function that renders the Jinja template
                def render_plan_template(inputs):
                    rendered = jinja_template.render(
                        message=inputs.get("message", ""),
                        intent=inputs.get("intent", ""),
                        state_summary=inputs.get("state_summary", ""),
                        context=inputs.get("context", "")
                    )
                    return rendered
                
                # Use RunnableLambda to create a custom runnable
                template_runnable = RunnableLambda(render_plan_template)
                
                return template_runnable | self.llm | JsonOutputParser()
                
            except Exception as e:
                print(f"Warning: Could not load Jinja template: {e}")
                # Fall back to simple template
        
        # Fallback template with basic species detection
        plan_template = """
        Generate a step-by-step plan for the user's request. Consider the current state
        and available tools. Return a JSON list of steps. Do not include any comments with '//'. 
        
        User message: {message}
        Intent: {intent}
        Current state: {state_summary}
        Retrieved context: {context}
        
        IMPORTANT: If the user mentions "mouse" or "mouse data", set species="mouse" in compute_qc_metrics params.
        If the user mentions "human" or "human data", set species="human" in compute_qc_metrics params.
        
        Available tools:
        - load_data: Load AnnData files
        - compute_qc_metrics: Calculate QC metrics (species: human/mouse/other)
        - plot_qc: Generate QC plots
        - apply_qc_filters: Filter cells/genes
        - quick_graph: PCA→neighbors→UMAP→Leiden
        - run_scar: scAR denoising
        - run_scvi: scVI integration
        - detect_doublets: Doublet detection
        - apply_doublet_filter: Remove doublets
        - final_graph: Final analysis
        
        Return a JSON array of step objects with "tool", "description", and "params":
        """
        
        prompt = PromptTemplate(
            template=plan_template,
            input_variables=["message", "intent", "state_summary", "context"]
        )
        
        return prompt | self.llm | JsonOutputParser()
    
    def _create_tool_exec_chain(self) -> Optional[Any]:
        """Create tool execution chain."""
        return RunnableLambda(self._execute_tool_step)
    
    def _execute_tool_step(self, step: Dict[str, Any]) -> ToolResult:
        """Execute a single tool step."""
        tool_name = step.get("tool", "")
        params = step.get("params", {})
        
        if tool_name in self.tools:
            try:
                # Validate parameters if schema exists
                if tool_name in TOOL_SCHEMAS:
                    params = validate_tool_input(tool_name, params)
                
                # Execute the tool
                return self.tools[tool_name](params)
            except Exception as e:
                return ToolResult(
                    message=f"❌ Error executing {tool_name}: {str(e)}",
                    state_delta={},
                    artifacts=[],
                    citations=[]
                )
        else:
            return ToolResult(
                message=f"❌ Unknown tool: {tool_name}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
    
    def _create_validation_chain(self) -> Optional[Any]:
        """Create validation chain."""
        return RunnableLambda(self._validate_results)
    
    def _validate_results(self, results: List[ToolResult]) -> Dict[str, Any]:
        """Validate tool execution results."""
        total_steps = len(results)
        success_count = sum(1 for r in results if not r.message.startswith("❌"))
        error_count = total_steps - success_count
        
        return {
            "total_steps": total_steps,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / total_steps if total_steps > 0 else 0,
            "has_errors": error_count > 0
        }
    
    def _create_summarize_chain(self, prompts_dir: Path) -> Optional[Any]:
        """Create result summarization chain."""
        if not self.llm:
            return None
        
        # Use the detailed Jinja2 template for comprehensive summaries
        try:
            from jinja2 import Environment, FileSystemLoader
            env = Environment(loader=FileSystemLoader(str(prompts_dir)))
            template = env.get_template("summarize.j2")
            
            def render_summary(inputs):
                """Render summary using Jinja2 template."""
                import json
                
                # Parse JSON inputs back to objects for template
                try:
                    plan = json.loads(inputs["plan"]) if isinstance(inputs["plan"], str) else inputs["plan"]
                    tool_results = json.loads(inputs["tool_results"]) if isinstance(inputs["tool_results"], str) else inputs["tool_results"]
                except json.JSONDecodeError:
                    plan = []
                    tool_results = []
                
                artifacts = inputs["artifacts"].split(", ") if inputs["artifacts"] else []
                citations = inputs["citations"].split(", ") if inputs["citations"] else []
                
                return template.render(
                    message=inputs["message"],
                    plan=plan,
                    tool_results=tool_results,
                    artifacts=artifacts,
                    citations=citations,
                    state_summary=f"Run ID: {self.state.run_id}"
                )
            
            return render_summary
            
        except Exception as e:
            # Fallback to basic template if Jinja2 fails
            summary_template = """
            Summarize the workflow execution results for the user.
            
            Original request: {message}
            Plan executed: {plan}
            Tool results: {tool_results}
            Artifacts created: {artifacts}
            Citations used: {citations}
            
            Provide a clear, concise summary of what was accomplished:
            """
            
            prompt = PromptTemplate(
                template=summary_template,
                input_variables=["message", "plan", "tool_results", "artifacts", "citations"]
            )
            
            return prompt | self.llm | StrOutputParser()
    
    def _init_tool_registry(self) -> None:
        """Initialize registry of available tools."""
        self.tools = {
            "load_data": self._load_data_tool,
            "compute_qc_metrics": self._compute_qc_tool,
            "plot_qc": self._plot_qc_tool,
            "apply_qc_filters": self._apply_qc_filters_tool,
            "quick_graph": self._quick_graph_tool,
            "run_scar": self._run_scar_tool,
            "run_scvi": self._run_scvi_tool,
            "detect_doublets": self._detect_doublets_tool,
            "apply_doublet_filter": self._apply_doublet_filter_tool,
            "final_graph": self._final_graph_tool,
            "batch_diagnostics": self._batch_diagnostics_tool,
        }
    
    def _add_phase8_tools(self) -> None:
        """Add Phase 8 stretch goal tools to registry."""
        # Tissue-aware QC priors (integrated into existing tools)
        self.tools["suggest_tissue_thresholds"] = self._suggest_tissue_thresholds_tool
        
        # Ambient RNA correction tools
        self.tools["scar_ambient"] = self._scar_ambient_tool
        
        # Batch diagnostics tools
        self.tools["batch_diagnostics"] = self._batch_diagnostics_tool
        #self.tools["lisi_analysis"] = self._lisi_analysis_tool #removed to use only python implementation
        # self.tools["batch_diagnostics_summary"] = self._batch_diagnostics_summary_tool #removed to use only python implementation
    
    def load_state(self) -> None:
        """Load session state from file."""
        self.state = SessionState.load(self.state_path)
    
    def save_state(self) -> None:
        """Save current session state to file."""
        self.state.save(self.state_path)
    
    def chat(self, message: str, mode: str = "plan") -> Dict[str, Any]:
        """Enhanced chat interface with planning and execution phases.
        
        Args:
            message: Natural language message from user
            mode: "plan" (generate plan for approval) or "execute" (execute approved plan)
            
        Returns:
            Dictionary with plan or execution results
        """
        chat_run_dir = self._create_chat_run_dir()
        
        try:
            if mode == "plan":
                return self._planning_phase(message, chat_run_dir)
            elif mode == "execute":
                return self._execution_phase(message, chat_run_dir)
            else:
                return {"error": f"Unknown mode: {mode}"}
        except Exception as e:
            return {"error": str(e)}

    def _planning_phase(self, message: str, chat_run_dir: Path) -> Dict[str, Any]:
        """Planning phase - generate and present plan for user approval."""
        # Step 1: Intent Classification
        intent = self._classify_intent(message)
        
        # Step 2: Plan Generation (with RAG context)
        plan = self._generate_plan(message, intent)
        
        # Save planning artifacts
        self._save_chat_artifacts(chat_run_dir, {
            "message": message,
            "intent": intent,
            "plan": plan,
            "mode": "planning"
        })
        
        return {
            "message": message,
            "intent": intent,
            "plan": plan,
            "mode": "planning",
            "chat_run_dir": str(chat_run_dir),
            "status": "plan_ready",
            "next_steps": "Review the plan and use 'execute' to run it, or provide feedback to modify it."
        }

    def _execution_phase(self, message: str, chat_run_dir: Path) -> Dict[str, Any]:
        """Execution phase - execute the approved plan."""
        # Load the plan from the most recent planning session
        # (You'd need to implement plan storage/retrieval)
        
        # For now, regenerate plan if needed
        intent = self._classify_intent(message)
        plan = self._generate_plan(message, intent)
        
        # Step 3: Tool Execution
        tool_results = self._execute_plan(plan)
        
        # Step 4: Validation
        validation_results = self._validate_execution(tool_results)
        
        # Step 5: Summarization
        summary = self._summarize_results(message, plan, tool_results)
        
        # Save execution artifacts
        self._save_chat_artifacts(chat_run_dir, {
            "message": message,
            "intent": intent,
            "plan": plan,
            "tool_results": [r.model_dump() for r in tool_results],
            "validation": validation_results,
            "summary": summary,
            "mode": "execution"
        })
        
        # Collect all citations and artifacts
        all_citations = []
        all_artifacts = []
        for result in tool_results:
            all_citations.extend(result.citations)
            all_artifacts.extend([str(p) for p in result.artifacts])
        
        return {
            "message": message,
            "intent": intent,
            "plan": plan,
            "tool_results": [r.model_dump() for r in tool_results],
            "validation": validation_results,
            "summary": summary,
            "citations": all_citations,
            "artifacts": all_artifacts,
            "chat_run_dir": str(chat_run_dir),
            "mode": "execution",
            "status": "completed"
        }
        
    def _create_chat_run_dir(self) -> Path:
        """Create directory for this chat session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / self.state.run_id / f"chat_{len(self.state.history):03d}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _classify_intent(self, message: str) -> str:
        """Classify user intent."""
        if not self.intent_chain:
            # Fallback classification
            return self._fallback_classify_intent(message)
        
        state_summary = self._get_state_summary()
        result = self.intent_chain.invoke({
            "message": message,
            "state_summary": state_summary
        })
        return result.strip().lower()
    
    def _generate_plan(self, message: str, intent: str) -> List[Dict[str, Any]]:
        """Generate execution plan with RAG context and tissue-aware priors."""
        # Retrieve relevant context
        context = ""
        if self.retriever:
            try:
                docs = self.retriever.retrieve(message, k=3)
                context = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                print(f"Warning: Retrieval failed: {e}")
        
        # Enhance context with tissue-aware priors if available
        tissue_context = self._get_tissue_context(message)
        if tissue_context:
            context += "\n\nTissue-specific recommendations:\n" + tissue_context
        
        if not self.plan_chain:
            # Fallback planning with tissue awareness
            return self._fallback_generate_plan(message, intent)
        
        state_summary = self._get_state_summary()
        result = self.plan_chain.invoke({
            "message": message,
            "intent": intent,
            "state_summary": state_summary,
            "context": context
        })
        
        if isinstance(result, list):
            # Enhance plan with tissue-specific parameters
            enhanced_plan = self._enhance_plan_with_tissue_priors(result, message)
            return enhanced_plan
        else:
            # Fallback if JSON parsing failed
            return self._fallback_generate_plan(message, intent)
    
    def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute the generated plan."""
        results = []
        for step in plan:
            tool_name = step.get("tool", "")
            params = step.get("params", {})
            
            if tool_name in self.tools:
                try:
                    # Validate parameters using Pydantic schemas
                    if tool_name in TOOL_SCHEMAS:
                        try:
                            validated_params = validate_tool_input(tool_name, params)
                            params = validated_params
                        except ValueError as ve:
                            error_result = ToolResult(
                                message=f"❌ Invalid parameters for {tool_name}: {str(ve)}",
                                state_delta={},
                                artifacts=[],
                                citations=[]
                            )
                            results.append(error_result)
                            continue
                    
                    # Execute the tool (wrapper functions handle state internally)
                    result = self.tools[tool_name](params)
                    results.append(result)
                    
                    # CRITICAL FIX: Apply state_delta to session state after each tool
                    if result.state_delta:
                        # Update adata_path if it's in the state_delta
                        if "adata_path" in result.state_delta:
                            self.state.adata_path = result.state_delta["adata_path"]
                        
                        # Update metadata with any other state changes
                        self.state.update_metadata(result.state_delta)
                    
                    # Artifacts are handled by individual tools via state.add_artifact()
                    # No need for duplicate registration here
                except Exception as e:
                    error_result = ToolResult(
                        message=f"❌ Error executing {tool_name}: {str(e)}",
                        state_delta={},
                        artifacts=[],
                        citations=[]
                    )
                    results.append(error_result)
            else:
                error_result = ToolResult(
                    message=f"❌ Unknown tool: {tool_name}",
                    state_delta={},
                    artifacts=[],
                    citations=[]
                )
                results.append(error_result)
        
        # CRITICAL FIX: Save session state after executing all tools
        self.save_state()
        
        return results
    
    def _validate_execution(self, tool_results: List[ToolResult]) -> Dict[str, Any]:
        """Validate execution results."""
        validation = {
            "success_count": sum(1 for r in tool_results if not r.message.startswith("❌")),
            "error_count": sum(1 for r in tool_results if r.message.startswith("❌")),
            "total_artifacts": sum(len(r.artifacts) for r in tool_results),
            "has_errors": any(r.message.startswith("❌") for r in tool_results)
        }
        return validation
    
    def _summarize_results(self, message: str, plan: List[Dict[str, Any]], 
                          tool_results: List[ToolResult]) -> str:
        """Summarize the execution results."""
        # Force use of fallback for consistent structured output
        return self._fallback_summarize(message, plan, tool_results)
        
        # DISABLED: LLM-based summarization for debugging
        # if not self.summarize_chain:
        #     return self._fallback_summarize(message, plan, tool_results)
        # 
        # all_artifacts = []
        # all_citations = []
        # for result in tool_results:
        #     all_artifacts.extend([str(p) for p in result.artifacts])
        #     all_citations.extend(result.citations)
        # 
        # # Call the render function directly since we changed it to return a function
        # summary = self.summarize_chain({
        #     "message": message,
        #     "plan": json.dumps(plan, indent=2),
        #     "tool_results": json.dumps([r.model_dump() for r in tool_results], indent=2),
        #     "artifacts": ", ".join(all_artifacts),
        #     "citations": ", ".join(all_citations)
        # })
        # 
        # return summary
    
    def _save_chat_artifacts(self, run_dir: Path, data: Dict[str, Any]) -> None:
        """Save chat session artifacts."""
        # Save messages
        with open(run_dir / "messages.json", "w") as f:
            json.dump(data, f, indent=2)
        
        # Save plan separately for easy access
        if "plan" in data:
            with open(run_dir / "plan.json", "w") as f:
                json.dump(data["plan"], f, indent=2)
    
    def _get_state_summary(self) -> str:
        """Get a summary of current session state."""
        summary = f"Run ID: {self.state.run_id}\n"
        summary += f"Steps completed: {len(self.state.history)}\n"
        summary += f"Artifacts: {len(self.state.artifacts)}\n"
        
        if self.state.metadata:
            if "adata_path" in self.state.metadata:
                summary += f"Data loaded: {self.state.metadata['adata_path']}\n"
        
        return summary
    
    # Fallback methods for when LangChain is not available
    def _fallback_classify_intent(self, message: str) -> str:
        """Fallback intent classification using keywords."""
        text_lower = message.lower()
        
        if any(keyword in text_lower for keyword in ["load", "data", "file", "import"]):
            return "load_data"
        elif any(keyword in text_lower for keyword in ["qc", "quality", "metrics"]):
            return "compute_qc"
        elif any(keyword in text_lower for keyword in ["plot", "visualiz", "graph", "show"]):
            return "plot_qc"
        elif any(keyword in text_lower for keyword in ["filter", "clean", "remove", "apply"]):
            return "apply_filters"
        elif any(keyword in text_lower for keyword in ["scar", "denois", "ambient"]):
            return "run_scar"
        elif any(keyword in text_lower for keyword in ["scvi", "batch", "integrat"]):
            return "run_scvi"
        elif any(keyword in text_lower for keyword in ["doublet", "multi-cell"]):
            return "detect_doublets"
        elif any(keyword in text_lower for keyword in ["umap", "embedding", "cluster", "leiden"]):
            return "graph_analysis"
        else:
            return "other"
    
    def _fallback_generate_plan(self, message: str, intent: str) -> List[Dict[str, Any]]:
        """Fallback plan generation using rule-based logic."""
        plan = []
        
        # Extract species from message
        species = self._extract_species_from_message(message)
        
        if intent == "load_data":
            plan.append({
                "tool": "load_data",
                "description": "Load AnnData file",
                "params": {}
            })
        elif intent == "compute_qc":
            params = {}
            if species:
                params["species"] = species
            plan.append({
                "tool": "compute_qc_metrics",
                "description": f"Compute QC metrics{f' for {species} data' if species else ''}",
                "params": params
            })
        elif intent == "plot_qc":
            # For plot_qc + compute_qc combo (like your "mouse data" message)
            params = {}
            if species:
                params["species"] = species
            
            plan.append({
                "tool": "plot_qc",
                "description": "Generate QC plots",
                "params": {"stage": "pre"}
            })
            plan.append({
                "tool": "compute_qc_metrics",
                "description": f"Compute QC metrics{f' for {species} data' if species else ''}",
                "params": params
            })
        elif intent == "apply_filters":
            plan.append({
                "tool": "apply_qc_filters",
                "description": "Apply QC filters",
                "params": {}
            })
        elif intent == "graph_analysis":
            plan.append({
                "tool": "quick_graph",
                "description": "Quick graph analysis",
                "params": {}
            })
        else:
            # Default case - include species if detected
            params = {}
            if species:
                params["species"] = species
            plan.append({
                "tool": "compute_qc_metrics",
                "description": f"Default: Compute QC metrics{f' for {species} data' if species else ''}",
                "params": params
            })
        
        return plan
    
    def _extract_species_from_message(self, message: str) -> Optional[str]:
        """Extract species information from user message."""
        message_lower = message.lower()
        
        if "mouse" in message_lower:
            return "mouse"
        elif "human" in message_lower:
            return "human"
        else:
            return None
    
    def _fallback_summarize(self, message: str, plan: List[Dict[str, Any]], 
                           tool_results: List[ToolResult]) -> str:
        """Fallback summarization with detailed output in the style the user wants."""
        success_count = sum(1 for r in tool_results if not r.message.startswith("❌"))
        error_count = sum(1 for r in tool_results if r.message.startswith("❌"))
        total_count = len(tool_results)
        
        # Collect all artifacts and citations
        all_artifacts = []
        all_citations = []
        successful_steps = []
        error_steps = []
        
        for i, result in enumerate(tool_results, 1):
            all_artifacts.extend([str(p) for p in result.artifacts])
            all_citations.extend(result.citations)
            
            if result.message.startswith("❌"):
                error_steps.append(f"{i}. {result.message}")
            else:
                # Extract meaningful accomplishment from message
                message_clean = result.message.replace("✅", "").strip()
                successful_steps.append(message_clean)
        
        # Build the structured summary
        if "qc" in message.lower() or "quality" in message.lower():
            workflow_type = "QC workflow"
        elif "doublet" in message.lower():
            workflow_type = "doublet detection workflow"
        else:
            workflow_type = "workflow"
        
        summary = f"The {workflow_type} has been {'partially ' if error_count > 0 else ''}executed"
        if error_count > 0:
            summary += " with some errors"
        summary += ".\n\n"
        
        # Accomplishments section
        if successful_steps:
            summary += "**Accomplishments:**\n\n"
            for i, step in enumerate(successful_steps, 1):
                # Extract key information from step messages
                if "QC metrics computed" in step:
                    summary += f"{i}. **QC metrics computed**: {step}\n"
                elif "Generated" in step and "plot" in step:
                    summary += f"{i}. **QC plots generated**: {step}\n"
                elif "doublet" in step.lower():
                    summary += f"{i}. **Doublet detection**: {step}\n"
                elif "filter" in step.lower():
                    summary += f"{i}. **Data filtering**: {step}\n"
                else:
                    summary += f"{i}. **{step.split(':')[0] if ':' in step else 'Analysis step'}**: {step}\n"
            
            # Add artifacts info
            artifact_count = len(all_artifacts)
            if artifact_count > 0:
                summary += f"{len(successful_steps) + 1}. **Artifacts created**: {artifact_count} files were generated, including data snapshots, plots, and summary files.\n"
            summary += "\n"
        
        # Errors section
        if error_steps:
            summary += "**Errors:**\n\n"
            for i, error in enumerate(error_steps, 1):
                # Clean up error message
                error_clean = error.replace("❌", "").strip()
                if "Error executing" in error_clean:
                    error_clean = error_clean.replace("Error executing ", "").replace(":", " failed:")
                summary += f"{i}. **{error_clean}**\n"
            summary += "\n"
        
        # Citations
        if all_citations:
            # Remove duplicates while preserving order
            unique_citations = list(dict.fromkeys(all_citations))
            if len(unique_citations) == 1:
                summary += f"**Citation used:** {unique_citations[0]}\n\n"
            else:
                summary += "**Citations used:** " + ", ".join(unique_citations) + "\n\n"
        
        return summary
    
    # Phase 8 enhancement methods
    def _get_tissue_context(self, message: str) -> str:
        """Extract tissue-specific context from message and provide recommendations."""
        if not TISSUE_PRIORS_AVAILABLE:
            return ""
        
        # Simple tissue detection from message
        message_lower = message.lower()
        tissue_keywords = {
            "brain": ["brain", "neural", "neuron", "cortex", "hippocampus"],
            "pbmc": ["pbmc", "blood", "immune", "lymphocyte", "monocyte"],
            "liver": ["liver", "hepatic", "hepatocyte"],
            "heart": ["heart", "cardiac", "cardiomyocyte"],
            "kidney": ["kidney", "renal", "nephron"],
            "lung": ["lung", "pulmonary", "respiratory"],
            "intestine": ["intestine", "gut", "epithelial"],
            "skin": ["skin", "dermal", "epidermal"],
            "tumor": ["tumor", "cancer", "malignant"],
            "embryonic": ["embryonic", "developmental", "embryo"]
        }
        
        detected_tissue = None
        for tissue, keywords in tissue_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_tissue = tissue
                break
        
        if not detected_tissue:
            return ""
        
        try:
            # Get tissue-specific thresholds
            thresholds = suggest_thresholds(tissue=detected_tissue, stringency="default")
            
            context_lines = [
                f"Detected tissue type: {detected_tissue}",
                f"Recommended QC thresholds for {detected_tissue}:",
                f"  - Min genes per cell: {thresholds.get('min_genes', 200)}",
                f"  - Max genes per cell: {thresholds.get('max_genes', 7000)}",
                f"  - Max mitochondrial %: {thresholds.get('max_pct_mt', 20.0)}",
                f"  - Expected doublet rate: {thresholds.get('doublet_rate', 0.08):.1%}",
                f"  - Notes: {thresholds.get('notes', 'Standard tissue-specific thresholds')}"
            ]
            
            return "\n".join(context_lines)
            
        except Exception as e:
            return f"Detected tissue: {detected_tissue} (error getting thresholds: {e})"
    
    def _enhance_plan_with_tissue_priors(self, plan: List[Dict[str, Any]], message: str) -> List[Dict[str, Any]]:
        """Enhance plan steps with tissue-specific parameters."""
        # Always enhance with species detection, even if tissue priors not available
        enhanced_plan = []
        
        # Extract species from message as a safety net
        detected_species = self._extract_species_from_message(message)
        
        # Detect tissue from message if tissue priors available
        detected_tissue = None
        thresholds = {}
        if TISSUE_PRIORS_AVAILABLE:
            message_lower = message.lower()
            detected_tissue = self._detect_tissue_from_message(message_lower)
            
            if detected_tissue:
                try:
                    # Get tissue-specific thresholds
                    thresholds = suggest_thresholds(tissue=detected_tissue, stringency="default")
                except Exception:
                    thresholds = {}
        
        # Enhance each step with species and tissue-specific parameters
        for step in plan:
            enhanced_step = step.copy()
            tool = step.get("tool", "")
            params = step.get("params", {}).copy()
            
            # Enhance QC-related tools
            if tool == "compute_qc_metrics":
                # CRITICAL: Add species detection if not specified
                if "species" not in params or not params["species"]:
                    if detected_species:
                        params["species"] = detected_species
                        print(f"🐭 Enhanced plan: Setting species={detected_species} for compute_qc_metrics")
                    # If no species detected, let auto-detection handle it (don't set a default)
                
            elif tool == "apply_qc_filters" and thresholds:
                # Suggest tissue-specific thresholds
                if "min_genes" not in params:
                    params["min_genes"] = thresholds.get("min_genes", 200)
                if "max_genes" not in params:
                    params["max_genes"] = thresholds.get("max_genes", 7000)
                if "max_pct_mt" not in params:
                    params["max_pct_mt"] = thresholds.get("max_pct_mt", 20.0)
                
                # Add tissue context to description
                if detected_tissue:
                    enhanced_step["description"] = f"{step.get('description', 'Apply QC filters')} (using {detected_tissue} tissue priors)"
            
            elif tool == "detect_doublets" and thresholds:
                # Use tissue-specific doublet rate
                if "expected_doublet_rate" not in params:
                    params["expected_doublet_rate"] = thresholds.get("doublet_rate", 0.08)
                
                if detected_tissue:
                    enhanced_step["description"] = f"{step.get('description', 'Detect doublets')} (expected rate: {thresholds.get('doublet_rate', 0.08):.1%} for {detected_tissue})"
            
            enhanced_step["params"] = params
            enhanced_plan.append(enhanced_step)
        
        return enhanced_plan
    
    def _detect_tissue_from_message(self, message_lower: str) -> Optional[str]:
        """Detect tissue type from message text."""
        tissue_keywords = {
            "brain": ["brain", "neural", "neuron", "cortex", "hippocampus"],
            "pbmc": ["pbmc", "blood", "immune", "lymphocyte", "monocyte"],
            "liver": ["liver", "hepatic", "hepatocyte"],
            "heart": ["heart", "cardiac", "cardiomyocyte"],
            "kidney": ["kidney", "renal", "nephron"],
            "lung": ["lung", "pulmonary", "respiratory"],
            "intestine": ["intestine", "gut", "epithelial"],
            "skin": ["skin", "dermal", "epidermal"],
            "tumor": ["tumor", "cancer", "malignant"],
            "embryonic": ["embryonic", "developmental", "embryo"]
        }
        
        for tissue, keywords in tissue_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return tissue
        
        return None

    # Tool wrapper methods - these integrate with existing tool modules
    def _load_data_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Load data tool wrapper."""
        try:
            from ..tools.io import load_anndata
            file_path = params.get('file_path')
            if not file_path:
                return ToolResult(
                    message="❌ Error: file_path parameter is required",
                    state_delta={},
                    artifacts=[],
                    citations=[]
                )
            
            # Call the real load function
            result = load_anndata(file_path)
            
            # Update session state if successful
            if not result.message.startswith("❌"):
                self.state.adata_path = file_path
                # Merge state_delta
                result.state_delta.update({"adata_path": file_path})
            
            return result
            
        except ImportError:
            return ToolResult(
                message="❌ Data loading tools not available",
                state_delta={},
                artifacts=[],
                citations=[]
            )
    
        
    def _validate_data_loaded(self) -> bool:
        """Check if AnnData file is loaded."""
        return bool(self.state.adata_path and Path(self.state.adata_path).exists())

    def _compute_qc_tool(self, params: Dict[str, Any]) -> ToolResult:
        """QC computation tool wrapper."""
        if not self._validate_data_loaded():
            return ToolResult(
                message="❌ No AnnData file loaded. Use 'load_data' first.",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        from ..tools.qc import compute_qc_metrics
        return compute_qc_metrics(self.state, **params)
    
    def _plot_qc_tool(self, params: Dict[str, Any]) -> ToolResult:
        """QC plotting tool wrapper."""
        from ..tools.qc import plot_qc_metrics
        return plot_qc_metrics(self.state, **params)
    
    def _apply_qc_filters_tool(self, params: Dict[str, Any]) -> ToolResult:
        """QC filtering tool wrapper."""
        from ..tools.qc import apply_qc_filters
        return apply_qc_filters(self.state, **params)
    
    def _quick_graph_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Quick graph tool wrapper."""
        from ..tools.graph import quick_graph
        return quick_graph(self.state, **params)
    
    def _run_scar_tool(self, params: Dict[str, Any]) -> ToolResult:
        """scAR tool wrapper."""
        from ..tools.scar import run_scar
        return run_scar(self.state, **params)
    
    def _run_scvi_tool(self, params: Dict[str, Any]) -> ToolResult:
        """scVI tool wrapper."""
        from ..tools.scvi import run_scvi
        return run_scvi(self.state, **params)
    
    def _detect_doublets_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Doublet detection tool wrapper."""
        from ..tools.doublets import detect_doublets
        return detect_doublets(self.state, **params)
    
    def _apply_doublet_filter_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Doublet filtering tool wrapper."""
        from ..tools.doublets import apply_doublet_filter
        return apply_doublet_filter(self.state, **params)
    
    def _final_graph_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Final graph tool wrapper."""
        from ..tools.graph import final_graph
        return final_graph(self.state, **params)
    
    # Phase 8 tool wrappers
    def _suggest_tissue_thresholds_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Tissue-aware threshold suggestion tool wrapper."""
        if not TISSUE_PRIORS_AVAILABLE:
            return ToolResult(
                message="❌ Tissue priors module not available",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
        try:
            tissue = params.get("tissue", "default")
            stringency = params.get("stringency", "default")
            species = params.get("species", "human")
            
            thresholds = suggest_thresholds(
                tissue=tissue, 
                stringency=stringency, 
                species=species
            )
            
            message = (
                f"✅ Tissue-aware thresholds suggested for {tissue} tissue.\n"
                f"   Min genes: {thresholds.get('min_genes', 200)}\n"
                f"   Max genes: {thresholds.get('max_genes', 7000)}\n"
                f"   Max pct MT: {thresholds.get('max_pct_mt', 20.0)}%\n"
                f"   Doublet rate: {thresholds.get('doublet_rate', 0.08):.1%}"
            )
            
            return ToolResult(
                message=message,
                state_delta={"tissue_thresholds": thresholds},
                artifacts=[],
                citations=["Tissue-specific QC threshold recommendations"]
            )
            
        except Exception as e:
            return ToolResult(
                message=f"❌ Tissue threshold suggestion failed: {str(e)}",
                state_delta={},
                artifacts=[],
                citations=[]
            )
    
    def _scar_ambient_tool(self, params: Dict[str, Any]) -> ToolResult:
        """scAR ambient RNA correction tool wrapper."""
        try:
            from ..tools.ambient import scar_ambient_removal
            return scar_ambient_removal(self.state, **params)
        except ImportError:
            return ToolResult(
                message="❌ scAR ambient RNA correction not available. Install with: pip install scar",
                state_delta={},
                artifacts=[],
                citations=[]
            )
    
    def _compare_ambient_methods_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Ambient RNA method comparison tool wrapper."""
        try:
            from ..tools.ambient import compare_ambient_methods
            return compare_ambient_methods(self.state, **params)
        except ImportError:
            return ToolResult(
                message="❌ Ambient RNA correction tools not available",
                state_delta={},
                artifacts=[],
                citations=[]
            )
    
    def _kbet_analysis_tool(self, params: Dict[str, Any]) -> ToolResult:
        """kBET batch diagnostics tool wrapper."""
        try:
            from ..tools.batch_diag import kbet_analysis
            return kbet_analysis(self.state, **params)
        except ImportError:
            return ToolResult(
                message="❌ Batch diagnostics tools not available",
                state_delta={},
                artifacts=[],
                citations=[]
            )
    
    def _lisi_analysis_tool(self, params: Dict[str, Any]) -> ToolResult:
        """LISI batch diagnostics tool wrapper."""
        try:
            from ..tools.batch_diag import lisi_analysis
            return lisi_analysis(self.state, **params)
        except ImportError:
            return ToolResult(
                message="❌ Batch diagnostics tools not available",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        
    # Replace the existing _kbet_analysis_tool and _lisi_analysis_tool (lines 891-916) with:
    def _batch_diagnostics_tool(self, params: Dict[str, Any]) -> ToolResult:
        """Batch diagnostics tool using scib-metrics."""
        try:
            from ..tools.batch_diag_scib import run_batch_diagnostics, DiagConfig
            import scanpy as sc
            
            # Load data
            adata = sc.read_h5ad(self.state.adata_path)
                            
            # Create config from parameters
            config = DiagConfig(
                rep_key=params.get('embedding_key', 'X_scVI'),
                label_key=params.get('label_key', 'cell_type'),
                batch_key=params.get('batch_key', 'batch'),
                n_neighbors=params.get('n_neighbors', 15),
                subsample=params.get('subsample', None),
                seed=params.get('seed', 0)
            )
            
            # Run diagnostics
            results = run_batch_diagnostics(adata, config)
            
            # Create step directory for artifacts
            from ..tools.io import ensure_run_dir
            step_dir = ensure_run_dir(self.state.run_id, "batch_diagnostics")
            
            # Update state
            state_delta = {
                "last_tool": "batch_diagnostics",
                "batch_diag_results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Format message
            message_parts = [f"✅ Batch diagnostics completed using {config.rep_key}"]
            
            if results.get("has_scib"):
                message_parts.extend([
                    f"   kBET acceptance: {results.get('kbet_acceptance', 0):.3f}",
                    f"   iLISI (batch mixing): {results.get('ilisi', 0):.2f}",
                    f"   cLISI (structure preservation): {results.get('clisi', 0):.2f}",
                    f"   Graph connectivity: {results.get('graph_connectivity', 0):.3f}"
                ])
            else:
                message_parts.extend([
                    f"   ASW batch: {results.get('asw_batch', 0):.3f}",
                    f"   NN batch purity: {results.get('nn_batch_purity', 0):.3f}",
                    f"   NN batch entropy: {results.get('nn_batch_entropy', 0):.3f}"
                ])
            
            message_parts.append(f"   Evaluated {results.get('n_cells_eval', 0):,} cells")
            
            return ToolResult(
                message="\n".join(message_parts),
                state_delta=state_delta,
                artifacts=[],  # Your scib implementation already saves to files
                citations=["Luecken et al. (2022) Benchmarking atlas-level data integration"]
            )
            
        except ImportError:
            return ToolResult(
                message="❌ scib-metrics not available. Install with: pip install scib-metrics",
                state_delta={},
                artifacts=[],
                citations=[]
            )
        except Exception as e:
            return ToolResult(
                message=f"❌ Batch diagnostics failed: {str(e)}",
                state_delta={},
                artifacts=[],
                citations=[]
            )

    # Legacy method for backward compatibility
    def handle_message(self, text: str) -> Dict[str, Any]:
        """Handle a natural language message (legacy interface)."""
        result = self.chat(text)
        
        # Convert to legacy format
        plan_steps = []
        if "plan" in result:
            for i, step in enumerate(result["plan"], 1):
                plan_steps.append(f"{i}. {step.get('description', 'Unknown step')}")
        
        return {
            "message": text,
            "plan": plan_steps,
            "tool_results": result.get("tool_results", []),
            "status": result.get("status", "completed")
        }

