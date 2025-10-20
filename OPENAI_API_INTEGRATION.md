# OpenAI API Integration

This document outlines the OpenAI backend integration for scAgent, including configuration requirements and Streamlit UI behavior.

## Summary of Changes

- Introduced an `LLM_PROVIDER` environment flag that selects between the existing Ollama backend (`ollama`) and the new OpenAI backend (`openai`).
- Updated the agent runtime to lazily import `langchain_openai.ChatOpenAI` only when the OpenAI provider is active and to emit clear runtime errors when `OPENAI_API_KEY` is missing.
- Added an OpenAI-aware initialization path that accepts optional overrides via `OPENAI_MODEL` and `OPENAI_API_BASE`.
- Expanded the Streamlit app sidebar with a provider toggle and active backend status, including automatic agent reloads when the provider changes.

## Configuration Steps

1. **Install dependencies** (only required for OpenAI usage):
   ```bash
   pip install langchain-openai
   ```
2. **Select the provider**:
   ```bash
   export LLM_PROVIDER=openai  # use "ollama" to force local execution (default)
   ```
3. **Set the OpenAI credentials**:
   ```bash
   export OPENAI_API_KEY=sk-...
   export OPENAI_MODEL=gpt-4o-mini      # optional, defaults to gpt-4o-mini
   export OPENAI_API_BASE=https://...   # optional for Azure/OpenAI proxies
   ```

If `LLM_PROVIDER` is unset or set to `ollama`, the agent follows the existing local Ollama path and no OpenAI configuration is required.

## Streamlit UI Workflow

- A new **LLM Provider** section appears at the top of the sidebar with a selector for **Local Ollama** or **OpenAI API**.
- The current backend is displayed in both the sidebar and beneath the main page title so users can confirm which model is active.
- Changing the provider clears the cached agent instance, reloads the session state, and re-creates the agent with the selected backend.
- If the OpenAI provider is selected without an `OPENAI_API_KEY`, the sidebar displays an inline warning before the backend attempts to initialize.

## Error Handling Expectations

- Missing OpenAI credentials raise a runtime error during agent initialization, ensuring Streamlit surfaces a clear message to the user.
- When `langchain-openai` is not installed, the agent instructs the user to add the dependency.
- Ollama remains the default provider; if the Ollama service is unavailable, the agent falls back gracefully with warning logs, matching the prior behavior.
