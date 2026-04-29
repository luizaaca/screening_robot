# Screening Robot Agent

A stateful clinical screening assistant built with LangGraph for demo and local development scenarios. The project combines patient chart lookup from SQLite with symptom analysis powered by either a local GGUF runtime, provider-specific OpenAI/OpenRouter integrations, or a generic OpenAI-compatible API.

By default, the repository is now configured for a **local mock mode**, so the assistant can be demoed end-to-end even when no external model endpoint is available.

## Overview

This repository originally focused on notebook-based clinical screening experiments. It now also includes the first implementation slice of a production-style assistant architecture with:

- an LLM router that selects the correct workflow per user request;
- tool calling for patient record retrieval;
- session state with a single active patient;
- prompt-specialized nodes for lookup, symptom analysis, usage help, and invalid requests;
- structured audit events for observability and future validation.

The repository also now includes a simplified educational notebook that demonstrates the core LangGraph routing pattern with one shared LLM, one router node, and two specialist nodes.

## Features

- LangGraph-based orchestration with short-term session state.
- SQLite patient repository with support for lookup by fictional security number or by name.
- Enumerated disambiguation flow when multiple patients match a name.
- Chainlit chat UI wired to the LangGraph application.
- Chainlit streaming that shows only the final user-facing answer tokens from the dedicated `final_answer` node.
- Deterministic demo-data seeding for a ready-to-run SQLite database.
- Deterministic mock control-model and clinical-model runtimes for local demos and end-to-end tests.
- Dedicated clinical backend abstraction for:
  - OpenAI API inference;
  - OpenRouter API inference;
  - generic OpenAI-compatible API inference;
  - future local GGUF inference integration.
- Structured internal clinical result for audit and testing.
- Optional full-graph terminal debug output using LangGraph streaming in JSON lines format.

## Repository Structure

- `screening_robot.ipynb`: notebook with model training and export experiments.
- `langgraph_router_specialists_simple.ipynb`: simplified LangGraph notebook showing a router that dispatches questions to one of two specialists with a single shared LLM.
- `process_clinical_batches.py`: prior batch-oriented clinical enrichment pipeline.
- `app_chainlit.py`: Chainlit entry point for the conversational UI.
- `seed_demo_data.py`: script that seeds the SQLite database with deterministic demo patients.
- `src/screening_agent/`: application package for the assistant runtime.
- `tests/`: initial automated tests for deterministic project components.

## Requirements

- Python 3.13+
- A virtual environment
- SQLite database with synthetic patient records
- One of the following model setups:
  - OpenAI provider endpoint for the control model and/or clinical model
  - OpenRouter provider endpoint for the control model and/or clinical model
  - generic OpenAI-compatible endpoint for the control model and/or clinical model
  - local GGUF file for future local runtime support

## Setup

1. Create and activate a virtual environment.
2. Install the project in editable mode so the `src/` package is importable.
3. Copy `.env.example` to `.env` and fill in the values needed for your model setup.
4. Create or point to a SQLite database for patient data.

Example:

- `pip install -e .`

The provided `.env.example` starts in mock mode:

- `SCREENING_AGENT_CONTROL_BACKEND=mock`
- `SCREENING_AGENT_CLINICAL_BACKEND=mock`

You can switch those values to real backends later without changing the application code.

### Provider backend options

The control and clinical model settings now support these backend values:

- `mock`: deterministic offline demo mode.
- `openai`: official OpenAI integration via `ChatOpenAI`.
- `openrouter`: official OpenRouter integration via `ChatOpenRouter`.
- `openai_compatible`: generic OpenAI-compatible endpoint via `ChatOpenAI(base_url=...)`.
- `gguf`: clinical backend only; reserved for the local runtime path.

Use provider-specific integrations whenever possible. The generic `openai_compatible`
path is kept for endpoints that expose an OpenAI-style API but do not have a dedicated
LangChain provider package.

## How to Run or Use

The repository now includes both the application core and a first conversational UI.

Typical usage flow:

1. Keep the default mock mode or configure `.env` for real model endpoints.
2. Seed the SQLite database with demo patients.
3. Keep checkpointing enabled for the Chainlit app.
4. Start the Chainlit application.
5. Open the local Chainlit URL and chat with the assistant.

Try it:

python seed_demo_data.py
chainlit run app_chainlit.py

### Simplified LangGraph notebook demo

If you want a smaller, documentation-aligned example before looking at the full application, open:

- `langgraph_router_specialists_simple.ipynb`

The notebook:

- uses a single shared chat model configured from `SCREENING_AGENT_CONTROL_*` variables in `.env`;
- builds a `StateGraph` with one router node and two specialist nodes;
- demonstrates conditional routing with two sample questions;
- keeps the example intentionally smaller than the production-style assistant in `src/screening_agent/`.

If you prefer to use the package directly, compile the graph with `build_screening_graph(...)` or `build_default_graph(...)` and invoke it with a `thread_id` in the LangGraph config.

### Terminal console debug

To stream detailed debug information to the terminal while keeping the Chainlit UI focused on the final user-facing answer, enable:

- `SCREENING_AGENT_CONSOLE_DEBUG=true`

The Chainlit UI now streams only the tokens emitted by the dedicated `final_answer` node.

When console debug is enabled, the application uses LangGraph `astream(..., version="v2", subgraphs=True)` and writes structured JSON lines for:

- `debug` stream parts for task/checkpoint metadata;
- `custom` stream parts for exact prompts, structured-output retries, and tool metadata.

By default, raw token-level `messages` events are suppressed in the terminal so the console stays consolidated.

If you explicitly want token-by-token terminal logs as well, enable:

- `SCREENING_AGENT_CONSOLE_DEBUG_VERBOSE=true`

The Chainlit app also requires checkpoint-backed state, so keep:

- `SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER=true`

After the stream completes, the app reads the authoritative final response from the latest root checkpoint state before updating the UI.

### Switching from mock mode to a real backend

#### OpenAI

Set values such as:

- `SCREENING_AGENT_CONTROL_BACKEND=openai`
- `SCREENING_AGENT_CONTROL_MODEL=gpt-4.1-mini`
- `SCREENING_AGENT_CLINICAL_BACKEND=openai`
- `SCREENING_AGENT_CLINICAL_MODEL=gpt-4.1-mini`
- `SCREENING_AGENT_CONTROL_API_KEY=...`
- `SCREENING_AGENT_CLINICAL_API_KEY=...`

#### OpenRouter

Set values such as:

- `SCREENING_AGENT_CONTROL_BACKEND=openrouter`
- `SCREENING_AGENT_CONTROL_MODEL=openai/gpt-4.1-mini`
- `SCREENING_AGENT_CLINICAL_BACKEND=openrouter`
- `SCREENING_AGENT_CLINICAL_MODEL=openai/gpt-4.1-mini`
- `SCREENING_AGENT_CONTROL_API_KEY=...`
- `SCREENING_AGENT_CLINICAL_API_KEY=...`

#### Generic OpenAI-compatible endpoint

Set values such as:

- `SCREENING_AGENT_CONTROL_BACKEND=openai_compatible`
- `SCREENING_AGENT_CLINICAL_BACKEND=openai_compatible`
- `SCREENING_AGENT_CONTROL_BASE_URL=...`
- `SCREENING_AGENT_CLINICAL_BASE_URL=...`
- `SCREENING_AGENT_CONTROL_API_KEY=...`
- `SCREENING_AGENT_CLINICAL_API_KEY=...`

The routing layer now fails closed if structured output cannot be produced after
bounded retry attempts, and the patient lookup subgraph uses `ToolNode` with bounded
tool-repair retries instead of custom manual tool dispatch.

## Data, Models, and Artifacts

- Patient data lives in SQLite and can be prepopulated with `seed_demo_data.py` for local demos.
- Clinical model artifacts currently originate from the notebooks in this repository.
- Local GGUF runtime support is scaffolded as an integration point and can be completed once the final GGUF artifact path and runtime package strategy are confirmed.

## Limitations and Disclaimers

- This assistant is for software prototyping and workflow validation only.
- It is not a substitute for a licensed clinician, emergency assessment, or medical diagnosis.
- The current codebase now includes a first Chainlit UI and seeded demo-data flow, but production auth, persistence hardening, and full end-to-end model validation are still in progress.
