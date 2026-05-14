# Screening Robot

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-6f42c1)](https://docs.langchain.com/oss/python/langgraph/overview)
[![Chainlit](https://img.shields.io/badge/ui-Chainlit-12b886)](https://docs.chainlit.io/)
[![QLoRA](https://img.shields.io/badge/fine--tuning-QLoRA-orange)](https://unsloth.ai/docs)
[![Hugging%20Face](https://img.shields.io/badge/model%20registry-Hugging%20Face-yellow)](https://huggingface.co/luizaaca)

🇧🇷 [Read in Portuguese](README_pt-br.md)

Clinical screening assistant that connects **fine-tuned Qwen models**, a **LangGraph orchestration layer**, a **Chainlit conversational UI**, and **SQLite-backed patient retrieval** into a single end-to-end project.

The project covers the full lifecycle: dataset engineering, QLoRA fine-tuning, structured clinical inference, retrieval-augmented patient context, observability, and a modular Python runtime.

## Table of Contents

- [Overview](#overview)
- [Architecture and implementation checklist](#architecture-and-implementation-checklist)
- [Architecture at a glance](#architecture-at-a-glance)
- [Fine-tuning pipeline and model evolution](#fine-tuning-pipeline-and-model-evolution)
- [Published model registry](#published-model-registry)
- [LangGraph assistant runtime](#langgraph-assistant-runtime)
- [Security, validation, and explainability](#security-validation-and-explainability)
- [Repository structure](#repository-structure)
- [Technology stack](#technology-stack)
- [Getting started](#getting-started)
- [Example prompts](#example-prompts)
- [Notebooks, scripts, and datasets](#notebooks-scripts-and-datasets)
- [Testing](#testing)
- [Official references](#official-references)
- [Limitations](#limitations)

## Overview

`Screening Robot` is an agentic clinical screening prototype built around two complementary layers:

1. **Control layer** — routes user requests, manages tools, handles patient lookup, and composes the final answer.
2. **Clinical specialist layer** — produces structured clinical screening output that can be consumed reliably by the graph.

The project evolved from notebook-only experiments into a modular Python application under `src/screening_agent/`, with:

- **LangGraph** for stateful orchestration and conditional routing;
- **LangChain** abstractions for models, tools, and message handling;
- **Chainlit** for the interactive web chat experience;
- **Pydantic** schemas for structured outputs and validation;
- **SQLite** for retrieval-augmented patient context;
- **Unsloth + QLoRA** for efficient fine-tuning of Qwen models;
- **GGUF compatibility** for local clinical inference.

The project uses public clinical symptom datasets, curated synthetic enrichment, and synthetic patient records in SQLite, enabling reproducible demonstrations without exposing sensitive information.

## Architecture and implementation checklist

| Component | Where it is implemented | Evidence in this repository |
| --- | --- | --- |
| LLM fine-tuning with medical data | Training notebooks + custom dataset pipeline | `screening_robot.ipynb`, `screening_robot_qwen3_1_7b_json.ipynb`, `process_clinical_batches.py` |
| Dataset preprocessing and curation | Normalization + synthetic data strategy | `process_clinical_batches.py`, `dataset_augmentation.ipynb`, `seed_demo_data.py`, synthetic SQLite demo DB |
| LangChain-based assistant runtime | Model/tool abstractions and prompt-driven workflow | `src/screening_agent/model/`, `src/screening_agent/tools/`, `src/screening_agent/prompts/` |
| LangGraph orchestration | Stateful graph + subgraphs + conditional edges | `src/screening_agent/graph/` |
| Structured data access and contextualization | SQLite retrieval and patient activation flow | `src/screening_agent/data/patient_repository.py`, `src/screening_agent/tools/patient_tools.py` |
| Security and fail-closed validation | Bounded retries, disclaimers, schema validation | `src/screening_agent/model/structured_output.py`, `src/screening_agent/graph/nodes/processing_error.py` |
| Observability and audit trail | Event logging + debug modes | `src/screening_agent/audit.py`, `.env.example` |
| Explainability and source traceability | Router rationale, structured output, patient context trace | `src/screening_agent/graph/state.py`, `specialist_tool.py`, `finalize_response.py` |
| Modular Python architecture | Clear separation of concerns | `src/screening_agent/` |

## Architecture at a glance

From a software perspective, the project is better described as a composition of **six functional blocks**: interface, configuration/observability, orchestration, services and contracts, data/persistence, and inference backends. This view shows how the repository modules fit together without duplicating the detailed LangGraph node flow documented later.

```mermaid
flowchart LR
    USER["User / Clinician"]

    subgraph ENTRY["Interface and application entrypoint"]
        CL["app_chainlit.py<br/>Chainlit UI, session, and streaming"]
    end

    subgraph CROSS["Configuration and observability"]
        CFG["config.py<br/>AppSettings and backend selection"]
        AUD["audit.py<br/>audit trail and console debug"]
    end

    subgraph ORCH["Application orchestration"]
        GRAPH["graph/builder.py + graph/state.py<br/>root graph, state, and routing"]
        LOOKUP["graph/subgraphs/patient_lookup.py<br/>patient lookup via tool-calling"]
        ANALYSIS["graph/nodes/symptom_analysis.py<br/>structured clinical analysis"]
        FINAL["graph/nodes/finalize_response.py<br/>final response composition"]
    end

    subgraph SERVICES["Services and contracts"]
        CONTROL["model/factory.py + model/control_models.py<br/>control-model adapters"]
        SPECIALIST["tools/specialist_tool.py<br/>clinical invoker and JSON contract"]
        PTOOLS["tools/patient_tools.py<br/>patient lookup and activation tools"]
        STRUCT["model/structured_output.py<br/>fallback and structured validation"]
    end

    subgraph DATA["Data and persistence"]
        REPO["data/patient_repository.py<br/>patient repository"]
        DB[(SQLite)]
    end

    subgraph BACKENDS["Inference backends"]
        CTRLB["Control<br/>mock / openai / openrouter / openai_compatible"]
        CLINB["Clinical<br/>mock / openai / openrouter / openai_compatible / gguf"]
    end

    USER --> CL
    CL --> GRAPH
    CL --> CFG
    GRAPH --> LOOKUP
    GRAPH --> ANALYSIS
    GRAPH --> FINAL
    GRAPH --> CONTROL
    LOOKUP --> PTOOLS
    ANALYSIS --> SPECIALIST
    SPECIALIST --> STRUCT
    PTOOLS --> REPO
    REPO --> DB
    # Screening Robot

    [![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
    [![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-6f42c1)](https://docs.langchain.com/oss/python/langgraph/overview)
    [![Chainlit](https://img.shields.io/badge/ui-Chainlit-12b886)](https://docs.chainlit.io/)
    [![QLoRA](https://img.shields.io/badge/fine--tuning-QLoRA-orange)](https://unsloth.ai/docs)
    [![Hugging%20Face](https://img.shields.io/badge/model%20registry-Hugging%20Face-yellow)](https://huggingface.co/luizaaca)

    🇧🇷 [Read in Portuguese](README_pt-br.md)

    Clinical screening assistant that connects **fine-tuned Qwen models**, a **LangGraph orchestration layer**, a **Chainlit conversational UI**, and **SQLite-backed patient retrieval** into a single end-to-end project.

    The project covers the full lifecycle: dataset engineering, QLoRA fine-tuning, structured clinical inference, retrieval-augmented patient context, observability, and a modular Python runtime.

    ## Table of Contents

    - [Overview](#overview)
    - [Architecture and implementation checklist](#architecture-and-implementation-checklist)
    - [Architecture at a glance](#architecture-at-a-glance)
    - [Fine-tuning pipeline and model evolution](#fine-tuning-pipeline-and-model-evolution)
    - [Published model registry](#published-model-registry)
    - [LangGraph assistant runtime](#langgraph-assistant-runtime)
    - [Security, validation, and explainability](#security-validation-and-explainability)
    - [Repository structure](#repository-structure)
    - [Technology stack](#technology-stack)
    - [Getting started](#getting-started)
    - [Example prompts](#example-prompts)
    - [Notebooks, scripts, and datasets](#notebooks-scripts-and-datasets)
    - [Testing](#testing)
    - [Official references](#official-references)
    - [Limitations](#limitations)

    ## Overview

    `Screening Robot` is a prototype clinical screening assistant built around two complementary concerns:

    1. **Control layer** — routes user requests, manages tools, performs patient lookup, and composes the final answer.
    2. **Clinical specialist layer** — produces structured clinical screening output that the graph consumes reliably.

    The project evolved from notebook experiments into a modular Python application under `src/screening_agent/`, with:

    - **LangGraph** for stateful orchestration and conditional routing;
    - **LangChain** abstractions for models, tools, and messages;
    - **Chainlit** for the interactive web chat UI;
    - **Pydantic** schemas for structured outputs and validation;
    - **SQLite** for retrieval-augmented patient context;
    - **Unsloth + QLoRA** for efficient fine-tuning of Qwen models;
    - **GGUF compatibility** for local clinical inference.

    The project uses public clinical symptom datasets, curated synthetic enrichment, and synthetic patient records in SQLite, enabling reproducible demonstrations without exposing sensitive information.

    ## Architecture and implementation checklist

    | Component | Where it is implemented | Evidence in this repository |
    | --- | --- | --- |
    | LLM fine-tuning with medical data | Training notebooks + custom dataset pipeline | `screening_robot.ipynb`, `screening_robot_qwen3_1_7b_json.ipynb`, `process_clinical_batches.py` |
    | Dataset preprocessing and curation | Normalization + synthetic data strategy | `process_clinical_batches.py`, `dataset_augmentation.ipynb`, `seed_demo_data.py`, synthetic SQLite demo DB |
    | LangChain-based assistant runtime | Model/tool abstractions and prompt-driven workflow | `src/screening_agent/model/`, `src/screening_agent/tools/`, `src/screening_agent/prompts/` |
    | LangGraph orchestration | Stateful graph + subgraphs + conditional edges | `src/screening_agent/graph/` |
    | Structured data access and contextualization | SQLite retrieval and patient activation flow | `src/screening_agent/data/patient_repository.py`, `src/screening_agent/tools/patient_tools.py` |
    | Security and fail-closed validation | Bounded retries, disclaimers, schema validation | `src/screening_agent/model/structured_output.py`, `src/screening_agent/graph/nodes/processing_error.py` |
    | Observability and audit trail | Event logging + debug modes | `src/screening_agent/audit.py`, `.env.example` |
    | Explainability and source traceability | Router rationale, structured output, patient context trace | `src/screening_agent/graph/state.py`, `specialist_tool.py`, `finalize_response.py` |
    | Modular Python architecture | Clear separation of concerns | `src/screening_agent/` |

    ## Architecture at a glance

    From a software perspective, the project is best described as a composition of **six functional blocks**: interface, configuration/observability, orchestration, services and contracts, data/persistence, and inference backends. This view shows how the repository modules fit together without duplicating the detailed LangGraph node flow documented later.

    ```mermaid
    flowchart LR
        USER["User / Clinician"]

        subgraph ENTRY["Interface and application entrypoint"]
            CL["app_chainlit.py<br/>Chainlit UI, session, and streaming"]
        end

        subgraph CROSS["Configuration and observability"]
            CFG["config.py<br/>AppSettings and backend selection"]
            AUD["audit.py<br/>audit trail and console debug"]
        end

        subgraph ORCH["Application orchestration"]
            GRAPH["graph/builder.py + graph/state.py<br/>root graph, state, and routing"]
            LOOKUP["graph/subgraphs/patient_lookup.py<br/>patient lookup via tool-calling"]
            ANALYSIS["graph/nodes/symptom_analysis.py<br/>structured clinical analysis"]
            FINAL["graph/nodes/finalize_response.py<br/>final response composition"]
        end

        subgraph SERVICES["Services and contracts"]
            CONTROL["model/factory.py + model/control_models.py<br/>control-model adapters"]
            SPECIALIST["tools/specialist_tool.py<br/>clinical invoker and JSON contract"]
            PTOOLS["tools/patient_tools.py<br/>patient lookup and activation tools"]
            STRUCT["model/structured_output.py<br/>fallback and structured validation"]
        end

        subgraph DATA["Data and persistence"]
            REPO["data/patient_repository.py<br/>patient repository"]
            DB[(SQLite)]
        end

        subgraph BACKENDS["Inference backends"]
            CTRLB["Control<br/>mock / openai / openrouter / openai_compatible"]
            CLINB["Clinical<br/>mock / openai / openrouter / openai_compatible / gguf"]
        end

        USER --> CL
        CL --> GRAPH
        CL --> CFG
        GRAPH --> LOOKUP
        GRAPH --> ANALYSIS
        GRAPH --> FINAL
        GRAPH --> CONTROL
        LOOKUP --> PTOOLS
        ANALYSIS --> SPECIALIST
        SPECIALIST --> STRUCT
        PTOOLS --> REPO
        REPO --> DB
        CONTROL --> CTRLB
        SPECIALIST --> CLINB
    ```

    ### Main blocks

    **Interface and application entrypoint**
    - `app_chainlit.py` is the Chainlit entrypoint.
    - The UI creates or reuses a per-session `thread_id`, loads `AppSettings`, compiles the graph with `build_default_graph(...)`, and streams only the tokens generated by the `final_answer` node.
    - The interface does not embed clinical or lookup business rules; it delegates each turn to the orchestrated runtime.

    **Configuration and observability**
    - `config.py` centralizes environment-driven configuration: SQLite database, control backend, clinical backend, checkpoint mode, and debug level.
    - `audit.py` concentrates audit events and console-debug emission across the workflow.
    - These modules are cross-cutting concerns rather than business-flow components.

    **Application orchestration**
    - `src/screening_agent/graph/` implements the main LangGraph runtime.
    - `src/screening_agent/prompts/` centralizes the system prompts consumed by nodes and subflows.
    - The root graph coordinates routing, usage instructions, patient lookup, symptom analysis, context clearing, error handling, and final response composition.
    - Patient lookup and clinical analysis are encapsulated as specialized subflows while sharing the same session state.

    **Services and contracts**
    - The **control model** does more than intent classification: it also drives tool-calling, support nodes, and final response rendering.
    - The **clinical invoker** in `tools/specialist_tool.py` encapsulates the `ClinicalScreeningOutput` contract and abstracts the configured clinical backend.
    - `tools/patient_tools.py` implements security-number lookup, name search, and patient activation operations.
    - `model/structured_output.py` adds fallback and JSON repair for remote specialist backends; the `mock` and `gguf` paths validate outputs through their own mechanisms.

    **Inference Backends**
    - The **control** path supports `mock`, `openai`, `openrouter`, and `openai_compatible`.
    - The **clinical** path supports `mock`, `openai`, `openrouter`, `openai_compatible`, and `gguf`.
    - The fine-tuned Qwen artifacts belong to the clinical path; they are not an architectural requirement for the control path.

    **Data and persistence**
    - `data/patient_repository.py` provides structured SQLite access for fictional security-number lookup, name search, and clinical-context retrieval.
    - The repository operates on synthetic patient data and feeds the active context used during clinical analysis.
    - This is structured retrieval over SQLite, not a vector index.

    ### Important architectural relationships

    - Chainlit is the presentation layer; the main logic lives in the graph and service modules.
    - LangGraph is the application runtime and state coordinator; it does not replace the data layer or the model-integration layer.
    - Patient lookup and clinical analysis both use tool-calling, but they represent different business responsibilities.
    - Final-answer composition is a separate step that turns internal artifacts into the text shown to the user.

    For details on node routing and LangGraph flow, see the **LangGraph assistant runtime** section below.

    ### Assistant state

    The runtime extends LangGraph's `MessagesState` with assistant-specific fields such as:

    - `active_patient`
    - `patient_lookup_status`
    - `patient_lookup_candidates`
    - `router_intent`
    - `router_rationale`
    - `specialist_output_json`
    - `last_response`

    This state design allows the assistant to preserve short-term context across turns without hard-coding business logic into the UI layer.

    ## Fine-tuning pipeline and model evolution

    The project contains **two fine-tuning generations**, and the difference between them is central to understanding the final solution.

    ### Version 1 — `screening_robot.ipynb`

    The first notebook fine-tunes **Qwen3-0.6B** with QLoRA in a symptom→disease formulation where the model returns a final textual answer with a fixed clinical disclaimer.

    Key characteristics:

    - lighter base model for fast experimentation;
    - mixed `/think` and `/no_think` prompting styles;
    - evaluation with **accuracy**, **macro-F1**, **Cohen's Kappa**, **confusion matrices**, and **BERTScore**;
    - LoRA + GGUF export path.

    Why it was not selected for the final agent:

    - the reasoning behavior was not sufficiently consolidated for downstream orchestration;
    - the disclaimer-oriented free-text shape was hard to integrate reliably as a specialist tool;
    - free-form answers were harder to validate and to use in a tool-calling agent.

    ### Version 2 — `screening_robot_qwen3_1_7b_json.ipynb`

    The second notebook fine-tunes **Qwen3-1.7B-Base** with **Unsloth + QLoRA** for a narrower, production-friendly objective: emitting a validated JSON payload designed for the specialist tool in the runtime.

    Target schema:

    ```json
    {
      "support_status": "supported | inconclusive",
      "candidate_diseases": ["..."],
      "recommended_exams_tests": ["..."]
    }
    ```

    Why this version became the chosen model:

    - **better alignment with the agent design**;
    - **structured output is easier to validate, retry, and audit**;
    - **1.7B parameters offered a better trade-off** between capacity and local efficiency for the intended use case;
    - it integrates naturally with the `run_symptom_specialist` tool and the `final_answer` composition step.

    ### Dataset engineering and curation

    The training data is produced by a multi-step enrichment pipeline rather than a single raw CSV.

    1. Merge symptom/disease sources into `combined_diseases_symptoms_2.csv`.
    2. Run `process_clinical_batches.py` to generate:
       - `support_status`
       - `candidate_diseases`
       - `recommended_exams_tests`
       - `reasoning`
    3. Persist the enriched artifact to `combined_diseases_symptoms_2_enriched_with_exams_v2.csv`.
    4. Publish the resulting dataset to Kaggle: [`luizaaca/symptoms-to-diseases-with-reasoning`](https://www.kaggle.com/datasets/luizaaca/symptoms-to-diseases-with-reasoning).

    The enrichment pipeline includes:

    - symptom normalization;
    - heuristic support scoring;
    - TF-IDF-style symptom weighting;
    - web-assisted exam suggestion gathering;
    - stable disease candidate ordering;
    - curation for structured downstream training.

    For the runtime, patient records are intentionally **synthetic** and stored in SQLite. Demo records use fictional identifiers and narrative clinical context so the repository can be published safely.

    ### Training configuration used in the 1.7B JSON experiment

    | Parameter | Value |
    | --- | --- |
    | Base model | `Qwen/Qwen3-1.7B-Base` |
    | Fine-tuning method | QLoRA with Unsloth |
    | Quantization | 4-bit |
    | LoRA rank | 16 |
    | LoRA alpha | 32 |
    | Max sequence length | 1024 |
    | Batch size | 2 |
    | Gradient accumulation | 4 |
    | Warmup steps | 20 |
    | Max steps | 400 |
    | Learning rate | `2e-4` |
    | Weight decay | `0.01` |
    | Loss masking | response-only supervision |

    ### Evaluation methodology

    Notebooks assess both predictive quality and operational reliability.

    **Clinical prediction metrics**

    - Accuracy
    - F1-macro
    - Cohen's Kappa
    - Confusion matrix analysis
    - BERTScore

    **Structured-output reliability metrics**

    - JSON parse rate
    - Schema validation rate
    - support-status validity
    - presence of the target disease in `candidate_diseases`
    - non-empty recommended-exam list rate

    Final model selection considered both task performance and **schema adherence / downstream compatibility with the LangGraph agent**.

    ## Published model registry

    The fine-tuned models are published on Hugging Face and are referenced from this repository.

    | Model | Link | Purpose | Output style |
    | --- | --- | --- | --- |
    | Qwen3-0.6B Clinical Screening | [`luizaaca/qwen3-0.6b-clinical-screening`](https://huggingface.co/luizaaca/qwen3-0.6b-clinical-screening) | First iteration to validate task framing | Free-form clinical answer with fixed disclaimer |
    | Qwen3-1.7B Clinical Screening | [`luizaaca/qwen3-1.7b-clinical-screening`](https://huggingface.co/luizaaca/qwen3-1.7b-clinical-screening) | Final specialist-oriented model for structured agent integration | Structured clinical JSON aligned with tool calling |

    Notes:

    - Both Hugging Face model pages list artifacts under **CC-BY-4.0**.
    - Local inference can use a **GGUF export** configured via `SCREENING_AGENT_GGUF_MODEL_PATH`.
    - The runtime keeps the **control** and **clinical** models logically independent.

    ## LangGraph assistant runtime

    The runtime graph is implemented in `src/screening_agent/graph/` and compiled by `build_default_graph(...)`.

    ```mermaid
    flowchart LR
        START --> router
        router -->|usage_instructions| usage_instructions
        router -->|patient_lookup| patient_lookup
        router -->|patient_lookup_then_analysis| patient_lookup
        router -->|symptom_analysis| symptom_analysis
        router -->|clear_active_patient| clear_active_patient
        router -->|invalid_request| invalid_request
        router -->|structured output failure| processing_error

        patient_lookup --> route_after_lookup
        route_after_lookup -->|lookup complete| symptom_analysis
        route_after_lookup -->|selection required / not found| final_answer

        usage_instructions --> final_answer
        symptom_analysis --> final_answer
        clear_active_patient --> final_answer
        invalid_request --> final_answer
        processing_error --> final_answer
        final_answer --> END
    ```

    ### Main nodes and subgraphs

    | Component | Responsibility |
    | --- | --- |
    | `router` | Classifies the request intent with structured output (`RouteDecision`) |
    | `usage_instructions` | Explains how the assistant should be used |
    | `patient_lookup` | Runs the patient retrieval tool flow |
    | `route_after_lookup` | Decides whether to continue to analysis or answer immediately |
    | `symptom_analysis` | Invokes the specialist tool and captures structured clinical output |
    | `clear_active_patient` | Clears patient context safely |
    | `invalid_request` | Handles unsupported requests |
    | `processing_error` | Fail-closed fallback for orchestration failures |
    | `final_answer` | Composes the final clinician-facing response |

    ### Retrieval-augmented patient context

    The patient lookup flow is structured retrieval over SQLite (RAG-style for clinical context):

    - `find_by_security_number(...)` retrieves a single patient by fictional ID;
    - `search_by_name(...)` ranks results by exact match, prefix match, and substring match;
    - `activate_patient_selection(...)` resolves name ambiguity across turns;
    - the retrieved `clinical_context` is injected into the symptom-analysis workflow.

    The seeded demo database contains **six synthetic patients** and can be initialized with `python seed_demo_data.py`.

    ### Specialist tool and backends

    The specialist tool is implemented in `src/screening_agent/tools/specialist_tool.py` and supports multiple execution strategies:

    1. **mock** — deterministic offline behavior for development and tests;
    2. **remote structured model** — OpenAI / OpenRouter / OpenAI-compatible endpoints;
    3. **GGUF local runtime** — for local inference using a GGUF artifact.

    Supported backends matrix:

    | Layer | Supported backends |
    | --- | --- |
    | Control model | `mock`, `openai`, `openrouter`, `openai_compatible` |
    | Clinical model | `mock`, `openai`, `openrouter`, `openai_compatible`, `gguf` |

    ## Security, validation, and explainability

    Healthcare systems need strong guardrails. This project implements safety where necessary.

    ### Safety boundaries

    - The assistant is presented as a **clinical screening support tool**, not a diagnostic device;
    - Final answers include a clear disclaimer that the system **does not replace professional medical judgment**;
    - Invalid or unsupported requests are routed to dedicated guardrail responses.

    ### Fail-closed structured output

    `ResilientStructuredOutputInvoker` in `src/screening_agent/model/structured_output.py` implements bounded retries:

    1. native structured output attempt;
    2. JSON repair fallback;
    3. final JSON repair attempt;
    4. raise a bounded error and route to a deterministic failure path.

    This prevents silent corruption when a model drifts from the expected schema.

    ### Audit trail and debug modes

    Audit events are emitted across the workflow with fields such as:

    - `timestamp_utc`
    - `event_type`
    - `status`
    - `node_name`
    - `detail`

    Terminal debug levels:

    - `SCREENING_AGENT_CONSOLE_DEBUG_MODE=none`
    - `SCREENING_AGENT_CONSOLE_DEBUG_MODE=info`
    - `SCREENING_AGENT_CONSOLE_DEBUG_MODE=debug`

    Sensitive values such as security numbers are masked before printing.

    ### Explainability features

    The runtime preserves interpretable intermediate artifacts instead of hiding everything inside a prompt:

    - router rationale is kept in state;
    - specialist output exposes `support_status`, `candidate_diseases` and `recommended_exams_tests`;
    - patient context comes from a known repository source;
    - final answers are composed from structured upstream artifacts.

    ## Repository structure

    ```text
    .
    ├── app_chainlit.py
    ├── process_clinical_batches.py
    ├── screening_robot.ipynb
    ├── screening_robot_qwen3_1_7b_json.ipynb
    ├── langgraph_router_specialists_simple.ipynb
    ├── langgraph_router_specialists_simple_v2.ipynb
    ├── langgraph_router_specialists_simple_v3.ipynb
    ├── seed_demo_data.py
    ├── src/
    │   └── screening_agent/
    │       ├── audit.py
    │       ├── config.py
    │       ├── data/
    │       ├── graph/
    │       ├── model/
    │       ├── prompts/
    │       └── tools/
    └── tests/
    ```

    ### Key folders

    - `src/screening_agent/data/` — SQLite schema, repository, and demo seeding helpers
    - `src/screening_agent/graph/` — state schema, graph builder, nodes, and subgraphs
    - `src/screening_agent/model/` — model adapters, mock runtime, structured-output fallback
    - `src/screening_agent/prompts/` — system prompts by node responsibility
    - `src/screening_agent/tools/` — patient lookup tools and clinical specialist tool
    - `tests/` — deterministic automated coverage for the core runtime

    ## Technology stack

    ### Runtime dependencies

    - `chainlit>=2.11.1`
    - `langchain>=1.2.15`
    - `langgraph>=1.1.10`
    - `langchain-openai>=1.2.1`
    - `langchain-openrouter>=0.2.1`
    - `openai>=2.32.0`
    - `pydantic>=2.13.2`
    - `python-dotenv>=1.2.2`

    ### Training and evaluation stack

    - Unsloth
    - PyTorch
    - TRL / SFTTrainer
    - scikit-learn
    - evaluate / BERTScore
    - pandas / NumPy / seaborn / matplotlib
    - llama-cpp-python (for GGUF runtime experiments)

    ## Getting started

    ### Prerequisites

    - Python **3.13+**
    - virtual environment support
    - Git
    - optional model endpoint or local GGUF artifact if you want real inference instead of mock mode

    ### Installation

    ```bash
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -e .[dev]
    ```

    ### Configuration

    Copy `.env.example` to `.env` and configure the model backends as desired.

    Important variables:

    - `SCREENING_AGENT_CONTROL_BACKEND`
    - `SCREENING_AGENT_CONTROL_MODEL`
    - `SCREENING_AGENT_CLINICAL_BACKEND`
    - `SCREENING_AGENT_CLINICAL_MODEL`
    - `SCREENING_AGENT_GGUF_MODEL_PATH`
    - `SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER=true`
    - `SCREENING_AGENT_CONSOLE_DEBUG_MODE`

    Default (safe demo) configuration:

    ```env
    SCREENING_AGENT_CONTROL_BACKEND=mock
    SCREENING_AGENT_CLINICAL_BACKEND=mock
    SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER=true
    ```

    ### Seed the demo database

    ```bash
    python seed_demo_data.py
    ```

    ### Run the application

    ```bash
    chainlit run app_chainlit.py
    ```

    Open the local Chainlit URL shown in the terminal and start chatting.

    ### Provider switching

    Supported setups without code changes:

    - **Mock mode** — demos, tests, offline development
    - **OpenAI** — hosted inference
    - **OpenRouter** — provider-agnostic remote inference
    - **OpenAI-compatible** — local/self-hosted endpoints (LM Studio-style)
    - **GGUF** — local clinical backend for the specialist tool

    See `.env.example` for concrete examples.

    ## Example prompts

    - `Find patient Maria Silva`
    - `Lookup patient 12003456`
    - `Patient 55667788 has fatigue and frequent urination`
    - `Clear active patient`
    - `How should I use this assistant?`

    ## Notebooks, scripts, and datasets

    | Artifact | Purpose |
    | --- | --- |
    | `screening_robot.ipynb` | First end-to-end fine-tuning experiment with Qwen3-0.6B |
    | `screening_robot_qwen3_1_7b_json.ipynb` | Final structured-output fine-tuning experiment with Qwen3-1.7B |
    | `langgraph_router_specialists_simple.ipynb` | Minimal LangGraph routing concept |
    | `langgraph_router_specialists_simple_v2.ipynb` | Intermediate notebook with richer patient context and backend flexibility |
    | `langgraph_router_specialists_simple_v3.ipynb` | Production-oriented pattern for structured specialist integration |
    | `dataset_augmentation.ipynb` | LLM-assisted dataset validation and augmentation experiments |
    | `process_clinical_batches.py` | Batch enrichment pipeline for support status, candidates, exams, and reasoning |
    | `combined_diseases_symptoms_2_enriched_with_exams_v2.csv` | Enriched custom training dataset |
    | `data/patients.sqlite3` | Synthetic patient database used by the app |

    ## Testing

    Run the automated test suite with:

    ```bash
    python -m pytest tests/
    ```

    Current tests cover:

    - patient repository lookups and ranking behavior;
    - graph routing and node transitions;
    - end-to-end mock-mode conversations;
    - structured-output fallback logic;
    - demo data seeding;
    - state helpers and console debug behavior.

    ## Official references

    - LangChain overview: <https://docs.langchain.com/oss/python/langchain/overview>
    - LangGraph overview: <https://docs.langchain.com/oss/python/langgraph/overview>
    - LangGraph graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api>
    - Chainlit overview: <https://docs.chainlit.io/>
    - Chainlit installation: <https://docs.chainlit.io/get-started/installation>
    - Unsloth documentation: <https://unsloth.ai/docs>
    - Pydantic documentation: <https://pydantic.dev/docs/validation/latest/get-started/>
    - Qwen organization on Hugging Face: <https://huggingface.co/Qwen>
    - Hugging Face model repository — Qwen3-0.6B clinical screening: <https://huggingface.co/luizaaca/qwen3-0.6b-clinical-screening>
    - Hugging Face model repository — Qwen3-1.7B clinical screening: <https://huggingface.co/luizaaca/qwen3-1.7b-clinical-screening>
    - Kaggle dataset — symptoms to diseases with reasoning: <https://www.kaggle.com/datasets/luizaaca/symptoms-to-diseases-with-reasoning>

    ## Limitations

    - This is **not** a medical device and must not be used as a substitute for a licensed clinician.
    - The public repository uses **synthetic patient records** and public datasets rather than real hospital data.
    - The retrieval layer is **structured SQLite retrieval**, not a vector-search knowledge base.
    - Production concerns such as authentication, long-term persistence, and deployment hardening are intentionally out of scope for this version.

    To explore the project, start with `langgraph_router_specialists_simple_v3.ipynb` to understand how the agent works with LangGraph, then `screening_robot_qwen3_1_7b_json.ipynb` to review the final model training pipeline, and `app_chainlit.py` for the conversational interface.

### Configuration

Copy `.env.example` to `.env` and adjust the model configuration for your preferred setup.

Important settings:

- `SCREENING_AGENT_CONTROL_BACKEND`
- `SCREENING_AGENT_CONTROL_MODEL`
- `SCREENING_AGENT_CLINICAL_BACKEND`
- `SCREENING_AGENT_CLINICAL_MODEL`
- `SCREENING_AGENT_GGUF_MODEL_PATH`
- `SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER=true`
- `SCREENING_AGENT_CONSOLE_DEBUG_MODE`

The default configuration is intentionally safe for local demos:

```env
SCREENING_AGENT_CONTROL_BACKEND=mock
SCREENING_AGENT_CLINICAL_BACKEND=mock
SCREENING_AGENT_USE_IN_MEMORY_CHECKPOINTER=true
```

### Seed the demo database

```bash
python seed_demo_data.py
```

### Run the application

```bash
chainlit run app_chainlit.py
```

Open the local Chainlit URL shown in the terminal and start chatting.

### Provider switching

The repository supports multiple inference setups without changing the Python code:

- **Mock mode** — best for demos, tests, and offline development
- **OpenAI** — direct hosted inference
- **OpenRouter** — provider-agnostic remote inference
- **OpenAI-compatible** — local or self-hosted endpoints such as LM Studio style APIs
- **GGUF** — local clinical backend for the specialist tool

See `.env.example` for concrete variable names and examples.

## Example prompts

- `Find patient Maria Silva`
- `Lookup patient 12003456`
- `Patient 55667788 has fatigue and frequent urination`
- `Clear active patient`
- `How should I use this assistant?`

## Notebooks, scripts, and datasets

| Artifact | Purpose |
| --- | --- |
| `screening_robot.ipynb` | First end-to-end fine-tuning experiment with Qwen3-0.6B |
| `screening_robot_qwen3_1_7b_json.ipynb` | Final structured-output fine-tuning experiment with Qwen3-1.7B |
| `langgraph_router_specialists_simple.ipynb` | Minimal LangGraph routing concept |
| `langgraph_router_specialists_simple_v2.ipynb` | Intermediate notebook with richer patient context and backend flexibility |
| `langgraph_router_specialists_simple_v3.ipynb` | Production-oriented notebook pattern for structured specialist integration |
| `dataset_augmentation.ipynb` | LLM-assisted dataset validation and augmentation experiments |
| `process_clinical_batches.py` | Batch enrichment pipeline for support status, candidates, exams, and reasoning |
| `combined_diseases_symptoms_2_enriched_with_exams_v2.csv` | Enriched custom training dataset |
| `data/patients.sqlite3` | Synthetic patient database used by the app |

## Testing

Run the automated test suite with:

```bash
python -m pytest tests/
```

Current tests cover:

- patient repository lookups and ranking behavior;
- graph routing and node transitions;
- end-to-end mock-mode conversations;
- structured-output fallback logic;
- demo data seeding;
- state helpers and console debug behavior.

## Official references

- LangChain overview: <https://docs.langchain.com/oss/python/langchain/overview>
- LangGraph overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph graph API: <https://docs.langchain.com/oss/python/langgraph/graph-api>
- Chainlit overview: <https://docs.chainlit.io/>
- Chainlit installation: <https://docs.chainlit.io/get-started/installation>
- Unsloth documentation: <https://unsloth.ai/docs>
- Pydantic documentation: <https://pydantic.dev/docs/validation/latest/get-started/>
- Qwen organization on Hugging Face: <https://huggingface.co/Qwen>
- Hugging Face model repository — Qwen3-0.6B clinical screening: <https://huggingface.co/luizaaca/qwen3-0.6b-clinical-screening>
- Hugging Face model repository — Qwen3-1.7B clinical screening: <https://huggingface.co/luizaaca/qwen3-1.7b-clinical-screening>
- Kaggle dataset — symptoms to diseases with reasoning: <https://www.kaggle.com/datasets/luizaaca/symptoms-to-diseases-with-reasoning>

## Limitations

- This is **not** a medical device and must not be used as a substitute for a licensed clinician.
- The public repository uses **synthetic patient records** and public datasets rather than real hospital data.
- The retrieval layer is **structured SQLite retrieval**, not a vector-search knowledge base.
- Production concerns such as authentication, long-term persistence, and deployment hardening are intentionally out of scope for this version.
To explore the project, start with `screening_robot_qwen3_1_7b_json.ipynb` for the final model, then `src/screening_agent/graph/builder.py` for the runtime, and `app_chainlit.py` for the conversational interface.
