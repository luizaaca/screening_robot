"""Specialist tool and invoker for structured symptom analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, field_validator


VALID_SUPPORT_STATUSES: tuple[str, str] = ("supported", "inconclusive")


class ClinicalScreeningOutput(BaseModel):
    """Structured output emitted by the symptom specialist."""

    support_status: Literal["supported", "inconclusive"] = Field(
        description="Whether the current evidence strongly supports a shortlist or remains inconclusive."
    )
    candidate_diseases: list[str] = Field(
        min_length=1,
        description="Ordered disease candidates, from strongest to weakest match.",
    )
    recommended_exams_tests: list[str] = Field(
        min_length=1,
        description="Ordered exams or tests that could help confirm or clarify the differential.",
    )

    @field_validator("support_status", mode="before")
    @classmethod
    def _normalize_support_status(cls, value: Any) -> str:
        """Normalize and validate the support status string."""
        normalized_value = str(value).strip().lower()
        if normalized_value not in VALID_SUPPORT_STATUSES:
            raise ValueError(
                f"support_status must be one of {VALID_SUPPORT_STATUSES!r}; got {normalized_value!r}."
            )
        return normalized_value

    @field_validator("candidate_diseases", "recommended_exams_tests", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        """Normalize ordered string lists and remove duplicates."""
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings.")
        cleaned_values = [str(item).strip() for item in value if str(item).strip()]
        seen: set[str] = set()
        unique: list[str] = []
        for item in cleaned_values:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        if not unique:
            raise ValueError("The list must contain at least one non-empty string.")
        return unique


SpecialistInvoker = Callable[[str, str], ClinicalScreeningOutput]

SYSTEM_PROMPT = """
You are a clinical screening assistant focused on symptom analysis.

Your job is to:
- interpret the user's complaint in context;
- recommend relevant confirmatory exams or next assessment steps;

Output requirements:
- Respond in the same language as the user, eg., English, Spanish, French.
- Produce a structured internal result in JSON format.
- Return only the JSON object.
- Use exactly these keys in the JSON object: support_status, candidate_diseases, recommended_exams_tests.
""".strip()


def build_clinical_input(clinical_request: str, clinical_context: str) -> str:
    """Build the training-aligned clinical input for the specialist.

    Args:
        clinical_request: The user's symptom question.
        clinical_context: Active patient clinical context string.

    Returns:
        Formatted prompt text for the specialist.
    """

    return (
        "Clinical request:\n"
        f"{clinical_request.strip()}\n\n"
        "Active patient context:\n"
        f"{clinical_context.strip()}"
    )


def _extract_json_object(text: str) -> str:
    """Extract the first JSON object from raw text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("The model response did not contain a JSON object.")
    return text[start: end + 1]


def _normalize_raw_content(content: Any) -> str:
    """Normalize raw backend content into a plain string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, Mapping):
                text_val = item.get("text") or item.get("content")
                if text_val:
                    parts.append(str(text_val).strip())
        return "\n".join(part for part in parts if part)
    return str(content).strip()


def create_mock_specialist_invoker() -> SpecialistInvoker:
    """Create a deterministic mock specialist for testing."""

    def invoke(clinical_request: str, clinical_context: str) -> ClinicalScreeningOutput:
        request_lower = clinical_request.lower()
        if "fatigue" in request_lower or "urination" in request_lower or "diabetes" in clinical_context.lower():
            return ClinicalScreeningOutput(
                support_status="supported",
                candidate_diseases=["Diabetes mellitus or poor glycemic control"],
                recommended_exams_tests=["Serum glucose", "HbA1c", "Urinalysis"],
            )
        if "cough" in request_lower or "fever" in request_lower:
            return ClinicalScreeningOutput(
                support_status="supported",
                candidate_diseases=["Acute respiratory infection"],
                recommended_exams_tests=["Complete blood count (CBC)", "Chest X-Ray"],
            )
        return ClinicalScreeningOutput(
            support_status="inconclusive",
            candidate_diseases=["Further clinical evaluation needed"],
            recommended_exams_tests=["Physical examination"],
        )

    return invoke


def create_gguf_specialist_invoker(model_path: str | Path, temperature: float = 0.0) -> SpecialistInvoker:
    """Create a GGUF-backed specialist invoker.

    Args:
        model_path: Path to the GGUF model file.
        temperature: Sampling temperature.

    Returns:
        A specialist invoker function.
    """

    from llama_cpp import Llama, LlamaGrammar  # type: ignore[import-untyped]

    gguf_llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        chat_format="chatml",
        verbose=False,
    )
    response_format: Any = {
        "type": "json_object",
        "schema": ClinicalScreeningOutput.model_json_schema(),
    }

    def invoke(clinical_request: str, clinical_context: str) -> ClinicalScreeningOutput:
        messages: Any = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_clinical_input(clinical_request, clinical_context)},
        ]

        def _extract_gguf_text(completion: Any) -> str:
            if not isinstance(completion, Mapping):
                raise ValueError("GGUF completion did not return a mapping payload.")
            choices = completion.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError("GGUF completion did not return any choices.")
            first = choices[0]
            if not isinstance(first, Mapping):
                raise ValueError("GGUF completion choice has an unexpected format.")
            msg = first.get("message")
            if not isinstance(msg, Mapping):
                raise ValueError("GGUF completion did not contain a chat message.")
            return _normalize_raw_content(msg.get("content", ""))

        try:
            completion = gguf_llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                response_format=response_format,
            )
            raw = _extract_gguf_text(completion)
            return ClinicalScreeningOutput.model_validate(json.loads(_extract_json_object(raw)))
        except Exception:
            grammar = LlamaGrammar.from_json_schema(
                json.dumps(ClinicalScreeningOutput.model_json_schema())
            )
            completion = gguf_llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                grammar=grammar,
            )
            raw = _extract_gguf_text(completion)
            return ClinicalScreeningOutput.model_validate(json.loads(_extract_json_object(raw)))

    return invoke


def create_remote_specialist_invoker(
    model: Any,
) -> SpecialistInvoker:
    """Create a remote structured specialist invoker.

    Args:
        model: A LangChain chat model instance.

    Returns:
        A specialist invoker function.
    """

    def invoke(clinical_request: str, clinical_context: str) -> ClinicalScreeningOutput:
        from screening_agent.model.structured_output import ResilientStructuredOutputInvoker

        invoker = ResilientStructuredOutputInvoker(
            model=model,
            schema=ClinicalScreeningOutput,
            max_attempts=2,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_clinical_input(clinical_request, clinical_context)),
        ]
        return invoker.invoke(messages)

    return invoke


def build_symptom_specialist_tool(invoker: SpecialistInvoker) -> BaseTool:
    """Create a LangChain tool that wraps the structured symptom specialist.

    Args:
        invoker: Backend-specific specialist invoker function.

    Returns:
        A LangChain tool compatible with ToolNode execution.
    """

    @tool
    def run_symptom_specialist(clinical_request: str, active_patient_context: str) -> str:
        """Run the structured symptom specialist and return validated JSON.

        Args:
            clinical_request: Symptom-focused user request.
            active_patient_context: Current active patient clinical context.

        Returns:
            A compact JSON string with the specialist contract.
        """

        payload = invoker(clinical_request, active_patient_context)
        return json.dumps(payload.model_dump(), ensure_ascii=False)

    return cast(BaseTool, run_symptom_specialist)
