"""Deterministic mock runtimes for local demos and end-to-end tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, cast

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from screening_agent.graph.state import PatientRecord
from screening_agent.model.clinical_backend import ClinicalAnalysisPayload
from screening_agent.model.control_models import ControlModel, StructuredOutputInvoker, ToolBoundControlModel

_SYMPTOM_TERMS = (
    "pain",
    "dor",
    "fever",
    "febre",
    "cough",
    "tosse",
    "fatigue",
    "fadiga",
    "shortness of breath",
    "falta de ar",
    "palpitations",
    "palpitações",
    "palpitacoes",
    "headache",
    "cefaleia",
    "nausea",
    "náusea",
    "nausea",
    "urination",
    "urinar",
    "thirst",
    "sede",
    "chest tightness",
    "chest pain",
    "aperto no peito",
)


class MockControlModel(ControlModel):
    """Deterministic control model for local demos without external LLM access."""

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Generate a deterministic plain-text AI response.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            AI message for usage, invalid-request, or clear-context nodes.
        """

        system_text = _get_system_text(messages)
        user_text = _get_latest_human_text(messages)
        is_ptbr = _looks_like_portuguese(user_text)
        if "explaining how the product can be used" in system_text:
            content = _usage_response(is_ptbr=is_ptbr)
        elif "handling an out-of-scope request" in system_text:
            content = _invalid_response(is_ptbr=is_ptbr)
        elif "confirming patient-context reset" in system_text:
            had_active_patient = "Active patient before clearing: none" not in system_text
            content = _clear_response(is_ptbr=is_ptbr, had_active_patient=had_active_patient)
        elif "final response composer" in system_text:
            content = _final_answer_response(messages)
        else:
            content = _generic_response(is_ptbr=is_ptbr)
        return AIMessage(content=content)

    def bind_tools(self, tools: list[object]) -> ToolBoundControlModel:
        """Bind patient-lookup tools for deterministic tool-calling behavior.

        Args:
            tools: Tool definitions exposed to the model.

        Returns:
            Tool-bound mock control model.
        """

        return _MockToolBoundControlModel(tools)

    def with_structured_output(
        self,
        schema: type[BaseModel],
    ) -> StructuredOutputInvoker[BaseModel]:
        """Return a wrapper that emits deterministic structured output.

        Args:
            schema: Pydantic schema expected by the caller.

        Returns:
            Structured-output wrapper.
        """

        return _MockStructuredControlModel(schema)


class MockClinicalBackend:
    """Deterministic clinical backend for demos and tests."""

    backend_name = "mock"

    def analyze(
        self,
        *,
        user_message: str,
        active_patient: PatientRecord | None,
    ) -> ClinicalAnalysisPayload:
        """Return a deterministic structured clinical analysis.

        Args:
            user_message: Latest user complaint.
            active_patient: Optional active patient context.

        Returns:
            Structured clinical analysis payload.
        """

        lowered = user_message.lower()
        is_ptbr = _looks_like_portuguese(user_message)
        patient_reference = (
            active_patient["full_name"] if active_patient is not None else ("the current report" if not is_ptbr else "o relato atual")
        )

        status = "analysis_ready"
        primary_hypothesis = "Non-specific symptomatic syndrome"
        differential_hypotheses = ["Viral syndrome", "Medication side effect"]
        recommended_exams = ["Complete blood count (CBC)", "Basic metabolic panel (BMP)"]
        reasoning_summary = (
            f"Mock analysis reviewed {patient_reference} and found a non-specific symptom pattern that still needs confirmatory testing."
            if not is_ptbr
            else f"A análise mock revisou {patient_reference} e encontrou um padrão sintomático inespecífico que ainda precisa de testes confirmatórios."
        )
        safety_notes = []

        if _contains_any(lowered, ["fatigue", "frequent urination", "polydipsia", "sede", "fadiga", "urination"]):
            primary_hypothesis = "Diabetes mellitus or poor glycemic control"
            differential_hypotheses = ["Urinary tract disorder", "Dehydration"]
            recommended_exams = ["Serum glucose", "HbA1c", "Urinalysis"]
            reasoning_summary = (
                "The complaint suggests a metabolic pattern compatible with diabetes or inadequate glycemic control."
                if not is_ptbr
                else "A queixa sugere um padrão metabólico compatível com diabetes ou controle glicêmico inadequado."
            )
        elif _contains_any(lowered, ["chest pain", "chest tightness", "palpitations", "shortness of breath", "aperto no peito", "falta de ar"]):
            status = "urgent_attention"
            primary_hypothesis = "Cardiopulmonary cause requiring urgent assessment"
            differential_hypotheses = ["Acute coronary syndrome", "Arrhythmia", "Anxiety or panic episode"]
            recommended_exams = ["Electrocardiogram (ECG)", "Pulse oximetry", "Chest X-ray"]
            reasoning_summary = (
                "Chest symptoms with possible cardiopulmonary involvement justify urgent triage and targeted testing."
                if not is_ptbr
                else "Sintomas torácicos com possível envolvimento cardiopulmonar justificam triagem urgente e exames direcionados."
            )
            safety_notes = [
                "Seek immediate emergency evaluation if symptoms are severe, persistent, or associated with syncope."
                if not is_ptbr
                else "Procure avaliação de urgência imediatamente se os sintomas forem intensos, persistentes ou associados a síncope."
            ]
        elif _contains_any(lowered, ["cough", "fever", "wheezing", "tosse", "febre", "chiado"]):
            primary_hypothesis = "Respiratory infection or airway exacerbation"
            differential_hypotheses = ["Asthma exacerbation", "Bronchitis"]
            recommended_exams = ["Pulse oximetry", "Chest X-ray", "Complete blood count (CBC)"]
            reasoning_summary = (
                "Respiratory symptoms raise the possibility of infection or obstructive-airway worsening."
                if not is_ptbr
                else "Sintomas respiratórios levantam a possibilidade de infecção ou piora obstrutiva das vias aéreas."
            )
        elif _contains_any(lowered, ["headache", "photophobia", "nausea", "cefaleia", "fotofobia", "náusea"]):
            primary_hypothesis = "Migraine or primary headache syndrome"
            differential_hypotheses = ["Tension headache", "Secondary headache cause"]
            recommended_exams = ["Neurologic examination", "Blood pressure assessment"]
            reasoning_summary = (
                "The headache pattern is compatible with a primary headache syndrome but still requires focused screening."
                if not is_ptbr
                else "O padrão da cefaleia é compatível com uma síndrome de cefaleia primária, mas ainda exige triagem focada."
            )

        if is_ptbr:
            user_response = (
                f"Hipótese principal: {primary_hypothesis}. Exames sugeridos: {', '.join(recommended_exams)}. "
                f"Resumo: {reasoning_summary}"
            )
        else:
            user_response = (
                f"Primary hypothesis: {primary_hypothesis}. Suggested exams: {', '.join(recommended_exams)}. "
                f"Summary: {reasoning_summary}"
            )

        return ClinicalAnalysisPayload(
            status=cast(Any, status),
            primary_hypothesis=primary_hypothesis,
            differential_hypotheses=differential_hypotheses,
            recommended_exams=recommended_exams,
            reasoning_summary=reasoning_summary,
            safety_notes=safety_notes,
            user_response=user_response,
        )


@dataclass(frozen=True)
class _MockStructuredControlModel(StructuredOutputInvoker[BaseModel]):
    """Structured-output wrapper for router decisions."""

    schema: type[BaseModel]

    def invoke(self, messages: list[Any]) -> BaseModel:
        """Return a deterministic router decision object.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            Schema instance populated with deterministic routing data.
        """

        user_text = _get_latest_human_text(messages)
        system_text = _get_system_text(messages)
        pending_candidates = _extract_pending_candidate_count(system_text)
        intent = _decide_intent(user_text, pending_candidates=pending_candidates)
        rationale = _build_rationale(intent)
        return self.schema.model_validate({"intent": intent, "rationale": rationale})


@dataclass(frozen=True)
class _MockToolBoundControlModel(ToolBoundControlModel):
    """Tool-bound wrapper that decides which patient tool should run next."""

    tools: list[object]

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Return a deterministic tool call or final lookup response.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            AI message with tool calls or plain lookup content.
        """

        if messages and isinstance(messages[-1], ToolMessage):
            return AIMessage(content=_coerce_content(messages[-1].content))

        user_text = _get_latest_human_text(messages)
        system_text = _get_system_text(messages)
        pending_candidates = _extract_pending_candidate_count(system_text)
        selection_index = _extract_selection_index(user_text)
        if pending_candidates > 0 and selection_index is not None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "activate_patient_selection",
                        "args": {"selection_index": selection_index},
                        "id": "call_activate_patient_selection",
                        "type": "tool_call",
                    },
                ],
            )

        security_number = _extract_security_number(user_text)
        if security_number is not None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_patient_by_security_number",
                        "args": {"security_number": security_number},
                        "id": "call_lookup_patient_by_security_number",
                        "type": "tool_call",
                    },
                ],
            )

        name_query = _extract_name_query(user_text)
        if name_query is not None:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_patient_by_name",
                        "args": {"full_name": name_query},
                        "id": "call_lookup_patient_by_name",
                        "type": "tool_call",
                    },
                ],
            )

        is_ptbr = _looks_like_portuguese(user_text)
        return AIMessage(
            content=(
                "Por favor, informe o nome do paciente ou o número fictício de 8 dígitos."
                if is_ptbr
                else "Please provide the patient name or the fictional 8-digit security number."
            ),
        )



def _decide_intent(user_text: str, *, pending_candidates: int) -> str:
    """Classify the latest user message with deterministic routing heuristics.

    Args:
        user_text: Latest user message text.
        pending_candidates: Number of disambiguation candidates currently pending.

    Returns:
        Intent string understood by the router node.
    """

    lowered = user_text.lower()
    if pending_candidates > 0 and _extract_selection_index(user_text) is not None:
        return "patient_lookup"
    if _contains_any(
        lowered,
        [
            "how to use",
            "how do i use",
            "use this assistant",
            "what can you do",
            "help",
            "como usar",
            "como eu uso",
            "ajuda",
            "o que você pode fazer",
            "o que voce pode fazer",
        ],
    ):
        return "usage_instructions"
    if _contains_any(lowered, ["clear active patient", "clear patient", "forget patient", "reset patient", "limpar paciente", "esquecer paciente", "remover paciente ativo"]):
        return "clear_active_patient"
    if _contains_any(lowered, ["weather", "capital of", "tell me a joke", "write a poem", "piada", "previsão do tempo"]):
        return "invalid_request"

    has_identifier = _extract_security_number(user_text) is not None or _extract_name_query(user_text) is not None
    has_symptom_request = _contains_any(lowered, list(_SYMPTOM_TERMS))
    if has_identifier and has_symptom_request:
        return "patient_lookup_then_analysis"
    if has_identifier:
        return "patient_lookup"
    if has_symptom_request:
        return "symptom_analysis"
    return "invalid_request"



def _build_rationale(intent: str) -> str:
    """Return a short deterministic rationale for a router decision.

    Args:
        intent: Selected router intent.

    Returns:
        One-sentence rationale.
    """

    rationale_map = {
        "usage_instructions": "The message asks for product guidance or capabilities.",
        "patient_lookup": "The message focuses on identifying a patient or choosing from candidates.",
        "patient_lookup_then_analysis": "The message combines patient identification with a clinical complaint.",
        "symptom_analysis": "The message describes symptoms or requests a clinical screening interpretation.",
        "clear_active_patient": "The message asks to reset the active patient context.",
        "invalid_request": "The message is outside the assistant scope.",
    }
    return rationale_map[intent]



def _get_system_text(messages: list[Any]) -> str:
    """Extract the first system message text from a message list.

    Args:
        messages: LangChain-compatible message list.

    Returns:
        Concatenated system message content.
    """

    for message in messages:
        if isinstance(message, SystemMessage):
            return _coerce_content(message.content)
    return ""



def _get_latest_human_text(messages: list[Any]) -> str:
    """Extract the latest human message text from a message list.

    Args:
        messages: LangChain-compatible message list.

    Returns:
        Latest human message content.
    """

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _coerce_content(message.content)
    return ""



def _extract_pending_candidate_count(system_text: str) -> int:
    """Parse the pending-candidate count embedded in lookup/router prompts.

    Args:
        system_text: System prompt text.

    Returns:
        Parsed pending-candidate count.
    """

    match = re.search(r"Pending (?:patient )?candidates: (\d+)", system_text)
    return int(match.group(1)) if match else 0



def _extract_security_number(text: str) -> str | None:
    """Extract a fictional 8-digit security number from text.

    Args:
        text: User message text.

    Returns:
        The first matching security number, if present.
    """

    match = re.search(r"\b(\d{8})\b", text)
    return match.group(1) if match else None



def _extract_selection_index(text: str) -> int | None:
    """Extract a numeric disambiguation choice from text.

    Args:
        text: User message text.

    Returns:
        Integer selection index when the text is a simple choice.
    """

    normalized = text.strip()
    if re.fullmatch(r"\d+", normalized):
        return int(normalized)
    match = re.search(r"(?:option|choose|pick|selecione|escolho|quero)\s+(\d+)", normalized, re.IGNORECASE)
    return int(match.group(1)) if match else None



def _extract_name_query(text: str) -> str | None:
    """Extract a likely patient name from a lookup-oriented message.

    Args:
        text: User message text.

    Returns:
        Normalized patient name when one can be inferred.
    """

    explicit_patterns = [
        r"(?:find|lookup|search for)\s+(?:patient\s+)?([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)+?)(?:\s+(?:has|with|reports|complains)|$)",
        r"patient\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)+?)(?:\s+(?:has|with|reports|complains)|$)",
        r"(?:buscar|encontrar|localizar)\s+(?:paciente\s+)?([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)+?)(?:\s+(?:tem|com|relata)|$)",
        r"paciente\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)+?)(?:\s+(?:tem|com|relata)|$)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_spaces(match.group(1))
    if re.search(r"\d{8}", text):
        return None
    title_case_match = re.findall(r"[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+", text)
    if title_case_match:
        return _normalize_spaces(title_case_match[-1])
    return None



def _usage_response(*, is_ptbr: bool) -> str:
    """Return a deterministic usage message.

    Args:
        is_ptbr: Whether the conversation appears to be in Portuguese.

    Returns:
        Usage instructions in the selected language.
    """

    if is_ptbr:
        return (
            "Posso localizar um paciente por nome ou número fictício de 8 dígitos, manter um único paciente ativo por sessão, "
            "limpar esse contexto quando solicitado e analisar sintomas para sugerir hipóteses e exames relevantes dentro do escopo de triagem."
        )
    return (
        "I can look up a patient by name or fictional 8-digit security number, keep one active patient in session context, "
        "clear that context on request, and analyze symptoms to suggest likely conditions and relevant exams within a screening workflow."
    )



def _invalid_response(*, is_ptbr: bool) -> str:
    """Return a deterministic out-of-scope response.

    Args:
        is_ptbr: Whether the conversation appears to be in Portuguese.

    Returns:
        Out-of-scope message.
    """

    if is_ptbr:
        return (
            "Posso apenas explicar o uso do assistente, localizar pacientes, limpar o paciente ativo e analisar sintomas com exames sugeridos."
        )
    return (
        "I can only explain how to use the assistant, retrieve a patient record, clear the active patient, and analyze symptoms with suggested exams."
    )



def _clear_response(*, is_ptbr: bool, had_active_patient: bool) -> str:
    """Return a deterministic clear-context confirmation.

    Args:
        is_ptbr: Whether the conversation appears to be in Portuguese.
        had_active_patient: Whether a patient was active before the clear request.

    Returns:
        Clear-context confirmation.
    """

    if is_ptbr:
        return "O contexto do paciente ativo foi limpo." if had_active_patient else "Não havia paciente ativo carregado."
    return "The active patient context was cleared." if had_active_patient else "There was no active patient loaded."



def _generic_response(*, is_ptbr: bool) -> str:
    """Return a generic deterministic control-model fallback.

    Args:
        is_ptbr: Whether the conversation appears to be in Portuguese.

    Returns:
        Generic fallback text.
    """

    return "Resposta mock gerada." if is_ptbr else "Mock response generated."


def _final_answer_response(messages: list[Any]) -> str:
    """Assemble the final answer from the JSON payload used by the final node.

    Args:
        messages: LangChain-compatible message list.

    Returns:
        Deterministically assembled final response.
    """

    payload_text = _get_latest_human_text(messages)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return _generic_response(is_ptbr=False)

    header = str(payload.get("active_patient_header") or "").strip()
    response_body = str(payload.get("response_body") or "I do not have a response yet.").strip()
    response_sections = [section for section in [header, response_body] if section]
    if bool(payload.get("response_requires_disclaimer")):
        disclaimer = str(payload.get("clinical_disclaimer") or "").strip()
        if disclaimer:
            response_sections.append(disclaimer)
    return "\n\n".join(response_sections)



def _coerce_content(content: object) -> str:
    """Normalize message content into a string.

    Args:
        content: Message content object.

    Returns:
        Plain-text representation.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for block in content:
            if isinstance(block, str):
                fragments.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                fragments.append(str(block.get("text", "")))
        return "\n".join(fragment for fragment in fragments if fragment).strip()
    return str(content)



def _contains_any(text: str, options: list[str]) -> bool:
    """Return whether any candidate substring is present.

    Args:
        text: Text to scan.
        options: Candidate substrings.

    Returns:
        `True` when at least one candidate appears.
    """

    return any(option in text for option in options)



def _looks_like_portuguese(text: str) -> bool:
    """Estimate whether the user text is in Portuguese.

    Args:
        text: User text to inspect.

    Returns:
        `True` when Portuguese cues are detected.
    """

    lowered = text.lower()
    return _contains_any(
        lowered,
        [
            " paciente",
            " sintomas",
            " exame",
            " como ",
            " dor",
            " febre",
            " falta de ar",
            " limpar",
            "ajuda",
            "tem ",
            "relata",
        ],
    )



def _normalize_spaces(text: str) -> str:
    """Normalize repeated whitespace in a text string.

    Args:
        text: Raw text.

    Returns:
        Whitespace-normalized text.
    """

    return re.sub(r"\s+", " ", text).strip()
