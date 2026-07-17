"""Deterministic mock runtimes for local demos and end-to-end tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

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
_VIDEO_TERMS = (
    "video",
    "vídeo",
    "posture",
    "postura",
    "expression",
    "expressao",
    "expressão",
    "transcription",
    "transcricao",
    "transcrição",
    "face",
    "facial",
)
_VIDEO_ANALYSIS_TERMS = (
    "analyze video",
    "analyze this video",
    "analyse video",
    "analyse this video",
    "process video",
    "process this video",
    "summarize video",
    "summarize this video",
    "load video",
    "analise o video",
    "analise o vídeo",
    "processar video",
    "processar vídeo",
    "resuma o video",
    "resuma o vídeo",
)
_VIDEO_CLINICAL_TERMS = (
    "symptom",
    "symptoms",
    "condition",
    "conditions",
    "disease",
    "diseases",
    "worsening",
    "aggravation",
    "exam",
    "exams",
    "test",
    "tests",
    "sintoma",
    "sintomas",
    "condicao",
    "condição",
    "doenca",
    "doença",
    "agravamento",
    "piora",
    "exame",
    "exames",
)
_NAME_PARTICLES = frozenset({"da", "de", "di", "do", "das", "des", "dos", "e"})


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
        elif "structured clinical-context extractor for video evidence" in system_text:
            content = _video_clinical_extraction_response(messages)
        elif "clinical video analysis specialist" in system_text:
            content = _video_qa_response(messages)
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
        intent = _decide_intent(
            user_text,
            pending_candidates=pending_candidates,
            has_incoming_video_path=_extract_yes_no_session_flag(
                system_text,
                "New video path provided this turn",
            ),
            has_video_path=_extract_yes_no_session_flag(
                system_text,
                "Current video path available",
            ),
            has_video_result=_extract_yes_no_session_flag(
                system_text,
                "Video analysis result available",
            ),
        )
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
            followup_lookup = _next_patient_lookup_tool_call(messages, self.tools)
            if followup_lookup is not None:
                return followup_lookup
            return AIMessage(content=_coerce_content(messages[-1].content))

        if _supports_tool(self.tools, "run_symptom_specialist"):
            user_text = _get_latest_human_text(messages)
            active_patient_context = _extract_active_patient_context(_get_system_text(messages))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_symptom_specialist",
                        "args": {
                            "clinical_request": user_text,
                            "active_patient_context": active_patient_context,
                        },
                        "id": "call_run_symptom_specialist",
                        "type": "tool_call",
                    },
                ],
            )

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

        if _is_list_patients_request(user_text):
            return _patient_lookup_tool_call(
                name="list_patients",
                args={},
                call_id="call_list_patients",
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



def _decide_intent(
    user_text: str,
    *,
    pending_candidates: int,
    has_incoming_video_path: bool = False,
    has_video_path: bool = False,
    has_video_result: bool = False,
) -> str:
    """Classify the latest user message with deterministic routing heuristics.

    Args:
        user_text: Latest user message text.
        pending_candidates: Number of disambiguation candidates currently pending.
        has_incoming_video_path: Whether this turn supplied a new video path/upload.
        has_video_path: Whether a video path is already present in graph state.
        has_video_result: Whether a video analysis result is already present.

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

    has_video_reference = (
        has_incoming_video_path
        or has_video_path
        or has_video_result
        or _contains_any(lowered, list(_VIDEO_TERMS))
        or "video_path=" in lowered
        or "path:" in lowered
    )
    has_symptom_request = _contains_any(lowered, list(_SYMPTOM_TERMS))
    has_video_clinical_request = has_video_reference and (
        has_symptom_request or _contains_any(lowered, list(_VIDEO_CLINICAL_TERMS))
    )
    if has_video_clinical_request:
        return "video_symptom_analysis"
    if has_incoming_video_path:
        return "video_analysis"
    if has_video_reference and _contains_any(lowered, list(_VIDEO_ANALYSIS_TERMS)):
        return "video_analysis"
    if has_video_reference:
        return "video_interpretation"

    if _is_list_patients_request(user_text):
        return "patient_lookup"

    has_identifier = _extract_security_number(user_text) is not None or _extract_name_query(user_text) is not None
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
        "video_analysis": "The message asks to process or summarize a video.",
        "video_interpretation": "The message asks a general question about video evidence.",
        "video_symptom_analysis": "The message asks for clinical symptom analysis based on video evidence.",
        "video_upload_confirmation": "The message answers a pending video upload confirmation.",
        "video_qa": "The message asks a question about video evidence.",
        "clear_active_patient": "The message asks to reset the active patient context.",
        "invalid_request": "The message is outside the assistant scope.",
    }
    return rationale_map[intent]


def _next_patient_lookup_tool_call(messages: list[Any], tools: list[object]) -> AIMessage | None:
    """Return the next patient lookup tool call after a normal not-found result."""

    if not messages or not isinstance(messages[-1], ToolMessage):
        return None

    tool_message_text = _coerce_content(messages[-1].content).lower()
    if not (
        tool_message_text.startswith("no patient was found")
        or tool_message_text.startswith("no patient matched")
    ):
        return None

    latest_user_text = _get_latest_human_text(messages)
    if _supports_tool(tools, "lookup_patient_by_security_number"):
        attempted_numbers = _attempted_tool_arg_values(
            messages,
            tool_name="lookup_patient_by_security_number",
            arg_name="security_number",
        )
        for security_number in _extract_security_numbers(latest_user_text):
            if security_number not in attempted_numbers:
                return _patient_lookup_tool_call(
                    name="lookup_patient_by_security_number",
                    args={"security_number": security_number},
                    call_id=f"call_lookup_patient_by_security_number_{security_number}",
                )

    if _supports_tool(tools, "lookup_patient_by_name"):
        attempted_names = _attempted_tool_arg_values(
            messages,
            tool_name="lookup_patient_by_name",
            arg_name="full_name",
        )
        name_query = _extract_name_query(latest_user_text)
        if name_query is not None:
            for variant in _name_query_variants(name_query):
                if variant not in attempted_names:
                    return _patient_lookup_tool_call(
                        name="lookup_patient_by_name",
                        args={"full_name": variant},
                        call_id=f"call_lookup_patient_by_name_{_tool_call_id_suffix(variant)}",
                    )
    return None


def _attempted_tool_arg_values(
    messages: list[Any],
    *,
    tool_name: str,
    arg_name: str,
) -> set[str]:
    """Collect argument values already used in previous tool calls."""

    values: set[str] = set()
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or tool_call.get("name") != tool_name:
                continue
            args = tool_call.get("args")
            if not isinstance(args, dict):
                continue
            value = args.get(arg_name)
            if value is not None:
                values.add(str(value))
    return values


def _patient_lookup_tool_call(
    *,
    name: str,
    args: dict[str, object],
    call_id: str,
) -> AIMessage:
    """Build a deterministic patient-lookup tool call message."""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            },
        ],
    )



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


def _extract_yes_no_session_flag(system_text: str, label: str) -> bool:
    """Parse a yes/no session-context flag embedded in a router prompt."""

    match = re.search(rf"{re.escape(label)}: (yes|no)", system_text, re.IGNORECASE)
    return bool(match and match.group(1).lower() == "yes")



def _extract_security_number(text: str) -> str | None:
    """Extract a fictional 8-digit security number from text.

    Args:
        text: User message text.

    Returns:
        The first matching security number, if present.
    """

    security_numbers = _extract_security_numbers(text)
    return security_numbers[0] if security_numbers else None


def _extract_security_numbers(text: str) -> list[str]:
    """Extract all fictional 8-digit security numbers from text."""

    return list(dict.fromkeys(re.findall(r"\b(\d{8})\b", text)))


def _is_list_patients_request(text: str) -> bool:
    """Return whether the message asks to enumerate available patients."""

    lowered = text.lower()
    return _contains_any(
        lowered,
        [
            "list all patients",
            "list patients",
            "show all patients",
            "show patients",
            "available patients",
            "listar pacientes",
            "liste pacientes",
            "liste os pacientes",
            "listar todos",
            "liste todos",
            "mostrar pacientes",
            "mostre os pacientes",
            "todos os pacientes",
            "todas as pacientes",
        ],
    )


def _name_query_variants(name_query: str) -> list[str]:
    """Build deterministic fallback name queries after an initial miss."""

    normalized = _normalize_spaces(name_query)
    tokens = re.findall(r"[A-Za-zÀ-ÿ]+", normalized)
    significant_tokens = [
        token for token in tokens if token.lower() not in _NAME_PARTICLES
    ]
    variants = [normalized]
    if significant_tokens:
        simplified = " ".join(significant_tokens)
        variants.append(simplified)
        if len(significant_tokens) >= 2:
            variants.append(f"{significant_tokens[0]} {significant_tokens[-1]}")
            variants.append(f"{significant_tokens[-1]} {significant_tokens[0]}")
        variants.append(significant_tokens[0])
        if len(significant_tokens) > 1:
            variants.append(significant_tokens[-1])
    return list(dict.fromkeys(_normalize_spaces(variant) for variant in variants if variant.strip()))


def _tool_call_id_suffix(value: str) -> str:
    """Build a stable safe suffix for deterministic mock tool-call IDs."""

    suffix = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return suffix or "query"



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
        r"(?:buscar|busque|encontrar|encontre|localizar|localize)\s+(?:os\s+dados\s+d[ao]\s+)?(?:paciente\s+)?([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)+?)(?:\s+(?:tem|com|relata)|$)",
        r"dados\s+d[ao]\s+paciente\s+([A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)+?)(?:\s+(?:tem|com|relata)|$)",
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
    latest_user_message = str(payload.get("latest_user_message") or "")
    draft_response = str(payload.get("draft_response") or "I do not have a response yet.").strip()
    specialist_output = payload.get("specialist_output")
    response_instruction = str(payload.get("response_instruction") or "").strip()
    active_patient_record = payload.get("active_patient_record")
    is_ptbr = _looks_like_portuguese(latest_user_message or draft_response)

    if isinstance(specialist_output, dict):
        candidate_diseases = [str(item) for item in specialist_output.get("candidate_diseases", [])]
        recommended_exams = [
            str(item) for item in specialist_output.get("recommended_exams_tests", [])
        ]
        support_status = str(specialist_output.get("support_status") or "inconclusive")
        if is_ptbr:
            if support_status == "inconclusive":
                body = (
                    "As informações atuais são inconclusivas. "
                    f"Condições candidatas: {', '.join(candidate_diseases)}. "
                    f"Exames recomendados: {', '.join(recommended_exams)}."
                )
            else:
                body = (
                    f"Condições mais prováveis: {', '.join(candidate_diseases)}. "
                    f"Exames recomendados: {', '.join(recommended_exams)}."
                )
        else:
            if support_status == "inconclusive":
                body = (
                    "The available information is inconclusive. "
                    f"Candidate conditions: {', '.join(candidate_diseases)}. "
                    f"Recommended exams/tests: {', '.join(recommended_exams)}."
                )
            else:
                body = (
                    f"Most likely conditions: {', '.join(candidate_diseases)}. "
                    f"Recommended exams/tests: {', '.join(recommended_exams)}."
                )
        response_sections = [section for section in [header, body] if section]
        return "\n\n".join(response_sections)

    if (
        response_instruction
        and "only loaded a patient record" in response_instruction
        and isinstance(active_patient_record, dict)
    ):
        clinical_context = str(active_patient_record.get("clinical_context") or "").strip()
        if clinical_context:
            body = (
                f"Contexto do paciente carregado com sucesso. Resumo da ficha: {clinical_context}"
                if is_ptbr
                else f"Patient context loaded successfully. Record summary: {clinical_context}"
            )
        else:
            body = draft_response
        response_sections = [section for section in [header, body] if section]
        return "\n\n".join(response_sections)

    response_sections = [section for section in [header, draft_response] if section]
    return "\n\n".join(response_sections)


def _video_qa_response(messages: list[Any]) -> str:
    """Assemble a deterministic video QA response from the JSON payload."""

    payload_text = _get_latest_human_text(messages)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return "I could not read the video analysis payload."

    latest_user_message = str(payload.get("latest_user_message") or "")
    summary = str(payload.get("video_analysis_summary") or "").strip()
    active_patient = payload.get("active_patient")
    patient_line = ""
    if isinstance(active_patient, dict) and active_patient.get("full_name"):
        patient_line = f" Active patient context: {active_patient['full_name']}."

    if _looks_like_portuguese(latest_user_message):
        base = summary or "A analise de video esta disponivel, mas sem achados resumidos."
        return (
            f"Com base no video processado: {base}{patient_line} "
            "Isto e suporte de triagem, nao diagnostico definitivo."
        )

    base = summary or "The video analysis is available, but no summarized findings were captured."
    return (
        f"Based on the processed video: {base}{patient_line} "
        "This is screening support, not a definitive diagnosis."
    )


def _video_clinical_extraction_response(messages: list[Any]) -> str:
    """Return deterministic structured clinical context for video tests."""

    payload_text = _get_latest_human_text(messages)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = {}

    summary = str(payload.get("video_analysis_summary") or "").lower()
    symptoms: list[str] = []
    signs: list[str] = []
    evidence: list[dict[str, object]] = []

    if "tired" in summary or "fatigue" in summary:
        symptoms.append("fatigue or tiredness mentioned in speech")
        evidence.append(
            {
                "observation": "fatigue or tiredness mentioned in speech",
                "source": "transcription",
                "time_range_s": [1.0, 4.0],
            },
        )
    if "sad_expression" in summary or "sad expression" in summary:
        signs.append("sad facial expression pattern")
        evidence.append(
            {
                "observation": "sad facial expression pattern",
                "source": "expression",
                "time_range_s": [0.0, 8.0],
            },
        )
    if "head_down" in summary or "head down" in summary:
        signs.append("head-down posture pattern")
        evidence.append(
            {
                "observation": "head-down posture pattern",
                "source": "posture",
                "time_range_s": [0.0, 8.0],
            },
        )

    return json.dumps(
        {
            "reported_or_inferred_symptoms": symptoms,
            "observable_signs": signs,
            "evidence": evidence,
            "limitations": ["Video evidence is supportive and cannot establish a diagnosis."],
            "uncertainties": ["Clinical significance depends on exam and full history."],
            "clinical_attention_points": ["Consider whether observed behavior aligns with reported symptoms."],
        },
        ensure_ascii=False,
    )


def _supports_tool(tools: list[object], tool_name: str) -> bool:
    """Return whether a bound tool list contains the requested tool name.

    Args:
        tools: Bound tool definitions.
        tool_name: Tool name to search for.

    Returns:
        `True` when the tool is available.
    """

    for tool in tools:
        if getattr(tool, "name", None) == tool_name:
            return True
    return False


def _extract_active_patient_context(system_text: str) -> str:
    """Extract the active patient context block embedded in the specialist prompt.

    Args:
        system_text: System prompt text.

    Returns:
        Active patient context string.
    """

    marker = "Resolved clinical context:\n"
    if marker not in system_text:
        marker = "Resolved active patient context:\n"
    start_index = system_text.find(marker)
    if start_index == -1:
        return "No active patient context was loaded."
    remainder = system_text[start_index + len(marker) :]
    end_marker = "\n\nLatest user message:"
    end_index = remainder.find(end_marker)
    if end_index == -1:
        return remainder.strip() or "No active patient context was loaded."
    return remainder[:end_index].strip() or "No active patient context was loaded."



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
            "listar",
            "liste",
            "todos",
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
