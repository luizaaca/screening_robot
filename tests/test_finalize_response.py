"""Tests for final-answer prompt payload composition."""

from __future__ import annotations

from typing import Any
import json

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from screening_agent.graph.nodes.finalize_response import build_finalize_response_node


class _RecordingFinalAnswerModel:
    """Control-model stub that records the final-answer JSON payload."""

    def __init__(self, response_content: str = "Loaded patient response.") -> None:
        self.payloads: list[dict[str, object]] = []
        self.system_prompts: list[str] = []
        self.response_content = response_content

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.system_prompts.append(str(messages[0].content))
        payload = json.loads(str(messages[-1].content))
        self.payloads.append(payload)
        return AIMessage(content=self.response_content)


def _recorded_context(model: _RecordingFinalAnswerModel) -> dict[str, object]:
    payload = model.payloads[0]
    context = payload["final_answer_context"]
    assert isinstance(context, dict)
    return context


def test_patient_lookup_finalizer_receives_full_sanitized_context() -> None:
    """Loaded lookup turns should expose complete sanitized state to the finalizer."""

    model = _RecordingFinalAnswerModel(
        response_content="Loaded patient response with security number 11112222."
    )
    node = build_finalize_response_node(model)

    result = node(
        {
            "messages": [
                HumanMessage(content="Lookup patient 11112222"),
                AIMessage(content="Calling lookup tool for 11112222."),
                ToolMessage(
                    content="Loaded patient John Souza with security number 11112222.",
                    name="lookup_patient_by_security_number",
                    tool_call_id="call_lookup",
                ),
            ],
            "router_intent": "patient_lookup",
            "patient_lookup_status": "loaded",
            "active_patient": {
                "security_number": "11112222",
                "full_name": "John Souza",
                "clinical_context": (
                    "47-year-old male. Conditions: Type 2 diabetes mellitus. "
                    "Medications: Metformin."
                ),
            },
            "last_response": (
                "Patient context loaded successfully.\n"
                "Name: John Souza\n"
                "Security number: 11112222\n"
                "Clinical context: 47-year-old male. Conditions: Type 2 diabetes mellitus."
            ),
        }
    )

    context = _recorded_context(model)
    payload_text = json.dumps(model.payloads[0], ensure_ascii=False)
    state_snapshot = context["state_snapshot"]
    conversation_history = context["conversation_history"]
    derived_context = context["derived_context"]

    assert "final_answer_context" in model.payloads[0]
    assert "active_patient_record" not in model.payloads[0]
    assert "clinical_disclaimer" not in model.payloads[0]
    assert "final_answer_context" in model.system_prompts[0]
    assert "Respond in the same language as `latest_user_message`" in model.system_prompts[0]
    assert isinstance(state_snapshot, dict)
    assert isinstance(conversation_history, list)
    assert state_snapshot["messages"] == conversation_history
    assert conversation_history[0]["role"] == "user"
    assert conversation_history[2]["role"] == "tool"
    assert conversation_history[2]["name"] == "lookup_patient_by_security_number"
    assert state_snapshot["active_patient"]["security_number"] == "****2222"
    assert "Type 2 diabetes mellitus" in str(state_snapshot["active_patient"]["clinical_context"])
    assert derived_context["response_instruction"]
    assert "11112222" not in payload_text
    assert "11112222" not in str(result["last_response"])
    assert "****2222" in str(result["last_response"])


def test_video_correlation_finalizer_receives_full_state_context() -> None:
    """Video follow-ups with an active patient should expose patient and video context."""

    model = _RecordingFinalAnswerModel()
    node = build_finalize_response_node(model)

    node(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "poderia haver alguma correlação entre a situação apresentada "
                        "pelo vídeo e o histórico da paciente?"
                    )
                )
            ],
            "router_intent": "video_interpretation",
            "active_patient": {
                "security_number": "87654321",
                "full_name": "Maria Silva",
                "clinical_context": (
                    "34-year-old female. Conditions: Anxiety. "
                    "History: reports fear and intimate partner violence."
                ),
            },
            "video_analysis_summary": "Transcription reports fear and violence.",
            "video_interpretation": (
                "A fala do vídeo menciona medo e possível violência; a expressão facial "
                "permanece predominantemente neutra."
            ),
        }
    )

    context = _recorded_context(model)
    state_snapshot = context["state_snapshot"]
    derived_context = context["derived_context"]

    assert isinstance(state_snapshot, dict)
    assert isinstance(derived_context, dict)
    assert derived_context["response_instruction"]
    assert state_snapshot["active_patient"]["security_number"] == "****4321"
    assert "intimate partner violence" in str(state_snapshot["active_patient"]["clinical_context"])
    assert "possível violência" in str(state_snapshot["video_interpretation"])
    assert "87654321" not in json.dumps(model.payloads[0], ensure_ascii=False)


def test_processing_error_finalizer_marks_stale_evidence_without_dropping_snapshot() -> None:
    """Processing-error turns should preserve state but mark stale evidence clearly."""

    model = _RecordingFinalAnswerModel()
    node = build_finalize_response_node(model)
    error_detail = (
        "I could not complete this request safely after repeated structured-processing attempts."
    )

    node(
        {
            "messages": [
                HumanMessage(content="correlacione o histórico da paciente com o vídeo")
            ],
            "router_intent": "video_interpretation",
            "active_patient": {
                "security_number": "87654321",
                "full_name": "Maria Silva",
                "clinical_context": "History: anxiety.",
            },
            "video_interpretation": "Stale previous video interpretation.",
            "last_response": error_detail,
            "turn_outcome": {
                "type": "processing_error",
                "detail": error_detail,
            },
        }
    )

    context = _recorded_context(model)
    state_snapshot = context["state_snapshot"]
    derived_context = context["derived_context"]

    assert isinstance(state_snapshot, dict)
    assert isinstance(derived_context, dict)
    assert state_snapshot["active_patient"]["security_number"] == "****4321"
    assert state_snapshot["video_interpretation"] == "Stale previous video interpretation."
    assert derived_context["draft_response"] == error_detail
    assert "processing_error" in str(context["response_task"])
    assert "stale" in " ".join(str(note) for note in context["context_notes"])
    assert "87654321" not in json.dumps(model.payloads[0], ensure_ascii=False)


def test_symptom_followup_finalizer_receives_patient_specialist_and_video_context() -> None:
    """Regression: pre-condition follow-ups need patient, specialist, and video state."""

    model = _RecordingFinalAnswerModel()
    node = build_finalize_response_node(model)
    specialist_output = {
        "support_status": "inconclusive",
        "candidate_diseases": [
            "Hypothyroidism control issue",
            "Orthostatic hypotension",
        ],
        "recommended_exams_tests": ["TSH", "Free T4", "ECG"],
    }
    video_context = {
        "reported_or_inferred_symptoms": ["tontura", "sensação de desmaio"],
        "clinical_attention_points": ["correlacionar com histórico tireoidiano"],
    }

    node(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "poderia ter relação com as pre-condições dela conforme seu histórico?"
                    )
                )
            ],
            "router_intent": "symptom_analysis",
            "patient_lookup_status": "loaded",
            "active_patient": {
                "security_number": "55553847",
                "full_name": "Beatriz Lima",
                "clinical_context": (
                    "42-year-old female. Conditions: Hypothyroidism and hypertension. "
                    "Medications: Levothyroxine."
                ),
            },
            "specialist_output_json": json.dumps(specialist_output, ensure_ascii=False),
            "video_clinical_context_json": json.dumps(video_context, ensure_ascii=False),
            "video_interpretation": "The video suggests dizziness and near-fainting.",
        }
    )

    context = _recorded_context(model)
    payload_text = json.dumps(model.payloads[0], ensure_ascii=False)
    state_snapshot = context["state_snapshot"]
    derived_context = context["derived_context"]

    assert isinstance(state_snapshot, dict)
    assert isinstance(derived_context, dict)
    assert state_snapshot["active_patient"]["full_name"] == "Beatriz Lima"
    assert state_snapshot["active_patient"]["security_number"] == "****3847"
    assert "Hypothyroidism" in str(state_snapshot["active_patient"]["clinical_context"])
    assert "Levothyroxine" in str(state_snapshot["active_patient"]["clinical_context"])
    assert derived_context["specialist_output"] == specialist_output
    assert derived_context["video_clinical_context"] == video_context
    assert "55553847" not in payload_text
