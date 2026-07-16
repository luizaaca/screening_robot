"""Tests for final-answer prompt payload composition."""

from __future__ import annotations

import json
from typing import Any

from langchain.messages import AIMessage, HumanMessage

from screening_agent.graph.nodes.finalize_response import build_finalize_response_node


class _RecordingFinalAnswerModel:
    """Control-model stub that records the final-answer JSON payload."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.system_prompts: list[str] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.system_prompts.append(str(messages[0].content))
        payload = json.loads(str(messages[-1].content))
        self.payloads.append(payload)
        return AIMessage(content="Loaded patient response.")


def test_patient_lookup_finalizer_receives_record_summary_instruction() -> None:
    """Loaded lookup turns should give the finalizer a masked record summary contract."""

    model = _RecordingFinalAnswerModel()
    node = build_finalize_response_node(model)

    node(
        {
            "messages": [HumanMessage(content="Lookup patient 11112222")],
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

    payload = model.payloads[0]
    active_patient_record = payload["active_patient_record"]

    assert payload["response_instruction"]
    assert "clinical_disclaimer" not in payload
    assert "Respond in the same language as `latest_user_message`" in model.system_prompts[0]
    assert isinstance(active_patient_record, dict)
    assert active_patient_record["masked_security_number"] == "****2222"
    assert "Type 2 diabetes mellitus" in str(active_patient_record["clinical_context"])
    assert "11112222" not in str(payload["draft_response"])
    assert "11112222" not in json.dumps(active_patient_record)


def test_video_correlation_finalizer_receives_active_patient_record() -> None:
    """Video follow-ups with an active patient should expose masked record context."""

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
            "video_interpretation": (
                "A fala do vídeo menciona medo e possível violência; a expressão facial "
                "permanece predominantemente neutra."
            ),
        }
    )

    payload = model.payloads[0]
    active_patient_record = payload["active_patient_record"]

    assert payload["response_instruction"]
    assert isinstance(active_patient_record, dict)
    assert active_patient_record["masked_security_number"] == "****4321"
    assert "intimate partner violence" in str(active_patient_record["clinical_context"])
    assert "87654321" not in json.dumps(active_patient_record)


def test_processing_error_finalizer_does_not_reuse_stale_video_interpretation() -> None:
    """Processing-error turns should not answer from stale video interpretation state."""

    model = _RecordingFinalAnswerModel()
    node = build_finalize_response_node(model)
    error_detail = (
        "I could not complete this request safely after repeated structured-processing attempts."
    )

    node(
        {
            "messages": [
                HumanMessage(
                    content="correlacione o histórico da paciente com o vídeo"
                )
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

    payload = model.payloads[0]

    assert payload["draft_response"] == error_detail
    assert payload["active_patient_record"] is None
    assert payload["response_instruction"] is None
    assert "Stale previous video interpretation" not in str(payload["draft_response"])
