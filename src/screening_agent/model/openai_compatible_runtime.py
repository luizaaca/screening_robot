"""Remote structured clinical backend implementation."""

from __future__ import annotations

from langchain.messages import HumanMessage, SystemMessage

from screening_agent.audit import emit_custom_debug_event
from screening_agent.graph.state import PatientRecord
from screening_agent.model.clinical_backend import ClinicalAnalysisPayload, build_clinical_input
from screening_agent.model.structured_output import ResilientStructuredOutputInvoker, StructuredOutputCapableModel
from screening_agent.prompts import CLINICAL_ANALYSIS_SYSTEM_PROMPT


class OpenAICompatibleClinicalBackend:
    """Run clinical analysis through a structured remote chat model."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        model: StructuredOutputCapableModel,
        *,
        backend_name: str = "openai_compatible",
    ) -> None:
        """Initialize the backend.

        Args:
            model: Chat model configured for a structured remote API.
            backend_name: Human-readable backend family used in state.
        """

        self.backend_name = backend_name
        self._model = ResilientStructuredOutputInvoker(model, ClinicalAnalysisPayload)

    def analyze(
        self,
        *,
        user_message: str,
        active_patient: PatientRecord | None,
    ) -> ClinicalAnalysisPayload:
        """Analyze a clinical request using structured output.

        Args:
            user_message: Latest user complaint or question.
            active_patient: Optional active patient record.

        Returns:
            Structured clinical analysis payload.
        """

        messages = [
            SystemMessage(content=CLINICAL_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=build_clinical_input(user_message, active_patient)),
        ]
        emit_custom_debug_event(
            "clinical_analysis_prompt",
            backend_name=self.backend_name,
            payload={"messages": messages},
        )
        result = self._model.invoke(messages)
        emit_custom_debug_event(
            "clinical_analysis_result",
            backend_name=self.backend_name,
            payload={"parsed_payload": result.model_dump()},
        )
        return result
