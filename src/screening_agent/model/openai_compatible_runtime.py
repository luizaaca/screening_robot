"""OpenAI-compatible clinical backend implementation."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage

from screening_agent.graph.state import PatientRecord
from screening_agent.model.clinical_backend import ClinicalAnalysisPayload, build_clinical_input
from screening_agent.model.structured_output import ResilientStructuredOutputInvoker
from screening_agent.prompts import CLINICAL_ANALYSIS_SYSTEM_PROMPT


class OpenAICompatibleClinicalBackend:
    """Run clinical analysis through an OpenAI-compatible chat model."""

    backend_name = "openai_compatible"

    def __init__(self, model: BaseChatModel) -> None:
        """Initialize the backend.

        Args:
            model: Chat model configured for an OpenAI-compatible API.
        """

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
        return self._model.invoke(messages)
