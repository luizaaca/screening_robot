"""Regression tests for structured-output fallbacks."""

from __future__ import annotations

from typing import Any

from langchain.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from screening_agent.graph.nodes.router import RouteDecision, build_router_node
from screening_agent.model.control_models import ControlModel, StructuredOutputInvoker, ToolBoundControlModel
from screening_agent.model.openai_compatible_runtime import OpenAICompatibleClinicalBackend


class _FailingStructuredInvoker(StructuredOutputInvoker[BaseModel]):
    """Test invoker that always reproduces a provider JSON validation failure."""

    def __init__(self, schema: type[BaseModel]) -> None:
        """Store the schema that should fail validation.

        Args:
            schema: Structured schema requested by the caller.
        """

        self._schema = schema

    def invoke(self, messages: list[Any]) -> BaseModel:
        """Raise a validation error that mimics a non-JSON provider response.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            This method never returns successfully.
        """

        return self._schema.model_validate_json(
            "invalid_request: The user asked for something outside the supported scope.",
        )


class _UnsupportedToolBinding(ToolBoundControlModel):
    """Placeholder tool-bound model used to satisfy the control-model protocol."""

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Reject tool binding in tests that do not need it.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            This method never returns successfully.

        Raises:
            NotImplementedError: Always, because the tests do not require tool calls.
        """

        raise NotImplementedError("Tool binding is not used in these fallback tests.")


class _FallbackLabelControlModel(ControlModel):
    """Control-model stub that falls back to label-style text responses."""

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Return a label-style route decision instead of JSON.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            Plain AI response in `<intent>: <rationale>` format.
        """

        return AIMessage(
            content="invalid_request: The message is outside the assistant clinical scope.",
        )

    def bind_tools(self, tools: list[object]) -> ToolBoundControlModel:
        """Return a placeholder tool-bound model.

        Args:
            tools: Tool definitions exposed to the model.

        Returns:
            Placeholder tool-bound model.
        """

        return _UnsupportedToolBinding()

    def with_structured_output(
        self,
        schema: type[BaseModel],
    ) -> StructuredOutputInvoker[BaseModel]:
        """Return a failing native structured-output invoker.

        Args:
            schema: Structured output schema.

        Returns:
            Invoker that raises a validation error.
        """

        return _FailingStructuredInvoker(schema)


class _FallbackJsonClinicalModel:
    """Chat-model stub that succeeds only through the manual JSON fallback path."""

    def invoke(self, messages: list[Any]) -> AIMessage:
        """Return valid JSON for clinical analysis.

        Args:
            messages: LangChain-compatible message list.

        Returns:
            Plain AI message containing a JSON object.
        """

        return AIMessage(
            content=(
                '{"status":"analysis_ready","primary_hypothesis":"Migraine",'
                '"differential_hypotheses":["Tension headache"],'
                '"recommended_exams":["Neurologic examination"],'
                '"reasoning_summary":"Symptoms fit a primary headache pattern.",'
                '"safety_notes":[],'
                '"user_response":"This looks compatible with a primary headache syndrome."}'
            ),
        )

    def with_structured_output(
        self,
        schema: type[BaseModel],
    ) -> StructuredOutputInvoker[BaseModel]:
        """Return a failing native structured-output invoker.

        Args:
            schema: Structured output schema.

        Returns:
            Invoker that raises a validation error.
        """

        return _FailingStructuredInvoker(schema)


def test_router_accepts_label_style_fallback_after_json_failure() -> None:
    """Ensure routing still works when a provider returns label-style text."""

    router_node = build_router_node(_FallbackLabelControlModel())

    command = router_node(
        {
            "messages": [HumanMessage(content="Tell me a joke")],
            "audit_events": [],
        },
    )

    assert command.goto == "invalid_request"
    assert command.update["router_intent"] == "invalid_request"
    assert "outside the assistant clinical scope" in str(command.update["router_rationale"]).lower()


def test_clinical_backend_retries_with_manual_json_when_native_mode_fails() -> None:
    """Ensure clinical analysis survives providers without native structured output."""

    backend = OpenAICompatibleClinicalBackend(_FallbackJsonClinicalModel())

    result = backend.analyze(user_message="The patient reports headache", active_patient=None)

    assert result.status == "analysis_ready"
    assert result.primary_hypothesis == "Migraine"
    assert result.recommended_exams == ["Neurologic examination"]