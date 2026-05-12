"""Factories for creating LangChain models and specialist invokers."""

from __future__ import annotations

from typing import Any, cast

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from screening_agent.config import AppSettings
from screening_agent.model.control_models import ChatModelControlAdapter, ControlModel
from screening_agent.model.mock_runtime import MockControlModel
from screening_agent.tools.specialist_tool import (
    SpecialistInvoker,
    create_gguf_specialist_invoker,
    create_mock_specialist_invoker,
    create_remote_specialist_invoker,
)


def create_control_model(settings: AppSettings) -> ControlModel:
    """Create the control model used by router and tool-calling nodes.

    Args:
        settings: Application settings loaded from environment variables.

    Returns:
        A configured chat model instance.
    """

    if settings.control_model.backend == "mock":
        return MockControlModel()

    return ChatModelControlAdapter(
        _create_remote_chat_model(
            backend=settings.control_model.backend,
            model_name=settings.control_model.model,
            base_url=settings.control_model.base_url,
            api_key=settings.control_model.api_key,
            temperature=settings.control_model.temperature,
        ),
        backend=settings.control_model.backend,
    )


def create_specialist_invoker(settings: AppSettings) -> SpecialistInvoker:
    """Create the structured symptom specialist invoker.

    Args:
        settings: Application settings loaded from environment variables.

    Returns:
        A specialist invoker function that accepts (clinical_request, clinical_context)
        and returns a validated structured output.

    Raises:
        ValueError: If required configuration is missing.
    """

    if settings.clinical_backend.backend == "mock":
        return create_mock_specialist_invoker()

    if settings.clinical_backend.backend == "gguf":
        if settings.clinical_backend.gguf_model_path is None:
            raise ValueError("SCREENING_AGENT_GGUF_MODEL_PATH is required for GGUF mode.")
        return create_gguf_specialist_invoker(
            settings.clinical_backend.gguf_model_path,
            temperature=settings.clinical_backend.temperature,
        )

    model_name = settings.clinical_backend.model or settings.control_model.model
    base_url = settings.clinical_backend.base_url or settings.control_model.base_url
    api_key = settings.clinical_backend.api_key or settings.control_model.api_key
    remote_model = ChatModelControlAdapter(
        _create_remote_chat_model(
            backend=settings.clinical_backend.backend,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=settings.clinical_backend.temperature,
        ),
        backend=settings.clinical_backend.backend,
    )
    return create_remote_specialist_invoker(
        remote_model,
    )


def _create_remote_chat_model(
    *,
    backend: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
) -> BaseChatModel:
    """Create a remote chat model for the selected provider backend.

    Args:
        backend: Provider backend kind.
        model_name: Provider-specific model identifier.
        base_url: Optional base URL for OpenAI-compatible endpoints.
        api_key: Optional provider API key.
        temperature: Sampling temperature.

    Returns:
        Configured LangChain chat model.

    Raises:
        ValueError: If required configuration for the chosen backend is missing.
    """

    if backend == "openai":
        return ChatOpenAI(
            model=model_name,
            api_key=_to_secret_str(api_key),
            temperature=temperature,
        )

    if backend == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "langchain-openrouter is required when SCREENING_AGENT_*_BACKEND is set to 'openrouter'.",
            ) from exc

        return ChatOpenRouter(
            model=model_name,
            api_key=cast(Any, _to_secret_str(api_key)),
            temperature=temperature,
        )

    if backend == "openai_compatible":
        if not base_url:
            raise ValueError(
                "SCREENING_AGENT_*_BASE_URL is required when the backend is 'openai_compatible'.",
            )
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=_to_secret_str(api_key or "local-openai-compatible"),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported remote backend: {backend!r}")


def _to_secret_str(value: str | None) -> SecretStr | None:
    """Convert a plain API key into the secret-string type expected by providers.

    Args:
        value: Raw API key string.

    Returns:
        Secret string wrapper, or `None` when no key is configured.
    """

    if value is None:
        return None
    return SecretStr(value)
