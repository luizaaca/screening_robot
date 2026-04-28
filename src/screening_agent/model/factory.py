"""Factories for creating LangChain models and clinical backends."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from screening_agent.config import AppSettings
from screening_agent.model.clinical_backend import ClinicalBackend
from screening_agent.model.control_models import ChatModelControlAdapter, ControlModel
from screening_agent.model.gguf_runtime import GGUFClinicalBackend
from screening_agent.model.mock_runtime import MockClinicalBackend, MockControlModel
from screening_agent.model.openai_compatible_runtime import OpenAICompatibleClinicalBackend



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



def create_clinical_backend(settings: AppSettings) -> ClinicalBackend:
    """Create the clinical analysis backend configured for the project.

    Args:
        settings: Application settings loaded from environment variables.

    Returns:
        A clinical backend implementation.

    Raises:
        ValueError: If required configuration is missing.
    """

    if settings.clinical_backend.backend == "mock":
        return MockClinicalBackend()

    if settings.clinical_backend.backend == "gguf":
        if settings.clinical_backend.gguf_model_path is None:
            raise ValueError("SCREENING_AGENT_GGUF_MODEL_PATH is required for GGUF mode.")
        return GGUFClinicalBackend(settings.clinical_backend.gguf_model_path)

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
    return OpenAICompatibleClinicalBackend(
        remote_model,
        backend_name=settings.clinical_backend.backend,
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
            api_key=api_key,
            temperature=temperature,
        )

    if backend == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:  # pragma: no cover - depends on local environment setup.
            raise ImportError(
                "langchain-openrouter is required when SCREENING_AGENT_*_BACKEND is set to 'openrouter'.",
            ) from exc

        return ChatOpenRouter(
            model=model_name,
            api_key=api_key,
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
            api_key=api_key or "local-openai-compatible",
            temperature=temperature,
        )

    raise ValueError(f"Unsupported remote backend: {backend!r}")
