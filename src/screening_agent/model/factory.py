"""Factories for creating LangChain models and clinical backends."""

from __future__ import annotations

from langchain.chat_models import init_chat_model

from screening_agent.config import AppSettings
from screening_agent.model.clinical_backend import ClinicalBackend
from screening_agent.model.control_models import ControlModel
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

    return init_chat_model(
        model=settings.control_model.model,
        model_provider="openai",
        base_url=settings.control_model.base_url,
        api_key=settings.control_model.api_key or "local-openai-compatible",
        temperature=settings.control_model.temperature,
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
    api_key = settings.clinical_backend.api_key or settings.control_model.api_key or "local-openai-compatible"
    remote_model = init_chat_model(
        model=model_name,
        model_provider="openai",
        base_url=base_url,
        api_key=api_key,
        temperature=settings.clinical_backend.temperature,
    )
    return OpenAICompatibleClinicalBackend(remote_model)
