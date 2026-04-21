"""Graph builders and state definitions for the screening assistant."""

from __future__ import annotations

from typing import Any

from .state import AssistantState

__all__ = ["AssistantState", "build_default_graph", "build_screening_graph"]


def build_default_graph(*args: Any, **kwargs: Any) -> Any:
    """Proxy import for the default graph builder.

    Args:
        *args: Positional arguments forwarded to the real builder.
        **kwargs: Keyword arguments forwarded to the real builder.

    Returns:
        The compiled LangGraph workflow.
    """

    from .builder import build_default_graph as _build_default_graph

    return _build_default_graph(*args, **kwargs)


def build_screening_graph(*args: Any, **kwargs: Any) -> Any:
    """Proxy import for the main graph builder.

    Args:
        *args: Positional arguments forwarded to the real builder.
        **kwargs: Keyword arguments forwarded to the real builder.

    Returns:
        The compiled LangGraph workflow.
    """

    from .builder import build_screening_graph as _build_screening_graph

    return _build_screening_graph(*args, **kwargs)
