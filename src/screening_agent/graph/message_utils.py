"""Utilities for extracting text from LangChain message objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage

MessageT = TypeVar("MessageT", bound=BaseMessage)


def get_message_text(message: BaseMessage) -> str:
    """Return plain text from a LangChain message.

    Args:
        message: LangChain message instance.

    Returns:
        Plain-text message content.
    """

    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if text_attr is not None and not callable(text_attr):
        return str(text_attr)

    content = message.content
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


def get_last_message_of_type(
    messages: Sequence[BaseMessage],
    message_type: type[MessageT],
) -> MessageT:
    """Return the latest message matching the requested LangChain type.

    Args:
        messages: Conversation message history.
        message_type: Desired LangChain message subclass.

    Returns:
        The latest matching message instance.

    Raises:
        ValueError: If no matching message exists.
    """

    for message in reversed(messages):
        if isinstance(message, message_type):
            return message
    raise ValueError(f"No {message_type.__name__} is available in the state.")


def get_last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage:
    """Return the latest AI message from the conversation state.

    Args:
        messages: Conversation message history.

    Returns:
        Latest AI message.
    """

    return get_last_message_of_type(messages, AIMessage)


def get_last_ai_message_text(messages: Sequence[BaseMessage]) -> str:
    """Return the textual content of the latest AI message.

    Args:
        messages: Conversation message history.

    Returns:
        The latest AI message content as plain text.
    """

    return get_message_text(get_last_ai_message(messages))


def get_last_human_message(messages: Sequence[BaseMessage]) -> HumanMessage:
    """Return the latest human message from the conversation state.

    Args:
        messages: Conversation message history.

    Returns:
        Latest human message.
    """

    return get_last_message_of_type(messages, HumanMessage)


def get_last_human_message_text(messages: Sequence[BaseMessage]) -> str:
    """Return the textual content of the latest human message.

    Args:
        messages: Conversation message history.

    Returns:
        The latest human message content as plain text.
    """

    return get_message_text(get_last_human_message(messages))


def get_last_tool_message(messages: Sequence[BaseMessage]) -> ToolMessage:
    """Return the latest tool message from the conversation state.

    Args:
        messages: Conversation message history.

    Returns:
        Latest tool message.
    """

    return get_last_message_of_type(messages, ToolMessage)
