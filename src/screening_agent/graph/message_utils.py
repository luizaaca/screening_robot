"""Utilities for extracting text from LangChain message objects."""

from __future__ import annotations

from collections.abc import Sequence

from langchain.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage



def get_last_ai_message_text(messages: Sequence[BaseMessage]) -> str:
    """Return the textual content of the latest AI message.

    Args:
        messages: Conversation message history.

    Returns:
        The latest AI message content as plain text.

    Raises:
        ValueError: If no AI message exists.
    """

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return coerce_message_text(message.content)
    raise ValueError("No AI message is available in the state.")



def get_last_human_message_text(messages: Sequence[BaseMessage]) -> str:
    """Return the textual content of the latest human message.

    Args:
        messages: Conversation message history.

    Returns:
        The latest human message content as plain text.

    Raises:
        ValueError: If no human message exists.
    """

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return coerce_message_text(message.content)
    raise ValueError("No human message is available in the state.")



def coerce_message_text(content: object) -> str:
    """Normalize LangChain message content into a plain string.

    Args:
        content: Message content that may be string or content blocks.

    Returns:
        String representation of the message content.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[str] = []
        for block in content:
            if isinstance(block, str):
                blocks.append(block)
                continue
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append(str(block.get("text", "")))
        return "\n".join(blocks).strip()
    return str(content)
