from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class ModelResponse:
    text: str | None = None
    tool_call: ToolCall | None = None


class ChatModel(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, object]],
        *,
        temperature: float,
        timeout_seconds: int,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        raise NotImplementedError
