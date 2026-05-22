from __future__ import annotations

import json
from collections.abc import Callable
from urllib import request

from icebreaker.config import BackendConfig, KnowledgeConfig
from icebreaker.llm.base import ChatModel, ModelResponse, ToolCall, ToolDefinition


UrlOpen = Callable[..., object]


class OpenAICompatibleModel(ChatModel):
    """Adapter for OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        config: BackendConfig,
        knowledge: KnowledgeConfig | None = None,
        urlopen: UrlOpen = request.urlopen,
    ) -> None:
        self.config = config
        self.knowledge = knowledge
        self._urlopen = urlopen

    async def complete(
        self,
        messages: list[dict[str, object]],
        *,
        temperature: float,
        timeout_seconds: int,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        http_request = self._build_request(messages, temperature=temperature, tools=tools)
        with self._urlopen(http_request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tool_call = tool_calls[0]
            arguments = json.loads(tool_call["function"]["arguments"])
            return ModelResponse(
                tool_call=ToolCall(
                    name=str(tool_call["function"]["name"]),
                    arguments=arguments,
                    call_id=str(tool_call["id"]),
                )
            )
        return ModelResponse(text=str(message.get("content") or ""))

    def _build_request(
        self,
        messages: list[dict[str, object]],
        *,
        temperature: float,
        tools: list[ToolDefinition] | None,
    ) -> request.Request:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
