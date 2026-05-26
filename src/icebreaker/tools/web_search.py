from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request
from urllib.parse import urljoin

from icebreaker.config import WebSearchConfig
from icebreaker.llm.base import ToolDefinition


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchPrompt:
    name: str
    text: str


class WebSearchTool:
    def __init__(self, config: WebSearchConfig, urlopen=request.urlopen) -> None:
        self.config = config
        self._urlopen = urlopen
        self._cached_prompt: WebSearchPrompt | None = None

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.config.tool_name,
            description="Search the web for recent information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "categories": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        )

    def search(
        self,
        *,
        query: str,
        categories: str | None = None,
        page: int = 1,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            raise WebSearchError("Web search is disabled.")
        stream = self._open_sse()
        endpoint = self._read_endpoint(stream)
        self._initialize(stream, endpoint)
        self._notify_initialized(endpoint)
        tools = self._list_tools(stream, endpoint)
        if self.config.tool_name not in {tool.get("name") for tool in tools}:
            raise WebSearchError("Web search tool is not available in the MCP server response.")
        result = self._call_tool(
            stream,
            endpoint,
            query=query,
            categories=categories,
            page=page,
            max_results=max_results,
        )
        payload = result.get("result")
        if isinstance(payload, dict):
            return payload
        return {"result": payload}

    def prompt(self) -> WebSearchPrompt | None:
        if not self.config.prompt_enabled:
            return None
        if self._cached_prompt:
            return self._cached_prompt
        stream = self._open_sse()
        endpoint = self._read_endpoint(stream)
        self._initialize(stream, endpoint)
        self._notify_initialized(endpoint)
        prompt = self._read_prompt(stream, endpoint)
        self._cached_prompt = prompt
        return prompt

    def _open_sse(self):
        headers = {"Accept": "text/event-stream"}
        if self.config.api_key_env:
            api_key = self._get_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        http_request = request.Request(self.config.base_url, headers=headers, method="GET")
        return self._urlopen(http_request, timeout=self.config.timeout_seconds)

    def _read_endpoint(self, stream) -> str:
        while True:
            event = _read_next_event(stream)
            if event is None:
                break
            if event.get("event") == "endpoint":
                endpoint = str(event.get("data", ""))
                if endpoint:
                    return endpoint
        raise WebSearchError("Unable to read MCP endpoint from SSE stream.")

    def _initialize(self, stream, endpoint: str) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "icebreaker", "version": "0.1.0"},
            },
        }
        self._post(endpoint, payload)
        _read_until_id(stream, 1)

    def _notify_initialized(self, endpoint: str) -> None:
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        self._post(endpoint, payload)

    def _list_tools(self, stream, endpoint: str) -> list[dict[str, Any]]:
        payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        self._post(endpoint, payload)
        response = _read_until_id(stream, 2)
        tools = response.get("result", {}).get("tools")
        if not isinstance(tools, list):
            raise WebSearchError("Web search tools list response is malformed.")
        return tools

    def _read_prompt(self, stream, endpoint: str) -> WebSearchPrompt:
        payload = {"jsonrpc": "2.0", "id": 4, "method": "prompts/list", "params": {}}
        self._post(endpoint, payload)
        response = _read_until_id(stream, 4)
        prompts = response.get("result", {}).get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise WebSearchError("Web search prompt list response is malformed.")
        name = str(prompts[0].get("name", "")).strip()
        if not name:
            raise WebSearchError("Web search prompt list did not return a name.")
        payload = {"jsonrpc": "2.0", "id": 5, "method": "prompts/get", "params": {"name": name}}
        self._post(endpoint, payload)
        response = _read_until_id(stream, 5)
        prompt_payload = response.get("result")
        text = _extract_prompt_text(prompt_payload)
        if not text:
            raise WebSearchError("Web search prompt payload is malformed.")
        return WebSearchPrompt(name=name, text=text)

    def _call_tool(
        self,
        stream,
        endpoint: str,
        *,
        query: str,
        categories: str | None,
        page: int,
        max_results: int | None,
    ) -> dict[str, Any]:
        max_results_value = max_results if max_results is not None else self.config.max_results
        max_results_value = min(max(int(max_results_value), 1), 20)
        arguments: dict[str, object] = {
            "query": query,
            "page": max(int(page), 1),
            "max_results": max_results_value,
        }
        if isinstance(categories, str) and categories.strip():
            arguments["categories"] = categories
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": self.config.tool_name, "arguments": arguments},
        }
        self._post(endpoint, payload)
        return _read_until_id(stream, 3)

    def _post(self, endpoint: str, payload: dict[str, object]) -> None:
        url = urljoin(self.config.base_url, endpoint)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = self._get_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        http_request = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._urlopen(http_request, timeout=self.config.timeout_seconds):
            return None

    def _get_api_key(self) -> str | None:
        if not self.config.api_key_env:
            return None
        return str(os.getenv(self.config.api_key_env) or "") or None


def _read_until_id(stream, message_id: int) -> dict[str, Any]:
    while True:
        event = _read_next_event(stream)
        if event is None:
            break
        if event.get("event") != "message":
            continue
        payload = event.get("data")
        if isinstance(payload, dict) and payload.get("id") == message_id:
            return payload
    raise WebSearchError("Web search SSE stream did not return expected response.")


def _read_next_event(stream) -> dict[str, object] | None:
    current: dict[str, object] = {}
    while True:
        line = stream.readline()
        if not line:
            return current or None
        text = line.decode("utf-8").rstrip("\r\n")
        if not text:
            if current:
                return current
            continue
        if text.startswith("event:"):
            current["event"] = text[len("event:") :].strip()
        if text.startswith("data:"):
            data = text[len("data:") :].strip()
            current["data"] = _parse_data(data)
    return None


def _parse_data(raw: str) -> object:
    if not raw:
        return ""
    if raw.startswith("{") or raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _extract_prompt_text(payload: object) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        for key in ("text", "prompt", "template", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        messages = payload.get("messages")
        if isinstance(messages, list):
            parts: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
            if parts:
                return "\n".join(parts)
    return None
