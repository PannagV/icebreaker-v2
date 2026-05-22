from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request
from urllib.parse import urljoin

from icebreaker.config import KnowledgeConfig
from icebreaker.llm.base import ToolDefinition


class LocalKnowledgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeResult:
    title: str
    snippet: str
    source: str
    score: float
    category: str

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
            "category": self.category,
        }


@dataclass(frozen=True)
class KnowledgeSearchResponse:
    query: str
    category: str | None
    results: list[KnowledgeResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "category": self.category,
            "results": [result.to_dict() for result in self.results],
        }


class LocalKnowledgeTool:
    def __init__(self, config: KnowledgeConfig, urlopen=request.urlopen) -> None:
        self.config = config
        self._urlopen = urlopen

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.config.tool_name,
            description="Search the local knowledge base for relevant context.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": ["string", "null"]},
                },
                "required": ["query"],
            },
        )

    def search(self, *, query: str, category: str | None = None) -> KnowledgeSearchResponse:
        if not self.config.enabled:
            raise LocalKnowledgeError("Local knowledge search is disabled.")
        stream = self._open_sse()
        endpoint = self._read_endpoint(stream)
        self._initialize(stream, endpoint)
        self._notify_initialized(endpoint)
        tools = self._list_tools(stream, endpoint)
        if self.config.tool_name not in {tool["name"] for tool in tools}:
            raise LocalKnowledgeError("Knowledge tool is not available in the MCP server response.")
        result = self._call_tool(stream, endpoint, query=query, category=category)
        return self._parse_result(query, category, result)

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
        raise LocalKnowledgeError("Unable to read MCP endpoint from SSE stream.")

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
            raise LocalKnowledgeError("Knowledge tools list response is malformed.")
        return tools

    def _call_tool(self, stream, endpoint: str, *, query: str, category: str | None) -> dict[str, Any]:
        arguments: dict[str, object] = {"query": query}
        if isinstance(category, str) and category.strip():
            arguments["category"] = category
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": self.config.tool_name, "arguments": arguments},
        }
        self._post(endpoint, payload)
        return _read_until_id(stream, 3)

    def _parse_result(
        self, query: str, category: str | None, result: dict[str, Any]
    ) -> KnowledgeSearchResponse:
        payload = result.get("result", {})
        content = payload.get("structuredContent")
        if isinstance(content, dict):
            raw_results = content.get("results")
            if isinstance(raw_results, dict):
                raw_results = [raw_results]
            if isinstance(raw_results, list):
                parsed = [
                    KnowledgeResult(
                        title=str(item.get("title", "")),
                        snippet=str(item.get("snippet", "")),
                        source=str(item.get("source", "")),
                        score=float(item.get("score", 0.0)),
                        category=str(item.get("category", "")),
                    )
                    for item in raw_results
                ]
                return KnowledgeSearchResponse(query=query, category=category, results=parsed)

        fallback = _extract_content_text(payload.get("content"))
        if fallback:
            return KnowledgeSearchResponse(
                query=query,
                category=category,
                results=[
                    KnowledgeResult(
                        title="Knowledge response",
                        snippet=fallback,
                        source="mcp",
                        score=0.0,
                        category=str(category or ""),
                    )
                ],
            )

        raise LocalKnowledgeError("Knowledge tool result malformed.")

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
    raise LocalKnowledgeError("Knowledge SSE stream did not return expected response.")


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


def _extract_content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text.strip() or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
            if item.get("type") == "json" and isinstance(item.get("json"), dict):
                json_text = json.dumps(item["json"], ensure_ascii=True)
                parts.append(json_text)
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return None
