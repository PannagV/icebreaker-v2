import json

import pytest

from icebreaker.config import KnowledgeConfig
from icebreaker.tools.local_knowledge import LocalKnowledgeError, LocalKnowledgeTool


class DummyResponse:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload or {}).encode("utf-8")


class DummySSEStream:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.lines = []
        for event in events:
            if "event" in event:
                self.lines.append(f"event: {event['event']}\n".encode("utf-8"))
            if "data" in event:
                data = event["data"]
                rendered = data if isinstance(data, str) else json.dumps(data)
                self.lines.append(f"data: {rendered}\n".encode("utf-8"))
            self.lines.append(b"\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        return b""


def test_local_knowledge_uses_mcp_sse_handshake(monkeypatch) -> None:
    captured: list[tuple[str, object | None, str, int]] = []

    def fake_urlopen(http_request, timeout):
        captured.append((http_request.full_url, _decode_body(http_request), http_request.get_method(), timeout))
        if http_request.get_method() == "GET":
            return DummySSEStream(
                [
                    {"event": "endpoint", "data": "/messages"},
                    {"event": "message", "data": {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}},
                    {
                        "event": "message",
                        "data": {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "result": {"tools": [{"name": "search_local_knowledge"}]},
                        },
                    },
                    {
                        "event": "message",
                        "data": {
                            "jsonrpc": "2.0",
                            "id": 3,
                            "result": {
                                "structuredContent": {
                                    "results": [
                                        {
                                            "title": "Guide",
                                            "snippet": "Use MFA.",
                                            "source": "auth-guide",
                                            "score": 0.95,
                                            "category": "authentication",
                                        }
                                    ]
                                }
                            },
                        },
                    },
                ]
            )
        return DummyResponse()

    monkeypatch.setenv("KNOWLEDGE_API_KEY", "secret")
    tool = LocalKnowledgeTool(
        KnowledgeConfig(
            enabled=True,
            base_url="http://192.168.1.50:8002/sse",
            timeout_seconds=12,
            api_key_env="KNOWLEDGE_API_KEY",
            max_results=4,
            tool_name="search_local_knowledge",
        ),
        urlopen=fake_urlopen,
    )

    response = tool.search(query="authentication best practices", category="authentication")

    assert captured[0] == ("http://192.168.1.50:8002/sse", None, "GET", 12)
    assert captured[1][0] == "http://192.168.1.50:8002/messages"
    assert captured[1][2] == "POST"
    assert captured[1][1]["method"] == "initialize"
    assert captured[2][1]["method"] == "notifications/initialized"
    assert captured[3][1]["method"] == "tools/list"
    assert captured[4][1]["method"] == "tools/call"
    assert captured[4][1]["params"]["arguments"] == {
        "query": "authentication best practices",
        "category": "authentication",
        "max_results": 4,
    }
    assert response.results[0].source == "auth-guide"


def test_local_knowledge_handles_malformed_tool_result() -> None:
    def fake_urlopen(http_request, timeout):
        del timeout
        if http_request.get_method() == "GET":
            return DummySSEStream(
                [
                    {"event": "endpoint", "data": "/messages"},
                    {"event": "message", "data": {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}},
                    {
                        "event": "message",
                        "data": {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "search_local_knowledge"}]}},
                    },
                    {"event": "message", "data": {"jsonrpc": "2.0", "id": 3, "result": {"content": []}}},
                ]
            )
        return DummyResponse()

    tool = LocalKnowledgeTool(KnowledgeConfig(enabled=True), urlopen=fake_urlopen)

    with pytest.raises(LocalKnowledgeError, match="malformed"):
        tool.search(query="latest cybersecurity threats", category=None)


def _decode_body(http_request) -> object | None:
    if http_request.data is None:
        return None
    return json.loads(http_request.data.decode("utf-8"))
