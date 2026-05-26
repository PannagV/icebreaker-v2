import json

import pytest

from icebreaker.config import WebSearchConfig
from icebreaker.tools.web_search import WebSearchError, WebSearchTool


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


def test_web_search_uses_mcp_sse_handshake(monkeypatch) -> None:
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
                            "result": {"tools": [{"name": "search_web"}]},
                        },
                    },
                    {
                        "event": "message",
                        "data": {
                            "jsonrpc": "2.0",
                            "id": 3,
                            "result": {"content": "ok"},
                        },
                    },
                ]
            )
        return DummyResponse()

    monkeypatch.setenv("WEB_SEARCH_API_KEY", "secret")
    tool = WebSearchTool(
        WebSearchConfig(
            enabled=True,
            base_url="http://192.168.1.50:8003/sse",
            timeout_seconds=12,
            api_key_env="WEB_SEARCH_API_KEY",
            max_results=7,
            tool_name="search_web",
            prompt_enabled=False,
        ),
        urlopen=fake_urlopen,
    )

    response = tool.search(query="latest ai", categories="news,science", page=2)

    assert captured[0] == ("http://192.168.1.50:8003/sse", None, "GET", 12)
    assert captured[1][0] == "http://192.168.1.50:8003/messages"
    assert captured[1][2] == "POST"
    assert captured[1][1]["method"] == "initialize"
    assert captured[2][1]["method"] == "notifications/initialized"
    assert captured[3][1]["method"] == "tools/list"
    assert captured[4][1]["method"] == "tools/call"
    assert captured[4][1]["params"]["arguments"] == {
        "query": "latest ai",
        "categories": "news,science",
        "page": 2,
        "max_results": 7,
    }
    assert response["content"] == "ok"


def test_web_search_prompt_fetch(monkeypatch) -> None:
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
                            "id": 4,
                            "result": {"prompts": [{"name": "web-search"}]},
                        },
                    },
                    {
                        "event": "message",
                        "data": {
                            "jsonrpc": "2.0",
                            "id": 5,
                            "result": {"text": "Use web search responsibly."},
                        },
                    },
                ]
            )
        return DummyResponse()

    monkeypatch.setenv("WEB_SEARCH_API_KEY", "secret")
    tool = WebSearchTool(
        WebSearchConfig(
            enabled=True,
            base_url="http://192.168.1.50:8003/sse",
            timeout_seconds=12,
            api_key_env="WEB_SEARCH_API_KEY",
            max_results=10,
            tool_name="search_web",
            prompt_enabled=True,
        ),
        urlopen=fake_urlopen,
    )

    prompt = tool.prompt()

    assert prompt is not None
    assert prompt.name == "web-search"
    assert prompt.text == "Use web search responsibly."
    assert captured[0] == ("http://192.168.1.50:8003/sse", None, "GET", 12)
    assert captured[1][1]["method"] == "initialize"
    assert captured[2][1]["method"] == "notifications/initialized"
    assert captured[3][1]["method"] == "prompts/list"
    assert captured[4][1]["method"] == "prompts/get"


def test_web_search_missing_tool_raises() -> None:
    def fake_urlopen(http_request, timeout):
        del timeout
        if http_request.get_method() == "GET":
            return DummySSEStream(
                [
                    {"event": "endpoint", "data": "/messages"},
                    {"event": "message", "data": {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}},
                    {"event": "message", "data": {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}},
                ]
            )
        return DummyResponse()

    tool = WebSearchTool(WebSearchConfig(enabled=True), urlopen=fake_urlopen)

    with pytest.raises(WebSearchError, match="not available"):
        tool.search(query="latest ai")


def _decode_body(http_request) -> object | None:
    if http_request.data is None:
        return None
    return json.loads(http_request.data.decode("utf-8"))
