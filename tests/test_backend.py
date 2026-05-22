import asyncio
import json
from pathlib import Path

from icebreaker.config import BackendConfig, Config, KnowledgeConfig, render_config
from icebreaker.llm.base import ToolDefinition
from icebreaker.llm.factory import build_backend, resolve_backend_config
from icebreaker.llm.openai_compatible import OpenAICompatibleModel


class DummyResponse:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or {"choices": [{"message": {"content": "ok"}}]}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_adapter_sends_expected_request() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(http_request, timeout):
        captured["url"] = http_request.full_url
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        captured["auth"] = http_request.get_header("Authorization")
        captured["timeout"] = timeout
        return DummyResponse()

    backend = OpenAICompatibleModel(
        BackendConfig(
            name="openai",
            type="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_env=None,
        ),
        urlopen=fake_urlopen,
    )

    response = asyncio.run(
        backend.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.4,
            timeout_seconds=45,
        )
    )

    assert response.text == "ok"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["body"] == {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.4,
    }
    assert captured["auth"] is None
    assert captured["timeout"] == 45


def test_auth_header_only_present_when_configured(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(http_request, timeout):
        captured["auth"] = http_request.get_header("Authorization")
        return DummyResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    backend = OpenAICompatibleModel(
        BackendConfig(
            name="openai",
            type="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        ),
        urlopen=fake_urlopen,
    )

    asyncio.run(
        backend.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.2,
            timeout_seconds=30,
        )
    )

    assert captured["auth"] == "Bearer secret"


def test_openai_compatible_adapter_includes_tools_when_knowledge_enabled() -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(http_request, timeout):
        captured["body"] = json.loads(http_request.data.decode("utf-8"))
        return DummyResponse()

    backend = OpenAICompatibleModel(
        BackendConfig(
            name="openai",
            type="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_env=None,
        ),
        knowledge=KnowledgeConfig(enabled=True),
        urlopen=fake_urlopen,
    )

    asyncio.run(
        backend.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.4,
            timeout_seconds=45,
            tools=[
                ToolDefinition(
                    name="search_local_knowledge",
                    description="Search LAN knowledge.",
                    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                )
            ],
        )
    )

    assert captured["body"]["tools"][0]["function"]["name"] == "search_local_knowledge"


def test_openai_compatible_adapter_parses_tool_calls() -> None:
    def fake_urlopen(_http_request, timeout):
        del timeout
        return DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "search_local_knowledge",
                                        "arguments": json.dumps({"query": "threats", "category": None}),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )

    backend = OpenAICompatibleModel(
        BackendConfig(
            name="openai",
            type="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://api.openai.com/v1",
            api_key_env=None,
        ),
        urlopen=fake_urlopen,
    )

    response = asyncio.run(
        backend.complete(
            [{"role": "user", "content": "hello"}],
            temperature=0.4,
            timeout_seconds=45,
        )
    )

    assert response.tool_call is not None
    assert response.tool_call.name == "search_local_knowledge"
    assert response.tool_call.arguments == {"query": "threats", "category": None}
    assert response.tool_call.call_id == "call_1"


def test_backend_selection_resolves_default_and_explicit_names(tmp_path: Path) -> None:
    config_path = tmp_path / "icebreaker.toml"
    config_path.write_text(
        render_config(
            default_backend="local",
            backends={
                "local": {
                    "type": "openai_compatible",
                    "model": "local-model",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "api_key_env": "",
                },
                "openai": {
                    "type": "openai_compatible",
                    "model": "gpt-4.1-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                },
            },
        ),
        encoding="utf-8",
    )
    config = Config.load(config_path)

    assert resolve_backend_config(config).name == "local"
    assert resolve_backend_config(config, "openai").name == "openai"
    assert isinstance(build_backend(config), OpenAICompatibleModel)
