from pathlib import Path

import pytest

from icebreaker.config import Config, render_config, write_default_config


def test_loads_named_backends_and_default(tmp_path: Path) -> None:
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
            temperature=0.3,
            timeout_seconds=90,
            knowledge_enabled=True,
            knowledge_base_url="http://192.168.1.50:8002/sse",
            knowledge_timeout_seconds=7,
            knowledge_api_key_env="KNOWLEDGE_API_KEY",
            knowledge_max_results=8,
            knowledge_tool_name="search_local_knowledge",
            web_search_enabled=True,
            web_search_base_url="http://192.168.1.50:8003/sse",
            web_search_timeout_seconds=9,
            web_search_api_key_env="WEB_SEARCH_API_KEY",
            web_search_max_results=12,
            web_search_tool_name="search_web",
            web_search_prompt_enabled=True,
        ),
        encoding="utf-8",
    )

    config = Config.load(config_path)

    assert config.default_backend == "local"
    assert config.chat.temperature == 0.3
    assert config.chat.timeout_seconds == 90
    assert config.knowledge.enabled is True
    assert config.knowledge.base_url == "http://192.168.1.50:8002/sse"
    assert config.knowledge.timeout_seconds == 7
    assert config.knowledge.api_key_env == "KNOWLEDGE_API_KEY"
    assert config.knowledge.max_results == 8
    assert config.knowledge.tool_name == "search_local_knowledge"
    assert config.web_search.enabled is True
    assert config.web_search.base_url == "http://192.168.1.50:8003/sse"
    assert config.web_search.timeout_seconds == 9
    assert config.web_search.api_key_env == "WEB_SEARCH_API_KEY"
    assert config.web_search.max_results == 12
    assert config.web_search.tool_name == "search_web"
    assert config.web_search.prompt_enabled is True
    assert config.resolve_backend().name == "local"
    assert config.resolve_backend("openai").model == "gpt-4.1-mini"


def test_rejects_missing_default_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "icebreaker.toml"
    config_path.write_text(
        """
[chat]
temperature = 0.2
timeout_seconds = 120

[storage]
session_dir = ".icebreaker/sessions"

[backends.local]
type = "openai_compatible"
model = "local-model"
base_url = "http://127.0.0.1:1234/v1"
api_key_env = ""
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_backend"):
        Config.load(config_path)


def test_preserves_env_var_secret_references(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "icebreaker.toml"
    write_default_config(config_path)
    config_path.write_text(
        render_config(
            default_backend="openai",
            backends={
                "openai": {
                    "type": "openai_compatible",
                    "model": "gpt-4.1-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "ICEBREAKER_API_KEY",
                }
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ICEBREAKER_API_KEY", "secret")

    config = Config.load(config_path)

    assert config.resolve_backend().api_key_env == "ICEBREAKER_API_KEY"
    assert config.resolve_backend().api_key == "secret"


def test_rejects_invalid_knowledge_timeout(tmp_path: Path) -> None:
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
                }
            },
            knowledge_enabled=True,
            knowledge_timeout_seconds=0,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Knowledge timeout"):
        Config.load(config_path)


def test_rejects_invalid_web_search_timeout(tmp_path: Path) -> None:
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
                }
            },
            web_search_enabled=True,
            web_search_timeout_seconds=0,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Web search timeout"):
        Config.load(config_path)
