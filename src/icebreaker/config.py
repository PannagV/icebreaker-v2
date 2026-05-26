from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BACKEND_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

DEFAULT_COMMAND_BLOCK_PATTERNS = [
    "rm -rf",
    "rm -fr",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "mkfs",
    "dd ",
]


DEFAULT_CONFIG = """\
default_backend = "local"

[chat]
temperature = 0.2
timeout_seconds = 120

[storage]
session_dir = ".icebreaker/sessions"

[knowledge]
enabled = false
base_url = "http://127.0.0.1:8002/sse"
timeout_seconds = 12
api_key_env = ""
max_results = 6
tool_name = "search_local_knowledge"

[web_search]
enabled = false
base_url = "http://127.0.0.1:8003/sse"
timeout_seconds = 12
api_key_env = ""
max_results = 10
tool_name = "search_web"
prompt_enabled = false

[command]
enabled = true
timeout_seconds = 30
max_output_chars = 40000
block_patterns = ["rm -rf", "rm -fr", "shutdown", "reboot", "poweroff", "halt", "mkfs", "dd "]
tool_name = "run_command"

[backends.local]
type = "openai_compatible"
model = "local-model"
base_url = "http://127.0.0.1:1234/v1"
api_key_env = ""
"""


@dataclass(frozen=True)
class ChatConfig:
    temperature: float = 0.2
    timeout_seconds: int = 120


@dataclass(frozen=True)
class StorageConfig:
    session_dir: Path


@dataclass(frozen=True)
class KnowledgeConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8002/sse"
    timeout_seconds: int = 12
    api_key_env: str | None = None
    max_results: int = 6
    tool_name: str = "search_local_knowledge"


@dataclass(frozen=True)
class WebSearchConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8003/sse"
    timeout_seconds: int = 12
    api_key_env: str | None = None
    max_results: int = 10
    tool_name: str = "search_web"
    prompt_enabled: bool = False


@dataclass(frozen=True)
class CommandConfig:
    enabled: bool = True
    timeout_seconds: int = 30
    max_output_chars: int = 40000
    block_patterns: list[str] = field(default_factory=list)
    tool_name: str = "run_command"


@dataclass(frozen=True)
class BackendConfig:
    name: str
    type: str
    model: str
    base_url: str
    api_key_env: str | None = None

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.getenv(self.api_key_env)


@dataclass(frozen=True)
class Config:
    default_backend: str
    chat: ChatConfig
    storage: StorageConfig
    knowledge: KnowledgeConfig
    web_search: WebSearchConfig
    command: CommandConfig
    backends: dict[str, BackendConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}. Run `icebreaker init` first.")

        data = tomllib.loads(path.read_text(encoding="utf-8"))
        backends_data = data.get("backends", {})
        default_backend = str(data.get("default_backend", "")).strip()

        if not default_backend:
            raise ValueError("Config is missing `default_backend`.")
        if not isinstance(backends_data, dict) or not backends_data:
            raise ValueError("Config must define at least one backend under `[backends.<name>]`.")
        if default_backend not in backends_data:
            raise ValueError(f"Default backend `{default_backend}` is not defined in `[backends]`.")

        chat = data.get("chat", {})
        storage = data.get("storage", {})
        knowledge = data.get("knowledge", {})
        web_search = data.get("web_search", {})
        command = data.get("command", {})

        backends = {
            name: _parse_backend_config(name, backend)
            for name, backend in backends_data.items()
        }

        return cls(
            default_backend=default_backend,
            chat=ChatConfig(
                temperature=float(chat.get("temperature", 0.2)),
                timeout_seconds=int(chat.get("timeout_seconds", 120)),
            ),
            storage=StorageConfig(session_dir=Path(storage.get("session_dir", ".icebreaker/sessions"))),
            knowledge=_parse_knowledge_config(knowledge),
            web_search=_parse_web_search_config(web_search),
            command=_parse_command_config(command),
            backends=backends,
        )

    def resolve_backend(self, name: str | None = None) -> BackendConfig:
        backend_name = name or self.default_backend
        try:
            return self.backends[backend_name]
        except KeyError as exc:
            raise ValueError(f"Backend `{backend_name}` is not defined in the config.") from exc


def write_default_config(path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG, encoding="utf-8")


def render_config(
    *,
    backends: dict[str, dict[str, Any]],
    default_backend: str,
    temperature: float = 0.2,
    timeout_seconds: int = 120,
    session_dir: str = ".icebreaker/sessions",
    knowledge_enabled: bool = False,
    knowledge_base_url: str = "http://127.0.0.1:8002/sse",
    knowledge_timeout_seconds: int = 12,
    knowledge_api_key_env: str = "",
    knowledge_max_results: int = 6,
    knowledge_tool_name: str = "search_local_knowledge",
    web_search_enabled: bool = False,
    web_search_base_url: str = "http://127.0.0.1:8003/sse",
    web_search_timeout_seconds: int = 12,
    web_search_api_key_env: str = "",
    web_search_max_results: int = 10,
    web_search_tool_name: str = "search_web",
    web_search_prompt_enabled: bool = False,
    command_enabled: bool = False,
    command_timeout_seconds: int = 30,
    command_max_output_chars: int = 40000,
    command_block_patterns: list[str] | None = None,
    command_tool_name: str = "run_command",
) -> str:
    if default_backend not in backends:
        raise ValueError(f"Default backend `{default_backend}` is not present in the backend definitions.")
    for name in backends:
        validate_backend_name(name)

    command_patterns = command_block_patterns or list(DEFAULT_COMMAND_BLOCK_PATTERNS)
    header = [
        f"default_backend = {_toml_string(default_backend)}",
        "",
        "[chat]",
        f"temperature = {temperature}",
        f"timeout_seconds = {timeout_seconds}",
        "",
        "[storage]",
        f"session_dir = {_toml_string(session_dir)}",
        "",
        "[knowledge]",
        f"enabled = {str(knowledge_enabled).lower()}",
        f"base_url = {_toml_string(knowledge_base_url)}",
        f"timeout_seconds = {knowledge_timeout_seconds}",
        f"api_key_env = {_toml_string(knowledge_api_key_env)}",
        f"max_results = {knowledge_max_results}",
        f"tool_name = {_toml_string(knowledge_tool_name)}",
        "",
        "[web_search]",
        f"enabled = {str(web_search_enabled).lower()}",
        f"base_url = {_toml_string(web_search_base_url)}",
        f"timeout_seconds = {web_search_timeout_seconds}",
        f"api_key_env = {_toml_string(web_search_api_key_env)}",
        f"max_results = {web_search_max_results}",
        f"tool_name = {_toml_string(web_search_tool_name)}",
        f"prompt_enabled = {str(web_search_prompt_enabled).lower()}",
        "",
        "[command]",
        f"enabled = {str(command_enabled).lower()}",
        f"timeout_seconds = {command_timeout_seconds}",
        f"max_output_chars = {command_max_output_chars}",
        f"block_patterns = [{', '.join(_toml_string(pattern) for pattern in command_patterns)}]",
        f"tool_name = {_toml_string(command_tool_name)}",
    ]
    blocks = [render_backend_block(name, backend) for name, backend in backends.items()]
    return "\n".join(header + [""] + blocks) + "\n"


def render_backend_block(name: str, backend: dict[str, Any]) -> str:
    validate_backend_name(name)
    backend_type = str(backend.get("type", "openai_compatible"))
    model = str(backend.get("model", "local-model"))
    base_url = str(backend.get("base_url", "http://127.0.0.1:1234/v1"))
    api_key_env = str(backend.get("api_key_env", ""))
    return "\n".join(
        [
            f"[backends.{name}]",
            f"type = {_toml_string(backend_type)}",
            f"model = {_toml_string(model)}",
            f"base_url = {_toml_string(base_url)}",
            f"api_key_env = {_toml_string(api_key_env)}",
        ]
    )


def write_rendered_config(path: Path, content: str, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _toml_string(value: str) -> str:
    return json.dumps(value)


def validate_backend_name(name: str) -> None:
    if not BACKEND_NAME_PATTERN.fullmatch(name):
        raise ValueError("Backend names may contain only letters, numbers, underscores, and hyphens.")


def _parse_backend_config(name: str, backend: object) -> BackendConfig:
    validate_backend_name(name)
    if not isinstance(backend, dict):
        raise ValueError(f"Backend `{name}` must be a TOML table.")
    return BackendConfig(
        name=name,
        type=str(backend.get("type", "openai_compatible")),
        model=str(backend.get("model", "local-model")),
        base_url=str(backend.get("base_url", "http://127.0.0.1:1234/v1")),
        api_key_env=str(backend.get("api_key_env", "")).strip() or None,
    )


def _parse_knowledge_config(knowledge: object) -> KnowledgeConfig:
    if not isinstance(knowledge, dict):
        return KnowledgeConfig()
    enabled = bool(knowledge.get("enabled", False))
    timeout_seconds = int(knowledge.get("timeout_seconds", 12))
    if enabled and timeout_seconds <= 0:
        raise ValueError("Knowledge timeout must be a positive integer.")
    return KnowledgeConfig(
        enabled=enabled,
        base_url=str(knowledge.get("base_url", "http://127.0.0.1:8002/sse")),
        timeout_seconds=timeout_seconds,
        api_key_env=str(knowledge.get("api_key_env", "")).strip() or None,
        max_results=int(knowledge.get("max_results", 6)),
        tool_name=str(knowledge.get("tool_name", "search_local_knowledge")),
    )


def _parse_web_search_config(web_search: object) -> WebSearchConfig:
    if not isinstance(web_search, dict):
        return WebSearchConfig()
    enabled = bool(web_search.get("enabled", False))
    timeout_seconds = int(web_search.get("timeout_seconds", 12))
    if enabled and timeout_seconds <= 0:
        raise ValueError("Web search timeout must be a positive integer.")
    return WebSearchConfig(
        enabled=enabled,
        base_url=str(web_search.get("base_url", "http://127.0.0.1:8003/sse")),
        timeout_seconds=timeout_seconds,
        api_key_env=str(web_search.get("api_key_env", "")).strip() or None,
        max_results=int(web_search.get("max_results", 10)),
        tool_name=str(web_search.get("tool_name", "search_web")),
        prompt_enabled=bool(web_search.get("prompt_enabled", False)),
    )


def _parse_command_config(command: object) -> CommandConfig:
    if not isinstance(command, dict):
        return CommandConfig()
    if "block_patterns" in command:
        block_patterns = command.get("block_patterns")
        if not isinstance(block_patterns, list):
            raise ValueError("Command block_patterns must be a list.")
        block_patterns = [str(pattern) for pattern in block_patterns]
    else:
        block_patterns = list(DEFAULT_COMMAND_BLOCK_PATTERNS)
    return CommandConfig(
        enabled=bool(command.get("enabled", False)),
        timeout_seconds=int(command.get("timeout_seconds", 30)),
        max_output_chars=int(command.get("max_output_chars", 40000)),
        block_patterns=block_patterns,
        tool_name=str(command.get("tool_name", "run_command")),
    )
