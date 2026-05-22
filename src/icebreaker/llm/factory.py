from __future__ import annotations

from collections.abc import Callable

from icebreaker.config import BackendConfig, Config, KnowledgeConfig
from icebreaker.llm.base import ChatModel
from icebreaker.llm.openai_compatible import OpenAICompatibleModel


BackendBuilder = Callable[[BackendConfig, KnowledgeConfig | None], ChatModel]
_BACKEND_BUILDERS: dict[str, BackendBuilder] = {
    "openai_compatible": OpenAICompatibleModel,
}


def register_backend(type_name: str, builder: BackendBuilder) -> None:
    _BACKEND_BUILDERS[type_name] = builder


def resolve_backend_config(config: Config, backend_name: str | None = None) -> BackendConfig:
    return config.resolve_backend(backend_name)


def build_backend(config: Config | BackendConfig, backend_name: str | None = None) -> ChatModel:
    backend = resolve_backend_config(config, backend_name) if isinstance(config, Config) else config
    knowledge = config.knowledge if isinstance(config, Config) else None
    try:
        builder = _BACKEND_BUILDERS[backend.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend type `{backend.type}`.") from exc
    return builder(backend, knowledge)
