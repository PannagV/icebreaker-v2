from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from icebreaker.config import render_config, validate_backend_name, write_rendered_config
from icebreaker.ui.console import Console


@dataclass(frozen=True)
class BackendDraft:
    name: str
    type: str
    model: str
    base_url: str
    api_key_env: str


def ensure_config(path: Path, console: Console | None = None) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"Config not found: {path}. Run `icebreaker init` first.")


def run_setup_wizard(path: Path, console: Console, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite it.")

    console.markdown("Icebreaker backend setup")

    drafts: list[BackendDraft] = []
    while True:
        draft = _prompt_backend(drafts)
        drafts.append(draft)
        if not _prompt_yes_no("Add another backend?", default=False):
            break

    temperature = _prompt_float("Default temperature", 0.2)
    timeout_seconds = _prompt_int("Default request timeout seconds", 120)
    default_backend = _choose_default_backend(drafts)
    knowledge_enabled = _prompt_yes_no("Enable local knowledge search tool?", default=False)
    knowledge_base_url = "http://127.0.0.1:8002/sse"
    knowledge_timeout_seconds = 12
    knowledge_api_key_env = ""
    knowledge_max_results = 6
    knowledge_tool_name = "search_local_knowledge"
    if knowledge_enabled:
        knowledge_base_url = _prompt("Knowledge base URL", knowledge_base_url)
        knowledge_timeout_seconds = _prompt_int("Knowledge timeout seconds", knowledge_timeout_seconds)
        knowledge_use_key = _prompt_yes_no("Use a knowledge API key env var?", default=False)
        if knowledge_use_key:
            knowledge_api_key_env = _prompt("Knowledge API key env var name", "KNOWLEDGE_API_KEY")
        knowledge_max_results = _prompt_int("Knowledge max results", knowledge_max_results)
        knowledge_tool_name = _prompt("Knowledge tool name", knowledge_tool_name)

    content = render_config(
        backends={
            draft.name: {
                "type": draft.type,
                "model": draft.model,
                "base_url": draft.base_url,
                "api_key_env": draft.api_key_env,
            }
            for draft in drafts
        },
        default_backend=default_backend,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        knowledge_enabled=knowledge_enabled,
        knowledge_base_url=knowledge_base_url,
        knowledge_timeout_seconds=knowledge_timeout_seconds,
        knowledge_api_key_env=knowledge_api_key_env,
        knowledge_max_results=knowledge_max_results,
        knowledge_tool_name=knowledge_tool_name,
    )
    write_rendered_config(path, content, force=force)
    console.success(f"Wrote {path}")


def _prompt_backend(existing: list[BackendDraft]) -> BackendDraft:
    backend_choice = _prompt_choice(
        "Backend type",
        {
            "1": "openai_compatible",
        },
        default="1",
    )
    backend_name = _prompt_unique_backend_name(existing)
    model = _prompt("Model identifier", "local-model")
    base_url = _prompt("Base URL", "http://127.0.0.1:1234/v1")
    use_api_key = _prompt_yes_no("Use an API key environment variable?", default=False)
    api_key_env = _prompt("API key env var name", "OPENAI_API_KEY") if use_api_key else ""
    if api_key_env and not os.getenv(api_key_env):
        print(f"[!] {api_key_env} is not currently set.")
    return BackendDraft(
        name=backend_name,
        type="openai_compatible" if backend_choice == "1" else "openai_compatible",
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def _prompt_unique_backend_name(existing: list[BackendDraft]) -> str:
    taken = {draft.name for draft in existing}
    default_name = "local" if not existing else f"backend{len(existing) + 1}"
    while True:
        name = _prompt("Backend name", default_name)
        try:
            validate_backend_name(name)
        except ValueError as exc:
            print(exc)
            continue
        if name not in taken:
            return name
        print(f"Backend `{name}` already exists.")


def _choose_default_backend(drafts: list[BackendDraft]) -> str:
    choices = {str(index): draft.name for index, draft in enumerate(drafts, start=1)}
    selected = _prompt_choice("Default backend", choices, default="1")
    return choices[selected]


def _prompt_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    print(prompt)
    for key, label in choices.items():
        suffix = " [default]" if key == default else ""
        print(f"  {key}. {label}{suffix}")

    while True:
        value = _read_input("> ", default)
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def _prompt(prompt: str, default: str) -> str:
    return _read_input(f"{prompt} [{default}]: ", default)


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        value = _read_input(f"{prompt} [{default}]: ", "")
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("Enter a number.")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        value = _read_input(f"{prompt} [{default}]: ", "")
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Enter an integer.")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        value = _read_input(f"{prompt} [{default_label}]: ", "").lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Enter y or n.")


def _read_input(prompt: str, default: str) -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        print()
        return default
    return value or default
