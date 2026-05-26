import asyncio
import json
from pathlib import Path

from icebreaker.config import Config, render_config
from icebreaker.llm.base import ModelResponse, ToolCall
from icebreaker.repl.chat import ChatRepl
from icebreaker.storage.sessions import SessionStore
from icebreaker.tools.local_knowledge import KnowledgeSearchResponse, KnowledgeResult
from icebreaker.ui.console import Console


class CaptureConsole(Console):
    def __init__(self) -> None:
        self.markdown_messages: list[str] = []
        self.json_payloads: list[object] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.statuses: list[str] = []

    def status(self, message: str) -> None:
        self.statuses.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def markdown(self, content: str) -> None:
        self.markdown_messages.append(content)

    def print_json(self, payload) -> None:
        self.json_payloads.append(payload)


class ScriptedBackend:
    def __init__(self, responses: list[ModelResponse | str], transcripts: list[list[dict[str, str]]]) -> None:
        self.responses = responses
        self.transcripts = transcripts

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        timeout_seconds: int,
        tools=None,
    ) -> ModelResponse:
        self.transcripts.append([dict(message) for message in messages])
        response = self.responses.pop(0)
        if isinstance(response, str):
            return ModelResponse(text=response)
        return response


class ScriptedKnowledgeTool:
    def __init__(self, results: list[KnowledgeSearchResponse]) -> None:
        self.results = results
        self.calls: list[tuple[str, str | None]] = []

    def search(self, *, query: str, category: str | None = None) -> KnowledgeSearchResponse:
        self.calls.append((query, category))
        return self.results.pop(0)


class ScriptedWebSearchTool:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.calls: list[tuple[str, str | None, int, int | None]] = []

    def search(
        self,
        *,
        query: str,
        categories: str | None = None,
        page: int = 1,
        max_results: int | None = None,
    ) -> dict[str, object]:
        self.calls.append((query, categories, page, max_results))
        return self.results.pop(0)


def _make_config(tmp_path: Path) -> Config:
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
            session_dir=str(tmp_path / "sessions"),
        ),
        encoding="utf-8",
    )
    return Config.load(config_path)


def test_multi_turn_chat_persists_full_history(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    transcripts: list[list[dict[str, str]]] = []
    backend = ScriptedBackend(["first reply", "second reply"], transcripts)
    inputs = iter(["hello", "follow up"])
    console = CaptureConsole()
    repl = ChatRepl(
        config=config,
        console=console,
        backend_factory=lambda *_args: backend,
        input_func=lambda _: next(inputs),
    )

    asyncio.run(repl.run())

    store = SessionStore(config.storage.session_dir)
    sessions = store.list_sessions()
    saved = store.load_session(sessions[0].session_id)

    assert console.markdown_messages == ["first reply", "second reply"]
    assert len(transcripts) == 2
    assert [message["role"] for message in transcripts[1][-2:]] == ["assistant", "user"]
    assert [message["content"] for message in saved.messages[1:]] == [
        "hello",
        "first reply",
        "follow up",
        "second reply",
    ]


def test_reset_starts_new_session_and_preserves_old_one(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    backend = ScriptedBackend(["reply one", "reply two"], [])
    inputs = iter(["first", "/reset", "second"])
    repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: backend,
        input_func=lambda _: next(inputs),
    )

    asyncio.run(repl.run())

    store = SessionStore(config.storage.session_dir)
    sessions = store.list_sessions()

    assert len(sessions) == 2
    latest = store.load_session(sessions[0].session_id)
    oldest = store.load_session(sessions[1].session_id)
    assert [message["content"] for message in latest.messages[1:]] == ["second", "reply two"]
    assert [message["content"] for message in oldest.messages[1:]] == ["first", "reply one"]


def test_sessions_command_lists_persisted_metadata(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    backend = ScriptedBackend(["reply"], [])
    inputs = iter(["hello", "/sessions"])
    console = CaptureConsole()
    repl = ChatRepl(
        config=config,
        console=console,
        backend_factory=lambda *_args: backend,
        input_func=lambda _: next(inputs),
    )

    asyncio.run(repl.run())

    assert console.json_payloads
    listing = console.json_payloads[-1]
    assert listing[0]["backend"] == "local"
    assert listing[0]["message_count"] >= 3


def test_load_restores_prior_thread_and_continues_it(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    transcripts: list[list[dict[str, str]]] = []
    first_backend = ScriptedBackend(["first reply"], transcripts)
    first_inputs = iter(["hello"])
    first_repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: first_backend,
        input_func=lambda _: next(first_inputs),
    )
    asyncio.run(first_repl.run())

    store = SessionStore(config.storage.session_dir)
    session_id = store.list_sessions()[0].session_id

    second_backend = ScriptedBackend(["continued reply"], transcripts)
    second_inputs = iter([f"/load {session_id}", "continue"])
    second_repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: second_backend,
        input_func=lambda _: next(second_inputs),
    )
    asyncio.run(second_repl.run())

    saved = store.load_session(session_id)

    assert len(transcripts[-1]) == 4
    assert [message["content"] for message in saved.messages[1:]] == [
        "hello",
        "first reply",
        "continue",
        "continued reply",
    ]


def test_eof_and_keyboard_interrupt_exit_without_corrupting_sessions(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    eof_backend = ScriptedBackend(["reply"], [])
    eof_inputs = iter(["hello"])
    eof_repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: eof_backend,
        input_func=lambda _: next(eof_inputs),
    )
    asyncio.run(eof_repl.run())

    def interrupting_input(_: str) -> str:
        raise KeyboardInterrupt

    interrupt_repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: ScriptedBackend([], []),
        input_func=interrupting_input,
    )
    asyncio.run(interrupt_repl.run())

    store = SessionStore(config.storage.session_dir)
    for summary in store.list_sessions():
        loaded = store.load_session(summary.session_id)
        assert loaded.session_id
        assert loaded.messages[0]["role"] == "system"


def test_tool_call_executes_knowledge_search_and_persists_tool_message(tmp_path: Path) -> None:
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
            session_dir=str(tmp_path / "sessions"),
        ),
        encoding="utf-8",
    )
    config = Config.load(config_path)
    transcripts: list[list[dict[str, str]]] = []
    backend = ScriptedBackend(
        [
            ModelResponse(tool_call=ToolCall("search_local_knowledge", {"query": "latest threats", "category": None}, "call_1")),
            ModelResponse(text="Here is the summary."),
        ],
        transcripts,
    )
    knowledge_tool = ScriptedKnowledgeTool(
        [
            KnowledgeSearchResponse(
                query="latest threats",
                category=None,
                results=[
                    KnowledgeResult(
                        title="Threat Bulletin",
                        snippet="Ransomware campaigns are active.",
                        source="bulletin-1",
                        score=0.91,
                        category="threat-intelligence",
                    )
                ],
            )
        ]
    )
    repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: backend,
        input_func=lambda _prompt, inputs=iter(["hello"]): next(inputs),
    )
    repl._knowledge_tool = knowledge_tool

    asyncio.run(repl.run())

    store = SessionStore(config.storage.session_dir)
    saved = store.load_session(store.list_sessions()[0].session_id)

    assert knowledge_tool.calls == [("latest threats", None)]
    assert any(message["role"] == "tool" for message in saved.messages)
    tool_message = next(message for message in saved.messages if message["role"] == "tool")
    assert tool_message["name"] == "search_local_knowledge"
    assert json.loads(tool_message["content"])["results"][0]["source"] == "bulletin-1"
    assert transcripts[-1][-1]["role"] == "tool"


def test_tool_failure_warns_without_corrupting_session(tmp_path: Path) -> None:
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
            session_dir=str(tmp_path / "sessions"),
        ),
        encoding="utf-8",
    )
    config = Config.load(config_path)
    console = CaptureConsole()
    backend = ScriptedBackend(
        [ModelResponse(tool_call=ToolCall("search_local_knowledge", {"query": ""}, "call_1"))],
        [],
    )
    repl = ChatRepl(
        config=config,
        console=console,
        backend_factory=lambda *_args: backend,
        input_func=lambda _prompt, inputs=iter(["hello"]): next(inputs),
    )

    asyncio.run(repl.run())

    assert console.warnings
    assert "requires a non-empty string `query`" in console.warnings[-1]


def test_tool_call_executes_web_search_and_persists_tool_message(tmp_path: Path) -> None:
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
            session_dir=str(tmp_path / "sessions"),
        ),
        encoding="utf-8",
    )
    config = Config.load(config_path)
    transcripts: list[list[dict[str, str]]] = []
    backend = ScriptedBackend(
        [
            ModelResponse(
                tool_call=ToolCall(
                    "search_web",
                    {"query": "latest ai", "categories": "news", "page": 1, "max_results": 5},
                    "call_1",
                )
            ),
            ModelResponse(text="Here is the summary."),
        ],
        transcripts,
    )
    web_tool = ScriptedWebSearchTool(
        [
            {
                "results": [
                    {"title": "AI News", "url": "https://example.com", "snippet": "Update"}
                ]
            }
        ]
    )
    repl = ChatRepl(
        config=config,
        console=CaptureConsole(),
        backend_factory=lambda *_args: backend,
        input_func=lambda _prompt, inputs=iter(["hello"]): next(inputs),
    )
    repl._web_search_tool = web_tool

    asyncio.run(repl.run())

    store = SessionStore(config.storage.session_dir)
    saved = store.load_session(store.list_sessions()[0].session_id)

    assert web_tool.calls == [("latest ai", "news", 1, 5)]
    assert any(message["role"] == "tool" for message in saved.messages)
    tool_message = next(message for message in saved.messages if message["role"] == "tool")
    assert tool_message["name"] == "search_web"
    assert json.loads(tool_message["content"])["results"][0]["title"] == "AI News"
    assert transcripts[-1][-1]["role"] == "tool"
