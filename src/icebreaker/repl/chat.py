from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from icebreaker.config import Config
from icebreaker.llm.base import ChatModel, ModelResponse, ToolCall, ToolDefinition
from icebreaker.llm.factory import build_backend
from icebreaker.repl.commands import ReplCommand, parse_repl_command
from icebreaker.storage.sessions import SessionStore, SessionStoreError, StoredSession, timestamp_now
from icebreaker.tools.command_exec import CommandExecutionError, CommandExecutor
from icebreaker.tools.local_knowledge import LocalKnowledgeError, LocalKnowledgeTool
from icebreaker.tools.web_search import WebSearchError, WebSearchTool
from icebreaker.ui.console import Console


SYSTEM_PROMPT = """\
You are Icebreaker, a conservative security assistant working in a terminal chat session.

Stay within the information available in the conversation. If something is uncertain, say so directly.
Do not invent findings, access, test results, or external actions that did not happen.
Do not provide harmful, destructive, or evasive operational guidance.
Offer safe, bounded reasoning and clearly state when the user needs to validate something themselves.
"""


HELP_TEXT = """\
Commands:
  /help                 Show this help.
  /reset                Start a new in-memory session and persist it immediately.
  /sessions             List saved sessions.
  /load <session-id>    Load a saved session and continue that thread.
  /exit                 Quit the REPL.
"""


@dataclass
class ActiveSession:
    session_id: str
    backend_name: str
    created_at: str
    updated_at: str
    messages: list[dict[str, object]]


class ChatRepl:
    def __init__(
        self,
        config: Config,
        console: Console,
        store: SessionStore | None = None,
        backend_factory=build_backend,
        input_func=input,
        **_: object,
    ) -> None:
        self.config = config
        self.console = console
        self.store = store or SessionStore(config.storage.session_dir)
        self.backend_factory = backend_factory
        self.input_func = input_func
        self._backend_cache: dict[str, ChatModel] = {}
        self._knowledge_tool = LocalKnowledgeTool(config.knowledge) if config.knowledge.enabled else None
        self._web_search_tool = WebSearchTool(config.web_search) if config.web_search.enabled else None
        self._command_tool = CommandExecutor(config.command) if config.command.enabled else None

    async def run(self, backend_name: str | None = None) -> None:
        session = self._new_session(self.config.resolve_backend(backend_name).name)
        self.console.status("Type /help for commands, /exit to quit.")

        while True:
            try:
                line = self.input_func("icebreaker> ")
            except (EOFError, KeyboardInterrupt, StopIteration):
                print()
                break

            line = line.strip()
            if not line:
                continue

            command = parse_repl_command(line)
            if command:
                should_continue, session = self._handle_command(command, session)
                if not should_continue:
                    break
                continue

            await self._handle_user_message(session, line)

    def _handle_command(self, command: ReplCommand, session: ActiveSession) -> tuple[bool, ActiveSession]:
        if command.kind == "exit":
            return False, session

        if command.kind == "help":
            self.console.markdown(HELP_TEXT)
            return True, session

        if command.kind == "reset":
            reset_session = self._new_session(session.backend_name)
            return True, reset_session

        if command.kind == "sessions":
            try:
                summaries = self.store.list_sessions()
            except SessionStoreError as exc:
                self.console.warn(str(exc))
                return True, session
            self.console.print_json(
                [
                    {
                        "id": summary.session_id,
                        "backend": summary.backend_name,
                        "created_at": summary.created_at,
                        "updated_at": summary.updated_at,
                        "message_count": summary.message_count,
                        "preview": summary.preview,
                    }
                    for summary in summaries
                ]
            )
            return True, session

        if command.kind == "load":
            try:
                loaded = self.store.load_session(command.args["session_id"])
                self.config.resolve_backend(loaded.backend_name)
            except (SessionStoreError, ValueError) as exc:
                self.console.warn(str(exc))
                return True, session
            active = self._activate_session(loaded)
            self.console.success(f"Loaded session {active.session_id}")
            return True, active

        if command.kind == "error":
            self.console.warn(command.args["message"])
            return True, session

        self.console.warn(f"Unhandled command: {command.kind}")
        return True, session

    async def _handle_user_message(self, session: ActiveSession, content: str) -> None:
        session.messages.append({"role": "user", "content": content})
        self._save_session(session)

        backend = self._get_backend(session.backend_name)
        try:
            response = await backend.complete(
                session.messages,
                temperature=self.config.chat.temperature,
                timeout_seconds=self.config.chat.timeout_seconds,
                tools=self._tool_definitions(),
            )
        except Exception as exc:
            self.console.warn(f"Backend request failed: {exc}")
            return
        await self._handle_model_response(session, response)

    def _new_session(self, backend_name: str) -> ActiveSession:
        now = timestamp_now()
        session = ActiveSession(
            session_id=uuid.uuid4().hex,
            backend_name=backend_name,
            created_at=now,
            updated_at=now,
            messages=[{"role": "system", "content": self._system_prompt()}],
        )
        self._save_session(session)
        self.console.success(f"Started session {session.session_id}")
        return session

    def _save_session(self, session: ActiveSession) -> None:
        session.updated_at = timestamp_now()
        self.store.save_session(
            StoredSession(
                session_id=session.session_id,
                backend_name=session.backend_name,
                created_at=session.created_at,
                updated_at=session.updated_at,
                messages=list(session.messages),
            )
        )

    def _activate_session(self, stored: StoredSession) -> ActiveSession:
        return ActiveSession(
            session_id=stored.session_id,
            backend_name=stored.backend_name,
            created_at=stored.created_at,
            updated_at=stored.updated_at,
            messages=list(stored.messages),
        )

    def _get_backend(self, backend_name: str) -> ChatModel:
        backend = self._backend_cache.get(backend_name)
        if backend is None:
            backend = self.backend_factory(self.config, backend_name)
            self._backend_cache[backend_name] = backend
        return backend

    def _system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT
        tool_lines = []
        if self._knowledge_tool:
            tool_lines.append("- You can call a local knowledge search tool when helpful.")
        if self._web_search_tool:
            tool_lines.append("- You can call a web search tool when fresh context is needed.")
        if self._command_tool:
            tool_lines.append("- You can run commands via the command tool for user-approved tasks.")
        if tool_lines:
            prompt = prompt + "\n" + "\n".join(tool_lines)
        if self._web_search_tool:
            try:
                web_prompt = self._web_search_tool.prompt()
            except WebSearchError as exc:
                self.console.warn(str(exc))
                web_prompt = None
            if web_prompt and web_prompt.text.strip():
                prompt = prompt + "\n\n" + web_prompt.text.strip()
        return prompt

    def _tool_definitions(self) -> list[ToolDefinition] | None:
        tools: list[ToolDefinition] = []
        if self._knowledge_tool and hasattr(self._knowledge_tool, "definition"):
            tools.append(self._knowledge_tool.definition())
        if self._web_search_tool and hasattr(self._web_search_tool, "definition"):
            tools.append(self._web_search_tool.definition())
        if self._command_tool and hasattr(self._command_tool, "definition"):
            tools.append(self._command_tool.definition())
        return tools or None

    async def _handle_model_response(self, session: ActiveSession, response: ModelResponse) -> None:
        if response.tool_call:
            tool_message = self._execute_tool_call(response.tool_call)
            if tool_message is None:
                return
            session.messages.append(tool_message)
            self._save_session(session)

            backend = self._get_backend(session.backend_name)
            try:
                followup = await backend.complete(
                    session.messages,
                    temperature=self.config.chat.temperature,
                    timeout_seconds=self.config.chat.timeout_seconds,
                    tools=self._tool_definitions(),
                )
            except Exception as exc:
                self.console.warn(f"Backend request failed: {exc}")
                return
            if followup.text is not None:
                session.messages.append({"role": "assistant", "content": followup.text})
                self._save_session(session)
                self.console.markdown(followup.text)
            return

        if response.text is not None:
            session.messages.append({"role": "assistant", "content": response.text})
            self._save_session(session)
            self.console.markdown(response.text)

    def _execute_tool_call(self, tool_call: ToolCall) -> dict[str, object] | None:
        if tool_call.name == self.config.knowledge.tool_name:
            return self._run_knowledge_tool(tool_call)
        if tool_call.name == self.config.web_search.tool_name:
            return self._run_web_search_tool(tool_call)
        if tool_call.name == self.config.command.tool_name:
            return self._run_command_tool(tool_call)
        self.console.warn(f"Unknown tool: {tool_call.name}")
        return None

    def _run_knowledge_tool(self, tool_call: ToolCall) -> dict[str, object] | None:
        if not self._knowledge_tool:
            self.console.warn("Knowledge tool is not enabled.")
            return None
        query = tool_call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            self.console.warn("Knowledge search requires a non-empty string `query`.")
            return None
        category = tool_call.arguments.get("category")
        if category is not None and not isinstance(category, str):
            self.console.warn("Knowledge search `category` must be a string or null.")
            return None
        try:
            response = self._knowledge_tool.search(query=query.strip(), category=category)
        except LocalKnowledgeError as exc:
            self.console.warn(str(exc))
            return None
        return {
            "role": "tool",
            "name": tool_call.name,
            "tool_call_id": tool_call.call_id,
            "content": json.dumps(response.to_dict()),
        }

    def _run_command_tool(self, tool_call: ToolCall) -> dict[str, object] | None:
        if not self._command_tool:
            self.console.warn("Command tool is not enabled.")
            return None
        command = tool_call.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            self.console.warn("Command tool requires a non-empty string `command`.")
            return None
        try:
            result = self._command_tool.run(command.strip())
        except CommandExecutionError as exc:
            self.console.warn(str(exc))
            return None
        return {
            "role": "tool",
            "name": tool_call.name,
            "tool_call_id": tool_call.call_id,
            "content": json.dumps(result.to_dict()),
        }

    def _run_web_search_tool(self, tool_call: ToolCall) -> dict[str, object] | None:
        if not self._web_search_tool:
            self.console.warn("Web search tool is not enabled.")
            return None
        query = tool_call.arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            self.console.warn("Web search requires a non-empty string `query`.")
            return None
        categories = tool_call.arguments.get("categories")
        if categories is not None and not isinstance(categories, str):
            self.console.warn("Web search `categories` must be a string or null.")
            return None
        page = tool_call.arguments.get("page", 1)
        if not isinstance(page, int):
            self.console.warn("Web search `page` must be an integer.")
            return None
        max_results = tool_call.arguments.get("max_results")
        if max_results is not None and not isinstance(max_results, int):
            self.console.warn("Web search `max_results` must be an integer.")
            return None
        try:
            result = self._web_search_tool.search(
                query=query.strip(),
                categories=categories,
                page=page,
                max_results=max_results,
            )
        except WebSearchError as exc:
            self.console.warn(str(exc))
            return None
        return {
            "role": "tool",
            "name": tool_call.name,
            "tool_call_id": tool_call.call_id,
            "content": json.dumps(result),
        }
