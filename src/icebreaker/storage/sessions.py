from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class SessionStoreError(RuntimeError):
    pass


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    backend_name: str
    created_at: str
    updated_at: str
    messages: list[dict[str, object]]

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def preview(self) -> str:
        for message in reversed(self.messages):
            if message["role"] != "system":
                return _truncate(message["content"])
        return "Empty session"


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    backend_name: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: StoredSession) -> None:
        self._path_for(session.session_id).write_text(
            json.dumps(
                {
                    "id": session.session_id,
                    "backend": session.backend_name,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "messages": session.messages,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def load_session(self, session_id: str) -> StoredSession:
        path = self._path_for(session_id)
        if not path.exists():
            raise SessionStoreError(f"Session `{session_id}` was not found.")
        return self._load_path(path)

    def list_sessions(self) -> list[SessionSummary]:
        sessions = [self._load_path(path) for path in sorted(self.path.glob("*.json"))]
        summaries = [
            SessionSummary(
                session_id=session.session_id,
                backend_name=session.backend_name,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=session.message_count,
                preview=session.preview,
            )
            for session in sessions
        ]
        return sorted(summaries, key=lambda session: session.updated_at, reverse=True)

    def _load_path(self, path: Path) -> StoredSession:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            messages = payload["messages"]
            if not isinstance(messages, list):
                raise TypeError("messages must be a list")
            return StoredSession(
                session_id=str(payload["id"]),
                backend_name=str(payload["backend"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                messages=[_normalize_message(message) for message in messages],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SessionStoreError(f"Session file `{path.name}` is malformed.") from exc

    def _path_for(self, session_id: str) -> Path:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise SessionStoreError(f"Invalid session id `{session_id}`.")
        return self.path / f"{session_id}.json"


def timestamp_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_message(message: object) -> dict[str, object]:
    if not isinstance(message, dict):
        raise TypeError("message must be an object")
    role = str(message["role"])
    content = str(message["content"])
    normalized = dict(message)
    normalized["role"] = role
    normalized["content"] = content
    return normalized


def _truncate(value: str, limit: int = 80) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped or "Empty session"
    return stripped[: limit - 3] + "..."
