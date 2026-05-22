from pathlib import Path

import pytest

from icebreaker.storage.sessions import SessionStore, SessionStoreError, StoredSession


def test_auto_save_creates_valid_json_transcript_files(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = StoredSession(
        session_id="session-1",
        backend_name="local",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hello"},
        ],
    )

    store.save_session(session)
    loaded = store.load_session("session-1")

    assert loaded.session_id == "session-1"
    assert loaded.messages[-1]["content"] == "hello"


def test_persists_tool_message_metadata(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = StoredSession(
        session_id="session-tool",
        backend_name="local",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "tool", "name": "search_local_knowledge", "tool_call_id": "call_1", "content": '{"results":[]}'},
        ],
    )

    store.save_session(session)
    loaded = store.load_session("session-tool")

    assert loaded.messages[-1]["name"] == "search_local_knowledge"
    assert loaded.messages[-1]["tool_call_id"] == "call_1"


def test_list_sessions_reads_metadata(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.save_session(
        StoredSession(
            session_id="older",
            backend_name="local",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:01+00:00",
            messages=[{"role": "system", "content": "prompt"}],
        )
    )
    store.save_session(
        StoredSession(
            session_id="newer",
            backend_name="openai",
            created_at="2026-01-01T00:00:02+00:00",
            updated_at="2026-01-01T00:00:03+00:00",
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "test preview"},
            ],
        )
    )

    sessions = store.list_sessions()

    assert [session.session_id for session in sessions] == ["newer", "older"]
    assert sessions[0].preview == "test preview"


def test_loading_missing_or_malformed_session_fails_cleanly(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    (tmp_path / "sessions" / "broken.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SessionStoreError, match="not found"):
        store.load_session("missing")

    with pytest.raises(SessionStoreError, match="malformed"):
        store.list_sessions()
