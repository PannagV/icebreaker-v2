from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReplCommand:
    kind: str
    args: dict[str, Any]


def parse_repl_command(line: str) -> ReplCommand | None:
    if not line.startswith("/"):
        return None

    command, _, rest = line.partition(" ")
    command = command.lower()
    rest = rest.strip()

    if command == "/exit":
        return ReplCommand("exit", {})
    if command == "/help":
        return ReplCommand("help", {})
    if command == "/reset":
        return ReplCommand("reset", {})
    if command == "/sessions":
        return ReplCommand("sessions", {})
    if command == "/load":
        if not rest:
            return ReplCommand("error", {"message": "Usage: /load <session-id>"})
        return ReplCommand("load", {"session_id": rest})

    return ReplCommand("error", {"message": f"Unknown command: {command}"})
