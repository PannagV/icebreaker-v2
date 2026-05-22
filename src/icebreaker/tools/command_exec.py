from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from icebreaker.config import CommandConfig
from icebreaker.llm.base import ToolDefinition


class CommandExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "truncated": self.truncated,
        }


class CommandExecutor:
    def __init__(self, config: CommandConfig) -> None:
        self.config = config
        self._block_patterns = [re.compile(pattern) for pattern in config.block_patterns]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.config.tool_name,
            description="Run a shell command on the local machine.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        )

    def run(self, command: str) -> CommandResult:
        if not command.strip():
            raise CommandExecutionError("Command must be a non-empty string.")
        self._ensure_allowed(command)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandExecutionError(
                f"Command timed out after {self.config.timeout_seconds}s."
            ) from exc
        stdout, stdout_truncated = _truncate_output(completed.stdout, self.config.max_output_chars)
        stderr, stderr_truncated = _truncate_output(completed.stderr, self.config.max_output_chars)
        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=completed.returncode,
            truncated=stdout_truncated or stderr_truncated,
        )

    def _ensure_allowed(self, command: str) -> None:
        for pattern in self._block_patterns:
            if pattern.search(command):
                raise CommandExecutionError("Command blocked by policy.")


def _truncate_output(value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return value, False
    if len(value) <= max_chars:
        return value, False
    return value[: max_chars - 16] + "... [truncated]", True
