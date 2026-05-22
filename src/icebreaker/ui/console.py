from __future__ import annotations

import json
import sys
from typing import Any


class Console:
    def status(self, message: str) -> None:
        print(f"[*] {message}", file=sys.stderr)

    def success(self, message: str) -> None:
        print(f"[+] {message}")

    def warn(self, message: str) -> None:
        print(f"[!] {message}", file=sys.stderr)

    def block(self, message: str) -> None:
        print(f"[blocked] {message}", file=sys.stderr)

    def markdown(self, content: str) -> None:
        print(content)

    def print_json(self, payload: Any) -> None:
        print(json.dumps(payload, indent=2, sort_keys=True))

