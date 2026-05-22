from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from icebreaker.config import Config
from icebreaker.repl.chat import ChatRepl
from icebreaker.setup import ensure_config, run_setup_wizard
from icebreaker.ui.console import Console


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    if args.command is None:
        parser.print_help()
        return

    config_path = Path(args.config)

    try:
        if args.command == "init":
            run_setup_wizard(config_path, console=console, force=args.force)
            return

        if args.command == "chat":
            ensure_config(config_path, console)
            config = Config.load(config_path)
            repl = ChatRepl(config=config, console=console)
            asyncio.run(repl.run(args.backend))
            return
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        console.warn(str(exc))
        raise SystemExit(1) from exc

    parser.print_help()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="icebreaker",
        description="Terminal-first security assistant with local session persistence.",
    )
    parser.add_argument("--config", default="icebreaker.toml", help="Path to the TOML config file.")

    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Create a backend config file interactively.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing config file.")

    chat_parser = subparsers.add_parser("chat", help="Start the interactive chat REPL.")
    chat_parser.add_argument("--backend", help="Use a configured backend other than the default.")

    return parser


if __name__ == "__main__":
    main()
