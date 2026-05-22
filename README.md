# Icebreaker

Icebreaker is a terminal-first security assistant with a deliberately small surface:

- `icebreaker init` writes a TOML config for one or more named model backends
- `icebreaker chat` starts a REPL that persists each session as local JSON

V1 is intentionally narrow. It is a chat REPL backed by a configured model provider and local JSON session files.

## Quick Start

Initialize a config:

```bash
PYTHONPATH=src python3 -m icebreaker init
```

Start the chat REPL:

```bash
PYTHONPATH=src python3 -m icebreaker chat
PYTHONPATH=src python3 -m icebreaker chat --backend openai
```

## REPL Commands

```text
/help
/reset
/sessions
/load <session-id>
/exit
```

Sessions are written as JSON files under `.icebreaker/sessions` by default. `/load` resumes the same thread rather than creating a fork.

## Configuration

The config stores a default backend, shared chat defaults, and one or more named backends:

```toml
default_backend = "openai"

[chat]
temperature = 0.2
timeout_seconds = 120

[storage]
session_dir = ".icebreaker/sessions"

[backends.openai]
type = "openai_compatible"
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"

[backends.local]
type = "openai_compatible"
model = "local-model"
base_url = "http://127.0.0.1:1234/v1"
api_key_env = ""
```

Secrets stay in the environment. The config stores only env var names:

```bash
export OPENAI_API_KEY=...
```

## Notes

The system prompt is intentionally conservative. Icebreaker is meant to avoid overclaiming, unsupported actions, and harmful guidance.
