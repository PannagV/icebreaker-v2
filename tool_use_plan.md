Plan: Command Tooling For LLM
Add a structured command-execution tool path to the REPL so the model can request shell commands, the host executes them with safety gates, and the results are returned to the model. This aligns with the existing REPL architecture while extending ChatModel.complete() to support tool-call responses. We will add a command runner service with allow/deny rules, output limits, and timeouts, integrate it into the REPL loop, and update session persistence to preserve tool metadata. The OpenAI-compatible backend will be extended to send tool definitions and parse tool calls. This approach uses the tool-call interface you selected, allows full output, and blocks destructive commands.

Steps

Extend the LLM response model to support tool calls and tool results: update ChatModel.complete() contract and add a response type (e.g., ModelResponse with content, tool_calls) in base.py:6-15.
Implement a command-execution tool with safety checks (deny patterns like rm -rf, enforce timeouts, limit environment, allowlist vs denylist per config) and stdout/stderr capture in a new module under tools. Keep output full per your preference but apply max bytes/lines to prevent runaway output.
Wire tool definitions and parsing into the OpenAI-compatible backend: send tool schema, detect tool calls, and return them in the new response type in openai_compatible.py:21-53.
Update the REPL flow to handle tool calls: in ChatRepl._handle_user_message(), call the tool runner when a tool call is returned, then feed the tool result back to the model for final response in chat.py:61-152.
Persist tool metadata in sessions so history remains consistent across reloads: extend serialization format beyond {role, content} in sessions.py:89-120.
Add config knobs for command tool enablement and safety policy (deny patterns, timeouts, max output) in config.py:33-180, then update CLI/help text in cli.py.
Verification

Run unit tests and add new tests for tool-call parsing and command execution safety: pytest.
Manual REPL checks: run a safe command (e.g., ls), confirm full output, try a blocked command and confirm denial.
Decisions

Use tool calls instead of slash commands.
Allow full output but still cap output size for safety.
Block destructive operations (e.g., rm -rf, shutdown).