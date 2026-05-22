from icebreaker.repl.commands import parse_repl_command


def test_parse_load_command() -> None:
    command = parse_repl_command("/load abc123")

    assert command is not None
    assert command.kind == "load"
    assert command.args["session_id"] == "abc123"


def test_parse_reset_command() -> None:
    command = parse_repl_command("/reset")

    assert command is not None
    assert command.kind == "reset"
