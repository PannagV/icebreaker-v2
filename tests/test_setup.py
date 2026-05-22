from pathlib import Path

import pytest

from icebreaker.config import Config
from icebreaker.setup import ensure_config, run_setup_wizard
from icebreaker.ui.console import Console


def test_run_setup_wizard_writes_valid_minimal_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "icebreaker.toml"
    responses = iter(
        [
            "1",
            "local",
            "local-model",
            "http://127.0.0.1:1234/v1",
            "n",
            "n",
            "0.1",
            "45",
            "1",
            "n",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    run_setup_wizard(config_path, console=Console())
    config = Config.load(config_path)

    assert config.default_backend == "local"
    assert config.chat.temperature == 0.1
    assert config.chat.timeout_seconds == 45
    assert config.knowledge.enabled is False
    assert config.resolve_backend().base_url == "http://127.0.0.1:1234/v1"


def test_ensure_config_is_explicit_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="icebreaker init"):
        ensure_config(tmp_path / "missing.toml")
