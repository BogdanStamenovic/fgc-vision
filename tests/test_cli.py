from __future__ import annotations

import pytest

from fgc_vision.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "fgc-vision" in capsys.readouterr().out


def test_usage_error_is_2() -> None:
    assert main(["nonsense"]) == 2
