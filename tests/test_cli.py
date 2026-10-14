from pathlib import Path

import pytest

from chess_personas.cli import main


@pytest.mark.parametrize("arguments", [[], ["--player", "hikaru", "fabianocaruana"]])
def test_training_requires_exactly_one_player(arguments):
    with pytest.raises(SystemExit) as error:
        main(["train", *arguments])
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--player", "--players"])
def test_single_player_training_dispatch(flag, monkeypatch, capsys):
    pytest.importorskip("torch")
    from chess_personas import training

    calls = []

    def capture_train(**kwargs):
        calls.append(kwargs)
        return Path(kwargs["output"]) / "best.pt"

    monkeypatch.setattr(training, "train", capture_train)
    main(["train", flag, "Hikaru", "--output", "runs/single-player/hikaru"])
    assert calls[0]["players"] == ["hikaru"]
    assert "best.pt" in capsys.readouterr().out
