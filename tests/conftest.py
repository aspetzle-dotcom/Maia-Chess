import pytest


@pytest.fixture
def synthetic_pgn():
    # Deliberately fabricated games for software tests; never training evidence.
    return '''[Event "Synthetic software test"]
[Site "test-game"]
[Date "2026.01.01"]
[White "Hikaru"]
[Black "FabianoCaruana"]
[WhiteElo "3000"]
[BlackElo "2900"]
[TimeControl "180+2"]
[Result "*"]

1. e4 {[%clk 0:03:00]} e5 {[%clk 0:03:00]}
2. Nf3 {[%clk 0:02:59]} Nc6 {[%clk 0:02:58]}
3. Bb5 {[%clk 0:02:57]} a6 {[%clk 0:02:56]} *
'''


@pytest.fixture
def synthetic_dataset(tmp_path, synthetic_pgn):
    from chess_personas.data import prepare
    games = [synthetic_pgn.replace("2026.01.01", f"2026.01.{day:02d}").replace("test-game", f"test-game-{day}")
             for day in range(1, 13)]
    source = tmp_path / "games.pgn"
    source.write_text("\n\n".join(games), encoding="utf-8")
    output = tmp_path / "processed"
    report = prepare([source], output, ["hikaru", "fabianocaruana"])
    return output, report
