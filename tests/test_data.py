import json

import pytest

from chess_personas.clocks import elapsed_seconds, time_control
from chess_personas.data import game_rows, prepare


@pytest.mark.parametrize("value,expected", [("180+2", (180, 2)), ("60", (60, 0)),
    ("40/7200:3600", None), ("1/86400", None), ("-", None), ("0+1", None)])
def test_time_controls(value, expected):
    assert time_control(value) == expected


def test_elapsed_increment_and_invalid_labels():
    assert elapsed_seconds(180, 179, 2) == 3
    assert elapsed_seconds(180, 182, 2) == 0
    assert elapsed_seconds(180, 183, 2) is None
    assert elapsed_seconds(None, 179, 2) is None
    assert elapsed_seconds(180, 179, 2, first_move=True) is None
    assert elapsed_seconds(1, 0, 2) is None


def test_only_target_moves_and_pre_move_clocks(synthetic_pgn):
    rows = game_rows(synthetic_pgn, {"hikaru"})
    assert [r["move"] for r in rows] == ["e2e4", "g1f3", "f1b5"]
    assert rows[0]["elapsed_seconds"] is None
    assert rows[1]["self_clock"] == 180
    assert rows[1]["elapsed_seconds"] == 3
    assert rows[1]["opponent_clock"] == 180
    assert [h["move"] for h in rows[1]["history"]] == ["e2e4", "e7e5"]
    assert rows[1]["board_history"][-1] == rows[1]["fen"]


def test_missing_clock_does_not_bridge_two_turns(synthetic_pgn):
    pgn = synthetic_pgn.replace("Nf3 {[%clk 0:02:59]}", "Nf3")
    rows = game_rows(pgn, {"hikaru"})
    assert rows[1]["elapsed_seconds"] is None
    assert rows[2]["self_clock"] is None
    assert rows[2]["elapsed_seconds"] is None


def test_changing_label_clock_does_not_change_current_inputs(synthetic_pgn):
    before = game_rows(synthetic_pgn, {"hikaru"})[1]
    after = game_rows(synthetic_pgn.replace("0:02:59", "0:02:49"), {"hikaru"})[1]
    assert before["elapsed_seconds"] != after["elapsed_seconds"]
    assert {k: v for k, v in before.items() if k != "elapsed_seconds"} == {
        k: v for k, v in after.items() if k != "elapsed_seconds"}


def test_whole_games_chronological_and_disjoint(synthetic_dataset):
    directory, report = synthetic_dataset
    rows = {s: [json.loads(line) for line in (directory / f"{s}.jsonl").read_text().splitlines()]
            for s in ("train", "val", "test")}
    ids = {s: {r["game_id"] for r in rs} for s, rs in rows.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert max(r["timestamp"] for r in rows["train"]) < min(r["timestamp"] for r in rows["val"])
    assert max(r["timestamp"] for r in rows["val"]) < min(r["timestamp"] for r in rows["test"])
    assert report["games"] == 12
    for split in rows.values():
        assert {r["player"] for r in split} == {"hikaru", "fabianocaruana"}


def test_duplicates_do_not_inflate_dataset(tmp_path, synthetic_pgn):
    source = tmp_path / "games.pgn"
    games = [synthetic_pgn.replace("2026.01.01", f"2026.01.0{day}") for day in range(1, 5)]
    source.write_text("\n\n".join(games + games), encoding="utf-8")
    report = prepare([source], tmp_path / "out", ["hikaru"])
    assert report["games"] == 4
    assert report["audit"]["duplicate_games"] == 4
