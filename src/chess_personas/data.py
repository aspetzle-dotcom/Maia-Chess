"""Turn PGNs into pre-move inputs and chronologically separated labels."""

import hashlib
import io
import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn

from .clocks import elapsed_seconds, time_control


def _rating(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def iter_sources(paths):
    for source in paths:
        path = Path(source)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("games", []):
                if record.get("rules", "chess") == "chess" and record.get("time_class") != "daily":
                    yield record.get("pgn", ""), record
        else:
            with path.open(encoding="utf-8-sig") as handle:
                while (game := chess.pgn.read_game(handle)) is not None:
                    if game.errors:
                        raise ValueError(f"Malformed PGN in {path}: {game.errors[0]}")
                    yield str(game), {}


def game_rows(pgn: str, players: set[str], metadata=None):
    metadata = metadata or {}
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None or game.errors:
        return []
    if game.headers.get("Variant", "Standard") not in ("Standard", "Chess"):
        return []
    names = {chess.WHITE: game.headers.get("White", "").lower(),
             chess.BLACK: game.headers.get("Black", "").lower()}
    if not players.intersection(names.values()):
        return []
    control = time_control(game.headers.get("TimeControl", ""))
    # Unsupported timing regimes cannot be interpreted by this first version.
    if control is None:
        return []
    base, increment = control
    timestamp = metadata.get("end_time")
    if timestamp is None:
        date = game.headers.get("UTCDate", game.headers.get("Date", ""))
        clock = game.headers.get("UTCTime", "00:00:00")
        try:
            timestamp = int(datetime.strptime(date + " " + clock, "%Y.%m.%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            return []  # Unknown dates cannot support a reliable chronological split.
    nodes = list(game.mainline())
    identity = {"white": names[chess.WHITE], "black": names[chess.BLACK],
                "date": game.headers.get("UTCDate", game.headers.get("Date")),
                "time": game.headers.get("UTCTime"), "start": game.board().fen(),
                "moves": [node.move.uci() for node in nodes],
                "site": metadata.get("url", game.headers.get("Site", ""))}
    game_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    board = game.board()
    clocks = {chess.WHITE: base, chess.BLACK: base}
    counts = {chess.WHITE: 0, chess.BLACK: 0}
    history = deque(maxlen=16)
    positions = deque(maxlen=8)
    result = []
    for ply, node in enumerate(nodes):
        side = board.turn
        positions.append(board.fen())
        before = clocks[side]
        after = node.clock()
        elapsed = elapsed_seconds(before, after, increment, first_move=counts[side] == 0)
        if names[side] in players:
            result.append({
                "schema_version": 1, "game_id": game_id, "timestamp": int(timestamp),
                "player": names[side], "opponent": names[not side],
                "time_class": metadata.get("time_class", "unknown"),
                "fen": board.fen(), "board_history": list(positions),
                "history": list(history), "ply": ply,
                "base_seconds": base, "increment_seconds": increment,
                "self_clock": before, "opponent_clock": clocks[not side],
                "self_rating": _rating(game.headers.get("WhiteElo" if side else "BlackElo")),
                "opponent_rating": _rating(game.headers.get("BlackElo" if side else "WhiteElo")),
                "move": node.move.uci(), "elapsed_seconds": elapsed,
            })
        # Apply observations only after emitting the pre-move input.
        history.append({"move": node.move.uci(), "white": side, "elapsed_seconds": elapsed})
        clocks[side] = after  # None intentionally breaks the next same-side difference.
        counts[side] += 1
        board.push(node.move)
    return result


def prepare(paths, output, players, *, time_classes=None, train_fraction=0.8, val_fraction=0.1):
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Split fractions must leave nonempty train, validation and test fractions")
    players = {name.lower() for name in players}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    destinations = [out / f"{name}.jsonl" for name in ("train", "val", "test")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Use a new output directory; prepared splits already exist")
    spool = out / "positions.partial.jsonl"
    games = {}
    audit = Counter()
    with spool.open("w", encoding="utf-8") as handle:
        for pgn, metadata in iter_sources(paths):
            audit["source_games"] += 1
            if time_classes and metadata.get("time_class") not in time_classes:
                audit["excluded_time_class"] += 1
                continue
            rows = game_rows(pgn, players, metadata)
            if not rows:
                audit["excluded_or_unusable_games"] += 1
                continue
            key = rows[0]["game_id"]
            if key in games:
                audit["duplicate_games"] += 1
                continue
            games[key] = rows[0]["timestamp"]
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    # A global time boundary prevents a game between both target players leaking
    # across per-player splits. Equal timestamps stay together.
    timestamps = sorted(set(games.values()))
    if len(timestamps) < 3:
        raise ValueError("Need games at at least three distinct timestamps for chronological splits")
    train_end = max(1, min(len(timestamps) - 2, int(len(timestamps) * train_fraction)))
    val_end = max(train_end + 1, min(len(timestamps) - 1, int(len(timestamps) * (train_fraction + val_fraction))))
    cuts = (timestamps[train_end], timestamps[val_end])
    audit_by_split = {name: {} for name in ("train", "val", "test")}
    handles = {p.stem: p.open("w", encoding="utf-8") for p in destinations}
    try:
        with spool.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                stamp = row["timestamp"]
                split = "train" if stamp < cuts[0] else "val" if stamp < cuts[1] else "test"
                handles[split].write(line)
                key = row["player"] + "/" + row["time_class"]
                counts = audit_by_split[split].setdefault(key, {"positions": 0, "timed_positions": 0})
                counts["positions"] += 1
                counts["timed_positions"] += row["elapsed_seconds"] is not None
    finally:
        for handle in handles.values():
            handle.close()
    spool.unlink()
    report = {"schema_version": 1, "players": sorted(players), "games": len(games),
              "audit": dict(audit), "split_cutoff_timestamps": cuts,
              "by_split_player_time_class": audit_by_split}
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
