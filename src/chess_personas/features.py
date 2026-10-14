"""Feature construction has no dependency on the current move/time labels."""

import math

import chess
import numpy as np
import torch

PROMOTIONS = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
CONTEXT_SIZE = 15
HISTORY_LENGTH = 16
TIME_EDGES = np.concatenate(([0.0], np.geomspace(1e-4, 1.0, 32))).astype(np.float32)


def canonical_move(move, white):
    return (move.from_square if white else chess.square_mirror(move.from_square),
            move.to_square if white else chess.square_mirror(move.to_square),
            PROMOTIONS.index(move.promotion))


def canonical_uci(move, white):
    start, end, _ = canonical_move(move, white)
    return chess.Move(start, end, promotion=move.promotion).uci()


def _log(value):
    return math.log1p(max(0.0, value or 0.0)) / 10


def context_features(row):
    board = chess.Board(row["fen"])
    clocks = [row.get("self_clock"), row.get("opponent_clock")]
    ratings = [row.get("self_rating"), row.get("opponent_rating")]
    return [
        _log(row["base_seconds"]), _log(row["increment_seconds"]),
        *[_log(x) for x in clocks], *[float(x is not None) for x in clocks],
        *[(x or 0) / 4000 for x in ratings], *[float(x is not None) for x in ratings],
        min(board.fullmove_number / 100, 5), float(board.is_check()),
        float(board.has_kingside_castling_rights(board.turn)),
        float(board.has_queenside_castling_rights(board.turn)),
        board.halfmove_clock / 100,
    ]


def duration_bin(seconds, clock):
    if seconds is None or clock is None or clock <= 0 or not (0 <= seconds <= clock + 1e-6):
        return -1
    fraction = min(1.0, seconds / clock)
    return int(np.clip(np.searchsorted(TIME_EDGES, fraction, side="right") - 1, 0, 31))


def bin_centers():
    centers = np.sqrt(TIME_EDGES[:-1] * TIME_EDGES[1:])
    centers[0] = TIME_EDGES[1] / 2
    return centers


def collate_rows(rows, players):
    """Produce padded legal candidates; labels are returned in separate tensors."""
    mapping = {name: i for i, name in enumerate(players)}
    legal_moves = [list(chess.Board(r["fen"]).legal_moves) for r in rows]
    if any(not moves for moves in legal_moves):
        raise ValueError("Cannot predict a move in a terminal position")
    batch, candidates = len(rows), max(map(len, legal_moves))
    moves = torch.zeros(batch, candidates, 3, dtype=torch.long)
    mask = torch.zeros(batch, candidates, dtype=torch.bool)
    history_ids = torch.zeros(batch, HISTORY_LENGTH, 4, dtype=torch.long)
    history_values = torch.zeros(batch, HISTORY_LENGTH, 2)
    lengths, targets, bins, duration_labels = [], [], [], []
    for i, (row, legal) in enumerate(zip(rows, legal_moves)):
        board = chess.Board(row["fen"])
        for j, move in enumerate(legal):
            moves[i, j] = torch.tensor(canonical_move(move, board.turn))
        mask[i, :len(legal)] = True
        label = chess.Move.from_uci(row["move"]) if row.get("move") else None
        if label is not None and label not in legal:
            raise ValueError(f"Illegal training label {label} for {row['fen']}")
        targets.append(legal.index(label) if label is not None else -1)
        history = row.get("history", [])[-HISTORY_LENGTH:]
        lengths.append(max(1, len(history)))
        for j, event in enumerate(history):
            move = chess.Move.from_uci(event["move"])
            history_ids[i, j] = torch.tensor((*canonical_move(move, board.turn), int(event["white"] == board.turn)))
            elapsed = event.get("elapsed_seconds")
            history_values[i, j] = torch.tensor([_log(elapsed), float(elapsed is not None)])
        elapsed = row.get("elapsed_seconds")
        bins.append(duration_bin(elapsed, row.get("self_clock")))
        duration_labels.append(elapsed if elapsed is not None else 0)
    return {
        "rows": rows, "legal_moves": [[m.uci() for m in ms] for ms in legal_moves],
        "moves": moves, "legal_mask": mask,
        "context": torch.tensor([context_features(row) for row in rows], dtype=torch.float32),
        "players": torch.tensor([mapping[r["player"]] for r in rows]),
        "history_ids": history_ids, "history_values": history_values,
        "history_lengths": torch.tensor(lengths), "target_move": torch.tensor(targets),
        "target_time": torch.tensor(bins), "elapsed": torch.tensor(duration_labels),
        "clock": torch.tensor([r.get("self_clock") or 0.0 for r in rows]),
    }


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
