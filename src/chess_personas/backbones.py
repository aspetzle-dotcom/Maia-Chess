"""Optional frozen Maia integration, plus an explicitly untrained smoke backbone."""

from types import SimpleNamespace

import chess
import torch
from torch import nn

from .features import canonical_uci


class MaiaBackbone(nn.Module):
    def __init__(self, model_name="maia3-5m", checkpoint=None, device="cpu", load_weights=True):
        super().__init__()
        try:
            from maia3.model_registry import resolve_model_spec, resolve_checkpoint_path
            from maia3.models import MAIA3Model
            from maia3.utils import get_all_possible_moves
        except ImportError as exc:
            raise RuntimeError("Install requirements-maia.txt to enable the pretrained Maia backbone") from exc
        spec = resolve_model_spec(model_name)
        self.cfg = SimpleNamespace(**spec.config)
        self.model = MAIA3Model(self.cfg)
        if load_weights:
            path = checkpoint or resolve_checkpoint_path(spec)
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = state.get("model_state_dict", state)
            state = {k.replace("smolgen", "gab"): v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.dim = self.cfg.dim_vit
        self.vocab = {move: i for i, move in enumerate(get_all_possible_moves())}
        self.to(device)
        self.eval()

    def train(self, mode=True):
        # Calling train() on the parent must never enable backbone dropout.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, batch):
        from maia3.dataset import tokenize_board, get_historical_tokens
        device = next(self.model.parameters()).device
        tokens = []
        for row in batch["rows"]:
            fens = row.get("board_history", [row["fen"]])[-self.cfg.history:]
            if not fens or fens[-1] != row["fen"]:
                raise ValueError("Board history must end at the pre-move position")
            history = [tokenize_board(chess.Board(fen)) for fen in fens]
            # The upstream API accepts a label slot. Always pass zero here.
            tokens.append(get_historical_tokens(history, self.cfg, row["base_seconds"],
                          row["increment_seconds"], row.get("self_clock") or 0.0, 0.0))
        self_elos = torch.tensor([r.get("self_rating") or 2500 for r in batch["rows"]], device=device)
        oppo_elos = torch.tensor([r.get("opponent_rating") or 2500 for r in batch["rows"]], device=device)
        captured = []
        hook = self.model.transformer.register_forward_hook(lambda module, args, output: captured.append(output))
        try:
            logits, _, _ = self.model(torch.stack(tokens).to(device), self_elos, oppo_elos)
        finally:
            hook.remove()
        indices = torch.zeros_like(batch["moves"][..., 0], device=device)
        for i, (row, moves) in enumerate(zip(batch["rows"], batch["legal_moves"])):
            white = chess.Board(row["fen"]).turn
            indices[i, :len(moves)] = torch.tensor([
                self.vocab[canonical_uci(chess.Move.from_uci(m), white)] for m in moves
            ], device=device)
        return captured[0], logits.gather(1, indices)


class SmokeBackbone(nn.Module):
    """Random frozen features for pipeline tests. This has no learned chess skill."""
    dim = 32

    def __init__(self):
        super().__init__()
        self.piece = nn.Embedding(13, self.dim)
        self.square = nn.Embedding(64, self.dim)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, batch):
        device = self.piece.weight.device
        pieces = []
        for row in batch["rows"]:
            board = chess.Board(row["fen"])
            if not board.turn:
                board = board.mirror()
            pieces.append([0 if (p := board.piece_at(sq)) is None else p.piece_type + (0 if p.color else 6)
                           for sq in chess.SQUARES])
        tokens = self.piece(torch.tensor(pieces, device=device)) + self.square(torch.arange(64, device=device))
        return tokens, torch.zeros(batch["moves"].shape[:2], device=device)


def build_backbone(kind="maia3", *, model_name="maia3-5m", checkpoint=None, device="cpu", load_weights=True):
    if kind == "smoke":
        return SmokeBackbone().to(device)
    if kind != "maia3":
        raise ValueError(f"Unknown backbone: {kind}")
    return MaiaBackbone(model_name, checkpoint, device, load_weights)
