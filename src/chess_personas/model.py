"""Residual legal-move policy and move-conditioned duration histogram."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .features import CONTEXT_SIZE, bin_centers


class PersonaModel(nn.Module):
    def __init__(self, backbone, players, hidden=128):
        super().__init__()
        self.backbone = backbone
        self.player_names = list(players)
        self.hidden = hidden
        self.player_embedding = nn.Embedding(len(players), 32)
        self.square_embedding = nn.Embedding(64, 16)
        self.promotion_embedding = nn.Embedding(5, 8)
        self.history_gru = nn.GRU(43, 64, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(backbone.dim + CONTEXT_SIZE + 32 + 64, hidden), nn.GELU())
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, hidden)) for _ in players
        ])
        self.candidate = nn.Sequential(nn.Linear(backbone.dim * 2 + 8 + hidden, hidden), nn.GELU())
        self.policy = nn.Linear(hidden, 1)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        self.time_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 32))
        self.register_buffer("time_centers", torch.tensor(bin_centers()))

    def forward(self, batch):
        squares, base_logits = self.backbone(batch)
        ids, values = batch["history_ids"], batch["history_values"]
        history = torch.cat([self.square_embedding(ids[..., 0]), self.square_embedding(ids[..., 1]),
                             self.promotion_embedding(ids[..., 2]), ids[..., 3:].float(), values], dim=-1)
        packed = pack_padded_sequence(history, batch["history_lengths"].cpu(), batch_first=True, enforce_sorted=False)
        _, history_state = self.history_gru(packed)
        context = self.fusion(torch.cat([squares.mean(dim=1), batch["context"],
                               self.player_embedding(batch["players"]), history_state[-1]], dim=-1))
        adjusted = context.clone()
        for index, adapter in enumerate(self.adapters):
            selected = batch["players"] == index
            if selected.any():
                adjusted[selected] = context[selected] + adapter(context[selected])
        moves = batch["moves"]
        start = squares.gather(1, moves[..., 0, None].expand(-1, -1, squares.size(-1)))
        end = squares.gather(1, moves[..., 1, None].expand(-1, -1, squares.size(-1)))
        candidate = self.candidate(torch.cat([start, end, self.promotion_embedding(moves[..., 2]),
                                              adjusted[:, None, :].expand(-1, moves.size(1), -1)], dim=-1))
        # Finite padding avoids 0 * infinity in KL regularization.
        base_logits = base_logits.masked_fill(~batch["legal_mask"], -1e9)
        logits = (base_logits + self.policy(candidate).squeeze(-1)).masked_fill(~batch["legal_mask"], -1e9)
        return {"move_logits": logits, "base_logits": base_logits, "time_logits": self.time_head(candidate)}


def losses(output, batch, time_weight=0.5, kl_weight=0.02):
    policy = F.cross_entropy(output["move_logits"], batch["target_move"])
    log_policy = F.log_softmax(output["move_logits"], dim=-1)
    base_probs = F.softmax(output["base_logits"], dim=-1)
    regularization = F.kl_div(log_policy, base_probs, reduction="batchmean")
    timing_valid = batch["target_time"] >= 0
    chosen_times = output["time_logits"][torch.arange(len(batch["target_move"]), device=log_policy.device), batch["target_move"]]
    timing = (F.cross_entropy(chosen_times[timing_valid], batch["target_time"][timing_valid])
              if timing_valid.any() else chosen_times.sum() * 0)
    return {"loss": policy + time_weight * timing + kl_weight * regularization,
            "move_loss": policy, "time_loss": timing, "kl": regularization}
