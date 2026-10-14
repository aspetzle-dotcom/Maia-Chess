"""Optional contract tests use the pinned upstream source without downloading weights."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("maia3")

from chess_personas.backbones import MaiaBackbone
from chess_personas.data import game_rows
from chess_personas.features import collate_rows
from chess_personas.model import PersonaModel, losses


def test_real_maia_contract_and_no_label_dependency(synthetic_pgn):
    torch.set_num_threads(1)
    rows = game_rows(synthetic_pgn, {"hikaru", "fabianocaruana"})
    batch = collate_rows(rows, ["hikaru", "fabianocaruana"])
    backbone = MaiaBackbone(load_weights=False)
    squares, logits = backbone(batch)
    assert squares.shape == (6, 64, 256)
    assert logits.shape == batch["legal_mask"].shape
    rows[2]["elapsed_seconds"] = 999
    new_batch = collate_rows(rows, ["hikaru", "fabianocaruana"])
    new_squares, new_logits = backbone(new_batch)
    assert torch.equal(squares, new_squares)
    assert torch.equal(logits, new_logits)
    model = PersonaModel(backbone, ["hikaru", "fabianocaruana"])
    model.train()
    assert not model.backbone.model.training
    objective = losses(model(batch), batch)
    objective["loss"].backward()
    assert torch.isfinite(objective["loss"])
    assert all(p.grad is None for p in backbone.parameters())
