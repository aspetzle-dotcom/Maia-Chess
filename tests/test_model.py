import json

import chess
import pytest

torch = pytest.importorskip("torch")

from chess_personas.backbones import SmokeBackbone
from chess_personas.data import game_rows
from chess_personas.features import canonical_move, collate_rows, duration_bin
from chess_personas.model import PersonaModel, losses
from chess_personas.training import train, load_model, predict, evaluate_checkpoint


def test_black_promotions_and_fraction_bounds():
    assert canonical_move(chess.Move.from_uci("a2b1n"), False) == (chess.A7, chess.B8, 1)
    assert duration_bin(0, 10) == 0
    assert duration_bin(10, 10) == 31
    assert duration_bin(11, 10) == -1
    assert duration_bin(None, 10) == -1


def test_gradients_masks_and_frozen_backbone(synthetic_pgn):
    torch.manual_seed(7)
    rows = game_rows(synthetic_pgn, {"hikaru", "fabianocaruana"})
    batch = collate_rows(rows, ["hikaru", "fabianocaruana"])
    model = PersonaModel(SmokeBackbone(), ["hikaru", "fabianocaruana"])
    output = model(batch)
    assert torch.allclose(output["move_logits"], output["base_logits"])
    assert torch.equal(output["move_logits"].softmax(-1)[~batch["legal_mask"]], torch.zeros_like(output["move_logits"][~batch["legal_mask"]]))
    objective = losses(output, batch)
    assert torch.isfinite(objective["loss"])
    objective["loss"].backward()
    assert model.policy.weight.grad.abs().sum() > 0
    assert model.time_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_all_missing_times_keep_policy_training(synthetic_pgn):
    rows = game_rows(synthetic_pgn, {"hikaru"})
    for row in rows:
        row["elapsed_seconds"] = None
    batch = collate_rows(rows, ["hikaru"])
    model = PersonaModel(SmokeBackbone(), ["hikaru"])
    objective = losses(model(batch), batch)
    assert objective["time_loss"] == 0
    objective["loss"].backward()
    assert model.policy.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_backbone_keeps_policy_mask_finite(synthetic_pgn, dtype):
    class LowPrecisionBackbone(SmokeBackbone):
        def forward(self, batch):
            squares, logits = super().forward(batch)
            return squares.to(dtype), logits.to(dtype)

    rows = game_rows(synthetic_pgn, {"hikaru", "fabianocaruana"})
    batch = collate_rows(rows, ["hikaru", "fabianocaruana"])
    assert (~batch["legal_mask"]).any()
    model = PersonaModel(LowPrecisionBackbone(), ["hikaru", "fabianocaruana"])
    output = model(batch)
    assert output["move_logits"].dtype == torch.float32
    assert torch.isfinite(output["move_logits"]).all()
    assert (output["move_logits"].softmax(-1)[~batch["legal_mask"]] == 0).all()
    objective = losses(output, batch)
    assert torch.isfinite(objective["loss"])
    objective["loss"].backward()
    assert torch.isfinite(model.policy.weight.grad).all()
    assert model.policy.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("players", [["hikaru"], ["fabianocaruana"], ["hikaru", "fabianocaruana"]])
def test_training_checkpoint_evaluation_and_label_free_prediction(synthetic_dataset, tmp_path, players):
    torch.set_num_threads(1)
    directory, _ = synthetic_dataset
    path = train(directory / "train.jsonl", directory / "val.jsonl", tmp_path / "run",
                 players, backbone="smoke", epochs=1, batch_size=8, device="cpu")
    model, device = load_model(path, "cpu")
    assert model.player_names == players
    assert len(model.adapters) == len(players)
    assert model.player_embedding.num_embeddings == len(players)
    report = evaluate_checkpoint(path, directory / "test.jsonl", device="cpu")
    test_rows = [json.loads(line) for line in (directory / "test.jsonl").read_text().splitlines()]
    selected_rows = [row for row in test_rows if row["player"] in players]
    assert report["overall"]["positions"] == len(selected_rows)
    assert {key for key in report if key != "overall" and "/" not in key} == set(players)
    assert report["overall"]["timed_positions"] > 0
    row = next(row for row in selected_rows if row["self_clock"] is not None)
    first = predict(path, row, device="cpu")
    row["move"] = "a1a8"  # Impossible label must never influence inference.
    row["elapsed_seconds"] = 999999
    second = predict(path, row, device="cpu")
    assert first == second
    board = chess.Board(row["fen"])
    assert all(chess.Move.from_uci(m["uci"]) in board.legal_moves for m in first["moves"])
    assert 0 <= first["marginal_time_mean_seconds"] <= row["self_clock"]
    assert abs(sum(first["marginal_time_probabilities"]) - 1) < 1e-5
    if len(players) == 1:
        without_player = {key: value for key, value in row.items() if key != "player"}
        assert predict(path, without_player, device="cpu") == first
        other_player = next(name for name in ("hikaru", "fabianocaruana") if name not in players)
        with pytest.raises(ValueError, match="Checkpoint supports"):
            predict(path, {**row, "player": other_player}, device="cpu")
