import json
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from chess_personas.timing_baseline import TimingBaseline
from chess_personas.training import JsonlDataset, evaluate_checkpoint, train


def test_capped_dataset_samples_whole_source_reproducibly(tmp_path):
    path = tmp_path / "positions.jsonl"
    rows = [{"player": "hikaru", "game_id": f"g{i}", "timestamp": i} for i in range(200)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    first = JsonlDataset(path, ["hikaru"], 20, seed=7)
    random.seed(999)
    second = JsonlDataset(path, ["hikaru"], 20, seed=7)
    assert first.offsets == second.offsets
    assert first.offsets != JsonlDataset(path, ["hikaru"], 20, seed=8).offsets
    assert max(first.timestamps) > 150  # A prefix would stop at timestamp 19.
    assert first.available_counts["hikaru"] == 200 and len(first) == 20
    assert len(first.source_game_ids) == 200
    assert len(JsonlDataset(path, ["hikaru"])) == 200


def test_shuffle_covers_each_position_once_independent_of_model_rng():
    def order(extra_random_draws):
        torch.rand(extra_random_draws)
        loader = DataLoader(list(range(71)), batch_size=8, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        return [int(index) for batch in loader for index in batch]
    first = order(1)
    assert first == order(1000)
    assert len(first) == len(set(first)) == 71
    assert first != list(range(71))


def test_timing_baseline_smoothing_sparse_fallback_and_no_label_use():
    rows = [{"player": "hikaru", "base_seconds": 60, "increment_seconds": 0,
             "self_clock": 50, "elapsed_seconds": 2} for _ in range(25)]
    rows.append({**rows[0], "base_seconds": 300, "elapsed_seconds": 9})
    baseline = TimingBaseline.fit(rows)
    observation = {**rows[0], "base_seconds": 180, "elapsed_seconds": 100000}
    first = baseline.predict(observation)
    assert first["group"] == "hikaru"  # Unseen control falls back to player.
    assert first["median_seconds"] == 2
    assert np.all(first["probabilities"] > 0)
    assert sum(first["probabilities"]) == pytest.approx(1)
    restored = TimingBaseline.from_dict(json.loads(json.dumps(baseline.to_dict())))
    second = restored.predict({**observation, "elapsed_seconds": None})
    assert np.array_equal(first["probabilities"], second["probabilities"])
    assert restored.predict({**observation, "self_clock": 1})["median_seconds"] == 1
    assert TimingBaseline.fit([]).predict(observation) is None


def test_exact_update_budget_and_comparable_evaluation(synthetic_dataset, tmp_path):
    torch.set_num_threads(1)
    data, _ = synthetic_dataset
    run = tmp_path / "corrected"
    checkpoint = train(data / "train.jsonl", data / "val.jsonl", run, ["hikaru"], backbone="smoke",
                       epochs=1, batch_size=8, max_updates=7, eval_every=3, train_eval_size=10, device="cpu")
    history = json.loads((run / "metrics.json").read_text())
    summary = json.loads((run / "run-summary.json").read_text())
    audit = json.loads((run / "data-audit.json").read_text())
    assert [item["update"] for item in history] == [3, 6, 7]
    assert summary["completed_updates"] == summary["attempted_updates"] == 7
    assert summary["unique_positions_seen"] == audit["training"]["positions"]
    assert summary["passes_started"] > 1  # Update budget overrides epochs=1.
    assert torch.load(run / "last.pt", weights_only=True)["update"] == 7
    assert torch.load(checkpoint, weights_only=True)["update"] == summary["best_update"]
    assert (run / "best-move.pt").exists()
    assert history[-1]["train_evaluation"]["overall"]["positions"] == 10
    assert all(np.isfinite(item["train_move_loss"]) and np.isfinite(item["train_time_loss"]) for item in history)
    predictions = tmp_path / "predictions.jsonl"
    report = evaluate_checkpoint(checkpoint, data / "val.jsonl", device="cpu",
                                 baseline_path=run / "timing-baseline.json", predictions_path=predictions)
    records = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert len(records) == audit["validation"]["positions"] == report["overall"]["positions"]
    timed = [row for row in records if "time_nll" in row]
    assert report["overall"]["time_nll"] == pytest.approx(np.mean([row["time_nll"] for row in timed]))
    assert report["overall"]["time_interval_80_width_seconds"] > 0
    assert report["overall"]["baseline_timed_positions"] == len(timed)
    assert all(row["game_id"] and row["time_control"] == "180+2" for row in records)
    assert all(0 <= row["time_median_seconds"] <= row["self_clock"] for row in timed)


def test_validation_never_capped(synthetic_dataset, tmp_path):
    torch.set_num_threads(1)
    data, _ = synthetic_dataset
    run = tmp_path / "capped"
    train(data / "train.jsonl", data / "val.jsonl", run, ["hikaru"], backbone="smoke",
          max_per_player=1, max_updates=1, train_eval_size=1, device="cpu")
    audit = json.loads((run / "data-audit.json").read_text())
    assert audit["training"]["positions"] == 1
    assert audit["validation"]["positions"] == len(JsonlDataset(data / "val.jsonl", ["hikaru"])) > 1
