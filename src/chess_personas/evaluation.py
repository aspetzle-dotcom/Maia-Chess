"""Common train/validation evaluator and game-identifiable prediction records."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .features import TIME_EDGES, to_device
from .timing_baseline import clock_band, control_key


@torch.no_grad()
def evaluate(model, loader, device, *, timing_baseline=None, predictions_path=None):
    model.eval()
    buckets = defaultdict(Counter)
    handle = None
    if predictions_path:
        Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
        handle = Path(predictions_path).open("w", encoding="utf-8")
    try:
        for batch in loader:
            batch = to_device(batch, device)
            result = model(batch)
            targets = batch["target_move"]
            policy = result["move_logits"].softmax(-1)
            conditional = result["time_logits"].softmax(-1)
            marginal = (policy[..., None] * conditional).sum(1)
            chosen = conditional[torch.arange(len(targets), device=device), targets]
            # Transfer once per batch, avoiding scalar GPU synchronization per metric.
            move_nll = F.cross_entropy(result["move_logits"], targets, reduction="none").cpu().numpy()
            base_nll = F.cross_entropy(result["base_logits"], targets, reduction="none").cpu().numpy()
            correct = (result["move_logits"].argmax(-1) == targets).cpu().numpy()
            base_correct = (result["base_logits"].argmax(-1) == targets).cpu().numpy()
            probabilities, conditional_probabilities = marginal.cpu().numpy(), chosen.cpu().numpy()
            labels = batch["target_time"].cpu().numpy()
            centers = model.time_centers.cpu().numpy()
            for i, row in enumerate(batch["rows"]):
                record = {"player": row["player"], "game_id": row["game_id"],
                          "timestamp": row["timestamp"], "ply": row.get("ply"), "fen": row["fen"],
                          "time_class": row.get("time_class", "unknown"),
                          "time_control": control_key(row), "clock_band": clock_band(row),
                          "move_nll": float(move_nll[i]), "base_move_nll": float(base_nll[i]),
                          "move_correct": int(correct[i]), "base_move_correct": int(base_correct[i])}
                timed = int(labels[i]) >= 0
                values = {"positions": 1, "timed_positions": int(timed),
                          "move_nll_sum": record["move_nll"], "base_nll_sum": record["base_move_nll"],
                          "correct": record["move_correct"], "base_correct": record["base_move_correct"]}
                if timed:
                    label, clock, elapsed = int(labels[i]), row["self_clock"], row["elapsed_seconds"]
                    probs = probabilities[i]
                    cdf = np.cumsum(probs)
                    low, median, high = np.minimum(np.searchsorted(cdf, [0.1, 0.5, 0.9]), 31)
                    mean_seconds = float(np.dot(probs, centers)) * clock
                    median_seconds = float(centers[median]) * clock
                    interval = [float(TIME_EDGES[low]) * clock, float(TIME_EDGES[high + 1]) * clock]
                    record.update({"elapsed_seconds": elapsed, "self_clock": clock,
                                   "time_nll": float(-np.log(max(float(probs[label]), 1e-12))),
                                   "conditional_time_nll": float(-np.log(max(float(conditional_probabilities[i, label]), 1e-12))),
                                   "time_mean_seconds": mean_seconds, "time_median_seconds": median_seconds,
                                   "time_absolute_error_seconds": abs(mean_seconds - elapsed),
                                   "time_median_absolute_error_seconds": abs(median_seconds - elapsed),
                                   "time_interval_80_seconds": interval,
                                   "time_interval_80_width_seconds": interval[1] - interval[0],
                                   "time_interval_80_covered": int(interval[0] <= elapsed <= interval[1])})
                    values.update({"time_nll_sum": record["time_nll"], "conditional_time_nll_sum": record["conditional_time_nll"],
                                   "time_mae_sum": record["time_absolute_error_seconds"],
                                   "time_median_mae_sum": record["time_median_absolute_error_seconds"],
                                   "covered": record["time_interval_80_covered"],
                                   "width_sum": record["time_interval_80_width_seconds"]})
                    baseline = timing_baseline.predict(row) if timing_baseline else None
                    if baseline is not None:
                        record.update({"baseline_time_nll": float(-np.log(baseline["probabilities"][label])),
                                       "baseline_time_median_seconds": baseline["median_seconds"],
                                       "baseline_time_absolute_error_seconds": abs(baseline["median_seconds"] - elapsed),
                                       "baseline_group": baseline["group"]})
                        values.update({"baseline_positions": 1, "baseline_nll_sum": record["baseline_time_nll"],
                                       "baseline_mae_sum": record["baseline_time_absolute_error_seconds"]})
                player = row["player"]
                for key in ("overall", player, player + "/" + record["time_class"],
                            player + "/control=" + record["time_control"],
                            player + "/control=" + record["time_control"] + "/clock=" + record["clock_band"]):
                    buckets[key].update(values)
                if handle:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    finally:
        if handle:
            handle.close()
    report = {}
    for key, count in buckets.items():
        n, timed, baseline_n = count["positions"], count["timed_positions"], count["baseline_positions"]
        report[key] = {"positions": n, "timed_positions": timed,
                       "move_accuracy": count["correct"] / n, "move_nll": count["move_nll_sum"] / n,
                       "base_move_accuracy": count["base_correct"] / n, "base_move_nll": count["base_nll_sum"] / n,
                       "baseline_timed_positions": baseline_n,
                       "baseline_time_nll": count["baseline_nll_sum"] / baseline_n if baseline_n else None,
                       "baseline_time_median_mae_seconds": count["baseline_mae_sum"] / baseline_n if baseline_n else None}
        for name, total in {"time_nll": "time_nll_sum", "conditional_time_nll": "conditional_time_nll_sum",
                            "time_mae_seconds": "time_mae_sum", "time_median_mae_seconds": "time_median_mae_sum",
                            "time_interval_80_coverage": "covered", "time_interval_80_width_seconds": "width_sum"}.items():
            report[key][name] = count[total] / timed if timed else None
    return report
