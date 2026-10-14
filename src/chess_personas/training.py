"""Small-GPU training and evaluation. No cloud job is launched automatically."""

import json
import random
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .backbones import build_backbone
from .features import TIME_EDGES, collate_rows, to_device
from .model import PersonaModel, losses


class JsonlDataset(Dataset):
    """Index byte offsets, retaining neither board strings nor full games in RAM."""
    def __init__(self, path, players, max_per_player=0):
        self.path = Path(path)
        self.offsets, self.player_ids = [], []
        self.counts = Counter()
        self.game_ids, self.timestamps = set(), []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row = json.loads(line)
                name = row["player"]
                if name not in players or (max_per_player and self.counts[name] >= max_per_player):
                    continue
                self.offsets.append(offset)
                self.player_ids.append(name)
                self.counts[name] += 1
                self.game_ids.add(row["game_id"])
                self.timestamps.append(row["timestamp"])
        if not self.offsets:
            raise ValueError(f"No usable positions for {players} in {path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())


def choose_device(value="auto"):
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    buckets = defaultdict(Counter)
    edges = torch.tensor(TIME_EDGES, device=device)
    for batch in loader:
        batch = to_device(batch, device)
        result = model(batch)
        targets = batch["target_move"]
        policy = result["move_logits"].softmax(-1)
        move_nll = F.cross_entropy(result["move_logits"], targets, reduction="none")
        base_nll = F.cross_entropy(result["base_logits"], targets, reduction="none")
        correct = result["move_logits"].argmax(-1) == targets
        base_correct = result["base_logits"].argmax(-1) == targets
        # Marginal timing is the valid pre-move prediction; conditional timing
        # is also reported, explicitly labeled as knowing the played move.
        conditional = result["time_logits"].softmax(-1)
        marginal = (policy[..., None] * conditional).sum(1)
        mean_seconds = (marginal * model.time_centers).sum(-1) * batch["clock"]
        for i, row in enumerate(batch["rows"]):
            for key in ("overall", row["player"], row["player"] + "/" + row.get("time_class", "unknown")):
                metrics = buckets[key]
                metrics["positions"] += 1
                metrics["move_nll_sum"] += float(move_nll[i])
                metrics["base_nll_sum"] += float(base_nll[i])
                metrics["correct"] += int(correct[i])
                metrics["base_correct"] += int(base_correct[i])
                label = int(batch["target_time"][i])
                if label >= 0:
                    metrics["timed_positions"] += 1
                    metrics["time_nll_sum"] += float(-marginal[i, label].clamp_min(1e-12).log())
                    metrics["conditional_time_nll_sum"] += float(-conditional[i, targets[i], label].clamp_min(1e-12).log())
                    metrics["time_mae_sum"] += abs(float(mean_seconds[i] - batch["elapsed"][i]))
                    cdf = marginal[i].cumsum(0)
                    low = int(torch.searchsorted(cdf, torch.tensor(0.1, device=device)).clamp(max=31))
                    high = int(torch.searchsorted(cdf, torch.tensor(0.9, device=device)).clamp(max=31))
                    fraction = batch["elapsed"][i] / batch["clock"][i]
                    metrics["covered"] += int(edges[low] <= fraction <= edges[high + 1])
    report = {}
    for key, count in buckets.items():
        n, timed = count["positions"], count["timed_positions"]
        report[key] = {
            "positions": n, "timed_positions": timed,
            "move_accuracy": count["correct"] / n, "move_nll": count["move_nll_sum"] / n,
            "base_move_accuracy": count["base_correct"] / n, "base_move_nll": count["base_nll_sum"] / n,
            "time_nll": count["time_nll_sum"] / timed if timed else None,
            "conditional_time_nll": count["conditional_time_nll_sum"] / timed if timed else None,
            "time_mae_seconds": count["time_mae_sum"] / timed if timed else None,
            "time_interval_80_coverage": count["covered"] / timed if timed else None,
        }
    return report


def train(train_path, val_path, output, players, *, backbone="maia3", model_name="maia3-5m",
          checkpoint=None, epochs=3, batch_size=32, learning_rate=3e-4, max_per_player=10000,
          device="auto", seed=42, time_weight=0.5, kl_weight=0.02):
    if epochs < 1 or batch_size < 1 or max_per_player < 0 or learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    if len(set(players)) != len(players) or not players:
        raise ValueError("Players must be a nonempty list of unique usernames")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(device)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "best.pt").exists():
        raise FileExistsError("Use a new run directory; best.pt already exists")
    training = JsonlDataset(train_path, players, max_per_player)
    validation = JsonlDataset(val_path, players, max_per_player)
    if training.game_ids & validation.game_ids:
        raise ValueError("Training and validation contain overlapping games")
    if max(training.timestamps) >= min(validation.timestamps):
        raise ValueError("Validation must be chronologically later than training")
    missing = set(players) - set(training.counts)
    if missing:
        raise ValueError(f"No training data for {sorted(missing)}")
    missing = set(players) - set(validation.counts)
    if missing:
        raise ValueError(f"No validation data for {sorted(missing)}; collect more recent games or adjust splits")
    weights = [1 / training.counts[name] for name in training.player_ids]
    sampler = WeightedRandomSampler(weights, len(training), replacement=True)
    collate = partial(collate_rows, players=players)
    train_loader = DataLoader(training, batch_size=batch_size, sampler=sampler, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate)
    encoder = build_backbone(backbone, model_name=model_name, checkpoint=checkpoint, device=device)
    model = PersonaModel(encoder, players).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    config = {"players": players, "backbone": backbone, "model_name": model_name,
              "hidden": model.hidden, "seed": seed, "batch_size": batch_size,
              "learning_rate": learning_rate, "time_weight": time_weight, "kl_weight": kl_weight,
              "max_per_player": max_per_player, "train_path": str(train_path), "val_path": str(val_path),
              "maia_source_revision": "1e13597c42d4858b7cfd7cfdae01e297263364b2",
              "duration_representation": "32_log_fraction_of_remaining_clock_bins"}
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        total, examples = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=device.startswith("cuda")):
                output_values = model(batch)
                objective = losses(output_values, batch, time_weight, kl_weight)
            if not torch.isfinite(objective["loss"]):
                raise FloatingPointError("Training loss is not finite")
            scaler.scale(objective["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            n = len(batch["rows"])
            total += float(objective["loss"].detach()) * n
            examples += n
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step} loss={total / examples:.4f}", flush=True)
        metrics = evaluate(model, val_loader, device)
        # Equal player weighting matches the balanced training objective.
        score = sum(metrics[p]["move_nll"] + time_weight * (metrics[p]["time_nll"] or 0) for p in players) / len(players)
        history.append({"epoch": epoch + 1, "train_loss": total / examples, "selection_score": score, "validation": metrics})
        print(json.dumps(history[-1]), flush=True)
        if score < best:
            best = score
            torch.save({"format_version": 1, "config": config,
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": epoch + 1, "validation": metrics}, directory / "best.pt")
        (directory / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return directory / "best.pt"


def load_model(path, device="auto"):
    device = choose_device(device)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = saved["config"]
    backbone = build_backbone(config["backbone"], model_name=config["model_name"], device=device, load_weights=False)
    model = PersonaModel(backbone, config["players"], hidden=config["hidden"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, device


def evaluate_checkpoint(checkpoint, data, *, device="auto", batch_size=32, max_per_player=0):
    model, device = load_model(checkpoint, device)
    dataset = JsonlDataset(data, model.player_names, max_per_player)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_rows, players=model.player_names))
    return evaluate(model, loader, device)


@torch.no_grad()
def predict(checkpoint, row, *, device="auto", top_k=5):
    model, device = load_model(checkpoint, device)
    # Labels are removed even if a prepared training record was supplied.
    row = {k: v for k, v in row.items() if k not in ("move", "elapsed_seconds")}
    batch = to_device(collate_rows([row], model.player_names), device)
    result = model(batch)
    move_probs = result["move_logits"][0].softmax(-1)
    times = result["time_logits"][0].softmax(-1)
    marginal = (move_probs[:, None] * times).sum(0)
    clock = row.get("self_clock")
    values, indices = move_probs.topk(min(top_k, len(batch["legal_moves"][0])))
    return {"player": row["player"],
            "marginal_time_mean_seconds": float((marginal * model.time_centers).sum()) * clock if clock else None,
            "duration_bin_edges_fraction_of_clock": TIME_EDGES.tolist(),
            "marginal_time_probabilities": marginal.tolist() if clock else None,
            "moves": [{"uci": batch["legal_moves"][0][int(index)], "probability": float(probability),
                       "conditional_time_mean_seconds": float((times[index] * model.time_centers).sum()) * clock if clock else None}
                      for probability, index in zip(values, indices)]}
