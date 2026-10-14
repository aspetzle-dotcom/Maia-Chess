"""Small-GPU training and evaluation. No cloud job is launched automatically."""

import json
import hashlib
import random
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .backbones import build_backbone
from .features import TIME_EDGES, collate_rows, to_device
from .model import PersonaModel, losses
from .evaluation import evaluate
from .timing_baseline import TimingBaseline, control_key


class JsonlDataset(Dataset):
    """Index byte offsets, retaining neither board strings nor full games in RAM."""
    def __init__(self, path, players, max_per_player=0, seed=42):
        self.path = Path(path)
        if max_per_player < 0:
            raise ValueError("Position cap must be nonnegative")
        rng = random.Random(seed)
        selected = {name: [] for name in players}
        self.available_counts = Counter()
        self.source_game_ids, self.source_timestamps = set(), []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row = json.loads(line)
                name = row["player"]
                if name not in players:
                    continue
                self.available_counts[name] += 1
                self.source_game_ids.add(row["game_id"])
                self.source_timestamps.append(row["timestamp"])
                item = (offset, name, row["game_id"], row["timestamp"])
                if not max_per_player or len(selected[name]) < max_per_player:
                    selected[name].append(item)
                else:
                    # Reservoir sampling covers the whole file, not its prefix.
                    index = rng.randrange(self.available_counts[name])
                    if index < max_per_player:
                        selected[name][index] = item
        records = sorted(item for values in selected.values() for item in values)
        self.offsets = [item[0] for item in records]
        self.player_ids = [item[1] for item in records]
        self.counts = Counter(self.player_ids)
        self.game_ids = {item[2] for item in records}
        self.timestamps = [item[3] for item in records]
        if not self.offsets:
            raise ValueError(f"No usable positions for {players} in {path}")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
            row["_source_offset"] = self.offsets[index]
            return row


def choose_device(value="auto"):
    return "cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_audit(dataset):
    controls = Counter(control_key(row) for row in dataset)
    return {"path": str(dataset.path), "sha256": file_sha256(dataset.path),
            "positions": len(dataset), "available_by_player": dict(dataset.available_counts),
            "selected_by_player": dict(dataset.counts), "games": len(dataset.game_ids),
            "timestamp_min": min(dataset.timestamps), "timestamp_max": max(dataset.timestamps),
            "positions_by_time_control": dict(controls),
            "selected_offsets_sha256": hashlib.sha256(json.dumps(dataset.offsets).encode()).hexdigest()}


def train(train_path, val_path, output, players, *, backbone="maia3", model_name="maia3-5m",
          checkpoint=None, epochs=3, batch_size=32, learning_rate=3e-4, max_per_player=0,
          device="auto", seed=42, time_weight=0.5, kl_weight=0.02,
          max_updates=None, eval_every=250, train_eval_size=1024):
    if (epochs < 1 or batch_size < 1 or max_per_player < 0 or learning_rate <= 0
            or eval_every < 1 or train_eval_size < 1 or (max_updates is not None and max_updates < 1)):
        raise ValueError("Invalid training configuration")
    if len(set(players)) != len(players) or not players:
        raise ValueError("Players must be a nonempty list of unique usernames")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(device)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("best.pt", "last.pt", "config.json")):
        raise FileExistsError("Use a new run directory; run files already exist")
    training = JsonlDataset(train_path, players, max_per_player, seed)
    validation = JsonlDataset(val_path, players)  # Never cap the validation set.
    if training.source_game_ids & validation.source_game_ids:
        raise ValueError("Training and validation contain overlapping games")
    if max(training.source_timestamps) >= min(validation.source_timestamps):
        raise ValueError("Validation must be chronologically later than training")
    for label, dataset in (("training", training), ("validation", validation)):
        missing = set(players) - set(dataset.counts)
        if missing:
            raise ValueError(f"No {label} data for {sorted(missing)}")
    collate = partial(collate_rows, players=players)
    # Dedicated RNGs keep initialization and evaluation from changing batch order.
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=batch_size, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed + 1))
    subset_indices = sorted(random.Random(seed + 2).sample(range(len(training)), min(train_eval_size, len(training))))
    train_eval_loader = DataLoader(Subset(training, subset_indices), batch_size=batch_size, collate_fn=collate,
                                   generator=torch.Generator().manual_seed(seed + 3))
    baseline = TimingBaseline.fit(training)
    (directory / "timing-baseline.json").write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
    audit = {"training": dataset_audit(training), "validation": dataset_audit(validation),
             "train_eval_indices": subset_indices, "train_eval_positions": len(subset_indices)}
    (directory / "data-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    encoder = build_backbone(backbone, model_name=model_name, checkpoint=checkpoint, device=device)
    model = PersonaModel(encoder, players).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    budget = max_updates if max_updates is not None else epochs * len(train_loader)
    config = {"players": players, "backbone": backbone, "model_name": model_name,
              "hidden": model.hidden, "seed": seed, "batch_size": batch_size,
              "learning_rate": learning_rate, "time_weight": time_weight, "kl_weight": kl_weight,
              "max_per_player": max_per_player, "train_path": str(train_path), "val_path": str(val_path),
              "max_updates": max_updates, "update_budget": budget, "eval_every": eval_every,
              "train_eval_size": train_eval_size, "sampling": "shuffle_without_replacement",
              "selection_metric": "move_nll_plus_weighted_marginal_time_nll",
              "maia_source_revision": "1e13597c42d4858b7cfd7cfdae01e297263364b2",
              "duration_representation": "32_log_fraction_of_remaining_clock_bins"}
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best, best_move = float("inf"), float("inf")
    history, seen = [], set()
    order_digest = hashlib.sha256()
    update, epoch, skipped, attempts = 0, 0, 0, 0
    examples_seen = 0
    window = Counter()
    started = time.monotonic()
    print(json.dumps({"training_positions": len(training), "validation_positions": len(validation),
                      "update_budget": budget, "sampling": config["sampling"]}), flush=True)
    while update < budget:
        epoch += 1
        for batch in train_loader:
            model.train()
            attempts += 1
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
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
                if skipped > 100:
                    raise FloatingPointError("Too many AMP-skipped updates")
                continue  # The requested budget counts actual optimizer updates.
            update += 1
            n = len(batch["rows"])
            timed = int((batch["target_time"] >= 0).sum())
            examples_seen += n
            window["positions"] += n
            window["timed_positions"] += timed
            for name in ("loss", "move_loss", "kl"):
                window[name] += float(objective[name].detach()) * n
            window["time_loss"] += float(objective["time_loss"].detach()) * timed
            for row in batch["rows"]:
                offset = row["_source_offset"]
                seen.add(offset)
                order_digest.update(f"{offset},".encode())
            if update % 100 == 0:
                print(f"update={update}/{budget} pass={epoch} loss={window['loss'] / window['positions']:.4f}", flush=True)
            due = (update % eval_every == 0 if max_updates is not None else update % len(train_loader) == 0)
            if due or update == budget:
                metrics = evaluate(model, val_loader, device, timing_baseline=baseline,
                                   predictions_path=directory / "predictions" / f"validation-{update:06d}.jsonl")
                train_metrics = evaluate(model, train_eval_loader, device, timing_baseline=baseline)
                score = sum(metrics[p]["move_nll"] + time_weight * (metrics[p]["time_nll"] or 0) for p in players) / len(players)
                move_score = sum(metrics[p]["move_nll"] for p in players) / len(players)
                entry = {"epoch": epoch, "update": update, "examples_seen": examples_seen,
                         "unique_positions_seen": len(seen), "elapsed_seconds": time.monotonic() - started,
                         "train_loss": window["loss"] / window["positions"],
                         "train_move_loss": window["move_loss"] / window["positions"],
                         "train_time_loss": window["time_loss"] / window["timed_positions"] if window["timed_positions"] else None,
                         "train_kl": window["kl"] / window["positions"],
                         "selection_score": score, "validation": metrics, "train_evaluation": train_metrics}
                history.append(entry)
                print(json.dumps({"update": update, "train_loss": entry["train_loss"],
                                  "train_move_loss": entry["train_move_loss"], "train_time_loss": entry["train_time_loss"],
                                  "selection_score": score, "validation": metrics["overall"],
                                  "train_evaluation": train_metrics["overall"]}), flush=True)
                saved = {"format_version": 1, "config": config,
                         "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                         "epoch": epoch, "update": update, "validation": metrics}
                if score < best:
                    best = score
                    torch.save(saved, directory / "best.pt")
                if move_score < best_move:
                    best_move = move_score
                    torch.save(saved, directory / "best-move.pt")
                if update == budget:
                    torch.save(saved, directory / "last.pt")
                (directory / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                window.clear()
            if update == budget:
                break
    summary = {"completed_updates": update, "attempted_updates": attempts, "amp_skipped_updates": skipped,
               "passes_started": epoch, "training_positions": len(training), "unique_positions_seen": len(seen),
               "examples_seen": examples_seen, "batch_order_sha256": order_digest.hexdigest(),
               "elapsed_seconds": time.monotonic() - started,
               "best_update": min(history, key=lambda item: item["selection_score"])["update"],
               "best_move_update": min(history, key=lambda item: item["validation"]["overall"]["move_nll"])["update"],
               "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None}
    (directory / "run-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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


def evaluate_checkpoint(checkpoint, data, *, device="auto", batch_size=32, max_per_player=0,
                        baseline_path=None, predictions_path=None):
    model, device = load_model(checkpoint, device)
    dataset = JsonlDataset(data, model.player_names, max_per_player)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_rows, players=model.player_names))
    baseline = TimingBaseline.from_dict(json.loads(Path(baseline_path).read_text())) if baseline_path else None
    return evaluate(model, loader, device, timing_baseline=baseline, predictions_path=predictions_path)


@torch.no_grad()
def predict(checkpoint, row, *, device="auto", top_k=5):
    model, device = load_model(checkpoint, device)
    # Labels are removed even if a prepared training record was supplied.
    row = {k: v for k, v in row.items() if k not in ("move", "elapsed_seconds")}
    if "player" not in row and len(model.player_names) == 1:
        row["player"] = model.player_names[0]
    if row.get("player") not in model.player_names:
        raise ValueError(f"Checkpoint supports {model.player_names}, not player {row.get('player')!r}")
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
