"""Training-only empirical timing baselines with sparse-group fallback."""

from collections import defaultdict

import numpy as np

from .features import duration_bin


def control_key(row):
    return f"{row['base_seconds']:g}+{row['increment_seconds']:g}"


def clock_band(row):
    clock = row.get("self_clock")
    if clock is None:
        return "missing"
    return ("0-10", "10-30", "30-60", "60-180", "180+")[int(np.searchsorted([10, 30, 60, 180], clock, side="left"))]


def group_keys(row):
    player = row["player"]
    control = player + "/control=" + control_key(row)
    return [player, control, control + "/clock=" + clock_band(row)]


class TimingBaseline:
    def __init__(self, groups, min_count=20, smoothing=32.0):
        self.groups, self.min_count, self.smoothing = groups, min_count, smoothing

    @classmethod
    def fit(cls, rows, min_count=20, smoothing=32.0):
        counts = defaultdict(lambda: np.zeros(32, dtype=np.int64))
        seconds = defaultdict(list)
        for row in rows:
            label = duration_bin(row.get("elapsed_seconds"), row.get("self_clock"))
            if label < 0:
                continue
            for key in group_keys(row):
                counts[key][label] += 1
                seconds[key].append(row["elapsed_seconds"])
        groups = {key: {"count": len(seconds[key]), "bins": bins.tolist(),
                        "median_seconds": float(np.median(seconds[key]))}
                  for key, bins in counts.items()}
        return cls(groups, min_count, smoothing)

    def predict(self, row):
        keys = group_keys(row)
        parent = self.groups.get(keys[0])
        if parent is None:
            return None
        prior = np.asarray(parent["bins"], dtype=float) + 1.0
        prior /= prior.sum()
        selected, selected_key = parent, keys[0]
        for key in keys[1:]:
            candidate = self.groups.get(key)
            if candidate is not None and candidate["count"] >= self.min_count:
                selected, selected_key = candidate, key
        probabilities = np.asarray(selected["bins"], dtype=float) + self.smoothing * prior
        probabilities /= probabilities.sum()
        return {"probabilities": probabilities, "median_seconds": min(selected["median_seconds"], row["self_clock"]),
                "group": selected_key}

    def to_dict(self):
        return {"fit_split": "training_only", "min_count": self.min_count,
                "smoothing": self.smoothing, "groups": self.groups}

    @classmethod
    def from_dict(cls, value):
        return cls(value["groups"], value["min_count"], value["smoothing"])
