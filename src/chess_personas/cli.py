import argparse
import glob
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate player-specific chess move and duration models")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Cache public monthly Chess.com game archives")
    download.add_argument("players", nargs="+")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--since", help="First month, YYYY-MM")
    download.add_argument("--until", help="Last month, YYYY-MM")
    download.add_argument("--max-months", type=int, default=3, help="Newest matching months per player; 0 = all")
    download.add_argument("--user-agent", default="chess-personas/0.1")
    download.add_argument("--refresh", action="store_true")
    prepare = commands.add_parser("prepare", help="Extract labels and split whole games chronologically")
    prepare.add_argument("sources", nargs="+", help="PGNs, monthly JSONs, or quoted glob patterns")
    prepare.add_argument("--players", nargs="+", default=["hikaru", "fabianocaruana"])
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--time-classes", nargs="+", choices=["bullet", "blitz", "rapid"])
    prepare.add_argument("--train-fraction", type=float, default=0.8)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    train = commands.add_parser("train", help="Train an independent move and duration model for one player")
    train.add_argument("--train", default="data/processed/train.jsonl")
    train.add_argument("--val", default="data/processed/val.jsonl")
    train.add_argument("--output", default="runs/first")
    train.add_argument("--player", "--players", dest="players", nargs=1, required=True,
                       help="One target username per checkpoint; run once for each player")
    train.add_argument("--backbone", choices=["maia3", "smoke"], default="maia3")
    train.add_argument("--model-name", default="maia3-5m")
    train.add_argument("--checkpoint", help="Optional local upstream Maia weights")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--max-per-player", type=int, default=10000, help="Position cap in each split; 0 = all")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--time-weight", type=float, default=0.5)
    train.add_argument("--kl-weight", type=float, default=0.02)
    evaluate = commands.add_parser("evaluate", help="Score a checkpoint on a held-out JSONL split")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--data", default="data/processed/test.jsonl")
    evaluate.add_argument("--output", help="Optional JSON report path")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--max-per-player", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    predict = commands.add_parser("predict", help="Predict from one pre-move JSON observation")
    predict.add_argument("checkpoint")
    predict.add_argument("--record", required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--top-k", type=int, default=5)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "download":
        from .download import download_player
        players = args.pop("players")
        result = [download_player(player, **args) for player in players]
    elif command == "prepare":
        from .data import prepare as prepare_data
        patterns = args.pop("sources")
        paths = sorted(set(path for pattern in patterns for path in glob.glob(pattern)))
        paths = [p for p in paths if Path(p).name != "manifest.json"]
        if not paths:
            parser.error("No input files matched")
        result = prepare_data(paths, **args)
    elif command == "train":
        from .training import train as train_model
        args["players"] = [p.lower() for p in args["players"]]
        args["train_path"] = args.pop("train")
        args["val_path"] = args.pop("val")
        result = {"checkpoint": str(train_model(**args))}
    elif command == "evaluate":
        from .training import evaluate_checkpoint
        output = args.pop("output")
        result = evaluate_checkpoint(**args)
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        from .training import predict as predict_position
        record = json.loads(Path(args.pop("record")).read_text(encoding="utf-8"))
        if args["top_k"] < 1:
            parser.error("--top-k must be positive")
        result = predict_position(row=record, **args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
