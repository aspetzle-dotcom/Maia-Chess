"""Sequential, cached access to Chess.com's public monthly archive API."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def read_json(url: str, *, user_agent: str, retries: int = 4):
    for attempt in range(retries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": user_agent}), timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
            time.sleep(min(delay, 60))
    raise RuntimeError("Unreachable retry state")


def download_player(username, output, *, since=None, until=None, max_months=3,
                    user_agent="chess-personas/0.1", refresh=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid Chess.com username")
    for month in (since, until):
        if month is not None:
            datetime.strptime(month, "%Y-%m")
    if since and until and since > until:
        raise ValueError("--since must be no later than --until")
    if max_months < 0:
        raise ValueError("--max-months must be nonnegative; 0 means all")
    username = username.lower()
    directory = Path(output) / username
    directory.mkdir(parents=True, exist_ok=True)
    root = f"https://api.chess.com/pub/player/{username}"
    archives = read_json(root + "/games/archives", user_agent=user_agent)["archives"]
    selected = []
    for url in archives:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "api.chess.com":
            raise ValueError("Unexpected archive host")
        match = re.fullmatch(rf"/pub/player/{re.escape(username)}/games/(\d{{4}})/(\d{{2}})", parsed.path, re.I)
        if not match:
            raise ValueError("Unexpected archive path")
        month = "-".join(match.groups())
        if (since is None or month >= since) and (until is None or month <= until):
            selected.append((month, url))
    selected.sort()
    if max_months:
        selected = selected[-max_months:]
    report = {"username": username, "fetched_at": datetime.now(timezone.utc).isoformat(), "months": []}
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    for month, url in selected:
        path = directory / f"{month}.json"
        if refresh or month == current_month or not path.exists():
            payload = read_json(url, user_agent=user_agent)
            if not isinstance(payload.get("games"), list):
                raise ValueError(f"Invalid archive response for {month}")
            temporary = path.with_suffix(".json.partial")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        report["months"].append({"month": month, "games": len(payload["games"]), "source": url})
        print(f"{username} {month}: {len(payload['games'])} games", flush=True)
    (directory / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
