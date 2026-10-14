import json

import pytest

from chess_personas.download import download_player


def test_downloader_filters_and_uses_cache(tmp_path, monkeypatch):
    calls = []
    def fake_read(url, **kwargs):
        calls.append(url)
        if url.endswith("archives"):
            return {"archives": [f"https://api.chess.com/pub/player/hikaru/games/2024/{month:02d}" for month in (1, 2, 3)]}
        return {"games": [{"pgn": "synthetic"}]}
    monkeypatch.setattr("chess_personas.download.read_json", fake_read)
    result = download_player("Hikaru", tmp_path, since="2024-01", until="2024-02", max_months=1)
    assert [m["month"] for m in result["months"]] == ["2024-02"]
    assert len(calls) == 2
    download_player("Hikaru", tmp_path, since="2024-01", until="2024-02", max_months=1)
    assert len(calls) == 3
    assert json.loads((tmp_path / "hikaru/2024-02.json").read_text())["games"]


def test_archive_cannot_redirect_to_unexpected_host(tmp_path, monkeypatch):
    monkeypatch.setattr("chess_personas.download.read_json", lambda *a, **kw: {"archives": ["https://example.org/private"]})
    with pytest.raises(ValueError, match="archive host"):
        download_player("hikaru", tmp_path)
