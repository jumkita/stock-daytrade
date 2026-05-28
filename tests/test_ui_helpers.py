"""ui_helpers の単体テスト。"""
from __future__ import annotations

from ui_helpers import (
    filter_signals,
    history_json_url,
    items_to_dataframe,
    parse_signals_payload,
    sort_signals,
    with_cache_buster,
)


def test_filter_win_rate_70():
    items = [
        {"ticker": "7203.T", "win_rate": 0.75, "quadrant_score": 50},
        {"ticker": "9984.T", "win_rate": 0.60, "quadrant_score": 90},
    ]
    out = filter_signals(items, "win_rate_70", backtest_format=True)
    assert len(out) == 1
    assert out[0]["ticker"] == "7203.T"


def test_filter_sector_good():
    items = [
        {"ticker": "7203.T", "sector_label": "良"},
        {"ticker": "9984.T", "sector_label": "悪"},
    ]
    out = filter_signals(items, "sector_good")
    assert len(out) == 1


def test_sort_by_quadrant_score():
    items = [
        {"quadrant_score": 50},
        {"quadrant_score": 90},
    ]
    out = sort_signals(items, "quadrant_score")
    assert out[0]["quadrant_score"] == 90


def test_parse_backtest_payload():
    data = {
        "backtest_driven": True,
        "items": [{"ticker": "7203.T"}],
        "signal_count": 1,
        "unique_tickers": 1,
        "updated": "2026-05-28",
    }
    parsed = parse_signals_payload(data)
    assert parsed["backtest_format"] is True
    assert len(parsed["items"]) == 1


def test_history_json_url():
    base = "https://raw.githubusercontent.com/u/r/main/daily_buy_signals.json"
    assert history_json_url(base, "2026-05-28").endswith("daily_buy_signals_2026-05-28.json")


def test_with_cache_buster_adds_ts():
    url = with_cache_buster("https://example.com/a.json", ts=123)
    assert "_ts=123" in url


def test_items_to_dataframe_mobile():
    items = [{"ticker": "7203.T", "name": "Toyota", "quadrant_score": 80, "win_rate": 0.7}]
    df = items_to_dataframe(items, mobile=True)
    assert "銘柄コード" in df.columns
    assert "4象限" in df.columns
