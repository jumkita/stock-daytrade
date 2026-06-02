# -*- coding: utf-8 -*-
"""verify_signal_returns の利確日オフセット（holding_days）の再現テスト。"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

import verify_signal_returns as vsr


def _sample_ohlcv() -> pd.DataFrame:
    """シグナル日=2024-01-15 とその後2営業日のダミー日足。"""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"]),
            "Close": [100.0, 110.0, 121.0],
        }
    )


@patch.object(vsr, "fetch_ohlcv", return_value=_sample_ohlcv())
def test_verify_returns_holding_one_exits_next_session_day(mock_fetch):
    """holding_days=1 のとき利確は翌営業日（iloc[1]）の終値。"""
    payload = {
        "updated": "2024-01-15T06:00:00+00:00",
        "items": [
            {"ticker": "7203.T", "entry": 100.0, "name": "Test", "pattern_name": "P"},
        ],
    }
    rows = vsr.verify_returns(payload, holding_days=1)
    assert len(rows) == 1
    r = rows[0]
    assert r.get("error") is None
    assert r["exit_price"] == 110.0
    assert r["exit_date"] == "2024-01-16"
    assert r["return_pct"] == 10.0


@patch.object(vsr, "fetch_ohlcv", return_value=_sample_ohlcv())
def test_verify_returns_default_holding_is_one(mock_fetch):
    """verify_returns の既定 holding_days は 1（翌営業日引け）。"""
    payload = {
        "updated": "2024-01-15T06:00:00+00:00",
        "items": [
            {"ticker": "7203.T", "entry": 100.0, "name": "Test", "pattern_name": "P"},
        ],
    }
    rows = vsr.verify_returns(payload)
    assert rows[0]["exit_date"] == "2024-01-16"


@patch.object(vsr, "fetch_ohlcv", return_value=_sample_ohlcv())
def test_verify_returns_holding_three_exits_third_forward_bar(mock_fetch):
    """holding_days=3 で十分な行があるとき iloc[3] の終値で利確。"""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"]
            ),
            "Close": [100.0, 101.0, 102.0, 103.0, 400.0],
        }
    )
    mock_fetch.return_value = df
    payload = {
        "updated": "2024-01-15T06:00:00+00:00",
        "items": [
            {"ticker": "7203.T", "entry": 100.0, "name": "Test", "pattern_name": "P"},
        ],
    }
    rows = vsr.verify_returns(payload, holding_days=3)
    assert rows[0]["exit_price"] == 103.0
    assert rows[0]["exit_date"] == "2024-01-18"
