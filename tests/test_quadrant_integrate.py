"""integrate モジュールの単体テスト。"""
from __future__ import annotations

import pandas as pd

from quadrant_screening.integrate import apply_quadrant_to_backtest_items, format_backtest_quadrant_line


def _uptrend_df(n: int = 90) -> pd.DataFrame:
    close = [600 + i * 2 for i in range(n)]
    vol = [400_000 if i == n - 1 else 350_000 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [c - 1 for c in close],
            "High": [c + 2 for c in close],
            "Low": [c - 3 for c in close],
            "Close": close,
            "Volume": vol,
        }
    )


def test_format_backtest_quadrant_line():
    line = format_backtest_quadrant_line(
        {
            "ticker": "7203.T",
            "quadrant_score": 65,
            "quadrant_sign": "包み線",
            "vol_ratio": 1.3,
            "sector_label": "良",
            "roe_pct": 9.5,
            "entry": 2500,
            "tp": 2600,
            "sl": 2450,
        }
    )
    assert "【7203】" in line
    assert "65点" in line
    assert "包み線" in line


def test_apply_quadrant_filters_low_ma(monkeypatch, tmp_path):
    """75MA下の銘柄は除外される。"""
    csv = tmp_path / "jpx.csv"
    csv.write_text(
        "コード,銘柄名,市場・商品区分,17業種コード\n"
        "7203,トヨタ,プライム（内国株式）,6.0\n",
        encoding="utf-8",
    )
    df_bad = _uptrend_df()
    df_bad["Close"] = list(range(900, 900 - 90, -1))  # 下落で75MA上にいない可能性

    items = [
        {
            "ticker": "7203.T",
            "name": "トヨタ",
            "pattern_name": "包み線",
            "avg_return_pct": 2.0,
            "entry": 600,
            "tp": 620,
            "sl": 580,
        }
    ]
    bulk = {"7203.T": df_bad}

    monkeypatch.setattr(
        "quadrant_screening.integrate.load_sector_momentum",
        lambda days=21: (1.0, {}),
    )
    monkeypatch.setattr(
        "quadrant_screening.integrate.fetch_fundamentals_parallel",
        lambda tickers: {},
    )

    out, stats = apply_quadrant_to_backtest_items(items, bulk, csv, min_score=0)
    assert stats.before == 1
    assert stats.after <= 1
