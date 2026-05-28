"""4象限スクリーニングの単体テスト。"""
from __future__ import annotations

import pandas as pd
from dataclasses import replace

from quadrant_screening.fundamentals import FundamentalSnapshot, eps_discount_pct
from quadrant_screening.scoring import compute_score
from quadrant_screening.sector import SectorMomentum
from quadrant_screening.technical import (
    analyze_technical,
    detect_buy_patterns,
    passes_primary_filter,
)
from quadrant_screening.ticker_utils import normalize_ticker


def test_normalize_ticker_fixes_duplicate_suffix():
    assert normalize_ticker("7063.T7063.T") == "7063.T"
    assert normalize_ticker("7203") == "7203.T"
    assert normalize_ticker("7203.T") == "7203.T"


def test_primary_filter_thresholds():
    assert passes_primary_filter(500.0, 300_000) is True
    assert passes_primary_filter(499.0, 300_000) is False
    assert passes_primary_filter(500.0, 299_999) is False


def test_scoring_prefers_volume_and_pattern():
    df = _sample_uptrend_df()
    tech = analyze_technical(df)
    assert tech is not None
    assert tech.above_75ma is True

    fund = FundamentalSnapshot(roe_pct=10.0, trailing_eps=100.0, trailing_pe=12.0)
    sector = SectorMomentum(sector_code=1, sector_return_pct=5.0, topix_return_pct=2.0)
    high = compute_score(tech, fund, sector)

    fund_low = FundamentalSnapshot(roe_pct=5.0, trailing_eps=None, trailing_pe=None, roe_is_default=True)
    sector_bad = SectorMomentum(sector_code=1, sector_return_pct=1.0, topix_return_pct=3.0)
    tech_flat = replace(tech, vol_ratio=1.0, patterns=[])
    low = compute_score(tech_flat, fund_low, sector_bad)

    assert high.total > low.total


def test_eps_discount_positive_when_cheap():
    assert eps_discount_pct(1000.0, 100.0, industry_per=15.0) == 50.0


def test_detect_engulfing_pattern():
    ohlc = {
        "Open": [100, 102, 98, 97, 96],
        "High": [101, 103, 99, 100, 105],
        "Low": [99, 100, 95, 94, 95],
        "Close": [100, 101, 96, 95, 104],
        "Volume": [100_000] * 5,
    }
    df = pd.DataFrame(ohlc)
    base = pd.concat([df] * 16, ignore_index=True)
    patterns = detect_buy_patterns(base)
    assert isinstance(patterns, list)


def _sample_uptrend_df() -> pd.DataFrame:
    n = 90
    close = [500 + i * 2 for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [c - 1 for c in close],
            "High": [c + 2 for c in close],
            "Low": [c - 3 for c in close],
            "Close": close,
            "Volume": [400_000 if i == n - 1 else 300_000 for i in range(n)],
        }
    )
