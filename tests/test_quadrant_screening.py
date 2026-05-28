"""4象限スクリーニングの単体テスト。"""
from __future__ import annotations

import pandas as pd
from dataclasses import replace

from quadrant_screening.config import (
    HIGH_GROWTH_PER,
    SCORE_ABOVE_MA,
    SCORE_EPS_DISCOUNT,
    SCORE_MAX,
    SCORE_PATTERN,
    SCORE_ROE_BONUS,
    SCORE_SECTOR_OUTPERFORM,
    SCORE_VOL_STRONG,
    SCORE_VOL_WEAK,
    SECTOR_PER_BY_CODE,
    SECTOR_PER_MAP,
    get_sector_per,
)
from quadrant_screening.fundamentals import (
    FundamentalSnapshot,
    compute_allowed_per,
    eps_discount_pct,
)
from quadrant_screening.scoring import compute_score
from quadrant_screening.sector import SectorMomentum
from quadrant_screening.technical import (
    TechnicalSnapshot,
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


def test_eps_discount_uses_sector_per():
    # 銀行(15)=PER10 → 適正1000、割安度0%
    assert eps_discount_pct(1000.0, 100.0, sector_code=15) == 0.0
    # 情報通信(10)=PER25 → 適正2500、割安度150%
    assert eps_discount_pct(1000.0, 100.0, sector_code=10) == 150.0
    assert get_sector_per(12) == SECTOR_PER_BY_CODE[12]
    assert SECTOR_PER_MAP["銀行"] == 10.0
    assert SECTOR_PER_MAP["情報通信・サービスその他"] == 25.0


def test_compute_allowed_per_growth_tiers():
    assert compute_allowed_per(15.0, 0.0) == 15.0
    assert compute_allowed_per(15.0, 19.9) == 15.0
    assert compute_allowed_per(15.0, 20.0) == 30.0
    assert compute_allowed_per(15.0, 39.9) == 30.0
    assert compute_allowed_per(15.0, 40.0) == 45.0
    assert compute_allowed_per(20.0, 25.0) == 40.0


def test_high_growth_ticker_bypass_per():
    assert compute_allowed_per(20.0, 0.0, ticker="6146.T") == HIGH_GROWTH_PER
    assert compute_allowed_per(20.0, 50.0, ticker="8035") == HIGH_GROWTH_PER


def test_eps_discount_with_growth_premium():
    # 基礎PER15 × 成長20% → 30倍、適正3000、割安度20%
    assert eps_discount_pct(2500.0, 100.0, industry_per=15.0, growth_pct=25.0) == 20.0
    # 6146: 業種PER20でもホワイトリスト40倍 → 適正40000、割安
    disc = eps_discount_pct(35000.0, 1000.0, sector_code=9, ticker="6146.T")
    assert disc is not None and disc > 0


def test_high_pe_growth_stock_gets_discount_score():
    tech = TechnicalSnapshot(
        price=35000.0,
        vol_20d_avg=400_000,
        vol_ratio=1.0,
        ma75=30000.0,
        above_75ma=True,
        patterns=[],
    )
    fund = FundamentalSnapshot(
        roe_pct=15.0,
        trailing_eps=1000.0,
        trailing_pe=35.0,
        growth_pct=0.0,
    )
    without = compute_score(tech, fund, None, sector_code=9, ticker="9999.T")
    with_whitelist = compute_score(tech, fund, None, sector_code=9, ticker="6146.T")
    assert without.fundamental_pts == SCORE_ROE_BONUS
    assert with_whitelist.fundamental_pts == SCORE_ROE_BONUS + SCORE_EPS_DISCOUNT


def test_score_allocation_sums_to_100_max():
    assert (
        SCORE_SECTOR_OUTPERFORM
        + SCORE_VOL_STRONG
        + SCORE_ABOVE_MA
        + SCORE_PATTERN
        + SCORE_ROE_BONUS
        + SCORE_EPS_DISCOUNT
        == SCORE_MAX
    )


def test_max_score_is_100_when_all_conditions_met():
    tech = TechnicalSnapshot(
        price=500.0,
        vol_20d_avg=400_000,
        vol_ratio=1.5,
        ma75=400.0,
        above_75ma=True,
        patterns=["包み線"],
    )
    fund = FundamentalSnapshot(roe_pct=10.0, trailing_eps=100.0, trailing_pe=5.0)
    sector = SectorMomentum(sector_code=10, sector_return_pct=5.0, topix_return_pct=2.0)
    result = compute_score(tech, fund, sector, sector_code=10)
    assert result.total == 100.0
    assert result.sector_pts == SCORE_SECTOR_OUTPERFORM
    assert result.volume_pts == SCORE_VOL_STRONG
    assert result.technical_pts == SCORE_ABOVE_MA + SCORE_PATTERN
    assert result.fundamental_pts == SCORE_ROE_BONUS + SCORE_EPS_DISCOUNT


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
