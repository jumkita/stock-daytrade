"""4象限スクリーニングの閾値（過去シグナル検証に基づく調整値）。"""
from __future__ import annotations

# 一次フィルター（高速足切り）
MIN_PRICE = 500.0
MIN_VOL_20D = 300_000

# 需給（1.5倍は件数が少なすぎるため 1.25 / 1.15 に緩和）
VOL_SPIKE_STRONG = 1.25
VOL_SPIKE_WEAK = 1.15

# テクニカル
MA_PERIOD = 75
VOL_RATIO_LOOKBACK = 5

# ファンダメンタル（欠損時フォールバック・ハード除外はしない）
ROE_DEFAULT_PCT = 5.0
ROE_BONUS_THRESHOLD = 8.0
DEFAULT_INDUSTRY_PER = 15.0

# セクター（約1ヶ月）
SECTOR_MOMENTUM_DAYS = 21
TOPIX_ETF = "1306.T"

# 17業種コード → 東証17業種ETF
SECTOR_ETF_BY_CODE: dict[int, str] = {
    1: "1617.T",
    2: "1618.T",
    3: "1619.T",
    4: "1620.T",
    5: "1621.T",
    6: "1622.T",
    7: "1623.T",
    8: "1624.T",
    9: "1625.T",
    10: "1626.T",
    11: "1627.T",
    12: "1628.T",
    13: "1629.T",
    14: "1630.T",
    15: "1631.T",
    16: "1632.T",
    17: "1633.T",
}

# 実行
BULK_CHUNK_SIZE = 80
FUNDAMENTAL_MAX_WORKERS = 8
OHLCV_PERIOD = "4mo"
TOP_N_DEFAULT = 5

# スコア配点（100点満点）
SCORE_SECTOR_OUTPERFORM = 10
SCORE_VOL_STRONG = 20
SCORE_VOL_WEAK = 10
SCORE_PATTERN = 25
SCORE_ROE_BONUS = 5
SCORE_EPS_DISCOUNT = 10
SCORE_ABOVE_MA = 15
