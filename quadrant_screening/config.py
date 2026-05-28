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

# 東証17業種コード → 業種名・中央値PER（割安度算出用）
SECTOR_NAME_BY_CODE: dict[int, str] = {
    1: "食品",
    2: "エネルギー資源",
    3: "建設・資材",
    4: "素材・化学",
    5: "医薬品",
    6: "自動車・輸送機",
    7: "鉄鋼・非鉄",
    8: "機械",
    9: "電気機器",
    10: "情報通信・サービスその他",
    11: "電力・ガス",
    12: "運輸・物流",
    13: "商社・卸売",
    14: "小売",
    15: "銀行",
    16: "金融（除く銀行）",
    17: "不動産",
}

SECTOR_PER_BY_CODE: dict[int, float] = {
    1: 20.0,
    2: 10.0,
    3: 12.0,
    4: 14.0,
    5: 25.0,
    6: 12.0,
    7: 8.0,
    8: 16.0,
    9: 20.0,
    10: 25.0,
    11: 12.0,
    12: 7.0,
    13: 10.0,
    14: 18.0,
    15: 10.0,
    16: 14.0,
    17: 12.0,
}

# 業種名 → PER（参照・テスト用。未登録は DEFAULT_INDUSTRY_PER）
SECTOR_PER_MAP: dict[str, float] = {
    SECTOR_NAME_BY_CODE[code]: per for code, per in SECTOR_PER_BY_CODE.items()
}


def get_sector_per(sector_code: int | None) -> float:
    """17業種コードから適正PERを返す。不明時は DEFAULT_INDUSTRY_PER。"""
    if sector_code is None:
        return DEFAULT_INDUSTRY_PER
    return SECTOR_PER_BY_CODE.get(int(sector_code), DEFAULT_INDUSTRY_PER)


# 成長率に応じた許容PER拡張（基礎PER × 乗数）
GROWTH_PER_TIER2_PCT = 20.0
GROWTH_PER_TIER3_PCT = 40.0
GROWTH_PER_MULT_NORMAL = 1.0
GROWTH_PER_MULT_TIER2 = 2.0
GROWTH_PER_MULT_TIER3 = 3.0

# 半導体・高成長テーマ（17業種では電気機器に混在するためホワイトリスト）
HIGH_GROWTH_PER = 40.0
HIGH_GROWTH_TICKERS: tuple[str, ...] = (
    "8035.T",  # 東京エレクトロン
    "6920.T",  # レーザーテック
    "6146.T",  # ディスコ
    "6857.T",  # アドバンテスト
    "7735.T",  # SCREENホールディングス
    "3436.T",  # SUMCO
    "6963.T",  # ローム
    "6988.T",  # 日東電工
)
HIGH_GROWTH_TICKER_SET: frozenset[str] = frozenset(t.upper() for t in HIGH_GROWTH_TICKERS)

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

# スコア配点（100点満点: 15 + 25 + 40 + 20）
SCORE_MAX = 100
SCORE_SECTOR_OUTPERFORM = 15
SCORE_VOL_STRONG = 25
SCORE_VOL_WEAK = 10
SCORE_PATTERN = 25
SCORE_ROE_BONUS = 10
SCORE_EPS_DISCOUNT = 10
SCORE_ABOVE_MA = 15
