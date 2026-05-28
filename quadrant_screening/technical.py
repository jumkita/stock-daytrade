"""一次フィルター・需給・75MA・買いパターン。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quadrant_screening.config import (
    MA_PERIOD,
    MIN_PRICE,
    MIN_VOL_20D,
    VOL_RATIO_LOOKBACK,
)


@dataclass
class TechnicalSnapshot:
    price: float
    vol_20d_avg: float
    vol_ratio: float
    ma75: float
    above_75ma: bool
    patterns: list[str]
    atr14: float | None = None


def _atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period + 1:
        return None
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    val = tr.iloc[-period:].mean()
    return float(val) if pd.notna(val) else None


def passes_primary_filter(price: float, vol_20d: float) -> bool:
    return price >= MIN_PRICE and vol_20d >= MIN_VOL_20D


def detect_buy_patterns(df: pd.DataFrame) -> list[str]:
    """直近足の買いパターン（pandas のみ・主要6種）。"""
    if df is None or len(df) < 3:
        return []
    o = df["Open"].astype(float).values
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    c = df["Close"].astype(float).values
    i = len(df) - 1
    found: list[str] = []

    body = abs(c[i] - o[i])
    rng = h[i] - l[i]
    if rng <= 0:
        return found

    lower = min(o[i], c[i]) - l[i]
    upper = h[i] - max(o[i], c[i])

    # ハンマー / ピンバー（下ヒゲ長い）
    if lower >= body * 2 and upper <= body * 0.5 and c[i] >= o[i]:
        found.append("ハンマー")

    # 包み線（陽線が前日陰線を包む）
    if i >= 1 and c[i] > o[i] and o[i - 1] > c[i - 1]:
        if c[i] >= o[i - 1] and o[i] <= c[i - 1]:
            found.append("包み線")

    # 赤三兵（3連陽線）
    if i >= 2 and all(c[j] > o[j] for j in (i - 2, i - 1, i)):
        if c[i] > c[i - 1] > c[i - 2]:
            found.append("赤三兵")

    # 明けの明星（簡易）
    if i >= 2:
        body0 = abs(c[i - 2] - o[i - 2])
        body1 = abs(c[i - 1] - o[i - 1])
        if c[i - 2] < o[i - 2] and body1 <= body0 * 0.4 and c[i] > o[i]:
            if c[i] > (o[i - 2] + c[i - 2]) / 2:
                found.append("明けの明星")

    # 上昇三法（簡易: 大陽線→小足→ブレイク）
    if i >= 2 and c[i - 2] > o[i - 2] and c[i] > h[i - 1] and c[i] > c[i - 2]:
        found.append("上昇三法")

    # たくり線（下ヒゲ）
    if lower >= rng * 0.55 and body <= rng * 0.35:
        found.append("たくり線")

    return found


def analyze_technical(df: pd.DataFrame) -> TechnicalSnapshot | None:
    if df is None or len(df) < MA_PERIOD:
        return None
    try:
        close = float(df["Close"].iloc[-1])
        vol20 = float(df["Volume"].iloc[-20:].mean())
        vol5 = float(df["Volume"].iloc[-VOL_RATIO_LOOKBACK:].mean())
        vol_today = float(df["Volume"].iloc[-1])
        ma75 = float(df["Close"].iloc[-MA_PERIOD:].mean())
    except (TypeError, ValueError):
        return None

    if close <= 0 or pd.isna(vol20) or pd.isna(ma75):
        return None

    vol_ratio = vol_today / vol5 if vol5 > 0 else 0.0
    patterns = detect_buy_patterns(df)

    return TechnicalSnapshot(
        price=close,
        vol_20d_avg=vol20,
        vol_ratio=vol_ratio,
        ma75=ma75,
        above_75ma=close > ma75,
        patterns=patterns,
        atr14=_atr(df),
    )


def compute_tp_sl(price: float, atr: float | None, rr: float = 2.0) -> tuple[float, float]:
    """SL=2ATRまたは3%、TP=リスクリワード2倍。"""
    if price <= 0:
        return price, price
    risk = (2.0 * atr) if atr and atr > 0 else price * 0.03
    sl = max(price - risk, price * 0.92)
    tp = price + (price - sl) * rr
    return round(tp, 2), round(sl, 2)
