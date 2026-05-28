"""17業種ETF vs TOPIX のセクターモメンタム。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from quadrant_screening.config import (
    SECTOR_ETF_BY_CODE,
    SECTOR_MOMENTUM_DAYS,
    TOPIX_ETF,
)
from quadrant_screening.market_data import normalize_ohlcv


@dataclass(frozen=True)
class SectorMomentum:
    sector_code: int
    sector_return_pct: float
    topix_return_pct: float

    @property
    def outperforms_topix(self) -> bool:
        return self.sector_return_pct > self.topix_return_pct


def _return_over_days(df: pd.DataFrame, days: int) -> float | None:
    if df.empty or len(df) < days + 1 or "Close" not in df.columns:
        return None
    close = df["Close"].astype(float)
    start = float(close.iloc[-days - 1])
    end = float(close.iloc[-1])
    if start <= 0:
        return None
    return (end - start) / start * 100.0


def load_sector_momentum(days: int = SECTOR_MOMENTUM_DAYS) -> tuple[float | None, dict[int, SectorMomentum]]:
    """
    TOPIX ETF と17業種ETFの直近騰落率を一括取得。
    Returns: (topix_return_pct, {sector_code: SectorMomentum})
    """
    tickers = [TOPIX_ETF] + list(SECTOR_ETF_BY_CODE.values())
    try:
        raw = yf.download(
            tickers,
            period="3mo",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        return None, {}

    etf_to_sector = {v: k for k, v in SECTOR_ETF_BY_CODE.items()}
    returns: dict[str, float | None] = {}

    if len(tickers) == 1:
        returns[tickers[0]] = _return_over_days(normalize_ohlcv(raw), days)
    else:
        for t in tickers:
            try:
                sub = raw[t].dropna(how="all")
                returns[t] = _return_over_days(normalize_ohlcv(sub), days)
            except Exception:
                returns[t] = None

    topix_ret = returns.get(TOPIX_ETF)
    out: dict[int, SectorMomentum] = {}
    for etf, sec_code in etf_to_sector.items():
        sec_ret = returns.get(etf)
        if sec_ret is None or topix_ret is None:
            continue
        out[sec_code] = SectorMomentum(
            sector_code=sec_code,
            sector_return_pct=sec_ret,
            topix_return_pct=topix_ret,
        )
    return topix_ret, out
