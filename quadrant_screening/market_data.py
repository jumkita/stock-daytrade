"""OHLCV バルク取得・正規化。"""
from __future__ import annotations

from typing import Iterator

import pandas as pd
import yfinance as yf

from quadrant_screening.config import BULK_CHUNK_SIZE, OHLCV_PERIOD


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df.copy())
    if "Date" not in df.columns:
        df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=False).dt.tz_localize(None)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace("¥", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
    return df.dropna(subset=["Close"]).reset_index(drop=True)


def chunk_list(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_ohlcv_bulk(tickers: list[str], period: str = OHLCV_PERIOD) -> dict[str, pd.DataFrame]:
    """銘柄リストをチャンク単位で一括ダウンロード。"""
    result: dict[str, pd.DataFrame] = {}
    if not tickers:
        return result

    for chunk in chunk_list(tickers, BULK_CHUNK_SIZE):
        try:
            raw = yf.download(
                chunk,
                period=period,
                interval="1d",
                group_by="ticker",
                progress=False,
                auto_adjust=True,
                threads=True,
            )
        except Exception:
            raw = None

        if raw is None or getattr(raw, "empty", True):
            for t in chunk:
                result[t] = pd.DataFrame()
            continue

        if len(chunk) == 1:
            t = chunk[0]
            result[t] = normalize_ohlcv(raw)
            continue

        for t in chunk:
            try:
                sub = raw[t].dropna(how="all")
                result[t] = normalize_ohlcv(sub) if not sub.empty else pd.DataFrame()
            except Exception:
                result[t] = pd.DataFrame()
    return result
