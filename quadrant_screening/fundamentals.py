"""ROE / EPS 取得（欠損フォールバック付き）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import yfinance as yf

from quadrant_screening.config import (
    DEFAULT_INDUSTRY_PER,
    FUNDAMENTAL_MAX_WORKERS,
    ROE_DEFAULT_PCT,
)


@dataclass
class FundamentalSnapshot:
    roe_pct: float
    trailing_eps: float | None
    trailing_pe: float | None
    roe_is_default: bool = False


def _fetch_one(ticker: str) -> tuple[str, FundamentalSnapshot]:
    roe_pct = ROE_DEFAULT_PCT
    eps: float | None = None
    pe: float | None = None
    is_default = True
    try:
        info = yf.Ticker(ticker).info or {}
        raw_roe = info.get("returnOnEquity")
        if raw_roe is not None and isinstance(raw_roe, (int, float)) and raw_roe == raw_roe:
            roe_pct = float(raw_roe) * 100.0 if abs(float(raw_roe)) <= 1.5 else float(raw_roe)
            is_default = False
        raw_eps = info.get("trailingEps")
        if raw_eps is not None and isinstance(raw_eps, (int, float)) and raw_eps == raw_eps:
            eps = float(raw_eps)
        raw_pe = info.get("trailingPE")
        if raw_pe is not None and isinstance(raw_pe, (int, float)) and raw_pe == raw_pe:
            pe = float(raw_pe)
    except Exception:
        pass
    return ticker, FundamentalSnapshot(
        roe_pct=roe_pct,
        trailing_eps=eps,
        trailing_pe=pe,
        roe_is_default=is_default,
    )


def fetch_fundamentals_parallel(tickers: list[str]) -> dict[str, FundamentalSnapshot]:
    out: dict[str, FundamentalSnapshot] = {}
    if not tickers:
        return out
    workers = min(FUNDAMENTAL_MAX_WORKERS, len(tickers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futs):
            try:
                ticker, snap = fut.result()
                out[ticker] = snap
            except Exception:
                t = futs[fut]
                out[t] = FundamentalSnapshot(
                    roe_pct=ROE_DEFAULT_PCT,
                    trailing_eps=None,
                    trailing_pe=None,
                    roe_is_default=True,
                )
    return out


def eps_discount_pct(price: float, eps: float | None, industry_per: float = DEFAULT_INDUSTRY_PER) -> float | None:
    """適正株価(EPS×PER)に対する割安度（%）。プラス=割安。"""
    if price <= 0 or eps is None or eps <= 0:
        return None
    fair = eps * industry_per
    if fair <= 0:
        return None
    return (fair - price) / price * 100.0
