"""ROE / EPS / 成長率 取得（欠損フォールバック付き）。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import yfinance as yf

from quadrant_screening.config import (
    FUNDAMENTAL_MAX_WORKERS,
    GROWTH_PER_MULT_NORMAL,
    GROWTH_PER_MULT_TIER2,
    GROWTH_PER_MULT_TIER3,
    GROWTH_PER_TIER2_PCT,
    GROWTH_PER_TIER3_PCT,
    HIGH_GROWTH_PER,
    HIGH_GROWTH_TICKER_SET,
    ROE_DEFAULT_PCT,
    get_sector_per,
)
from quadrant_screening.ticker_utils import normalize_ticker


@dataclass
class FundamentalSnapshot:
    roe_pct: float
    trailing_eps: float | None
    trailing_pe: float | None
    growth_pct: float = 0.0
    roe_is_default: bool = False


def _parse_growth_pct(info: dict) -> float:
    """earningsGrowth を優先し、欠損時は revenueGrowth。いずれも無ければ 0%。"""
    for key in ("earningsGrowth", "revenueGrowth"):
        raw = info.get(key)
        if raw is not None and isinstance(raw, (int, float)) and raw == raw:
            v = float(raw)
            if abs(v) <= 1.5:
                v *= 100.0
            return v
    return 0.0


def compute_allowed_per(
    base_per: float,
    growth_pct: float,
    ticker: str | None = None,
) -> float:
    """
    業種別基礎PERに成長率プレミアムを乗算して許容PERを返す。
    HIGH_GROWTH_TICKERS は無条件で HIGH_GROWTH_PER（40倍）を適用。
    """
    norm = normalize_ticker(ticker) if ticker else None
    if norm and norm in HIGH_GROWTH_TICKER_SET:
        return HIGH_GROWTH_PER

    if growth_pct >= GROWTH_PER_TIER3_PCT:
        multiplier = GROWTH_PER_MULT_TIER3
    elif growth_pct >= GROWTH_PER_TIER2_PCT:
        multiplier = GROWTH_PER_MULT_TIER2
    else:
        multiplier = GROWTH_PER_MULT_NORMAL
    return base_per * multiplier


def _fetch_one(ticker: str) -> tuple[str, FundamentalSnapshot]:
    roe_pct = ROE_DEFAULT_PCT
    eps: float | None = None
    pe: float | None = None
    growth_pct = 0.0
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
        growth_pct = _parse_growth_pct(info)
    except Exception:
        pass
    return ticker, FundamentalSnapshot(
        roe_pct=roe_pct,
        trailing_eps=eps,
        trailing_pe=pe,
        growth_pct=growth_pct,
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
                    growth_pct=0.0,
                    roe_is_default=True,
                )
    return out


def eps_discount_pct(
    price: float,
    eps: float | None,
    *,
    sector_code: int | None = None,
    industry_per: float | None = None,
    growth_pct: float = 0.0,
    ticker: str | None = None,
) -> float | None:
    """適正株価(EPS×補正後許容PER)に対する割安度（%）。プラス=割安。"""
    if price <= 0 or eps is None or eps <= 0:
        return None
    base_per = industry_per if industry_per is not None else get_sector_per(sector_code)
    allowed_per = compute_allowed_per(base_per, growth_pct, ticker)
    fair = eps * allowed_per
    if fair <= 0:
        return None
    return (fair - price) / price * 100.0
