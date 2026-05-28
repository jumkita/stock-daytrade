"""auto_post 向け: バックテストシグナルへの4象限スコア付与・フィルタ。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quadrant_screening.fundamentals import FundamentalSnapshot, fetch_fundamentals_parallel
from quadrant_screening.config import ROE_DEFAULT_PCT
from quadrant_screening.scoring import ScoreBreakdown, compute_score
from quadrant_screening.sector import SectorMomentum, load_sector_momentum
from quadrant_screening.technical import (
    analyze_technical,
    compute_tp_sl,
    passes_primary_filter,
)
from quadrant_screening.ticker_utils import display_code, normalize_ticker
from quadrant_screening.universe import load_prime_universe

QuadrantMode = Literal["rank", "filter"]


@dataclass
class QuadrantApplyStats:
    before: int
    after: int
    dropped_primary: int
    dropped_ma: int
    dropped_score: int
    unscored_kept: int = 0
    mode: str = "rank"


def _quadrant_mode() -> QuadrantMode:
    raw = os.environ.get("QUADRANT_MODE", "rank").strip().lower()
    return "filter" if raw == "filter" else "rank"


def _sector_map_from_csv(csv_path: Path) -> dict[str, int | None]:
    uni = load_prime_universe(csv_path)
    out: dict[str, int | None] = {}
    for _, row in uni.iterrows():
        sec = row.get("sector_code_17")
        out[row["ticker"]] = int(sec) if pd.notna(sec) else None
    return out


def format_backtest_quadrant_line(item: dict[str, Any]) -> str:
    """4象限スコア付きの1行出力（auto_post / JSON 用）。"""
    code = display_code(str(item.get("ticker") or ""))
    score = item.get("quadrant_score")
    score_s = f"{float(score):.0f}" if score is not None else "—"
    sign = item.get("quadrant_sign") or item.get("pattern_name") or "—"
    vol = item.get("vol_ratio")
    vol_s = f"{float(vol):.1f}" if vol is not None else "—"
    sector = item.get("sector_label") or "—"
    roe = item.get("roe_pct")
    roe_s = f"{float(roe):.1f}" if roe is not None else "—"
    entry = item.get("entry")
    tp = item.get("tp")
    sl = item.get("sl")
    try:
        entry_f = float(entry) if entry is not None else None
        tp_f = float(tp) if tp is not None else None
        sl_f = float(sl) if sl is not None else None
    except (TypeError, ValueError):
        entry_f = tp_f = sl_f = None
    entry_s = f"¥{entry_f:,.0f}" if entry_f else "—"
    tp_s = f"¥{tp_f:,.0f}" if tp_f else "—"
    sl_s = f"¥{sl_f:,.0f}" if sl_f else "—"
    return (
        f"【{code}】総合スコア: {score_s}点 | "
        f"サイン: {sign} | 需給(出来高): {vol_s}倍 | "
        f"セクター: {sector} | ROE: {roe_s}% | "
        f"現在値: {entry_s} | TP: {tp_s} | SL: {sl_s}"
    )


def _enrich_item(
    item: dict[str, Any],
    ticker: str,
    tech: Any,
    fund: FundamentalSnapshot,
    sec_mom: SectorMomentum | None,
    breakdown: ScoreBreakdown,
) -> dict[str, Any]:
    pattern = str(item.get("pattern_name") or "")
    patterns = tech.patterns or ([pattern] if pattern else [])
    sign = "、".join(dict.fromkeys(patterns)) if patterns else pattern or "トレンド継続"
    tp, sl = compute_tp_sl(tech.price, tech.atr14)
    entry = item.get("entry")
    try:
        entry_f = float(entry) if entry is not None else tech.price
    except (TypeError, ValueError):
        entry_f = tech.price

    new_item = {
        **item,
        "ticker": ticker,
        "quadrant_score": breakdown.total,
        "quadrant_breakdown": {
            "sector": breakdown.sector_pts,
            "volume": breakdown.volume_pts,
            "technical": breakdown.technical_pts,
            "fundamental": breakdown.fundamental_pts,
        },
        "quadrant_sign": sign,
        "vol_ratio": round(tech.vol_ratio, 2),
        "sector_label": breakdown.sector_label,
        "roe_pct": round(fund.roe_pct, 2),
        "entry": entry_f,
        "tp": item.get("tp") if item.get("tp") is not None else tp,
        "sl": item.get("sl") if item.get("sl") is not None else sl,
    }
    new_item["formatted_line"] = format_backtest_quadrant_line(new_item)
    return new_item


def apply_quadrant_to_backtest_items(
    items: list[dict[str, Any]],
    bulk_ohlcv: dict[str, pd.DataFrame],
    csv_path: Path,
    min_score: float | None = None,
    top_n: int | None = None,
    mode: QuadrantMode | None = None,
) -> tuple[list[dict[str, Any]], QuadrantApplyStats]:
    """
    バックテスト通過シグナルに4象限スコアを付与。

    mode=rank（既定）: シグナルは落とさずスコア付きで並べ替えのみ。
    mode=filter: 一次・75MA・min_score で足切り（スタンドアロン向け）。
    """
    mode = mode or _quadrant_mode()
    if min_score is None:
        try:
            default_min = "25" if mode == "filter" else "0"
            min_score = float(os.environ.get("QUADRANT_MIN_SCORE", default_min))
        except ValueError:
            min_score = 25.0 if mode == "filter" else 0.0
    if top_n is None:
        try:
            top_n = int(os.environ.get("QUADRANT_TOP_N", "0")) or None
        except ValueError:
            top_n = None

    sector_map = _sector_map_from_csv(csv_path)
    _topix, sector_momentum = load_sector_momentum()

    unique_tickers: list[str] = []
    seen: set[str] = set()
    for it in items:
        t = normalize_ticker(str(it.get("ticker") or "")) or str(it.get("ticker") or "")
        if t and t not in seen:
            seen.add(t)
            unique_tickers.append(t)
    fundamentals = fetch_fundamentals_parallel(unique_tickers)

    enriched: list[dict[str, Any]] = []
    dropped_primary = dropped_ma = dropped_score = unscored_kept = 0

    for item in items:
        raw_t = str(item.get("ticker") or "")
        ticker = normalize_ticker(raw_t) or raw_t
        df = bulk_ohlcv.get(ticker)
        if df is None:
            df = bulk_ohlcv.get(raw_t)

        if df is None or df.empty:
            if mode == "filter":
                dropped_primary += 1
                continue
            enriched.append(dict(item))
            unscored_kept += 1
            continue

        tech = analyze_technical(df)
        if tech is None:
            if mode == "filter":
                dropped_primary += 1
                continue
            enriched.append(dict(item))
            unscored_kept += 1
            continue

        if mode == "filter":
            if not passes_primary_filter(tech.price, tech.vol_20d_avg):
                dropped_primary += 1
                continue
            if not tech.above_75ma:
                dropped_ma += 1
                continue

        fund = fundamentals.get(ticker) or fundamentals.get(raw_t)
        if fund is None:
            fund = FundamentalSnapshot(
                roe_pct=ROE_DEFAULT_PCT,
                trailing_eps=None,
                trailing_pe=None,
                roe_is_default=True,
            )

        sec_code = sector_map.get(ticker)
        sec_mom: SectorMomentum | None = None
        if sec_code is not None and sec_code in sector_momentum:
            sec_mom = sector_momentum[sec_code]

        breakdown: ScoreBreakdown = compute_score(tech, fund, sec_mom)
        if mode == "filter" and breakdown.total < min_score:
            dropped_score += 1
            continue

        enriched.append(_enrich_item(item, ticker, tech, fund, sec_mom, breakdown))

    def _sort_key(x: dict[str, Any]) -> tuple[float, float]:
        qs = x.get("quadrant_score")
        score = float(qs) if qs is not None else -1.0
        ar = x.get("avg_return_pct")
        avg = float(ar) if ar is not None else 0.0
        return (-score, -avg)

    enriched.sort(key=_sort_key)
    if top_n and top_n > 0:
        enriched = enriched[:top_n]

    stats = QuadrantApplyStats(
        before=len(items),
        after=len(enriched),
        dropped_primary=dropped_primary,
        dropped_ma=dropped_ma,
        dropped_score=dropped_score,
        unscored_kept=unscored_kept,
        mode=mode,
    )
    return enriched, stats
