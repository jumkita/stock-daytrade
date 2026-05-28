"""4象限スクリーニングエンジン（バルク取得・並列ファンダ）。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quadrant_screening.config import TOP_N_DEFAULT
from quadrant_screening.fundamentals import fetch_fundamentals_parallel
from quadrant_screening.market_data import fetch_ohlcv_bulk
from quadrant_screening.output import ScreenCandidate, format_output
from quadrant_screening.scoring import compute_score
from quadrant_screening.sector import SectorMomentum, load_sector_momentum
from quadrant_screening.technical import (
    analyze_technical,
    compute_tp_sl,
    passes_primary_filter,
)
from quadrant_screening.universe import load_prime_universe


@dataclass
class RunStats:
    universe_size: int
    primary_pass: int
    ma_pass: int
    pattern_or_volume: int
    elapsed_sec: float


def run_quadrant_screen(
    csv_path: Path,
    top_n: int = TOP_N_DEFAULT,
    require_pattern: bool = False,
    limit: int | None = None,
) -> tuple[list[ScreenCandidate], RunStats]:
    """
    プライム母集団 → 一次フィルター → 75MA必須 → スコア順 top_n。
    require_pattern=True のとき買いパターン点灯銘柄のみ（厳格モード）。
    """
    t0 = time.perf_counter()
    universe = load_prime_universe(csv_path)
    if limit is not None and limit > 0:
        universe = universe.head(limit)
    tickers = universe["ticker"].tolist()
    name_map = dict(zip(universe["ticker"], universe["name"]))
    sector_map: dict[str, int | None] = {}
    for _, row in universe.iterrows():
        sec = row.get("sector_code_17")
        sector_map[row["ticker"]] = int(sec) if pd.notna(sec) else None

    topix_ret, sector_momentum = load_sector_momentum()
    ohlcv = fetch_ohlcv_bulk(tickers)

    primary_tickers: list[str] = []
    tech_snap: dict[str, object] = {}

    for t in tickers:
        df = ohlcv.get(t)
        if df is None or df.empty:
            continue
        tech = analyze_technical(df)
        if tech is None:
            continue
        if not passes_primary_filter(tech.price, tech.vol_20d_avg):
            continue
        if not tech.above_75ma:
            continue
        if require_pattern and not tech.patterns:
            continue
        primary_tickers.append(t)
        tech_snap[t] = tech

    fundamentals = fetch_fundamentals_parallel(primary_tickers)

    candidates: list[ScreenCandidate] = []
    pattern_or_vol = 0
    for t in primary_tickers:
        tech = tech_snap[t]
        fund = fundamentals.get(t)
        if fund is None:
            continue
        sec_code = sector_map.get(t)
        sec_mom: SectorMomentum | None = None
        if sec_code is not None and sec_code in sector_momentum:
            sec_mom = sector_momentum[sec_code]

        score = compute_score(tech, fund, sec_mom, sector_code=sec_code, ticker=t)
        sign = "、".join(tech.patterns) if tech.patterns else "トレンド継続"
        if tech.patterns or tech.vol_ratio >= 1.15:
            pattern_or_vol += 1

        tp, sl = compute_tp_sl(tech.price, tech.atr14)
        candidates.append(
            ScreenCandidate(
                ticker=t,
                name=name_map.get(t, t),
                score=score,
                sign=sign,
                vol_ratio=tech.vol_ratio,
                roe_pct=fund.roe_pct,
                price=tech.price,
                tp=tp,
                sl=sl,
            )
        )

    candidates.sort(key=lambda c: c.score.total, reverse=True)
    elapsed = time.perf_counter() - t0
    stats = RunStats(
        universe_size=len(tickers),
        primary_pass=len(primary_tickers),
        ma_pass=len(primary_tickers),
        pattern_or_volume=pattern_or_vol,
        elapsed_sec=elapsed,
    )
    return candidates[:top_n], stats


def run_and_format(csv_path: Path, top_n: int = TOP_N_DEFAULT, limit: int | None = None) -> str:
    top, stats = run_quadrant_screen(csv_path, top_n=top_n, limit=limit)
    header = (
        f"# 4象限スクリーニング | 母集団 {stats.universe_size} | "
        f"通過 {stats.primary_pass} | 所要 {stats.elapsed_sec:.1f}s | TOPIX比較済"
    )
    body = format_output(top) if top else "（該当銘柄なし）"
    return f"{header}\n{body}"
