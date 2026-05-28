"""新配点で daily_buy_signals を再スコアリングして確認する。"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrant_screening.config import SCORE_MAX
from quadrant_screening.integrate import apply_quadrant_to_backtest_items
from quadrant_screening.market_data import fetch_ohlcv_bulk

JSON_URL = "https://raw.githubusercontent.com/jumkita/stock-daytrade/main/daily_buy_signals.json"


def main() -> None:
    d = json.loads(urllib.request.urlopen(JSON_URL).read())
    items = d.get("items") or []
    tickers = sorted({x["ticker"] for x in items})
    bulk = fetch_ohlcv_bulk(tickers, period="6mo")
    csv = ROOT / "jpx_all_tickers.csv"
    out, stats = apply_quadrant_to_backtest_items(items, bulk, csv, mode="rank")
    scores = [x["quadrant_score"] for x in out if x.get("quadrant_score") is not None]
    print(f"SCORE_MAX={SCORE_MAX}")
    print(f"scored={len(scores)}/{len(out)} max={max(scores)} min={min(scores)}")
    for x in out[:8]:
        b = x.get("quadrant_breakdown") or {}
        total = x.get("quadrant_score")
        bsum = round(sum(b.values()), 1)
        ok = "OK" if bsum == total else "MISMATCH"
        print(
            f"{x['ticker']} {x.get('pattern_name')} "
            f"total={total} breakdown={bsum} [{ok}] {b}"
        )


if __name__ == "__main__":
    main()
