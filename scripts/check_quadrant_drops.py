"""本番 daily_buy_signals の銘柄が filter モードならどうなるか診断する。"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrant_screening.config import MA_PERIOD, MIN_PRICE, MIN_VOL_20D
from quadrant_screening.integrate import apply_quadrant_to_backtest_items
from quadrant_screening.market_data import fetch_ohlcv_bulk
from quadrant_screening.technical import analyze_technical, passes_primary_filter

JSON_URL = "https://raw.githubusercontent.com/jumkita/stock-daytrade/main/daily_buy_signals.json"
MIN_SCORE = 25


def main() -> None:
    d = json.loads(urllib.request.urlopen(JSON_URL).read())
    items = d.get("items") or []
    stats = d.get("quadrant_stats") or {}
    tickers = sorted({x["ticker"] for x in items})
    bulk = fetch_ohlcv_bulk(tickers, period="6mo")
    csv = ROOT / "jpx_all_tickers.csv"

    filtered, fstats = apply_quadrant_to_backtest_items(
        items, bulk, csv, mode="filter", min_score=MIN_SCORE
    )

    print("=== 現在の本番 (rank) ===")
    print(f"mode={stats.get('mode')} | before={stats.get('before')} after={stats.get('after')}")
    print(
        f"dropped: primary={stats.get('dropped_primary')} "
        f"ma={stats.get('dropped_ma')} score={stats.get('dropped_score')}"
    )
    print()
    print(f"=== filter 再シミュレーション (一次+75MA+{MIN_SCORE}点) ===")
    print(f"通過: {fstats.after} / {fstats.before}")
    print(
        f"一次除外: {fstats.dropped_primary}, "
        f"75MA除外: {fstats.dropped_ma}, スコア除外: {fstats.dropped_score}"
    )
    print()

    passed_keys = {(x["ticker"], x.get("pattern_name")) for x in filtered}
    dropped: list[str] = []
    passed: list[str] = []

    print("=== 銘柄別 ===")
    for it in items:
        t = it["ticker"]
        pat = it.get("pattern_name")
        df = bulk.get(t)
        tech = analyze_technical(df) if df is not None and not df.empty else None
        qs = it.get("quadrant_score")
        reasons: list[str] = []
        if tech is None:
            n = len(df) if df is not None else 0
            reasons.append(f"75MA不可(足{n}本<{MA_PERIOD})")
        else:
            if not passes_primary_filter(tech.price, tech.vol_20d_avg):
                reasons.append(
                    f"一次NG(株価{tech.price:.0f}<{MIN_PRICE} "
                    f"or 出来高{tech.vol_20d_avg:.0f}<{MIN_VOL_20D})"
                )
            if not tech.above_75ma:
                reasons.append(f"75MA下(終値{tech.price:.0f} < MA75 {tech.ma75:.0f})")
            if qs is not None and qs < MIN_SCORE:
                reasons.append(f"スコア{qs}点<{MIN_SCORE}")

        line = (
            f"{t} | {pat} | スコア{qs}点 | "
            f"出来高{it.get('vol_ratio')}倍 | {it.get('sector_label')}"
        )
        if (t, pat) in passed_keys:
            passed.append(line)
            print(f"[通過] {line}")
        else:
            dropped.append(f"{line} | 理由: {' / '.join(reasons) or '不明'}")
            print(f"[足切り] {line} | 理由: {' / '.join(reasons) or '不明'}")

    print()
    print(f"=== まとめ: filterなら足切り {len(dropped)}件 ===")
    for x in dropped:
        print(f"  - {x}")


if __name__ == "__main__":
    main()
