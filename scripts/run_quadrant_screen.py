#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""4象限スクリーニング CLI（GitHub Actions 5分以内想定）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quadrant_screening.engine import run_and_format, run_quadrant_screen


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="4象限スクリーニング（プライム・最適化版）")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "jpx_all_tickers.csv",
        help="JPX銘柄CSV（プライム抽出用）",
    )
    parser.add_argument("--top", type=int, default=5, help="出力件数")
    parser.add_argument("--limit", type=int, default=None, help="母集団を先頭N件に制限（デバッグ用）")
    parser.add_argument(
        "--require-pattern",
        action="store_true",
        help="買いパターン点灯銘柄のみ（件数が少なくなる）",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"CSVが見つかりません: {args.csv}", file=sys.stderr)
        sys.exit(1)

    if args.require_pattern:
        top, stats = run_quadrant_screen(
            args.csv, top_n=args.top, require_pattern=True, limit=args.limit
        )
        header = (
            f"# 4象限スクリーニング（パターン必須）| 母集団 {stats.universe_size} | "
            f"通過 {stats.primary_pass} | 所要 {stats.elapsed_sec:.1f}s"
        )
        from quadrant_screening.output import format_output

        print(header)
        print(format_output(top) if top else "（該当銘柄なし）")
    else:
        print(run_and_format(args.csv, top_n=args.top, limit=args.limit))


if __name__ == "__main__":
    main()
