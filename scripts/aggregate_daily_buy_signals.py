#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock-daytrade 形式の daily_buy_signals_YYYY-MM-DD.json を集計する。
売りシグナル（items_sell）は扱わず、買い（items）のみ。

出力:
  - 件数・avg_return_pct の合計・平均
  - ポートフォリオ累積リターン（仮定）: 各日のシグナルに元本を均等割りし、その日の
    平均 avg_return_pct を日次リターンとみなし、日付順に幾何積で複利合成。

注意:
  - avg_return_pct はパターンのバックテスト平均であり、実現リターンではない。
  - stock-daytrade の logic.py では「3営業日ホールド」で算出される:
    パターン点灯日の翌営業日寄りでエントリー、さらに3営業日後の大引けで決済
    （run_backtest_3day_vectorized）。シグナル日引け買いの検証は verify_signal_returns.py
    （既定 holding_days=1＝翌営業日引け）とエントリー価格の定義が異なる点に注意。
  - プレースホルダー（...）のみの JSON はスキップされる。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _iter_signal_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("daily_buy_signals_*.json"))
    return [p for p in files if p.name != "daily_buy_signals.json"]


def _load_json(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _date_from_filename(name: str) -> str | None:
    m = re.search(r"daily_buy_signals_(\d{4}-\d{2}-\d{2})\.json", name)
    return m.group(1) if m else None


def _tp_pct_buy(it: dict) -> float:
    en, tp = it.get("entry"), it.get("tp")
    try:
        en_f, tp_f = float(en), float(tp)
        if en_f and en_f > 0:
            return (tp_f - en_f) / en_f * 100.0
    except (TypeError, ValueError):
        pass
    return 0.0


def aggregate_directory(directory: Path, notional_yen: float | None) -> None:
    files = _iter_signal_files(directory)
    if not files:
        print(f"該当ファイルなし: {directory}", file=sys.stderr)
        sys.exit(1)

    total_buy = 0
    sum_buy_ar = 0.0
    sum_buy_tp = 0.0
    skipped: list[str] = []
    per_file: list[tuple[str, int, float, float, float | None]] = []
    # 日次平均リターン（%）→ ポートフォリオ幾何積用
    daily_mean_returns: list[tuple[str, float]] = []

    for path in files:
        data = _load_json(path)
        if data is None:
            skipped.append(path.name)
            continue
        items = data.get("items") or []
        nb = len(items)
        sb = sum(float(x.get("avg_return_pct") or 0) for x in items)
        tpb = sum(_tp_pct_buy(x) for x in items)
        total_buy += nb
        sum_buy_ar += sb
        sum_buy_tp += tpb
        mean_day = (sb / nb) if nb else None
        if mean_day is not None:
            daily_mean_returns.append((path.name, mean_day))
        per_file.append((path.name, nb, sb, tpb, mean_day))

    # ポートフォリオ累積（同日均等配分・日次複利）
    factor = 1.0
    for _fname, mean_pct in daily_mean_returns:
        factor *= 1.0 + mean_pct / 100.0
    portfolio_cumulative_pct = (factor - 1.0) * 100.0

    n_files = len(files) - len(skipped)
    print(f"# 対象ディレクトリ: {directory.resolve()}")
    print(f"# 集計対象: **買いシグナル（items）のみ**（items_sell は無視）")
    print(f"# 読み取り成功: {n_files} ファイル / スキップ（JSON不正）: {len(skipped)}")
    if skipped:
        print(f"# スキップ: {', '.join(skipped)}")
    print()
    print("## トータル（全ファイル合算）")
    print(f"- 買いシグナル件数: **{total_buy}**")
    print()
    print("### avg_return_pct（パターンのバックテスト平均リターン・%）")
    if total_buy:
        print(f"- 合計: **{sum_buy_ar:.2f}%**")
        print(f"- 1件あたり算術平均: **{(sum_buy_ar / total_buy):.4f}%**")
    print()
    print("### ポートフォリオ累積リターン（仮定）")
    print(
        "- **仮定**: 各日付ファイル＝1営業日。その日の全買いシグナルに元本を均等割りし、"
        "その日のリターンは各シグナルの avg_return_pct の**算術平均**。"
        "日と日の間は**幾何積**でつなぐ（複利）。"
    )
    print(f"- 対象日数（シグナル1件以上の日）: **{len(daily_mean_returns)}** 日")
    print(f"- **累積リターン（概算）: {portfolio_cumulative_pct:.2f}%**")
    print()
    print("### TP 到達までの価格変化率（(TP-entry)/entry×100）の合計（参考）")
    print(f"- 合計: **{sum_buy_tp:.2f}%**")

    if notional_yen and notional_yen > 0 and total_buy:
        pnl_yen = 0.0
        for path in files:
            d = _load_json(path)
            if not d:
                continue
            for x in d.get("items") or []:
                ar = x.get("avg_return_pct")
                if ar is None:
                    continue
                pnl_yen += notional_yen * float(ar) / 100.0
        print()
        print(f"### 概算損益（各買いシグナル {notional_yen:,.0f} 円×1回と仮定・売りなし）")
        print(f"- 合計: **{pnl_yen:,.0f} 円**（手数料・税・約定差は未考慮）")

    print()
    print("## ファイル別（日付順・買いのみ）")
    for name, nb, sb, tpb, mean_day in per_file:
        mean_s = f"{mean_day:.4f}%" if mean_day is not None else "—"
        print(f"- {name}: 買い{nb}件 | avg_return合計 {sb:.2f}% | 日平均 {mean_s} | TP変化率合計 {tpb:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="daily_buy_signals_*.json の集計（買いのみ）")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="JSON が置いてあるディレクトリ（既定: カレント）",
    )
    parser.add_argument(
        "--notional-yen",
        type=float,
        default=None,
        help="1シグナルあたりの想定元本（円）。指定時のみ概算損益を表示。",
    )
    args = parser.parse_args()
    aggregate_directory(Path(args.directory), args.notional_yen)


if __name__ == "__main__":
    main()
