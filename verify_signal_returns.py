# -*- coding: utf-8 -*-
"""
GitHub（daily_buy_signals.json）で出力された銘柄を「シグナル日の引けで購入」した場合の
実際のリターンを検証する。

【前提】GitHub Actions は平日 15:00 JST に実行され、daily_buy_signals.json を上書きする。
        JSON の "updated" はその実行日時（UTC）、"items" はその日に買いシグナルが出た銘柄リスト。

【検証ロジック】
1. シグナル日: "updated" を JST に変換した「日付」をシグナル日とする（15:00 実行ならその日がシグナル日）。
2. エントリー: 各銘柄の "entry"（＝その日時点の終値に相当）を「引けで約定した買い価格」とする。
3. 利確: シグナル日を 0 営業日目とし、その「N 営業日後」の終値を利確価格とする（N は既定 3、変更可）。
4. リターン: (利確価格 - エントリー価格) / エントリー価格 × 100（％）。
5. データ: 各銘柄について yfinance で約 2 営業月の日足を取得し、上記の日付に対応する終値を参照。

【注意】JSON は毎回上書きされるため、過去のシグナルを検証するには、その日の JSON が別途必要
       （ワークフローで日付付きコピーを保存する、または GitHub の過去コミットから取得する）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pandas as pd

# スクリプト配置を path に
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in (os.getcwd(), ""):
    import sys
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

from logic import fetch_ohlcv

JST = timezone(timedelta(hours=9))


def _parse_signal_date_jst(updated_iso: Optional[str]) -> Optional[datetime]:
    """updated の ISO 文字列を JST の datetime に。日付のみ使うので date でも可。"""
    if not updated_iso or not isinstance(updated_iso, str):
        return None
    try:
        dt = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(JST)
    except Exception:
        return None


def _to_date_key(dt) -> Optional[Any]:
    """DataFrame の Date 列と比較しやすい形に。"""
    if hasattr(dt, "date"):
        return dt.date()
    if hasattr(dt, "normalize"):
        return dt.normalize().date() if hasattr(dt.normalize(), "date") else dt
    return dt


def verify_returns(
    json_path_or_data: str | dict,
    holding_days: int = 3,
) -> list[dict[str, Any]]:
    """
    daily_buy_signals の内容を読み、各銘柄について「シグナル日引けで購入 → 保有期間後の終値で利確」とした
    リターン（%）を算出する。

    Args:
        json_path_or_data: JSON ファイルパス、または {"items": [...], "updated": "ISO"} の dict（アプリ用）
        holding_days: 保有営業日数（利確日）

    Returns:
        各 item ごとの dict: ticker, name, pattern_name, entry, exit_price, return_pct, signal_date, exit_date
    """
    if isinstance(json_path_or_data, dict):
        data = json_path_or_data
    else:
        if not os.path.isfile(json_path_or_data):
            return []
        try:
            with open(json_path_or_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []

    items_raw = data.get("items") or []
    # アプリからは list で渡されることがある（items がそのまま）
    items = items_raw if isinstance(items_raw, list) else []
    if not items:
        return []

    updated = data.get("updated")
    signal_dt = _parse_signal_date_jst(updated)
    if signal_dt is None:
        signal_dt = datetime.now(JST)
    signal_date = signal_dt.date()

    results: list[dict[str, Any]] = []
    for it in items:
        ticker = it.get("ticker")
        if not ticker:
            continue
        entry = it.get("entry")
        try:
            entry_f = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            entry_f = None
        if entry_f is None or entry_f <= 0:
            continue

        # シグナル日を含む十分な期間の OHLCV を取得（最大で約2営業月）
        df = fetch_ohlcv(ticker, period="2mo", interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            results.append({
                "ticker": ticker,
                "name": it.get("name") or "",
                "pattern_name": it.get("pattern_name") or "",
                "entry": entry_f,
                "exit_price": None,
                "return_pct": None,
                "signal_date": str(signal_date),
                "exit_date": None,
                "error": "データ取得不可",
            })
            continue

        if "Date" not in df.columns and df.index.name is not None:
            df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], utc=False).dt.tz_localize(None)
        df["_date"] = df[date_col].apply(lambda x: x.date() if hasattr(x, "date") else x)

        # シグナル日以降の行だけに絞り、営業日順で N 日後を取得
        after = df[df["_date"] >= signal_date].sort_values(date_col).reset_index(drop=True)
        if after.empty:
            results.append({
                "ticker": ticker,
                "name": it.get("name") or "",
                "pattern_name": it.get("pattern_name") or "",
                "entry": entry_f,
                "exit_price": None,
                "return_pct": None,
                "signal_date": str(signal_date),
                "exit_date": None,
                "error": "シグナル日以降の足なし",
            })
            continue

        # エントリー価格: シグナル日当日の終値（なければ翌日以降の最初の終値）
        entry_row = after.iloc[0]
        entry_close = float(entry_row["Close"])
        if entry_close <= 0:
            entry_close = entry_f

        # 保有 N 営業日後の終値
        if len(after) <= holding_days:
            exit_row = after.iloc[-1]
            exit_date_actual = after.iloc[-1]["_date"]
        else:
            exit_row = after.iloc[holding_days]
            exit_date_actual = after.iloc[holding_days]["_date"]
        exit_price = float(exit_row["Close"])
        return_pct = (exit_price - entry_close) / entry_close * 100.0 if entry_close and entry_close > 0 else None

        results.append({
            "ticker": ticker,
            "name": it.get("name") or "",
            "pattern_name": it.get("pattern_name") or "",
            "entry": entry_f,
            "exit_price": exit_price,
            "return_pct": round(return_pct, 2) if return_pct is not None else None,
            "signal_date": str(signal_date),
            "exit_date": str(exit_date_actual) if exit_date_actual is not None else None,
            "error": None,
        })

    return results


def summary_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """検証結果のサマリ（銘柄数、勝率、平均リターン%）。"""
    valid = [r for r in results if r.get("return_pct") is not None]
    if not valid:
        return {"count": 0, "win_rate": None, "avg_return_pct": None, "total_return_pct": None}
    returns = [r["return_pct"] for r in valid]
    wins = sum(1 for x in returns if x > 0)
    return {
        "count": len(valid),
        "win_rate": round(wins / len(returns) * 100, 1) if returns else None,
        "avg_return_pct": round(sum(returns) / len(returns), 2) if returns else None,
        "total_return_pct": round(sum(returns), 2),
    }


if __name__ == "__main__":
    import sys
    path = os.environ.get("DAILY_SIGNALS_JSON_PATH", "").strip() or os.path.join(_script_dir, "daily_buy_signals.json")
    holding = 3
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            holding = int(sys.argv[2])
        except ValueError:
            pass
    rows = verify_returns(path, holding_days=holding)
    stats = summary_stats(rows)
    print(f"# シグナル引け購入検証（保有 {holding} 営業日）")
    print(f"# 対象: {path}")
    print()
    for r in rows:
        err = r.get("error") or ""
        ret = r.get("return_pct")
        ret_s = f"{ret:+.2f}%" if ret is not None else err or "—"
        print(f"【{r['ticker']}】{r.get('pattern_name','')} | エントリー: ¥{r['entry']:,.0f} → 利確: ¥{r.get('exit_price') or 0:,.0f} | リターン: {ret_s}")
    print()
    print(f"# サマリ: 有効 {stats['count']} 件 | 勝率 {stats['win_rate'] or '—'}% | 平均リターン {stats['avg_return_pct'] or '—'}%")
