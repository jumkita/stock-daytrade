# -*- coding: utf-8 -*-
"""
買いシグナル点灯銘柄を抽出し、X (Twitter) へ自動投稿するバッチ。
乖離率・AI判定は使わず、大引け日（その日足）で買いサインが出ている銘柄のみを対象とする。
"""
from __future__ import annotations

import os
import sys

# スクリプト配置ディレクトリとカレントを sys.path の先頭に（Actions 等で import 失敗しないよう）
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir and _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_cwd = os.getcwd()
if _cwd and _cwd not in sys.path:
    sys.path.insert(0, _cwd)

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

# yfinance の delisted/404 などの ERROR ログを抑制（銘柄スキャン時に大量に出るため）
logging.getLogger("yfinance").setLevel(logging.WARNING)

import pandas as pd

from logic import (
    fetch_ohlcv,
    detect_all_patterns,
    classify_signal_status,
    compute_tp_sl,
    build_signal_rationale,
    compute_conviction_score,
    is_provisional_market_session,
)
from screener import TARGET_TICKERS, get_ticker_name

MAX_TWEET_LEN = 280
PICK_MAX = 3
SLEEP_SEC = 0.5


def scan_buy_signal_only() -> dict[str, list[dict[str, Any]]]:
    """
    大引け日で買いサインが出た銘柄を「本命」「出来高待ち」「押し目待ち」に分類して返す。
    Returns:
        {"active": [本命のみ], "watch": [出来高待ち・押し目待ち]}。X投稿・JSON の picked は active のみ使用。
    """
    provisional = is_provisional_market_session()
    active: list[dict[str, Any]] = []
    watch: list[dict[str, Any]] = []
    for idx, ticker in enumerate(TARGET_TICKERS):
        time.sleep(SLEEP_SEC)
        try:
            df = fetch_ohlcv(ticker, period="3mo", interval="1d")
        except Exception:
            continue
        if df is None or len(df) < 76:
            continue
        try:
            patterns = detect_all_patterns(df)
        except Exception:
            patterns = []
        n = len(df)
        buy_on_last_day = [(i, name, side) for i, name, side in patterns if side == "buy" and i == n - 1]
        if not buy_on_last_day:
            continue
        status_reason = classify_signal_status(df, n - 1, provisional=provisional)
        if status_reason is None:
            continue
        status, reason_short = status_reason
        buy_names = [name for _, name, _ in buy_on_last_day]
        tp_sl = compute_tp_sl(df, bar_index=n - 1)
        rationale = build_signal_rationale(
            df, n - 1,
            multiple=1.4 if provisional else 1.5,
            ma_pct_display=7 if provisional else None,
        )
        name = get_ticker_name(ticker)
        conviction = compute_conviction_score(df, n - 1)
        item = {
            "ticker": ticker,
            "name": name,
            "buy_signals": ", ".join(buy_names),
            "signal_count": len(buy_names),
            "entry": tp_sl.get("entry"),
            "tp": tp_sl.get("tp"),
            "sl": tp_sl.get("sl"),
            "rationale": rationale,
            "provisional": provisional,
            "status": status,
            "reason_short": reason_short,
            "conviction_score": conviction,
        }
        if status == "active":
            active.append(item)
        else:
            watch.append(item)
    return {"active": active, "watch": watch}


def build_tweet(picked: list[dict[str, Any]]) -> str:
    """投稿文を組み立てる（確信度上位最大3銘柄・エントリー・TP・SL・根拠付き）。280文字を超えないよう調整する。"""
    provisional = any(r.get("provisional") for r in picked)
    lines = ["【本日の厳選3銘柄】"]
    if provisional:
        lines.append("※15:15暫定（大引け前の暫定値・TP/SLは暫定終値ベース）")
    for r in picked:
        lines.append(f"■ {r['name']} ({r['ticker']})")
        lines.append(f"・シグナル: {r['buy_signals']}")
        entry = r.get("entry")
        tp_val = r.get("tp")
        sl_val = r.get("sl")
        if entry is not None:
            lines.append(f"・エントリー想定: ¥{entry:,.0f}")
        if tp_val is not None:
            lines.append(f"・利確(TP): ¥{tp_val:,.0f}")
        if sl_val is not None:
            lines.append(f"・損切り(SL): ¥{sl_val:,.0f}")
        rationale = r.get("rationale") or "—"
        lines.append(f"・根拠: {rationale}")
    lines.append("")
    lines.append("※機械的スクリーニング結果。投資判断は自己責任で。")
    lines.append("#日本株 #プライスアクション")
    text = "\n".join(lines)
    if len(text) <= MAX_TWEET_LEN:
        return text
    if len(picked) > 1:
        return build_tweet(picked[:1])
    single = {**picked[0], "buy_signals": (picked[0].get("buy_signals") or "")[:47] + "…"}
    return build_tweet([single])


def post_to_x(text: str) -> tuple[bool, str | None]:
    """
    tweepy で X に投稿する。
    Returns:
        (成功したか, エラーメッセージ or None)
    """
    try:
        import tweepy
    except ImportError:
        print("tweepy がインストールされていません。pip install tweepy", file=sys.stderr)
        return False, "ImportError"

    api_key = os.environ.get("X_API_KEY", "").strip()
    api_secret = os.environ.get("X_API_SECRET", "").strip()
    access_token = os.environ.get("X_ACCESS_TOKEN", "").strip()
    access_secret = os.environ.get("X_ACCESS_TOKEN_SECRET", "").strip()

    if not all([api_key, api_secret, access_token, access_secret]):
        print("X API の環境変数が未設定です。X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET", file=sys.stderr)
        return False, "env"

    try:
        client = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
        )
        client.create_tweet(text=text)
        return True, None
    except Exception as e:
        err = str(e)
        print(f"X 投稿エラー: {e}", file=sys.stderr)
        return False, err


def main() -> int:
    scan_result = scan_buy_signal_only()
    active = scan_result["active"]
    watch = scan_result["watch"]
    combined = active + watch
    combined.sort(key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
    picked = combined[:PICK_MAX]
    all_results = active

    tweet_text = "本日は買いシグナル点灯銘柄はありませんでした。"
    if picked:
        tweet_text = build_tweet(picked)
        print(tweet_text)
        print("---")

    # ワークフロー用: 本命(active)・ウォッチ(watch) を JSON で保存。X 投稿は確信度上位最大3銘柄。
    json_path = os.environ.get("DAILY_SIGNALS_JSON_PATH", "").strip()
    if json_path:
        try:
            data = {
                "updated": datetime.now(timezone.utc).isoformat(),
                "all": all_results,
                "picked": picked,
                "tweet_text": tweet_text,
                "watch": watch,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"結果を保存しました: {json_path}")
        except Exception as e:
            print(f"JSON 保存エラー: {e}", file=sys.stderr)

    if not picked:
        return 0

    ok, err = post_to_x(tweet_text)
    if ok:
        print("投稿しました。")
        return 0
    # 402 Payment Required = API のクレジット不足。スクリーニングは成功しているので exit 0 にする
    if err and ("402" in err or "Payment Required" in err):
        print("※X 投稿はスキップしました（API クレジット不足 402）。上記はスクリーニング結果です。", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
