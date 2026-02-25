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
from typing import Any, Optional

# yfinance の delisted/404 などの ERROR ログを抑制（銘柄スキャン時に大量に出るため）
logging.getLogger("yfinance").setLevel(logging.WARNING)

import pandas as pd

from logic import (
    fetch_ohlcv,
    detect_all_patterns,
    compute_tp_sl,
    build_signal_rationale,
    compute_conviction_score,
    hybrid_classify_signal,
    watchlist_score,
    watchlist_eligible,
    build_watchlist_reason_short,
    get_volume_ratio,
    get_ma_deviation,
    is_provisional_market_session,
)
from screener import TARGET_TICKERS, get_ticker_name

MAX_TWEET_LEN = 280
PICK_MAX = 3
WATCHLIST_TOP_N = 5
SLEEP_SEC = 0.5


def _safe_float_for_sort(val: Any) -> float:
    """ソート用。NaN/None は 0 にし、比較エラーを防ぐ。"""
    if val is None:
        return 0.0
    try:
        f = float(val)
        if f != f:
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def scan_hybrid() -> dict[str, Any]:
    """
    ハイブリッド判定で 本命(active)・注目(high_potential)・監視(watch) を返す。
    - 本命: Type-A/Type-B で全条件合致（確信度高）。
    - 注目: 条件の8割以上充足（確信度中）。
    - 監視: 条件A（パターン点灯+出来高1.2倍+MA5%以内）または条件B（パターン未点灯+出来高2倍+MA3%以内）を満たす銘柄のみスコア順で上位5件。
    """
    tickers = list(TARGET_TICKERS)
    print(f"DEBUG: 読み込まれた銘柄数 = {len(tickers)}")
    if len(tickers) == 0:
        return {"active": [], "high_potential": [], "watch": []}

    provisional = is_provisional_market_session()
    active: list[dict[str, Any]] = []
    high_potential: list[dict[str, Any]] = []
    watch_candidates: list[dict[str, Any]] = []

    for ticker in tickers:
        print(f"DEBUG: スキャン開始 -> {ticker}")
        time.sleep(SLEEP_SEC)
        try:
            df = fetch_ohlcv(ticker, period="3mo", interval="1d")
        except Exception:
            continue
        if df is None:
            continue
        n = len(df)
        bar = n - 1
        name = get_ticker_name(ticker)

        # 25本以上でパターン検出を1回だけ実行（監視・本命の両方で利用）
        patterns: list = []
        if n >= 25:
            try:
                patterns = detect_all_patterns(df)
            except Exception:
                pass
        buy_on_last_day = [(i, nm, side) for i, nm, side in patterns if side == "buy" and i == bar]

        # 監視リスト: 条件AまたはBを満たす銘柄のみ候補に追加（買いパターンを評価の主軸にしたニアミス）
        if n >= 25:
            vol_ratio = get_volume_ratio(df, bar)
            ma_dev = get_ma_deviation(df, bar)
            pattern_found = len(buy_on_last_day) > 0
            pattern_names = ", ".join(nm for _, nm, _ in buy_on_last_day) if buy_on_last_day else ""
            eligible, condition_ab = watchlist_eligible(pattern_found, vol_ratio, ma_dev)
            if eligible and condition_ab:
                score = watchlist_score(df, bar, pattern_found)
                reason_short = build_watchlist_reason_short(
                    pattern_found, pattern_names, vol_ratio, ma_dev, condition_ab
                )
                tp_sl = compute_tp_sl(df, bar_index=bar)
                watch_candidates.append({
                    "ticker": ticker,
                    "name": name,
                    "watchlist_score": score,
                    "reason_short": reason_short,
                    "entry": tp_sl.get("entry"),
                    "tp": tp_sl.get("tp"),
                    "sl": tp_sl.get("sl"),
                    "conviction_score": score,
                })

        # 本命・注目: 76本以上かつ買いパターン当日点灯の銘柄のみ
        if n < 76:
            continue
        if not buy_on_last_day:
            continue
        classification = hybrid_classify_signal(df, bar, has_buy_pattern=True, provisional=provisional)
        if classification is None:
            continue
        status, reason_short_h = classification
        buy_names = [nm for _, nm, _ in buy_on_last_day]
        tp_sl = compute_tp_sl(df, bar_index=bar)
        rationale = build_signal_rationale(
            df, bar,
            multiple=1.4 if provisional else 1.5,
            ma_pct_display=7 if provisional else None,
        )
        conviction = compute_conviction_score(df, bar)
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
            "reason_short": reason_short_h,
            "conviction_score": conviction,
        }
        if status == "active":
            active.append(item)
        else:
            high_potential.append(item)

    # 監視 = スコア順ソートの上位5件を必ず返す（早期returnなし）
    watch_candidates.sort(key=lambda x: _safe_float_for_sort(x.get("watchlist_score")), reverse=True)
    watch = watch_candidates[:WATCHLIST_TOP_N]

    return {"active": active, "high_potential": high_potential, "watch": watch}


def scan_buy_signal_only() -> dict[str, list[dict[str, Any]]]:
    """
    後方互換: scan_hybrid() を呼び、active / high_potential を active に、
    watch を watch にマッピングして返す（本命＋注目を active に統合する旧形式）。
    """
    r = scan_hybrid()
    active = r["active"]
    high_potential = r["high_potential"]
    watch = r["watch"]
    return {"active": active + high_potential, "watch": watch}


def build_tweet(picked: list[dict[str, Any]], watch_names: Optional[list[str]] = None) -> str:
    """投稿文を組み立てる（本命・注目から確信度上位最大3銘柄。監視は1行で言及）。280文字を超えないよう調整する。"""
    provisional = any(r.get("provisional") for r in picked)
    lines = ["【本日の厳選3銘柄】"]
    if provisional:
        lines.append("※15:15暫定（大引け前の暫定値・TP/SLは暫定終値ベース）")
    for r in picked:
        label = "本命" if r.get("status") == "active" else "注目"
        lines.append(f"■ [{label}] {r['name']} ({r['ticker']})")
        lines.append(f"・シグナル: {r.get('buy_signals', '—')}")
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
    if watch_names and len(text) + 2 + len("【監視】 " + " / ".join(watch_names[:5])) <= MAX_TWEET_LEN:
        text = text + "\n\n【監視】 " + " / ".join(watch_names[:5])
    if len(text) <= MAX_TWEET_LEN:
        return text
    if len(picked) > 1:
        return build_tweet(picked[:1], watch_names)
    single = {**picked[0], "buy_signals": (picked[0].get("buy_signals") or "")[:47] + "…"}
    return build_tweet([single], watch_names)


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
    scan_result = scan_hybrid()
    active = scan_result["active"]
    high_potential = scan_result["high_potential"]
    watch = scan_result["watch"]
    combined = active + high_potential
    combined.sort(key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
    picked = combined[:PICK_MAX]
    watch_names = [w["name"] for w in watch] if watch else None

    tweet_text = "本日は買いシグナル点灯銘柄はありませんでした。"
    if picked:
        tweet_text = build_tweet(picked, watch_names)
        print(tweet_text)
        print("---")

    # ワークフロー用: 本命(active)・注目(high_potential)・監視(watch) を JSON で保存。X 投稿は本命・注目から確信度上位最大3銘柄。
    json_path = os.environ.get("DAILY_SIGNALS_JSON_PATH", "").strip()
    if json_path:
        try:
            data = {
                "updated": datetime.now(timezone.utc).isoformat(),
                "active": active,
                "high_potential": high_potential,
                "watch": watch,
                "all": active + high_potential,
                "picked": picked,
                "tweet_text": tweet_text,
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
