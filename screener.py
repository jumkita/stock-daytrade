# -*- coding: utf-8 -*-
"""
市場スクリーニング: 注目銘柄リストを一括スキャンし、乖離率＋直近買いパターンでフィルタ
API制限回避のためリクエスト間に time.sleep(1.0) を挿入。デバッグ用に全銘柄のスキャン結果を返す。
"""
import copy
import time
from typing import Callable, Optional

import pandas as pd

from backtest_engine import BacktestEngine
from logic import (
    sotp_full,
    fetch_ohlcv,
    detect_all_patterns,
    get_downtrend_mask,
    qualitative_audit_batch_with_gemini,
)

# yfinance の連続アクセス規制回避のための待機秒数
SCREENER_SLEEP_SECONDS = 0.5

# 対象銘柄: CSV または東証リストから取得、失敗時は日経225でフォールバック
from ticker_universe import get_ticker_universe
TARGET_TICKERS = get_ticker_universe()


def get_ticker_name(ticker: str) -> str:
    """銘柄コードから表示用名称を取得（yfinance の info から）。"""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        for key in ("shortName", "longName", "symbol"):
            if isinstance(info.get(key), str) and info[key].strip():
                return info[key].strip()
    except Exception:
        pass
    return ticker


def _fmt_num(x, decimals=1):
    """数値を文字列に。None/不正は '-' を返す。"""
    if x is None:
        return "-"
    try:
        if isinstance(x, (int, float)):
            return str(round(float(x), decimals)) if x == x else "-"
        return str(x)
    except Exception:
        return "-"


def get_fundamental_line(
    ticker: str,
    current_price: Optional[float],
    upside: Optional[float],
    buy_signals: str,
) -> str:
    """
    yfinance の info から PER, PBR, ROE, 配当利回り, 時価総額, 業種 を取得し、
    AIに送る1行文字列を組み立てる。取得失敗時は '-' とする。
    ※AIには「数字そのものではなく、その数字が示唆する企業の体質を読み解け」と別途プロンプトで指示する。
    """
    per_str = pbr_str = roe_str = div_str = mcap_str = sector_str = "-"
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        sector_str = (info.get("sector") or "-") if isinstance(info.get("sector"), str) else "-"
        trailing_pe = info.get("trailingPE")
        per_str = f"{_fmt_num(trailing_pe)}x" if trailing_pe is not None else "-"
        price_to_book = info.get("priceToBook")
        pbr_str = f"{_fmt_num(price_to_book)}x" if price_to_book is not None else "-"
        roe = info.get("returnOnEquity")
        if roe is not None and isinstance(roe, (int, float)):
            roe_str = f"{round(roe * 100, 1)}%" if roe == roe else "-"
        else:
            roe_str = "-"
        div = info.get("dividendYield")
        if div is not None and isinstance(div, (int, float)):
            div_str = f"{round(div * 100, 1)}%" if div == div else "-"
        else:
            div_str = "-"
        mcap = info.get("marketCap")
        if mcap is not None and isinstance(mcap, (int, float)) and mcap == mcap:
            if mcap >= 1e12:
                mcap_str = f"{mcap / 1e12:.1f}T"
            elif mcap >= 1e9:
                mcap_str = f"{mcap / 1e9:.1f}B"
            elif mcap >= 1e6:
                mcap_str = f"{mcap / 1e6:.1f}M"
            else:
                mcap_str = _fmt_num(mcap, 0)
        else:
            mcap_str = "-"
    except Exception:
        pass
    price_str = f"{int(current_price)}円" if current_price is not None else "-"
    upside_str = f"+{upside}%" if upside is not None else "-"
    pattern_str = buy_signals if buy_signals else "-"
    if pattern_str != "-" and any(p in pattern_str for p in ("逆三尊", "ピンバー")):
        pattern_str = f"{pattern_str} [NEW SIGNAL]"
    return f"{ticker} | 現在値:{price_str} (Upside {upside_str}) | 業種:{sector_str} | PBR:{pbr_str} | PER:{per_str} | ROE:{roe_str} | 配当:{div_str} | 時価:{mcap_str} | パターン:{pattern_str}"


def run_screen(
    ebitda_multiple: float = 10.0,
    min_deviation_pct: float = 20.0,
    recent_days: int = 3,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    enable_gemini_audit: bool = False,
    streamlit_secrets: Optional[object] = None,
    audit_progress_callback: Optional[Callable[..., None]] = None,
    holding_days: int = 5,
    stop_loss_pct: float = 0.05,
    min_win_rate: float = 0.5,
) -> dict:
    """
    注目銘柄リストを一括スキャンし、条件を満たす銘柄を返す。
    条件: 直近 recent_days 以内に「勝率・収益性の高いサイン」（バックテストで Win Rate >= min_win_rate,
    Profit Factor >= 1.0, Total Trades >= 5 を満たすシグナル）が1つ以上出た銘柄。
    さらに乖離率 >= min_deviation_pct の場合は表示（min_deviation_pct=0 なら乖離率は問わない）。
    戻り値: {"results": [...], "debug": [...]}
    """
    results = []
    debug_list = []
    total = len(TARGET_TICKERS)
    engine = BacktestEngine()

    for idx, ticker in enumerate(TARGET_TICKERS):
        if stop_check and stop_check():
            break
        if progress_callback:
            progress_callback(idx + 1, total, ticker)

        time.sleep(SCREENER_SLEEP_SECONDS)

        try:
            df = fetch_ohlcv(ticker, period="1mo", interval="1d")
        except Exception:
            debug_list.append({
                "ticker": ticker,
                "price": None,
                "model_type": "—",
                "theoretical_price": None,
                "upside_pct": None,
                "status": "OHLC Error",
            })
            continue
        if df is None or len(df) < 2:
            continue

        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("¥", "", regex=False).str.replace(",", "", regex=False),
                    errors="coerce",
                )

        try:
            patterns = detect_all_patterns(df)
        except Exception:
            patterns = []

        for i, name, side in patterns:
            col = f"Buy_{name}" if side == "buy" else f"Sell_{name}"
            if col not in df.columns:
                df[col] = False
            df.loc[df.index[i], col] = True

        signal_cols = [c for c in df.columns if c.startswith("Buy_")]
        if not signal_cols:
            continue

        try:
            bt_results = engine.run(
                df,
                signal_columns=signal_cols,
                holding_period_days=holding_days,
                stop_loss_pct=stop_loss_pct,
            )
        except Exception:
            continue

        valid_results = bt_results[
            (bt_results["Total Trades"] >= 5)
            & (bt_results["Win Rate"] >= min_win_rate)
            & (bt_results["Profit Factor"] >= 1.0)
        ]
        valid_signals = valid_results["Signal Name"].tolist()
        if not valid_signals:
            continue

        n = len(df)
        fired_valid = []
        for i in range(max(0, n - recent_days), n):
            for sig in valid_signals:
                if sig in df.columns and df[sig].iloc[i]:
                    label = sig.replace("Buy_", "")
                    if label not in fired_valid:
                        fired_valid.append(label)
        if not fired_valid:
            continue

        time.sleep(SCREENER_SLEEP_SECONDS)
        try:
            sotp = sotp_full(ticker, ebitda_multiple=ebitda_multiple)
        except Exception:
            debug_list.append({
                "ticker": ticker,
                "price": None,
                "model_type": "—",
                "theoretical_price": None,
                "upside_pct": None,
                "status": "SOTP Error",
            })
            continue

        _raw = sotp.get("current_price")
        current = None
        if _raw is not None:
            try:
                current = pd.to_numeric(str(_raw).replace("¥", "").replace(",", ""), errors="coerce")
            except (TypeError, ValueError):
                current = None
        if current is not None and (pd.isna(current) or current <= 0):
            current = None

        theoretical = sotp.get("theoretical_price")
        deviation_pct = sotp.get("deviation_pct")
        if theoretical is not None and current is not None and current > 0:
            deviation_pct = round((theoretical - current) / current * 100, 1) if deviation_pct is None else round(deviation_pct, 1)
        else:
            deviation_pct = None

        debug_list.append({
            "ticker": ticker,
            "price": current,
            "model_type": sotp.get("model_type") or "—",
            "theoretical_price": theoretical,
            "upside_pct": deviation_pct,
            "status": "Success",
        })

        if min_deviation_pct > 0 and (deviation_pct is None or deviation_pct < min_deviation_pct):
            continue

        name = get_ticker_name(ticker)
        stop_loss_price = float(current * 0.95) if current and current > 0 else None
        record = {
            "ticker": ticker,
            "name": name,
            "current_price": current,
            "stop_loss_price": stop_loss_price,
            "theoretical_price": theoretical,
            "deviation_pct": deviation_pct,
            "buy_signals": ", ".join(fired_valid),
            "ai_rank": "—",
            "strategist_eye": "",
            "verdict": "OK",
        }
        results.append(record)

    results.sort(key=lambda x: (x["deviation_pct"] if x.get("deviation_pct") is not None else -999), reverse=True)

    # 監査は3銘柄ごとのバッチで実行し、バッチごとに結果を画面へ反映（ファンダメンタル注入）
    if enable_gemini_audit and results:
        items = []
        for r in results:
            line = get_fundamental_line(
                r["ticker"],
                r.get("current_price"),
                r.get("deviation_pct"),
                r.get("buy_signals") or "",
            )
            items.append({
                "ticker": r["ticker"],
                "upside": r["deviation_pct"],
                "line": line,
            })
        try:
            batch_size = 3
            chunks = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
            all_audits = []
            for batch_idx, chunk in enumerate(chunks):
                audits_chunk = qualitative_audit_batch_with_gemini(
                    chunk,
                    streamlit_secrets=streamlit_secrets,
                    batch_size=batch_size,
                )
                all_audits.extend(audits_chunk)
                for i, audit in enumerate(all_audits):
                    if i < len(results):
                        results[i]["ai_rank"] = audit.get("rank", "—")
                        results[i]["strategist_eye"] = audit.get("conclusion", "")
                        results[i]["verdict"] = audit.get("verdict", "OK")
                if audit_progress_callback:
                    msg = f"API制限回避のため、10秒ごとに慎重に精査中です（全{len(items)}銘柄）... 現在 {len(all_audits)}/{len(items)} 銘柄"
                    try:
                        audit_progress_callback(
                            len(all_audits), len(items), msg,
                            results_so_far=copy.deepcopy(results),
                        )
                    except TypeError:
                        audit_progress_callback(len(all_audits), len(items), msg)
        except Exception:
            pass

    return {"results": results, "debug": debug_list}
