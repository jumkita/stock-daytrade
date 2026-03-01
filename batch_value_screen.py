# -*- coding: utf-8 -*-
"""
銘柄リストから適正株価の乖離率を一括算出し、
バリュートラップを排除した「本命の割安株」のみを抽出するバッチ処理。

- 財務データはローカルCSVキャッシュを参照（不足時のみ並列APIで補完してキャッシュ保存）
- セクター別平均PERはローカル辞書で定義（API非依存）
- 足切り: ROE≥8%、現在値>75MA、直近20日平均出来高≥10万株
"""
from __future__ import annotations

import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd

from logic import BULK_CHUNK_SIZE, fetch_ohlcv_bulk
from ticker_universe import get_ticker_universe, get_skip_tickers

try:
    import yfinance as yf
except ImportError:
    yf = None

# ----- キャッシュパス -----
_ROOT = os.path.dirname(os.path.abspath(__file__))
FUNDAMENTALS_CACHE_PATH = os.environ.get(
    "VALUE_SCREEN_FUNDAMENTALS_CACHE",
    os.path.join(_ROOT, "data", "fundamentals_cache.csv"),
)
PRICE_MA_VOLUME_CACHE_PATH = os.environ.get(
    "VALUE_SCREEN_PRICE_CACHE",
    os.path.join(_ROOT, "data", "price_ma_volume_cache.csv"),
)

# ----- セクター別平均PER（ローカル辞書・API非依存） -----
SECTOR_AVG_PER: dict[str, float] = {
    "Technology": 18.0,
    "Information Technology": 18.0,
    "Consumer Cyclical": 14.0,
    "Consumer Defensive": 14.0,
    "Financial Services": 10.0,
    "Financials": 10.0,
    "Healthcare": 22.0,
    "Industrials": 14.0,
    "Basic Materials": 10.0,
    "Energy": 8.0,
    "Utilities": 12.0,
    "Real Estate": 12.0,
    "Communication Services": 14.0,
    "Real Estate Investment Trusts": 12.0,
    "Banks": 8.0,
    "Insurance": 10.0,
}
DEFAULT_SECTOR_PER = 12.0  # 未定義セクター用

# ----- 足切り条件 -----
MIN_ROE_PCT = 8.0
MA_WINDOW = 75
MIN_AVG_VOLUME_20D = 100_000
TOP_N = 10


def _ensure_data_dir():
    d = os.path.dirname(FUNDAMENTALS_CACHE_PATH)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    d = os.path.dirname(PRICE_MA_VOLUME_CACHE_PATH)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if hasattr(x, "item"):
            x = x.item()
        v = float(x)
        return v if v == v and v != float("inf") else default
    except (TypeError, ValueError):
        return default


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s if s and s.lower() not in ("nan", "none", "-", "—") else ""


def load_fundamentals_cache() -> dict[str, dict[str, Any]]:
    """CSV から 銘柄 -> {eps, sector, roe} を読み込む。"""
    out: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(FUNDAMENTALS_CACHE_PATH):
        return out
    try:
        with open(FUNDAMENTALS_CACHE_PATH, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                t = _safe_str(row.get("ticker", "")).strip()
                if not t or not t.endswith(".T"):
                    continue
                raw_eps = row.get("eps", "")
                eps = None
                if raw_eps is not None and _safe_str(raw_eps):
                    v = _safe_float(raw_eps, default=0.0)
                    if v > 0 and v == v:
                        eps = v
                sector = _safe_str(row.get("sector", ""))
                raw_roe = row.get("roe", "")
                roe = None
                if raw_roe is not None and _safe_str(raw_roe):
                    v = _safe_float(raw_roe, default=0.0)
                    if v == v:
                        # 0.08 形式または 8.0 形式の両対応
                        roe = v if v <= 1.5 else v / 100.0
                out[t] = {"eps": eps, "sector": sector or None, "roe": roe}
    except Exception:
        pass
    return out


def save_fundamentals_cache(data: dict[str, dict[str, Any]]) -> None:
    _ensure_data_dir()
    with open(FUNDAMENTALS_CACHE_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "eps", "sector", "roe"])
        w.writeheader()
        for ticker, v in sorted(data.items()):
            eps = v.get("eps")
            w.writerow({
                "ticker": ticker,
                "eps": "" if eps is None else f"{eps:.6f}",
                "sector": v.get("sector") or "",
                "roe": "" if v.get("roe") is None else f"{v.get('roe'):.6f}",
            })


def _fetch_single_fundamentals(ticker: str) -> Optional[dict[str, Any]]:
    """1銘柄の info から eps, sector, roe を取得（並列ワーカー用）。"""
    if not yf:
        return None
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", None) or {}
        eps = info.get("trailingEps")
        eps_f = _safe_float(eps, default=None)
        if eps_f is not None and (eps_f <= 0 or eps_f != eps_f):
            eps_f = None
        sector = _safe_str(info.get("sector") or info.get("industry", ""))
        roe = info.get("returnOnEquity")
        roe_f = _safe_float(roe, default=None) if roe is not None else None
        return {"ticker": ticker, "eps": eps_f, "sector": sector or None, "roe": roe_f}
    except Exception:
        return None


def refill_fundamentals_parallel(tickers: list[str], max_workers: int = 12) -> dict[str, dict[str, Any]]:
    """
    キャッシュにない銘柄について、並列で yfinance .info を取得し、
    結果を {ticker: {eps, sector, roe}} で返す。1件ずつの直列リクエストは行わない。
    """
    if not yf or not tickers:
        return {}
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_single_fundamentals, t): t for t in tickers}
        for fut in as_completed(futures):
            try:
                row = fut.result()
                if row and row.get("ticker"):
                    results[row["ticker"]] = {
                        "eps": row.get("eps"),
                        "sector": row.get("sector"),
                        "roe": row.get("roe"),
                    }
            except Exception:
                pass
    return results


def load_price_ma_volume_cache() -> dict[str, dict[str, Any]]:
    """CSV から 銘柄 -> {current_price, ma75, avg_volume_20d} を読み込む。"""
    out: dict[str, dict[str, Any]] = {}
    if not os.path.isfile(PRICE_MA_VOLUME_CACHE_PATH):
        return out
    try:
        with open(PRICE_MA_VOLUME_CACHE_PATH, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                t = _safe_str(row.get("ticker", "")).strip()
                if not t or not t.endswith(".T"):
                    continue
                out[t] = {
                    "current_price": _safe_float(row.get("current_price"), 0.0),
                    "ma75": _safe_float(row.get("ma75"), 0.0),
                    "avg_volume_20d": _safe_float(row.get("avg_volume_20d"), 0.0),
                }
    except Exception:
        pass
    return out


def save_price_ma_volume_cache(data: dict[str, dict[str, Any]]) -> None:
    _ensure_data_dir()
    with open(PRICE_MA_VOLUME_CACHE_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ticker", "current_price", "ma75", "avg_volume_20d"],
        )
        w.writeheader()
        for ticker, v in sorted(data.items()):
            w.writerow({
                "ticker": ticker,
                "current_price": f"{v.get('current_price', 0):.2f}",
                "ma75": f"{v.get('ma75', 0):.2f}",
                "avg_volume_20d": f"{v.get('avg_volume_20d', 0):.0f}",
            })


def compute_ma75_and_volume_from_bulk(
    bulk: dict[str, pd.DataFrame],
    ma_window: int = MA_WINDOW,
    vol_days: int = 20,
) -> dict[str, dict[str, Any]]:
    """
    fetch_ohlcv_bulk の戻り値から、銘柄ごとに current_price, ma75, avg_volume_20d を算出。
    """
    result: dict[str, dict[str, Any]] = {}
    for ticker, df in bulk.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        if len(df) < ma_window:
            continue
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < ma_window:
            continue
        current_price = float(close.iloc[-1])
        ma75 = float(close.rolling(window=ma_window, min_periods=ma_window).mean().iloc[-1])
        avg_volume_20d = 0.0
        if "Volume" in df.columns:
            vol = pd.to_numeric(df["Volume"], errors="coerce").dropna().tail(vol_days)
            if len(vol) >= 1:
                avg_volume_20d = float(vol.mean())
        result[ticker] = {
            "current_price": current_price,
            "ma75": ma75,
            "avg_volume_20d": avg_volume_20d,
        }
    return result


def refill_price_ma_volume_bulk(
    tickers: list[str],
    period: str = "6mo",
    chunk_size: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """
    キャッシュにない銘柄について、OHLCV を一括取得（fetch_ohlcv_bulk）し、
    current_price, ma75, avg_volume_20d を算出して返す。1銘柄ずつのリクエストは行わない。
    """
    if not tickers:
        return {}
    chunk_size = chunk_size or BULK_CHUNK_SIZE
    bulk = fetch_ohlcv_bulk(tickers, period=period, chunk_size=chunk_size)
    return compute_ma75_and_volume_from_bulk(bulk, ma_window=MA_WINDOW, vol_days=20)


def get_sector_per(sector: Optional[str]) -> float:
    """セクター名からローカル辞書で平均PERを返す。"""
    if not sector:
        return DEFAULT_SECTOR_PER
    s = _safe_str(sector).strip()
    if s in SECTOR_AVG_PER:
        return SECTOR_AVG_PER[s]
    for key in SECTOR_AVG_PER:
        if key.lower() in s.lower() or s.lower() in key.lower():
            return SECTOR_AVG_PER[key]
    return DEFAULT_SECTOR_PER


def run_batch(
    tickers: Optional[list[str]] = None,
    refill_missing: bool = True,
    top_n: int = TOP_N,
) -> list[dict[str, Any]]:
    """
    銘柄リストから適正株価乖離率を算出し、足切りを通した上で乖離率上位 top_n 件を返す。

    足切り条件:
    - クオリティ: ROE ≥ 8%
    - モメンタム: 現在値 > 75日移動平均
    - 流動性: 直近20日平均出来高 ≥ 10万株
    - 理論株価 > 現在値（割安のみ）
    """
    if tickers is None:
        tickers = [t for t in get_ticker_universe() if t not in get_skip_tickers()]
    if not tickers:
        return []

    # 財務キャッシュ
    fund = load_fundamentals_cache()
    missing_fund = [t for t in tickers if t not in fund]
    if refill_missing and missing_fund and yf:
        new_fund = refill_fundamentals_parallel(missing_fund)
        fund.update(new_fund)
        save_fundamentals_cache(fund)

    # 価格・MA・出来高キャッシュ
    price_cache = load_price_ma_volume_cache()
    missing_price = [t for t in tickers if t not in price_cache]
    if refill_missing and missing_price:
        new_price = refill_price_ma_volume_bulk(missing_price)
        price_cache.update(new_price)
        save_price_ma_volume_cache(price_cache)

    # マージして乖離率算出・フィルター
    rows: list[dict[str, Any]] = []
    for t in tickers:
        f = fund.get(t)
        p = price_cache.get(t)
        if not f or not p:
            continue
        eps = f.get("eps")
        if eps is None or eps <= 0:
            continue
        sector = f.get("sector")
        roe_pct = f.get("roe")
        if roe_pct is None:
            continue
        roe_pct *= 100.0  # 0.08 -> 8
        if roe_pct < MIN_ROE_PCT:
            continue
        current_price = p.get("current_price") or 0
        ma75 = p.get("ma75") or 0
        avg_vol = p.get("avg_volume_20d") or 0
        if current_price <= 0 or ma75 <= 0:
            continue
        if current_price <= ma75:
            continue
        if avg_vol < MIN_AVG_VOLUME_20D:
            continue
        sector_per = get_sector_per(sector)
        theoretical = eps * sector_per
        if theoretical <= current_price:
            continue
        deviation_pct = (theoretical / current_price - 1.0) * 100.0
        rows.append({
            "ticker": t,
            "current_price": current_price,
            "theoretical_price": theoretical,
            "deviation_pct": round(deviation_pct, 1),
            "roe_pct": round(roe_pct, 1),
            "ma75": ma75,
            "avg_volume_20d": avg_vol,
        })

    rows.sort(key=lambda x: x["deviation_pct"], reverse=True)
    return rows[:top_n]


def format_line(row: dict[str, Any]) -> str:
    """出力例形式の1行を返す。"""
    t = row.get("ticker", "")
    cur = row.get("current_price", 0)
    theo = row.get("theoretical_price", 0)
    dev = row.get("deviation_pct", 0)
    roe = row.get("roe_pct", 0)
    return (
        f"【{t}】現在値: ¥{cur:,.0f} | 理論株価: ¥{theo:,.0f} | "
        f"乖離率: +{dev:.1f}% | ROE: {roe:.1f}% | 75MA: 突破済"
    )


def main():
    import sys
    top_n = TOP_N
    if len(sys.argv) > 1:
        try:
            top_n = int(sys.argv[1])
        except ValueError:
            pass
    # 銘柄リストは ticker_universe.get_ticker_universe()（CSV/東証/日経225）。
    # 少数で試す場合は環境変数 JPX_TICKERS_CSV で別CSVを指定するか、run_batch(tickers=[...]) で渡す。
    rows = run_batch(refill_missing=True, top_n=top_n)
    print(f"# 本命の割安株（乖離率上位 {len(rows)} 銘柄・バリュートラップ排除済）")
    print("# キャッシュ: data/fundamentals_cache.csv, data/price_ma_volume_cache.csv")
    print()
    for r in rows:
        print(format_line(r))
    if not rows:
        print("（条件を満たす銘柄はありませんでした）")


if __name__ == "__main__":
    main()
