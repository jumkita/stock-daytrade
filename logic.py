# -*- coding: utf-8 -*-
"""
金融分析ロジック: 理論株価算出 (EV/EBITDA + ROE連動PBR) + 24種買い / 26種売りパターン検知
Ambiguous 対策済み。Model A (Business) / B (Financials) の2軸。app/screener 互換の API を提供。
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, time as dt_time, timezone, timedelta
from typing import Any, Optional

# 日本時間（JST = UTC+9）
JST = timezone(timedelta(hours=9))
PROVISIONAL_START = dt_time(15, 10)  # 15:10 JST
PROVISIONAL_END = dt_time(15, 30)    # 15:30 JST（yfinance 約15分遅延を考慮）

import numpy as np
import pandas as pd

import yfinance as yf

# yfinance の「possibly delisted」等のログはノイズが多いため CRITICAL 未満を抑制
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TA-Lib（オプション）
try:
    import talib
    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False


# ========== Safe Helpers (Ambiguous Error対策) ==========

def _to_float_safe(x):
    """値を安全にfloatに変換する。DataFrame/Seriesが来てもスカラーにする"""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return 0.0  # 混入時は default のみ返し、iloc で取り出さない（ambiguous 防止）
        if isinstance(x, (int, float)) and np.isnan(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _safe_str(x):
    """値を安全に文字列にする"""
    if x is None:
        return ""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return ""
    return str(x)


def _safe_get(d, key, default=None):
    """辞書から安全に値を取得する。DataFrame/Series は返さない。"""
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    if val is None:
        return default
    if isinstance(val, (pd.DataFrame, pd.Series)):
        return default
    return val


def fuzzy_get_value(df, keywords):
    """
    データフレームからキーワードに一致する行を探し、その最新値を返す（安全版）
    """
    if df is None:
        return 0.0
    if getattr(df, "empty", True):
        return 0.0

    try:
        pattern = "|".join([k.lower() for k in keywords])
        if not hasattr(df, "index"):
            return 0.0

        mask = df.index.astype(str).str.lower().str.contains(pattern, case=False, na=False)

        if not mask.any():
            return 0.0

        matched_rows = df[mask]
        if matched_rows.empty:
            return 0.0

        return _to_float_safe(matched_rows.iloc[0, 0])

    except Exception as e:
        logger.error("Error in fuzzy_get_value: %s", e)
        return 0.0


# ========== Core SOTP Logic ==========

def select_valuation_logic(data):
    """
    業種・セクターに基づいて評価モデルを選択する（2軸）。
    Model B: 金融・不動産（負債を事業原資とするセクター）
    Model A: それ以外すべて（一般事業会社）
    """
    industry = _safe_str(data.get("industry", "")).lower()
    sector = _safe_str(data.get("sector", "")).lower()
    search_text = f"{industry} {sector}"

    if any(x in search_text for x in (
        "banks", "insurance", "real estate",
        "銀行", "保険", "不動産", "financial",
    )):
        return "B", "ROE-linked PBR"

    return "A", "EV/EBITDA"


def get_sotp_data(ticker):
    """
    データ取得＆補完ロジック
    """
    try:
        info = ticker.info

        try:
            fin_df = ticker.financials
            bs_df = ticker.balance_sheet
        except Exception:
            fin_df = None
            bs_df = None

        def get_val(keys_info, df_source, keys_df):
            for k in keys_info:
                val = _safe_get(info, k)
                if val is not None:
                    f_val = _to_float_safe(val)
                    if f_val != 0:
                        return f_val
            if df_source is not None and not getattr(df_source, "empty", True):
                return fuzzy_get_value(df_source, keys_df)
            return 0.0

        # Net Income
        net_income = get_val(
            ["netIncomeToCommon", "netIncome"],
            fin_df,
            ["net income", "net income common", "profit", "net profit"],
        )

        # Total Equity
        total_equity = get_val(
            ["totalStockholderEquity"],
            bs_df,
            ["total stockholder equity", "stockholder equity", "total equity", "equity"],
        )

        # Shares & Price
        shares = _to_float_safe(_safe_get(info, "sharesOutstanding", 1))
        current_price = _to_float_safe(_safe_get(info, "currentPrice"))
        if current_price == 0:
            current_price = _to_float_safe(_safe_get(info, "regularMarketPrice"))

        # 補完: Equityが取れなくてBookValueがある場合
        if total_equity == 0:
            bv = _to_float_safe(_safe_get(info, "bookValue"))
            if bv != 0 and shares != 0:
                total_equity = bv * shares

        # ROE
        roe = _to_float_safe(_safe_get(info, "returnOnEquity"))
        if roe == 0 and total_equity > 0:
            roe = net_income / total_equity

        # EBITDA
        ebitda = get_val(
            ["ebitda"],
            fin_df,
            ["ebitda", "normalized ebitda", "operating income"],
        )

        # Cash & Debt
        total_cash = get_val(
            ["totalCash"],
            bs_df,
            ["cash", "cash and cash equivalents"],
        )
        total_debt = get_val(
            ["totalDebt"],
            bs_df,
            ["total debt", "long term debt", "debt"],
        )

        return {
            "industry": _safe_str(_safe_get(info, "industry")),
            "sector": _safe_str(_safe_get(info, "sector")),
            "current_price": current_price,
            "shares": shares,
            "net_income": net_income,
            "total_equity": total_equity,
            "roe": roe,
            "ebitda": ebitda,
            "total_cash": total_cash,
            "total_debt": total_debt,
            "currency": _safe_str(_safe_get(info, "currency", "JPY")),
        }

    except Exception as e:
        logger.error("Error in get_sotp_data: %s", e)
        return None


def calculate_sotp(ticker_symbol, ebitda_multiple: Optional[float] = None):
    """
    理論株価算出のメイン関数。
    Model A: EV/EBITDA（一般事業会社）
    Model B: ROE 連動 PBR（金融・不動産）
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = get_sotp_data(ticker)

        if data is None or data["current_price"] == 0:
            return None

        model_id, model_name = select_valuation_logic(data)

        shares = data["shares"]
        if shares == 0:
            shares = 1

        # --- Insolvency Check: 債務超過は強制シャットアウト ---
        if data["total_equity"] <= 0:
            return {
                "ticker": ticker_symbol,
                "current_price": data["current_price"],
                "theoretical_price": 0.0,
                "upside": -100.0,
                "model_name": "Avoid",
                "details": "Avoid: Negative Equity (債務超過)",
                "raw_data": data,
            }

        target_price = 0.0
        details = ""

        if model_id == "B":
            # --- Model B: ROE-linked PBR（東証基準 + 赤字ペナルティ） ---
            bps = data["total_equity"] / shares
            roe = data["roe"]
            if roe >= 0.08:
                target_pbr = 1.2
            elif roe >= 0.05:
                target_pbr = 1.0
            elif roe >= 0.0:
                target_pbr = 0.6
            else:
                target_pbr = 0.4
            target_price = bps * target_pbr
            details = f"Model B: ROE-linked PBR (Target: {target_pbr}倍)"

        else:
            # --- Model A: Acquirer's Multiple (EV/EBITDA) ---
            if ebitda_multiple is not None:
                base_multiple = float(ebitda_multiple)
            else:
                base_multiple = 8.0
                industry_lower = _safe_str(data.get("industry", "")).lower()
                sector_lower = _safe_str(data.get("sector", "")).lower()
                if "automotive" in industry_lower or "auto" in industry_lower:
                    base_multiple = 6.0
                elif "technology" in sector_lower:
                    base_multiple = 12.0

            ebitda = data["ebitda"]
            if ebitda > 0:
                ev = ebitda * base_multiple
                equity_value = ev + data["total_cash"] - data["total_debt"]
                target_price = equity_value / shares
                details = f"Model A: EV/EBITDA ({base_multiple}倍)"
            else:
                target_price = (data["total_equity"] * 0.5) / shares
                details = "Model A: 解散価値 (Equity×0.5, EBITDA≤0)"

        return {
            "ticker": ticker_symbol,
            "current_price": data["current_price"],
            "theoretical_price": target_price,
            "upside": (target_price - data["current_price"]) / data["current_price"] * 100,
            "model_name": model_name,
            "details": details,
            "raw_data": data,
        }

    except Exception as e:
        logger.error("Calculation failed for %s: %s", ticker_symbol, e)
        return None


# ========== app/screener 互換 API ==========

def sotp_full(
    ticker_symbol: str,
    ebitda_multiple: float = 10.0,
    financial_bps_multiple: float = 1.2,
) -> dict[str, Any]:
    """銘柄シンボルから SOTP 一括実行。calculate_sotp の結果を app 用形式に変換。"""
    res = calculate_sotp(ticker_symbol, ebitda_multiple=ebitda_multiple)
    if res is None:
        return {
            "theoretical_price": None,
            "current_price": None,
            "deviation_pct": None,
            "message": "取得失敗",
            "valuation_logic": None,
            "model_type": None,
            "raw": None,
        }
    model_id, _ = select_valuation_logic(res["raw_data"])
    return {
        "theoretical_price": res["theoretical_price"],
        "current_price": res["current_price"],
        "deviation_pct": res["upside"],
        "message": res["details"],
        "valuation_logic": res["details"],
        "model_type": model_id,
        "raw": res["raw_data"],
    }


def flatten_ohlcv_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    yfinance が返す DataFrame の columns が MultiIndex の場合に必ずフラット化する。
    全てのデータ取得箇所で例外なく呼ぶこと。
    """
    if df is None or getattr(df, "empty", True):
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_ohlcv(ticker_symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    yfinance で OHLCV を取得。直近数日分を含む period を指定し、最新行を当日バーとして扱う。
    15:10〜15:30 JST ではその最新行を「未確定の1日足」として暫定値に利用する想定。
    データ欠損・取得不可時はエラー出力せず None を返す。
    MultiIndex フラット化: 取得直後に flatten_ohlcv_columns を漏れなく1回だけ実行する。
    """
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if df is None or getattr(df, "empty", True):
            return None
        df = flatten_ohlcv_columns(df)
        if df is None:
            return None
        for c in ("Open", "High", "Low", "Close"):
            if c not in df.columns:
                return None
        cols = ["Open", "High", "Low", "Close"]
        if "Volume" in df.columns:
            cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].dropna(how="all")
        return df.reset_index()
    except Exception:
        return None


# バルク取得のチャンクサイズ。小さいほど接続エラー時の影響が少ない（GitHub Actions 等で安定）
# 環境変数 BULK_CHUNK_SIZE で上書き可（例: 80）
try:
    _chunk = int(os.environ.get("BULK_CHUNK_SIZE", "50"))
    BULK_CHUNK_SIZE = max(20, min(200, _chunk))
except (TypeError, ValueError):
    BULK_CHUNK_SIZE = 50
PREFILTER_PERIOD = "2mo"  # 20営業日以上の出来高・株価を得るため


def _single_df_from_bulk(raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """バルク取得した MultiIndex DataFrame から1銘柄分の DataFrame を切り出し。"""
    if raw is None or getattr(raw, "empty", True):
        return None
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                return None
            sub = raw[ticker].copy()
        else:
            sub = raw.copy()
        if sub is None or sub.empty:
            return None
        sub = flatten_ohlcv_columns(sub)
        for c in ("Open", "High", "Low", "Close"):
            if c not in sub.columns:
                return None
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in sub.columns]
        sub = sub[cols].dropna(how="all").reset_index()
        return sub if len(sub) >= 5 else None
    except Exception:
        return None


def fetch_ohlcv_bulk(
    tickers: list[str],
    period: str = "3y",
    interval: str = "1d",
    chunk_size: int = BULK_CHUNK_SIZE,
) -> dict[str, pd.DataFrame]:
    """
    複数銘柄をチャンク単位で一括ダウンロード。for ループで1銘柄ずつ呼ばない。
    Returns: ticker -> DataFrame (OHLCV, reset_index 済み)。取得失敗はスキップ。
    """
    result: dict[str, pd.DataFrame] = {}
    chunk_size = max(1, min(chunk_size, 200))
    # GitHub Actions 等ではチャンク間に短い待機でレート制限・接続リセットを緩和
    chunk_delay = 1.0 if os.environ.get("GITHUB_ACTIONS") else 0.0
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        if not chunk:
            continue
        if chunk_delay > 0 and start > 0:
            time.sleep(chunk_delay)
        try:
            raw = yf.download(
                " ".join(chunk),
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        if len(chunk) == 1 and not isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = pd.MultiIndex.from_product([[chunk[0]], raw.columns])
        for t in chunk:
            df = _single_df_from_bulk(raw, t)
            if df is not None:
                result[t] = df
    return result


# 分足でザラ場の最新価格を取得するチャンクサイズ（一括取得のみ・直列禁止）
INTRADAY_CHUNK_SIZE = 80


def fetch_intraday_latest_close_bulk(
    tickers: list[str],
    period: str = "1d",
    interval: str = "5m",
    chunk_size: int = INTRADAY_CHUNK_SIZE,
) -> dict[str, float]:
    """
    当日の分足をバルク取得し、各銘柄の「最新のClose」を返す。1銘柄ずつの直列取得は行わない。
    15:00時点のザラ場価格を日足の仮終値として使うために利用する。
    Returns: ticker -> latest Close (float)。取得失敗はスキップ。
    """
    result: dict[str, float] = {}
    chunk_size = max(1, min(chunk_size, 150))
    chunk_delay = 0.8 if os.environ.get("GITHUB_ACTIONS") else 0.0
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        if not chunk:
            continue
        if chunk_delay > 0 and start > 0:
            time.sleep(chunk_delay)
        try:
            raw = yf.download(
                " ".join(chunk),
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        if len(chunk) == 1 and not isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = pd.MultiIndex.from_product([[chunk[0]], raw.columns])
        for t in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[t].copy()
                else:
                    sub = raw.copy()
                if sub is None or sub.empty:
                    continue
                sub = flatten_ohlcv_columns(sub)
                if "Close" not in sub.columns:
                    continue
                close_series = sub["Close"].dropna()
                if len(close_series) == 0:
                    continue
                last_close = float(close_series.iloc[-1])
                if last_close == last_close and last_close > 0:
                    result[t] = last_close
            except Exception:
                continue
    return result


def merge_intraday_into_last_bar(
    bulk_dfs: dict[str, pd.DataFrame],
    ticker_to_latest_close: dict[str, float],
) -> None:
    """
    日足の直近1行の Close を、分足で取得した最新価格で上書きする（in-place）。
    15:00時点のザラ場価格を当日の仮終値としてバックテストに反映する。
    """
    for ticker, df in bulk_dfs.items():
        if df is None or len(df) < 1 or "Close" not in df.columns:
            continue
        latest = ticker_to_latest_close.get(ticker)
        if latest is None or (isinstance(latest, float) and (latest != latest or latest <= 0)):
            continue
        try:
            df.iloc[-1, df.columns.get_loc("Close")] = float(latest)
        except Exception:
            continue


def prefilter_tickers_bulk(
    tickers: list[str],
    min_volume_20d: float = 100_000,
    min_price: float = 200.0,
    period: str = PREFILTER_PERIOD,
    chunk_size: int = BULK_CHUNK_SIZE,
) -> list[str]:
    """
    全銘柄の重いバックテストの前に、軽量な一括取得で「直近20日平均出来高10万株未満」「株価200円未満」を除外する。
    バルクで period=2mo を取得し、足切りした銘柄リストを返す。
    """
    if not tickers:
        return []
    vol_min = min_volume_20d
    price_min = min_price
    passed: list[str] = []
    bulk = fetch_ohlcv_bulk(tickers, period=period, chunk_size=chunk_size)
    for t, df in bulk.items():
        if df is None or len(df) < 20:
            continue
        try:
            if "Volume" not in df.columns:
                continue
            vol = df["Volume"].iloc[-20:]
            avg_vol = vol.mean()
            if pd.isna(avg_vol) or float(avg_vol) < vol_min:
                continue
            close = float(df["Close"].iloc[-1])
            if pd.isna(close) or close < price_min:
                continue
            passed.append(t)
        except Exception:
            continue
    return passed


# プレフィルター（API制限・ノイズ排除）のしきい値
PREFILTER_MIN_VOLUME = 100_000   # 直近平均出来高 100,000株未満は除外
PREFILTER_MIN_PRICE = 100.0      # 現在値 100円未満（ペニーストック）は除外


def prefilter_ticker(
    ticker_symbol: str,
    period: str = "5d",
    min_volume: float = PREFILTER_MIN_VOLUME,
    min_price: float = PREFILTER_MIN_PRICE,
) -> bool:
    """
    軽量データ（短期足）で流動性・株価の足切りを行う。True のときのみ本スキャン対象とする。
    - 流動性: 直近の平均出来高が min_volume 未満なら False
    - 株価: 現在値が min_price 未満なら False
    """
    df = fetch_ohlcv(ticker_symbol, period=period, interval="1d")
    if df is None or len(df) < 1:
        return False
    try:
        if "Volume" in df.columns:
            vol = df["Volume"]
            avg_vol = vol.mean()
            if pd.isna(avg_vol) or float(avg_vol) < min_volume:
                return False
        if "Close" in df.columns:
            close = float(df["Close"].iloc[-1])
            if pd.isna(close) or close < min_price:
                return False
        return True
    except Exception:
        return False


def get_downtrend_mask(df: pd.DataFrame, window: int = 25) -> pd.Series:
    """終値が SMA(window) より下の行を True とするマスク。"""
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return pd.Series(dtype=bool)
    close = df["Close"]
    sma = close.rolling(window=window, min_periods=window).mean()
    return (close < sma) & sma.notna()


def calc_stop_loss_line(df: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    """直近安値ベースの損切りライン。"""
    if df is None or getattr(df, "empty", True) or len(df) < 1 or "Low" not in df.columns:
        return None
    try:
        return float(df["Low"].tail(lookback).min())
    except Exception:
        return None


def _is_financial_sector(sector_str: str) -> bool:
    """金融セクターかどうか。"""
    if not sector_str:
        return False
    s = _safe_str(sector_str).lower()
    return (
        "financial" in s or "bank" in s or "insurance" in s
        or "証券" in s or "銀行" in s or "金融" in s
    )


def _sector_base_multiple_fallback(sector_str: str) -> tuple[float, str]:
    """セクター別の基準倍率フォールバック。"""
    if not sector_str:
        return 6.0, "その他"
    s = _safe_str(sector_str).lower()
    if "technology" in s or "tech" in s or "healthcare" in s or "情報" in s or "医薬" in s:
        return 10.0, "Tech/Healthcare"
    if "consumer" in s or "cyclical" in s or "industrials" in s or "工業" in s or "消費" in s:
        return 6.0, "Consumer/Industrials"
    if "basic materials" in s or "materials" in s or "energy" in s or "utilities" in s or "素材" in s or "エネルギー" in s or "公益" in s:
        return 4.0, "Materials/Energy/Utilities"
    return 6.0, "その他"


def calculate_base_multiple(industry_str: str, sector_str: str) -> tuple[float, str]:
    """industry ベースで Base Multiple を決定。"""
    text = (_safe_str(industry_str) or "").strip().lower()
    if not text:
        return _sector_base_multiple_fallback(sector_str)
    for kw in ("software", "information", "internet", "ソフトウェア", "情報", "インターネット"):
        if kw in text:
            return 16.0, "Tier1(Software/Info/Internet)"
    for kw in ("semiconductor", "medical", "biotech", "electronics", "半導体", "医薬", "医療", "電子"):
        if kw in text:
            return 12.0, "Tier2(Semi/Medical/Electronics)"
    for kw in ("entertainment", "beverages", "food", "retail", "consumer", "エンタメ", "飲料", "食品", "小売", "消費"):
        if kw in text:
            return 8.0, "Tier3(Entertainment/Food/Retail)"
    for kw in ("machinery", "construction", "railroads", "chemicals", "機械", "建設", "鉄道", "化学"):
        if kw in text:
            return 6.0, "Tier4(Machinery/Chemicals)"
    for kw in ("steel", "metals", "oil", "gas", "marine", "shipping", "trading", "auto", "automotive", "鉄鋼", "金属", "石油", "海運", "商社", "自動車"):
        if kw in text:
            return 4.0, "Tier5(Auto/Steel/Oil/Shipping/Trading)"
    return _sector_base_multiple_fallback(sector_str)


def suggest_ebitda_multiple(
    sector_str: str,
    industry_str: str = "",
    revenue_growth: float = 0.0,
    margin: float = 0.0,
) -> tuple[Optional[float], float, str]:
    """提案倍率を算出。金融の場合は (None, 0, '金融')。"""
    if _is_financial_sector(sector_str):
        return None, 0.0, "金融"
    base, label = calculate_base_multiple(industry_str, sector_str)
    add = (revenue_growth or 0.0) * 0.2 + (margin or 0.0) * 0.1
    suggested = max(1.0, min(25.0, base + add))
    return round(suggested, 1), base, label


def get_sotp_suggested_multiple(ticker_symbol: str) -> dict[str, Any]:
    """銘柄の業種に応じた提案倍率。Model B では multiplier_disabled=True。"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        raw = get_sotp_data(ticker)
    except Exception:
        raw = None
    if raw is None:
        return {
            "suggested_multiple": 8.0,
            "base": 8.0,
            "sector_label": "—",
            "is_financial": False,
            "multiplier_disabled": False,
            "valuation_logic": None,
        }
    model_id, logic_name = select_valuation_logic(raw)
    multiplier_disabled = model_id == "B"
    sector = _safe_str(raw.get("sector") or "")
    industry = _safe_str(raw.get("industry") or "")
    suggested, base, label = suggest_ebitda_multiple(sector, industry, 0.0, 0.0)
    return {
        "suggested_multiple": None if multiplier_disabled else (suggested if suggested is not None else 8.0),
        "base": base,
        "sector_label": label,
        "is_financial": _is_financial_sector(sector),
        "multiplier_disabled": multiplier_disabled,
        "valuation_logic": logic_name,
    }


# ========== テクニカル: ローソク足ユーティリティ ==========

def _body(s: pd.Series) -> float:
    return abs(s["Close"] - s["Open"])


def _upper_shadow(s: pd.Series) -> float:
    return s["High"] - max(s["Open"], s["Close"])


def _lower_shadow(s: pd.Series) -> float:
    return min(s["Open"], s["Close"]) - s["Low"]


def _range_hl(s: pd.Series) -> float:
    return s["High"] - s["Low"]


def _bull(s: pd.Series) -> bool:
    return s["Close"] > s["Open"]


def _bear(s: pd.Series) -> bool:
    return s["Open"] > s["Close"]


def _body_is_tiny(s: pd.Series, min_r: float = 1e-10) -> bool:
    r = _range_hl(s)
    return r <= min_r or _body(s) < r * 0.1


# ========== 24種買い / 26種売りパターン ==========

def _talib_func(name: str):
    if not _TALIB_AVAILABLE:
        return None
    return getattr(talib, name, None)


def _detect_talib_pattern(o, h, l, c, func, name_ja: str, side: str) -> list[tuple[int, str, str]]:
    out = []
    if not _TALIB_AVAILABLE or func is None:
        return out
    try:
        res = func(o, h, l, c)
        if res is None:
            return out
        for i in range(len(res)):
            if side == "buy" and res[i] > 0:
                out.append((i, name_ja, "buy"))
            elif side == "sell" and res[i] < 0:
                out.append((i, name_ja, "sell"))
    except Exception:
        pass
    return out


def _custom_buy_patterns(df: pd.DataFrame) -> list[tuple[int, str, str]]:
    out = []
    n = len(df)
    for i in range(1, n):
        try:
            r0, r1 = df.iloc[i], df.iloc[i - 1]
        except Exception:
            continue
        body0, body1 = _body(r0), _body(r1)
        range0, range1 = _range_hl(r0), _range_hl(r1)
        ls0, ls1 = _lower_shadow(r0), _lower_shadow(r1)
        us0 = _upper_shadow(r0)
        if range0 > 0 and range1 > 0 and ls0 >= body0 * 2 and ls1 >= body1 * 2:
            out.append((i, "二本たくり線", "buy"))
        if _bear(r1) and _bull(r0) and r0["Close"] > r1["Close"]:
            out.append((i, "陰線後の陽線", "buy"))
        # ピンバー: 実体が極小かつ下ヒゲが実体の2倍以上。上ヒゲが下ヒゲより短いことを要求しノイズ削減。
        if range0 > 0 and _body_is_tiny(r0) and ls0 > body0 * 2 and us0 < ls0:
            out.append((i, "ピンバー", "buy"))
        if _bull(r0) and range0 > 0 and ls0 >= range0 * 0.6:
            out.append((i, "スパイクロー", "buy"))
        if i >= 5 and _bull(r0):
            prev_low = min(df.iloc[k]["Low"] for k in range(i - 5, i))
            if r0["Low"] <= prev_low and r0["Close"] > df.iloc[i - 1]["Close"]:
                out.append((i, "リバーサルロー", "buy"))
        if range1 > 0 and r0["High"] < r1["High"] and r0["Low"] > r1["Low"]:
            out.append((i, "インサイドバー", "buy"))
        if body1 > 0 and _bull(r0) and r0["Open"] < r1["Close"] and r0["Close"] > r1["Open"]:
            out.append((i, "包み線", "buy"))
    return out


def _custom_sell_patterns(df: pd.DataFrame) -> list[tuple[int, str, str]]:
    out = []
    n = len(df)
    for i in range(1, n):
        try:
            r0, r1 = df.iloc[i], df.iloc[i - 1]
        except Exception:
            continue
        range0, range1 = _range_hl(r0), _range_hl(r1)
        us0, ls0 = _upper_shadow(r0), _lower_shadow(r0)
        body0, body1 = _body(r0), _body(r1)
        if range0 > 0 and _body_is_tiny(r0) and us0 > body0 * 2:
            out.append((i, "たくり", "sell"))
        if range0 > 0 and us0 >= range0 * 0.6:
            out.append((i, "スパイクハイ", "sell"))
        if i >= 5 and _bear(r0):
            prev_high = max(df.iloc[k]["High"] for k in range(i - 5, i))
            if r0["High"] >= prev_high and r0["Close"] < df.iloc[i - 1]["Close"]:
                out.append((i, "リバーサルハイ", "sell"))
        if range1 > 0 and r0["High"] > r1["High"] and r0["Low"] < r1["Low"]:
            out.append((i, "アウトサイドバー", "sell"))
        if body1 > 0 and _bear(r0) and r0["Open"] > r1["Close"] and r0["Close"] < r1["Open"]:
            out.append((i, "陰のつつみ", "sell"))
        if i >= 3:
            gaps = 0
            for k in range(i, i - 3, -1):
                if k < 1:
                    break
                curr, prev = df.iloc[k], df.iloc[k - 1]
                if prev["High"] < curr["Low"]:
                    gaps += 1
                else:
                    break
            if gaps >= 3:
                out.append((i, "三空踏み上げ", "sell"))
    return out


# ========== 3営業日ホールド バックテスト（統計ベース） ==========
HOLD_DAYS = 3
BACKTEST_MIN_SAMPLES = 3
BACKTEST_MIN_WIN_RATE = 0.60
BACKTEST_MIN_AVG_RETURN_PCT = 1.5
BACKTEST_LIQUIDITY_VOL_20D = 100_000
BACKTEST_MIN_PRICE = 200.0


def run_backtest_3day(df: pd.DataFrame, ticker: str = "") -> list[tuple[str, float]]:
    """
    日足で買いパターン点灯日の「翌日寄り付き」でエントリー、「3営業日後の大引け」で決済した場合の
    リターン(%)を1トレードずつ算出する。戻り値: [(pattern_name, return_pct), ...]
    （バックテスト一括実行時は run_backtest_3day_vectorized を使用すること）
    """
    if df is None or getattr(df, "empty", True) or len(df) < 5:
        return []
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return []
    out: list[tuple[str, float]] = []
    try:
        patterns = detect_all_patterns(df)
        buy_only = [(i, name, side) for i, name, side in patterns if side == "buy"]
        for bar_i, pattern_name, _ in buy_only:
            next_i = bar_i + 1
            exit_i = bar_i + 4
            if exit_i >= len(df):
                continue
            entry_price = float(df["Open"].iloc[next_i])
            exit_price = float(df["Close"].iloc[exit_i])
            if entry_price <= 0:
                continue
            ret_pct = (exit_price - entry_price) / entry_price * 100.0
            out.append((pattern_name, ret_pct))
    except Exception:
        pass
    return out


def detect_buy_patterns_vectorized(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    買いパターン点灯バーをベクトル化演算で検出。iterrows/for 行ループは使わない。
    Returns: pattern_name -> array of bar indices (int)
    """
    if df is None or getattr(df, "empty", True) or len(df) < 2:
        return {}
    for c in ("Open", "High", "Low", "Close"):
        if c not in df.columns:
            return {}
    out: dict[str, list[int]] = {}
    o = np.asarray(df["Open"], dtype=np.float64)
    h = np.asarray(df["High"], dtype=np.float64)
    l_ = np.asarray(df["Low"], dtype=np.float64)
    c = np.asarray(df["Close"], dtype=np.float64)
    n = len(df)

    # TA-Lib: 配列一括でシグナル取得し np.where でインデックス化
    for name_ja, func in BUY_PATTERNS_TALIB:
        if not _TALIB_AVAILABLE or func is None:
            continue
        try:
            res = func(o, h, l_, c)
            if res is None:
                continue
            idx = np.where(res > 0)[0]
            if len(idx) > 0:
                out.setdefault(name_ja, []).extend(idx.tolist())
        except Exception:
            continue

    # カスタム買いパターン: shift/rolling 等のベクトル演算のみ
    body = (df["Close"] - df["Open"]).abs()
    range_hl = df["High"] - df["Low"]
    low_min = df[["Open", "Close"]].min(axis=1)
    high_max = df[["Open", "Close"]].max(axis=1)
    lower = low_min - df["Low"]
    upper = df["High"] - high_max
    bull = df["Close"] > df["Open"]
    bear = df["Open"] > df["Close"]
    body1 = body.shift(1)
    range1 = range_hl.shift(1)
    lower1 = lower.shift(1)
    close1 = df["Close"].shift(1)
    open1 = df["Open"].shift(1)
    high1 = df["High"].shift(1)
    low1 = df["Low"].shift(1)

    # 二本たくり線
    cond = (range_hl > 0) & (range1 > 0) & (lower >= body * 2) & (lower1 >= body1 * 2)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        idx = idx[idx >= 1]
        if len(idx):
            out.setdefault("二本たくり線", []).extend(idx.tolist())
    # 陰線後の陽線
    cond = bear.shift(1) & bull & (df["Close"] > close1)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        idx = idx[idx >= 1]
        if len(idx):
            out.setdefault("陰線後の陽線", []).extend(idx.tolist())
    # ピンバー
    cond = (range_hl > 0) & (body < range_hl * 0.1) & (lower > body * 2) & (upper < lower)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        out.setdefault("ピンバー", []).extend(idx.tolist())
    # スパイクロー
    cond = bull & (range_hl > 0) & (lower >= range_hl * 0.6)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        out.setdefault("スパイクロー", []).extend(idx.tolist())
    # リバーサルロー
    prev_low_5 = df["Low"].rolling(5, min_periods=5).min().shift(1)
    cond = bull & (df["Low"] <= prev_low_5) & (df["Close"] > close1)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        idx = idx[idx >= 5]
        if len(idx):
            out.setdefault("リバーサルロー", []).extend(idx.tolist())
    # インサイドバー
    cond = (range1 > 0) & (df["High"] < high1) & (df["Low"] > low1)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        idx = idx[idx >= 1]
        if len(idx):
            out.setdefault("インサイドバー", []).extend(idx.tolist())
    # 包み線
    cond = (body1 > 0) & bull & (df["Open"] < close1) & (df["Close"] > open1)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        idx = idx[idx >= 1]
        if len(idx):
            out.setdefault("包み線", []).extend(idx.tolist())
    # 三空叩き込み
    gap = df["Low"].shift(1) > df["High"]
    cond = gap & gap.shift(1) & gap.shift(2)
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx):
        idx = idx[idx >= 3]
        if len(idx):
            out.setdefault("三空叩き込み", []).extend(idx.tolist())

    return {k: np.unique(np.array(v, dtype=np.int64)) for k, v in out.items() if v}


def run_backtest_3day_vectorized(df: pd.DataFrame, ticker: str = "") -> list[tuple[str, float]]:
    """
    3営業日ホールドのリターンをベクトル化で算出。iterrows/行ループは使わない。
    翌日寄りエントリー・3営業日後終値決済。pattern_name ごとに shift で entry/exit を一括取得。
    """
    if df is None or getattr(df, "empty", True) or len(df) < 5:
        return []
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return []
    open_arr = np.asarray(df["Open"], dtype=np.float64)
    close_arr = np.asarray(df["Close"], dtype=np.float64)
    n = len(df)
    out: list[tuple[str, float]] = []
    try:
        by_pattern = detect_buy_patterns_vectorized(df)
        for pattern_name, bar_inds in by_pattern.items():
            bar_inds = np.asarray(bar_inds, dtype=np.int64)
            next_i = bar_inds + 1
            exit_i = bar_inds + 4
            valid = (exit_i < n) & (next_i < n)
            bar_inds = bar_inds[valid]
            next_i = next_i[valid]
            exit_i = exit_i[valid]
            if len(bar_inds) == 0:
                continue
            entry_prices = open_arr[next_i]
            exit_prices = close_arr[exit_i]
            valid2 = entry_prices > 0
            if not np.any(valid2):
                continue
            entry_prices = entry_prices[valid2]
            exit_prices = exit_prices[valid2]
            ret_pct = (exit_prices - entry_prices) / entry_prices * 100.0
            for r in ret_pct:
                out.append((pattern_name, float(r)))
    except Exception:
        pass
    return out


def aggregate_backtest_stats(
    trades: list[tuple[str, str, float]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    (ticker, pattern_name, return_pct) のリストを集計し、
    (ticker, pattern_name) -> {win_rate, avg_return_pct, sample_count} を返す。
    """
    from collections import defaultdict
    key_to_returns: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ticker, pattern_name, ret_pct in trades:
        key_to_returns[(ticker, pattern_name)].append(ret_pct)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for (ticker, pattern_name), returns in key_to_returns.items():
        n = len(returns)
        if n == 0:
            continue
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / n
        avg_ret = sum(returns) / n
        result[(ticker, pattern_name)] = {
            "win_rate": round(win_rate, 4),
            "avg_return_pct": round(avg_ret, 4),
            "sample_count": n,
        }
    return result


def backtest_liquidity_ok(
    df: pd.DataFrame,
    min_volume_20d: float = BACKTEST_LIQUIDITY_VOL_20D,
    min_price: float = BACKTEST_MIN_PRICE,
) -> bool:
    """流動性・株価の足切り。直近20日平均出来高 >= min_volume_20d、現在値 >= min_price。"""
    if df is None or getattr(df, "empty", True) or len(df) < 20:
        return False
    try:
        if "Volume" not in df.columns:
            return False
        vol = df["Volume"].iloc[-20:]
        avg_vol = vol.mean()
        if pd.isna(avg_vol) or float(avg_vol) < min_volume_20d:
            return False
        close = float(df["Close"].iloc[-1])
        if pd.isna(close) or close < min_price:
            return False
        return True
    except Exception:
        return False


def run_full_backtest_universe(
    tickers: list[str],
    period: str = "3y",
    liquidity_filter: bool = True,
    chunk_size: int = BULK_CHUNK_SIZE,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    全銘柄で3営業日バックテストを実行。(ticker, pattern_name) ごとに win_rate, avg_return_pct, sample_count を返す。
    - 事前足切り: 軽量バルクで直近2ヶ月取得し、出来高10万株未満・200円未満を除外してから本取得。
    - 本取得: 通過銘柄のみ 3年分をチャンク一括ダウンロード（for で1銘柄ずつ取得しない）。
    - 計算: ベクトル化パターン検知・リターン算出（iterrows/行ループなし）。
    """
    if not tickers:
        return {}
    passed_tickers = prefilter_tickers_bulk(tickers, chunk_size=chunk_size) if liquidity_filter else list(tickers)
    if not passed_tickers:
        return {}
    bulk = fetch_ohlcv_bulk(passed_tickers, period=period, chunk_size=chunk_size)
    all_trades: list[tuple[str, str, float]] = []
    for ticker, df in bulk.items():
        if df is None or len(df) < 5:
            continue
        for pattern_name, ret_pct in run_backtest_3day_vectorized(df, ticker):
            all_trades.append((ticker, pattern_name, ret_pct))
    return aggregate_backtest_stats(all_trades)


def passes_backtest_filters(stats: dict[str, Any]) -> bool:
    """サンプル3回以上・勝率60%以上・平均リターン+1.5%以上。"""
    if not stats:
        return False
    n = stats.get("sample_count", 0)
    wr = stats.get("win_rate", 0.0)
    avg = stats.get("avg_return_pct", 0.0)
    return n >= BACKTEST_MIN_SAMPLES and wr >= BACKTEST_MIN_WIN_RATE and avg >= BACKTEST_MIN_AVG_RETURN_PCT


def tp_sl_from_avg_return(current_price: float, avg_return_pct: float) -> dict[str, float]:
    """
    リスクリワードを考慮: TP = 現在値 + 平均リターン、SL = 現在値 - (平均リターン÷2)。
    平均リターンは%なので current * (avg_return_pct/100) で価格差に変換。
    """
    if current_price <= 0:
        return {"entry": current_price, "tp": current_price, "sl": current_price}
    delta = current_price * (avg_return_pct / 100.0)
    tp = current_price + delta
    sl = current_price - (delta / 2.0)
    return {"entry": round(current_price, 2), "tp": round(tp, 2), "sl": round(max(0, sl), 2)}


# ========== プロフィルタ（出来高スパイク・MA近接）と TP/SL 算出 ==========


def is_provisional_market_session() -> bool:
    """
    現在時刻が 15:10〜15:30 JST かどうか。
    この帯では直近の未確定1日足を「当日暫定」として扱い、判定に5%程度のバッファを許容する。
    """
    now_jst = datetime.now(JST).time()
    return PROVISIONAL_START <= now_jst <= PROVISIONAL_END


def _volume_spike_ok(df: pd.DataFrame, bar_index: int, avg_days: int = 5, multiple: float = 1.5) -> bool:
    """直近 avg_days 営業日の平均出来高に対し、当日出来高が multiple 倍以上であることを確認。"""
    if df is None or bar_index < avg_days or "Volume" not in df.columns:
        return False
    try:
        vol = df["Volume"]
        if pd.isna(vol.iloc[bar_index]) or vol.iloc[bar_index] <= 0:
            return False
        start = bar_index - avg_days
        avg_vol = vol.iloc[start:bar_index].mean()
        if pd.isna(avg_vol) or avg_vol <= 0:
            return False
        return float(vol.iloc[bar_index]) >= multiple * float(avg_vol)
    except Exception:
        return False


def _ma_proximity_ok(df: pd.DataFrame, bar_index: int, pct: float = 0.02, windows: tuple[int, int] = (25, 75)) -> bool:
    """現在価格が指定 MA の ±pct 以内にあるか（反転の節目で出たサインのみ採用）。"""
    if df is None or "Close" not in df.columns:
        return False
    try:
        close = float(df["Close"].iloc[bar_index])
        if close <= 0 or pd.isna(close):
            return False
        for w in windows:
            if bar_index + 1 < w:
                continue
            ma = df["Close"].iloc[bar_index - w + 1 : bar_index + 1].mean()
            if pd.isna(ma) or ma <= 0:
                continue
            dev = abs(close - ma) / ma
            if dev <= pct:
                return True
        return False
    except Exception:
        return False


def get_volume_ratio(df: pd.DataFrame, bar_index: int, avg_days: int = 5) -> Optional[float]:
    """直近 avg_days 営業日の平均出来高に対する当日出来高の倍率。取得不可時は None。"""
    if df is None or bar_index < avg_days or "Volume" not in df.columns:
        return None
    try:
        vol = df["Volume"]
        if pd.isna(vol.iloc[bar_index]) or vol.iloc[bar_index] <= 0:
            return None
        start = bar_index - avg_days
        avg_vol = vol.iloc[start:bar_index].mean()
        if pd.isna(avg_vol) or avg_vol <= 0:
            return None
        return float(vol.iloc[bar_index]) / float(avg_vol)
    except Exception:
        return None


def get_ma_deviation(df: pd.DataFrame, bar_index: int, windows: tuple[int, int] = (25, 75)) -> Optional[float]:
    """現在価格の MA からの最小絶対乖離（比率）。例: 0.02 = ±2%。取得不可時は None。"""
    if df is None or "Close" not in df.columns:
        return None
    try:
        close = float(df["Close"].iloc[bar_index])
        if close <= 0 or pd.isna(close):
            return None
        min_dev: Optional[float] = None
        for w in windows:
            if bar_index + 1 < w:
                continue
            ma = df["Close"].iloc[bar_index - w + 1 : bar_index + 1].mean()
            if pd.isna(ma) or ma <= 0:
                continue
            dev = abs(close - ma) / ma
            if min_dev is None or dev < min_dev:
                min_dev = float(dev)
        return min_dev
    except Exception:
        return None


def get_ma_25_deviation_signed(df: pd.DataFrame, bar_index: int) -> Optional[float]:
    """25日線からの符号付き乖離率。(close - MA25)/MA25。例: -0.05 = -5%。取得不可時は None。"""
    if df is None or "Close" not in df.columns or bar_index + 1 < 25:
        return None
    try:
        close = float(df["Close"].iloc[bar_index])
        ma25 = df["Close"].iloc[bar_index - 24 : bar_index + 1].mean()
        if pd.isna(close) or pd.isna(ma25) or ma25 <= 0:
            return None
        return float((close - ma25) / ma25)
    except Exception:
        return None


def get_rsi(df: pd.DataFrame, bar_index: int, period: int = 14) -> Optional[float]:
    """RSI(period)。0〜100。取得不可時は None。"""
    if df is None or "Close" not in df.columns or bar_index < period:
        return None
    try:
        if _TALIB_AVAILABLE:
            c = np.asarray(df["Close"], dtype=np.float64)
            rsi = talib.RSI(c, timeperiod=period)
            v = rsi[bar_index]
            if np.isnan(v):
                return None
            return float(v)
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        start = bar_index - period + 1
        avg_gain = gain.iloc[start : bar_index + 1].mean()
        avg_loss = loss.iloc[start : bar_index + 1].mean()
        if pd.isna(avg_gain):
            avg_gain = 0.0
        if pd.isna(avg_loss) or avg_loss == 0:
            return 100.0
        rs = float(avg_gain) / float(avg_loss)
        return round(100.0 - (100.0 / (1.0 + rs)), 2)
    except Exception:
        return None


def is_perfect_order(df: pd.DataFrame, bar_index: int) -> bool:
    """5日線 > 25日線 > 75日線（パーフェクトオーダー）か。"""
    if df is None or "Close" not in df.columns or bar_index + 1 < 75:
        return False
    try:
        c = df["Close"]
        ma5 = c.iloc[bar_index - 4 : bar_index + 1].mean()
        ma25 = c.iloc[bar_index - 24 : bar_index + 1].mean()
        ma75 = c.iloc[bar_index - 74 : bar_index + 1].mean()
        if pd.isna(ma5) or pd.isna(ma25) or pd.isna(ma75):
            return False
        return float(ma5) > float(ma25) > float(ma75)
    except Exception:
        return False


def is_20d_high_update(df: pd.DataFrame, bar_index: int) -> bool:
    """直近20日で高値更新しているか（当日高値 >= 20日高値）。"""
    if df is None or "High" not in df.columns or bar_index < 20:
        return False
    try:
        high_today = float(df["High"].iloc[bar_index])
        high_20d = float(df["High"].iloc[bar_index - 19 : bar_index + 1].max())
        return not pd.isna(high_today) and not pd.isna(high_20d) and high_today >= high_20d
    except Exception:
        return False


def compute_conviction_score(df: pd.DataFrame, bar_index: int) -> float:
    """
    確信度スコア（Conviction Score）を算出する。
    スコア = (当日出来高/5日平均出来高) + (1 / (1 + |MAからの乖離率|))
    出来高が爆発しており、かつMAに極めて近い銘柄ほど高スコア。
    """
    vol_ratio = get_volume_ratio(df, bar_index)
    ma_dev = get_ma_deviation(df, bar_index)
    term_vol = float(vol_ratio) if vol_ratio is not None else 0.0
    term_ma = 1.0 / (1.0 + (float(ma_dev) if ma_dev is not None else 1.0))
    return round(term_vol + term_ma, 4)


def classify_signal_status(
    df: pd.DataFrame,
    bar_index: int,
    provisional: bool = False,
) -> Optional[tuple[str, str]]:
    """
    本命 / 出来高待ち / 押し目待ち のいずれかに分類する。
    15:15 実行時は大引け出来高未反映のため、本命の出来高条件は 1.2 倍に統一。
    Returns:
        ("active", "—") 本命
        ("volume_watch", "出来高不足") 出来高待ち
        ("price_watch", "MA乖離（押し目待ち）") 押し目待ち
        該当なしなら None
    """
    vol_ratio = get_volume_ratio(df, bar_index)
    ma_dev = get_ma_deviation(df, bar_index)
    if vol_ratio is None or ma_dev is None:
        return None
    # 本命の出来高条件: 1.5倍→1.2倍（15:15 時間補正）
    vol_active = 1.2
    ma_active = 0.07 if provisional else 0.02
    if vol_ratio >= vol_active and ma_dev <= ma_active:
        return ("active", "—")
    if ma_dev <= 0.02 and 1.0 <= vol_ratio < vol_active:
        return ("volume_watch", "出来高不足")
    if vol_ratio >= vol_active and 0.02 < ma_dev <= 0.05:
        return ("price_watch", "MA乖離（押し目待ち）")
    return None


# ---------- ハイブリッド判定（Type-A トレンド追随 / Type-B リバウンド） ----------
VOL_BONUS_THRESH = 1.2  # 出来高1.2倍以上で加点
MA25_REBOUND_THRESH = -0.05  # 25日線乖離 -5% 以下で Type-B 条件
RSI_REBOUND_THRESH = 40  # RSI(14) 40以下で Type-B 条件


def _type_a_score(
    df: pd.DataFrame, bar_index: int, provisional: bool = False
) -> tuple[bool, int, int]:
    """
    Type-A（トレンド追随型）。
    Returns: (条件成立, 条件1, 加点数). 条件=5日>25日>75日。加点=20日高値更新(1)+出来高1.2倍(1)。最大3。
    """
    cond = is_perfect_order(df, bar_index)
    bonus = 0
    if is_20d_high_update(df, bar_index):
        bonus += 1
    vol_ratio = get_volume_ratio(df, bar_index)
    if vol_ratio is not None and vol_ratio >= VOL_BONUS_THRESH:
        bonus += 1
    return (cond, 1 if cond else 0, bonus)


def _type_b_score(
    df: pd.DataFrame, bar_index: int, has_buy_pattern: bool, provisional: bool = False
) -> tuple[bool, int, int]:
    """
    Type-B（リバウンド型）。
    条件: 25日線乖離≤-5% または RSI(14)≤40。加点=買いパターン点灯(1)+出来高1.2倍(1)。最大3。
    """
    ma25_dev = get_ma_25_deviation_signed(df, bar_index)
    rsi = get_rsi(df, bar_index, 14)
    cond = False
    if ma25_dev is not None and ma25_dev <= MA25_REBOUND_THRESH:
        cond = True
    if rsi is not None and rsi <= RSI_REBOUND_THRESH:
        cond = True
    bonus = 0
    if has_buy_pattern:
        bonus += 1
    vol_ratio = get_volume_ratio(df, bar_index)
    if vol_ratio is not None and vol_ratio >= VOL_BONUS_THRESH:
        bonus += 1
    return (cond, 1 if cond else 0, bonus)


def hybrid_classify_signal(
    df: pd.DataFrame,
    bar_index: int,
    has_buy_pattern: bool,
    provisional: bool = False,
) -> Optional[tuple[str, str]]:
    """
    ハイブリッド判定: Type-A / Type-B のスコアで 本命(active) / 注目(high_potential) を返す。
    Returns:
        ("active", "—") 全条件合致（確信度高）
        ("high_potential", "…") 条件の8割以上充足（確信度中）
        該当なしなら None
    """
    a_cond, a_cond_n, a_bonus = _type_a_score(df, bar_index, provisional)
    b_cond, b_cond_n, b_bonus = _type_b_score(df, bar_index, has_buy_pattern, provisional)
    a_total = a_cond_n + a_bonus  # max 3
    b_total = b_cond_n + b_bonus  # max 3
    # 本命: いずれかで 3/3
    if a_total >= 3:
        return ("active", "Type-A トレンド追随（パーフェクトオーダー+高値更新+出来高）")
    if b_total >= 3:
        return ("active", "Type-B リバウンド（押し安+買いパターン+出来高）")
    # 注目: 8割以上 = 2以上/3 または 2.4 以上 → 2以上で注目
    if a_total >= 2:
        return ("high_potential", "Type-A 一部充足（トレンド形は良いが条件未達）")
    if b_total >= 2:
        return ("high_potential", "Type-B 一部充足（リバウンド形は良いが条件未達）")
    return None


def _safe_num(val: Optional[float], default: float = 0.0) -> float:
    """NaN/None を default にし、ソートエラーを防ぐ。"""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:
            return default
        return f
    except (TypeError, ValueError):
        return default


# 監視銘柄（Watchlist）用しきい値
WATCH_VOL_RATIO_A = 1.0   # 条件A: 1.0倍＝前日と同等で資金流入とみなす（1.2→1.0に緩和）
WATCH_VOL_RATIO_B = 2.0   # 条件B: 出来高2.0倍以上
WATCH_MA_PCT_A = 0.05     # 条件A: MA乖離5%以内
WATCH_MA_PCT_B = 0.03     # 条件B: MA乖離3%以内
WATCH_PATTERN_MULTIPLIER = 3  # パターン点灯時の基礎スコア倍率


def watchlist_score(
    df: pd.DataFrame, bar_index: int, pattern_found: bool = False
) -> float:
    """
    監視リスト用スコア。買いパターン点灯を絶対優遇する。
    - 基礎スコア = 出来高比。パターン点灯時は3倍。
    - MA乖離減点: 乖離するほど下げるが、パターン点灯時は減点幅を緩和。
    NaN 時は 0 を返しソート崩壊を防ぐ。
    """
    vol_ratio = get_volume_ratio(df, bar_index)
    ma_dev = get_ma_deviation(df, bar_index)
    base = _safe_num(vol_ratio, 0.0)
    if pattern_found:
        base *= WATCH_PATTERN_MULTIPLIER
    ma_gap = _safe_num(ma_dev, 0.0)
    # MA乖離減点: パターン点灯時は係数0.4、未点灯時は1.2で緩和差をつける
    penalty_coef = 0.4 if pattern_found else 1.2
    penalty = min(0.9, ma_gap * penalty_coef)
    score = base * max(0.1, 1.0 - penalty)
    return round(score, 4)


def watchlist_eligible(
    pattern_found: bool,
    vol_ratio: Optional[float],
    ma_dev: Optional[float],
) -> tuple[bool, Optional[str]]:
    """
    監視銘柄の足切り。意味のあるニアミスのみ。
    Returns: (合格するか, "A"|"B"|None)
    - 条件A（形完成・エネルギー待ち）: パターン点灯 かつ 出来高1.0倍以上 かつ MA乖離5%以内
    - 条件B（エネルギー爆発・形未完成）: パターン未点灯 かつ 出来高2.0倍以上 かつ MA乖離3%以内
    """
    v = _safe_num(vol_ratio, 0.0)
    ma = _safe_num(ma_dev, 1.0)
    if pattern_found and v >= WATCH_VOL_RATIO_A and ma <= WATCH_MA_PCT_A:
        return (True, "A")
    if not pattern_found and v >= WATCH_VOL_RATIO_B and ma <= WATCH_MA_PCT_B:
        return (True, "B")
    return (False, None)


def build_watchlist_reason_short(
    pattern_found: bool,
    pattern_names: str,
    vol_ratio: Optional[float],
    ma_dev: Optional[float],
    condition_ab: Optional[str],
) -> str:
    """
    監視銘柄の「何が足りないか」を条件に沿って出し分け。緩和ロジックに合わせた文面（1.5倍超えの固定表現は廃止）。
    - 条件A: パターン名点灯。出来高増加待ち
    - 条件B: 強烈な資金流入（出来高〇倍）あり。買いパターンの形成待ち
    - MA乖離3%〜5%時は「 | 押し目形成中」を付与
    """
    ma = _safe_num(ma_dev, 0.0)
    ma_in_3_5 = 0.03 < ma <= 0.05
    suffix = " | 押し目形成中" if ma_in_3_5 else ""

    if condition_ab == "A":
        name = pattern_names.strip() or "買いパターン"
        return f"{name}点灯。出来高増加待ち{suffix}".strip()
    if condition_ab == "B":
        v = _safe_num(vol_ratio, 0.0)
        vol_str = f"{v:.1f}" if v == v else "—"
        return f"強烈な資金流入（出来高{vol_str}倍）あり。買いパターンの形成待ち{suffix}".strip()
    return "条件未達（要観察）"


def filter_signals_by_pro_filters(
    df: pd.DataFrame,
    patterns: list[tuple[int, str, str]],
    side: str = "buy",
    provisional: bool = False,
) -> list[tuple[int, str, str]]:
    """
    出来高スパイク・MA近接の両方を満たすシグナルのみに絞る（本命のみ）。
    本命の出来高条件は 1.2 倍に統一（15:15 大引け未反映を考慮）。
    """
    if df is None or not patterns:
        return []
    vol_multiple = 1.2  # 本命: 1.5→1.2（時間補正）
    ma_pct = 0.07 if provisional else 0.02
    out = []
    for i, name, s in patterns:
        if s != side:
            continue
        if _volume_spike_ok(df, i, multiple=vol_multiple) and _ma_proximity_ok(df, i, pct=ma_pct):
            out.append((i, name, s))
    return out


def compute_tp_sl(
    df: pd.DataFrame,
    bar_index: Optional[int] = None,
    atr_period: int = 14,
    rr_ratio: float = 2.0,
) -> dict[str, Any]:
    """
    利確(TP)・損切り(SL)を算出。
    SL = max(直近5日安値, 現在値 - 2*ATR) のうち高い方（保守的）。
    TP = 現在値 + (現在値 - SL) * rr_ratio（リスク:リワード = 1:rr_ratio）。
    """
    if df is None or getattr(df, "empty", True) or len(df) < 2:
        return {"entry": None, "sl": None, "tp": None, "risk": None, "reward": None}
    n = len(df)
    i = bar_index if bar_index is not None else n - 1
    i = max(0, min(i, n - 1))
    try:
        close = float(df["Close"].iloc[i])
        low_5d = float(df["Low"].iloc[max(0, i - 4) : i + 1].min())
    except Exception:
        return {"entry": None, "sl": None, "tp": None, "risk": None, "reward": None}

    atr_val: Optional[float] = None
    if _TALIB_AVAILABLE and i >= atr_period:
        try:
            h = np.asarray(df["High"], dtype=np.float64)
            l_ = np.asarray(df["Low"], dtype=np.float64)
            c = np.asarray(df["Close"], dtype=np.float64)
            atr = talib.ATR(h, l_, c, timeperiod=atr_period)
            if not np.isnan(atr[i]) and atr[i] > 0:
                atr_val = float(atr[i])
        except Exception:
            pass
    if atr_val is None:
        atr_val = 0.0

    sl_candidate_atr = close - 2.0 * atr_val if atr_val > 0 else None
    sl = max(low_5d, sl_candidate_atr) if sl_candidate_atr is not None else low_5d
    if sl >= close:
        sl = low_5d
    risk = close - sl
    reward = risk * rr_ratio
    tp = close + reward

    return {
        "entry": round(close, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "risk": round(risk, 2),
        "reward": round(reward, 2),
    }


def build_signal_rationale(
    df: pd.DataFrame,
    bar_index: int,
    avg_days: int = 5,
    multiple: float = 1.2,
    ma_pct_display: Optional[float] = None,
) -> str:
    """出来高・MA近接の根拠テキストを生成。本命は multiple=1.2（15:15 時間補正）。provisional 時は ma_pct_display=7 で ±7% 表示。"""
    parts = []
    if df is None or "Volume" not in df.columns:
        return "—"
    try:
        vol = df["Volume"]
        if bar_index >= avg_days and not pd.isna(vol.iloc[bar_index]):
            start = bar_index - avg_days
            avg_vol = vol.iloc[start:bar_index].mean()
            if not pd.isna(avg_vol) and avg_vol > 0:
                ratio = float(vol.iloc[bar_index]) / float(avg_vol)
                parts.append(f"出来高{ratio:.1f}倍")
    except Exception:
        pass
    ma_thresh = ma_pct_display if ma_pct_display is not None else 2.0
    try:
        close = float(df["Close"].iloc[bar_index])
        for w in (25, 75):
            if bar_index + 1 < w:
                continue
            ma = df["Close"].iloc[bar_index - w + 1 : bar_index + 1].mean()
            if pd.isna(ma) or ma <= 0:
                continue
            dev_pct = (close - ma) / ma * 100
            if abs(dev_pct) <= ma_thresh:
                parts.append(f"MA{w}±{int(ma_thresh)}%")
                break
    except Exception:
        pass
    return " ".join(parts) if parts else "—"


BUY_PATTERNS_TALIB = [
    ("赤三兵", _talib_func("CDL3WHITESOLDIERS")),
    ("明けの明星", _talib_func("CDLMORNINGSTAR")),
    ("上げ三法", _talib_func("CDLRISEFALL3METHODS")),
    ("抱きの本立ち", _talib_func("CDLBELTHOLD")),
    ("陽のつつみ線", _talib_func("CDLENGULFING")),
    ("はらみ線", _talib_func("CDLHARAMI")),
    ("切り込み線", _talib_func("CDLPIERCING")),
    ("陽のたすき線", _talib_func("CDLTASUKIGAP")),
    ("ピンバー(ハンマー)", _talib_func("CDLHAMMER")),
    ("逆ハンマー", _talib_func("CDLINVERTEDHAMMER")),
    ("スラストアップ", _talib_func("CDLTHRUSTING")),
    ("包み線", _talib_func("CDLENGULFING")),
]

SELL_PATTERNS_TALIB = [
    ("同時三羽", _talib_func("CDL3BLACKCROWS")),
    ("流れ星", _talib_func("CDLSHOOTINGSTAR")),
    ("首吊り線", _talib_func("CDLHANGINGMAN")),
    ("下げ三法", _talib_func("CDLRISEFALL3METHODS")),
    ("黒三兵", _talib_func("CDL3BLACKCROWS")),
    ("宵の明星", _talib_func("CDLEVENINGSTAR")),
    ("かぶせ", _talib_func("CDLDARKCLOUDCOVER")),
    ("陰のつつみ", _talib_func("CDLENGULFING")),
    ("赤三兵の先詰", _talib_func("CDLADVANCEBLOCK")),
    ("星", _talib_func("CDLDOJISTAR")),
]


def detect_all_patterns(df: pd.DataFrame) -> list[tuple[int, str, str]]:
    """
    OHLC DataFrame に対して買い・売りパターンを検知。
    複数日複雑パターン（逆三尊・三尊）は誤検知が多いため除外。ピンバー・包み線・はらみ線・明けの明星など単一〜3日足の数学的に検証可能なパターンを優先。
    """
    if df is None or getattr(df, "empty", True) or len(df) < 2:
        return []
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return []

    o = np.asarray(df["Open"], dtype=np.float64).copy()
    h = np.asarray(df["High"], dtype=np.float64).copy()
    l = np.asarray(df["Low"], dtype=np.float64).copy()
    c = np.asarray(df["Close"], dtype=np.float64).copy()
    for arr in (o, h, l, c):
        if np.any(np.isnan(arr)):
            s = pd.Series(arr)
            s = s.ffill().bfill()
            arr[:] = s.values

    result: list[tuple[int, str, str]] = []
    for name_ja, func in BUY_PATTERNS_TALIB:
        result.extend(_detect_talib_pattern(o, h, l, c, func, name_ja, "buy"))
    for name_ja, func in SELL_PATTERNS_TALIB:
        result.extend(_detect_talib_pattern(o, h, l, c, func, name_ja, "sell"))
    result.extend(_custom_buy_patterns(df))
    result.extend(_custom_sell_patterns(df))

    # 逆三尊・三尊は複数日にまたがる複雑パターンのため誤検知が多く、一時的に除外。単一〜3日足のノイズの少ないパターンを優先。
    try:
        from signal_scanner import CandlePatterns
        cp = CandlePatterns(df)
        for i in range(len(df)):
            try:
                # if cp.is_gyakusanzun(i):
                #     result.append((i, "逆三尊", "buy"))
                # if cp.is_sanzun(i):
                #     result.append((i, "三尊", "sell"))
                if cp.is_akenomyojo(i):
                    result.append((i, "明けの明星", "buy"))
                if cp.is_yoinomyojo(i):
                    result.append((i, "宵の明星", "sell"))
                if cp.is_aka_sanpei(i):
                    result.append((i, "赤三兵", "buy"))
                if cp.is_kuro_sanpei(i):
                    result.append((i, "黒三兵", "sell"))
                if cp.is_nihon_takuri(i):
                    result.append((i, "二本たくり線", "buy"))
                if cp.is_aka_sanpei_sakizume(i):
                    result.append((i, "赤三兵の先詰", "sell"))
            except Exception:
                continue
    except ImportError:
        pass

    for i in range(3, len(df)):
        gaps = 0
        for k in range(i, i - 3, -1):
            if k < 1:
                break
            curr, prev = df.iloc[k], df.iloc[k - 1]
            if prev["Low"] > curr["High"]:
                gaps += 1
            else:
                break
        if gaps >= 3:
            result.append((i, "三空叩き込み", "buy"))

    return result


# ========== Gemini API（gemini_brain の再エクスポート） ==========

from gemini_brain import (
    get_gemini_api_key,
    qualitative_audit,
    qualitative_audit_batch,
    generate_one_shot,
)


def gemini_echo_ticker(
    ticker_name: str,
    api_key: Optional[str] = None,
    streamlit_secrets: Optional[Any] = None,
) -> str:
    """
    銘柄名を Gemini に投げ、一言で返す疎通確認用。
    logic 経由で app から呼ばれる。実体は gemini_brain.generate_one_shot。
    """
    prompt = f"銘柄名「{ticker_name}」について、投資の文脈で一言だけ短く答えてください。"
    return generate_one_shot(prompt, api_key=api_key, secrets=streamlit_secrets)


def qualitative_audit_with_gemini(
    ticker_symbol: str,
    upside: float,
    api_key: Optional[str] = None,
    streamlit_secrets: Optional[Any] = None,
) -> dict[str, Any]:
    """
    銘柄に対して Gemini で定性監査。実体は gemini_brain.qualitative_audit。
    screener / app との互換のため streamlit_secrets を secrets に渡す。
    """
    return qualitative_audit(
        ticker_symbol,
        upside,
        api_key=api_key,
        secrets=streamlit_secrets,
    )


def qualitative_audit_batch_with_gemini(
    items: list[dict[str, Any]],
    api_key: Optional[str] = None,
    streamlit_secrets: Optional[Any] = None,
    batch_size: int = 5,
    progress_callback: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """
    複数銘柄を一括で定性監査。実体は gemini_brain.qualitative_audit_batch。
    items: [{"ticker": str, "upside": float}, ...]
    progress_callback(done_count, total_count, message) で監査進捗を通知。
    """
    return qualitative_audit_batch(
        items,
        api_key=api_key,
        secrets=streamlit_secrets,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
