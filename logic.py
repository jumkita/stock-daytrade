# -*- coding: utf-8 -*-
"""
金融分析ロジック: 理論株価算出 (EV/EBITDA + ROE連動PBR) + 24種買い / 26種売りパターン検知
Ambiguous 対策済み。Model A (Business) / B (Financials) の2軸。app/screener 互換の API を提供。
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone, timedelta
from typing import Any, Optional

# 日本時間（JST = UTC+9）
JST = timezone(timedelta(hours=9))
PROVISIONAL_START = time(15, 10)  # 15:10 JST
PROVISIONAL_END = time(15, 30)    # 15:30 JST（yfinance 約15分遅延を考慮）

import numpy as np
import pandas as pd

import yfinance as yf

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


def fetch_ohlcv(ticker_symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    yfinance で OHLCV を取得。直近数日分を含む period を指定し、最新行を当日バーとして扱う。
    15:10〜15:30 JST ではその最新行を「未確定の1日足」として暫定値に利用する想定。
    データ欠損・取得不可時はエラー出力せず None を返す。
    """
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if df is None or getattr(df, "empty", True):
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
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
        if range0 > 0 and _body_is_tiny(r0) and ls0 > body0 * 2:
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


def filter_signals_by_pro_filters(
    df: pd.DataFrame,
    patterns: list[tuple[int, str, str]],
    side: str = "buy",
    provisional: bool = False,
) -> list[tuple[int, str, str]]:
    """
    出来高スパイク・MA近接の両方を満たすシグナルのみに絞る。
    provisional=True のときは 15:15 暫定用にバッファを許容（出来高1.4倍以上・MA±7%）。
    """
    if df is None or not patterns:
        return []
    vol_multiple = 1.4 if provisional else 1.5
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
    multiple: float = 1.5,
    ma_pct_display: Optional[float] = None,
) -> str:
    """出来高・MA近接の根拠テキストを生成。provisional 時は ma_pct_display=7 で ±7% 表示。"""
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
    """OHLC DataFrame に対して 24種買い + 26種売りパターンを検知。"""
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

    try:
        from signal_scanner import CandlePatterns
        cp = CandlePatterns(df)
        for i in range(len(df)):
            try:
                if cp.is_gyakusanzun(i):
                    result.append((i, "逆三尊", "buy"))
                if cp.is_sanzun(i):
                    result.append((i, "三尊", "sell"))
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
