"""Streamlit UI 向けの純粋関数（テスト可能）。"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd

DISPLAY_MODES = ("score", "win_rate_70", "all", "sector_good")
SORT_KEYS = ("quadrant_score", "win_rate", "avg_return_pct")

MOBILE_COLUMNS = [
    "銘柄コード",
    "銘柄名",
    "4象限",
    "パターン名",
    "勝率",
    "エントリー想定",
]
DESKTOP_EXTRA_COLUMNS = [
    "サンプル数",
    "平均リターン%",
    "セクター",
    "出来高倍率",
    "ROE%",
    "利確(TP)",
    "損切り(SL)",
    "sector_pts",
    "volume_pts",
    "technical_pts",
    "fundamental_pts",
]


def with_cache_buster(url: str, ts: int | None = None) -> str:
    if not url:
        return url
    try:
        parsed = urlparse(url)
        q = dict(parse_qsl(parsed.query))
        q["_ts"] = str(ts if ts is not None else int(time.time()))
        return urlunparse(parsed._replace(query=urlencode(q)))
    except Exception:
        return url


def resolve_daily_json_url() -> str:
    url = os.environ.get("DAILY_SIGNALS_JSON_URL", "").strip()
    return url


def parse_signals_payload(data: dict[str, Any]) -> dict[str, Any]:
    """JSON ペイロードを session_state 用 dict に正規化。"""
    if data.get("backtest_driven") and data.get("items"):
        items = data.get("items") or []
        n_tickers = data.get("unique_tickers", len({x.get("ticker") for x in items if x.get("ticker")}))
        n_signals = data.get("signal_count", len(items))
        return {
            "items": items,
            "updated": data.get("updated"),
            "items_sell": data.get("items_sell") or [],
            "high_potential": [],
            "watch": [],
            "backtest_format": True,
            "quadrant_stats": data.get("quadrant_stats"),
            "unique_tickers": n_tickers,
            "signal_count": n_signals,
        }
    return {
        "items": data.get("active", data.get("all", [])) or [],
        "updated": data.get("updated"),
        "items_sell": data.get("items_sell") or [],
        "high_potential": data.get("high_potential", []) or [],
        "watch": data.get("watch", []) or [],
        "backtest_format": False,
        "quadrant_stats": None,
        "unique_tickers": None,
        "signal_count": None,
    }


def fetch_signals_json(url: str, timeout: int = 10) -> dict[str, Any]:
    bust = with_cache_buster(url)
    with urllib.request.urlopen(bust, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def filter_signals(
    items: list[dict[str, Any]],
    mode: str,
    *,
    backtest_format: bool = True,
) -> list[dict[str, Any]]:
    if not items:
        return []
    out = list(items)
    if mode == "win_rate_70" and backtest_format:
        out = [x for x in out if (x.get("win_rate") or 0) >= 0.70]
    elif mode == "sector_good":
        out = [x for x in out if str(x.get("sector_label") or "") == "良"]
    return out


def sort_signals(items: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if not items:
        return []

    def _key(x: dict[str, Any]) -> float:
        if sort_key == "quadrant_score":
            v = x.get("quadrant_score")
        elif sort_key == "win_rate":
            v = x.get("win_rate")
        elif sort_key == "avg_return_pct":
            v = x.get("avg_return_pct")
        else:
            v = x.get("quadrant_score")
        try:
            return float(v) if v is not None else -1.0
        except (TypeError, ValueError):
            return -1.0

    return sorted(items, key=_key, reverse=True)


def fmt_price(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        v = float(x)
        return f"¥{v:,.0f}" if v == v else "—"
    except (TypeError, ValueError):
        return "—"


def breakdown_from_item(item: dict[str, Any]) -> dict[str, float]:
    bd = item.get("quadrant_breakdown") or {}
    return {
        "sector": float(bd.get("sector") or 0),
        "volume": float(bd.get("volume") or 0),
        "technical": float(bd.get("technical") or 0),
        "fundamental": float(bd.get("fundamental") or 0),
    }


def items_to_dataframe(items: list[dict[str, Any]], *, mobile: bool = False) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for x in items:
        bd = breakdown_from_item(x)
        wr = x.get("win_rate")
        rows.append(
            {
                "銘柄コード": x.get("ticker"),
                "銘柄名": x.get("name"),
                "4象限": x.get("quadrant_score"),
                "パターン名": x.get("pattern_name") or x.get("quadrant_sign"),
                "勝率": wr,
                "サンプル数": x.get("sample_count"),
                "平均リターン%": x.get("avg_return_pct"),
                "セクター": x.get("sector_label"),
                "出来高倍率": x.get("vol_ratio"),
                "ROE%": x.get("roe_pct"),
                "エントリー想定": x.get("entry"),
                "利確(TP)": x.get("tp"),
                "損切り(SL)": x.get("sl"),
                "sector_pts": bd["sector"],
                "volume_pts": bd["volume"],
                "technical_pts": bd["technical"],
                "fundamental_pts": bd["fundamental"],
                "_raw": x,
            }
        )
    df = pd.DataFrame(rows)
    if mobile:
        cols = [c for c in MOBILE_COLUMNS if c in df.columns]
        return df[cols + (["_raw"] if "_raw" in df.columns else [])]
    return df


def format_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "勝率" in out.columns:
        out["勝率"] = out["勝率"].apply(
            lambda x: f"{float(x) * 100:.0f}%" if x is not None and x == x else "—"
        )
    if "平均リターン%" in out.columns:
        out["平均リターン%"] = out["平均リターン%"].apply(
            lambda x: f"+{float(x):.2f}%" if x is not None and x == x else "—"
        )
    if "4象限" in out.columns:
        out["4象限"] = out["4象限"].apply(
            lambda x: f"{float(x):.0f}点" if x is not None and x == x else "—"
        )
    if "出来高倍率" in out.columns:
        out["出来高倍率"] = out["出来高倍率"].apply(
            lambda x: f"{float(x):.2f}倍" if x is not None and x == x else "—"
        )
    if "ROE%" in out.columns:
        out["ROE%"] = out["ROE%"].apply(
            lambda x: f"{float(x):.1f}%" if x is not None and x == x else "—"
        )
    for col in ("エントリー想定", "利確(TP)", "損切り(SL)"):
        if col in out.columns:
            out[col] = out[col].apply(fmt_price)
    return out


def display_columns(*, mobile: bool) -> list[str]:
    base = list(MOBILE_COLUMNS)
    if not mobile:
        base.extend([c for c in DESKTOP_EXTRA_COLUMNS if c not in base])
    return base


def history_json_url(base_url: str, date_str: str) -> str:
    """daily_buy_signals.json の URL から日付付き URL を生成。"""
    if not base_url:
        return ""
    if "daily_buy_signals.json" in base_url:
        return base_url.replace("daily_buy_signals.json", f"daily_buy_signals_{date_str}.json")
    return base_url.rstrip("/") + f"/daily_buy_signals_{date_str}.json"
