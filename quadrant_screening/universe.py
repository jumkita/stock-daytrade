"""東証プライム母集団の読み込み。"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from quadrant_screening.ticker_utils import normalize_ticker


def _normalize_jpx_code(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if re.match(r"^\d+\.0$", s):
        s = str(int(float(s)))
    s = s.upper()
    if re.match(r"^\d{4}$", s):
        return s
    if re.match(r"^\d+[A-Z]$", s):
        return s
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 4:
        return digits[:4].zfill(4)
    return None


def load_prime_universe(csv_path: Path) -> pd.DataFrame:
    """
    jpx_all_tickers.csv からプライム上場銘柄を返す。
    columns: ticker, name, sector_code_17
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp932")

    market_col = next((c for c in df.columns if "市場" in str(c)), None)
    code_col = next((c for c in df.columns if "コード" in str(c) and "業種" not in str(c)), None)
    name_col = next((c for c in df.columns if "銘柄名" in str(c)), None)
    sec_col = next((c for c in df.columns if "17業種コード" in str(c)), None)

    if not code_col or not name_col:
        return pd.DataFrame(columns=["ticker", "name", "sector_code_17"])

    out = pd.DataFrame()
    out["code_raw"] = df[code_col].apply(_normalize_jpx_code)
    out["name"] = df[name_col].astype(str).str.strip()
    if market_col:
        out["market"] = df[market_col].astype(str)
        out = out[out["market"].str.contains("プライム", na=False)]
    if sec_col:
        out["sector_code_17"] = pd.to_numeric(df[sec_col], errors="coerce").astype("Int64")
    else:
        out["sector_code_17"] = pd.NA

    out["ticker"] = out["code_raw"].apply(lambda x: normalize_ticker(x) if x else None)
    out = out.dropna(subset=["ticker", "name"]).drop_duplicates(subset=["ticker"])
    return out[["ticker", "name", "sector_code_17"]].reset_index(drop=True)
