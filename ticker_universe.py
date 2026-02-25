# -*- coding: utf-8 -*-
"""
対象銘柄（ユニバース）の取得: CSV 優先、未設定時は東証リストを動的取得、失敗時は日経225でフォールバック。
CSV形式: 1列目または code/銘柄コード/コード 列に 4桁コード（例 7203）または 7203.T を記載。
環境変数 JPX_TICKERS_CSV でパス指定可。未指定時は jpx_all_tickers.csv を参照。
"""
from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Dict, List, Optional

# 日経225（フォールバック用）
NIKKEI_225_CODES = [
    "1332", "1605", "1721", "1801", "1802", "1803", "1808", "1812", "1925", "1928",
    "1963", "2002", "2269", "2282", "2413", "2432", "2501", "2502", "2503", "2531",
    "2768", "2801", "2802", "2871", "2914", "3086", "3099", "3101", "3103", "3289",
    "3382", "3401", "3402", "3405", "3407", "3436", "3659", "3861", "3863", "4004",
    "4005", "4021", "4042", "4043", "4061", "4063", "4151", "4183", "4188", "4208",
    "4324", "4385", "4452", "4502", "4503", "4506", "4507", "4519", "4523", "4543",
    "4568", "4578", "4613", "4631", "4661", "4689", "4704", "4751", "4755", "4901",
    "4902", "4911", "5019", "5020", "5101", "5108", "5201", "5202", "5214", "5232",
    "5233", "5301", "5332", "5333", "5401", "5406", "5411", "5631", "5703", "5706",
    "5711", "5713", "5714", "5801", "5802", "5803", "5831", "5901", "6098", "6103",
    "6113", "6146", "6178", "6301", "6302", "6305", "6326", "6361", "6367", "6471",
    "6472", "6473", "6479", "6501", "6503", "6504", "6506", "6526", "6594", "6645",
    "6674", "6701", "6702", "6703", "6723", "6724", "6752", "6753", "6758", "6762",
    "6770", "6841", "6857", "6861", "6902", "6920", "6954", "6971", "6976", "6981",
    "6988", "7004", "7011", "7012", "7013", "7186", "7201", "7202", "7203", "7205",
    "7211", "7261", "7267", "7269", "7270", "7272", "7731", "7733", "7735", "7751",
    "7752", "7762", "7832", "7911", "7912", "7951", "7974", "8001", "8002", "8015",
    "8031", "8035", "8053", "8058", "8233", "8252", "8253", "8267", "8304", "8306",
    "8308", "8309", "8316", "8331", "8354", "8355", "8411", "8473", "8591", "8601",
    "8604", "8630", "8697", "8725", "8750", "8766", "8795", "8801", "8802", "8804",
    "8830", "9001", "9005", "9007", "9008", "9009", "9020", "9021", "9022", "9064",
    "9101", "9104", "9107", "9147", "9201", "9202", "9301", "9432", "9433", "9434",
    "9501", "9502", "9503", "9531", "9532", "9602", "9613", "9735", "9766", "9843",
    "9983", "9984",
]

DEFAULT_CSV_PATH = "jpx_all_tickers.csv"
DEFAULT_TICKER_MAPPING_PATH = "ticker_mapping.json"
JPX_LIST_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"

_name_mapping_cache: Optional[Dict[str, str]] = None


def _normalize_code(raw: str) -> Optional[str]:
    """1行・1セルから銘柄コードを正規化。'7203' or '7203.T' -> '7203.T'。不正は None。"""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.upper() in ("NAN", "NAT", "#N/A", "-", "—"):
        return None
    s = s.replace(" ", "")
    if s.endswith(".T"):
        code = s[:-2]
    else:
        code = s
    if not code.isdigit() or len(code) > 6:
        return None
    return f"{code}.T"


def load_tickers_from_csv(path: str) -> List[str]:
    """
    CSV から銘柄コードを読み込む。列名は code / 銘柄コード / コード / symbol / ticker または先頭列。
    値は '7203' または '7203.T' を想定。
    """
    try:
        import pandas as pd
    except ImportError:
        return []
    if not path or not os.path.isfile(path):
        return []
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=str)
    except Exception:
        try:
            df = pd.read_csv(path, encoding="cp932", dtype=str)
        except Exception:
            return []
    if df is None or df.empty:
        return []
    # 銘柄コード列を探す
    code_col = None
    for c in ("code", "symbol", "ticker", "銘柄コード", "コード", "銘柄コード（数字4桁）"):
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        code_col = df.columns[0]
    codes: List[str] = []
    for v in df[code_col].dropna().astype(str):
        t = _normalize_code(str(v))
        if t and t not in codes:
            codes.append(t)
    return codes


def fetch_jpx_tickers() -> Optional[List[str]]:
    """
    東証上場銘柄一覧を Web から取得する。requests で HTML を取得し、CSV/Excel リンクを探してパース。
    失敗時は None。
    """
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.get(JPX_LIST_PAGE, timeout=15)
        r.raise_for_status()
        html = r.text
    except Exception:
        return None
    # リンク抽出: .csv / .xlsx / .xls
    base = "https://www.jpx.co.jp"
    pattern = re.compile(r'href="([^"]+\.(?:csv|xlsx|xls))"', re.I)
    found = pattern.findall(html)
    url = None
    for rel in found:
        link = rel[0] if isinstance(rel, tuple) else rel
        if link.startswith("http"):
            url = link
        else:
            url = base + link if link.startswith("/") else base + "/" + link
        break
    if not url:
        return None
    try:
        r2 = requests.get(url, timeout=30)
        r2.raise_for_status()
        raw = r2.content
    except Exception:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    codes: List[str] = []
    try:
        if url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(raw), encoding="utf-8", dtype=str)
        else:
            df = pd.read_excel(BytesIO(raw), dtype=str, engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(BytesIO(raw), dtype=str)
        except Exception:
            return None
    if df is None or df.empty:
        return None
    code_col = None
    for c in df.columns:
        cstr = str(c).strip()
        if "コード" in cstr or cstr.lower() in ("code", "symbol", "ticker"):
            code_col = c
            break
    if code_col is None:
        code_col = df.columns[0]
    for v in df[code_col].dropna().astype(str):
        t = _normalize_code(str(v))
        if t and t not in codes:
            codes.append(t)
    return codes if codes else None


def get_ticker_universe(csv_path: Optional[str] = None) -> List[str]:
    """
    対象銘柄リストを返す。
    1) 指定または環境変数 JPX_TICKERS_CSV / デフォルト jpx_all_tickers.csv から読み込み
    2) 無い場合・空の場合は fetch_jpx_tickers() で東証リストを取得
    3) それも失敗した場合は日経225でフォールバック
    """
    path = csv_path or os.environ.get("JPX_TICKERS_CSV", "").strip() or DEFAULT_CSV_PATH
    root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    tickers = load_tickers_from_csv(path)
    if tickers:
        return tickers
    tickers = fetch_jpx_tickers()
    if tickers:
        return tickers
    return [f"{c}.T" for c in NIKKEI_225_CODES]


def load_ticker_name_mapping() -> Dict[str, str]:
    """
    銘柄コード → 銘柄名（日本語）のローカルマッピングを返す。
    ticker_mapping.json を優先し、続けて CSV の name/銘柄名 列があればマージする。API 不要。
    """
    global _name_mapping_cache
    if _name_mapping_cache is not None:
        return _name_mapping_cache
    out: Dict[str, str] = {}
    root = os.path.dirname(os.path.abspath(__file__))
    # 1) ticker_mapping.json
    path_json = os.environ.get("TICKER_MAPPING_JSON", "").strip() or DEFAULT_TICKER_MAPPING_PATH
    if not os.path.isabs(path_json):
        path_json = os.path.join(root, path_json)
    if os.path.isfile(path_json):
        try:
            with open(path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, str) and v.strip():
                        key = k.strip()
                        if key.endswith(".T"):
                            out[key] = v.strip()
                        else:
                            out[f"{key}.T"] = v.strip()
        except Exception:
            pass
    # 2) CSV の name / 銘柄名 列
    path_csv = os.environ.get("JPX_TICKERS_CSV", "").strip() or DEFAULT_CSV_PATH
    if not os.path.isabs(path_csv):
        path_csv = os.path.join(root, path_csv)
    if os.path.isfile(path_csv):
        try:
            import pandas as pd
            df = pd.read_csv(path_csv, encoding="utf-8", dtype=str, nrows=5000)
        except Exception:
            try:
                import pandas as pd
                df = pd.read_csv(path_csv, encoding="cp932", dtype=str, nrows=5000)
            except Exception:
                df = None
        if df is not None and not df.empty:
            try:
                code_col = None
                name_col = None
                for c in df.columns:
                    cstr = str(c).strip()
                    if cstr.lower() in ("code", "symbol", "ticker") or "コード" in cstr:
                        code_col = c
                    if cstr.lower() in ("name", "銘柄名", "名称", "会社名"):
                        name_col = c
                if code_col is None:
                    code_col = df.columns[0]
                if name_col and name_col in df.columns:
                    for _, row in df.iterrows():
                        t = _normalize_code(str(row.get(code_col, "")))
                        name = str(row.get(name_col, "")).strip()
                        if t and name and name not in ("nan", "-", ""):
                            out[t] = name
            except Exception:
                pass
    _name_mapping_cache = out
    return out


def get_ticker_name_local(ticker: str) -> str:
    """
    銘柄コードから表示用名称を返す。ローカル（ticker_mapping.json / CSV）のみ参照し、yfinance は使わない。
    未登録時は【コード】形式で返し、重複表示を防ぐ。
    """
    if not ticker or not isinstance(ticker, str):
        return ticker or ""
    mapping = load_ticker_name_mapping()
    t = ticker.strip()
    if t in mapping:
        return mapping[t]
    code_only = t.replace(".T", "") if t.endswith(".T") else t
    if f"{code_only}.T" in mapping:
        return mapping[f"{code_only}.T"]
    # フォールバック: コードのみを【】で囲み、他で (ticker) と重ねない
    return f"【{t}】"
