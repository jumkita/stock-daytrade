"""ティッカー正規化（重複 .T バグ対策）。"""
from __future__ import annotations

import re


_DUP_SUFFIX = re.compile(r"^(\d{4}[A-Z]?)(?:\.T\1\.T)+$", re.IGNORECASE)
_CODE_ONLY = re.compile(r"^(\d{4}[A-Z]?)$", re.IGNORECASE)


def normalize_ticker(raw: str) -> str | None:
    """
    yfinance 用ティッカーに正規化。
    例: 7063.T7063.T → 7063.T、7203 → 7203.T
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().upper().replace(" ", "")
    if not s:
        return None

    m = _DUP_SUFFIX.match(s)
    if m:
        return f"{m.group(1).upper()}.T"

    if s.endswith(".T"):
        base = s[:-2]
        if _CODE_ONLY.match(base):
            return f"{base.upper()}.T"
        return None

    m2 = _CODE_ONLY.match(s)
    if m2:
        return f"{m2.group(1).upper()}.T"

    digits = re.sub(r"\D", "", s)
    if len(digits) >= 4:
        return f"{digits[:4].zfill(4)}.T"
    return None


def display_code(ticker: str) -> str:
    """表示用: 7063.T → 7063"""
    t = normalize_ticker(ticker) or ticker
    return t.replace(".T", "") if t.endswith(".T") else t
