#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提案4象限フィルタが過去シグナルの avg_return_pct を改善するかを検証する。

データ: stock-daytrade の daily_buy_signals_YYYY-MM-DD.json（GitHub main）
リターン指標: JSON の avg_return_pct（3営業日バックテスト由来）

検証は「シグナル日時点のOHLCV」で一次・需給・75MAを再現。ROE/セクターは
yfinance 欠損が多いため、取得できたサブセットで追加検証する。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

GITHUB_API = "https://api.github.com/repos/jumkita/stock-daytrade/contents/"
GITHUB_RAW = "https://raw.githubusercontent.com/jumkita/stock-daytrade/main/"

# 提案仕様
PROPOSED_MIN_PRICE = 500.0
PROPOSED_MIN_VOL_20D = 300_000
PROPOSED_VOL_SPIKE = 1.5
MA_PERIOD = 75

# 現行 stock-daytrade 一次フィルタ（logic.py）
CURRENT_MIN_PRICE = 200.0
CURRENT_MIN_VOL_20D = 100_000


@dataclass
class Signal:
    ticker: str
    signal_date: str
    entry: float
    avg_return_pct: float
    pattern_name: str
    win_rate: float | None = None
    sample_count: int | None = None
    # 検証で付与
    vol_20d: float | None = None
    vol_ratio: float | None = None
    above_75ma: bool | None = None
    roe_pct: float | None = None
    sector_momentum_ok: bool | None = None
    realized_return_pct: float | None = None  # シグナル日引け→3営業日後引け


def _fetch_json_files(local_dir: Path | None) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if local_dir and local_dir.is_dir():
        for p in sorted(local_dir.glob("daily_buy_signals_*.json")):
            if p.name == "daily_buy_signals.json":
                continue
            try:
                out.append((p.name, json.loads(p.read_text(encoding="utf-8"))))
            except json.JSONDecodeError:
                pass
        return out

    req = urllib.request.urlopen(GITHUB_API)
    listing = json.loads(req.read().decode())
    names = sorted(
        x["name"]
        for x in listing
        if x["name"].startswith("daily_buy_signals_")
        and x["name"].endswith(".json")
        and x["name"] != "daily_buy_signals.json"
    )
    for name in names:
        try:
            raw = urllib.request.urlopen(GITHUB_RAW + name).read().decode("utf-8")
            out.append((name, json.loads(raw)))
        except Exception:
            pass
        time.sleep(0.05)
    return out


def _date_from_name(name: str) -> str | None:
    m = re.search(r"daily_buy_signals_(\d{4}-\d{2}-\d{2})\.json", name)
    return m.group(1) if m else None


def load_signals(local_dir: Path | None) -> list[Signal]:
    signals: list[Signal] = []
    for fname, data in _fetch_json_files(local_dir):
        d = _date_from_name(fname)
        if not d:
            continue
        for it in data.get("items") or []:
            try:
                entry = float(it["entry"])
                ar = float(it["avg_return_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            if entry <= 0:
                continue
            wr = it.get("win_rate")
            sc = it.get("sample_count")
            signals.append(
                Signal(
                    ticker=str(it.get("ticker") or ""),
                    signal_date=d,
                    entry=entry,
                    avg_return_pct=ar,
                    pattern_name=str(it.get("pattern_name") or ""),
                    win_rate=float(wr) if wr is not None else None,
                    sample_count=int(sc) if sc is not None else None,
                )
            )
    return signals


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Date" not in df.columns:
        df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=False).dt.tz_localize(None)
    df["_date"] = df[date_col].dt.date
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace("¥", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
    return df


def _fetch_ticker_ohlcv(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        return _normalize_ohlcv(df)
    except Exception:
        return pd.DataFrame()


def _realized_return_3d(df: pd.DataFrame, signal_date: str, holding_days: int = 3) -> float | None:
    """verify_signal_returns と同様: シグナル日引け→N営業日後引け。"""
    if df.empty or "Close" not in df.columns:
        return None
    signal_d = datetime.strptime(signal_date, "%Y-%m-%d").date()
    after = df[df["_date"] >= signal_d].sort_values("_date").reset_index(drop=True)
    if after.empty or len(after) <= holding_days:
        return None
    entry_close = float(after.iloc[0]["Close"])
    exit_close = float(after.iloc[holding_days]["Close"])
    if entry_close <= 0:
        return None
    return (exit_close - entry_close) / entry_close * 100.0


def enrich_ohlcv(sig: Signal, cache: dict[str, pd.DataFrame]) -> None:
    ticker = sig.ticker
    if ticker not in cache:
        cache[ticker] = _fetch_ticker_ohlcv(ticker)
    df = cache[ticker]
    if df.empty:
        return

    sig.realized_return_pct = _realized_return_3d(df, sig.signal_date)

    signal_d = datetime.strptime(sig.signal_date, "%Y-%m-%d").date()
    hist = df[df["_date"] <= signal_d].sort_values("_date")
    if len(hist) < MA_PERIOD:
        return
    row = hist.iloc[-1]
    vol20 = hist["Volume"].iloc[-20:].mean()
    vol5 = hist["Volume"].iloc[-5:].mean()
    close = float(row["Close"])
    ma75 = hist["Close"].iloc[-MA_PERIOD:].mean()

    sig.vol_20d = float(vol20) if pd.notna(vol20) else None
    if vol5 and vol5 > 0 and pd.notna(row["Volume"]):
        sig.vol_ratio = float(row["Volume"]) / float(vol5)
    sig.above_75ma = close > float(ma75) if pd.notna(ma75) else None


def enrich_roe(sig: Signal, roe_cache: dict[str, float | None]) -> None:
    if sig.ticker in roe_cache:
        sig.roe_pct = roe_cache[sig.ticker]
        return
    import yfinance as yf

    val: float | None = None
    try:
        info = yf.Ticker(sig.ticker).info or {}
        roe = info.get("returnOnEquity")
        if roe is not None and isinstance(roe, (int, float)) and roe == roe:
            val = float(roe) * 100.0 if abs(float(roe)) <= 1.5 else float(roe)
    except Exception:
        pass
    roe_cache[sig.ticker] = val
    sig.roe_pct = val


def proposed_score(sig: Signal, roe_default: float = 5.0) -> float:
    """提案100点満点の近似（セクター未取得時は需給・テクニカル・一次・ROEのみ）。"""
    score = 0.0
    if sig.entry >= PROPOSED_MIN_PRICE and (sig.vol_20d or 0) >= PROPOSED_MIN_VOL_20D:
        score += 15
    if sig.above_75ma:
        score += 20
    if sig.pattern_name:
        score += 25
    vr = sig.vol_ratio or 0
    if vr >= PROPOSED_VOL_SPIKE:
        score += 20
    elif vr >= 1.2:
        score += 10
    roe = sig.roe_pct if sig.roe_pct is not None else roe_default
    if roe >= 8.0:
        score += 10
    if sig.sector_momentum_ok:
        score += 10
    return min(100.0, score)


@dataclass
class GroupStats:
    name: str
    n: int = 0
    n_realized: int = 0
    sum_bt_ret: float = 0.0
    sum_realized: float = 0.0
    wins_bt: int = 0
    wins_realized: int = 0
    sum_cost_100: float = 0.0
    sum_pnl_100: float = 0.0

    def add(self, sig: Signal) -> None:
        self.n += 1
        self.sum_bt_ret += sig.avg_return_pct
        if sig.avg_return_pct > 0:
            self.wins_bt += 1
        if sig.realized_return_pct is not None:
            self.n_realized += 1
            self.sum_realized += sig.realized_return_pct
            if sig.realized_return_pct > 0:
                self.wins_realized += 1
            cost = sig.entry * 100
            self.sum_cost_100 += cost
            self.sum_pnl_100 += cost * sig.realized_return_pct / 100.0

    @property
    def avg_bt_ret(self) -> float | None:
        return self.sum_bt_ret / self.n if self.n else None

    @property
    def avg_realized(self) -> float | None:
        return self.sum_realized / self.n_realized if self.n_realized else None

    @property
    def win_rate_realized(self) -> float | None:
        return self.wins_realized / self.n_realized * 100 if self.n_realized else None

    @property
    def weighted_realized(self) -> float | None:
        return self.sum_pnl_100 / self.sum_cost_100 * 100 if self.sum_cost_100 else None


def summarize(groups: dict[str, GroupStats]) -> None:
    print("| グループ | 件数 | BT平均% | 実現平均% | 実現勝率% | 100株加重% |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, g in groups.items():
        bt = f"{g.avg_bt_ret:.3f}" if g.avg_bt_ret is not None else "—"
        ar = f"{g.avg_realized:.3f}" if g.avg_realized is not None else "—"
        wr = f"{g.win_rate_realized:.1f}" if g.win_rate_realized is not None else "—"
        w = f"{g.weighted_realized:.3f}" if g.weighted_realized is not None else "—"
        print(f"| {name} | {g.n} | {bt} | {ar} | {wr} | {w} |")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="4象限フィルタの精度検証")
    parser.add_argument("--local-dir", type=Path, default=None, help="JSONローカルDir（未指定時GitHub）")
    parser.add_argument("--skip-yfinance", action="store_true", help="OHLCV/ROE取得をスキップ（entryのみ）")
    args = parser.parse_args()

    signals = load_signals(args.local_dir)
    if not signals:
        print("シグナルなし", file=sys.stderr)
        sys.exit(1)

    groups: dict[str, GroupStats] = {}

    def _grp(name: str) -> GroupStats:
        if name not in groups:
            groups[name] = GroupStats(name=name)
        return groups[name]

    if not args.skip_yfinance:
        ohlcv_cache: dict[str, pd.DataFrame] = {}
        roe_cache: dict[str, float | None] = {}
        tickers_done: set[str] = set()
        for i, sig in enumerate(signals):
            enrich_ohlcv(sig, ohlcv_cache)
            if sig.ticker not in tickers_done:
                enrich_roe(sig, roe_cache)
                tickers_done.add(sig.ticker)
            else:
                sig.roe_pct = roe_cache.get(sig.ticker)
            if (i + 1) % 50 == 0:
                print(f"# OHLCV enrich {i+1}/{len(signals)}", file=sys.stderr)

        for sig in signals:
            _grp("全シグナル（現行出力）").add(sig)
            if sig.entry >= CURRENT_MIN_PRICE:
                _grp("現行一次: 株価>=200").add(sig)
            if sig.entry >= PROPOSED_MIN_PRICE:
                _grp("提案一次: 株価>=500").add(sig)
            if sig.vol_20d is not None and sig.vol_20d >= PROPOSED_MIN_VOL_20D and sig.entry >= PROPOSED_MIN_PRICE:
                _grp("提案一次: 500円+30万株").add(sig)
            if sig.above_75ma:
                _grp("提案テクニカル: 75MA上").add(sig)
            if sig.vol_ratio is not None and sig.vol_ratio >= PROPOSED_VOL_SPIKE:
                _grp("提案需給: 出来高1.5倍").add(sig)
            if (
                sig.entry >= PROPOSED_MIN_PRICE
                and (sig.vol_20d or 0) >= PROPOSED_MIN_VOL_20D
                and sig.above_75ma
                and sig.vol_ratio is not None
                and sig.vol_ratio >= PROPOSED_VOL_SPIKE
            ):
                _grp("提案コア4条件").add(sig)
            roe = sig.roe_pct if sig.roe_pct is not None else 5.0
            if roe >= 8.0:
                _grp("ROE>=8%（欠損は5%扱いで除外）").add(sig)

        scored = [(proposed_score(s), s) for s in signals if s.above_75ma is not None]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            k = max(1, len(scored) // 5)
            for _, s in scored[:k]:
                _grp("総合スコア上位20%").add(s)
            for _, s in scored[k : k * 2]:
                _grp("総合スコア中位20%").add(s)
            for _, s in scored[-k:]:
                _grp("総合スコア下位20%").add(s)

    print(f"# 検証対象: {len(signals)} 件（買いシグナル）")
    print(f"# BT指標: JSON avg_return_pct（パターン過去BT・翌寄り3日）")
    print(f"# 実現指標: シグナル日引け買い→3営業日後引け（verify_signal_returns 準拠）")
    print()
    summarize(groups)

    base = groups.get("全シグナル（現行出力）")
    prop = groups.get("提案コア4条件")
    if base and prop and prop.n_realized and base.avg_realized is not None and prop.avg_realized is not None:
        delta = prop.avg_realized - base.avg_realized
        print()
        print("## 判定メモ（実現リターン基準）")
        print(f"- 提案コア4条件 vs 全体: 実現平均 **{delta:+.3f}pt**（{prop.n_realized}件 / 全体{base.n_realized}件）")
        if delta > 0.3:
            print("- **精度向上の可能性: 中**（実現リターンが明確に改善）")
        elif delta > 0:
            print("- **精度向上の可能性: 低**（微改善。件数減とのトレードオフ要確認）")
        else:
            print("- **精度向上の可能性: 低〜なし**（実現リターンが改善しない）")


if __name__ == "__main__":
    main()
