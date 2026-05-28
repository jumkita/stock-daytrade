"""スクリーニング結果の出力整形。"""
from __future__ import annotations

from dataclasses import dataclass

from quadrant_screening.scoring import ScoreBreakdown
from quadrant_screening.ticker_utils import display_code


@dataclass
class ScreenCandidate:
    ticker: str
    name: str
    score: ScoreBreakdown
    sign: str
    vol_ratio: float
    roe_pct: float
    price: float
    tp: float
    sl: float


def format_candidate_line(c: ScreenCandidate) -> str:
    code = display_code(c.ticker)
    sign = c.sign or "—"
    vol_s = f"{c.vol_ratio:.1f}" if c.vol_ratio == c.vol_ratio else "—"
    roe_s = f"{c.roe_pct:.1f}" if c.roe_pct == c.roe_pct else "—"
    return (
        f"【{code}】総合スコア: {c.score.total:.0f}点 | "
        f"サイン: {sign} | 需給(出来高): {vol_s}倍 | "
        f"セクター: {c.score.sector_label} | ROE: {roe_s}% | "
        f"現在値: ¥{c.price:,.0f} | TP: ¥{c.tp:,.0f} | SL: ¥{c.sl:,.0f}"
    )


def format_output(candidates: list[ScreenCandidate]) -> str:
    lines = [format_candidate_line(c) for c in candidates]
    return "\n".join(lines)
