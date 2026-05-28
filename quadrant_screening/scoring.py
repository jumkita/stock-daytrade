"""100点満点の4象限スコア。"""
from __future__ import annotations

from dataclasses import dataclass

from quadrant_screening.config import (
    ROE_BONUS_THRESHOLD,
    SCORE_ABOVE_MA,
    SCORE_EPS_DISCOUNT,
    SCORE_PATTERN,
    SCORE_ROE_BONUS,
    SCORE_SECTOR_OUTPERFORM,
    SCORE_VOL_STRONG,
    SCORE_VOL_WEAK,
    VOL_SPIKE_STRONG,
    VOL_SPIKE_WEAK,
)
from quadrant_screening.fundamentals import FundamentalSnapshot, eps_discount_pct
from quadrant_screening.sector import SectorMomentum
from quadrant_screening.technical import TechnicalSnapshot


@dataclass
class ScoreBreakdown:
    total: float
    sector_pts: float
    volume_pts: float
    technical_pts: float
    fundamental_pts: float
    sector_label: str


def compute_score(
    tech: TechnicalSnapshot,
    fund: FundamentalSnapshot,
    sector: SectorMomentum | None,
) -> ScoreBreakdown:
    sector_pts = 0.0
    sector_label = "—"
    if sector is not None:
        sector_label = "良" if sector.outperforms_topix else "悪"
        if sector.outperforms_topix:
            sector_pts = SCORE_SECTOR_OUTPERFORM

    volume_pts = 0.0
    if tech.vol_ratio >= VOL_SPIKE_STRONG:
        volume_pts = SCORE_VOL_STRONG
    elif tech.vol_ratio >= VOL_SPIKE_WEAK:
        volume_pts = SCORE_VOL_WEAK

    technical_pts = 0.0
    if tech.above_75ma:
        technical_pts += SCORE_ABOVE_MA
    if tech.patterns:
        technical_pts += SCORE_PATTERN

    fundamental_pts = 0.0
    if not fund.roe_is_default and fund.roe_pct >= ROE_BONUS_THRESHOLD:
        fundamental_pts += SCORE_ROE_BONUS
    disc = eps_discount_pct(tech.price, fund.trailing_eps)
    if disc is not None and disc > 0:
        fundamental_pts += SCORE_EPS_DISCOUNT

    total = min(100.0, sector_pts + volume_pts + technical_pts + fundamental_pts)
    return ScoreBreakdown(
        total=round(total, 1),
        sector_pts=sector_pts,
        volume_pts=volume_pts,
        technical_pts=technical_pts,
        fundamental_pts=fundamental_pts,
        sector_label=sector_label,
    )
