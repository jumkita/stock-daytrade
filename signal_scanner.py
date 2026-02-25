# -*- coding: utf-8 -*-
"""
ローソク足パターン検知モジュール
判定ロジック定義に基づく50種類のパターン検知
LINE Messaging API による通知対応
"""

import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Optional, List

# .env から環境変数を読み込む（python-dotenv がインストールされている場合）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

# ========== 基本定義 (共通変数) ==========
# 実体 (Body) = abs(Close - Open)
# 上ヒゲ (UpperShadow) = High - max(Open, Close)
# 下ヒゲ (LowerShadow) = min(Open, Close) - Low
# 陽線 (Bull) = Close > Open
# 陰線 (Bear) = Open > Close


def _body(row: pd.Series) -> float:
    """実体 = abs(Close - Open)"""
    return abs(row["Close"] - row["Open"])


def _upper_shadow(row: pd.Series) -> float:
    """上ヒゲ = High - max(Open, Close)"""
    return row["High"] - max(row["Open"], row["Close"])


def _lower_shadow(row: pd.Series) -> float:
    """下ヒゲ = min(Open, Close) - Low"""
    return min(row["Open"], row["Close"]) - row["Low"]


def _bull(row: pd.Series) -> bool:
    """陽線 = Close > Open"""
    return row["Close"] > row["Open"]


def _bear(row: pd.Series) -> bool:
    """陰線 = Open > Close"""
    return row["Open"] > row["Close"]


def _range_hl(row: pd.Series) -> float:
    """High - Low（極小判定用）"""
    return row["High"] - row["Low"]


# 極小実体の閾値: 全体の10%未満を「極小」とする
def _body_is_tiny(row: pd.Series, min_range: float = 1e-8) -> bool:
    r = _range_hl(row)
    if r <= min_range:
        return True
    return _body(row) < r * 0.1


def _body_is_small(row: pd.Series, avg_body: float, min_range: float = 1e-8) -> bool:
    """実体が平均に比べて極端に小さくないか（赤三兵で使用）"""
    if avg_body <= min_range:
        return False
    return _body(row) >= avg_body * 0.3


# ========== 監視対象銘柄 ==========
# 監視銘柄: 三菱化工機, SBIホールディングス, エヌエフホールディングス, 三菱重工業, エクサウィザーズ, 三菱商事, 三菱UFJフィナンシャルグループ
WATCH_SYMBOLS = ["6331.T", "8473.T", "6864.T", "7011.T", "4259.T", "8058.T", "8306.T"]

# 銘柄コード → 会社名（通知文表示用）
SYMBOL_TO_NAME = {
    "6331.T": "三菱化工機",
    "8473.T": "SBIホールディングス",
    "6864.T": "エヌエフホールディングス",
    "7011.T": "三菱重工業",
    "4259.T": "エクサウィザーズ",
    "8058.T": "三菱商事",
    "8306.T": "三菱UFJフィナンシャルグループ",
}


def get_display_name(symbol: str) -> str:
    """銘柄コードから通知用の表示名（会社名）を返す。未登録なら銘柄コードのまま。"""
    return SYMBOL_TO_NAME.get(symbol, symbol)


def fetch_ohlc_yfinance(
    symbol: str,
    period: str = "3mo",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    yfinance で株価OHLCを取得する。
    symbol: 銘柄コード（例: "7203.T"）
    period: 取得期間（"1mo", "3mo", "6mo", "1y" など）
    interval: 足種（"1d"=日足）
    戻り値: 列 Open, High, Low, Close を持つ DataFrame。失敗時は None。
    """
    try:
        import yfinance as yf
        from logic import flatten_ohlcv_columns
    except ImportError:
        return None
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        df = flatten_ohlcv_columns(df)
        if df is None:
            return None
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                return None
        df = df[["Open", "High", "Low", "Close"]].dropna(how="all")
        df = df.reset_index()
        return df
    except Exception:
        return None


# ========== パターン名（日本語）→ UI表示用 ==========
PATTERN_NAMES_JA = {
    "aka_sanpei": "赤三兵",
    "kuro_sanpei": "黒三兵",
    "sanzun": "三尊（ヘッドアンドショルダー）",
    "gyakusanzun": "逆三尊",
    "akenomyojo": "明けの明星",
    "yoinomyojo": "宵の明星",
    "shimohige_pinbar": "下ヒゲピンバー（たくり線）",
    "uehige_pinbar": "上ヒゲピンバー（首吊り線/流れ星）",
    "tonbo_toba": "トンボ/塔婆",
    "harami": "はらみ線",
    "tsutsumi": "つつみ線",
    "reversal_high": "リバーサルハイ",
    "reversal_low": "リバーサルロー",
    "sanku": "三空",
    "age_sanpo": "上げ三法",
    "sage_sanpo": "下げ三法",
    "nihon_takuri": "二本たくり",
    "aka_sanpei_sakizume": "赤三兵の先詰",
    "doji": "同事線（ドージ）",
    "harami_bull": "陽線はらみ",
    "harami_bear": "陰線はらみ",
    "tsutsumi_bull": "陽線つつみ",
    "tsutsumi_bear": "陰線つつみ",
    "doji_dragonfly": "トンボ十字",
    "doji_gravestone": "墓石十字",
    "long_white": "大陽線",
    "long_black": "大陰線",
    "hammer": "ハンマー",
    "hanging_man": "首吊り",
    "inverted_hammer": "逆ハンマー",
    "shooting_star": "流れ星",
    "three_white_soldiers": "赤三兵（英名）",
    "three_black_crows": "黒三兵（英名）",
    "morning_doji_star": "明けの明星（ドージ付き）",
    "evening_doji_star": "宵の明星（ドージ付き）",
    "bullish_engulfing": "陽線つつみ（買い）",
    "bearish_engulfing": "陰線つつみ（売り）",
    "bullish_harami": "陽線はらみ（買い）",
    "bearish_harami": "陰線はらみ（売り）",
    "gap_up": "窓開け（上）",
    "gap_down": "窓開け（下）",
    "tweezer_top": "トゥイーザートップ",
    "tweezer_bottom": "トゥイーザーボトム",
    "three_line_strike_bull": "三兵反撃（陽）",
    "three_line_strike_bear": "三兵反撃（陰）",
    "rising_three": "上昇三法",
    "falling_three": "下降三法",
    "upside_gap_two_crows": "上放れ二羽烏",
    "two_crows": "二羽烏",
    "three_stars_south": "南方三ツ星",
    "identical_three_crows": "同値三羽烏",
    "stick_sandwich": "棒サンドイッチ",
    "kicking_bull": "キッキング（陽）",
    "kicking_bear": "キッキング（陰）",
}

# パターン（日本語名）→ 買い/売り/中立（通知で「買い」「売り」を明示するため）
# 1本の足で複数パターンが検出されると買いと売りが混在することがあるため、通知では分けて表示する
PATTERN_SIDE = {
    "赤三兵": "買い",
    "黒三兵": "売り",
    "三尊（ヘッドアンドショルダー）": "売り",
    "逆三尊": "買い",
    "明けの明星": "買い",
    "宵の明星": "売り",
    "下ヒゲピンバー（たくり線）": "買い",
    "上ヒゲピンバー（首吊り線/流れ星）": "売り",
    "トンボ/塔婆": "中立",
    "はらみ線": "中立",
    "つつみ線": "中立",
    "リバーサルハイ": "売り",
    "リバーサルロー": "買い",
    "三空": "中立",
    "上げ三法": "買い",
    "下げ三法": "売り",
    "二本たくり": "買い",
    "赤三兵の先詰": "売り",
    "同事線（ドージ）": "中立",
    "陽線はらみ": "買い",
    "陰線はらみ": "売り",
    "陽線つつみ": "買い",
    "陰線つつみ": "売り",
    "トンボ十字": "買い",
    "墓石十字": "売り",
    "大陽線": "買い",
    "大陰線": "売り",
    "長大陰線後の反転陽線": "買い",
    "長大陽線後の反転陰線": "売り",
    "窓開け（上）": "買い",
    "窓開け（下）": "売り",
    "トゥイーザーボトム": "買い",
    "トゥイーザートップ": "売り",
    "三兵反撃（陽）": "買い",
    "三兵反撃（陰）": "売り",
    "二羽烏": "売り",
    "上放れ二羽烏": "売り",
    "南方三ツ星": "買い",
    "棒サンドイッチ": "買い",
    "キッキング（陽）": "買い",
    "キッキング（陰）": "売り",
    "同値三羽烏": "売り",
    "ハンマー": "買い",
    "首吊り": "売り",
    "逆ハンマー": "買い",
    "流れ星": "売り",
    "上昇三法": "買い",
    "下降三法": "売り",
}


class CandlePatterns:
    """
    ローソク足パターン検知クラス。
    pandas DataFrame (列: Open, High, Low, Close) とインデックス i を渡して各パターンを判定する。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._nan = np.nan

    def _row(self, i: int) -> Optional[pd.Series]:
        if i < 0 or i >= len(self.df):
            return None
        return self.df.iloc[i]

    def _safe(self, i: int, *more: int) -> bool:
        """必要な本数だけ過去にデータがあるか"""
        indices = [i] + list(more)
        return all(0 <= k < len(self.df) for k in indices)

    # ---------- 1. 基本・トレンド転換（重要度：高） ----------

    def is_aka_sanpei(self, i: int) -> bool:
        """赤三兵: 3本連続陽線 かつ 終値が連続して上昇 かつ 各実体が極端に小さくない。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not (_bull(r0) and _bull(r1) and _bull(r2)):
            return False
        if not (r2["Close"] < r1["Close"] < r0["Close"]):
            return False
        avg_body = (_body(r0) + _body(r1) + _body(r2)) / 3.0
        return _body_is_small(r0, avg_body) and _body_is_small(r1, avg_body) and _body_is_small(r2, avg_body)

    def is_kuro_sanpei(self, i: int) -> bool:
        """黒三兵: 3本連続陰線 かつ 終値が連続して下降。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not (_bear(r0) and _bear(r1) and _bear(r2)):
            return False
        return r2["Close"] > r1["Close"] > r0["Close"]

    def is_sanzun(self, i: int, lookback: int = 20) -> bool:
        """三尊: 直近lookback本の中で「高値・最高値・高値」の山を形成し、現在の価格がネックライン（谷の安値）を下回っている。"""
        if i < lookback or i >= len(self.df):
            return False
        highs = [self.df.iloc[k]["High"] for k in range(i - lookback, i + 1)]
        lows = [self.df.iloc[k]["Low"] for k in range(i - lookback, i + 1)]
        imax = np.argmax(highs)
        # 左肩: 0..imax-1 の最大、頭: imax、右肩: imax+1.. の最大
        if imax <= 0 or imax >= len(highs) - 1:
            return False
        left_shoulder = max(highs[:imax]) if imax > 0 else highs[0]
        head = highs[imax]
        right_shoulder = max(highs[imax + 1 :]) if imax + 1 < len(highs) else highs[-1]
        if not (left_shoulder < head and right_shoulder < head):
            return False
        # 谷の安値（ネックライン）: 左谷と右谷の安値のうち高い方（簡易）
        left_trough = min(lows[: imax + 1]) if imax >= 0 else lows[0]
        right_trough = min(lows[imax:]) if imax < len(lows) else lows[-1]
        neck = max(left_trough, right_trough)
        current_close = self.df.iloc[i]["Close"]
        return current_close < neck

    def is_gyakusanzun(self, i: int, lookback: int = 20) -> bool:
        """逆三尊: 直近lookback本の中で「安値・最安値・安値」の谷を形成し、ネックラインを上回っている。"""
        if i < lookback or i >= len(self.df):
            return False
        lows = [self.df.iloc[k]["Low"] for k in range(i - lookback, i + 1)]
        highs = [self.df.iloc[k]["High"] for k in range(i - lookback, i + 1)]
        imin = np.argmin(lows)
        if imin <= 0 or imin >= len(lows) - 1:
            return False
        left_bottom = min(lows[:imin]) if imin > 0 else lows[0]
        bottom = lows[imin]
        right_bottom = min(lows[imin + 1 :]) if imin + 1 < len(lows) else lows[-1]
        if not (left_bottom > bottom and right_bottom > bottom):
            return False
        left_ridge = max(highs[: imin + 1]) if imin >= 0 else highs[0]
        right_ridge = max(highs[imin:]) if imin < len(highs) else highs[-1]
        neck = min(left_ridge, right_ridge)
        current_close = self.df.iloc[i]["Close"]
        return current_close > neck

    def is_akenomyojo(self, i: int) -> bool:
        """明けの明星: [陰線(大)] → [コマ(小実体)が窓を開けて下落] → [陽線(大)が窓を開けてor被せて上昇]。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        # r2: 陰線(大)
        if not _bear(r2):
            return False
        body2 = _body(r2)
        range2 = _range_hl(r2)
        if range2 <= 0:
            return False
        if body2 < range2 * 0.5:
            return False
        # r1: コマ（小実体）が窓を開けて下落
        if not _body_is_tiny(r1):
            return False
        gap_down_2_1 = r1["High"] < r2["Low"]
        if not gap_down_2_1:
            return False
        # r0: 陽線(大)が窓を開けてor被せて上昇
        if not _bull(r0):
            return False
        body0 = _body(r0)
        if body0 < _range_hl(r0) * 0.5:
            return False
        gap_up_1_0 = r0["Low"] > r1["High"]  # 窓を開けて
        engulf_1_0 = r0["Open"] < r1["Low"] and r0["Close"] > r1["High"]  # 被せて（前日を覆う）
        return gap_up_1_0 or engulf_1_0

    def is_yoinomyojo(self, i: int) -> bool:
        """宵の明星: [陽線(大)] → [コマ(小実体)が窓を開けて上昇] → [陰線(大)が下落]。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not _bull(r2):
            return False
        body2 = _body(r2)
        if body2 < _range_hl(r2) * 0.5:
            return False
        if not _body_is_tiny(r1):
            return False
        gap_up_2_1 = r1["Low"] > r2["High"]
        if not gap_up_2_1:
            return False
        if not _bear(r0):
            return False
        return _body(r0) >= _range_hl(r0) * 0.5

    # ---------- 2. ヒゲ・ピンバー系（重要度：高） ----------

    def is_shimohige_pinbar(self, i: int) -> bool:
        """下ヒゲピンバー（たくり線）: 下ヒゲの長さ > 実体の2倍。実体は極小。"""
        r = self._row(i)
        if r is None:
            return False
        lb, body = _lower_shadow(r), _body(r)
        if body <= 0:
            return lb > 0
        return lb > body * 2 and _body_is_tiny(r)

    def is_uehige_pinbar(self, i: int) -> bool:
        """上ヒゲピンバー（首吊り線/流れ星）: 上ヒゲの長さ > 実体の2倍。実体は極小。"""
        r = self._row(i)
        if r is None:
            return False
        ub, body = _upper_shadow(r), _body(r)
        if body <= 0:
            return ub > 0
        return ub > body * 2 and _body_is_tiny(r)

    def is_tonbo_toba(self, i: int) -> bool:
        """トンボ/塔婆: 始値≒終値 で、長いヒゲが片方だけにあるもの。"""
        r = self._row(i)
        if r is None:
            return False
        body = _body(r)
        rng = _range_hl(r)
        if rng <= 0:
            return False
        if body > rng * 0.1:
            return False
        us, ls = _upper_shadow(r), _lower_shadow(r)
        long_upper = us > rng * 0.5
        long_lower = ls > rng * 0.5
        return (long_upper and not long_lower) or (long_lower and not long_upper)

    # ---------- 3. はらみ・つつみ系 ----------

    def is_harami(self, i: int) -> bool:
        """はらみ線: 前日の実体の中に、当日の実体が完全に収まっている。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        body1 = _body(r1)
        if body1 <= 0:
            return False
        open1, close1 = r1["Open"], r1["Close"]
        body0_open, body0_close = min(r0["Open"], r0["Close"]), max(r0["Open"], r0["Close"])
        body1_min, body1_max = min(open1, close1), max(open1, close1)
        return body1_min < body0_open and body0_close < body1_max

    def is_tsutsumi(self, i: int) -> bool:
        """つつみ線: 前日の実体を、当日の実体が完全に覆い隠している。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        body1 = _body(r1)
        if body1 <= 0:
            return False
        body0_min, body0_max = min(r0["Open"], r0["Close"]), max(r0["Open"], r0["Close"])
        body1_min, body1_max = min(r1["Open"], r1["Close"]), max(r1["Open"], r1["Close"])
        return body0_min < body1_min and body1_max < body0_max

    def is_reversal_high(self, i: int, n: int = 5) -> bool:
        """リバーサルハイ: 新高値を更新した後に反落して陰線引け。"""
        if i < n or i >= len(self.df):
            return False
        r = self._row(i)
        if not _bear(r):
            return False
        prev_high = max(self.df.iloc[k]["High"] for k in range(i - n, i))
        cur_high = self.df.iloc[i]["High"]
        if cur_high <= prev_high:
            return False
        return r["Close"] < self.df.iloc[i - 1]["Close"]

    def is_reversal_low(self, i: int, n: int = 5) -> bool:
        """リバーサルロー: 新安値を更新した後に反発して陽線引け。"""
        if i < n or i >= len(self.df):
            return False
        r = self._row(i)
        if not _bull(r):
            return False
        prev_low = min(self.df.iloc[k]["Low"] for k in range(i - n, i))
        cur_low = self.df.iloc[i]["Low"]
        if cur_low >= prev_low:
            return False
        return r["Close"] > self.df.iloc[i - 1]["Close"]

    # ---------- 4. 酒田五法・特殊形状 ----------

    def is_sanku(self, i: int) -> bool:
        """三空: 3回連続で「窓（Gap）」を開けている状態。"""
        if not self._safe(i, i - 1, i - 2, i - 3):
            return False
        # 窓: 前日High < 当日Low（上昇窓） or 前日Low > 当日High（下降窓）
        gaps = 0
        for k in range(i, i - 3, -1):
            curr = self.df.iloc[k]
            prev = self.df.iloc[k - 1]
            gap_up = prev["High"] < curr["Low"]
            gap_down = prev["Low"] > curr["High"]
            if gap_up or gap_down:
                gaps += 1
            else:
                break
        return gaps >= 3

    def is_age_sanpo(self, i: int) -> bool:
        """上げ三法: [大陽線]→[小陰線3つが陽線内にはらむ]→[大陽線]。"""
        if not self._safe(i, i - 1, i - 2, i - 3, i - 4):
            return False
        r0, r1, r2, r3, r4 = [self._row(i - k) for k in range(5)]
        body4 = _body(r4)
        range4 = _range_hl(r4)
        if not _bull(r4) or (range4 <= 0 or body4 < range4 * 0.5):
            return False
        body0 = _body(r0)
        range0 = _range_hl(r0)
        if not _bull(r0) or (range0 <= 0 or body0 < range0 * 0.5):
            return False
        high4, low4 = r4["High"], r4["Low"]
        for r in (r1, r2, r3):
            if not _bear(r):
                return False
            if r["High"] > high4 or r["Low"] < low4:
                return False
        return True

    def is_sage_sanpo(self, i: int) -> bool:
        """下げ三法: [大陰線]→[小陽線3つが陰線内にはらむ]→[大陰線]。"""
        if not self._safe(i, i - 1, i - 2, i - 3, i - 4):
            return False
        r0, r1, r2, r3, r4 = [self._row(i - k) for k in range(5)]
        if not _bear(r4) or _body(r4) < _range_hl(r4) * 0.5:
            return False
        if not _bear(r0) or _body(r0) < _range_hl(r0) * 0.5:
            return False
        high4, low4 = r4["High"], r4["Low"]
        for r in (r1, r2, r3):
            if not _bull(r):
                return False
            if r["High"] > high4 or r["Low"] < low4:
                return False
        return True

    def is_nihon_takuri(self, i: int) -> bool:
        """二本たくり: 下ヒゲの長いローソク足が2本連続。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        for r in (r0, r1):
            ls, body = _lower_shadow(r), _body(r)
            if body <= 0:
                if ls <= 0:
                    return False
                continue
            if ls <= body * 2:
                return False
        return True

    def is_aka_sanpei_sakizume(self, i: int) -> bool:
        """赤三兵の先詰: 赤三兵だが、3本目の実体や上ヒゲが短くなり勢いが衰えている。"""
        if not self.is_aka_sanpei(i):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        body0, body1 = _body(r0), _body(r1)
        us0 = _upper_shadow(r0)
        range0 = _range_hl(r0)
        return body0 <= body1 and (range0 <= 0 or us0 <= range0 * 0.3)

    def is_doji(self, i: int) -> bool:
        """同事線（Doji）: 実体がほぼゼロ ( abs(Open - Close) < (High - Low) * 0.1 )。"""
        r = self._row(i)
        if r is None:
            return False
        rng = _range_hl(r)
        if rng <= 0:
            return True
        return _body(r) < rng * 0.1

    # ---------- 派生・簡易パターン（UI用に名前を分けたもの） ----------

    def is_harami_bull(self, i: int) -> bool:
        """陽線はらみ: はらみで当日が陽線。"""
        return self.is_harami(i) and self._row(i) is not None and _bull(self._row(i))

    def is_harami_bear(self, i: int) -> bool:
        """陰線はらみ: はらみで当日が陰線。"""
        return self.is_harami(i) and self._row(i) is not None and _bear(self._row(i))

    def is_tsutsumi_bull(self, i: int) -> bool:
        """陽線つつみ: つつみで当日が陽線。"""
        return self.is_tsutsumi(i) and self._row(i) is not None and _bull(self._row(i))

    def is_tsutsumi_bear(self, i: int) -> bool:
        """陰線つつみ: つつみで当日が陰線。"""
        return self.is_tsutsumi(i) and self._row(i) is not None and _bear(self._row(i))

    def is_doji_dragonfly(self, i: int) -> bool:
        """トンボ十字: 実体極小で下ヒゲが長い。"""
        r = self._row(i)
        if r is None or not self.is_doji(i):
            return False
        return _lower_shadow(r) > _upper_shadow(r) and _lower_shadow(r) > _body(r)

    def is_doji_gravestone(self, i: int) -> bool:
        """墓石十字: 実体極小で上ヒゲが長い。"""
        r = self._row(i)
        if r is None or not self.is_doji(i):
            return False
        return _upper_shadow(r) > _lower_shadow(r) and _upper_shadow(r) > _body(r)

    def is_long_white(self, i: int, avg_body: Optional[float] = None) -> bool:
        """大陽線: 陽線で実体が大きい。"""
        r = self._row(i)
        if r is None or not _bull(r):
            return False
        if avg_body is None and len(self.df) > 0:
            avg_body = self.df.apply(_body, axis=1).mean()
        if avg_body is None or avg_body <= 0:
            return _body(r) >= _range_hl(r) * 0.6
        return _body(r) >= avg_body * 1.5

    def is_long_black(self, i: int, avg_body: Optional[float] = None) -> bool:
        """大陰線: 陰線で実体が大きい。"""
        r = self._row(i)
        if r is None or not _bear(r):
            return False
        if avg_body is None and len(self.df) > 0:
            avg_body = self.df.apply(_body, axis=1).mean()
        if avg_body is None or avg_body <= 0:
            return _body(r) >= _range_hl(r) * 0.6
        return _body(r) >= avg_body * 1.5

    # 長大陽線/陰線後の反転形状（鷹返し等の簡易実装）
    def is_long_candle_reversal_bull(self, i: int) -> bool:
        """長大陰線後の反転陽線（簡易）。※鷹返し等の厳密定義の代わり。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        return _bear(r1) and _body(r1) >= _range_hl(r1) * 0.5 and _bull(r0) and r0["Close"] > r1["Open"]

    def is_long_candle_reversal_bear(self, i: int) -> bool:
        """長大陽線後の反転陰線（簡易）。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        return _bull(r1) and _body(r1) >= _range_hl(r1) * 0.5 and _bear(r0) and r0["Close"] < r1["Open"]

    def is_gap_up(self, i: int) -> bool:
        """窓開け（上）: 前日High < 当日Low。"""
        if not self._safe(i, i - 1):
            return False
        return self.df.iloc[i - 1]["High"] < self.df.iloc[i]["Low"]

    def is_gap_down(self, i: int) -> bool:
        """窓開け（下）: 前日Low > 当日High。"""
        if not self._safe(i, i - 1):
            return False
        return self.df.iloc[i - 1]["Low"] > self.df.iloc[i]["High"]

    def is_tweezer_bottom(self, i: int) -> bool:
        """トゥイーザーボトム: 2本の安値がほぼ同値で下ヒゲが長い。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        if abs(r0["Low"] - r1["Low"]) > (r0["High"] - r0["Low"]) * 0.1:
            return False
        return _lower_shadow(r0) > _body(r0) and _lower_shadow(r1) > _body(r1)

    def is_tweezer_top(self, i: int) -> bool:
        """トゥイーザートップ: 2本の高値がほぼ同値で上ヒゲが長い。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        if abs(r0["High"] - r1["High"]) > (r0["High"] - r0["Low"]) * 0.1:
            return False
        return _upper_shadow(r0) > _body(r0) and _upper_shadow(r1) > _body(r1)

    def is_three_line_strike_bull(self, i: int) -> bool:
        """三兵反撃（陽）: 3本連続陰線の後、大陽線が1本目始値より上で引ける。"""
        if not self._safe(i, i - 1, i - 2, i - 3):
            return False
        r0, r1, r2, r3 = self._row(i), self._row(i - 1), self._row(i - 2), self._row(i - 3)
        if not (_bear(r1) and _bear(r2) and _bear(r3)):
            return False
        if not _bull(r0) or _body(r0) < _range_hl(r0) * 0.5:
            return False
        return r0["Close"] > r3["Open"]

    def is_three_line_strike_bear(self, i: int) -> bool:
        """三兵反撃（陰）: 3本連続陽線の後、大陰線が1本目始値より下で引ける。"""
        if not self._safe(i, i - 1, i - 2, i - 3):
            return False
        r0, r1, r2, r3 = self._row(i), self._row(i - 1), self._row(i - 2), self._row(i - 3)
        if not (_bull(r1) and _bull(r2) and _bull(r3)):
            return False
        if not _bear(r0) or _body(r0) < _range_hl(r0) * 0.5:
            return False
        return r0["Close"] < r3["Open"]

    def is_two_crows(self, i: int) -> bool:
        """二羽烏: 陽線の後、2本陰線で2本目が1本目実体をはらむ（高値は1本目より上）。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not _bull(r2) or not _bear(r1) or not _bear(r0):
            return False
        open1, close1 = r1["Open"], r1["Close"]
        body1_min, body1_max = min(open1, close1), max(open1, close1)
        body0_min, body0_max = min(r0["Open"], r0["Close"]), max(r0["Open"], r0["Close"])
        if not (body1_min < body0_min and body0_max < body1_max):
            return False
        return r0["High"] > r1["High"]

    def is_upside_gap_two_crows(self, i: int) -> bool:
        """上放れ二羽烏: 陽線の後、窓を開けて高値で始まり2本陰線、2本目が1本目をはらむ。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not _bull(r2) or not _bear(r1) or not _bear(r0):
            return False
        if r1["Low"] <= r2["High"]:
            return False
        open1, close1 = r1["Open"], r1["Close"]
        body1_min, body1_max = min(open1, close1), max(open1, close1)
        body0_min, body0_max = min(r0["Open"], r0["Close"]), max(r0["Open"], r0["Close"])
        return body1_min < body0_min and body0_max < body1_max

    def is_three_stars_south(self, i: int) -> bool:
        """南方三ツ星: 3本連続陰線で終値がほぼ同水準（下げ止まりの兆し）。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not (_bear(r0) and _bear(r1) and _bear(r2)):
            return False
        avg_close = (r0["Close"] + r1["Close"] + r2["Close"]) / 3
        range_avg = (r0["High"] - r0["Low"] + r1["High"] - r1["Low"] + r2["High"] - r2["Low"]) / 3
        if range_avg <= 0:
            return True
        return abs(r0["Close"] - avg_close) < range_avg * 0.2 and abs(r1["Close"] - avg_close) < range_avg * 0.2

    def is_stick_sandwich(self, i: int) -> bool:
        """棒サンドイッチ: 陰線→陽線→陰線で、2本の陰線の終値がほぼ同値。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not (_bear(r0) and _bull(r1) and _bear(r2)):
            return False
        range_avg = (r0["High"] - r0["Low"] + r2["High"] - r2["Low"]) / 2
        if range_avg <= 0:
            return True
        return abs(r0["Close"] - r2["Close"]) < range_avg * 0.2

    def is_kicking_bull(self, i: int) -> bool:
        """キッキング（陽）: 窓を開けて陽線（前日は陰線で窓を開けて下落した後）。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        if not _bear(r1) or not _bull(r0):
            return False
        return r0["Low"] > r1["High"]

    def is_kicking_bear(self, i: int) -> bool:
        """キッキング（陰）: 窓を開けて陰線（前日は陽線で窓を開けて上昇した後）。"""
        if not self._safe(i, i - 1):
            return False
        r0, r1 = self._row(i), self._row(i - 1)
        if not _bull(r1) or not _bear(r0):
            return False
        return r0["High"] < r1["Low"]

    def is_identical_three_crows(self, i: int) -> bool:
        """同値三羽烏: 3本連続陰線で終値がほぼ同水準（厳密には同値）。"""
        if not self._safe(i, i - 1, i - 2):
            return False
        r0, r1, r2 = self._row(i), self._row(i - 1), self._row(i - 2)
        if not (_bear(r0) and _bear(r1) and _bear(r2)):
            return False
        avg = (r0["Close"] + r1["Close"] + r2["Close"]) / 3
        rng = (r0["High"] - r0["Low"] + r1["High"] - r1["Low"] + r2["High"] - r2["Low"]) / 3
        if rng <= 0:
            return True
        return abs(r0["Close"] - avg) < rng * 0.1 and abs(r1["Close"] - avg) < rng * 0.1 and abs(r2["Close"] - avg) < rng * 0.1

    def is_hammer(self, i: int) -> bool:
        """ハンマー: 下ヒゲピンバー（陽線）の別名。"""
        return self.is_shimohige_pinbar(i) and self._row(i) is not None and _bull(self._row(i))

    def is_hanging_man(self, i: int) -> bool:
        """首吊り: 上昇後の下ヒゲピンバー（陰線）。"""
        return self.is_shimohige_pinbar(i) and self._row(i) is not None and _bear(self._row(i))

    def is_inverted_hammer(self, i: int) -> bool:
        """逆ハンマー: 上ヒゲピンバーで陽線（下落後の反転候補）。"""
        return self.is_uehige_pinbar(i) and self._row(i) is not None and _bull(self._row(i))

    def is_shooting_star(self, i: int) -> bool:
        """流れ星: 上ヒゲピンバーで陰線。"""
        return self.is_uehige_pinbar(i) and self._row(i) is not None and _bear(self._row(i))

    # ---------- 全パターン一覧（メソッド名 → 日本語名） ----------
    PATTERN_METHODS = [
        ("aka_sanpei", "赤三兵"),
        ("kuro_sanpei", "黒三兵"),
        ("sanzun", "三尊（ヘッドアンドショルダー）"),
        ("gyakusanzun", "逆三尊"),
        ("akenomyojo", "明けの明星"),
        ("yoinomyojo", "宵の明星"),
        ("shimohige_pinbar", "下ヒゲピンバー（たくり線）"),
        ("uehige_pinbar", "上ヒゲピンバー（首吊り線/流れ星）"),
        ("tonbo_toba", "トンボ/塔婆"),
        ("harami", "はらみ線"),
        ("tsutsumi", "つつみ線"),
        ("reversal_high", "リバーサルハイ"),
        ("reversal_low", "リバーサルロー"),
        ("sanku", "三空"),
        ("age_sanpo", "上げ三法"),
        ("sage_sanpo", "下げ三法"),
        ("nihon_takuri", "二本たくり"),
        ("aka_sanpei_sakizume", "赤三兵の先詰"),
        ("doji", "同事線（ドージ）"),
        ("harami_bull", "陽線はらみ"),
        ("harami_bear", "陰線はらみ"),
        ("tsutsumi_bull", "陽線つつみ"),
        ("tsutsumi_bear", "陰線つつみ"),
        ("doji_dragonfly", "トンボ十字"),
        ("doji_gravestone", "墓石十字"),
        ("long_white", "大陽線"),
        ("long_black", "大陰線"),
        ("long_candle_reversal_bull", "長大陰線後の反転陽線"),
        ("long_candle_reversal_bear", "長大陽線後の反転陰線"),
        ("gap_up", "窓開け（上）"),
        ("gap_down", "窓開け（下）"),
        ("tweezer_bottom", "トゥイーザーボトム"),
        ("tweezer_top", "トゥイーザートップ"),
        ("three_line_strike_bull", "三兵反撃（陽）"),
        ("three_line_strike_bear", "三兵反撃（陰）"),
        ("two_crows", "二羽烏"),
        ("upside_gap_two_crows", "上放れ二羽烏"),
        ("three_stars_south", "南方三ツ星"),
        ("stick_sandwich", "棒サンドイッチ"),
        ("kicking_bull", "キッキング（陽）"),
        ("kicking_bear", "キッキング（陰）"),
        ("identical_three_crows", "同値三羽烏"),
        ("hammer", "ハンマー"),
        ("hanging_man", "首吊り"),
        ("inverted_hammer", "逆ハンマー"),
        ("shooting_star", "流れ星"),
    ]

    def scan_index(self, i: int) -> List[str]:
        """インデックス i で検知されたパターン名（日本語）のリストを返す。"""
        result = []
        for method_name, ja_name in self.PATTERN_METHODS:
            method = getattr(self, "is_" + method_name, None)
            if method is None or not callable(method):
                continue
            try:
                if method(i):
                    result.append(ja_name)
            except Exception:
                continue
        return result

    def scan_all(self) -> pd.DataFrame:
        """全行をスキャンし、各行で検知されたパターン（日本語）を列 'detected_patterns' にリストで追加した DataFrame を返す。"""
        patterns_per_row = []
        for i in range(len(self.df)):
            patterns_per_row.append(self.scan_index(i))
        out = self.df.copy()
        out["detected_patterns"] = patterns_per_row
        return out


def scan_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    メインエントリ: OHLC DataFrame をスキャンし、検知パターン列を付与する。
    列に Open, High, Low, Close が必要。
    """
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame に列 '{col}' がありません。")
    cp = CandlePatterns(df)
    return cp.scan_all()


# ========== LINE Messaging API ==========
LINE_MESSAGING_API_PUSH = "https://api.line.me/v2/bot/message/push"
LINE_MESSAGING_API_BROADCAST = "https://api.line.me/v2/bot/message/broadcast"


def send_line_message(
    channel_access_token: str,
    message: str,
    to_user_ids: Optional[List[str]] = None,
) -> bool:
    """
    LINE Messaging API でメッセージを送信する。
    channel_access_token: チャネルアクセストークン（LINE Developers で発行）
    message: 送信するテキスト（5000文字まで）
    to_user_ids: 指定時はプッシュ（各ユーザーに送信）。None のときはブロードキャスト（友だち全員）。
    戻り値: 送信成功なら True、失敗なら False
    """
    if not channel_access_token or not message:
        return False
    text = message[:5000]
    body = {"messages": [{"type": "text", "text": text}]}
    try:
        if to_user_ids and len(to_user_ids) > 0:
            for uid in to_user_ids:
                if not uid:
                    continue
                req_body = {"to": uid, "messages": body["messages"]}
                req = urllib.request.Request(
                    LINE_MESSAGING_API_PUSH,
                    data=json.dumps(req_body).encode("utf-8"),
                    method="POST",
                    headers={
                        "Authorization": f"Bearer {channel_access_token}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=10) as res:
                    if res.status < 200 or res.status >= 300:
                        return False
            return True
        else:
            req = urllib.request.Request(
                LINE_MESSAGING_API_BROADCAST,
                data=json.dumps(body).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {channel_access_token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as res:
                return 200 <= res.status < 300
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        print(f"[LINE API エラー] HTTP {e.code}: {e.reason}", file=sys.stderr)
        print(f"[LINE API エラー] 詳細: {err_body[:500]}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[LINE 送信エラー] {type(e).__name__}: {e}", file=sys.stderr)
        return False


def scan_ohlc_and_notify_line(
    df: pd.DataFrame,
    symbol: str,
    channel_access_token: Optional[str] = None,
    line_user_ids: Optional[List[str]] = None,
    notify_latest_only: bool = True,
    date_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    OHLC をスキャンし、パターン検知時に LINE Messaging API で通知する。
    symbol: 銘柄コード（例: "7203.T"）。通知文に含まれる。
    channel_access_token: チャネルアクセストークン。未指定なら環境変数 LINE_CHANNEL_ACCESS_TOKEN を使用。
    line_user_ids: 指定時はプッシュ（該当ユーザーにのみ送信）。None のときはブロードキャスト（友だち全員）。
    notify_latest_only: True のとき直近1本のみ通知（定期実行向け）。False ならパターンが出た行すべて通知。
    date_column: 日付列名（あれば通知文に日付として表示）。
    戻り値: (scan_ohlc と同じ DataFrame, 送信した通知の件数)
    """
    result = scan_ohlc(df)
    token = channel_access_token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    sent_count = 0
    if not token:
        return result, sent_count

    n = len(result)
    if n == 0:
        return result, sent_count

    if notify_latest_only:
        indices_to_notify = [n - 1]
    else:
        indices_to_notify = [i for i in range(n) if result.iloc[i]["detected_patterns"]]

    for i in indices_to_notify:
        patterns = result.iloc[i]["detected_patterns"]
        if not patterns:
            continue
        row = result.iloc[i]
        date_str = ""
        if date_column and date_column in row.index:
            date_str = f"\n日付: {row[date_column]}"
        close_str = f"終値: {row['Close']}" if "Close" in row.index else ""
        display_name = get_display_name(symbol)
        # 買い/売り/中立で分けて表示（同じ足で両方出る場合は転換・迷いのサインの可能性）
        buy_list = [p for p in patterns if PATTERN_SIDE.get(p) == "買い"]
        sell_list = [p for p in patterns if PATTERN_SIDE.get(p) == "売り"]
        neutral_list = [p for p in patterns if PATTERN_SIDE.get(p) == "中立"]
        unknown = [p for p in patterns if PATTERN_SIDE.get(p) not in ("買い", "売り", "中立")]
        parts = []
        if buy_list:
            parts.append(f"買い: {', '.join(buy_list)}")
        if sell_list:
            parts.append(f"売り: {', '.join(sell_list)}")
        if neutral_list or unknown:
            other = neutral_list + unknown
            parts.append(f"その他: {', '.join(other)}")
        detect_str = "\n".join(parts) if parts else ", ".join(patterns)
        if buy_list and sell_list:
            detect_str += "\n※同一足で買い・売り両方＝転換・迷いの可能性"
        msg = f"【チャートシグナル】{display_name}\n{date_str}\n{close_str}\n{detect_str}"
        if send_line_message(token, msg.strip(), to_user_ids=line_user_ids):
            sent_count += 1

    return result, sent_count


if __name__ == "__main__":
    # サンプル: ダミーOHLCでスキャン
    n = 50
    np.random.seed(42)
    df = pd.DataFrame({
        "Open": np.cumsum(np.random.randn(n) * 0.5) + 100,
        "High": np.nan,
        "Low": np.nan,
        "Close": np.nan,
    })
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.rand(n) * 2
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.rand(n) * 2
    df["Close"] = df["Open"] + np.random.randn(n) * 0.8
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])
    df = df[["Open", "High", "Low", "Close"]]
    result = scan_ohlc(df)
    print("検知パターン付き DataFrame の最後5行 (detected_patterns):")
    print(result[["Close", "detected_patterns"]].tail())
    print("\nパターンが検知された行のみ:")
    hit = result[result["detected_patterns"].apply(len) > 0]
    print(hit[["Close", "detected_patterns"]].to_string())
