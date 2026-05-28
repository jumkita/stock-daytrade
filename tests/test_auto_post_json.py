"""auto_post JSON シリアライズのテスト。"""
from __future__ import annotations

from ui_helpers import serialize_buy_signal_item


def test_serialize_buy_signal_item_includes_quadrant_breakdown():
    item = {
        "ticker": "6981.T",
        "name": "MURATA",
        "pattern_name": "包み線",
        "win_rate": 0.63,
        "quadrant_score": 90.0,
        "quadrant_breakdown": {
            "sector": 15,
            "volume": 25,
            "technical": 40.0,
            "fundamental": 10.0,
        },
        "quadrant_sign": "包み線",
        "formatted_line": "line",
    }
    out = serialize_buy_signal_item(item)
    assert out["quadrant_breakdown"]["sector"] == 15
    assert out["quadrant_sign"] == "包み線"
    assert out["quadrant_score"] == 90.0


def test_serialize_buy_signal_item_preserves_none_breakdown():
    out = serialize_buy_signal_item({"ticker": "7203.T"})
    assert "quadrant_breakdown" in out
    assert out["quadrant_breakdown"] is None
