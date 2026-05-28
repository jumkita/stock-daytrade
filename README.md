# stock-daytrade

日次買いシグナル生成（`auto_post.py`）と、4象限統合スクリーニング。

## 4象限スクリーニング（検証済み最適化版）

テクニカル・ファンダ・需給・セクター（17業種ETF vs TOPIX）を100点満点でスコアリング。

```bash
pip install -r requirements.txt
python scripts/run_quadrant_screen.py
```

- 母集団: 東証プライム（`jpx_all_tickers.csv`）
- 一次: 500円以上・20日平均出来高30万株
- 必須: 75日移動平均線の上
- 需給: 出来高1.25倍で加点（1.5倍は厳しすぎるため緩和）
- ROE: yfinance 欠損時は5%で継続（8%以上のみ弱加点）

### 関連スクリプト

```bash
# 日次 JSON 集計（買いのみ）
python scripts/aggregate_daily_buy_signals.py .

# 4象限フィルタの過去精度検証
python scripts/verify_quadrant_filters.py

# 単体テスト
python -m pytest tests/test_quadrant_screening.py -q
```

## 日次シグナル（GitHub Actions）

平日 JST 15:00 頃に `auto_post.py` が実行され、`daily_buy_signals.json` を更新します。
