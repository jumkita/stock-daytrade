# stock-daytrade

日次買いシグナル生成（`auto_post.py`）、4象限統合スクリーニング、Streamlit ダッシュボード。

- **ダッシュボード:** https://stock-daytrade.streamlit.app
- **日次 JSON:** `daily_buy_signals.json`（GitHub Actions が平日更新）

## Streamlit UI

```bash
pip install -r requirements.txt
streamlit run app.py
```

タブ構成:

| タブ | 内容 |
|------|------|
| 📊 日次シグナル | GitHub JSON 自動読込、4象限スコア表、内訳 chart、買い/売り並列 |
| 🔍 銘柄分析 | 単銘柄チャート・適正株価、バッチ割安スキャン |
| 📈 検証 | 引け購入リターン、100株ポートフォリオ損益 |
| ⚙️ 設定 | Secrets 確認、Gemini 疎通 |

サイドバーで **表示モード**（4象限スコア順 / 勝率70% / セクター良 / 全件）と **並び順** を切り替え可能。

### Streamlit Secrets 例

```toml
DAILY_SIGNALS_JSON_URL = "https://raw.githubusercontent.com/jumkita/stock-daytrade/main/daily_buy_signals.json"
GEMINI_API_KEY = "..."
```

## 4象限スクリーニング（100点満点）

| 象限 | 配点 |
|------|------|
| セクター（17業種ETF vs TOPIX） | 最大 15 |
| 需給（出来高倍率 ≥1.25 / ≥1.15） | 最大 25 |
| テクニカル（75MA + 買いパターン） | 最大 40 |
| ファンダ（ROE≥8% + 業種別割安度） | 最大 20 |

ファンダの割安度:

- 東証17業種ごとの **中央値 PER** を使用（一律15倍は廃止）
- **成長率**（earningsGrowth / revenueGrowth）に応じて許容 PER を最大3倍まで拡張
- 半導体等は `HIGH_GROWTH_TICKERS` ホワイトリストで許容 PER 40倍

```bash
python scripts/run_quadrant_screen.py
```

## 日次シグナル（GitHub Actions）

平日 **06:00 UTC（JST 15:00 相当）** を目安に `schedule` でキューされます。GitHub Actions の仕様上、**開始時刻は保証されず**遅延することがあります。また GitHub 上で見える更新時刻は、多くの場合 **`git push` が完了した時刻（ジョブ終了付近）** です。15:00 JST ちょうどにコミットが付くことは期待しないでください。

精密に時刻を合わせたい場合の例:

1. **手動:** Actions の「Run workflow」から `workflow_dispatch` を実行する。
2. **外部スケジューラ（JST 固定）:** 指定時刻に GitHub API の `repository_dispatch` を呼び出し、イベント種別 `daily-buy-signals` で本ワークフローと同じジョブを起動する（PAT はリポジトリシークレットに置き、クライアント側に書かない）。

```bash
# 例: 毎営業日 15:05 JST に実行するジョブから（GITHUB_TOKEN はリポジトリ外のシークレットを想定）
curl -sS -L -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_PAT_DISPATCH}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/jumkita/stock-daytrade/dispatches \
  -d '{"event_type":"daily-buy-signals"}'
```

スケジュール遅延の説明は GitHub 公式の [Scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule) を参照してください。

`auto_post.py` はバックテスト通過後に **4象限スクリーニング**（rank モード: 除外せずスコア順）を適用します。

### 環境変数

| 変数 | 既定 | 説明 |
|------|------|------|
| `QUADRANT_SCREEN` | `1` | `0` で4象限を無効化 |
| `QUADRANT_MODE` | `rank` | `rank`=除外せず並べ替え / `filter`=足切り |
| `QUADRANT_MIN_SCORE` | `0`（rank時） | filter モード時の下限点 |
| `QUADRANT_TOP_N` | 未設定 | 設定時は上位N件に制限 |
| `DAILY_SIGNALS_JSON_PATH` | 未設定 | JSON 出力先（Actions で設定） |

### JSON の items フィールド

各買いシグナルに `quadrant_score`, `quadrant_breakdown`（sector/volume/technical/fundamental）, `quadrant_sign`, `vol_ratio`, `sector_label` 等を含みます。

## 関連スクリプト

```bash
# 日次 JSON 集計
python scripts/aggregate_daily_buy_signals.py .

# 4象限フィルタの過去精度検証
python scripts/verify_quadrant_filters.py

# スコア100点検証
python scripts/verify_score_100.py

# 単体テスト（CI でも実行）
python -m pytest tests/ -q
```

## CI

| ワークフロー | トリガー | 内容 |
|-------------|---------|------|
| `Tests` | push / PR | pytest 全件 |
| `Auto Post Buy Signals to X` | 平日 cron / 手動 / `repository_dispatch`（`daily-buy-signals`） | 日次 JSON 生成・push |
