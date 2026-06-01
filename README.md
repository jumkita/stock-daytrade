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

### いつ走るか（目標）

- **既定:** 平日 **06:15 UTC（JST 15:15）** を目安に `schedule` でキューされます。
- **狙い:** バックテスト＋ push が **おおよそ 15:25 JST まで**に終わり、Streamlit 側で **「最新データを読み込み」** またはページ再表示すれば Raw JSON が取り直せる状態にする（ジョブ所要は負荷により前後します）。`fetch_signals_json` は `Cache-Control: no-cache` とクエリ `_ts` でキャッシュを避けています。GitHub の CDN や Streamlit Cloud の挙動で数分ラグすることがあります。
- GitHub Actions の仕様上、**開始時刻は保証されません**（遅延の公式説明は [Scheduled events](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)）。コミット時刻は多くの場合 **`git push` 完了時刻**です。

### Automation で「15:15 起動」を組み込む

GitHub の `schedule` だけだとキュー遅延があり得るため、**JST 15:15 に確実にキューしたい**場合は、外部の定時サービスから **`repository_dispatch`** を叩いてください（イベント種別は `daily-buy-signals`）。リポジトリの `schedule` はフォールバックとして残せます。

設定例（[cron-job.org](https://cron-job.org) など）:

| 項目 | 値 |
|------|-----|
| スケジュール | 月〜金 **15:15**（タイムゾーン **Asia/Tokyo**） |
| メソッド | `POST` |
| URL | `https://api.github.com/repos/jumkita/stock-daytrade/dispatches` |
| ヘッダ | `Accept: application/vnd.github+json`、`Authorization: Bearer <PAT>`、`X-GitHub-Api-Version: 2022-11-28` |
| ボディ | `{"event_type":"daily-buy-signals"}` |

PAT は **クライアント側の画面に貼らず**、サービスが提供する秘密欄に保存してください。権限は GitHub 公式の [Create a repository dispatch event](https://docs.github.com/en/rest/repos/repos#create-a-repository-dispatch-event) に従い、classic なら通常 **`repo`**、fine-grained なら当該リポジトリで dispatch に必要な権限を付与します。

手動で同じことをする場合: Actions の「Run workflow」から `workflow_dispatch` を実行。

```bash
# 例: 自前サーバの cron から毎営業日 15:15 JST（cron で TZ=Asia/Tokyo を指定）
curl -sS -L -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_PAT_DISPATCH}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/jumkita/stock-daytrade/dispatches \
  -d '{"event_type":"daily-buy-signals"}'
```

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
