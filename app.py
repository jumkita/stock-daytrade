# -*- coding: utf-8 -*-
"""
Streamlit UI: 適正株価・バリュートラップ検知 + 24種買い/26種売りパターン
"""
import json
import os
import urllib.request
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from backtest_engine import BacktestEngine, plot_backtest_results

# ----- Gemini API Key Loading (Cloud: st.secrets | Local: env) -----
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    if not isinstance(gemini_api_key, str) or not gemini_api_key.strip():
        gemini_api_key = ""
    else:
        gemini_api_key = gemini_api_key.strip()
except (FileNotFoundError, KeyError):
    gemini_api_key = ""

if not gemini_api_key:
    gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

api_ready = bool(gemini_api_key)
GEMINI_SECRETS = {"GEMINI_API_KEY": gemini_api_key} if api_ready else {}
from logic import (
    fetch_ohlcv,
    detect_all_patterns,
    get_downtrend_mask,
    get_fair_value,
    check_value_trap,
    gemini_echo_ticker,
)
from auto_post import scan_hybrid, build_tweet
from batch_value_screen import run_batch, format_line
from verify_signal_returns import verify_returns, summary_stats


def _list_signal_history_files(root_dir: str) -> list[tuple[pd.Timestamp, str]]:
    """daily_buy_signals_YYYY-MM-DD.json を日付降順で返す。"""
    files: list[tuple[pd.Timestamp, str]] = []
    try:
        for name in os.listdir(root_dir):
            if not name.startswith("daily_buy_signals_") or not name.endswith(".json"):
                continue
            date_str = name.replace("daily_buy_signals_", "").replace(".json", "")
            try:
                dt = pd.to_datetime(date_str)
            except Exception:
                continue
            files.append((dt, os.path.join(root_dir, name)))
    except Exception:
        return []
    files.sort(key=lambda x: x[0], reverse=True)
    return files


def _compute_portfolio_pnl_100_shares(rows: list[dict]) -> tuple[float, float]:
    """
    各銘柄100株ずつ買った場合の合計損益(¥)と投下資金に対するリターン(%)を返す。
    exit_price / entry が欠損の銘柄は除外。
    """
    total_profit = 0.0
    total_invested = 0.0
    for r in rows:
        entry = r.get("entry")
        exit_price = r.get("exit_price")
        if entry is None or exit_price is None:
            continue
        try:
            e = float(entry)
            x = float(exit_price)
        except (TypeError, ValueError):
            continue
        if e <= 0:
            continue
        total_profit += (x - e) * 100.0
        total_invested += e * 100.0
    if total_invested <= 0:
        return 0.0, 0.0
    pct = total_profit / total_invested * 100.0
    return total_profit, pct


def _render_detail_chart(ticker: str, period: str) -> None:
    """
    単一銘柄の詳細（適正株価・バリュートラップ + ローソク足 + パターン）を描画。
    """
    try:
        df = fetch_ohlcv(ticker, period=period)
    except Exception as e:
        st.error(f"株価データ取得エラー: {e}")
        return
    if df is None or df.empty:
        st.warning("株価データを取得できませんでした。")
        return

    # ----- 適正株価の算出 & バリュートラップ検知（ボタンで実行） -----
    st.subheader("適正株価の算出 & バリュートラップ検知")
    run_fair_value = st.button("適正株価の算出", key="btn_fair_value")
    run_trap = st.button("バリュートラップ検知", key="btn_value_trap")
    if run_fair_value or run_trap:
        fv = get_fair_value(ticker)
        if fv.get("error"):
            st.error(f"**{fv['error']}** — {fv.get('message', '')}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                theo = fv.get("theoretical_price")
                st.metric("適正株価", f"¥{theo:,.0f}" if theo is not None else "—")
            with col2:
                cur = fv.get("current_price")
                st.metric("現在値", f"¥{cur:,.0f}" if cur is not None else "—")
            with col3:
                dev = fv.get("deviation_pct")
                st.metric("乖離率（適正÷現在−1）", f"{dev:+.1f}%" if dev is not None else "—")
            if fv.get("message"):
                st.caption(f"計算根拠: {fv['message']}")

        if run_trap or (run_fair_value and not fv.get("error")):
            cur_price = fv.get("current_price") if not fv.get("error") else None
            if cur_price is None and df is not None and not df.empty and "Close" in df.columns:
                cur_price = float(df["Close"].iloc[-1])
            if cur_price is not None and cur_price > 0:
                trap = check_value_trap(ticker, cur_price, df, roe_min=0.08, ma_window=75)
                dev = fv.get("deviation_pct") if not fv.get("error") else None
                deviation_ok = dev is not None and dev >= 20.0
                roe_ok = trap.get("roe_ok", False)
                trend_ok = trap.get("trend_ok", False)
                if deviation_ok and roe_ok and trend_ok:
                    st.success("**【本命バリュー】** 乖離率20%以上かつROE≥8%・75日MA上回りを満たしています。", icon="✅")
                elif deviation_ok and (not roe_ok or not trend_ok):
                    st.warning(
                        "**【トラップ警告：下落トレンドまたは資本効率低迷】** "
                        "乖離率は高いが、ROE8%未満または現在値が75日MAを下回っています。",
                        icon="⚠️",
                    )
                else:
                    dev_str = f"{dev:+.1f}%" if dev is not None else "—"
                    st.info(f"乖離率 {dev_str} のため本命/トラップ判定は行いません（20%以上で表示）。ROE: {trap.get('roe_ok')} / 75MA上: {trap.get('trend_ok')}")
            else:
                st.warning("現在値が取得できないためバリュートラップ検知をスキップしました。")
    else:
        st.caption("「適正株価の算出」または「バリュートラップ検知」を押すと、PERベースの適正株価とバリュートラップ判定を表示します。")

    try:
        patterns = detect_all_patterns(df)
    except Exception:
        patterns = []
    downtrend_mask = get_downtrend_mask(df, window=25)

    # バックテスト用: patterns から Buy_* / Sell_* 列を df に追加
    for i, name, side in patterns:
        col = f"Buy_{name}" if side == "buy" else f"Sell_{name}"
        if col not in df.columns:
            df[col] = False
        df.loc[df.index[i], col] = True

    # シグナル検証設定（サイドバー）
    st.sidebar.subheader("🔍 シグナル検証設定")
    enable_backtest = st.sidebar.checkbox("バックテストを実行して選別", value=True)

    signal_cols = [c for c in df.columns if c.startswith("Buy_")]
    valid_signals: list[str] = []

    if enable_backtest:
        holding_days = st.sidebar.slider("保有期間 (営業日)", 3, 20, 5)
        stop_loss_pct = st.sidebar.slider("損切りライン (%)", 1.0, 10.0, 5.0) / 100.0
        min_win_rate = st.sidebar.slider("採用する最低勝率 (%)", 0, 100, 50)

        if signal_cols:
            engine = BacktestEngine()
            raw_results = engine.run(
                df,
                signal_columns=signal_cols,
                holding_period_days=holding_days,
                stop_loss_pct=stop_loss_pct,
            )
            # Total Trades が 5 回未満のシグナルを除外（統計的信頼性）
            results = raw_results[raw_results["Total Trades"] >= 5].copy()

            if not results.empty:
                # ランキング表示（2列）
                rank_col1, rank_col2 = st.columns(2)
                with rank_col1:
                    st.subheader("📊 勝率ランキング (Top 5)")
                    win_rank = (
                        results.sort_values("Win Rate", ascending=False)
                        .head(5)[["Signal Name", "Win Rate", "Total Trades"]]
                        .reset_index(drop=True)
                    )
                    win_rank["順位"] = range(1, len(win_rank) + 1)
                    win_rank = win_rank[["順位", "Signal Name", "Win Rate", "Total Trades"]]
                    st.dataframe(
                        win_rank.style.format({"Win Rate": "{:.1%}"}),
                        width="stretch",
                        hide_index=True,
                    )
                with rank_col2:
                    st.subheader("📈 収益力ランキング (Top 5)")
                    pf_rank = (
                        results.sort_values("Profit Factor", ascending=False)
                        .head(5)[["Signal Name", "Profit Factor", "Win Rate"]]
                        .reset_index(drop=True)
                    )
                    pf_rank["順位"] = range(1, len(pf_rank) + 1)
                    pf_rank = pf_rank[["順位", "Signal Name", "Profit Factor", "Win Rate"]]
                    st.dataframe(
                        pf_rank.style.format({
                            "Profit Factor": "{:.2f}",
                            "Win Rate": "{:.1%}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

                # ヒートマップ（ランキングの下）
                heatmap_fig = plot_backtest_results(results, kind="heatmap")
                if heatmap_fig is not None:
                    st.plotly_chart(heatmap_fig, width="stretch")

                # チャート描画対象: min_win_rate 以上 かつ Profit Factor >= 1.0
                valid_signals = results[
                    (results["Win Rate"] >= min_win_rate / 100.0)
                    & (results["Profit Factor"] >= 1.0)
                ]["Signal Name"].tolist()
                if not valid_signals:
                    st.warning(
                        "採用条件（最低勝率・Profit Factor≥1.0）を満たすシグナルがありません。"
                    )
            else:
                st.warning("Total Trades 5回以上のシグナルがありません。")
                valid_signals = signal_cols

            with st.expander("📋 バックテスト成績表（全シグナル・クリックで展開）", expanded=False):
                st.dataframe(
                    raw_results.style.format({
                        "Win Rate": "{:.1%}",
                        "Avg Return": "{:.2f}%",
                        "Profit Factor": "{:.2f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
            st.sidebar.markdown(f"**有効シグナル数:** {len(valid_signals)} / {len(signal_cols)}")
        else:
            st.warning("検証可能なシグナル列が見つかりません。")
    else:
        valid_signals = signal_cols

    # 最新シグナルステータス（一目で判断できるダッシュボード）
    last_row = df.iloc[-1]
    date_val = last_row.get("Date", df.index[-1])
    date_str = str(date_val)[:10] if date_val is not None else str(df.index[-1])
    close_price = last_row.get("Close")
    try:
        v = None if close_price is None else float(close_price)
        close_str = f"¥{v:,.0f}" if v is not None and v == v and v >= 0 else "—"
    except (TypeError, ValueError):
        close_str = "—"

    active_buys = [
        c for c in valid_signals
        if c in df.columns and bool(df[c].fillna(False).iloc[-1])
    ]
    sell_cols = [c for c in df.columns if c.startswith("Sell_")]
    active_sells = [
        c for c in sell_cols
        if bool(df[c].fillna(False).iloc[-1])
    ]

    if active_buys:
        names = ", ".join(s.replace("Buy_", "") for s in active_buys)
        st.success(
            f"# 🚨 シグナル点灯: {names}\n\n"
            f"**日付:** {date_str}　**終値:** {close_str}",
            icon="🟢",
        )
    elif active_sells:
        names = ", ".join(s.replace("Sell_", "") for s in active_sells)
        st.error(
            f"# 🚨 シグナル点灯（売り）: {names}\n\n"
            f"**日付:** {date_str}　**終値:** {close_str}",
            icon="🔴",
        )
    else:
        st.info(
            "**本日は有効なシグナルはありません（Wait）**",
            icon="⏳",
        )

    st.divider()

    df_plot = df.copy()
    x = df_plot["Date"].tolist() if "Date" in df_plot.columns else df_plot.index.tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x, open=df_plot["Open"], high=df_plot["High"],
            low=df_plot["Low"], close=df_plot["Close"], name="OHLC",
        )
    )

    # 同一日・同一方向のパターンを集約: index -> [パターン名, ...]
    # バックテスト有効時は valid_signals に含まれる買いシグナルのみ描画
    buy_agg: dict[int, list[str]] = {}
    sell_agg: dict[int, list[str]] = {}
    for i, name, side in patterns:
        if side == "buy":
            if f"Buy_{name}" in valid_signals:
                buy_agg.setdefault(i, []).append(name)
        else:
            sell_agg.setdefault(i, []).append(name)

    # チャート上は「緑の▲」「赤の▼」マーカーのみ。パターン名はホバー時のみ表示（文字は一切描画しない）
    # 下落トレンド（Close < SMA25）の買いには ⚠️ Downtrend (Risky) を付与
    if buy_agg:
        indices_buy = list(buy_agg.keys())
        hover_parts = []
        for i in indices_buy:
            txt = "買い: " + ", ".join(buy_agg[i])
            if i < len(downtrend_mask) and downtrend_mask.iloc[i]:
                txt += " ⚠️ Downtrend (Risky)"
            hover_parts.append(txt)
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in indices_buy],
                y=[df_plot.iloc[i]["Low"] * 0.98 for i in indices_buy],
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="lime", line=dict(width=1, color="darkgreen")),
                name="買い",
                hovertext=hover_parts,
                hoverinfo="text",
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    if sell_agg:
        indices_sell = list(sell_agg.keys())
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in indices_sell],
                y=[df_plot.iloc[i]["High"] * 1.02 for i in indices_sell],
                mode="markers",
                marker=dict(symbol="triangle-down", size=14, color="red", line=dict(width=1, color="darkred")),
                name="売り",
                hovertext=["売り: " + ", ".join(sell_agg[i]) for i in indices_sell],
                hoverinfo="text",
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{ticker} ローソク足 & パターン",
        xaxis_title="日付", yaxis_title="株価",
        template="plotly_white", xaxis_rangeslider_visible=False, height=500,
    )
    st.plotly_chart(fig, width="stretch")

    if patterns:
        def _date_str(i):
            if "Date" in df_plot.columns:
                v = df_plot.iloc[i]["Date"]
                return str(v)[:10] if v is not None else str(i)
            return str(i)
        def _buy_label(i, name):
            base = f"{name} ({_date_str(i)})"
            if i < len(downtrend_mask) and downtrend_mask.iloc[i]:
                return base + " ⚠️ Downtrend (Risky)"
            return base
        buy_list = [
            _buy_label(i, name)
            for i, name, s in patterns
            if s == "buy" and f"Buy_{name}" in valid_signals
        ]
        sell_list = [f"{name} ({_date_str(i)})" for i, name, s in patterns if s == "sell"]
        c1, c2 = st.columns(2)
        with c1:
            st.write("**買い**", buy_list or "なし")
        with c2:
            st.write("**売り**", sell_list or "なし")
    else:
        st.info("検出されたパターンはありません。")


def main():
    st.set_page_config(page_title="日本株 適正株価・パターン分析", layout="wide")
    st.title("日本株 適正株価・バリュートラップ × 勝ちパターン分析")

    with st.sidebar:
        st.header("設定")
        if not api_ready:
            st.warning("⚠️ APIキーが設定されていません。AI分析機能は利用できません。")
        period = st.selectbox("分析期間", ["3mo", "6mo", "1y", "2y"], index=0)
        ticker = st.text_input("銘柄コード", value=st.session_state.get("ticker_input", "8473.T"), help="例: 7203.T, 8473.T", key="ticker_input")

        st.divider()
        with st.expander("Gemini API     疎通テスト"):
            if st.button(
                "実行（銘柄名: トヨタ）",
                key="gemini_test_btn",
                disabled=not api_ready,
            ):
                msg = gemini_echo_ticker("トヨタ", streamlit_secrets=GEMINI_SECRETS)
                st.session_state.gemini_test_msg = msg
            if st.session_state.get("gemini_test_msg") is not None:
                st.write(st.session_state.gemini_test_msg)

    # ----- 単一銘柄分析（常に表示） -----
    st.subheader(f"単一銘柄分析: {ticker}")
    _render_detail_chart(ticker, period)

    st.divider()

    # ----- 本命の割安株（バッチ一括スキャン） -----
    st.subheader("本命の割安株（バッチ一括スキャン）")
    st.caption(
        "銘柄リスト（CSV/東証）から適正株価の乖離率を一括算出し、"
        "ROE≥8%・75MA突破・出来高10万株以上の足切りを通した「本命」のみを乖離率上位で表示します。"
        "初回はキャッシュ取得のため数分かかることがあります。"
    )
    top_n_batch = st.sidebar.number_input("バッチで表示する上位銘柄数", min_value=1, max_value=50, value=10, key="batch_top_n")
    if st.button("本命の割安株を一括スキャン", type="primary", key="batch_value_screen_btn"):
        with st.spinner("銘柄リストを読み込み、キャッシュを参照／補完してスキャンしています…"):
            try:
                rows = run_batch(refill_missing=True, top_n=int(top_n_batch))
                if rows:
                    st.success(f"**{len(rows)} 銘柄**が条件を満たしました（乖離率上位）。")
                    for r in rows:
                        st.markdown(f"- {format_line(r)}")
                    df_batch = pd.DataFrame(rows)
                    df_batch = df_batch.rename(columns={
                        "ticker": "銘柄",
                        "current_price": "現在値",
                        "theoretical_price": "理論株価",
                        "deviation_pct": "乖離率(%)",
                        "roe_pct": "ROE(%)",
                    })
                    with st.expander("結果を表で表示", expanded=False):
                        st.dataframe(
                            df_batch[["銘柄", "現在値", "理論株価", "乖離率(%)", "ROE(%)"]].style.format({
                                "現在値": "¥{:,.0f}",
                                "理論株価": "¥{:,.0f}",
                                "乖離率(%)": "+{:.1f}%",
                                "ROE(%)": "{:.1f}%",
                            }),
                            hide_index=True,
                            use_container_width=True,
                        )
                else:
                    st.info("条件を満たす銘柄はありませんでした。キャッシュを更新するか、しばらく経ってから再実行してください。")
            except Exception as e:
                st.error(f"バッチスキャンエラー: {e}")

    st.divider()

    # ----- 本日の買いシグナル（16:00想定＝X投稿と同じ内容） -----
    st.subheader("本日の買いシグナル（16:00想定）")
    st.caption(
        "X 自動投稿（毎日16:00）と同じ条件で表示します。"
        " 大引け日で買いサインが出た銘柄のみ（乖離率・AI判定は使わない）。"
    )
    if "daily_buy_signals" not in st.session_state:
        st.session_state.daily_buy_signals = None
    if "daily_buy_signals_text" not in st.session_state:
        st.session_state.daily_buy_signals_text = None
    if "daily_buy_signals_watch" not in st.session_state:
        st.session_state.daily_buy_signals_watch = None
    if "daily_buy_signals_high_potential" not in st.session_state:
        st.session_state.daily_buy_signals_high_potential = None
    if "daily_buy_signals_backtest_format" not in st.session_state:
        st.session_state.daily_buy_signals_backtest_format = False
    if "daily_buy_signals_updated" not in st.session_state:
        st.session_state.daily_buy_signals_updated = None

    daily_json_url = os.environ.get("DAILY_SIGNALS_JSON_URL", "").strip()
    if not daily_json_url:
        try:
            daily_json_url = (st.secrets.get("DAILY_SIGNALS_JSON_URL") or "").strip()
        except Exception:
            pass

    # 初回ロード時に GitHub の JSON を自動ロード（ボタンを押さなくても最新結果を反映）
    if (
        daily_json_url
        and st.session_state.daily_buy_signals is None
        and st.session_state.daily_buy_signals_text is None
    ):
        try:
            with urllib.request.urlopen(daily_json_url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data.get("backtest_driven") and data.get("items"):
                items = data.get("items") or []
                n_tickers = data.get("unique_tickers", len({x.get("ticker") for x in items if x.get("ticker")}))
                n_signals = data.get("signal_count", len(items))
                summary = f"該当 {n_tickers} 銘柄（{n_signals} 件のシグナル）\n\n" + "\n".join(
                    x.get("formatted_line", "") for x in items
                )
                st.session_state.daily_buy_signals = items
                st.session_state.daily_buy_signals_updated = data.get("updated")
                st.session_state.daily_buy_signals_high_potential = []
                st.session_state.daily_buy_signals_watch = []
                st.session_state.daily_buy_signals_text = summary if items else "本日は該当銘柄はありませんでした。"
                st.session_state.daily_buy_signals_backtest_format = True
            else:
                active_list = data.get("active", data.get("all", []))
                high_potential_list = data.get("high_potential", [])
                watch_list = data.get("watch", [])
                tweet_text = data.get("tweet_text", "")
                st.session_state.daily_buy_signals = active_list if isinstance(active_list, list) else []
                st.session_state.daily_buy_signals_high_potential = high_potential_list if isinstance(high_potential_list, list) else []
                st.session_state.daily_buy_signals_text = tweet_text or "本日は買いシグナル点灯銘柄はありませんでした。"
                st.session_state.daily_buy_signals_watch = watch_list if isinstance(watch_list, list) else []
                st.session_state.daily_buy_signals_backtest_format = False
        except Exception:
            pass

    col_refresh, col_fetch, _ = st.columns([1, 1, 2])
    with col_refresh:
        if st.button("表示を更新", key="daily_signal_refresh"):
            with st.spinner("対象銘柄をスキャン中…（プレフィルター＋並列処理で10〜15分程度）"):
                try:
                    scan_result = scan_hybrid()
                    active_list = scan_result.get("active", [])
                    high_potential_list = scan_result.get("high_potential", [])
                    watch_list = scan_result.get("watch", [])
                    combined = active_list + high_potential_list
                    combined.sort(key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
                    picked = combined[:3]
                    if picked:
                        tweet_text = build_tweet(picked, watch_items=watch_list if watch_list else None)
                    else:
                        tweet_text = "本日は買いシグナル点灯銘柄はありませんでした。"
                    st.session_state.daily_buy_signals = active_list
                    st.session_state.daily_buy_signals_high_potential = high_potential_list
                    st.session_state.daily_buy_signals_text = tweet_text
                    st.session_state.daily_buy_signals_watch = watch_list
                except Exception as e:
                    st.session_state.daily_buy_signals = None
                    st.session_state.daily_buy_signals_high_potential = None
                    st.session_state.daily_buy_signals_text = None
                    st.session_state.daily_buy_signals_watch = None
                    st.error(f"スキャンエラー: {e}")
            st.rerun()
    with col_fetch:
        if daily_json_url and st.button("GitHub の結果を読み込み", key="daily_signal_fetch"):
            try:
                with urllib.request.urlopen(daily_json_url, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                # バックテスト駆動の新形式（backtest_driven + items）
                if data.get("backtest_driven") and data.get("items"):
                    items = data.get("items") or []
                    n_tickers = data.get("unique_tickers", len({x.get("ticker") for x in items if x.get("ticker")}))
                    n_signals = data.get("signal_count", len(items))
                    summary = f"該当 {n_tickers} 銘柄（{n_signals} 件のシグナル）\n\n" + "\n".join(
                        x.get("formatted_line", "") for x in items
                    )
                    st.session_state.daily_buy_signals = items
                    st.session_state.daily_buy_signals_updated = data.get("updated")
                    st.session_state.daily_buy_signals_high_potential = []
                    st.session_state.daily_buy_signals_watch = []
                    st.session_state.daily_buy_signals_text = summary if items else "本日は該当銘柄はありませんでした。"
                    st.session_state.daily_buy_signals_backtest_format = True
                else:
                    active_list = data.get("active", data.get("all", []))
                    high_potential_list = data.get("high_potential", [])
                    watch_list = data.get("watch", [])
                    tweet_text = data.get("tweet_text", "")
                    st.session_state.daily_buy_signals = active_list if isinstance(active_list, list) else []
                    st.session_state.daily_buy_signals_high_potential = high_potential_list if isinstance(high_potential_list, list) else []
                    st.session_state.daily_buy_signals_text = tweet_text or "本日は買いシグナル点灯銘柄はありませんでした。"
                    st.session_state.daily_buy_signals_watch = watch_list if isinstance(watch_list, list) else []
                    st.session_state.daily_buy_signals_backtest_format = False
                st.success("読み込みました。")
            except Exception as e:
                st.error(f"読み込みエラー: {e}")
            st.rerun()

    if not daily_json_url:
        st.caption("GitHub で自動更新された結果を表示するには、環境変数または Secrets で **DAILY_SIGNALS_JSON_URL** を設定してください（例: `https://raw.githubusercontent.com/ユーザ名/リポジトリ名/main/daily_buy_signals.json`）。")

    def _fmt_price(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        try:
            v = float(x)
            return f"¥{v:,.0f}" if v == v else "—"
        except (TypeError, ValueError):
            return "—"

    # ----- 本命（Active Signal）／バックテスト該当銘柄 -----
    st.subheader("本命（Active Signal）／バックテスト該当銘柄")
    backtest_format = st.session_state.get("daily_buy_signals_backtest_format", False)
    if backtest_format:
        st.caption("3営業日バックテスト統計ベース。サンプル3回以上・勝率60%以上・平均リターン+1.5%以上の銘柄のみ。")
    else:
        st.caption("全条件合致（確信度高）。Type-A トレンド追随 または Type-B リバウンドで 3/3 充足。X 投稿は本命・注目から確信度上位最大3銘柄。")
    if st.session_state.daily_buy_signals_text is not None:
        st.text_area(
            "結果サマリ",
            value=st.session_state.daily_buy_signals_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )
        if st.session_state.daily_buy_signals:
            full_list = list(st.session_state.daily_buy_signals)
            if backtest_format and full_list and full_list[0].get("pattern_name") is not None:
                # バックテスト駆動の items 形式
                n = len(full_list)
                st.caption(f"**全 {n} 件のシグナル**　※機械的スクリーニング結果。投資判断は自己責任で。")
                df_16 = pd.DataFrame(full_list)
                rename_bt = {
                    "ticker": "銘柄コード", "name": "銘柄名", "pattern_name": "パターン名",
                    "win_rate": "勝率", "sample_count": "サンプル数", "avg_return_pct": "平均リターン%",
                    "entry": "エントリー想定", "tp": "利確(TP)", "sl": "損切り(SL)",
                }
                df_16 = df_16.rename(columns={k: v for k, v in rename_bt.items() if k in df_16.columns})
                cols_bt = ["銘柄コード", "銘柄名", "パターン名", "勝率", "サンプル数", "平均リターン%", "エントリー想定", "利確(TP)", "損切り(SL)"]
                display_cols_bt = [c for c in cols_bt if c in df_16.columns]
                if "勝率" in df_16.columns:
                    df_16["勝率"] = df_16["勝率"].apply(lambda x: f"{float(x)*100:.0f}%" if x is not None and x == x else "—")
                if "平均リターン%" in df_16.columns:
                    df_16["平均リターン%"] = df_16["平均リターン%"].apply(lambda x: f"+{float(x):.2f}%" if x is not None and x == x else "—")
                for col in ("エントリー想定", "利確(TP)", "損切り(SL)"):
                    if col in df_16.columns:
                        df_16[col] = df_16[col].apply(_fmt_price)
                st.dataframe(df_16[display_cols_bt], hide_index=True, use_container_width=True)
            else:
                # 旧形式（本命・注目）
                high_for_top = st.session_state.get("daily_buy_signals_high_potential") or []
                merged_all = full_list + high_for_top
                merged_all.sort(key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
                top3_tickers = {x["ticker"] for x in merged_all[:3]}
                full_list.sort(key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
                n = len(full_list)
                provisional_note = any(x.get("provisional") for x in full_list if isinstance(x, dict))
                cap = f"**全 {n} 銘柄**　※機械的スクリーニング結果。投資判断は自己責任で。"
                if provisional_note:
                    cap += "　※15:15暫定（大引け前の暫定値・TP/SLは暫定終値ベース）"
                st.caption(cap)
                if n == 3:
                    st.info("3銘柄だけの場合は、GitHub の JSON が古い可能性があります。Actions でワークフローを1回実行すると「all」が入り全銘柄表示になります。「表示を更新」でも全銘柄取得できます。")
                df_16 = pd.DataFrame(full_list)
                rename_map = {
                    "ticker": "銘柄コード", "name": "銘柄名", "buy_signals": "検出シグナル", "signal_count": "シグナル数",
                    "entry": "エントリー想定", "tp": "利確(TP)", "sl": "損切り(SL)", "rationale": "根拠",
                    "conviction_score": "確信度スコア",
                }
                df_16 = df_16.rename(columns={k: v for k, v in rename_map.items() if k in df_16.columns})
                df_16["★高確信度"] = df_16["銘柄コード"].apply(lambda t: "★高確信度" if t in top3_tickers else "")
                base_cols = ["銘柄コード", "銘柄名", "★高確信度", "確信度スコア", "検出シグナル", "シグナル数"]
                opt_cols = ["エントリー想定", "利確(TP)", "損切り(SL)", "根拠"]
                display_cols = [c for c in base_cols + opt_cols if c in df_16.columns]
                if "エントリー想定" in df_16.columns:
                    df_16["エントリー想定"] = df_16["エントリー想定"].apply(_fmt_price)
                if "利確(TP)" in df_16.columns:
                    df_16["利確(TP)"] = df_16["利確(TP)"].apply(_fmt_price)
                if "損切り(SL)" in df_16.columns:
                    df_16["損切り(SL)"] = df_16["損切り(SL)"].apply(_fmt_price)
                st.dataframe(df_16[display_cols], hide_index=True, use_container_width=True)
        else:
            st.info("本日は該当銘柄はありませんでした。")
    else:
        st.caption("「GitHub の結果を読み込み」で daily_buy_signals.json を表示します。「表示を更新」で従来スキャンも実行できます。")

    # ----- シグナル検証（引け購入でどれだけ儲かったか） -----
    st.subheader("シグナル検証（引け購入でどれだけ儲かったか）")
    st.caption("GitHub で出力した銘柄を「シグナル日の引けで購入 → 指定営業日後に利確」した場合の実際のリターン（%）を検証します。")
    holding_days_verify = st.sidebar.number_input("検証時の保有営業日数", min_value=1, max_value=20, value=3, key="verify_holding_days")
    if st.button("引け購入リターンを検証する", key="btn_verify_returns"):
        full_list = st.session_state.get("daily_buy_signals") or []
        if not full_list:
            st.warning("先に「GitHub の結果を読み込み」または「表示を更新」でシグナルを読み込んでください。")
        else:
            payload = {
                "items": full_list,
                "updated": st.session_state.get("daily_buy_signals_updated"),
            }
            with st.spinner("各銘柄の価格を取得してリターンを計算しています…"):
                try:
                    rows = verify_returns(payload, holding_days=int(holding_days_verify))
                    stats = summary_stats(rows)
                except Exception as e:
                    st.error(f"検証エラー: {e}")
                    rows = []
                    stats = {"count": 0, "win_rate": None, "avg_return_pct": None, "total_return_pct": None}
            if rows:
                st.success(f"**有効 {stats['count']} 件** | 勝率: {stats['win_rate'] or '—'}% | 平均リターン: {stats['avg_return_pct'] or '—'}%")
                df_v = pd.DataFrame(rows)
                df_v = df_v.rename(columns={
                    "ticker": "銘柄", "pattern_name": "パターン", "entry": "エントリー(¥)",
                    "exit_price": "利確(¥)", "return_pct": "リターン(%)", "signal_date": "シグナル日", "exit_date": "利確日",
                })
                display_v = ["銘柄", "パターン", "エントリー(¥)", "利確(¥)", "リターン(%)", "シグナル日", "利確日"]
                display_v = [c for c in display_v if c in df_v.columns]
                st.dataframe(
                    df_v[display_v].style.format({
                        "エントリー(¥)": "¥{:,.0f}",
                        "利確(¥)": "¥{:,.0f}",
                        "リターン(%)": "{:+.2f}%",
                    }, na_rep="—"),
                    hide_index=True,
                    use_container_width=True,
                )
                for r in rows:
                    if r.get("error"):
                        st.caption(f"【{r.get('ticker')}】{r['error']}")
            else:
                st.info("検証対象のシグナルがありません（バックテスト形式の items に entry があるものを使用します）。")

    # ----- 過去3日分の「100株ずつ買っていた場合」のリターン -----
    root_dir = os.path.dirname(os.path.abspath(__file__))
    history_files = _list_signal_history_files(root_dir)
    if history_files:
        st.caption("直近のシグナルについて、各銘柄を100株ずつ買っていた場合のポートフォリオ損益を自動集計します。")
        horizons = [1, 2, 3]
        labels = {1: "1日後リターン（1日前の銘柄）", 2: "2日後リターン（2日前の銘柄）", 3: "3日後リターン（3日前の銘柄）"}
        for idx, h in enumerate(horizons, start=1):
            if len(history_files) <= idx:
                continue
            dt, path_hist = history_files[idx]
            try:
                with open(path_hist, "r", encoding="utf-8") as f_hist:
                    data_hist = json.load(f_hist)
            except Exception:
                continue
            rows_h = verify_returns(data_hist, holding_days=h)
            stats_h = summary_stats(rows_h)
            profit_yen, profit_pct = _compute_portfolio_pnl_100_shares(rows_h)
            st.markdown(
                f"- **{labels[h]}**: シグナル日 {dt.date().isoformat()} / "
                f"有効 {stats_h['count']} 件 / 勝率 {stats_h['win_rate'] or '—'}% / "
                f"平均リターン {stats_h['avg_return_pct'] or '—'}% / "
                f"100株ずつの合計損益 {profit_yen:+,.0f}円 (ポートフォリオ {profit_pct:+.2f}%)"
            )

    # ----- 注目（High Potential） -----
    st.subheader("注目（High Potential）")
    st.caption("条件の8割以上を充足（確信度中）。Type-A/Type-B で 2/3 以上。")
    high_potential_list = st.session_state.get("daily_buy_signals_high_potential") or []
    if high_potential_list:
        active_for_top = st.session_state.get("daily_buy_signals") or []
        merged_all_h = active_for_top + high_potential_list
        merged_all_h.sort(key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
        top3_tickers_h = {x["ticker"] for x in merged_all_h[:3]}
        high_potential_list = sorted(high_potential_list, key=lambda x: float(x.get("conviction_score", 0)), reverse=True)
        df_h = pd.DataFrame(high_potential_list)
        rename_h = {
            "ticker": "銘柄コード", "name": "銘柄名", "buy_signals": "検出シグナル", "signal_count": "シグナル数",
            "entry": "エントリー想定", "tp": "利確(TP)", "sl": "損切り(SL)", "rationale": "根拠",
            "reason_short": "不足理由", "conviction_score": "確信度スコア",
        }
        df_h = df_h.rename(columns={k: v for k, v in rename_h.items() if k in df_h.columns})
        df_h["★高確信度"] = df_h["銘柄コード"].apply(lambda t: "★高確信度" if t in top3_tickers_h else "")
        cols_h = ["銘柄コード", "銘柄名", "★高確信度", "確信度スコア", "不足理由", "検出シグナル", "シグナル数", "エントリー想定", "利確(TP)", "損切り(SL)", "根拠"]
        display_cols_h = [c for c in cols_h if c in df_h.columns]
        for col in ("エントリー想定", "利確(TP)", "損切り(SL)"):
            if col in df_h.columns:
                df_h[col] = df_h[col].apply(_fmt_price)
        st.dataframe(df_h[display_cols_h], hide_index=True, use_container_width=True)
    else:
        st.caption("注目銘柄はありません。")

    # ----- 監視（Watchlist） -----
    st.subheader("監視（Watchlist）")
    st.caption("24種の買いパターンを主軸にしたニアミス。条件A（パターン点灯+出来高1.0倍以上+MA5%以内）または条件B（パターン未点灯+出来高2倍+MA3%以内）を満たす銘柄からスコア上位5件。各銘柄に「何が足りないか」を表示。")
    watch_list = st.session_state.get("daily_buy_signals_watch") or []
    if watch_list:
        df_w = pd.DataFrame(watch_list)
        rename_w = {
            "ticker": "銘柄コード", "name": "銘柄名",
            "entry": "エントリー想定", "tp": "利確(TP)", "sl": "損切り(SL)",
            "reason_short": "不足理由", "watchlist_score": "ウォッチスコア",
        }
        df_w = df_w.rename(columns={k: v for k, v in rename_w.items() if k in df_w.columns})
        cols_w = ["銘柄コード", "銘柄名", "不足理由", "ウォッチスコア", "エントリー想定", "利確(TP)", "損切り(SL)"]
        display_cols_w = [c for c in cols_w if c in df_w.columns]
        for col in ("エントリー想定", "利確(TP)", "損切り(SL)"):
            if col in df_w.columns:
                df_w[col] = df_w[col].apply(_fmt_price)
        st.dataframe(df_w[display_cols_w], hide_index=True, use_container_width=True)
    else:
        st.caption("監視銘柄はありません。")

    st.divider()


if __name__ == "__main__":
    main()
