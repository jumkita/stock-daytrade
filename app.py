# -*- coding: utf-8 -*-
"""
Streamlit UI: SOTP理論株価 + 24種買い/26種売りパターン + 市場スクリーニング
"""
import json
import os
import threading
import time
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
    sotp_full,
    fetch_ohlcv,
    detect_all_patterns,
    get_downtrend_mask,
    calc_stop_loss_line,
    get_sotp_suggested_multiple,
    gemini_echo_ticker,
)
from screener import TARGET_TICKERS, run_screen
from auto_post import scan_hybrid, scan_buy_signal_only, build_tweet


def _render_detail_chart(ticker: str, ebitda_mult: float, period: str) -> None:
    """
    単一銘柄の詳細（SOTPカード + ローソク足 + パターン）を描画。
    単一銘柄モードとスクリーナー「詳細表示」の両方で利用。
    """
    try:
        sotp = sotp_full(ticker, ebitda_multiple=ebitda_mult)
    except Exception as e:
        st.error(f"SOTP 取得エラー: {e}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        theo = sotp.get("theoretical_price")
        st.metric("理論株価 (SOTP)", f"¥{theo:,.0f}" if theo is not None else "—")
    with col2:
        cur = sotp.get("current_price")
        st.metric("現在値", f"¥{cur:,.0f}" if cur is not None else "—")
    with col3:
        dev = sotp.get("deviation_pct")
        st.metric("乖離率（割安度）", f"{dev:+.1f}%" if dev is not None else "—")

    logic_name = sotp.get("valuation_logic")
    if logic_name:
        st.caption(f"**Evaluation Model:** {logic_name}")
    msg = sotp.get("message")
    if msg:
        st.caption(f"計算根拠: {msg}")

    try:
        df = fetch_ohlcv(ticker, period=period)
    except Exception as e:
        st.error(f"株価データ取得エラー: {e}")
        return
    if df is None or df.empty:
        st.warning("株価データを取得できませんでした。")
        return

    try:
        patterns = detect_all_patterns(df)
    except Exception:
        patterns = []
    downtrend_mask = get_downtrend_mask(df, window=25)
    latest_close = float(df["Close"].iloc[-1]) if len(df) > 0 else None
    stop_loss_price = round(latest_close * 0.95) if latest_close and latest_close > 0 else None
    if stop_loss_price is not None:
        st.caption(f"損切りライン（現在値×0.95）: ¥{stop_loss_price:,.0f}")

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
    close_str = f"¥{close_price:,.0f}" if close_price is not None and pd.notna(close_price) else "—"

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
    if stop_loss_price is not None:
        fig.add_hline(y=stop_loss_price, line_dash="dash", line_color="red", annotation_text=f"損切り ¥{stop_loss_price:,.0f}")
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
    st.set_page_config(page_title="日本株 SOTP・パターン分析", layout="wide")
    st.title("日本株 SOTP 理論株価 × 勝ちパターン分析")

    if "screen_results" not in st.session_state:
        st.session_state.screen_results = None
    if "screen_debug" not in st.session_state:
        st.session_state.screen_debug = None
    if "scan_shared" not in st.session_state:
        st.session_state.scan_shared = None
    if "scan_thread" not in st.session_state:
        st.session_state.scan_thread = None

    with st.sidebar:
        st.header("設定")
        if not api_ready:
            st.warning("⚠️ APIキーが設定されていません。AI分析機能は利用できません。")
        period = st.selectbox("分析期間", ["3mo", "6mo", "1y", "2y"], index=0)
        ticker = st.text_input("銘柄コード", value=st.session_state.get("ticker_input", "8473.T"), help="例: 7203.T, 8473.T", key="ticker_input")
        current_ticker = ticker

        valuation_logic = None
        multiplier_disabled = False
        if current_ticker:
            try:
                sug = get_sotp_suggested_multiple(current_ticker)
                default_mult = sug["suggested_multiple"] if sug.get("suggested_multiple") is not None else 8.0
                sector_label = sug["sector_label"]
                multiplier_disabled = sug.get("multiplier_disabled", False)
                valuation_logic = sug.get("valuation_logic")
            except Exception:
                default_mult = 8.0
                sector_label = "—"
        else:
            default_mult = 8.0
            sector_label = "—"

        if current_ticker is not None and st.session_state.get("sotp_ticker") != current_ticker:
            st.session_state["sotp_ticker"] = current_ticker
            st.session_state["ebitda_mult"] = default_mult
        if multiplier_disabled:
            st.caption(f"{valuation_logic or 'ROE-linked PBR'} のため倍率スライダーは無効")
            ebitda_mult = 8.0
        else:
            ebitda_mult = st.slider(
                "EBITDA 倍率",
                min_value=1.0,
                max_value=30.0,
                value=float(st.session_state.get("ebitda_mult", default_mult)),
                step=0.5,
                key="ebitda_slider",
            )
            st.session_state["ebitda_mult"] = ebitda_mult
            st.caption(f"(自動算出: {default_mult}倍 / 業種: {sector_label})")

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
    _render_detail_chart(ticker, ebitda_mult, period)

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

    daily_json_url = os.environ.get("DAILY_SIGNALS_JSON_URL", "").strip()
    if not daily_json_url:
        try:
            daily_json_url = (st.secrets.get("DAILY_SIGNALS_JSON_URL") or "").strip()
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
                    watch_names = [w["name"] for w in watch_list] if watch_list else None
                    if picked:
                        tweet_text = build_tweet(picked, watch_names)
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
                active_list = data.get("active", data.get("all", []))
                high_potential_list = data.get("high_potential", [])
                watch_list = data.get("watch", [])
                tweet_text = data.get("tweet_text", "")
                st.session_state.daily_buy_signals = active_list if isinstance(active_list, list) else []
                st.session_state.daily_buy_signals_high_potential = high_potential_list if isinstance(high_potential_list, list) else []
                st.session_state.daily_buy_signals_text = tweet_text or "本日は買いシグナル点灯銘柄はありませんでした。"
                st.session_state.daily_buy_signals_watch = watch_list if isinstance(watch_list, list) else []
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

    # ----- 本命（Active Signal） -----
    st.subheader("本命（Active Signal）")
    st.caption("全条件合致（確信度高）。Type-A トレンド追随 または Type-B リバウンドで 3/3 充足。X 投稿は本命・注目から確信度上位最大3銘柄。")
    if st.session_state.daily_buy_signals_text is not None:
        st.text_area(
            "X 投稿と同じフォーマット",
            value=st.session_state.daily_buy_signals_text,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )
        if st.session_state.daily_buy_signals:
            full_list = list(st.session_state.daily_buy_signals)
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
            st.info("本日は本命はありませんでした。")
    else:
        st.caption("「表示を更新」を押すと、本命・注目・監視を取得します。")

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
    st.caption("24種の買いパターンを主軸にしたニアミス。条件A（パターン点灯+出来高1.2倍+MA5%以内）または条件B（パターン未点灯+出来高2倍+MA3%以内）を満たす銘柄からスコア上位5件。各銘柄に「何が足りないか」を表示。")
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

    # ----- 市場スキャン -----
    st.subheader("市場スキャン（厳選銘柄）")
    st.caption(
        f"対象: {len(TARGET_TICKERS)} 銘柄（CSV/東証リストまたは日経225）— "
        "直近3日以内に「勝率・収益性の高いサイン」が1つ以上出た銘柄を抽出（バックテスト: 勝率50%以上・PF≥1.0・約定5回以上）。"
        " 乖離率20%以上でさらに絞り込み。"
    )

    # スキャン中は進捗と中断ボタンを表示（スレッドで実行中のため）
    scan_thread = st.session_state.get("scan_thread")
    scan_shared = st.session_state.get("scan_shared")
    scan_running = scan_thread is not None and scan_thread.is_alive()

    if scan_running and scan_shared is not None:
        cur, total, ticker = scan_shared.get("progress", (0, 1, ""))
        total = max(1, total)
        progress_bar = st.progress(cur / total, text=f"現在 {cur}/{total} 銘柄をスキャン中...")
        st.caption(f"処理中: {ticker}")
        audit_progress = scan_shared.get("audit_progress")
        if audit_progress is not None:
            adone, atotal, amsg = audit_progress
            atotal = max(1, atotal)
            st.progress(adone / atotal, text=amsg)
        partial = scan_shared.get("partial_audit_results")
        if partial:
            placeholder = st.empty()
            with placeholder.container():
                st.caption("監査結果（3銘柄ごとに更新）")
                df_part = pd.DataFrame(partial)
                df_part = df_part.rename(columns={
                    "ticker": "銘柄コード", "name": "銘柄名", "current_price": "現在値",
                    "theoretical_price": "理論株価", "deviation_pct": "乖離率(%)",
                    "buy_signals": "直近の買いサイン", "ai_rank": "AI判定",
                    "strategist_eye": "ストラテジストの眼", "verdict": "Verdict",
                })
                # 強制数値化してから損切り目安を計算・表示用に整形
                if "現在値" in df_part.columns:
                    raw = pd.to_numeric(
                        df_part["現在値"].astype(str).str.replace("¥", "", regex=False).str.replace(",", "", regex=False),
                        errors="coerce",
                    )
                    df_part["損切り目安"] = raw * 0.95
                    df_part["現在値"] = raw.apply(lambda x: f"¥{int(x):,}" if x is not None and pd.notna(x) and x == x and x > 0 else "—")
                    df_part["損切り目安"] = df_part["損切り目安"].apply(lambda x: f"¥{int(x):,}" if x is not None and pd.notna(x) and x == x and x > 0 else "—")
                if "理論株価" in df_part.columns:
                    df_part["理論株価"] = df_part["理論株価"].apply(lambda x: f"¥{int(x):,}" if x is not None and pd.notna(x) else "—")
                if "乖離率(%)" in df_part.columns:
                    df_part["乖離率(%)"] = df_part["乖離率(%)"].apply(lambda x: f"{x:+.1f}%" if x is not None else "—")
                cols = [c for c in ["銘柄コード", "銘柄名", "現在値", "損切り目安", "理論株価", "乖離率(%)", "AI判定", "ストラテジストの眼", "Verdict", "直近の買いサイン"] if c in df_part.columns]
                st.dataframe(df_part[cols], width="stretch")
        if st.button("中断", key="scan_stop_btn"):
            scan_shared["stop"] = True
            st.caption("中断リクエストを送りました。現在の銘柄処理後に停止します。")
            st.rerun()
        time.sleep(0.5)
        st.rerun()

    # スキャン終了直後: 結果を反映してスレッド・共有状態をクリア
    if not scan_running and scan_shared is not None:
        data = scan_shared.get("result")
        if data is not None:
            st.session_state.screen_results = data.get("results", [])
            st.session_state.screen_debug = data.get("debug", [])
        if scan_shared.get("stopped"):
            st.info("スキャンを中断しました。")
        if scan_shared.get("error"):
            st.error(f"スキャンエラー: {scan_shared['error']}")
        st.session_state.scan_shared = None
        st.session_state.scan_thread = None
        st.rerun()

    col_scan, col_stop = st.columns(2)
    with col_scan:
        if st.button("厳選銘柄をスキャン", type="primary", key="scan_start_btn"):
            shared = {
                "stop": False,
                "progress": (0, len(TARGET_TICKERS), ""),
                "audit_progress": None,
                "partial_audit_results": None,
                "result": None,
                "stopped": False,
            }
            st.session_state.scan_shared = shared

            gemini_secrets = GEMINI_SECRETS

            def worker(secrets_for_audit):
                def on_progress(current: int, total: int, t: str):
                    shared["progress"] = (current, total, t)

                def on_audit_progress(done: int, total: int, msg: str, results_so_far=None):
                    shared["audit_progress"] = (done, total, msg)
                    if results_so_far is not None:
                        shared["partial_audit_results"] = results_so_far

                try:
                    data = run_screen(
                        ebitda_multiple=ebitda_mult,
                        min_deviation_pct=20.0,
                        recent_days=3,
                        progress_callback=on_progress,
                        stop_check=lambda: shared.get("stop", False),
                        enable_gemini_audit=api_ready,
                        streamlit_secrets=secrets_for_audit,
                        audit_progress_callback=on_audit_progress,
                        holding_days=5,
                        stop_loss_pct=0.05,
                        min_win_rate=0.5,
                    )
                    shared["result"] = data
                    shared["stopped"] = shared.get("stop", False)
                except Exception as e:
                    shared["result"] = {"results": [], "debug": []}
                    shared["error"] = str(e)

            th = threading.Thread(target=worker, args=(gemini_secrets,))
            st.session_state.scan_thread = th
            th.start()
            st.rerun()

    if st.session_state.screen_results is not None:
        results = st.session_state.screen_results
        if not results:
            st.info("条件を満たす銘柄はありませんでした。")
        else:
            st.success(f"**{len(results)} 銘柄**が条件を満たしました。")
            for r in results:
                r.setdefault("ai_rank", "—")
                r.setdefault("strategist_eye", "")
                r.setdefault("verdict", "OK")
            df_display = pd.DataFrame(results)
            df_display = df_display.rename(columns={
                "ticker": "銘柄コード",
                "name": "銘柄名",
                "current_price": "現在値",
                "theoretical_price": "理論株価",
                "deviation_pct": "乖離率(%)",
                "buy_signals": "直近の買いサイン",
                "ai_rank": "AI判定",
                "strategist_eye": "ストラテジストの眼",
                "verdict": "Verdict",
            })
            # 強制数値化（文字列 '¥3,489' 混入で損切りが 0 になるのを防ぐ）
            df_display["現在値"] = pd.to_numeric(
                df_display["現在値"].astype(str).str.replace("¥", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
            df_display["理論株価"] = pd.to_numeric(
                df_display["理論株価"].astype(str).str.replace("¥", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
            # 損切り目安を数値で計算してから表示用に整形
            raw_price = df_display["現在値"]
            df_display["損切り目安"] = raw_price * 0.95
            df_display["現在値"] = raw_price.apply(lambda x: f"¥{int(x):,}" if x is not None and pd.notna(x) and x == x and x > 0 else "—")
            df_display["損切り目安"] = df_display["損切り目安"].apply(lambda x: f"¥{int(x):,}" if x is not None and pd.notna(x) and x == x and x > 0 else "—")
            df_display["理論株価"] = df_display["理論株価"].apply(lambda x: f"¥{int(x):,}" if x is not None and pd.notna(x) and x == x else "—")
            df_display["乖離率(%)"] = df_display["乖離率(%)"].apply(lambda x: f"{x:+.1f}%" if x is not None and pd.notna(x) else "—")
            # Rank D の場合は Verdict を強制 AVOID に（理論株価が高くても注意）
            if "Verdict" in df_display.columns:
                df_display["Verdict"] = df_display.apply(
                    lambda r: "AVOID" if str(r.get("AI判定", "")).strip() == "D" else r.get("Verdict", "OK"),
                    axis=1,
                )
            # 表示順
            col_order = ["銘柄コード", "銘柄名", "現在値", "損切り目安", "理論株価", "乖離率(%)", "AI判定", "ストラテジストの眼", "Verdict", "直近の買いサイン"]
            df_display = df_display[[c for c in col_order if c in df_display.columns]]
            # 行ハイライト: Rank A = 薄緑, Rank D = 薄赤
            def _row_style(row):
                rank = str(row.get("AI判定", "")).strip()
                if rank == "A":
                    return ["background-color: rgba(200,255,200,0.5)"] * len(row)
                if rank == "D":
                    return ["background-color: rgba(255,200,200,0.5)"] * len(row)
                return [""] * len(row)
            try:
                st.dataframe(
                    df_display.style.apply(_row_style, axis=1),
                    width="stretch",
                    hide_index=True,
                )
            except Exception:
                st.dataframe(df_display, width="stretch", hide_index=True)
            st.caption("🟢 Rank A: 割安に正当な理由あり　🔴 Rank D: 万年割安の可能性（Verdict=AVOID）")

            st.divider()
            st.subheader("詳細分析")
            options = [f"{r['ticker']} - {r['name']}" for r in results]
            selected = st.selectbox(
                "詳細表示する銘柄を選択（上段の単一銘柄分析に反映）",
                options=options,
                key="screener_detail_select",
            )
            if selected:
                ticker_for_detail = selected.split(" - ")[0].strip()
                if ticker_for_detail != st.session_state.get("ticker_input"):
                    st.session_state["ticker_input"] = ticker_for_detail
                    st.rerun()

        # デバッグ用: スキャンした全銘柄のリスト（理論株価 None/0 の可視化）
        debug_list = getattr(st.session_state, "screen_debug", None)
        if debug_list:
            st.divider()
            st.subheader("デバッグ用: 全銘柄スキャン結果")
            st.caption("条件（乖離率>20%）に関係なく、スキャンした全銘柄の取得結果です。理論株価が None/0 の原因特定に利用してください。")
            df_debug = pd.DataFrame(debug_list)
            df_debug = df_debug.rename(columns={
                "ticker": "Ticker",
                "price": "Price",
                "model_type": "Model Type",
                "theoretical_price": "Theoretical Price",
                "upside_pct": "Upside (%)",
                "status": "Status",
            })
            def _fmt_price(x):
                if x is None or (isinstance(x, float) and x != x):
                    return "—"
                return f"¥{x:,.0f}"
            def _fmt_theo(x):
                if x is None:
                    return "None"
                if isinstance(x, (int, float)) and x == x:
                    return f"¥{x:,.0f}"
                return str(x)
            df_debug["Price"] = df_debug["Price"].apply(_fmt_price)
            df_debug["Theoretical Price"] = df_debug["Theoretical Price"].apply(_fmt_theo)
            df_debug["Upside (%)"] = df_debug["Upside (%)"].apply(lambda x: f"{x:+.1f}%" if x is not None else "—")
            st.dataframe(df_debug, width="stretch", hide_index=True)
    else:
        st.info("「厳選銘柄をスキャン」ボタンで一括スキャンを実行してください。")


if __name__ == "__main__":
    main()
