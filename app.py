# -*- coding: utf-8 -*-
"""
Streamlit UI: 適正株価・バリュートラップ検知 + 日次買いシグナルダッシュボード
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from auto_post import build_tweet, scan_hybrid
from backtest_engine import BacktestEngine, plot_backtest_results
from batch_value_screen import format_line, run_batch
from logic import (
    check_value_trap,
    detect_all_patterns,
    fetch_ohlcv,
    gemini_echo_ticker,
    get_downtrend_mask,
    get_fair_value,
)
from ui_helpers import (
    breakdown_from_item,
    display_columns,
    fetch_signals_json,
    filter_signals,
    fmt_price,
    format_display_dataframe,
    history_json_url,
    items_to_dataframe,
    parse_signals_payload,
    resolve_daily_json_url,
    sort_signals,
    with_cache_buster,
)
from verify_signal_returns import summary_stats, verify_returns

# ----- Gemini API Key -----
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


def _list_signal_history_files(root_dir: str) -> list[tuple[pd.Timestamp, str]]:
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
    total_profit = 0.0
    total_invested = 0.0
    for r in rows:
        entry = r.get("entry")
        exit_price = r.get("exit_price")
        if entry is None or exit_price is None:
            continue
        try:
            e, x = float(entry), float(exit_price)
        except (TypeError, ValueError):
            continue
        if e <= 0:
            continue
        total_profit += (x - e) * 100.0
        total_invested += e * 100.0
    if total_invested <= 0:
        return 0.0, 0.0
    return total_profit, total_profit / total_invested * 100.0


def _init_session_state() -> None:
    defaults = {
        "daily_buy_signals": None,
        "daily_buy_signals_sell": None,
        "daily_buy_signals_updated": None,
        "daily_buy_signals_backtest_format": False,
        "daily_buy_signals_quadrant_stats": None,
        "daily_buy_signals_meta": {},
        "signals_source_label": "未読込",
        "compare_signals": None,
        "compare_label": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_daily_json_url() -> str:
    url = resolve_daily_json_url()
    if not url:
        try:
            url = (st.secrets.get("DAILY_SIGNALS_JSON_URL") or "").strip()
        except Exception:
            url = ""
    return url


def _apply_payload_to_session(payload: dict[str, Any], source: str) -> None:
    st.session_state.daily_buy_signals = payload["items"]
    st.session_state.daily_buy_signals_sell = payload["items_sell"]
    st.session_state.daily_buy_signals_updated = payload.get("updated")
    st.session_state.daily_buy_signals_backtest_format = payload["backtest_format"]
    st.session_state.daily_buy_signals_quadrant_stats = payload.get("quadrant_stats")
    st.session_state.daily_buy_signals_meta = {
        "unique_tickers": payload.get("unique_tickers"),
        "signal_count": payload.get("signal_count"),
    }
    st.session_state.signals_source_label = source


def _load_signals_from_url(url: str, source: str = "GitHub") -> None:
    data = fetch_signals_json(url)
    payload = parse_signals_payload(data)
    _apply_payload_to_session(payload, source)


def _load_signals_from_file(path: str, label: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    payload = parse_signals_payload(data)
    _apply_payload_to_session(payload, label)


def _quadrant_breakdown_figure(bd: dict[str, float], title: str) -> go.Figure:
    labels = ["セクター", "需給", "テクニカル", "ファンダ"]
    keys = ["sector", "volume", "technical", "fundamental"]
    values = [bd.get(k, 0) for k in keys]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.0f}" for v in values], textposition="outside"))
    fig.update_layout(title=title, yaxis_title="点数", height=220, margin=dict(t=40, b=20, l=20, r=20), yaxis_range=[0, max(values + [10]) * 1.2])
    return fig


def _render_signal_cards(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    top = items[:3]
    cols = st.columns(len(top))
    for col, item in zip(cols, top):
        code = str(item.get("ticker", "")).replace(".T", "")
        score = item.get("quadrant_score")
        score_s = f"{float(score):.0f}点" if score is not None else "—"
        with col:
            st.metric(
                label=f"{code} {item.get('pattern_name', '')}",
                value=score_s,
                delta=f"勝率 {float(item.get('win_rate', 0)) * 100:.0f}%" if item.get("win_rate") else None,
            )


def _render_buy_signals_table(items: list[dict[str, Any]], *, mobile: bool) -> None:
    df = items_to_dataframe(items, mobile=mobile)
    if df.empty:
        st.info("表示対象の買いシグナルがありません。")
        return
    raw_items = df.pop("_raw").tolist() if "_raw" in df.columns else items
    show = format_display_dataframe(df)
    cols = [c for c in display_columns(mobile=mobile) if c in show.columns]
    st.dataframe(show[cols], hide_index=True, use_container_width=True)
    with st.expander("銘柄詳細（4象限内訳・TP/SL）", expanded=False):
        for item in raw_items:
            code = str(item.get("ticker", "")).replace(".T", "")
            title = f"{code} {item.get('name', '')} — {item.get('pattern_name', '')}"
            bd = breakdown_from_item(item)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(
                    f"セクター: {item.get('sector_label', '—')} | "
                    f"出来高: {item.get('vol_ratio', '—')}倍 | ROE: {item.get('roe_pct', '—')}%"
                )
                st.write(
                    f"エントリー {fmt_price(item.get('entry'))} / "
                    f"TP {fmt_price(item.get('tp'))} / SL {fmt_price(item.get('sl'))}"
                )
            with c2:
                if sum(bd.values()) > 0:
                    st.plotly_chart(
                        _quadrant_breakdown_figure(bd, f"4象限 {item.get('quadrant_score', '—')}点"),
                        use_container_width=True,
                    )


def _render_sell_table(sell_list: list[dict[str, Any]]) -> None:
    if not sell_list:
        st.caption("本日の売りシグナルはありません。")
        return
    df = pd.DataFrame(sell_list)
    df = df.rename(columns={"ticker": "銘柄コード", "name": "銘柄名", "pattern_name": "パターン名", "entry": "現在値(¥)"})
    if "現在値(¥)" in df.columns:
        df["現在値(¥)"] = df["現在値(¥)"].apply(fmt_price)
    st.dataframe(df[["銘柄コード", "銘柄名", "パターン名", "現在値(¥)"]], hide_index=True, use_container_width=True)


def _render_app_header() -> None:
    items = st.session_state.get("daily_buy_signals") or []
    updated = st.session_state.get("daily_buy_signals_updated") or "—"
    meta = st.session_state.get("daily_buy_signals_meta") or {}
    n_sig = meta.get("signal_count") or len(items)
    n_ticker = meta.get("unique_tickers") or len({x.get("ticker") for x in items if x.get("ticker")})
    top_score = max((float(x.get("quadrant_score") or 0) for x in items), default=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最終更新", str(updated)[:19] if updated != "—" else "—")
    c2.metric("シグナル数", f"{n_sig}件")
    c3.metric("銘柄数", f"{n_ticker}銘柄")
    c4.metric("最高4象限", f"{top_score:.0f}点" if top_score else "—")
    st.caption(f"データソース: **{st.session_state.get('signals_source_label', '—')}**")


def _render_detail_chart(
    ticker: str,
    period: str,
    *,
    enable_backtest: bool,
    holding_days: int,
    stop_loss_pct: float,
    min_win_rate: int,
) -> None:
    try:
        df = fetch_ohlcv(ticker, period=period)
    except Exception as e:
        st.error(f"株価データ取得エラー: {e}")
        return
    if df is None or df.empty:
        st.warning("株価データを取得できませんでした。")
        return

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
                st.metric("乖離率", f"{dev:+.1f}%" if dev is not None else "—")
        if run_trap or (run_fair_value and not fv.get("error")):
            cur_price = fv.get("current_price") if not fv.get("error") else None
            if cur_price is None and not df.empty:
                cur_price = float(df["Close"].iloc[-1])
            if cur_price:
                trap = check_value_trap(ticker, cur_price, df, roe_min=0.08, ma_window=75)
                dev = fv.get("deviation_pct") if not fv.get("error") else None
                if dev is not None and dev >= 20 and trap.get("roe_ok") and trap.get("trend_ok"):
                    st.success("【本命バリュー】乖離率20%以上・ROE≥8%・75MA上", icon="✅")
                elif dev is not None and dev >= 20:
                    st.warning("【トラップ警告】乖離率高いがROEまたは75MA未達", icon="⚠️")
    else:
        st.caption("ボタンで PER ベースの適正株価・トラップ判定を実行します。")

    try:
        patterns = detect_all_patterns(df)
    except Exception:
        patterns = []
    downtrend_mask = get_downtrend_mask(df, window=25)
    for i, name, side in patterns:
        col = f"Buy_{name}" if side == "buy" else f"Sell_{name}"
        if col not in df.columns:
            df[col] = False
        df.loc[df.index[i], col] = True

    signal_cols = [c for c in df.columns if c.startswith("Buy_")]
    valid_signals: list[str] = signal_cols
    if enable_backtest and signal_cols:
        engine = BacktestEngine()
        raw_results = engine.run(df, signal_columns=signal_cols, holding_period_days=holding_days, stop_loss_pct=stop_loss_pct)
        results = raw_results[raw_results["Total Trades"] >= 5].copy()
        if not results.empty:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.subheader("📊 勝率ランキング Top5")
                win_rank = results.sort_values("Win Rate", ascending=False).head(5)
                st.dataframe(win_rank[["Signal Name", "Win Rate", "Total Trades"]].style.format({"Win Rate": "{:.1%}"}), hide_index=True)
            with rc2:
                st.subheader("📈 収益力ランキング Top5")
                pf_rank = results.sort_values("Profit Factor", ascending=False).head(5)
                st.dataframe(pf_rank[["Signal Name", "Profit Factor", "Win Rate"]].style.format({"Profit Factor": "{:.2f}", "Win Rate": "{:.1%}"}), hide_index=True)
            heatmap_fig = plot_backtest_results(results, kind="heatmap")
            if heatmap_fig is not None:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            valid_signals = results[
                (results["Win Rate"] >= min_win_rate / 100.0) & (results["Profit Factor"] >= 1.0)
            ]["Signal Name"].tolist()
        with st.expander("📋 バックテスト成績表（全シグナル）"):
            st.dataframe(raw_results.style.format({"Win Rate": "{:.1%}", "Avg Return": "{:.2f}%", "Profit Factor": "{:.2f}"}), hide_index=True)

    last_row = df.iloc[-1]
    date_str = str(last_row.get("Date", df.index[-1]))[:10]
    close_str = fmt_price(last_row.get("Close"))
    active_buys = [c for c in valid_signals if c in df.columns and bool(df[c].fillna(False).iloc[-1])]
    sell_cols = [c for c in df.columns if c.startswith("Sell_")]
    active_sells = [c for c in sell_cols if bool(df[c].fillna(False).iloc[-1])]
    if active_buys:
        st.success(f"🟢 買い点灯: {', '.join(s.replace('Buy_', '') for s in active_buys)} | {date_str} {close_str}")
    elif active_sells:
        st.error(f"🔴 売り点灯: {', '.join(s.replace('Sell_', '') for s in active_sells)} | {date_str} {close_str}")
    else:
        st.info("⏳ 本日は有効シグナルなし")

    df_plot = df.copy()
    x = df_plot["Date"].tolist() if "Date" in df_plot.columns else df_plot.index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x, open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"], name="OHLC"))
    buy_agg: dict[int, list[str]] = {}
    sell_agg: dict[int, list[str]] = {}
    for i, name, side in patterns:
        if side == "buy" and f"Buy_{name}" in valid_signals:
            buy_agg.setdefault(i, []).append(name)
        elif side == "sell":
            sell_agg.setdefault(i, []).append(name)
    if buy_agg:
        fig.add_trace(go.Scatter(
            x=[x[i] for i in buy_agg], y=[df_plot.iloc[i]["Low"] * 0.98 for i in buy_agg],
            mode="markers", marker=dict(symbol="triangle-up", size=12, color="lime"), name="買い",
        ))
    if sell_agg:
        fig.add_trace(go.Scatter(
            x=[x[i] for i in sell_agg], y=[df_plot.iloc[i]["High"] * 1.02 for i in sell_agg],
            mode="markers", marker=dict(symbol="triangle-down", size=12, color="red"), name="売り",
        ))
    fig.update_layout(title=f"{ticker} ローソク足 & パターン", height=480, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_signals_tab(daily_json_url: str, root_dir: str) -> None:
    st.subheader("本日の買い・売りシグナル")
    st.caption("GitHub Actions が平日15:00頃に更新。4象限スコア順がデフォルト表示です。")

    hc1, hc2 = st.columns([1, 3])
    with hc1:
        if st.button("🔄 最新データを読み込み", type="primary", use_container_width=True):
            if daily_json_url:
                try:
                    _load_signals_from_url(daily_json_url)
                    st.success("最新 JSON を読み込みました。")
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")
            else:
                local = os.path.join(root_dir, "daily_buy_signals.json")
                if os.path.isfile(local):
                    _load_signals_from_file(local, "ローカル daily_buy_signals.json")
                    st.success("ローカル JSON を読み込みました。")
                else:
                    st.warning("DAILY_SIGNALS_JSON_URL が未設定です。")
            st.rerun()

    history_files = _list_signal_history_files(root_dir)
    hist_labels = [dt.strftime("%Y-%m-%d") for dt, _ in history_files]
    with st.expander("📅 過去日付のシグナルを表示・比較"):
        if hist_labels:
            pick = st.selectbox("履歴日付", hist_labels, key="history_pick")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("この日付を表示", key="load_history"):
                    path = next(p for dt, p in history_files if dt.strftime("%Y-%m-%d") == pick)
                    _load_signals_from_file(path, f"履歴 {pick}")
                    st.rerun()
            with b2:
                if st.button("比較用に保持", key="load_compare"):
                    path = next(p for dt, p in history_files if dt.strftime("%Y-%m-%d") == pick)
                    with open(path, "r", encoding="utf-8") as f:
                        st.session_state.compare_signals = json.load(f).get("items") or []
                    st.session_state.compare_label = pick
                    st.success(f"{pick} を比較用に保持しました。")
        elif daily_json_url:
            pick = st.date_input("日付（GitHub Raw）", value=None, key="history_date_remote")
            if pick and st.button("GitHub から履歴読込", key="load_history_remote"):
                url = history_json_url(daily_json_url, pick.isoformat())
                try:
                    _load_signals_from_url(url, f"GitHub 履歴 {pick.isoformat()}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            st.info("履歴 JSON が見つかりません。")

    with st.expander("⚙️ ローカル全銘柄スキャン（10〜15分・通常は不要）"):
        if st.button("表示を更新（フルスキャン）", key="daily_signal_refresh"):
            with st.spinner("スキャン中…"):
                try:
                    scan_result = scan_hybrid()
                    st.session_state.daily_buy_signals = scan_result.get("active", [])
                    st.session_state.daily_buy_signals_backtest_format = False
                    st.session_state.signals_source_label = "ローカルスキャン"
                except Exception as e:
                    st.error(str(e))
            st.rerun()

    items = st.session_state.get("daily_buy_signals")
    if items is None:
        st.info("「最新データを読み込み」を押すか、初回自動読込をお待ちください。")
        return

    _render_app_header()
    mode = st.session_state.get("signal_display_mode", "score")
    sort_key = st.session_state.get("signal_sort_key", "quadrant_score")
    mobile = st.session_state.get("ui_mobile_compact", False)
    backtest_fmt = st.session_state.get("daily_buy_signals_backtest_format", False)

    filtered = filter_signals(list(items), mode, backtest_format=backtest_fmt)
    filtered = sort_signals(filtered, sort_key)
    n_total = len(items)
    n_show = len(filtered)

    mode_labels = {
        "score": "4象限スコア順（全件）",
        "win_rate_70": "勝率70%以上",
        "sector_good": "セクター「良」のみ",
        "all": "フィルターなし",
    }
    st.caption(f"**{mode_labels.get(mode, mode)}** — 全 {n_total} 件中 **{n_show} 件** 表示")

    stats = st.session_state.get("daily_buy_signals_quadrant_stats")
    if stats:
        st.caption(
            f"4象限: mode={stats.get('mode')} | before→after {stats.get('before')}→{stats.get('after')}"
        )

    _render_signal_cards(filtered)
    buy_col, sell_col = st.columns(2)
    with buy_col:
        st.markdown("#### 🟢 買いシグナル")
        _render_buy_signals_table(filtered, mobile=mobile)
    with sell_col:
        st.markdown("#### 🔴 売りシグナル")
        _render_sell_table(st.session_state.get("daily_buy_signals_sell") or [])

    compare = st.session_state.get("compare_signals")
    if compare:
        st.markdown(f"#### 📊 比較: {st.session_state.get('compare_label')} vs 現在")
        cur_tickers = {x.get("ticker") for x in items}
        cmp_tickers = {x.get("ticker") for x in compare}
        st.write(f"現在のみ: {len(cur_tickers - cmp_tickers)} 銘柄 / 履歴のみ: {len(cmp_tickers - cur_tickers)} 銘柄 / 共通: {len(cur_tickers & cmp_tickers)} 銘柄")


def _render_analysis_tab(ticker: str, period: str) -> None:
    st.subheader(f"単一銘柄分析: {ticker}")
    with st.expander("バックテスト設定（この銘柄のパターン検証）"):
        enable_bt = st.checkbox("バックテストでシグナル選別", value=True, key="analysis_enable_bt")
        holding = st.slider("保有期間(営業日)", 3, 20, 5, key="analysis_holding")
        sl = st.slider("損切り(%)", 1.0, 10.0, 5.0, key="analysis_sl") / 100.0
        min_wr = st.slider("最低勝率(%)", 0, 100, 50, key="analysis_min_wr")
    _render_detail_chart(ticker, period, enable_backtest=enable_bt, holding_days=holding, stop_loss_pct=sl, min_win_rate=min_wr)

    st.divider()
    st.subheader("本命の割安株（バッチ一括スキャン）")
    top_n = st.session_state.get("batch_top_n", 10)
    if st.button("本命の割安株を一括スキャン", type="primary", key="batch_value_screen_btn"):
        with st.spinner("スキャン中…"):
            try:
                rows = run_batch(refill_missing=True, top_n=int(top_n))
                if rows:
                    st.success(f"{len(rows)} 銘柄が条件を満たしました。")
                    for r in rows:
                        st.markdown(f"- {format_line(r)}")
                else:
                    st.info("該当銘柄なし")
            except Exception as e:
                st.error(str(e))


def _render_verify_tab(root_dir: str) -> None:
    st.subheader("シグナル検証（引け購入リターン）")
    holding = st.session_state.get("verify_holding_days", 3)
    if st.button("引け購入リターンを検証する", key="btn_verify_returns"):
        full_list = st.session_state.get("daily_buy_signals") or []
        if not full_list:
            st.warning("先に日次シグナルタブでデータを読み込んでください。")
        else:
            payload = {"items": full_list, "updated": st.session_state.get("daily_buy_signals_updated")}
            with st.spinner("計算中…"):
                rows = verify_returns(payload, holding_days=int(holding))
                stats = summary_stats(rows)
            if rows:
                st.success(f"有効 {stats['count']} 件 | 勝率 {stats['win_rate']}% | 平均 {stats['avg_return_pct']}%")
                df_v = pd.DataFrame(rows).rename(columns={
                    "ticker": "銘柄", "pattern_name": "パターン", "entry": "エントリー(¥)",
                    "exit_price": "利確(¥)", "return_pct": "リターン(%)",
                })
                st.dataframe(df_v, hide_index=True, use_container_width=True)

    history_files = _list_signal_history_files(root_dir)
    if history_files:
        st.markdown("#### 100株ずつポートフォリオ損益（直近3営業日）")
        for idx, h in enumerate([1, 2, 3], start=1):
            if len(history_files) <= idx:
                continue
            dt, path_hist = history_files[idx]
            with open(path_hist, "r", encoding="utf-8") as f:
                data_hist = json.load(f)
            rows_h = verify_returns(data_hist, holding_days=h)
            stats_h = summary_stats(rows_h)
            profit_yen, profit_pct = _compute_portfolio_pnl_100_shares(rows_h)
            st.markdown(
                f"- **{h}日後** ({dt.date()}): 有効 {stats_h['count']}件 / "
                f"勝率 {stats_h['win_rate'] or '—'}% / 損益 {profit_yen:+,.0f}円 ({profit_pct:+.2f}%)"
            )


def _render_settings_tab(daily_json_url: str) -> None:
    st.subheader("設定・接続")
    st.write("**DAILY_SIGNALS_JSON_URL**", daily_json_url or "（未設定 — ローカル JSON を使用）")
    st.write("**Gemini API**", "設定済み" if api_ready else "未設定")
    if st.button("Gemini 疎通テスト（トヨタ）", disabled=not api_ready):
        st.write(gemini_echo_ticker("トヨタ", streamlit_secrets=GEMINI_SECRETS))
    st.markdown("---")
    st.markdown("**Streamlit Secrets 例**")
    st.code(
        'DAILY_SIGNALS_JSON_URL = "https://raw.githubusercontent.com/jumkita/stock-daytrade/main/daily_buy_signals.json"\n'
        'GEMINI_API_KEY = "..."',
        language="toml",
    )


def main() -> None:
    st.set_page_config(page_title="日本株 適正株価・パターン分析", layout="wide", initial_sidebar_state="expanded")
    _init_session_state()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    daily_json_url = _get_daily_json_url()

    if daily_json_url and st.session_state.daily_buy_signals is None:
        try:
            _load_signals_from_url(daily_json_url, "GitHub（自動）")
        except Exception:
            local = os.path.join(root_dir, "daily_buy_signals.json")
            if os.path.isfile(local):
                try:
                    _load_signals_from_file(local, "ローカル（自動）")
                except Exception:
                    pass

    with st.sidebar:
        st.header("⚙️ 設定")
        if not api_ready:
            st.warning("Gemini API 未設定")
        period = st.selectbox("分析期間", ["3mo", "6mo", "1y", "2y"], index=0)
        ticker = st.text_input("銘柄コード", value="8473.T", key="ticker_input")
        st.divider()
        st.subheader("📊 シグナル表示")
        st.session_state.signal_display_mode = st.selectbox(
            "表示モード",
            ["score", "win_rate_70", "sector_good", "all"],
            format_func=lambda x: {"score": "4象限スコア順（全件）", "win_rate_70": "勝率70%以上", "sector_good": "セクター良のみ", "all": "全件（フィルターなし）"}[x],
            index=0,
        )
        st.session_state.signal_sort_key = st.selectbox(
            "並び順",
            ["quadrant_score", "win_rate", "avg_return_pct"],
            format_func=lambda x: {"quadrant_score": "4象限スコア", "win_rate": "勝率", "avg_return_pct": "平均リターン"}[x],
        )
        st.session_state.ui_mobile_compact = st.checkbox("モバイル向け簡易列", value=False)
        st.session_state.verify_holding_days = st.number_input("検証保有日数", 1, 20, 3)
        st.session_state.batch_top_n = st.number_input("バッチ上位N", 1, 50, 10)

    st.title("日本株 適正株価 × 勝ちパターン分析")

    tab_sig, tab_ana, tab_ver, tab_set = st.tabs(["📊 日次シグナル", "🔍 銘柄分析", "📈 検証", "⚙️ 設定"])

    with tab_sig:
        _render_signals_tab(daily_json_url, root_dir)
    with tab_ana:
        _render_analysis_tab(ticker, period)
    with tab_ver:
        _render_verify_tab(root_dir)
    with tab_set:
        _render_settings_tab(daily_json_url)


if __name__ == "__main__":
    main()
