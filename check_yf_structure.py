# -*- coding: utf-8 -*-
"""
yfinance 取得データの構造・出来高・MA・判定ロジックの確認用スクリプト。
単一銘柄でも yf.download は列が MultiIndex になるため、logic.flatten_ohlcv_columns でフラット化する。
"""
import pandas as pd
import yfinance as yf

from logic import flatten_ohlcv_columns

# 疑わしい銘柄コードを入力
ticker_code = "7203.T"
data = yf.download(ticker_code, period="100d", interval="1d", progress=False)
data = flatten_ohlcv_columns(data)

# 1. 構造の確認
print(f"--- Structure Check for {ticker_code} ---")
print(f"Columns structure: {data.columns.tolist()}")
print(f"Shape: {data.shape}")

# 2. 出来高の生データ確認
last_volume = data["Volume"].iloc[-1]
print(f"Latest Volume: {last_volume}")

# 3. トレンドライン（MA）の計算確認
ma25 = data["Close"].rolling(window=25).mean().iloc[-1]
ma75 = data["Close"].rolling(window=75).mean().iloc[-1]
print(f"25MA: {ma25}")
print(f"75MA: {ma75}")

# 4. 判定ロジックのシミュレーション
current_price = data["Close"].iloc[-1]
is_near_ma = abs(current_price - ma25) / ma25 < 0.05 if ma25 else False
print(f"Current Price: {current_price}")
print(f"Is near 25MA (within 5%): {is_near_ma}")
