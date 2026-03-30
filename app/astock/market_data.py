"""
A-share market data fetcher with caching.

Provides:
  - Single stock daily OHLCV (forward-adjusted)
  - Technical indicator computation
  - Limit-up / limit-down detection
"""

import os, hashlib
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "astock")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, start: str, end: str) -> str:
    key = hashlib.md5(f"{symbol}_{start}_{end}".encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{symbol}_{key}.csv")


def fetch_stock(symbol: str, start_date: str = "20200101",
                end_date: str = None) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cp = _cache_path(symbol, start_date, end_date)
    if os.path.exists(cp):
        mtime = datetime.fromtimestamp(os.path.getmtime(cp))
        if datetime.now() - mtime < timedelta(hours=6):
            return pd.read_csv(cp, parse_dates=["date"])

    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                             start_date=start_date, end_date=end_date, adjust="qfq")
    df.columns = ["date", "code", "open", "close", "high", "low",
                   "volume", "amount", "amplitude", "pct_chg", "chg", "turnover"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cp, index=False)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Moving averages
    for w in [5, 10, 20, 60]:
        d[f"ma{w}"] = d["close"].rolling(w).mean()

    # EMA
    for w in [12, 26]:
        d[f"ema{w}"] = d["close"].ewm(span=w, adjust=False).mean()

    # MACD
    d["macd_dif"] = d["ema12"] - d["ema26"]
    d["macd_dea"] = d["macd_dif"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = 2 * (d["macd_dif"] - d["macd_dea"])

    # RSI
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # Bollinger Bands
    d["bb_mid"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * bb_std
    d["bb_lower"] = d["bb_mid"] - 2 * bb_std
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / (d["bb_mid"] + 1e-10)

    # KDJ
    low_min = d["low"].rolling(9).min()
    high_max = d["high"].rolling(9).max()
    rsv = (d["close"] - low_min) / (high_max - low_min + 1e-10) * 100
    d["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    d["kdj_d"] = d["kdj_k"].ewm(com=2, adjust=False).mean()
    d["kdj_j"] = 3 * d["kdj_k"] - 2 * d["kdj_d"]

    # Volume ratio
    d["vol_ma5"] = d["volume"].rolling(5).mean()
    d["vol_ratio"] = d["volume"] / (d["vol_ma5"] + 1)

    # ATR
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"] - d["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    # Limit-up / limit-down detection (10% for main board, 20% for ChiNext/STAR)
    prev_close = d["close"].shift(1)
    is_chinext = str(d["code"].iloc[0]).startswith("3")
    is_star = str(d["code"].iloc[0]).startswith("68")
    limit_pct = 0.20 if (is_chinext or is_star) else 0.10

    d["limit_up_price"] = (prev_close * (1 + limit_pct)).round(2)
    d["limit_down_price"] = (prev_close * (1 - limit_pct)).round(2)
    d["is_limit_up"] = (d["close"] >= d["limit_up_price"] - 0.01) & (d["pct_chg"] > limit_pct * 100 * 0.95)
    d["is_limit_down"] = (d["close"] <= d["limit_down_price"] + 0.01) & (d["pct_chg"] < -limit_pct * 100 * 0.95)

    return d


# Pre-defined stock pools
STOCK_POOLS = {
    "沪深300成分股(示例)": [
        "600519", "601318", "000858", "600036", "601166",
        "000333", "600276", "601888", "300750", "002714",
        "600887", "000568", "601012", "002475", "600030",
        "000001", "600000", "601398", "601288", "600809",
    ],
    "科技龙头(示例)": [
        "002230", "300059", "603986", "688981", "002049",
        "300124", "002415", "300496", "688036", "300661",
    ],
    "消费龙头(示例)": [
        "600519", "000858", "000568", "600887", "002304",
        "603288", "600809", "000895", "002507", "600600",
    ],
}
