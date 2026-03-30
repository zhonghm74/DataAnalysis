"""
Trading signal models for A-shares.

Each model takes a stock DataFrame (with technical indicators) and returns
a signal series: +1 (buy), -1 (sell), 0 (hold).
"""

import numpy as np
import pandas as pd


def macd_cross_signal(df: pd.DataFrame) -> pd.Series:
    """MACD golden/dead cross signals."""
    sig = pd.Series(0, index=df.index)
    macd = df["macd_hist"]
    sig[(macd > 0) & (macd.shift(1) <= 0)] = 1   # golden cross
    sig[(macd < 0) & (macd.shift(1) >= 0)] = -1   # dead cross
    return sig


def ma_cross_signal(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """Moving average crossover signals."""
    sig = pd.Series(0, index=df.index)
    fast_ma = df[f"ma{fast}"] if f"ma{fast}" in df else df["close"].rolling(fast).mean()
    slow_ma = df[f"ma{slow}"] if f"ma{slow}" in df else df["close"].rolling(slow).mean()
    sig[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
    sig[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1
    return sig


def rsi_signal(df: pd.DataFrame, buy_thresh: int = 30, sell_thresh: int = 70) -> pd.Series:
    """RSI overbought/oversold signals."""
    sig = pd.Series(0, index=df.index)
    rsi = df["rsi14"]
    sig[(rsi < buy_thresh) & (rsi.shift(1) >= buy_thresh)] = 1
    sig[(rsi > sell_thresh) & (rsi.shift(1) <= sell_thresh)] = -1
    return sig


def bollinger_signal(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band breakout/reversion signals."""
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] < df["bb_lower"]) & (df["close"].shift(1) >= df["bb_lower"].shift(1))] = 1
    sig[(df["close"] > df["bb_upper"]) & (df["close"].shift(1) <= df["bb_upper"].shift(1))] = -1
    return sig


def kdj_signal(df: pd.DataFrame) -> pd.Series:
    """KDJ golden/dead cross signals."""
    sig = pd.Series(0, index=df.index)
    k, d = df["kdj_k"], df["kdj_d"]
    sig[(k > d) & (k.shift(1) <= d.shift(1)) & (k < 30)] = 1   # oversold golden cross
    sig[(k < d) & (k.shift(1) >= d.shift(1)) & (k > 70)] = -1  # overbought dead cross
    return sig


def volume_price_signal(df: pd.DataFrame) -> pd.Series:
    """Volume-price divergence signals."""
    sig = pd.Series(0, index=df.index)
    price_up = df["close"] > df["close"].shift(1)
    vol_up = df["vol_ratio"] > 1.5
    sig[price_up & vol_up] = 1
    price_down = df["close"] < df["close"].shift(1)
    vol_down = df["vol_ratio"] < 0.7
    sig[price_down & vol_down] = -1
    return sig


def composite_signal(df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """Weighted ensemble of all signal models."""
    if weights is None:
        weights = {
            "MACD交叉": 0.25, "均线交叉": 0.20, "RSI": 0.15,
            "布林带": 0.15, "KDJ": 0.15, "量价": 0.10,
        }

    signals = {
        "MACD交叉": macd_cross_signal(df),
        "均线交叉": ma_cross_signal(df),
        "RSI": rsi_signal(df),
        "布林带": bollinger_signal(df),
        "KDJ": kdj_signal(df),
        "量价": volume_price_signal(df),
    }

    weighted = sum(signals[name] * weights.get(name, 0) for name in signals)
    result = pd.Series(0, index=df.index)
    result[weighted > 0.2] = 1
    result[weighted < -0.2] = -1
    return result


SIGNAL_MODELS = {
    "MACD交叉": macd_cross_signal,
    "均线交叉 (MA5/MA20)": ma_cross_signal,
    "RSI超买超卖": rsi_signal,
    "布林带突破": bollinger_signal,
    "KDJ金叉死叉": kdj_signal,
    "量价配合": volume_price_signal,
    "综合信号 (加权)": composite_signal,
}
