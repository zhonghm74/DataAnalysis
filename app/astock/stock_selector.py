"""
Stock selection strategies for A-shares.

Each selector returns a list of (symbol, score, reason) tuples.
"""

import pandas as pd
import numpy as np
from .market_data import fetch_stock, add_technical_indicators


def _analyze(symbol: str, lookback: int = 120) -> dict:
    try:
        df = fetch_stock(symbol)
        if len(df) < lookback:
            return None
        df = add_technical_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        return {"symbol": symbol, "df": df, "latest": latest, "prev": prev}
    except Exception:
        return None


def momentum_selector(pool: list, top_n: int = 5) -> list:
    """Select stocks with strongest momentum (20-day return + volume surge)."""
    results = []
    for sym in pool:
        info = _analyze(sym)
        if info is None:
            continue
        df = info["df"]
        latest = info["latest"]

        ret_20 = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100 if len(df) > 21 else 0
        ret_5 = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100 if len(df) > 6 else 0
        vol_surge = latest.get("vol_ratio", 1)

        score = ret_20 * 0.4 + ret_5 * 0.3 + min(vol_surge, 3) * 10 * 0.3
        reasons = []
        if ret_20 > 5:
            reasons.append(f"20日涨{ret_20:.1f}%")
        if vol_surge > 1.5:
            reasons.append(f"放量{vol_surge:.1f}倍")
        if latest.get("rsi14", 50) > 50:
            reasons.append(f"RSI={latest['rsi14']:.0f}")
        if not reasons:
            reasons.append("动量一般")

        results.append((sym, round(score, 2), "; ".join(reasons)))

    return sorted(results, key=lambda x: -x[1])[:top_n]


def mean_reversion_selector(pool: list, top_n: int = 5) -> list:
    """Select oversold stocks near support (low RSI + near Bollinger lower band)."""
    results = []
    for sym in pool:
        info = _analyze(sym)
        if info is None:
            continue
        latest = info["latest"]

        rsi = latest.get("rsi14", 50)
        bb_pos = 0
        if latest.get("bb_upper", 0) != latest.get("bb_lower", 0):
            bb_pos = (latest["close"] - latest["bb_lower"]) / (latest["bb_upper"] - latest["bb_lower"] + 1e-10)

        score = (50 - rsi) * 0.5 + (1 - bb_pos) * 50 * 0.5
        reasons = []
        if rsi < 30:
            reasons.append(f"RSI超卖={rsi:.0f}")
        elif rsi < 40:
            reasons.append(f"RSI偏低={rsi:.0f}")
        if bb_pos < 0.2:
            reasons.append("触及布林下轨")
        if latest.get("kdj_j", 50) < 20:
            reasons.append(f"KDJ-J={latest['kdj_j']:.0f}")
        if not reasons:
            reasons.append("未达超卖")

        results.append((sym, round(score, 2), "; ".join(reasons)))

    return sorted(results, key=lambda x: -x[1])[:top_n]


def trend_following_selector(pool: list, top_n: int = 5) -> list:
    """Select stocks in strong uptrend (MA alignment + MACD golden cross)."""
    results = []
    for sym in pool:
        info = _analyze(sym)
        if info is None:
            continue
        latest = info["latest"]
        df = info["df"]

        ma_aligned = (latest.get("ma5", 0) > latest.get("ma10", 0) >
                      latest.get("ma20", 0) > latest.get("ma60", 0))
        macd_positive = latest.get("macd_hist", 0) > 0
        above_ma20 = latest["close"] > latest.get("ma20", latest["close"])

        score = 0
        reasons = []
        if ma_aligned:
            score += 40
            reasons.append("均线多头排列")
        if macd_positive:
            score += 30
            reasons.append("MACD红柱")
        if above_ma20:
            score += 20
            reasons.append("站上MA20")
        if latest.get("vol_ratio", 1) > 1.2:
            score += 10
            reasons.append(f"量比{latest['vol_ratio']:.1f}")
        if not reasons:
            reasons.append("趋势不明")

        results.append((sym, round(score, 2), "; ".join(reasons)))

    return sorted(results, key=lambda x: -x[1])[:top_n]


def breakout_selector(pool: list, top_n: int = 5) -> list:
    """Select stocks breaking out of consolidation (new 20-day high + volume)."""
    results = []
    for sym in pool:
        info = _analyze(sym)
        if info is None:
            continue
        df = info["df"]
        latest = info["latest"]

        high_20 = df["high"].iloc[-21:-1].max() if len(df) > 21 else df["high"].max()
        is_breakout = latest["close"] > high_20
        vol_surge = latest.get("vol_ratio", 1)
        bb_break = latest["close"] > latest.get("bb_upper", latest["close"] * 1.1)

        score = 0
        reasons = []
        if is_breakout:
            score += 50
            reasons.append(f"突破20日高点{high_20:.2f}")
        if vol_surge > 2:
            score += 30
            reasons.append(f"放量{vol_surge:.1f}倍")
        elif vol_surge > 1.5:
            score += 15
            reasons.append(f"温和放量{vol_surge:.1f}倍")
        if bb_break:
            score += 20
            reasons.append("突破布林上轨")
        if not reasons:
            reasons.append("未突破")

        results.append((sym, round(score, 2), "; ".join(reasons)))

    return sorted(results, key=lambda x: -x[1])[:top_n]


SELECTORS = {
    "动量策略": momentum_selector,
    "均值回归": mean_reversion_selector,
    "趋势跟踪": trend_following_selector,
    "突破策略": breakout_selector,
}

ML_SELECTORS = ["ML多因子选股 (LightGBM)"]
