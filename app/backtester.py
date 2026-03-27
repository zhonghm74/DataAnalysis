"""
Backtesting module — simulate a simple directional trading strategy.

Strategy:
  - If predicted Δy > threshold → go LONG (buy bond / expect yield to rise)
  - If predicted Δy < -threshold → go SHORT
  - Otherwise → HOLD (no position)

For bond trading: yield UP = bond price DOWN, yield DOWN = bond price UP.
We trade the *direction of yield change* and assume PnL = position × Δyield × notional.
"""

import numpy as np
import pandas as pd


def backtest(delta_true: np.ndarray, delta_pred: np.ndarray,
             dates: pd.DatetimeIndex, threshold: float = 0.0,
             notional: float = 1_000_000, bp_per_unit: float = 100) -> dict:
    """
    Run backtest on test period.

    Args:
        delta_true: actual daily yield changes
        delta_pred: predicted daily yield changes
        dates: date index
        threshold: minimum predicted change to take position
        notional: notional position size
        bp_per_unit: basis points per unit yield change

    Returns:
        dict with backtest results and daily PnL series
    """
    n = len(delta_true)
    positions = np.zeros(n)  # +1 long, -1 short, 0 flat
    pnl = np.zeros(n)

    for i in range(n):
        if delta_pred[i] > threshold:
            positions[i] = 1
        elif delta_pred[i] < -threshold:
            positions[i] = -1

        pnl[i] = positions[i] * delta_true[i] * notional * bp_per_unit / 10000

    cumulative_pnl = np.cumsum(pnl)
    total_pnl = cumulative_pnl[-1]
    n_trades = (positions != 0).sum()
    win_trades = ((positions * delta_true) > 0).sum()
    dir_correct = ((delta_pred > 0) == (delta_true > 0)).mean()

    # Max drawdown
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    max_dd = drawdown.max()

    # Sharpe (annualized, ~252 trading days)
    daily_returns = pnl / notional
    sharpe = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-10) if daily_returns.std() > 0 else 0

    # Buy-and-hold comparison (always long)
    bh_pnl = delta_true * notional * bp_per_unit / 10000
    bh_total = bh_pnl.sum()

    return {
        "总收益": round(total_pnl, 2),
        "交易天数": n,
        "持仓天数": int(n_trades),
        "空仓天数": int(n - n_trades),
        "盈利交易": int(win_trades),
        "亏损交易": int(n_trades - win_trades),
        "胜率(%)": round(win_trades / max(n_trades, 1) * 100, 1),
        "方向准确率(%)": round(dir_correct * 100, 1),
        "最大回撤": round(max_dd, 2),
        "年化Sharpe": round(sharpe, 3),
        "买入持有收益": round(bh_total, 2),
        "超额收益": round(total_pnl - bh_total, 2),
        "_pnl": pnl,
        "_cum_pnl": cumulative_pnl,
        "_positions": positions,
        "_dates": dates,
        "_bh_cum": np.cumsum(bh_pnl),
    }
